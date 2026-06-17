#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import sys
import time
import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from tqdm import tqdm

import sam3

# ===================== Simple Palette for Visualization =====================
PALETTE: Sequence[Tuple[int, int, int]] = (
    (239, 83, 80), (102, 187, 106), (66, 165, 245), (255, 202, 40),
    (171, 71, 188), (255, 112, 67), (38, 198, 218), (156, 204, 101),
)

IMAGE_EXT = {"jpg", "jpeg", "png", "bmp"}


# ===================== Post-processing Functions =====================

def smooth_mask(mask, kernel_size=5, iterations=1):
    if kernel_size <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return smoothed


def mask_to_polygons(mask, min_area=16.0, epsilon_ratio=0.01, keep_largest_only=False):
    mask_uint8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_data = [(c, cv2.contourArea(c)) for c in contours if cv2.contourArea(c) >= min_area]
    if not contour_data:
        return []
    if keep_largest_only:
        contour_data = [max(contour_data, key=lambda x: x[1])]
    polygons = []
    for contour, _ in contour_data:
        perimeter = max(cv2.arcLength(contour, True), 1.0)
        approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
        if approx.shape[0] < 3:
            continue
        poly = approx.reshape(-1, 2)
        polygons.append(poly.flatten().astype(float).tolist())
    return polygons


def mask_to_xywh(mask):
    ys, xs = np.nonzero(mask)
    return float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)


def compute_bbox_iou(box_a, box_b):
    """Compute IoU between two xyxy boxes (x1, y1, x2, y2)."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter_area = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def save_coco_per_image(json_dir, image_info, anns, categories):
    (json_dir / f"{image_info['file_name'].rsplit('.',1)[0]}.json").write_text(
        json.dumps({"images": [image_info], "annotations": anns,
                     "categories": [{"id": c, "name": n} for c, n in sorted(categories.items())]},
                   indent=2), encoding="utf-8")


def overlay_masks(image, masks, boxes, labels, alpha):
    out = image.copy()
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = None
    for idx, mask in enumerate(masks):
        color = PALETTE[idx % len(PALETTE)]
        img_np = np.array(out, dtype=np.float32)
        img_np[mask.astype(bool)] = (1 - alpha) * img_np[mask.astype(bool)] + alpha * np.array(color)
        out = Image.fromarray(img_np.astype(np.uint8))
        x1, y1, x2, y2 = boxes[idx]
        draw = ImageDraw.Draw(out)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        txt = str(labels[idx])
        if font:
            draw.text((x1 + 2, y1 + 2), txt, fill=color, font=font)
        else:
            draw.text((x1 + 2, y1 + 2), txt, fill=color)
    return out


# ===================== YOLO Label File Helpers =====================

def load_yolo_labels(label_path):
    entries = []
    if not label_path.exists():
        return entries
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            entries.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
        except ValueError:
            continue
    return entries


def yolo_to_xyxy(cx, cy, w, h, width, height):
    x1 = max(0.0, min((cx - w * 0.5) * width, width - 1.0))
    y1 = max(0.0, min((cy - h * 0.5) * height, height - 1.0))
    x2 = max(x1 + 1.0, min((cx + w * 0.5) * width, width - 1.0))
    y2 = max(y1 + 1.0, min((cy + h * 0.5) * height, height - 1.0))
    return x1, y1, x2, y2


# ===================== SAM3 Helpers =====================

GLOBAL_COUNTER = 1

def create_empty_datapoint():
    from sam3.train.data.sam3_image_dataset import Datapoint
    return Datapoint(find_queries=[], images=[])

def set_image(dp, pil_image):
    from sam3.train.data.sam3_image_dataset import Image as SAMImage
    w, h = pil_image.size
    dp.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]

def add_box_prompt(dp, boxes_xyxy, text_prompt="object"):
    global GLOBAL_COUNTER
    from sam3.train.data.sam3_image_dataset import InferenceMetadata, FindQueryLoaded
    w, h = dp.images[0].size
    dp.find_queries.append(FindQueryLoaded(
        query_text=text_prompt, image_id=0, object_ids_output=[],
        is_exhaustive=True, query_processing_order=0,
        input_bbox=torch.tensor(boxes_xyxy, dtype=torch.float).view(-1, 4),
        input_bbox_label=torch.tensor([True] * len(boxes_xyxy), dtype=torch.bool),
        inference_metadata=InferenceMetadata(
            coco_image_id=GLOBAL_COUNTER, original_image_id=GLOBAL_COUNTER,
            original_category_id=1, original_size=[w, h], object_id=0, frame_index=0)))
    GLOBAL_COUNTER += 1
    return GLOBAL_COUNTER - 1

def add_text_prompt(dp, text_query):
    global GLOBAL_COUNTER
    from sam3.train.data.sam3_image_dataset import InferenceMetadata, FindQueryLoaded
    w, h = dp.images[0].size
    dp.find_queries.append(FindQueryLoaded(
        query_text=text_query, image_id=0, object_ids_output=[],
        is_exhaustive=True, query_processing_order=0,
        inference_metadata=InferenceMetadata(
            coco_image_id=GLOBAL_COUNTER, original_image_id=GLOBAL_COUNTER,
            original_category_id=1, original_size=[w, h], object_id=0, frame_index=0)))
    GLOBAL_COUNTER += 1
    return GLOBAL_COUNTER - 1

def run_yolo_inference(yolo_model, pil_image, conf=0.3, classes=None):
    results = yolo_model(pil_image, conf=conf, classes=classes, verbose=False)
    if len(results) == 0 or results[0].boxes is None:
        return []
    return results[0].boxes.xyxy.cpu().numpy().tolist()


# ===================== Auto-detection =====================

def find_best_pt(search_root):
    """Recursively find best.pt under search_root."""
    matches = list(search_root.glob("**/best.pt"))
    return matches[0] if matches else None

def _has_images(d):
    return d.is_dir() and any(p.suffix.lower().lstrip(".") in IMAGE_EXT for p in d.iterdir())

def find_image_dir(search_root):
    """Find the main image directory: prefer production_data/, then images/, then any dir with images.
    Supports nested structures like production_data/子文件夹/*.jpg."""
    if _has_images(search_root):
        return search_root
    for candidate in ["production_data", "images", "sample"]:
        d = search_root / candidate
        if d.is_dir():
            if _has_images(d):
                return d
            for sub in sorted(d.iterdir()):
                if _has_images(sub):
                    return d
    for d in sorted(search_root.iterdir()):
        if d.is_dir():
            if _has_images(d):
                return d
            for sub in sorted(d.iterdir()):
                if _has_images(sub):
                    return d
    return None

def scan_material_folders(root, args):
    """
    Scan root for material folders. Returns list of (name, mode, config).
    mode is "dataset" or "yolo".
    """
    tasks = []

    # First: check if root itself is a dataset
    if (root / "dataset").is_dir():
        tasks.append((root.name, "dataset", {"root": root}))
        return tasks

    # Second: check for model/ + production_data/ layout (multi-model YOLO)
    model_dir = root / "model"
    prod_dir = root / "production_data"
    if model_dir.is_dir() and prod_dir.is_dir():
        for model_sub in sorted(model_dir.iterdir()):
            if not model_sub.is_dir():
                continue
            best_pt = find_best_pt(model_sub)
            if not best_pt:
                continue
            prod_sub = prod_dir / model_sub.name
            if not prod_sub.is_dir():
                continue
            img_dir = find_image_dir(prod_sub)
            if not img_dir:
                continue
            tasks.append((model_sub.name, "yolo", {
                "yolo_model": best_pt,
                "image_dir": img_dir,
                "instance_name": f"Instance_{model_sub.name}",
            }))
        if tasks:
            return tasks

    # Third: check if root itself is a YOLO-mode folder (has best.pt + images)
    best_pt = find_best_pt(root)
    img_dir = find_image_dir(root)
    if best_pt and img_dir:
        tasks.append((root.name, "yolo", {"yolo_model": best_pt, "image_dir": img_dir}))
        return tasks

    # Root has images + --prompt (no best.pt) → text-only mode
    if img_dir and args.prompt:
        tasks.append((root.name, "yolo", {"yolo_model": None, "image_dir": img_dir}))
        return tasks

    # Third: scan subfolders
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue

        # Subfolder has dataset/ → dataset mode
        if (sub / "dataset").is_dir():
            tasks.append((sub.name, "dataset", {"root": sub}))
            continue

        # Subfolder has best.pt somewhere + images → YOLO mode
        best_pt = find_best_pt(sub)
        img_dir = find_image_dir(sub)
        if best_pt and img_dir:
            tasks.append((sub.name, "yolo", {"yolo_model": best_pt, "image_dir": img_dir}))
            continue

        # Subfolder has images + --prompt (no best.pt) → text-only mode
        if img_dir and args.prompt:
            tasks.append((sub.name, "yolo", {"yolo_model": None, "image_dir": img_dir}))
            continue

    return tasks


# ===================== Inference Routines =====================

def process_dataset_task(args, model, collate, transform, postprocessor, name, cfg):
    """Process a single dataset-type folder (has dataset/ subfolder)."""
    dataset_root = cfg["root"] / "dataset"
    output_root = Path(args.output_dir) / name

    print(f"\n{'='*60}")
    print(f"[{name}] Dataset mode")
    print(f"  Path: {dataset_root}")
    print(f"  Splits: {args.splits}")
    print(f"  Output: {output_root}")
    print(f"{'='*60}")

    ann_id = 1
    total_processed = 0
    total_annotations = 0
    categories = {}

    for split_name in args.splits:
        image_dir = dataset_root / split_name / "images"
        label_dir = dataset_root / split_name / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            print(f"  [{split_name}] SKIP: images/labels 目录不存在")
            continue

        split_json_dir = output_root / split_name / "Instance" / "label"
        split_mask_dir = output_root / split_name / "Instance" / "mask"
        split_vis_dir = output_root / split_name / "Instance" / "vis"
        for d in [split_json_dir, split_mask_dir, split_vis_dir]:
            d.mkdir(parents=True, exist_ok=True)

        image_paths = sorted([p for p in image_dir.glob("*") if p.suffix.lower().lstrip(".") in IMAGE_EXT])
        if not image_paths:
            print(f"  [{split_name}] SKIP: 无图像")
            continue

        print(f"\n  [{split_name}] {len(image_paths)} images")

        all_dps, all_imgs, all_qids, all_paths, all_cls, all_yolo_boxes = [], [], [], [], [], []
        for img_path in tqdm(image_paths, desc=f"  Loading {split_name}", unit="img"):
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            labels = load_yolo_labels(label_dir / f"{img_path.stem}.txt")
            if not labels:
                continue
            w, h = img.size
            boxes, cls_ids = [], []
            for cid, cx, cy, bw, bh in labels:
                boxes.append(yolo_to_xyxy(cx, cy, bw, bh, w, h))
                cls_ids.append(cid)
                categories.setdefault(cid, f"class_{cid}")
            dp = create_empty_datapoint()
            set_image(dp, img)
            qid = add_box_prompt(dp, boxes, args.prompt if args.prompt else "object")
            dp = transform(dp)
            all_dps.append(dp)
            all_imgs.append(img)
            all_qids.append(qid)
            all_paths.append(img_path)
            all_cls.append(cls_ids)
            all_yolo_boxes.append([tuple(b) for b in boxes])

        if not all_dps:
            print(f"  [{split_name}] SKIP: 无有效图像")
            continue

        coco_images, coco_annotations = [], []
        img_id_counter = 1
        processed = 0
        skipped = 0

        for i in range(0, len(all_dps), args.batch_size):
            batch_dps = all_dps[i:i+args.batch_size]
            batch_imgs = all_imgs[i:i+args.batch_size]
            batch_qids = all_qids[i:i+args.batch_size]
            batch_paths = all_paths[i:i+args.batch_size]
            batch_cls = all_cls[i:i+args.batch_size]
            batch_yolo_boxes = all_yolo_boxes[i:i+args.batch_size]

            b = collate(batch_dps, dict_key="dummy")["dummy"]
            from sam3.model.utils.misc import copy_data_to_device
            b = copy_data_to_device(b, torch.device("cuda"), non_blocking=True)
            with torch.no_grad():
                output = model(b)
            results = postprocessor.process_results(output, b.find_metadatas)

            for j in range(len(batch_dps)):
                img = batch_imgs[j]
                qid = batch_qids[j]
                path = Path(batch_paths[j])
                w, h = img.size
                img_cls_ids = batch_cls[j]
                yolo_boxes = batch_yolo_boxes[j]

                if args.skip_existing and (split_json_dir / f"{path.stem}.json").exists() and (split_mask_dir / f"{path.stem}_mask.png").exists():
                    skipped += 1
                    continue

                r = results.get(qid)
                if r is None or r.get("masks") is None or r.get("scores") is None:
                    continue

                scores_np = r["scores"].float().detach().cpu().numpy().flatten()
                masks_np = r["masks"].float().detach().cpu().numpy().astype(np.uint8)
                if masks_np.ndim == 4:
                    masks_np = masks_np.squeeze(1)
                elif masks_np.ndim == 2:
                    masks_np = masks_np[None]

                masks_vis, boxes_vis, labels_vis, anns_img = [], [], [], []
                for k in range(masks_np.shape[0]):
                    mask_bin = masks_np[k].astype(np.uint8)
                    if float(mask_bin.sum()) < args.min_mask_area:
                        continue
                    if args.morph_kernel_size > 0:
                        mask_bin = smooth_mask(mask_bin, args.morph_kernel_size, args.morph_iterations)
                    polys = mask_to_polygons(mask_bin, args.min_poly_area, args.epsilon_ratio, args.keep_largest_only)
                    if not polys:
                        continue
                    area = float(mask_bin.sum())
                    x, y, bw_b, bh_b = mask_to_xywh(mask_bin)
                    if args.iou_threshold > 0 and yolo_boxes:
                        mask_xyxy = (x, y, x + bw_b - 1, y + bh_b - 1)
                        max_iou = max(compute_bbox_iou(mask_xyxy, yb) for yb in yolo_boxes)
                        if max_iou < args.iou_threshold:
                            continue
                    cid = img_cls_ids[k] if k < len(img_cls_ids) else 1
                    categories.setdefault(cid, f"class_{cid}")
                    anns_img.append({"id": ann_id, "image_id": img_id_counter, "category_id": cid,
                                      "segmentation": polys, "area": area, "bbox": [x, y, bw_b, bh_b],
                                      "iscrowd": 0, "score": float(scores_np[k]) if k < len(scores_np) else 1.0})
                    coco_annotations.append(anns_img[-1])
                    ann_id += 1
                    masks_vis.append(mask_bin)
                    boxes_vis.append((x, y, x + bw_b - 1, y + bh_b - 1))
                    labels_vis.append(cid)

                if not masks_vis:
                    continue

                info = {"id": img_id_counter, "file_name": path.name, "width": w, "height": h}
                coco_images.append(info)
                img_id_counter += 1

                save_coco_per_image(split_json_dir, info, anns_img, categories)

                combined = np.zeros((h, w), dtype=np.uint8)
                for m in masks_vis:
                    combined[m > 0] = 255
                Image.fromarray(combined, 'L').save(split_mask_dir / f"{path.stem}_mask.png")
                overlay_masks(img, masks_vis, boxes_vis, labels_vis, args.alpha).save(split_vis_dir / f"{path.stem}_sam3.png", quality=95)
                processed += 1

        if coco_images:
            cdata = {"images": coco_images, "annotations": coco_annotations,
                     "categories": [{"id": c, "name": n} for c, n in sorted(categories.items())]}
            json_path = output_root / split_name / "Instance" / f"instances_{split_name}.json"
            json_path.write_text(json.dumps(cdata, indent=2), encoding="utf-8")
            print(f"  [{split_name}] {processed} images, {len(coco_annotations)} annots" + (f" ({skipped} skipped)" if skipped else ""))

        total_processed += processed
        total_annotations += len(coco_annotations)

    return total_processed, total_annotations


def process_yolo_task(args, model, collate, transform, postprocessor, name, cfg):
    """Process a single YOLO-type folder (has best.pt + images) or text-only folder."""
    yolo_path = cfg["yolo_model"]
    image_dir = cfg["image_dir"]
    instance_name = cfg.get("instance_name", "Instance")
    output_root = Path(args.output_dir) / name

    mode_label = "YOLO" if yolo_path else "Text-only"
    print(f"\n{'='*60}")
    print(f"[{name}] {mode_label} mode")
    if yolo_path:
        print(f"  YOLO model: {yolo_path}")
    print(f"  Image dir: {image_dir}")
    if args.prompt:
        print(f"  Prompt: {args.prompt}")
    print(f"  Output: {output_root}")
    print(f"{'='*60}")

    yolo_model = None
    if yolo_path:
        from ultralytics import YOLO
        yolo_model = YOLO(str(yolo_path))
        print(f"  YOLO model loaded")

    json_dir = output_root / instance_name / "label"
    mask_dir = output_root / instance_name / "mask"
    vis_dir = output_root / instance_name / "vis"
    for d in [json_dir, mask_dir, vis_dir]:
        d.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in image_dir.glob("*") if p.suffix.lower().lstrip(".") in IMAGE_EXT])
    if not image_paths:
        image_paths = sorted([p for p in image_dir.rglob("*") if p.suffix.lower().lstrip(".") in IMAGE_EXT])
    if not image_paths:
        print(f"  No images in {image_dir}")
        return 0, 0
    print(f"\n  {len(image_paths)} images")

    all_dps, all_imgs, all_qids, all_paths, all_yolo_boxes = [], [], [], [], []
    for img_path in tqdm(image_paths, desc="  Loading", unit="img"):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        boxes = run_yolo_inference(yolo_model, img, conf=args.yolo_conf, classes=args.yolo_classes)
        dp = create_empty_datapoint()
        set_image(dp, img)
        if boxes:
            qid = add_box_prompt(dp, boxes, args.prompt if args.prompt else "object")
        elif args.prompt:
            qid = add_text_prompt(dp, args.prompt)
        else:
            continue
        dp = transform(dp)
        all_dps.append(dp)
        all_imgs.append(img)
        all_qids.append(qid)
        all_paths.append(img_path)
        all_yolo_boxes.append([tuple(b) for b in boxes] if boxes else [])

    if not all_dps:
        print("  No valid images")
        return 0, 0

    categories, coco_images, coco_annotations = {}, [], []
    ann_id, img_id_counter = 1, 1
    processed, skipped = 0, 0

    for i in range(0, len(all_dps), args.batch_size):
        batch_dps = all_dps[i:i+args.batch_size]
        batch_imgs = all_imgs[i:i+args.batch_size]
        batch_qids = all_qids[i:i+args.batch_size]
        batch_paths = all_paths[i:i+args.batch_size]
        batch_yolo_boxes = all_yolo_boxes[i:i+args.batch_size]

        b = collate(batch_dps, dict_key="dummy")["dummy"]
        from sam3.model.utils.misc import copy_data_to_device
        b = copy_data_to_device(b, torch.device("cuda"), non_blocking=True)
        with torch.no_grad():
            output = model(b)
        results = postprocessor.process_results(output, b.find_metadatas)

        for j in range(len(batch_dps)):
            img, qid = batch_imgs[j], batch_qids[j]
            path = Path(batch_paths[j])
            w, h = img.size
            yolo_boxes = batch_yolo_boxes[j]

            if args.skip_existing and (json_dir / f"{path.stem}.json").exists() and (mask_dir / f"{path.stem}_mask.png").exists():
                skipped += 1
                continue

            r = results.get(qid)
            if r is None or r.get("masks") is None or r.get("scores") is None:
                continue

            scores_np = r["scores"].float().detach().cpu().numpy().flatten()
            masks_np = r["masks"].float().detach().cpu().numpy().astype(np.uint8)
            if masks_np.ndim == 4:
                masks_np = masks_np.squeeze(1)
            elif masks_np.ndim == 2:
                masks_np = masks_np[None]

            # Get predicted boxes if available, else compute from mask
            boxes_xyxy = r.get("boxes")
            if boxes_xyxy is not None and isinstance(boxes_xyxy, torch.Tensor):
                boxes_np = boxes_xyxy.float().detach().cpu().numpy()
            else:
                boxes_np = None

            masks_vis, boxes_vis, labels_vis, anns_img = [], [], [], []
            for k in range(masks_np.shape[0]):
                mask_bin = masks_np[k].astype(np.uint8)
                if float(mask_bin.sum()) < args.min_mask_area:
                    continue
                if args.morph_kernel_size > 0:
                    mask_bin = smooth_mask(mask_bin, args.morph_kernel_size, args.morph_iterations)
                polys = mask_to_polygons(mask_bin, args.min_poly_area, args.epsilon_ratio, args.keep_largest_only)
                if not polys:
                    continue
                area = float(mask_bin.sum())
                x, y, bw_b, bh_b = mask_to_xywh(mask_bin)
                if args.iou_threshold > 0 and yolo_boxes:
                    mask_xyxy = (x, y, x + bw_b - 1, y + bh_b - 1)
                    max_iou = max(compute_bbox_iou(mask_xyxy, yb) for yb in yolo_boxes)
                    if max_iou < args.iou_threshold:
                        continue
                if boxes_np is not None and k < len(boxes_np):
                    box_xyxy = tuple(boxes_np[k].tolist())
                else:
                    box_xyxy = (x, y, x + bw_b - 1, y + bh_b - 1)

                cid = 1
                categories.setdefault(cid, "object")
                anns_img.append({"id": ann_id, "image_id": img_id_counter, "category_id": cid,
                                  "segmentation": polys, "area": area, "bbox": [x, y, bw_b, bh_b],
                                  "iscrowd": 0, "score": float(scores_np[k]) if k < len(scores_np) else 1.0})
                coco_annotations.append(anns_img[-1])
                ann_id += 1
                masks_vis.append(mask_bin)
                boxes_vis.append(box_xyxy)
                labels_vis.append(cid)

            if not masks_vis:
                continue

            info = {"id": img_id_counter, "file_name": path.name, "width": w, "height": h}
            coco_images.append(info)
            img_id_counter += 1

            save_coco_per_image(json_dir, info, anns_img, categories)

            combined = np.zeros((h, w), dtype=np.uint8)
            for m in masks_vis:
                combined[m > 0] = 255
            Image.fromarray(combined, 'L').save(mask_dir / f"{path.stem}_mask.png")
            overlay_masks(img, masks_vis, boxes_vis, labels_vis, args.alpha).save(vis_dir / f"{path.stem}_sam3.png", quality=95)
            processed += 1

    if coco_images:
        cdata = {"images": coco_images, "annotations": coco_annotations,
                 "categories": [{"id": c, "name": n} for c, n in sorted(categories.items())]}
        (output_root / instance_name / "instances_default.json").write_text(json.dumps(cdata, indent=2), encoding="utf-8")
        print(f"  [{name}] {processed} images, {len(coco_annotations)} annots" + (f" ({skipped} skipped)" if skipped else ""))

    return processed, len(coco_annotations)


def process_flat_folder(args, model, collate, transform, postprocessor, instance_name="Instance"):
    """Original flat folder / single YOLO model / text prompt mode."""
    IMAGE_EXT_LIST = list(IMAGE_EXT)
    image_paths = []
    for ext in IMAGE_EXT_LIST:
        image_paths += glob.glob(os.path.join(args.image_folder, f"*.{ext}"))
        image_paths += glob.glob(os.path.join(args.image_folder, f"*.{ext.upper()}"))
    print(f"Found {len(image_paths)} images")

    output_root = Path(args.output_dir)
    json_dir = output_root / instance_name / "label"
    mask_dir = output_root / instance_name / "mask"
    vis_dir = output_root / instance_name / "vis"
    for d in [json_dir, mask_dir, vis_dir]:
        d.mkdir(parents=True, exist_ok=True)

    yolo_model = None
    if args.yolo_model:
        from ultralytics import YOLO
        yolo_model = YOLO(args.yolo_model)
        print(f"YOLO model loaded: {args.yolo_model}")

    all_dps, all_imgs, all_qids, all_paths, all_yolo_boxes = [], [], [], [], []
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        dp = create_empty_datapoint()
        set_image(dp, img)
        boxes = run_yolo_inference(yolo_model, img, conf=args.yolo_conf, classes=args.yolo_classes) if yolo_model else []
        if args.prompt and boxes:
            qid = add_box_prompt(dp, boxes, args.prompt)
        elif boxes:
            qid = add_box_prompt(dp, boxes, args.prompt if args.prompt else "object")
        elif args.prompt:
            qid = add_text_prompt(dp, args.prompt)
        else:
            continue
        dp = transform(dp)
        all_dps.append(dp)
        all_imgs.append(img)
        all_qids.append(qid)
        all_paths.append(img_path)
        all_yolo_boxes.append([tuple(b) for b in boxes] if boxes else [])

    if not all_dps:
        print("No valid images, exiting")
        return 0, 0

    categories, coco_images, coco_annotations = {}, [], []
    ann_id, img_id_counter = 1, 1
    processed, skipped = 0, 0

    for i in range(0, len(all_dps), args.batch_size):
        batch_dps = all_dps[i:i+args.batch_size]
        batch_imgs = all_imgs[i:i+args.batch_size]
        batch_qids = all_qids[i:i+args.batch_size]
        batch_paths = all_paths[i:i+args.batch_size]
        batch_yolo_boxes = all_yolo_boxes[i:i+args.batch_size]

        b = collate(batch_dps, dict_key="dummy")["dummy"]
        from sam3.model.utils.misc import copy_data_to_device
        b = copy_data_to_device(b, torch.device("cuda"), non_blocking=True)
        with torch.no_grad():
            output = model(b)
        results = postprocessor.process_results(output, b.find_metadatas)

        for j in range(len(batch_dps)):
            img, qid = batch_imgs[j], batch_qids[j]
            path = Path(batch_paths[j])
            w, h = img.size
            yolo_boxes = batch_yolo_boxes[j]

            if args.skip_existing and (json_dir / f"{path.stem}.json").exists() and (mask_dir / f"{path.stem}_mask.png").exists():
                skipped += 1
                continue

            r = results.get(qid)
            if r is None or r.get("masks") is None or r.get("scores") is None:
                continue

            scores_np = r["scores"].float().detach().cpu().numpy().flatten()
            masks_np = r["masks"].float().detach().cpu().numpy().astype(np.uint8)
            if masks_np.ndim == 4:
                masks_np = masks_np.squeeze(1)
            elif masks_np.ndim == 2:
                masks_np = masks_np[None]

            boxes_xyxy = r.get("boxes")
            if boxes_xyxy is not None and isinstance(boxes_xyxy, torch.Tensor):
                boxes_np = boxes_xyxy.float().detach().cpu().numpy()
            else:
                boxes_np = None

            masks_vis, boxes_vis, labels_vis, anns_img = [], [], [], []
            for k in range(masks_np.shape[0]):
                mask_bin = masks_np[k].astype(np.uint8)
                if float(mask_bin.sum()) < args.min_mask_area:
                    continue
                if args.morph_kernel_size > 0:
                    mask_bin = smooth_mask(mask_bin, args.morph_kernel_size, args.morph_iterations)
                polys = mask_to_polygons(mask_bin, args.min_poly_area, args.epsilon_ratio, args.keep_largest_only)
                if not polys:
                    continue
                area = float(mask_bin.sum())
                x, y, bw_b, bh_b = mask_to_xywh(mask_bin)
                if args.iou_threshold > 0 and yolo_boxes:
                    mask_xyxy = (x, y, x + bw_b - 1, y + bh_b - 1)
                    max_iou = max(compute_bbox_iou(mask_xyxy, yb) for yb in yolo_boxes)
                    if max_iou < args.iou_threshold:
                        continue
                if boxes_np is not None and k < len(boxes_np):
                    box_xyxy = tuple(boxes_np[k].tolist())
                else:
                    box_xyxy = (x, y, x + bw_b - 1, y + bh_b - 1)
                cid = 1
                categories.setdefault(cid, "object")
                anns_img.append({"id": ann_id, "image_id": img_id_counter, "category_id": cid,
                                  "segmentation": polys, "area": area, "bbox": [x, y, bw_b, bh_b],
                                  "iscrowd": 0, "score": float(scores_np[k]) if k < len(scores_np) else 1.0})
                coco_annotations.append(anns_img[-1])
                ann_id += 1
                masks_vis.append(mask_bin)
                boxes_vis.append(box_xyxy)
                labels_vis.append(cid)

            if not masks_vis:
                continue

            info = {"id": img_id_counter, "file_name": path.name, "width": w, "height": h}
            coco_images.append(info)
            img_id_counter += 1

            save_coco_per_image(json_dir, info, anns_img, categories)
            combined = np.zeros((h, w), dtype=np.uint8)
            for m in masks_vis:
                combined[m > 0] = 255
            Image.fromarray(combined, 'L').save(mask_dir / f"{path.stem}_mask.png")
            overlay_masks(img, masks_vis, boxes_vis, labels_vis, args.alpha).save(vis_dir / f"{path.stem}_sam3.png", quality=95)
            processed += 1

    if coco_images:
        cdata = {"images": coco_images, "annotations": coco_annotations,
                 "categories": [{"id": c, "name": n} for c, n in sorted(categories.items())]}
        (output_root / instance_name / "instances_default.json").write_text(json.dumps(cdata, indent=2), encoding="utf-8")
        print(f"\nProcessed: {processed} images, {len(coco_annotations)} annotations" + (f" ({skipped} skipped)" if skipped else ""))

    return processed, len(coco_annotations)


# ===================== Main =====================

def main():
    parser = argparse.ArgumentParser(description="SAM3 批量图片分割推理 — 支持 auto-detect / dataset / flat folder")
    parser.add_argument("--dataset-root", type=str, default=None,
                        help="根目录: 自动探测子文件夹, 支持 dataset/ 分划 或 best.pt+图片 模式")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val", "test"],
                        help="要处理的split列表 (dataset 模式)")
    parser.add_argument("--image-folder", type=str, default=None, help="输入图片文件夹 (flat folder 模式)")
    parser.add_argument("--prompt", nargs="*", default=None, help="文本提示词（可选）")
    parser.add_argument("--yolo-model", type=str, default=None, help="YOLO 模型权重路径（可选, flat folder 模式）")
    parser.add_argument("--yolo-conf", type=float, default=0.3)
    parser.add_argument("--yolo-classes", type=int, nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--min-mask-area", type=float, default=200.0)
    parser.add_argument("--min-poly-area", type=float, default=16.0)
    parser.add_argument("--keep-largest-only", action="store_true")
    parser.add_argument("--morph-kernel-size", type=int, default=5)
    parser.add_argument("--morph-iterations", type=int, default=1)
    parser.add_argument("--epsilon-ratio", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.75,
                        help="SAM3 mask 与 YOLO box 的 IoU 阈值 (0=关闭)，低于此值的 mask 将被丢弃")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="/home/model/sam3_pth/sam3pt/sam3.pt")
    args = parser.parse_args()

    if args.prompt:
        args.prompt = " ".join(args.prompt)
    if not args.dataset_root and not args.image_folder:
        print("错误: 必须指定 --dataset-root 或 --image-folder 之一")
        sys.exit(1)
    if args.dataset_root and args.image_folder:
        print("错误: --dataset-root 和 --image-folder 不能同时使用")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Model setup ---
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    torch.inference_mode().__enter__()

    from sam3.train.data.collator import collate_fn_api as collate
    from sam3 import build_sam3_image_model
    from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
    from sam3.eval.postprocessors import PostProcessImage

    bpe_path = os.path.join(sam3_root, "assets/bpe_simple_vocab_16e6.txt.gz")
    model = build_sam3_image_model(checkpoint_path=args.checkpoint, bpe_path=bpe_path, load_from_HF=False).cuda().eval()

    transform = ComposeAPI([
        RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
        ToTensorAPI(), NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    postprocessor = PostProcessImage(max_dets_per_img=-1, iou_type="segm",
                                     use_original_sizes_box=True, use_original_sizes_mask=True,
                                     convert_mask_to_rle=False, detection_threshold=0.5, to_cpu=False)

    # --- Run ---
    if args.dataset_root:
        root = Path(args.dataset_root)
        tasks = scan_material_folders(root, args)

        if not tasks:
            print(f"未找到任何物料文件夹: {root}")
            sys.exit(1)

        print(f"\n找到 {len(tasks)} 个物料文件夹:")
        for name, mode, cfg in tasks:
            print(f"  [{mode}] {name}")

        total_p, total_a = 0, 0
        t0 = time.time()
        for name, mode, cfg in tasks:
            if mode == "dataset":
                p, a = process_dataset_task(args, model, collate, transform, postprocessor, name, cfg)
            else:
                p, a = process_yolo_task(args, model, collate, transform, postprocessor, name, cfg)
            total_p += p
            total_a += a

        t = time.time() - t0
        print(f"\n{'='*60}")
        print(f"[总计] {total_p} 图像, {total_a} 标注, {t:.1f}s ({t/total_p:.1f}s/图)" if total_p else f"[总计] 0 图像")
        print(f"{'='*60}")
    else:
        t0 = time.time()
        p, a = process_flat_folder(args, model, collate, transform, postprocessor)
        t = time.time() - t0
        print(f"\nTotal: {p} images, {a} anns, {t:.1f}s ({t/p:.1f}s/img)" if p else "\nNo results")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
