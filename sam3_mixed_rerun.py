#!/usr/bin/env python3
"""
SAM3 混合提示 (box + text) 重分割脚本

输入: JSON 文件, 格式:
  {
    "images": [
      {
        "image_path": "/path/to/img.jpg",
        "width": 1920,
        "height": 1080,
        "prompts": [
          {"bbox": [x, y, w, h], "text": "红色塑料袋包裹的辣椒包"},
          ...
        ]
      },
      ...
    ]
  }

输出: COCO JSON (instances_default.json) + 逐图 JSON
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import sam3

PALETTE: Sequence[Tuple[int, int, int]] = (
    (239, 83, 80), (102, 187, 106), (66, 165, 245), (255, 202, 40),
    (171, 71, 188), (255, 112, 67), (38, 198, 218), (156, 204, 101),
)

GLOBAL_COUNTER = 1

# ---------- Post-processing (reused from batch_01.py) ----------

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


# ---------- SAM3 Helpers ----------

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


# ---------- BBox conversion ----------

def xywh_to_xyxy(bbox_xywh):
    x, y, w, h = bbox_xywh
    return [x, y, x + w - 1, y + h - 1]


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="SAM3 mixed-prompt re-segmentation")
    parser.add_argument("--input-json", type=str, required=True,
                        help="JSON with image_paths and per-box prompts")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for COCO JSONs and vis")
    parser.add_argument("--checkpoint", type=str,
                        default="/home/model/sam3_pth/sam3pt/sam3.pt",
                        help="SAM3 checkpoint path")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--min-mask-area", type=float, default=200.0)
    parser.add_argument("--min-poly-area", type=float, default=16.0)
    parser.add_argument("--morph-kernel-size", type=int, default=5)
    parser.add_argument("--morph-iterations", type=int, default=1)
    parser.add_argument("--epsilon-ratio", type=float, default=0.01)
    parser.add_argument("--keep-largest-only", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    # Load input prompts
    with open(args.input_json, 'r') as f:
        input_data = json.load(f)

    images_list = input_data.get("images", [])
    if not images_list:
        print("[ERROR] No images in input JSON")
        sys.exit(1)

    output_root = Path(args.output_dir)
    instance_dir = output_root / "Instance"
    json_dir = instance_dir / "label"
    mask_dir = instance_dir / "mask"
    vis_dir = instance_dir / "vis"
    for d in [json_dir, mask_dir, vis_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # --- SAM3 Model Setup (reuses batch_01.py patterns) ---
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
    print(f"Loading SAM3 model: {args.checkpoint}")
    model = build_sam3_image_model(checkpoint_path=args.checkpoint, bpe_path=bpe_path, load_from_HF=False).cuda().eval()

    transform = ComposeAPI([
        RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
        ToTensorAPI(), NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    postprocessor = PostProcessImage(max_dets_per_img=-1, iou_type="segm",
                                      use_original_sizes_box=True, use_original_sizes_mask=True,
                                      convert_mask_to_rle=False, detection_threshold=0.5, to_cpu=False)

    # --- Process each image ---
    categories = {}
    coco_images = []
    coco_annotations = []
    ann_id = 1
    img_id_counter = 1
    
    all_dps, all_imgs, all_qids, all_img_metas = [], [], [], []

    print(f"Loading {len(images_list)} images...")
    for img_meta in tqdm(images_list, desc="Loading", unit="img"):
        img_path = img_meta["image_path"]
        width = img_meta.get("width", 0)
        height = img_meta.get("height", 0)
        prompts = img_meta.get("prompts", [])

        if not prompts:
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  Skip {img_path}: {e}")
            continue

        if width == 0 or height == 0:
            width, height = img.size

        dp = create_empty_datapoint()
        set_image(dp, img)

        # 一张图所有 bbox + 同一个文本 → 一次 add_box_prompt 调用
        all_xyxy = []
        text_desc = prompts[0].get("text", "object") if prompts else "object"
        for p in prompts:
            all_xyxy.append(xywh_to_xyxy(p["bbox"]))
        qid = add_box_prompt(dp, all_xyxy, text_prompt=text_desc)

        dp = transform(dp)
        all_dps.append(dp)
        all_imgs.append(img)
        all_qids.append(qid)
        all_img_metas.append(img_meta)

    # --- Batch inference ---
    if not all_dps:
        print("[ERROR] No valid datapoints")
        sys.exit(1)

    print(f"\nRunning SAM3 inference on {len(all_dps)} images...")
    for i in tqdm(range(0, len(all_dps), args.batch_size), desc="Inference", unit="batch"):
        batch_dps = all_dps[i:i+args.batch_size]
        batch_imgs = all_imgs[i:i+args.batch_size]
        batch_qids = all_qids[i:i+args.batch_size]
        batch_metas = all_img_metas[i:i+args.batch_size]

        b = collate(batch_dps, dict_key="dummy")["dummy"]
        from sam3.model.utils.misc import copy_data_to_device
        b = copy_data_to_device(b, torch.device("cuda"), non_blocking=True)
        with torch.no_grad():
            output = model(b)
        results = postprocessor.process_results(output, b.find_metadatas)

        for j in range(len(batch_dps)):
            img = batch_imgs[j]
            qid = batch_qids[j]
            w, h = img.size

            masks_vis, boxes_vis, labels_vis, anns_img = [], [], [], []

            r = results.get(qid)
            if r is not None and r.get("masks") is not None:
                scores_np = r["scores"].float().detach().cpu().numpy().flatten()
                masks_np = r["masks"].float().detach().cpu().numpy().astype(np.uint8)
                if masks_np.ndim == 4:
                    masks_np = masks_np.squeeze(1)
                elif masks_np.ndim == 2:
                    masks_np = masks_np[None]

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
                    cid = 1
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

            file_name = Path(batch_metas[j]["image_path"]).name
            info = {"id": img_id_counter, "file_name": file_name, "width": w, "height": h}
            coco_images.append(info)
            img_id_counter += 1

            save_coco_per_image(json_dir, info, anns_img, categories)

            try:
                combined = np.zeros((h, w), dtype=np.uint8)
                for m in masks_vis:
                    combined[m > 0] = 255
                Image.fromarray(combined, 'L').save(mask_dir / f"{file_name.rsplit('.',1)[0]}_mask.png")
                overlay_masks(img, masks_vis, boxes_vis, labels_vis, args.alpha).save(
                    vis_dir / f"{file_name.rsplit('.',1)[0]}_sam3.png", quality=95)
            except Exception as e:
                print(f"  Vis save error for {file_name}: {e}")

    # --- Save merged COCO JSON ---
    if coco_images:
        cdata = {"images": coco_images, "annotations": coco_annotations,
                 "categories": [{"id": c, "name": n} for c, n in sorted(categories.items())]}
        out_path = output_root / "instances_default.json"
        out_path.write_text(json.dumps(cdata, indent=2), encoding="utf-8")
        print(f"\n[Done] {len(coco_images)} images, {len(coco_annotations)} annotations")
        print(f"  Output: {out_path}")
    else:
        print("[WARN] No results produced")
        sys.exit(1)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
