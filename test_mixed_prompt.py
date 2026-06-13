#!/usr/bin/env python3
"""
最小测试: SAM3 混合提示 (box + text) 是否正常工作。

用法 (在 sam3_6000d 环境中):
  python /home/model/work/sam3_facebook/test_mixed_prompt.py \
      --image /path/to/test.jpg \
      --bbox 100 100 300 300 \
      --text "红色塑料袋包装的辣椒" \
      --checkpoint /home/model/sam3_pth/sam3pt/sam3.pt
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import sam3

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


def mask_to_polygons(mask, min_area=16.0, epsilon_ratio=0.01):
    mask_uint8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_data = [(c, cv2.contourArea(c)) for c in contours if cv2.contourArea(c) >= min_area]
    if not contour_data:
        return []
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


def main():
    parser = argparse.ArgumentParser(description="SAM3 混合提示最小测试")
    parser.add_argument("--image", type=str, required=True, help="测试图片路径")
    parser.add_argument("--bbox", type=int, nargs=4, required=True,
                        help="Bounding box: x y w h (像素坐标)")
    parser.add_argument("--text", type=str, default="object",
                        help="文本提示描述 (如 '红色塑料袋包装的辣椒')")
    parser.add_argument("--checkpoint", type=str,
                        default="/home/model/sam3_pth/sam3pt/sam3.pt",
                        help="SAM3 checkpoint")
    parser.add_argument("--output-dir", type=str, default="./test_mixed_output",
                        help="输出目录")
    args = parser.parse_args()

    # Check image exists
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"[ERROR] Image not found: {img_path}")
        sys.exit(1)

    bbox_xywh = args.bbox  # [x, y, w, h]
    text_desc = args.text

    print(f"=== SAM3 Mixed Prompt Test ===")
    print(f"  Image:   {img_path}")
    print(f"  BBox:    xywh={bbox_xywh}")
    print(f"  Text:    '{text_desc}'")
    print(f"  Model:   {args.checkpoint}")

    # Output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load SAM3 Model ----
    print(f"\n[1/4] Loading SAM3 model...")
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
    model = build_sam3_image_model(
        checkpoint_path=args.checkpoint,
        bpe_path=bpe_path,
        load_from_HF=False
    ).cuda().eval()

    transform = ComposeAPI([
        RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
        ToTensorAPI(), NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    postprocessor = PostProcessImage(
        max_dets_per_img=-1, iou_type="segm",
        use_original_sizes_box=True, use_original_sizes_mask=True,
        convert_mask_to_rle=False, detection_threshold=0.5, to_cpu=False
    )

    # ---- Load Image ----
    print(f"[2/4] Loading image: {img_path}")
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    print(f"  Size: {w}x{h}")

    # Convert bbox from xywh to xyxy
    x, y, bw, bh = bbox_xywh
    xyxy = [float(x), float(y), float(x + bw - 1), float(y + bh - 1)]
    print(f"  BBox xyxy: {xyxy}")

    # ---- Build Mixed Prompt (box + text) ----
    print(f"[3/4] Running SAM3 inference with mixed prompt...")
    dp = create_empty_datapoint()
    set_image(dp, img)
    qid = add_box_prompt(dp, [xyxy], text_prompt=text_desc)
    dp = transform(dp)

    b = collate([dp], dict_key="dummy")["dummy"]
    from sam3.model.utils.misc import copy_data_to_device
    b = copy_data_to_device(b, torch.device("cuda"), non_blocking=True)
    with torch.no_grad():
        output = model(b)
    results = postprocessor.process_results(output, b.find_metadatas)

    # ---- Parse Results ----
    print(f"[4/4] Parsing results...")
    r = results.get(qid)
    if r is None or r.get("masks") is None:
        print("[FAIL] SAM3 returned no masks for this prompt")
        # Try to get any results
        print(f"  Available query IDs: {list(results.keys())}")
        sys.exit(1)

    scores_np = r["scores"].float().detach().cpu().numpy().flatten()
    masks_np = r["masks"].float().detach().cpu().numpy().astype(np.uint8)
    if masks_np.ndim == 4:
        masks_np = masks_np.squeeze(1)
    elif masks_np.ndim == 2:
        masks_np = masks_np[None]

    print(f"\n=== Results ===")
    print(f"  Masks found: {masks_np.shape[0]}")

    valid_masks = 0
    for k in range(masks_np.shape[0]):
        mask_bin = masks_np[k].astype(np.uint8)
        mask_area = float(mask_bin.sum())
        score = float(scores_np[k]) if k < len(scores_np) else 0.0

        if mask_area < 200:
            print(f"  Mask {k}: area={mask_area:.0f} score={score:.4f} (SKIP: too small)")
            continue

        polys = mask_to_polygons(mask_bin)
        bbox_xywh_out = mask_to_xywh(mask_bin)
        valid_masks += 1
        print(f"  Mask {k}: area={mask_area:.0f} score={score:.4f} polygons={len(polys)} bbox_out={[round(v,1) for v in bbox_xywh_out]}")

        # Save mask as PNG
        mask_img = Image.fromarray(mask_bin * 255, 'L')
        mask_path = out_dir / f"mask_{k}.png"
        mask_img.save(str(mask_path))
        print(f"      Saved: {mask_path}")

        # Save overlay
        overlay = img.copy().convert("RGBA")
        overlay_np = np.array(overlay)
        overlay_np[mask_bin > 0, 0] = (overlay_np[mask_bin > 0, 0] * 0.5 + 128).astype(np.uint8)
        overlay_np[mask_bin > 0, 1] = (overlay_np[mask_bin > 0, 1] * 0.5 + 0).astype(np.uint8)
        overlay_np[mask_bin > 0, 2] = (overlay_np[mask_bin > 0, 2] * 0.5 + 0).astype(np.uint8)
        overlay_img = Image.fromarray(overlay_np)
        overlay_path = out_dir / f"overlay_{k}.png"
        overlay_img.save(str(overlay_path))
        print(f"      Overlay: {overlay_path}")

    if valid_masks == 0:
        print("\n[WARN] No valid masks generated. The text+box mixed prompt may not have found matching objects.")
        print("  Try adjusting the bounding box or text description.")
    else:
        print(f"\n[SUCCESS] Mixed prompt (box + text) works! Generated {valid_masks} valid mask(s).")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
