import os
import re
import tempfile
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

from vreason_bench.utils import (
    build_unified_results,
    get_eval_directories,
    save_json,
)


def _crop_box(arr: np.ndarray, xyxy: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    if xyxy is None:
        return arr
    x0, y0, x1, y1 = [int(i) for i in xyxy]
    return arr[y0:y1, x0:x1, ...] if arr.ndim == 3 else arr[y0:y1, x0:x1]

def _flatten_to_white(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError("Expected HxWxC array")
    if arr.shape[2] == 3:
        return arr
    if arr.shape[2] == 4:
        rgb = arr[..., :3].astype(np.float32)
        a   = arr[..., 3:4].astype(np.float32) / 255.0
        out = rgb * a + 255.0 * (1.0 - a)
        return out.clip(0, 255).astype(np.uint8)
    if arr.shape[2] == 1:
        return np.repeat(arr, 3, axis=2)
    raise ValueError("Unsupported channel count")


def _read_image_rgb_and_mask(path: str, size=None) -> Tuple[np.ndarray, np.ndarray]:
    im = Image.open(path)
    if size is not None:
        # W, H = size
        im = im.resize(size, resample=Image.BILINEAR)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        arr = np.array(im.convert("RGBA"))
        alpha = arr[..., 3]
        valid_mask = (alpha > 0).astype(bool)
        rgb = _flatten_to_white(arr)
        return rgb, valid_mask
    arr = np.array(im.convert("RGB"))
    valid_mask = (arr != 255).all(axis=-1)
    return arr, valid_mask


def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

def _color_dist_l2(rgb: np.ndarray, color=(255, 255, 255)) -> np.ndarray:
    c = np.array(color, np.float32)
    return np.linalg.norm(rgb.astype(np.float32) - c[None, None, :], axis=2)

def _grid_points_in_box(x0, y0, x1, y1, n_side: int, jitter: float = None):
    xs = np.linspace(x0, x1, n_side+2)[1:-1]
    ys = np.linspace(y0, y1, n_side+2)[1:-1]
    xv, yv = np.meshgrid(xs, ys)
    pts = np.stack([xv.ravel(), yv.ravel()], axis=-1)
    if jitter:
        cell_x = (x1 - x0) / (n_side + 1)
        cell_y = (y1 - y0) / (n_side + 1)
        noise = np.random.uniform(-jitter, jitter, pts.shape)
        pts[:, 0] += noise[:, 0] * cell_x
        pts[:, 1] += noise[:, 1] * cell_y
    return pts

def _sam2_refine(ref_rgb, ref_mask, gen_rgb, box_xyxy, model_cfg=None, ckpt=None, device=None) -> np.ndarray:
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    x0, y0, x1, y1 = box_xyxy
    gen_crop = np.zeros_like(gen_rgb) + 255
    gen_crop[y0:y1, x0:x1] = gen_rgb[y0:y1, x0:x1]

    vp = build_sam2_video_predictor(model_cfg, ckpt, device=device)

    with tempfile.TemporaryDirectory() as tmpd:
        Image.fromarray(ref_rgb).save(os.path.join(tmpd, "00000.jpg"))
        Image.fromarray(gen_crop).save(os.path.join(tmpd, "00001.jpg"))
        with torch.inference_mode(), torch.autocast(device if device == "cuda" else "cpu", dtype=torch.bfloat16, enabled=(device == "cuda")):
            state = vp.init_state(video_path=tmpd)
            vp.add_new_mask(inference_state=state, frame_idx=0, obj_id=1, mask=ref_mask)
            for f_idx, obj_ids, masks in vp.propagate_in_video(state, start_frame_idx=0):
                if f_idx != 1:
                    continue
                if masks is None or len(obj_ids) == 0:
                    break
                masks = masks.cpu().numpy()
                masks = (masks > 0.0).squeeze(-3)
                if masks.ndim == 2:
                    masks = masks[None, ...]
                if masks.all() or not masks.any():
                    break
                return masks
    del vp, state
    sam = build_sam2(model_cfg, ckpt, device=device)
    predictor = SAM2ImagePredictor(sam)
    pos_pts = _grid_points_in_box(x0, y0, x1, y1, n_side=3)
    masks_list = []
    with torch.inference_mode(), torch.autocast(device if device == "cuda" else "cpu", dtype=torch.bfloat16, enabled=(device == "cuda")):
        predictor.set_image(gen_crop)
        for i in range(len(pos_pts)):
            masks, _scores, _ = predictor.predict(
                point_coords=pos_pts[i:i+1],
                point_labels=np.ones([1,]),
                multimask_output=True,
            )
            if masks.ndim == 2:
                masks = masks[None, ...]
            masks_list.append(masks)
        masks = np.concatenate(masks_list, axis=0)
        masks = masks.astype(bool)
    return masks

def _calculate_mask_metrics(rgb: np.ndarray, mask: np.ndarray) -> Dict:
    """
    Calculate simple metrics for a masked region.
    
    Returns:
        dict with:
            - mean_color: (R, G, B) average color in mask
            - bbox_size: (width, height) of bounding box
            - area: number of pixels in mask
    """
    bbox = _bbox_from_mask(mask)
    if bbox is None:
        return {
            "mean_color": (0, 0, 0),
            "bbox_size": (0, 0),
            "area": 0
        }
    
    x0, y0, x1, y1 = bbox
    width = int(x1 - x0)
    height = int(y1 - y0)
    
    masked_pixels = rgb[mask]
    if len(masked_pixels) > 0:
        mean_color = tuple(int(x) for x in masked_pixels.mean(axis=0).astype(int))
    else:
        mean_color = (0, 0, 0)
    
    area = int(mask.sum())
    
    return {
        "mean_color": mean_color,
        "bbox_size": (width, height),
        "area": area
    }

def _compare_masks(pred_rgb: np.ndarray, pred_mask: np.ndarray, 
                   gt_rgb: np.ndarray, gt_mask: np.ndarray, init_rgb: np.ndarray) -> Tuple[float, Dict]:
    """
    Compare two masks based on:
    1. Bounding box size similarity
    2. Average color similarity
    
    Returns:
        score: float between 0-1
        details: dict with comparison metrics
    """
    pred_metrics = _calculate_mask_metrics(pred_rgb, pred_mask)
    gt_metrics = _calculate_mask_metrics(gt_rgb, gt_mask)
    
    pred_w, pred_h = pred_metrics["bbox_size"]
    gt_w, gt_h = gt_metrics["bbox_size"]
    
    if pred_w == 0 or pred_h == 0 or gt_w == 0 or gt_h == 0:
        return 0.0, {"error": "empty_mask"}
    
    pred_aspect = pred_w / pred_h
    gt_aspect = gt_w / gt_h
    aspect_diff = abs(pred_aspect - gt_aspect) / max(pred_aspect, gt_aspect)
    aspect_score = float(max(0, 1 - aspect_diff))
    
    pred_area = pred_metrics["area"]
    gt_area = gt_metrics["area"]
    size_ratio = min(pred_area, gt_area) / max(pred_area, gt_area)
    size_score = float(size_ratio)
    
    pred_color = np.array(pred_metrics["mean_color"], dtype=float)
    gt_color = np.array(gt_metrics["mean_color"], dtype=float)
    color_dist = np.linalg.norm(pred_color - gt_color) / (255 * np.sqrt(3))  # Normalize to [0,1]
    color_score = float(max(0, 1 - color_dist))
    init_w, init_h = init_rgb.shape[1], init_rgb.shape[0]

    example_inp = init_rgb[:init_h // 2, :init_w // 2 - 60]
    example_inp_mask = (example_inp != 255).all(axis=-1)
    example_inp = _crop_box(example_inp, _bbox_from_mask(example_inp_mask))

    example_gt = init_rgb[:init_h // 2, init_w // 2 + 60:]
    example_gt_mask = (example_gt != 255).all(axis=-1)
    example_gt = _crop_box(example_gt, _bbox_from_mask(example_gt_mask))

    example_inp_w, example_inp_h = example_inp.shape[1], example_inp.shape[0]
    example_gt_w, example_gt_h = example_gt.shape[1], example_gt.shape[0]
    gt_ratio = (example_gt_w * example_gt_h) / (example_inp_w * example_inp_h)

    init_crop = init_rgb[init_h // 2:, :init_w // 2 - 60, :]
    init_mask = (init_crop != 255).all(axis=-1)
    init_bbox = _bbox_from_mask(init_mask)
    init_crop = _crop_box(init_crop, init_bbox)
    init_w, init_h = init_crop.shape[1], init_crop.shape[0]
    pred_ratio = (pred_w * pred_h) / (init_w * init_h)

    if (gt_ratio < 1 and pred_ratio <= 0.8) or (gt_ratio > 1 and pred_ratio >= 1.2):
        bbox_weight = 0.5
        color_weight = 0.5
        final_score = bbox_weight * aspect_score + color_weight * color_score
    else:
        final_score = 0
    
    details = {
        "pred_metrics": pred_metrics,
        "gt_metrics": gt_metrics,
        "aspect_score": aspect_score,
        "size_score": size_score,
        "gt_ratio": gt_ratio,
        "pred_ratio": pred_ratio,
        "color_score": color_score,
        "final_score": float(final_score)
    }
    
    return float(final_score), details

def _eval_triple(init_rgb: np.ndarray, gen_rgb: np.ndarray, gt_rgb: np.ndarray, valid_mask: np.ndarray,
                 box_frac: Tuple[float, float], sam_ckpt: str, sam_type: str, thr: float = None) -> Dict:
    """
    Evaluate by comparing:
    1. Mask bounding box size (width, height)
    2. Average color within mask
    """
    H, W = gen_rgb.shape[:2]
    h = max(1, int(round(H * box_frac[0])))
    w = max(1, int(round(W * box_frac[1])))
    y1, x1 = H, W
    y0, x0 = H - h, W - w

    if thr:
        gen_rgb[_color_dist_l2(gen_rgb) < thr] = np.array([255, 255, 255], dtype=gen_rgb.dtype)

    pred_masks = _sam2_refine(gt_rgb, valid_mask, gen_rgb, (x0, y0, x1, y1), model_cfg=sam_type, ckpt=sam_ckpt)
    if pred_masks.ndim == 2:
        pred_masks = [pred_masks]

    score_best = None
    details_best = None
    for pred_mask in pred_masks:
        score, details = _compare_masks(gen_rgb, pred_mask, gt_rgb, valid_mask, init_rgb)
        if score_best is None or score > score_best:
            score_best = score
            details_best = details

    report = {
        "box_xyxy": {"x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1)}, 
        "score": float(score_best),
        "details": details_best
    }
    return report


def compute_visual_analogy(
    name: str, 
    local: bool = False, 
    mode: str = 'vreason_bench_standard', 
    **kwargs
) -> Dict:
    """
    Evaluate Visual Analogy using prediction frames stored in the `predictions` folder.
    
    Evaluation method (simplified):
    1. Extract mask from predicted image using SAM2
    2. Compare mask bounding box size (width, height) with GT
    3. Compare average color within mask region with GT
    4. Score = weighted combination of bbox similarity + color similarity
    
    Expects: evaluations/Visual_analogy/{inputs,GT,predictions}/<concept>/<*.png>
    Concepts: 2drotation, colour, reflect, resize
    
    Parameters:
        box_frac: (0.5, 0.5) - detection box fraction for SAM2
        sam_ckpt: "./checkpoints/sam2.1_hiera_large.pt"
        sam_type: "configs/sam2.1/sam2.1_hiera_l.yaml"
        thr: None - threshold for filtering white pixels
    """
    dirs = get_eval_directories("Visual_analogy")
    input_root = dirs["input_dir"]
    gt_root = dirs["gt_dir"]
    predictions_dir = dirs["predictions_dir"]
    eval_results_dir = dirs["eval_results_dir"]
    
    box_frac = kwargs.get('box_frac', (0.5, 0.5))
    sam_ckpt = "./checkpoints/sam2.1_hiera_large.pt"
    sam_type = "configs/sam2.1/sam2.1_hiera_l.yaml"
    thr = None
    
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(eval_results_dir, exist_ok=True)
    
    frame_files: List[str] = []
    for root, _, files in os.walk(predictions_dir):
        for fname in sorted(files):
            if fname.lower().endswith(".png"):
                frame_files.append(os.path.join(root, fname))

    results: Dict[str, Dict] = {}
    scores: List[float] = []

    for frame_path in frame_files:
        base_f = os.path.basename(frame_path)
        stem = os.path.splitext(base_f)[0]

        m = re.match(r"(?P<model>[^_]+)_(?P<idx>\d{2,4})_(?P<seed>seed[0-9A-Za-z]+)$", stem)
        
        model = m.group("model")
        idx_str = m.group("idx")

        rel = os.path.relpath(frame_path, predictions_dir)
        subdir = os.path.dirname(rel)
        concept = subdir.replace(os.sep, "_") if subdir and subdir != "." else ""

        input_path = os.path.join(input_root, subdir, f"{idx_str}.png")
        gt_path = os.path.join(gt_root, subdir, f"{idx_str}.png")

        if not os.path.exists(input_path) or not os.path.exists(gt_path):
            results[frame_path] = {
                "pred_frame": frame_path,
                "error": "input_or_gt_not_found",
                "input_path": input_path,
                "gt_path": gt_path,
                "concept": concept,
            }
            continue

        try:
            init_rgb, _ = _read_image_rgb_and_mask(input_path)
            init_w, init_h = init_rgb.shape[1], init_rgb.shape[0]

            gen_rgb, _ = _read_image_rgb_and_mask(frame_path, size=(init_w, init_h))
            gt_rgb, valid_mask = _read_image_rgb_and_mask(gt_path)

            report = _eval_triple(
                init_rgb,
                gen_rgb,
                gt_rgb,
                valid_mask=valid_mask,
                box_frac=box_frac,
                sam_ckpt=sam_ckpt,
                sam_type=sam_type,
                thr=thr,
            )

            score = report.get("score", 0.0)
            scores.append(score)
            subdir_key = subdir.replace(os.sep, "_") if subdir and subdir != "." else ""
            gt_index = f"{subdir_key}_{idx_str}" if subdir_key else idx_str

            results[frame_path] = {
                "pred_frame": frame_path,
                "input_path": input_path,
                "gt_path": gt_path,
                "score": float(score),
                "box_xyxy": report.get("box_xyxy"),
                "details": report.get("details"),
                "gt_index": gt_index,
                "concept": concept,
                "model": model,
            }
        except Exception as e:
            results[frame_path] = {
                "pred_frame": frame_path,
                "error": f"evaluation_failed: {str(e)}",
                "input_path": input_path,
                "gt_path": gt_path,
            }

    unified = build_unified_results(
        results, 
        score_key="score", 
        threshold=0.97, 
        predictions_dir=predictions_dir
    )
    
    unified["aggregate"]["task_metrics"] = {
        "average_score": float(sum(scores) / len(scores)) if scores else 0.0,
        "evaluation_method": "bbox_size_and_color_average",
        "scoring_components": {
            "bbox_weight": 0.5,
            "color_weight": 0.5,
            "bbox_includes": ["aspect_ratio", "size_ratio"],
            "color_metric": "RGB_L2_distance"
        },
        "sam_checkpoint": sam_ckpt,
        "sam_type": sam_type,
        "box_frac": list(box_frac),
        "threshold": float(thr) if thr else None,
    }
    
    save_json(unified, os.path.join(eval_results_dir, "visual_analogy_eval.json"))
    return unified

