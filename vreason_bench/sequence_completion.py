import os
import json
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .utils import (
    build_unified_results,
    get_eval_directories,
    save_json,
)


def _infer_gt_masks(idx_str: str, subdir: str, gt_root: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Build GT target and mask paths using subdir and idx_str."""
    gt_target = os.path.join(gt_root, subdir, f"{idx_str}.png")
    gt_mask_shape = os.path.join(gt_root, subdir, f"{idx_str}_mask_shape.png")
    gt_mask_background = os.path.join(gt_root, subdir, f"{idx_str}_mask_background.png")
    
    gt_target = gt_target if os.path.exists(gt_target) else None
    gt_mask_shape = gt_mask_shape if os.path.exists(gt_mask_shape) else None
    gt_mask_background = gt_mask_background if os.path.exists(gt_mask_background) else None
    return gt_target, gt_mask_shape, gt_mask_background


def _load_image_as_array(path: str) -> np.ndarray:
    """Load image as numpy array (RGB or L)."""
    img = Image.open(path)
    return np.array(img)


def _check_mask_colors(
    generated_img: np.ndarray,
    mask_shape: np.ndarray,
    mask_background: np.ndarray,
    color_tolerance: int = 10
) -> Dict:
    """
    Check if the generated image satisfies the color constraints:
    - Shape mask regions (mask_shape == 255) should be black (0, 0, 0)
    - Background mask regions (mask_background == 255) should be white (255, 255, 255)
    
    Args:
        generated_img: RGB image array (H, W, 3)
        mask_shape: Grayscale mask array (H, W), 255 indicates shape regions
        mask_background: Grayscale mask array (H, W), 255 indicates background regions
        color_tolerance: Tolerance for color matching (default: 10)
    
    Returns:
        Dictionary with accuracy metrics
    """
    if len(mask_shape.shape) == 3:
        mask_shape = mask_shape[:, :, 0]
    if len(mask_background.shape) == 3:
        mask_background = mask_background[:, :, 0]
    
    h, w = mask_shape.shape
    if generated_img.shape[:2] != (h, w):
        generated_pil = Image.fromarray(generated_img)
        generated_pil = generated_pil.resize((w, h), Image.BILINEAR)
        generated_img = np.array(generated_pil)
    
    shape_mask = mask_shape == 255
    shape_pixel_count = np.sum(shape_mask)
    
    background_mask = mask_background == 255
    background_pixel_count = np.sum(background_mask)
    
    shape_correct = 0
    background_correct = 0
    
    if shape_pixel_count > 0:
        shape_pixels = generated_img[shape_mask]
        black_target = np.array([0, 0, 0])
        shape_correct = np.sum(np.all(np.abs(shape_pixels - black_target) <= color_tolerance, axis=1))
    
    if background_pixel_count > 0:
        background_pixels = generated_img[background_mask]
        white_target = np.array([255, 255, 255])
        background_correct = np.sum(np.all(np.abs(background_pixels - white_target) <= color_tolerance, axis=1))
    
    shape_accuracy = float(shape_correct / shape_pixel_count) if shape_pixel_count > 0 else 0.0
    background_accuracy = float(background_correct / background_pixel_count) if background_pixel_count > 0 else 0.0
    
    total_pixels = shape_pixel_count + background_pixel_count
    if total_pixels > 0:
        overall_accuracy = float((shape_accuracy + background_accuracy) / 2)
    else:
        overall_accuracy = 0.0
    
    return {
        "shape_pixel_count": int(shape_pixel_count),
        "shape_correct": int(shape_correct),
        "shape_accuracy": shape_accuracy,
        "background_pixel_count": int(background_pixel_count),
        "background_correct": int(background_correct),
        "background_accuracy": background_accuracy,
        "total_pixels": int(total_pixels),
        "total_correct": int(shape_correct + background_correct),
        "overall_accuracy": overall_accuracy,
    }


def compute_sequence_completion(
    name: str,
    mode: str = 'vreason_bench_standard',
) -> Dict:
    """
    Evaluate Sequence Completion by reading prediction frames from the `predictions`
    folder and checking if the generated image satisfies the color constraints
    (shape regions black, background regions white).
    
    Expected layout:
      evaluations/Sequence_completion/
        inputs/<idx>.{png,csv}
        GT/<idx>.png, <idx>_mask_shape.png, <idx>_mask_background.png
        predictions/<model>_<idx>_seedK.png  or  sequence_completion_<model>_<idx>_seedK.png
        eval_results/sequence_completion_eval.json
    """
    dirs = get_eval_directories("Sequence_completion")
    gt_root = dirs["gt_dir"]
    predictions_dir = dirs["predictions_dir"]
    eval_results_dir = dirs["eval_results_dir"]
    
    color_tolerance = 10
    
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(eval_results_dir, exist_ok=True)
    
    frame_files: List[str] = []
    for root, _, files in os.walk(predictions_dir):
        for fname in sorted(files):
            if fname.lower().endswith(".png"):
                frame_files.append(os.path.join(root, fname))
    
    results: Dict[str, Dict] = {}
    accuracies: List[float] = []
    shape_accuracies: List[float] = []
    background_accuracies: List[float] = []
    
    for frame_path in frame_files:
        base_f = os.path.basename(frame_path)
        stem = os.path.splitext(base_f)[0]

        # Support two naming conventions:
        #   1) sequence_completion_<model>_<idx>_seedK
        #   2) <model>_<idx>_seedK
        model: Optional[str] = None
        idx_str: Optional[str] = None

        m_new = re.match(
            r"^sequence_completion_(?P<model>[^_]+)_(?P<idx>\d{2,3})_seed(?P<seed>[0-9A-Za-z]+)$",
            stem,
        )
        if m_new:
            model = m_new.group("model")
            idx_str = m_new.group("idx")
        else:
            m_old = re.match(
                r"^(?P<model>[^_]+)_(?P<idx>\d{2,3})_seed(?P<seed>[0-9A-Za-z]+)$",
                stem,
            )
            if m_old:
                model = m_old.group("model")
                idx_str = m_old.group("idx")

        if not (model and idx_str):
            results[frame_path] = {
                "pred_frame": frame_path,
                "error": "invalid_filename",
            }
            continue

        subdir = ""
        
        gt_target, gt_mask_shape, gt_mask_background = _infer_gt_masks(idx_str, subdir, gt_root)
        
        if not gt_mask_shape or not gt_mask_background:
            results[frame_path] = {
                "pred_frame": frame_path,
                "error": "masks_not_found",
                "gt_target": gt_target,
                "gt_mask_shape": gt_mask_shape,
                "gt_mask_background": gt_mask_background,
            }
            continue
        
        generated_img = _load_image_as_array(frame_path)
        mask_shape = _load_image_as_array(gt_mask_shape)
        mask_background = _load_image_as_array(gt_mask_background)
        
        metrics = _check_mask_colors(generated_img, mask_shape, mask_background, color_tolerance)
        
        accuracies.append(metrics["overall_accuracy"])
        shape_accuracies.append(metrics["shape_accuracy"])
        background_accuracies.append(metrics["background_accuracy"])
        
        results[frame_path] = {
            "pred_frame": frame_path,
            "model": model,
            "gt_target": gt_target,
            "gt_mask_shape": gt_mask_shape,
            "gt_mask_background": gt_mask_background,
            "metrics": metrics,
            "score": float(metrics["overall_accuracy"]),
            "gt_index": idx_str,
        }

    
    unified = build_unified_results(
        results, 
        score_key="score", 
        threshold=0.9,
        predictions_dir=predictions_dir
    )
    
    unified["aggregate"]["task_metrics"] = {
        "average_overall_accuracy": float(sum(accuracies) / len(accuracies)) if accuracies else 0.0,
        "average_shape_accuracy": float(sum(shape_accuracies) / len(shape_accuracies)) if shape_accuracies else 0.0,
        "average_background_accuracy": float(sum(background_accuracies) / len(background_accuracies)) if background_accuracies else 0.0,
        "color_tolerance": color_tolerance,
    }
    
    save_json(unified, os.path.join(eval_results_dir, "sequence_completion_eval.json"))
    return unified

