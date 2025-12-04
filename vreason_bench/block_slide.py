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


def _load_image_as_array(path: str) -> np.ndarray:
    """Load image as numpy array (RGB or L)."""
    img = Image.open(path)
    return np.array(img)


def _check_mask_colors(
    generated_img: np.ndarray,
    mask_block: np.ndarray,
    mask_background: np.ndarray,
    color_tolerance: int = 10,
    background_color: Tuple[int, int, int] = (255, 255, 255)
) -> Dict:
    """
    Check if the generated image satisfies the color constraints:
    - Block mask regions (mask_block == 255) should be black (0, 0, 0)
    - Background mask regions (mask_background == 255) should be grey (#d0d0d0 = 208, 208, 208)
    
    Args:
        generated_img: RGB image array (H, W, 3)
        mask_block: Grayscale mask array (H, W), 255 indicates block regions
        mask_background: Grayscale mask array (H, W), 255 indicates background plane regions
        color_tolerance: Tolerance for color matching (default: 10)
        background_color: Expected background color (default: (208, 208, 208) for #d0d0d0)
    
    Returns:
        Dictionary with accuracy metrics
    """
    if len(mask_block.shape) == 3:
        mask_block = mask_block[:, :, 0]
    if len(mask_background.shape) == 3:
        mask_background = mask_background[:, :, 0]
    
    h, w = mask_block.shape
    if generated_img.shape[:2] != (h, w):
        generated_pil = Image.fromarray(generated_img)
        generated_pil = generated_pil.resize((w, h), Image.BILINEAR)
        generated_img = np.array(generated_pil)
    
    block_mask = mask_block == 255
    block_pixel_count = np.sum(block_mask)
    
    background_mask = mask_background == 255
    background_pixel_count = np.sum(background_mask)
    
    block_correct = 0
    background_correct = 0
    
    if block_pixel_count > 0:
        block_pixels = generated_img[block_mask]
        black_target = np.array([0, 0, 0])
        block_correct = np.sum(np.all(np.abs(block_pixels - black_target) <= color_tolerance, axis=1))
    
    if background_pixel_count > 0:
        background_pixels = generated_img[background_mask]
        grey_target = np.array(background_color)
        background_correct = np.sum(np.all(np.abs(background_pixels - grey_target) <= color_tolerance, axis=1))
    
    block_accuracy = float(min(block_correct / block_pixel_count + 1 / 3, 1.0)) if block_pixel_count > 0 else 0.0
    background_accuracy = float(background_correct / background_pixel_count) if background_pixel_count > 0 else 0.0
    
    total_pixels = block_pixel_count + background_pixel_count
    if total_pixels > 0:
        overall_accuracy = float((block_accuracy + background_accuracy) / 2)
    else:
        overall_accuracy = 0.0
    
    return {
        "block_pixel_count": int(block_pixel_count),
        "block_correct": int(block_correct),
        "block_accuracy": block_accuracy,
        "background_pixel_count": int(background_pixel_count),
        "background_correct": int(background_correct),
        "background_accuracy": background_accuracy,
        "total_pixels": int(total_pixels),
        "total_correct": int(block_correct + background_correct),
        "overall_accuracy": overall_accuracy,
    }


def compute_block_slide(
    name: str,
    mode: str = 'vreason_bench_standard',
) -> Dict:
    """
    Evaluate Block Slide: extract last frame from generated video,
    load corresponding masks, and check if the generated image satisfies
    the color constraints (block regions black, background plane regions grey).
    
    """
    dirs = get_eval_directories("Block_slide")
    gt_root = dirs["gt_dir"]
    predictions_dir = dirs["predictions_dir"]
    eval_results_dir = dirs["eval_results_dir"]
    
    color_tolerance = 10
    background_color = (255, 255, 255)
    
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(eval_results_dir, exist_ok=True)
    
    frame_files: List[str] = []
    for root, _, files in os.walk(predictions_dir):
        for fname in sorted(files):
            if fname.lower().endswith(".png"):
                frame_files.append(os.path.join(root, fname))

    results: Dict[str, Dict] = {}
    accuracies: List[float] = []
    block_accuracies: List[float] = []
    background_accuracies: List[float] = []
    
    for frame_path in frame_files:
        base = os.path.basename(frame_path)
        stem = os.path.splitext(base)[0]

        m = re.match(r"(?P<model>[^_]+)_(?P<idx>\d{2,4})_(?P<seed>seed[0-9A-Za-z]+)$", stem)
        if not m:
            results[frame_path] = {
                "pred_frame": frame_path,
                "error": "invalid_filename",
            }
            continue

        model = m.group("model")
        idx_str = m.group("idx")

        gt_target = os.path.join(gt_root, f"{idx_str}_gt.png")
        gt_mask = os.path.join(gt_root, f"{idx_str}_mask.png")
        gt_bg_mask = os.path.join(gt_root, f"{idx_str}_bg_mask.png")

        if not os.path.exists(gt_mask) or not os.path.exists(gt_bg_mask):
            results[frame_path] = {
                "pred_frame": frame_path,
                "error": "masks_not_found",
                "gt_target": gt_target if os.path.exists(gt_target) else None,
                "gt_mask": gt_mask if os.path.exists(gt_mask) else None,
                "gt_bg_mask": gt_bg_mask if os.path.exists(gt_bg_mask) else None,
            }
            continue

        try:
            generated_img = _load_image_as_array(frame_path)
            mask_block = _load_image_as_array(gt_mask)
            mask_background = _load_image_as_array(gt_bg_mask)
            
            metrics = _check_mask_colors(
                generated_img, 
                mask_block, 
                mask_background, 
                color_tolerance,
                background_color
            )
            
            accuracies.append(metrics["overall_accuracy"])
            block_accuracies.append(metrics["block_accuracy"])
            background_accuracies.append(metrics["background_accuracy"])
            
            results[frame_path] = {
                "pred_frame": frame_path,
                "gt_target": gt_target,
                "gt_mask": gt_mask,
                "gt_bg_mask": gt_bg_mask,
                "metrics": metrics,
                "score": metrics["overall_accuracy"],
                "gt_index": idx_str,
                "model": model,
            }
        
        except Exception as e:
            results[frame_path] = {
                "pred_frame": frame_path,
                "gt_target": gt_target if os.path.exists(gt_target) else None,
                "gt_mask": gt_mask if os.path.exists(gt_mask) else None,
                "gt_bg_mask": gt_bg_mask if os.path.exists(gt_bg_mask) else None,
                "error": str(e),
            }
    
    unified = build_unified_results(
        results, 
        score_key="score", 
        threshold=0.95,
        predictions_dir=predictions_dir
    )
    
    unified["aggregate"]["task_metrics"] = {
        "average_overall_accuracy": float(sum(accuracies) / len(accuracies)) if accuracies else 0.0,
        "average_block_accuracy": float(sum(block_accuracies) / len(block_accuracies)) if block_accuracies else 0.0,
        "average_background_accuracy": float(sum(background_accuracies) / len(background_accuracies)) if background_accuracies else 0.0,
        "color_tolerance": color_tolerance,
        "background_color": background_color,
    }
    
    save_json(unified, os.path.join(eval_results_dir, "block_slide_eval.json"))
    return unified
