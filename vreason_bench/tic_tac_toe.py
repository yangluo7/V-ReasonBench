import os
import re
from typing import Dict, List

import cv2
import numpy as np

from .utils import (
    build_unified_results,
    compute_delta_e,
    get_eval_directories,
    load_image_rgb,
    rgb_to_lab,
    save_json,
)


def _crop_border(image: np.ndarray, black_threshold: int = 30, white_threshold: int = 180) -> np.ndarray:
    """Crop borders by detecting non-black and non-white content."""
    non_black = np.any(image > black_threshold, axis=2)
    luma = image.mean(axis=2)
    non_white = luma < white_threshold
    valid = non_black & non_white
    rows = np.any(valid, axis=1)
    cols = np.any(valid, axis=0)
    if not np.any(rows) or not np.any(cols):
        return image
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return image[r0:r1+1, c0:c1+1]


def _compute_grid_accuracy(
    gt_img: np.ndarray,
    gen_img: np.ndarray,
    grid_size: int = 3,
    delta_e_threshold: float = 15.0,
    black_threshold: int = 30,
    white_threshold: int = 180,
) -> Dict:
    """
    Compute cell-based accuracy using grid splitting similar to rule_follow.
    Divides the image into a grid_size × grid_size grid, computes average RGB
    per cell, converts to LAB, and compares using Delta E.
    
    Args:
        gt_img: Ground truth image (RGB)
        gen_img: Generated image (RGB)
        grid_size: Number of cells per row/column (default 3 for tic-tac-toe)
        delta_e_threshold: Delta E threshold for color matching (default 15.0)
        black_threshold: Intensity threshold for non-black detection (default 30)
        white_threshold: Intensity threshold for non-white detection (default 180)
    
    Returns:
        Dictionary with accuracy metrics including overall_accuracy
    """
    gt_img = _crop_border(gt_img, black_threshold, white_threshold)
    gen_img = _crop_border(gen_img, black_threshold, white_threshold)
    
    h, w = gt_img.shape[:2]
    if gen_img.shape[:2] != (h, w):
        gen_img = cv2.resize(gen_img, (w, h), interpolation=cv2.INTER_LINEAR)

    cell_h = h // grid_size
    cell_w = w // grid_size
    
    total_cells = grid_size * grid_size
    matched_cells = 0
    delta_es: List[float] = []

    for r in range(grid_size):
        for c in range(grid_size):
            y1 = r * cell_h
            y2 = (r + 1) * cell_h if r < grid_size - 1 else h
            x1 = c * cell_w
            x2 = (c + 1) * cell_w if c < grid_size - 1 else w
            
            mh = int((y2 - y1) * 0.1)
            mw = int((x2 - x1) * 0.1)
            yy1 = min(max(y1 + mh, y1), y2)
            yy2 = max(min(y2 - mh, y2), y1 + 1)
            xx1 = min(max(x1 + mw, x1), x2)
            xx2 = max(min(x2 - mw, x2), x1 + 1)
            
            gt_rgb = np.mean(gt_img[yy1:yy2, xx1:xx2], axis=(0, 1))
            gen_rgb = np.mean(gen_img[yy1:yy2, xx1:xx2], axis=(0, 1))
            
            gt_lab = rgb_to_lab(gt_rgb)
            gen_lab = rgb_to_lab(gen_rgb)
            de = compute_delta_e(gen_lab, gt_lab)
            delta_es.append(de)
            
            if de < delta_e_threshold:
                matched_cells += 1

    overall_accuracy = float(matched_cells / total_cells) if total_cells > 0 else 0.0

    max_delta_e = float(np.max(delta_es)) if delta_es else 0.0
    if overall_accuracy == 8 / 9 and max_delta_e < 8.0:
        overall_accuracy = 1.0
    
    return {
        'total_cells': int(total_cells),
        'matched_cells': int(matched_cells),
        'overall_accuracy': float(overall_accuracy),
        'grid_size': int(grid_size),
        'delta_e_threshold': float(delta_e_threshold),
        'black_threshold': int(black_threshold),
        'white_threshold': int(white_threshold),
        'mean_delta_e': float(np.mean(delta_es)) if delta_es else 0.0,
        'median_delta_e': float(np.median(delta_es)) if delta_es else 0.0,
        'max_delta_e': float(np.max(delta_es)) if delta_es else 0.0,
    }


def compute_tic_tac_toe(
    name: str,
    local: bool = False,
    mode: str = 'vreason_bench_standard',
    **kwargs,
) -> Dict:
    """
    Evaluate Tic-Tac-Toe by reading prediction frames from the `predictions` folder
    and using grid-based cell comparison similar to rule_follow.
    Divides the image into a grid_size × grid_size grid (default 3×3 for tic-tac-toe),
    computes average RGB per cell, converts to LAB, and compares using Delta E.
    
    Expected layout:
      evaluations/Tic_tac_toe/
        GT/<num_moves>/<idx>.png
        predictions/<num_moves>/<model>_<idx>_seedK.png
        eval_results/tic_tac_toe_eval.json
    """
    dirs = get_eval_directories("Tic_tac_toe")
    gt_dir = dirs["gt_dir"]
    predictions_dir = dirs["predictions_dir"]
    eval_results_dir = dirs["eval_results_dir"]
    
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(eval_results_dir, exist_ok=True)
    
    grid_size: int = 3
    delta_e_threshold: float = 5.0
    black_threshold: int = 30
    white_threshold: int = 180

    frame_files: List[str] = []
    for root, _, files in os.walk(predictions_dir):
        for fname in sorted(files):
            if fname.lower().endswith(".png"):
                frame_files.append(os.path.join(root, fname))

    results: Dict[str, Dict] = {}
    final_scores: List[float] = []
    grouped_by_gt: Dict[str, List[Dict]] = {}

    for frame_path in frame_files:
        rel = os.path.relpath(frame_path, predictions_dir)
        subdir = os.path.dirname(rel)
        
        base_f = os.path.basename(frame_path)
        stem = os.path.splitext(base_f)[0]

        # Expect naming: <model>_<idx>_seedK
        m = re.match(
            r"^(?P<model>[^_]+)_(?P<idx>\d{2})_seed(?P<seed>[0-9A-Za-z]+)$",
            stem,
        )
        if not m:
            results[frame_path] = {
                "pred_frame": frame_path,
                "error": "invalid_filename",
            }
            continue

        model = m.group("model")
        idx_str = m.group("idx")

        gt_path = os.path.join(gt_dir, subdir, f"{idx_str}.png")

        metrics = None
        if os.path.exists(gt_path):
            gt_img = load_image_rgb(gt_path)
            gen_img = load_image_rgb(frame_path)
            metrics = _compute_grid_accuracy(
                gt_img, gen_img,
                grid_size=grid_size,
                delta_e_threshold=delta_e_threshold,
                black_threshold=black_threshold,
                white_threshold=white_threshold
            )
        else:
            metrics = {"error": "gt_not_found", "gt_path": gt_path}

        if isinstance(metrics, dict) and ("overall_accuracy" in metrics):
            final_score = float(metrics["overall_accuracy"])
            final_scores.append(final_score)
        else:
            final_score = 0.0

        res_entry = {
            'pred_frame': frame_path,
            'model': model,
            'ground_truth': gt_path,
            'metrics': metrics,
            'score': float(final_score),
            'gt_index': f"{subdir.replace(os.sep, '_')}_{idx_str}" if subdir else idx_str,
        }
        results[frame_path] = res_entry
        grouped_by_gt.setdefault(idx_str, []).append({
            'pred_frame': frame_path,
            'final_score': float(final_score),
        })

    unified = build_unified_results(
        results,
        score_key="score",
        threshold=0.99,
        predictions_dir=predictions_dir,
    )
    unified['aggregate']['task_metrics'] = {
        'average_final_score': float(sum(final_scores) / len(final_scores)) if final_scores else 0.0,
        'grid_size': int(grid_size),
        'delta_e_threshold': float(delta_e_threshold),
        'black_threshold': int(black_threshold),
        'white_threshold': int(white_threshold),
    }

    save_json(unified, os.path.join(eval_results_dir, 'tic_tac_toe_eval.json'))
    return unified
