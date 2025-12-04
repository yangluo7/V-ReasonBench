import os
import re
from typing import Dict, List, Optional

import cv2
import numpy as np

from .utils import (
    build_unified_results,
    compute_delta_e,
    crop_black_white_border,
    get_eval_directories,
    load_image_rgb,
    rgb_to_lab,
    save_json,
)


def compute_rule_follow(
    name: str,
    local: bool = False,
    mode: str = 'vreason_bench_standard',
) -> Dict:
    """
    Evaluate rule-following (grid completion) using the same grid-based ΔE per-cell
    comparison method as Visual Symmetry. We split the image into a gh×gw grid,
    average RGB in each cell, convert to LAB, compute ΔE, and count cells below
    a threshold as correct. The final score is overall per-cell accuracy.

    Note: Visual Symmetry assets are predominantly on white backgrounds, while
    Rule Follow uses black backgrounds. We therefore crop borders using a
    "non-black" mask here (instead of the "non-black AND non-white" used in
    Visual Symmetry) so that black-only borders are excluded appropriately.

    Dataset structure note: In Rule Follow, each image contains six sub-grids,
    each of size gh×gw cells, arranged as shown in the GT example image (e.g.,
    01.png). The splitting into cells follows this layout when scoring.
    
    Grid size is determined by subdirectory:
      - Subdirectories '2' or '3': 7×7 grid per subgrid
      - Subdirectory '4': 9×9 grid per subgrid
    """
    dirs = get_eval_directories("Rule_follow")
    gt_dir = dirs["gt_dir"]
    q_dir = dirs["input_dir"]
    predictions_dir = dirs["predictions_dir"]
    eval_results_dir = dirs["eval_results_dir"]
    
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(eval_results_dir, exist_ok=True)
    
    subgrid_rows = 2
    subgrid_cols = 3
    delta_e_threshold = 15
    black_threshold = 30
    mask_threshold = 10

    # Gather prediction frames from predictions_dir using new naming
    frame_files: List[str] = []
    for root, _, files in os.walk(predictions_dir):
        for fname in sorted(files):
            if fname.lower().endswith(".png"):
                frame_files.append(os.path.join(root, fname))

    results: Dict[str, Dict] = {}
    final_scores: List[float] = []
    last_grid_scores: List[float] = []

    for frame_path in frame_files:
        rel = os.path.relpath(frame_path, predictions_dir)
        subdir = os.path.dirname(rel)

        # Expect predictions named as: <model>_<idx>_<seed>.png
        stem = os.path.splitext(os.path.basename(frame_path))[0]
        m_name = re.match(r"(?P<model>[^_]+)_(?P<idx>\d{2,4})_(?P<seed>seed[0-9A-Za-z]+)$", stem)
        if not m_name:
            results[frame_path] = {
                "pred_frame": frame_path,
                "error": "invalid_filename",
            }
            continue

        model = m_name.group("model")
        idx_str = m_name.group("idx")
        
        subdir_num = int(subdir) if subdir and subdir.isdigit() else 0
        if subdir_num == 4:
            grid_size = 9
        elif subdir_num in [2, 3]:
            grid_size = 7
        
        gh = gw = grid_size
        
        gt_path = os.path.join(gt_dir, subdir, f"{idx_str}.png")
        q_path = os.path.join(q_dir, subdir, f"{idx_str}.png")

        if not (os.path.exists(gt_path) and os.path.exists(q_path)):
            results[frame_path] = {
                "pred_frame": frame_path,
                "error": "missing_gt_or_question",
                "gt_path": gt_path if os.path.exists(gt_path) else None,
                "question_path": q_path if os.path.exists(q_path) else None,
            }
            continue

        pred = load_image_rgb(frame_path)
        gt = load_image_rgb(gt_path)
        pred = crop_black_white_border(pred, black_threshold=black_threshold, white_threshold=256)
        gt = crop_black_white_border(gt, black_threshold=black_threshold, white_threshold=256)
        th, tw = gt.shape[:2]
        if pred.shape[:2] != (th, tw):
            pred = cv2.resize(pred, (tw, th), interpolation=cv2.INTER_LINEAR)

        th, tw = gt.shape[:2]
        panel_h = th // subgrid_rows
        panel_w = tw // subgrid_cols
        total_cells = subgrid_rows * subgrid_cols * gh * gw
        matched = 0
        valid_cells = 0
        matched_valid = 0
        last_grid_total = gh * gw
        last_grid_matched = 0
        delta_es: List[float] = []
        last_grid_delta_es: List[float] = []
        m = None

        for sr in range(subgrid_rows):
            for sc in range(subgrid_cols):
                is_last_grid = (sr == subgrid_rows - 1 and sc == subgrid_cols - 1)
                y0 = sr * panel_h
                x0 = sc * panel_w
                y_end = th if sr == subgrid_rows - 1 else (y0 + panel_h)
                x_end = tw if sc == subgrid_cols - 1 else (x0 + panel_w)
                gt_panel = gt[y0:y_end, x0:x_end]
                pred_panel = pred[y0:y_end, x0:x_end]
                mask_panel = m[y0:y_end, x0:x_end] if m is not None else None
                nb = np.any(gt_panel > black_threshold, axis=2)
                rows = np.any(nb, axis=1)
                cols = np.any(nb, axis=0)
                if np.any(rows) and np.any(cols):
                    pr0, pr1 = np.where(rows)[0][[0, -1]]
                    pc0, pc1 = np.where(cols)[0][[0, -1]]
                    gt_panel = gt_panel[pr0:pr1+1, pc0:pc1+1]
                    pred_panel = pred_panel[pr0:pr1+1, pc0:pc1+1]
                    if mask_panel is not None:
                        mask_panel = mask_panel[pr0:pr1+1, pc0:pc1+1]
                if pred_panel.shape[:2] != gt_panel.shape[:2]:
                    pred_panel = cv2.resize(pred_panel, (gt_panel.shape[1], gt_panel.shape[0]), interpolation=cv2.INTER_LINEAR)
                ph = max(1, gt_panel.shape[0])
                pw = max(1, gt_panel.shape[1])
                cell_h = max(1, ph // gh)
                cell_w = max(1, pw // gw)

                for r in range(gh):
                    for c in range(gw):
                        y1 = r * cell_h
                        y2 = (r + 1) * cell_h if r < gh - 1 else ph
                        x1 = c * cell_w
                        x2 = (c + 1) * cell_w if c < gw - 1 else pw
                        mh = int((y2 - y1) * 0.1)
                        mw = int((x2 - x1) * 0.1)
                        yy1 = min(max(y1 + mh, y1), y2)
                        yy2 = max(min(y2 - mh, y2), y1 + 1)
                        xx1 = min(max(x1 + mw, x1), x2)
                        xx2 = max(min(x2 - mw, x2), x1 + 1)

                        pred_rgb = np.mean(pred_panel[yy1:yy2, xx1:xx2], axis=(0, 1))
                        gt_rgb = np.mean(gt_panel[yy1:yy2, xx1:xx2], axis=(0, 1))
                        de = compute_delta_e(rgb_to_lab(pred_rgb), rgb_to_lab(gt_rgb))
                        delta_es.append(de)
                        if de < delta_e_threshold:
                            matched += 1
                        
                        if is_last_grid:
                            last_grid_delta_es.append(de)
                            if de < delta_e_threshold:
                                last_grid_matched += 1

                        if mask_panel is not None:
                            cell_mean = float(np.mean(mask_panel[y1:y2, x1:x2]))
                            if cell_mean > mask_threshold:
                                valid_cells += 1
                                if de < delta_e_threshold:
                                    matched_valid += 1

        overall_accuracy = float(matched / total_cells) if total_cells > 0 else 0.0
        
        # Calculate last grid (bottom-right) accuracy
        last_grid_accuracy = float(last_grid_matched / last_grid_total) if last_grid_total > 0 else 0.0
        final_score = last_grid_accuracy
        final_scores.append(final_score)
        last_grid_scores.append(last_grid_accuracy)
      
        results[frame_path] = {
            'ground_truth': gt_path,
            'question': q_path,
            'pred_frame': frame_path,
            'model': model,
            'total_cells': int(total_cells),
            'matched_cells': int(matched),
            'valid_cells': int(valid_cells),
            'overall_accuracy': float(overall_accuracy),
            'delta_e_threshold': float(delta_e_threshold),
            'black_threshold': int(black_threshold),
            'mask_threshold': int(mask_threshold),
            'last_grid_accuracy': float(last_grid_accuracy),
            'last_grid_total_cells': int(last_grid_total),
            'score': float(final_score),
            'gt_index': f"{subdir.replace(os.sep, '_')}_{idx_str}" if subdir else idx_str,
        }

    unified = build_unified_results(
        results,
        score_key='score',
        threshold=1.0,
        predictions_dir=predictions_dir,
    )
    unified['aggregate']['task_metrics'] = {
        'average_final_score': float(sum(final_scores) / len(final_scores)) if final_scores else 0.0,
        'average_last_grid_accuracy': float(sum(last_grid_scores) / len(last_grid_scores)) if last_grid_scores else 0.0,
        'delta_e_threshold': float(delta_e_threshold),
    }
    save_json(unified, os.path.join(eval_results_dir, 'rule_follow_eval.json'))
    return unified
