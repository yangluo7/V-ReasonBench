import os
import re
from typing import Dict, List, Tuple

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




def _cell_avg_rgb(image: np.ndarray, row: int, col: int, gh: int, gw: int) -> np.ndarray:
    h, w = image.shape[:2]
    ch = h // gh
    cw = w // gw
    y1 = row * ch
    y2 = (row + 1) * ch
    x1 = col * cw + 3
    x2 = (col + 1) * cw + 3
    mh = int(ch * 0.1)
    mw = int(cw * 0.1)
    region = image[y1+mh:y2-mh, x1+mw:x2-mw]
    return np.mean(region, axis=(0, 1))


def _grid_dims(symmetry_type: str) -> Tuple[int, int]:
    if symmetry_type == 'horizontal':
        return 10, 16
    if symmetry_type in ['vertical', 'rotational']:
        return 10, 16
    if symmetry_type == 'diagonal':
        return 8, 8
    raise ValueError(f"Unknown symmetry type: {symmetry_type}")


def _region_names(symmetry_type: str) -> Tuple[str, str]:
    if symmetry_type == 'vertical':
        return 'left', 'right'
    if symmetry_type == 'horizontal':
        return 'top', 'bottom'
    if symmetry_type == 'diagonal':
        return 'lower', 'upper'
    if symmetry_type == 'rotational':
        return 'left', 'right'
    raise ValueError(f"Unknown symmetry type: {symmetry_type}")


def _region_idx(row: int, col: int, symmetry_type: str, gh: int, gw: int) -> int:
    if symmetry_type == 'vertical':
        return 0 if col < gw // 2 else 1
    if symmetry_type == 'horizontal':
        return 0 if row < gh // 2 else 1
    if symmetry_type == 'diagonal':
        return 0 if row >= col else 1
    if symmetry_type == 'rotational':
        return 0 if col < gw // 2 else 1
    raise ValueError(f"Unknown symmetry type: {symmetry_type}")


def _is_white(rgb: np.ndarray, thr: int = 240) -> bool:
    return bool(np.all(rgb > thr))


def _cell_match(pred_rgb: np.ndarray, gt_rgb: np.ndarray, delta_e_thr: float) -> Tuple[bool, float]:
    pred_lab = rgb_to_lab(pred_rgb)
    gt_lab = rgb_to_lab(gt_rgb)
    de = compute_delta_e(pred_lab, gt_lab)
    return (de < delta_e_thr), de

def compute_visual_symmetry(
    name: str,
    local: bool = False,
    mode: str = 'vreason_bench_standard',
    **kwargs,
) -> Dict:
    """
    Evaluate Visual Symmetry by comparing prediction frames in the `predictions` folder
    against GT per-grid cell with ΔE threshold.
    
    Assumes frames have already been extracted (e.g., via VReasonBench.distribute_videos)
    into `evaluations/Visual_symmetry/predictions/*/*.png`.
    """
    dirs = get_eval_directories("Visual_symmetry")
    delta_e_threshold = 15.0
    gt_root = dirs["gt_dir"]
    input_root = dirs["input_dir"]
    predictions_dir = dirs["predictions_dir"]
    eval_results_dir = dirs["eval_results_dir"]
    
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(eval_results_dir, exist_ok=True)

    frame_files: List[str] = []
    for root, _, files in os.walk(predictions_dir):
        for fname in sorted(files):
            if fname.lower().endswith(".png"):
                frame_files.append(os.path.join(root, fname))

    results: Dict[str, Dict] = {}
    accuracies: List[float] = []

    for frame_path in frame_files:
        stem = os.path.splitext(os.path.basename(frame_path))[0]

        m = re.match(r"(?P<model>[^_]+)_(?P<idx>\d{2,4})_(?P<seed>seed[0-9A-Za-z]+)$", stem)
        model: str = ""
        
        model = m.group("model")
        idx_str = m.group("idx")

        rel = os.path.relpath(frame_path, predictions_dir)
        subdir = os.path.dirname(rel)
        sym_candidates = {"vertical", "horizontal", "rotational", "diagonal"}
        parts = [p for p in subdir.replace("\\", "/").split("/") if p]
        sym_folder = parts[0] if parts else ""
        base_sym = sym_folder.split("_")[0] if sym_folder else ""
        if base_sym not in sym_candidates:
            raise ValueError(
                f"Invalid symmetry subfolder '{sym_folder}'. Expected one of {sorted(sym_candidates)}"
            )
        symmetry_type = base_sym
        gh, gw = _grid_dims(symmetry_type)
        region1, region2 = _region_names(symmetry_type)
        puzzle_path = os.path.join(input_root, subdir, f"{idx_str}.png")
        gt_path = os.path.join(gt_root, subdir, f"{idx_str}.png")

        pred = load_image_rgb(frame_path)
        puzzle = load_image_rgb(puzzle_path)
        gt = load_image_rgb(gt_path)

        pred = crop_black_white_border(pred)
        puzzle = crop_black_white_border(puzzle)
        gt = crop_black_white_border(gt)
        th, tw = gt.shape[:2]
        if pred.shape[:2] != (th, tw):
            pred = cv2.resize(pred, (tw, th), interpolation=cv2.INTER_LINEAR)
        if puzzle.shape[:2] != (th, tw):
            puzzle = cv2.resize(puzzle, (tw, th), interpolation=cv2.INTER_LINEAR)

        counters = {
            region1: {'colored': [0, 0], 'white': [0, 0]},
            region2: {'colored': [0, 0], 'white': [0, 0]},
        }

        all_delta_e: List[float] = []

        for r in range(gh):
            for c in range(gw):
                pred_rgb = _cell_avg_rgb(pred, r, c, gh, gw)
                gt_rgb = _cell_avg_rgb(gt, r, c, gh, gw)

                reg_idx = _region_idx(r, c, symmetry_type, gh, gw)
                reg_name = region1 if reg_idx == 0 else region2
                color_type = 'white' if _is_white(gt_rgb) else 'colored'

                match, de = _cell_match(pred_rgb, gt_rgb, delta_e_threshold)
                all_delta_e.append(de)
                counters[reg_name][color_type][1] += 1
                if match:
                    counters[reg_name][color_type][0] += 1

        res = {
            'symmetry_type': symmetry_type,
            'total_cells': gh * gw,
        }
        for reg in [region1, region2]:
            for ct in ['colored', 'white']:
                correct, total = counters[reg][ct]
                acc = (correct / total) if total > 0 else 0.0
                res[f'{reg}_{ct}_accuracy'] = acc
                res[f'{reg}_{ct}_correct'] = correct
                res[f'{reg}_{ct}_total'] = total
            reg_correct = counters[reg]['colored'][0] + counters[reg]['white'][0]
            reg_total = counters[reg]['colored'][1] + counters[reg]['white'][1]
            res[f'{reg}_total_accuracy'] = (reg_correct / reg_total) if reg_total > 0 else 0.0
            res[f'{reg}_total_correct'] = reg_correct
            res[f'{reg}_total_cells'] = reg_total

        total_correct = sum(counters[reg][ct][0] for reg in counters for ct in counters[reg])
        res['overall_accuracy'] = total_correct / res['total_cells']
        res['mean_delta_e'] = float(np.mean(all_delta_e)) if all_delta_e else 0.0
        res['max_delta_e'] = float(np.max(all_delta_e)) if all_delta_e else 0.0
        res['median_delta_e'] = float(np.median(all_delta_e)) if all_delta_e else 0.0

        res['score'] = res['overall_accuracy']
        res['model'] = model
        res['gt_index'] = f"{subdir.replace(os.sep, '_')}_{idx_str}" if subdir else idx_str
        results[frame_path] = res
        accuracies.append(res['overall_accuracy'])

    unified = build_unified_results(
        results, 
        score_key="score", 
        threshold=1.0,
        predictions_dir=predictions_dir
    )
    
    unified['aggregate']['task_metrics'] = {
        'average_overall_accuracy': float(sum(accuracies) / len(accuracies)) if accuracies else 0.0,
    }

    save_json(unified, os.path.join(eval_results_dir, 'visual_symmetry_eval.json'))
    return unified

