import os
import re
from typing import Dict, List

import numpy as np
from PIL import Image

from .utils import (
    build_unified_results,
    get_eval_directories,
    save_json,
)


def compute_communicating_vessels(
    name: str,
    mode: str = 'vreason_bench_standard',
) -> Dict:
    """
    Evaluate Communicating Vessels by comparing the generated last frame to GT using
    normalized RMSE over non-white regions.

    """
    dirs = get_eval_directories("Communicating_vessels")
    gt_root = dirs["gt_dir"]
    predictions_dir = dirs["predictions_dir"]
    eval_results_dir = dirs["eval_results_dir"]
    white_threshold = 240
    
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(eval_results_dir, exist_ok=True)

    # Gather prediction frames from predictions_dir (similar to visual_analogy / visual_symmetry)
    frame_files: List[str] = []
    for root, _, files in os.walk(predictions_dir):
        for fname in sorted(files):
            if fname.lower().endswith(".png"):
                frame_files.append(os.path.join(root, fname))

    results: Dict[str, Dict] = {}
    final_scores: List[float] = []
    normalized_rmses: List[float] = []

    for frame_path in frame_files:
        base = os.path.basename(frame_path)
        stem = os.path.splitext(base)[0]

        # Prefer new naming from distribute_videos: <model>_<idx>_<seed>.png
        m = re.match(r"(?P<model>[^_]+)_(?P<idx>\d{2,4})_(?P<seed>seed[0-9A-Za-z]+)$", stem)
        model = m.group("model")
        idx_str = m.group("idx")

        rel = os.path.relpath(frame_path, predictions_dir)
        subdir = os.path.dirname(rel)

        gt_image_path = os.path.join(gt_root, subdir, f"{idx_str}.png")
        mask_path = os.path.join(gt_root, subdir, f"{idx_str}_mask.png")

        def _load_rgb(path: str) -> np.ndarray:
            return np.array(Image.open(path).convert('RGB'))

        def _load_l(path: str) -> np.ndarray:
            return np.array(Image.open(path).convert('L')).astype(bool)

        metrics = {'error': 'gt_not_found'}
        score = 0.0
        norm_mse = 0.0
        norm_rmse = 0.0
        gen_ratio = 0.0
        if os.path.exists(gt_image_path):
            gt_img = _load_rgb(gt_image_path)
            gen_img = _load_rgb(frame_path)
            h, w = gt_img.shape[:2]
            if gen_img.shape[:2] != (h, w):
                gen_img = np.array(Image.fromarray(gen_img).resize((w, h), Image.BILINEAR))
            diff = gen_img.astype(np.float32) - gt_img.astype(np.float32)
            gt_non_white = np.any(gt_img < white_threshold, axis=2)
            gen_non_white = np.any(gen_img < white_threshold, axis=2)
            valid_mask = gt_non_white | gen_non_white

            valid_pixel_count = int(np.sum(valid_mask))
            gen_non_white_count = int(np.sum(gen_non_white))
            gen_ratio = float(gen_non_white_count / valid_pixel_count) if valid_pixel_count > 0 else 0.0

            if np.any(valid_mask):
                mse = float(np.mean((diff * diff)[valid_mask]))
                norm_mse = float(mse / (255.0 * 255.0))
                norm_rmse = float(np.sqrt(norm_mse))
            else:
                mse = 0.0
                norm_mse = 0.0
                norm_rmse = 0.0
            metrics = {
                'mse': mse,
                'normalized_rmse': norm_rmse,
                'gen_ratio': gen_ratio,
                'valid_pixel_count': valid_pixel_count,
                'white_threshold': int(white_threshold),
            }

            threshold1_gen_ratio = 0.95
            threshold2_norm_rmse = 0.08
            
            if gen_ratio >= threshold1_gen_ratio and norm_rmse <= threshold2_norm_rmse:
                score = 1.0
            else:
                score = 0.0
        final_scores.append(score)
        normalized_rmses.append(norm_rmse)

        subdir_key = subdir.replace(os.sep, '_') if subdir and subdir != '' else ''

        results[frame_path] = {
            'pred_frame': frame_path,
            'ground_truth': gt_image_path if os.path.exists(gt_image_path) else None,
            'mask_path': mask_path if os.path.exists(mask_path) else None,
            'metrics': metrics,
            'score': float(score),
            'gt_index': f"{subdir_key}_{idx_str}" if subdir_key else idx_str,
            'model': model,
        }

    unified = build_unified_results(
        results,
        score_key='score',
        threshold=0.90,
        predictions_dir=predictions_dir,
    )
    unified['aggregate']['task_metrics'] = {
        'average_mse': float(sum(final_scores) / len(final_scores)) if final_scores else 0.0,
        'average_normalized_rmse': float(sum(normalized_rmses) / len(normalized_rmses)) if normalized_rmses else 0.0,
        'mean_unified_score': float(np.mean([r['score'] for r in results.values() if 'error' not in r])) if results else 0.0,
    }
    save_json(unified, os.path.join(eval_results_dir, 'communicating_vessels_eval.json'))
    return unified
