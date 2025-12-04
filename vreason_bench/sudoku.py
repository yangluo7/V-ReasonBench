import os
import re
import csv
import json
import pandas as pd
from typing import Dict, List, Optional

import numpy as np

from .utils import (
    build_unified_results,
    extract_sudoku_from_image,
    get_eval_directories,
    parse_sudoku_result,
    save_json,
    save_sudoku_extraction_result,
)


def _load_csv_matrix(path: str) -> np.ndarray:
    """
    Load a Sudoku matrix from CSV.

    The CSV can be:
      - A flat row of N^2 numbers, which will be reshaped into an N x N grid
      - An already square matrix.

    Any parsing/shape error is raised to the caller, which can decide how to
    handle it (e.g., skip this sample but continue the overall evaluation).
    """
    df = pd.read_csv(path, header=None)
    data = df.values
    if data.shape[0] == 1:
        flat = data.flatten()
        n = int(len(flat) ** 0.5)
        if n * n != len(flat):
            raise ValueError("CSV length is not a perfect square")
        arr = flat.reshape(n, n)
    else:
        arr = data
        r, c = arr.shape
        if r != c:
            raise ValueError("Sudoku CSV must be square")
    return arr.astype(int)


def _is_valid_sequence(seq: np.ndarray, expected_size: int) -> bool:
    non_zero = seq[seq != 0]
    if len(non_zero) != len(np.unique(non_zero)):
        return False
    if len(non_zero) > 0:
        if np.min(non_zero) < 1 or np.max(non_zero) > expected_size:
            return False
    return True


def _check_rows_cols_validity(sudoku: np.ndarray) -> bool:
    size = sudoku.shape[0]
    for row in sudoku:
        if not _is_valid_sequence(row, size):
            return False
    for col in sudoku.T:
        if not _is_valid_sequence(col, size):
            return False
    return True


def _check_sudoku_validity(sudoku: np.ndarray) -> bool:
    size = sudoku.shape[0]
    sqrt_size = int(np.sqrt(size))
    if sqrt_size * sqrt_size != size:
        return _check_rows_cols_validity(sudoku)
    for row in sudoku:
        if not _is_valid_sequence(row, size):
            return False
    for col in sudoku.T:
        if not _is_valid_sequence(col, size):
            return False
    for i in range(0, size, sqrt_size):
        for j in range(0, size, sqrt_size):
            sub = sudoku[i:i+sqrt_size, j:j+sqrt_size].flatten()
            if not _is_valid_sequence(sub, size):
                return False
    return True


def _evaluate_sudoku(puzzle_csv: str, solution_csv: str, prediction_csv: str) -> Dict[str, float]:
    """
    Evaluate one Sudoku instance.

    Any CSV loading/parsing error is captured and returned as an "error"
    dictionary so that the caller can continue evaluating other instances.
    """
    try:
        puzzle = _load_csv_matrix(puzzle_csv)
        solution = _load_csv_matrix(solution_csv)
        prediction = _load_csv_matrix(prediction_csv)
    except Exception as e:
        return {
            "error": "failed_to_load_csv",
            "detail": str(e),
            "puzzle_csv": puzzle_csv,
            "solution_csv": solution_csv,
            "prediction_csv": prediction_csv,
        }
    if not (puzzle.shape == solution.shape == prediction.shape):
        return {"error": "matrix_shape_mismatch"}

    size = puzzle.shape[0]
    known = puzzle != 0
    empty = puzzle == 0
    preserved_correctly = int(np.sum(prediction[known] == puzzle[known]))
    total_known = int(np.sum(known))
    preservation_rate = (preserved_correctly / total_known) if total_known > 0 else 1.0

    filled_correctly = int(np.sum(prediction[empty] == solution[empty]))
    total_empty = int(np.sum(empty))
    fill_accuracy = (filled_correctly / total_empty) if total_empty > 0 else 1.0

    overall_accuracy = float(np.sum(prediction == solution) / (size * size))
    valid = bool(_check_sudoku_validity(prediction))

    return {
        "preservation_rate": float(preservation_rate),
        "fill_accuracy": float(fill_accuracy),
        "overall_accuracy": float(overall_accuracy),
        "is_valid_sudoku": valid,
        "total_cells": int(size * size),
        "known_cells": total_known,
        "empty_cells": total_empty,
        "preserved_correctly": preserved_correctly,
        "filled_correctly": filled_correctly,
        "sudoku_size": f"{size}x{size}",
    }


def _write_grid_csv(grid: List[List[int]], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in grid:
            writer.writerow(row)


def compute_sudoku(
    name: str,
    local: bool = False,
    mode: str = 'vreason_bench_standard',
) -> Dict:
    """
    Evaluate Sudoku by reading prediction frames from the `predictions` folder,
    using a VLM to read the Sudoku digits from the image into a prediction CSV,
    and evaluating against the ground-truth CSV.
    
    Expected layout:
      evaluations/Sudoku/
        inputs/<difficulty_base>/<idx>.{png,csv}
        GT/<difficulty_base>/<idx>.csv
        predictions/<difficulty_base>/<model>_<idx>_seedK.png
        eval_results/sudoku_eval.json
    """
    dirs = get_eval_directories("Sudoku")
    gt_root = dirs["gt_dir"]
    input_root = dirs["input_dir"]
    predictions_dir = dirs["predictions_dir"]
    eval_results_dir = dirs["eval_results_dir"]
    
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(eval_results_dir, exist_ok=True)

    # Gather prediction frames from predictions_dir
    frame_files: List[str] = []
    for root, _, files in os.walk(predictions_dir):
        for fname in sorted(files):
            if fname.lower().endswith(".png"):
                frame_files.append(os.path.join(root, fname))

    results: Dict[str, Dict] = {}
    overall_list: List[float] = []

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
        idx = m.group("idx")

        puzzle_csv = os.path.join(input_root, subdir, f"{idx}.csv")
        gt_csv = os.path.join(gt_root, subdir, f"{idx}.csv")

        if not (os.path.exists(puzzle_csv) and os.path.exists(gt_csv)):
            results[frame_path] = {
                "pred_frame": frame_path,
                "error": "missing_input_or_gt",
                "puzzle_csv": puzzle_csv,
                "gt_csv": gt_csv,
            }
            continue

        base_out = os.path.splitext(frame_path)[0]
        pred_json = base_out + ".json"
        pred_csv = base_out + ".csv"

        grid: Optional[List[List[int]]] = None

        if os.path.exists(pred_json):
            try:
                with open(pred_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data.get("sudoku_grid"), list):
                    grid = data["sudoku_grid"]
                elif isinstance(data.get("parsed_sudoku_grid"), list):
                    grid = data["parsed_sudoku_grid"]
            except Exception:
                grid = None

        if grid is None and os.path.exists(pred_csv):
            try:
                rows: List[List[int]] = []
                with open(pred_csv, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if row:
                            rows.append([int(x) for x in row])
                if rows:
                    grid = rows
            except Exception:
                grid = None

        # Extract base from subdir (e.g., "easy_2" -> 2)
        try:
            base_extraction = int(subdir.split('_')[1]) if '_' in subdir else 3
        except (IndexError, ValueError):
            base_extraction = 3

        if grid is None:
            result = extract_sudoku_from_image(frame_path, base=base_extraction)
            grid = parse_sudoku_result(result.get("extracted_numbers"))
            save_sudoku_extraction_result(result if isinstance(result, dict) else {}, grid, output_file=pred_json)

        if grid:
            _write_grid_csv(grid, pred_csv)

        stats = _evaluate_sudoku(puzzle_csv, gt_csv, pred_csv)

        if "error" not in stats and "overall_accuracy" in stats:
            overall_list.append(float(stats["overall_accuracy"]))
            score = float(stats.get("overall_accuracy", 0.0))
        else:
            score = 0.0

        stats["pred_frame"] = frame_path
        stats["model"] = model
        stats["score"] = score
        stats["gt_index"] = f"{subdir.replace(os.sep, '_')}_{idx}" if subdir else idx
        results[frame_path] = stats

    unified = build_unified_results(
        results, 
        score_key="score", 
        threshold=1.0,
        predictions_dir=predictions_dir
    )
    
    unified['aggregate']['task_metrics'] = {
        'average_overall_accuracy': float(sum(overall_list) / len(overall_list)) if overall_list else 0.0,
        'base': int(base_extraction),
    }

    save_json(unified, os.path.join(eval_results_dir, 'sudoku_eval.json'))
    return unified
