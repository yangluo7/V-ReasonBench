import os
import json
import csv
import re
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .utils import (
    build_unified_results,
    extract_math_from_image,
    get_eval_directories,
    load_csv_column,
    parse_math_result,
    save_json,
    save_math_extraction_result,
)


logger = logging.getLogger(__name__)




def _parse_math_expression(expr: str) -> Tuple[str, str]:
    s = str(expr).strip()
    if "=" in s:
        left, right = s.split("=", 1)
        return left.strip(), right.strip()
    return s, ""


def _expressions_equivalent(expr1: str, expr2: str) -> bool:
    try:
        expr1_clean = str(expr1).strip().replace(' ', '')
        expr2_clean = str(expr2).strip().replace(' ', '')
        if not expr1_clean and not expr2_clean:
            return True
        if not expr1_clean or not expr2_clean:
            return False

        def normalize_math_symbols(expr):
            expr = expr.replace('×', '*').replace('x', '*').replace('X', '*')
            expr = expr.replace('÷', '/')
            return expr

        expr1_normalized = normalize_math_symbols(expr1_clean)
        expr2_normalized = normalize_math_symbols(expr2_clean)

        return expr1_normalized == expr2_normalized

    except Exception as e:
        logger.warning(f"Error comparing expressions '{expr1}' and '{expr2}': {e}")
        return False


def _evaluate_math_set(input_csv: str, gt_csv: str, predicted_list: List[str]) -> Dict[str, Any]:
    problems = load_csv_column(input_csv)
    answers = load_csv_column(gt_csv)
    n = min(len(problems), len(answers), len(predicted_list))
    if n == 0:
        return {"error": "empty_data"}
    preservation_correct = 0
    answer_correct = 0
    overall_correct = 0
    items: List[Dict[str, Any]] = []
    for i in range(n):
        pred_p, pred_a = _parse_math_expression(predicted_list[i])
        gt_p = str(problems[i]).strip()
        gt_a = str(answers[i]).strip()
        preserved = _expressions_equivalent(pred_p, gt_p)
        ans_ok = False
        try:
            if pred_a and gt_a:
                ans_ok = abs(float(pred_a) - float(gt_a)) < 1e-6
            else:
                ans_ok = (pred_a.strip() == gt_a.strip())
        except Exception:
            ans_ok = (pred_a.strip() == gt_a.strip())
        both = preserved and ans_ok
        preservation_correct += 1 if preserved else 0
        answer_correct += 1 if ans_ok else 0
        overall_correct += 1 if both else 0
        items.append({
            "index": int(i),
            "input_problem": gt_p,
            "predicted_problem": pred_p,
            "predicted_answer": pred_a,
            "ground_truth_answer": gt_a,
            "problem_preserved": bool(preserved),
            "answer_correct": bool(ans_ok),
            "overall_correct": bool(both),
        })
    total = float(n)
    return {
        "preservation_rate": float(preservation_correct / total),
        "answer_accuracy": float(answer_correct / total),
        "overall_accuracy": float(overall_correct / total),
        "total_count": int(n),
        "items": items,
    }


def _extract_math_from_image(image_path: str) -> Optional[List[str]]:
    try:
        result = extract_math_from_image(image_path)
        return parse_math_result(result.get("extracted_problems"))
    except Exception:
        return None


def compute_math(
    name: str,
    mode: str = 'vreason_bench_standard',        
) -> Dict:
    """
    Evaluate Math by reading extracted frames from predictions/, using a VLM to parse
    visible expressions into a list, and comparing against inputs/GT CSVs.
    
    Directory layout:
      evaluations/Math/
        inputs/<level>_<numPerImage>/<idx>.{png,csv}
        GT/<level>_<numPerImage>/<idx>.csv
        predictions/<level>_<numPerImage>/<model>_<idx>_<seed>.png
        eval_results/math_eval.json
    """
    dirs = get_eval_directories("Math")
    gt_root = dirs["gt_dir"]
    input_root = dirs["input_dir"]
    predictions_dir = dirs["predictions_dir"]
    eval_results_dir = dirs["eval_results_dir"]
    
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(eval_results_dir, exist_ok=True)

    prediction_files: List[str] = []
    for root, dirs, files in os.walk(predictions_dir):
        for f in files:
            if f.lower().endswith('.png'):
                prediction_files.append(os.path.join(root, f))
    
    results: Dict[str, Dict] = {}
    pres_list: List[float] = []
    ans_list: List[float] = []
    overall_list: List[float] = []

    for frame_path in prediction_files:
        rel_path = os.path.relpath(frame_path, predictions_dir)
        subdir = os.path.dirname(rel_path)

        # Filename convention: <model>_<NN>_<seedK>.png under level folder(s)
        stem = os.path.splitext(os.path.basename(frame_path))[0]
        model: Optional[str] = None
        group: Optional[str] = None
        idx: Optional[str] = None

        m = re.match(
            r"(?P<model>[^_]+)_(?P<idx>\d{2,4})_(?P<seed>seed[0-9A-Za-z]+)$",
            stem,
        )
        if m:
            model = m.group("model")
            idx = m.group("idx")
            parts = [p for p in subdir.replace("\\", "/").split("/") if p]
            group = parts[0] if parts else None

        if not (model and group and idx):
            results[frame_path] = {
                "frame_path": frame_path,
                "error": "invalid_filename_or_group",
            }
            continue

        in_csv = os.path.join(input_root, group, f"{idx}.csv")
        gt_csv = os.path.join(gt_root, group, f"{idx}.csv")
        if not (os.path.exists(in_csv) and os.path.exists(gt_csv)):
            results[frame_path] = {"error": f"missing_input_or_gt_for_{group}/{idx}"}
            continue

        base_name = os.path.splitext(frame_path)[0]
        pred_json = base_name + ".json"
        pred_csv = base_name + ".csv"
        pred_list: List[str] = []

        if os.path.exists(pred_json):
            try:
                with open(pred_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data.get("math_problems"), list):
                    pred_list = [str(x) for x in data["math_problems"]]
                elif isinstance(data.get("parsed_math_problems"), list):
                    pred_list = [str(x) for x in data["parsed_math_problems"]]
            except Exception:
                pred_list = []

        if not pred_list and os.path.exists(pred_csv):
            try:
                with open(pred_csv, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader)
                    for row in reader:
                        if row:
                            pred_list.append(row[0])
            except Exception:
                pred_list = []

        if not pred_list:
            try:
                result = extract_math_from_image(frame_path)
                math_problems = parse_math_result(result.get("extracted_problems")) if isinstance(result, dict) else None
                save_math_extraction_result(result if isinstance(result, dict) else {}, math_problems, output_file=pred_json)
                if math_problems:
                    pred_list = [str(x) for x in math_problems]
            except Exception:
                pred_list = []

        stats = _evaluate_math_set(in_csv, gt_csv, pred_list)
        if all(k in stats for k in ["preservation_rate", "answer_accuracy", "overall_accuracy"]):
            pres_list.append(float(stats["preservation_rate"]))
            ans_list.append(float(stats["answer_accuracy"]))
            overall_list.append(float(stats["overall_accuracy"]))
        
        results[frame_path] = {
            "group": group,
            "index": idx,
            "model": model,
            "frame_path": frame_path,
            "score": float(stats.get("overall_accuracy", 0.0)),
            "gt_index": f"{group}_{idx}" if (group and idx) else idx or group or os.path.splitext(os.path.basename(frame_path))[0],
            **stats,
        }

    unified = build_unified_results(
        results, 
        score_key="score", 
        threshold=1.0,
        predictions_dir=predictions_dir
    )
    
    unified["aggregate"]["task_metrics"] = {
        "average_preservation_rate": float(sum(pres_list) / len(pres_list)) if pres_list else 0.0,
        "average_answer_accuracy": float(sum(ans_list) / len(ans_list)) if ans_list else 0.0,
        "average_overall_accuracy": float(sum(overall_list) / len(overall_list)) if overall_list else 0.0,
    }

    save_json(unified, os.path.join(eval_results_dir, "math_eval.json"))
    return unified



