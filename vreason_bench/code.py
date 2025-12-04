import os
import json
import re
import csv
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .utils import (
    build_unified_results,
    extract_code_from_image,
    get_eval_directories,
    load_csv_column,
    parse_python_result,
    save_extraction_result,
    save_json,
)




def _parse_input_code(s: str) -> Tuple[str, str]:
    if "# Input:" in s:
        head, tail = s.split("# Input:", 1)
        return head.strip(), tail.strip()
    return s.strip(), ""


def _parse_predicted_content(s: str) -> Tuple[str, str]:
    if "# Output:" in s:
        code, tail = s.split("# Output:", 1)
        ans = tail.strip().splitlines()[0].strip() if tail.strip() else ""
        if "# Input:" in code:
            code = code.split("# Input:")[0].strip()
        return code.strip(), ans
    lines = s.strip().splitlines()
    ans = ""
    cut = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        ln = lines[i].strip()
        if ln and not ln.startswith('#'):
            ans = ln
            cut = i
            break
        if ln.startswith('# Input:'):
            cut = i
            break
    code = "\n".join(lines[:cut]).strip()
    return code, ans


def _answers_equal(a: str, b: str) -> bool:
    pa = a.strip().lower()
    pb = b.strip().lower()
    if pa == pb:
        return True
    try:
        return abs(float(pa) - float(pb)) < 1e-6
    except Exception:
        pass
    try:
        import re as _re
        ca = _re.sub(r'[\[\]()]', '', pa)
        cb = _re.sub(r'[\[\]()]', '', pb)
        la = [x.strip() for x in ca.split(',') if x.strip()]
        lb = [x.strip() for x in cb.split(',') if x.strip()]
        return la == lb
    except Exception:
        return False


def _evaluate_code_set(input_csv: str, gt_csv: str, predicted_list: List[str]) -> Dict[str, Any]:
    inputs = load_csv_column(input_csv)
    answers = load_csv_column(gt_csv)
    n = min(len(inputs), len(answers), len(predicted_list))
    if n == 0:
        return {"error": "empty_data"}
    preserved = 0
    ans_ok = 0
    both = 0
    items: List[Dict[str, Any]] = []
    for i in range(n):
        pred_code, pred_ans = _parse_predicted_content(predicted_list[i])
        src_code, src_input = _parse_input_code(inputs[i])
        ok_code = _code_similarity_preserved(src_code, pred_code)
        ok_ans = _answers_equal(pred_ans, answers[i])
        preserved += 1 if ok_code else 0
        ans_ok += 1 if ok_ans else 0
        both += 1 if (ok_code and ok_ans) else 0
        items.append({
            'index': int(i),
            'original_code': src_code,
            'original_input': src_input,
            'predicted_code': pred_code,
            'predicted_answer': pred_ans,
            'ground_truth_answer': answers[i],
            'code_preserved': bool(ok_code),
            'answer_correct': bool(ok_ans),
            'overall_correct': bool(ok_code and ok_ans),
        })
    total = float(n)
    return {
        'preservation_rate': float(preserved / total),
        'answer_accuracy': float(ans_ok / total),
        'overall_accuracy': float(both / total),
        'total_count': int(n),
        'items': items,
    }


def _code_similarity_preserved(code_a: str, code_b: str) -> bool:
    def norm(s: str) -> List[str]:
        s = s.strip().replace('\r\n', '\n').replace('\r', '\n')
        lines = [ln.rstrip() for ln in s.split('\n')]
        out: List[str] = []
        prev_blank = False
        for ln in lines:
            if ln == '':
                if not prev_blank:
                    out.append(ln)
                prev_blank = True
            else:
                out.append(ln)
                prev_blank = False
        return out
    a = norm(code_a)
    b = norm(code_b)
    if a == b:
        return True
    max_len = max(len(a), len(b)) or 1
    same = sum(1 for i in range(min(len(a), len(b))) if a[i].strip() == b[i].strip())
    line_sim = same / max_len
    set_a = set("".join(a).lower())
    set_b = set("".join(b).lower())
    char_sim = (len(set_a & set_b) / len(set_a | set_b)) if (set_a | set_b) else 1.0
    score = 0.7 * line_sim + 0.3 * char_sim
    return score > 0.9


def compute_code(
    name: str,
    mode: str = 'vreason_bench_standard',
) -> Dict:
    """
    Evaluate Code (write_output) by extracting last frames, parsing code + output,
    and computing preservation/answer metrics.

    Directory layout:
      evaluations/Code/
        inputs/<difficulty>/<idx>.{png,csv}
        GT/<difficulty>/<idx>.csv
        video_outputs/<difficulty>/<idx>.mp4
        predictions/<difficulty>/<idx>.{png,json,csv}
        eval_results/code_eval.json
    """
    dirs = get_eval_directories("Code")
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
    pres_list: List[float] = []
    ans_list: List[float] = []
    overall_list: List[float] = []

    for frame_path in frame_files:
        rel = os.path.relpath(frame_path, predictions_dir)
        subdir = os.path.dirname(rel)

        stem = os.path.splitext(os.path.basename(frame_path))[0]
        m = re.match(r"(?P<model>[^_]+)_(?P<idx>\d{2,4})_(?P<seed>seed[0-9A-Za-z]+)$", stem)
        if not m:
            results[frame_path] = {
                "pred_frame": frame_path,
                "error": "invalid_filename",
            }
            continue

        model = m.group("model")
        idx = m.group("idx")

        parts = [p for p in subdir.replace("\\", "/").split("/") if p]
        group = parts[0] if parts else ""

        in_csv = os.path.join(input_root, group, f"{idx}.csv")
        gt_csv = os.path.join(gt_root, group, f"{idx}.csv")
        if not (os.path.exists(in_csv) and os.path.exists(gt_csv)):
            results[frame_path] = {
                "pred_frame": frame_path,
                "error": f"missing_input_or_gt_for_{group}/{idx}",
                "input_csv": in_csv,
                "gt_csv": gt_csv,
            }
            continue

        base_out = os.path.splitext(frame_path)[0]
        pred_json = base_out + ".json"
        pred_csv = base_out + ".csv"
        pred_list: List[str] = []
        if os.path.exists(pred_json):
            try:
                with open(pred_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if 'python_code' in data and isinstance(data['python_code'], list):
                    pred_list = [str(x) for x in data['python_code']]
                elif 'parsed_python_code' in data:
                    p = data['parsed_python_code']
                    if isinstance(p, list):
                        if len(p) >= 2:
                            pred_list = [str(p[0]) + "\n\n" + str(p[1])]
                        elif len(p) == 1:
                            pred_list = [str(p[0])]
                elif 'code' in data:
                    pred_list = [str(data['code'])]
            except Exception:
                pred_list = []
        if not pred_list and os.path.exists(pred_csv):
            try:
                with open(pred_csv, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader)
                    for row in reader:
                        if row:
                            pred_list.append(row[0])
            except Exception:
                pred_list = []

        if not pred_list:
            try:
                result = extract_code_from_image(frame_path)
                python_code = parse_python_result(result.get('extracted_code')) if isinstance(result, dict) else None
                save_extraction_result(result if isinstance(result, dict) else {}, python_code, output_file=pred_json)
                if python_code:
                    pred_list = [str(x) for x in python_code]
            except Exception:
                pred_list = []

        stats = _evaluate_code_set(in_csv, gt_csv, pred_list)
        if all(k in stats for k in ['preservation_rate', 'answer_accuracy', 'overall_accuracy']):
            pres_list.append(float(stats['preservation_rate']))
            ans_list.append(float(stats['answer_accuracy']))
            overall_list.append(float(stats['overall_accuracy']))

        results[frame_path] = {
            'group': group,
            'index': idx,
            'pred_frame': frame_path,
            'model': model,
            'score': float(stats.get('overall_accuracy', 0.0)),
            'gt_index': f"{group}_{idx}" if (group and idx) else idx or group or os.path.splitext(os.path.basename(frame_path))[0],
            **stats,
        }

    unified = build_unified_results(
        results, 
        score_key="score", 
        threshold=1.0,
        predictions_dir=predictions_dir
    )
    
    unified['aggregate']['task_metrics'] = {
        'average_preservation_rate': float(sum(pres_list) / len(pres_list)) if pres_list else 0.0,
        'average_answer_accuracy': float(sum(ans_list) / len(ans_list)) if ans_list else 0.0,
        'average_overall_accuracy': float(sum(overall_list) / len(overall_list)) if overall_list else 0.0,
    }

    save_json(unified, os.path.join(eval_results_dir, 'code_eval.json'))
    return unified



