import os
import re
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

from .utils import (
    build_unified_results,
    compare_images,
    get_eval_directories,
    parse_vlm_score,
    save_json,
)


COLOR_CONNECT_PROMPT = (
     """
You are evaluating a model’s ability to perform **color-connection reasoning**.

**Task Inputs:**
You are given two images:
1. The **first image** is the **Ground Truth (GT)** showing the correct configuration of circles and their correct same-color connections.
2. The **second image** is the **Candidate** (generated result) to evaluate.

**Your Goal:**
Judge whether the Candidate correctly connects circles of the same color as defined by the GT. In the GT there are exactly **four circles** total, consisting of **two colors**, and **each color appears exactly twice** (thus two same-color pairs).

Evaluate the Candidate using the criteria below. Provide per-criterion scores (0–100) inside <think>…</think> to justify your judgment. Then compute a **weighted final score** from these four criteria and output ONLY that number in <answer>…</answer>.

**Evaluation Criteria:**

**1. Circle & Color Identification (Weight: 0.20)**
- Does the Candidate contain **exactly four circles**?
- Do the circles partition into **two colors**, with **two circles per color** matching the GT’s color set?
- Score 0–100 (100 = exact match in counts and color set)

**2. Pair Integrity & Exclusivity (Weight: 0.35) — MANDATORY FOR PAIR CORRECTNESS**
- For each color, do the two circles of that color form **one intended pair** without any circle being connected to the wrong color?
- **No path may connect any circle to a different-colored circle.** Any cross-color connection **invalidates that color’s pair**.
- Score 0–100

**3. Path Continuity & Color Match (Weight: 0.35) — MANDATORY FOR PAIR CORRECTNESS**
- For each pair: Is there a **single continuous, unbroken path** connecting the two circles? (Curves allowed.)
- Does the path’s **stroke color match the pair’s color**?
- Crossings between different-colored paths are allowed **only if they do not merge or share a continuous segment**.
- Score 0–100

**4. Topological Consistency & Cleanliness (Weight: 0.10)**
- Are there **no spurious merges** between paths of different colors?
- Are extra stray strokes absent or, if present, do they **not** create cross-color connections?
- Is the overall routing topologically consistent with the GT (no path detours that imply unintended connections)?
- Score 0–100

**Final Weighted Score:**
Compute
Final Score = 0.20 × Criterion1 + 0.35 × Criterion2 + 0.35 × Criterion3 + 0.10 × Criterion4
The Final Score must be a single number in the range **0–100**.

**Output Format:**
1) Present detailed reasoning and the four criterion scores (0–100) within `<think>` and `</think>` tags.
2) Output ONLY the **Final Score** (a single number) inside `<answer>` tags, with **no other text**.

Example:
<think>
C1: 100 — Four circles detected; two colors; two per color.
C2: 100 — Each color forms an exclusive pair; no cross-color links.
C3: 95  — Paths continuous and same-colored; one minor gap suspicion dismissed.
C4: 90  — No spurious merges; minor stray stroke not creating cross-color link.
</think>
<answer>96.5</answer>

Be precise and consistent with the GT’s colors, counts, and pairings.
"""
).strip()


def compute_color_connect(
    name: str,
    local: bool = False,
    mode: str = "vreason_bench_standard",
) -> Dict:
    """
    Evaluate Color Connect task using a combined score of mask-based accuracy and VLM assessment.
    
    Scoring method:
        - Mask-based accuracy: Pixel-wise matching within white circle regions (from mask image)
        - VLM score: Semantic assessment of correct color connections (0-3, normalized to 0-1)
        - Final score = 0.1 * mask_accuracy + 0.9 * vlm_score_normalized

    Args:
        videos_path: Directory containing videos to evaluate.
        name: Evaluation name (used for saving outputs).
        local, mode: Reserved for future options.

    Returns:
        A dict with per-video results and aggregate statistics.
    """
    dirs = get_eval_directories("Color_connect")
    gt_dir = dirs["gt_dir"]
    predictions_dir = dirs["predictions_dir"]
    eval_results_dir = dirs["eval_results_dir"]
    
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(eval_results_dir, exist_ok=True)
    custom_prompt = COLOR_CONNECT_PROMPT
    
    # Gather prediction frames from predictions_dir using new naming
    frame_files: List[str] = []
    for root, _, files in os.walk(predictions_dir):
        for fname in sorted(files):
            if fname.lower().endswith(".png"):
                frame_files.append(os.path.join(root, fname))

    results: Dict[str, Dict] = {}
    scores: List[float] = []
    vlm_scores: List[float] = []

    for frame_path in frame_files:
        rel = os.path.relpath(frame_path, predictions_dir)
        subdir = os.path.dirname(rel)

        # Expect predictions named as: <model>_<idx>_<seed>.png
        stem = os.path.splitext(os.path.basename(frame_path))[0]
        m = re.match(r"(?P<model>[^_]+)_(?P<idx>\d{2,4})_(?P<seed>seed[0-9A-Za-z]+)$", stem)
        if not m:
            results[frame_path] = {
                "pred_frame": frame_path,
                "error": "invalid_filename",
            }
            continue

        model = m.group("model")
        idx_str = m.group("idx")

        gt_path = os.path.join(gt_dir, subdir, f"{idx_str}.png")

        try:
            cmp = compare_images(gt_path, frame_path, custom_prompt=custom_prompt)
            vlm_text = cmp.get("comparison") if isinstance(cmp, dict) else None
        except Exception:
            cmp = {}
            vlm_text = None
        vlm_score = parse_vlm_score(vlm_text) or 0.0
        
        final_score = vlm_score
        
        scores.append(float(final_score))
        vlm_scores.append(float(vlm_score))

        results[frame_path] = {
            "ground_truth": gt_path,
            "pred_frame": frame_path,
            "model": model,
            "score": float(final_score),
            "gt_index": f"{subdir.replace(os.sep, '_')}_{idx_str}" if subdir else idx_str,
            "vlm_text": vlm_text,
            "vlm_score": float(vlm_score), 
        }

    unified = build_unified_results(
        results, 
        score_key="score", 
        threshold=95,
        predictions_dir=predictions_dir
    )
    
    unified["aggregate"]["task_metrics"] = {
        "average_combined_score": float(sum(scores) / len(scores)) if scores else 0.0,
        "average_vlm_score_normalized": float(sum(vlm_scores) / len(vlm_scores)) if vlm_scores else 0.0,
    }

    save_json(unified, os.path.join(eval_results_dir, "color_connect_eval.json"))
    return unified


