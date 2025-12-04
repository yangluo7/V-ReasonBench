import os
import json
import re
from typing import Dict, List, Optional

from .utils import (
    build_unified_results,
    call_vlm_api,
    encode_image,
    get_eval_directories,
    save_json,
)

SHAPE_FIT_PROMPT = (
    '''
You are provided with two images for evaluation:
1. The **first image** shows a panel with multiple holes of various shapes. Some holes are placed correctly or incorrectly with pieces.
2. The **second image** shows the initial panel with unfilled holes and all the available pieces.

Your tasks:

- Hole Filling Assessment
1. For each hole in the panel (in the first image), determine if it is COMPLETELY and SEAMLESSLY placed with a piece of the SAME SHAPE and SIZE.
2. Count: The total number of holes and the number of holes that are fully and correctly filled.
3. Mark any hole that is incorrectly placed (wrong shape or size) by one piece or remains empty as "unfilled".
4. Note the hole that is placed by stacking pieces may be "filled" as the bottom one can be correct.

- Piece Comparison
1. Compare the pieces in both images.
2. If any piece in the first image has a different shape compared to its initial state in the second image (excluding flipping), set shape_changed to true. Otherwise, set it to false.
3. Flipping a piece is allowed; the shape itself must be unchanged. 
4. Creating new shapes or altering existing ones counts as a change.
5. If any hole changes the shape, mark "shape_changed" true.

Return your findings in the following JSON format:
{
  "total_holes": int,         
  "fully_filled_holes": int,
  "shape_changed": bool,
  "hole_1": {"filled": bool, "hole_shape": str, "piece_shape": List[str]},
  "hole_2": {"filled": bool, "hole_shape": str, "piece_shape": List[str]},
  ...
}
'''
).strip()




def _infer_input_image(idx_str: str, inputs_root: str) -> Optional[str]:
    """Find input image for a given index."""
    cand = os.path.join(inputs_root, f"{idx_str}.png")
    return cand if os.path.exists(cand) else None


def _vlm_shape_check(generated_image_path: str, first_frame_image_path: str, custom_prompt: str) -> Dict:
    """Call VLM API to check shape fitting results."""
    gen_b64 = encode_image(generated_image_path)
    first_b64 = encode_image(first_frame_image_path)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "First Image:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{gen_b64}"}},
                {"type": "text", "text": "Second Image:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{first_b64}"}},
                {"type": "text", "text": custom_prompt},
            ],
        }
    ]
    
    try:
        content, usage = call_vlm_api(messages, temperature=0.5)
        tokens = {
            "prompt": usage.get("prompt_tokens"),
            "completion": usage.get("completion_tokens"),
            "total": usage.get("total_tokens"),
        }
        return {"content": content, "tokens": tokens}
    except Exception:
        return {"content": None, "tokens": {"prompt": None, "completion": None, "total": None}}


def _parse_shape_json(text: Optional[str]) -> Optional[Dict]:
    if not text:
        return None
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        return json.loads(m.group(0))
    except Exception:
        return None


def compute_shape_fit(
    name: str,
    local: bool = False,
    mode: str = 'vreason_bench_standard',
) -> Dict:
    """
    Evaluate Shape Fitting by reading prediction frames from the `predictions` folder,
    finding corresponding initial input images, and running VLM JSON check to count
    fully filled holes and detect shape changes.
    
    Expected layout:
      evaluations/Shape_fit/
        inputs/<idx>.png
        predictions/<model>_<idx>_seedK.png
        eval_results/shape_fit_eval.json
    """
    dirs = get_eval_directories("Shape_fit")
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
    fill_rates: List[float] = []
    shape_changed_list: List[bool] = []
    total_holes_list: List[int] = []
    filled_list: List[int] = []

    for frame_path in frame_files:
        base_f = os.path.basename(frame_path)
        stem = os.path.splitext(base_f)[0]

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

        input_img = _infer_input_image(idx_str, input_root)
        if not input_img:
            results[frame_path] = {
                "pred_frame": frame_path,
                "error": "input_image_not_found",
                "expected_input": os.path.join(input_root, f"{idx_str}.png"),
            }
            continue

        vlm = _vlm_shape_check(frame_path, input_img, custom_prompt=SHAPE_FIT_PROMPT)
        parsed = _parse_shape_json(vlm.get("content"))

        total_holes = None
        filled = None
        shape_changed = None
        if parsed:
            total_holes = parsed.get("total_holes")
            filled = parsed.get("fully_filled_holes")
            shape_changed = parsed.get("shape_changed")

        shape_changed = shape_changed if shape_changed is not None else False
        total_holes = total_holes if total_holes is not None else 0
        filled = filled if filled is not None else 0
        
        fill_rate = 0.0
        if isinstance(total_holes, int) and total_holes > 0 and isinstance(filled, int):
            fill_rate = max(0.0, min(1.0, filled / total_holes))
            fill_rates.append(fill_rate)
        shape_changed_list.append(shape_changed)
        total_holes_list.append(total_holes)
        filled_list.append(filled)
        
        results[frame_path] = {
            "pred_frame": frame_path,
            "model": model,
            "input_image": input_img,
            "vlm_text": vlm.get("content"),
            "vlm_tokens": vlm.get("tokens"),
            "total_holes": total_holes,
            "fully_filled_holes": filled,
            "shape_changed": shape_changed,
            "fill_rate": float(fill_rate),
            "score": float(fill_rate) if shape_changed == False else 0.0,
            "gt_index": idx_str,
        }

    unified = build_unified_results(
        results, 
        score_key="score", 
        threshold=1.0,
        predictions_dir=predictions_dir
    )
    
    unified["aggregate"]["task_metrics"] = {
        "average_fill_rate": float(sum(fill_rates) / len(fill_rates)) if fill_rates else 0.0,
        "shape_changed_rate": float(sum(shape_changed_list) / len(shape_changed_list)) if shape_changed_list else 0.0,
        "total_holes_mean": float(sum(total_holes_list) / len(total_holes_list)) if total_holes_list else 0.0,
        "fully_filled_holes_mean": float(sum(filled_list) / len(filled_list)) if filled_list else 0.0,
    }

    save_json(unified, os.path.join(eval_results_dir, "shape_fit_eval.json"))
    return unified
