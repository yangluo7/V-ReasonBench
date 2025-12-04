import base64
import json
import csv
import os
import re
import io
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import requests
import numpy as np
from PIL import Image
import torch

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def extract_last_frame(video_path: str, output_path: str) -> str:
    """
    Extract the last frame from a video file and save it as an image.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last_frame = None
    
    if total_frames > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 1))
        ret, frame = cap.read()
        if ret and frame is not None:
            last_frame = frame
    
    if last_frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            last_frame = frame
            frame_count += 1
            
            if frame_count > 10000:
                break
    
    cap.release()
    
    if last_frame is None:
        raise ValueError(
            f"Cannot extract any frame from {video_path}. "
            f"Video may be completely corrupted (reported frames: {total_frames})"
        )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, last_frame)
    return output_path


def save_json(obj: Any, path: str) -> None:
    """Persist JSON to disk, creating parent directories when needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def encode_image(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def iter_video_files(videos_path: str, allowed_exts: Optional[Sequence[str]] = None) -> List[str]:
    """Walk a directory tree and return sorted video file paths."""
    exts = tuple(allowed_exts) if allowed_exts else VIDEO_EXTS
    files: List[str] = []
    for root, _, names in os.walk(videos_path):
        for name in sorted(names):
            if os.path.splitext(name)[1].lower() in exts:
                files.append(os.path.join(root, name))
    return files


def match_gt_for_video(video_filename: str, gt_dir: str, digits: int = 2) -> Optional[str]:
    """Match a video filename to a GT image based on numeric index."""
    base = os.path.basename(video_filename)
    tokens = re.findall(rf"(?<!\d)\d{{{digits}}}(?!\d)", base)
    if tokens:
        idx = int(tokens[0])
    else:
        match = re.search(r"(\d+)", base)
        if not match:
            return None
        idx = int(match.group(1))
    candidate = os.path.join(gt_dir, f"{idx:0{digits}d}.png")
    return candidate if os.path.exists(candidate) else None

# ---- VLM API helpers ----

# API configuration - should be set via environment variables or config file
_VLM_SECRET = os.getenv("VLM_API_KEY", "")
_VLM_URL = os.getenv("VLM_API_URL", "")
_VLM_MODEL = os.getenv("VLM_MODEL", "gemini-2.5-pro")
_VLM_HEADERS = {"Authorization": f"Bearer {_VLM_SECRET}"} if _VLM_SECRET else {}


def _build_image_messages(title: str, image_b64: str, prompt_text: str) -> List[Dict[str, Any]]:
    """Helper to construct a single-image VLM message payload."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": title},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


def _request_vlm(messages: Sequence[Dict[str, Any]], temperature: float) -> requests.Response:
    """Send a request to the VLM endpoint."""
    payload = {"messages": list(messages), "temperature": temperature, "model": _VLM_MODEL}
    return requests.post(url=_VLM_URL, data=json.dumps(payload), headers=_VLM_HEADERS)


def _parse_vlm_response(response: requests.Response) -> Tuple[str, Dict[str, Any]]:
    """Extract content and usage information from a VLM response object."""
    result = response.json()
    content = result["choices"][0]["message"]["content"]
    return content, result.get("usage", {})

_DEFAULT_COMPARISON_PROMPT = (
    """
You are given two images for comparison:
1. The **first image** is the **ground truth (GT)** image (reference image).
2. The **second image** is the **generated image**.

Please compare these two images carefully and provide a detailed analysis covering the following aspects:
- **Object Placement Accuracy:** Evaluate whether the objects in the generated image are in the correct positions compared to the ground truth. This includes checking if any objects are misplaced, missing, or incorrectly positioned.
- **Color Consistency:** Assess whether the colors of the objects, background, and other elements in the generated image match those in the ground truth. Consider whether color discrepancies impact the visual coherence.
- **Shape and Structure Accuracy:** Compare the shapes and structures of objects in the generated image to those in the ground truth. Look for distortions, incorrect proportions, or missing features.
- **Overall Alignment:** Evaluate the overall visual alignment between the generated image and the ground truth, including symmetry, perspective, and general visual coherence.
- **Artifact Detection:** Check for any visual artifacts, such as blurriness, noise, or strange object boundaries in the generated image.

Present your detailed reasoning and observations within <think> and </think> tags.
Then, provide a summary comparison score based on the following weighted metrics:
- **Object Placement Accuracy (Weight: 0.4)**
- **Color Consistency (Weight: 0.3)**
- **Shape and Structure Accuracy (Weight: 0.2)**
- **Overall Alignment (Weight: 0.1)**

Provide the final weighted comparison score within <answer> and </answer> tags, calculated as:
Final Score = 0.4 * Object Placement + 0.3 * Color Consistency + 0.2 * Shape Accuracy + 0.1 * Overall Alignment

Think step by step and be thorough in your comparison.
    """
).strip()


def call_vlm_api(messages: List[Dict], temperature: float = 0.1) -> Tuple[Optional[str], Dict]:
    """Call VLM API and return content and usage."""
    response = _request_vlm(messages, temperature)
    if response.status_code == 413:
        raise ValueError("Payload Too Large")
    content, usage = _parse_vlm_response(response)
    return content, usage


def compare_images(ground_truth_path: str, generated_path: str, custom_prompt: Optional[str] = None) -> Dict:
    """Compare two images using VLM API."""

    # resize the image's width and height to be 2 times smaller with same width and height ratio for both ground truth and generated image
    def _pil_to_b64(img: Image.Image, fmt: str = "JPEG") -> str:
        buffer = io.BytesIO()
        img.save(buffer, format=fmt)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _resize_half_keep_ratio(img: Image.Image) -> Image.Image:
        w, h = img.size
        new_w = max(1, int(round(w * 0.5)))
        new_h = max(1, int(round(h * 0.5)))
        return img.resize((new_w, new_h), Image.LANCZOS)

    gt_img = Image.open(ground_truth_path).convert("RGB")
    gen_img = Image.open(generated_path).convert("RGB")
    gt_resized = _resize_half_keep_ratio(gt_img)
    gen_resized = _resize_half_keep_ratio(gen_img)
    gt_b64 = _pil_to_b64(gt_resized, "JPEG")
    gen_b64 = _pil_to_b64(gen_resized, "JPEG")
    
    """Compare two images using VLM API."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Ground Truth Image:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{gt_b64}"}},
            {"type": "text", "text": "Generated Image:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{gen_b64}"}},
            {"type": "text", "text": custom_prompt or _DEFAULT_COMPARISON_PROMPT},
        ],
    }]
    
    try:
        content, usage = call_vlm_api(messages, temperature=1.0)
    except Exception as e:
        content = f"[VLM call failed] {e}"
    return {
        "ground_truth": ground_truth_path,
        "generated": generated_path,
        "comparison": content,
    }


_DEFAULT_CODE_EXTRACTION_PROMPT = (
'''
Please look at this image and extract exactly what you see written on it.

Instructions:
1. Carefully examine everything visible in the image
2. Extract all Python code, text, comments, or any other content exactly as it appears
3. If you see Python code (classes, functions, variables), include them exactly as shown
4. If you see input/output examples, include them; if not, don't add them
5. Return what you actually see, not what you think should be there
6. If there are multiple code blocks, extract them in order (left to right, top to bottom)
7. Preserve exact Python indentation and formatting

Output format:
Return ONLY a JSON object with the following structure:
{
  "python_code": ["code_block1", "code_block2", ...]
}

Where each code_block is exactly what appears in the image (e.g., complete class definitions, function definitions, etc.)

Example outputs:
{
  "python_code": ["class Solution:\n    def twoSum(self, nums, target):\n        # code here\n        return result"]
}
{
  "python_code": ["def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)", "# Input: n = 5\n# Output: 5"]
}

Please be very careful to:
- Extract exactly what is visible, nothing more, nothing less
- Maintain the correct order if multiple items exist
- Preserve Python indentation and syntax exactly
- Don't add missing parts or assume what should be there
- Return only the JSON format specified above
'''
).strip()



def extract_python_code(llm_prompt: str, python_image_b64: str) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
    messages = _build_image_messages("Python Code Image:", python_image_b64, llm_prompt)
    response = _request_vlm(messages, temperature=0.1)
    if response.status_code == 413:
        raise ValueError("Payload Too Large: The image is too large for the server to process.")
    try:
        content, usage = _parse_vlm_response(response)
        return (
            content,
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            usage.get("total_tokens"),
        )
    except Exception:
        return None, None, None, None


def extract_code_from_image(python_image_path: str, custom_prompt: Optional[str] = None) -> Dict[str, Optional[Any]]:
    if not os.path.exists(python_image_path):
        raise FileNotFoundError(f"Python code image not found: {python_image_path}")

    python_image_b64 = encode_image(python_image_path)
    prompt = custom_prompt if custom_prompt else _DEFAULT_CODE_EXTRACTION_PROMPT

    content, prompt_tokens, completion_tokens, total_tokens = extract_python_code(prompt, python_image_b64)
    return {
        "python_image": python_image_path,
        "extracted_code": content,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def parse_python_result(api_response: Optional[str]) -> Optional[List[str]]:
    try:
        if not api_response:
            return None
        import re as _re
        json_match = _re.search(r"\{.*\}", api_response, _re.DOTALL)
        if json_match:
            json_str = json_match.group()
            parsed = json.loads(json_str)
            if "python_code" in parsed:
                v = parsed["python_code"]
                if isinstance(v, list):
                    return [str(x) for x in v]
        return None
    except Exception:
        return None


def save_extraction_result(result: Dict[str, Optional[Any]], python_code: Optional[List[str]], output_file: str) -> None:
    save_data = {
        "raw_api_response": result.get("extracted_code"),
        "parsed_python_code": python_code,
        "extraction_success": bool(python_code is not None),
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    if python_code is not None:
        csv_file = output_file.replace(".json", ".csv")
        os.makedirs(os.path.dirname(csv_file) or ".", exist_ok=True)
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Code"])
            for code_item in python_code:
                writer.writerow([code_item])


# ---- Math problems extraction from image ----

_DEFAULT_MATH_EXTRACTION_PROMPT = (
'''
Please look at this image and extract exactly what you see written on it.

Instructions:
1. Carefully examine everything visible in the image
2. Extract all text, numbers, mathematical expressions, or any other content exactly as it appears
3. If you see mathematical expressions, include them exactly as shown (whether complete or incomplete)
4. If you see answers, include them; if not, don't add them
5. Return what you actually see, not what you think should be there
6. If there are multiple items, extract them in order (left to right, top to bottom)

Output format:
Return ONLY a JSON object with the following structure:
{
  "math_problems": ["content1", "content2", ...]
}

Where each content is exactly what appears in the image (e.g., "1+1", "1+1=2", "5-3=", "2*4=8", etc.)

Example outputs:
{
  "math_problems": ["1+1=2"]
}
{
  "math_problems": ["1+1=2", "3+2=5", "5-2", "2*3=6"]
}
{
  "math_problems": ["2+3", "6-1=", "4*2=8", "8/2", "5+4=9", "7-2", "3*3=", "9/3=3", "1+8"]
}

Please be very careful to:
- Extract exactly what is visible, nothing more, nothing less
- Maintain the correct order if multiple items exist
- Don't add missing parts or assume what should be there
- Return only the JSON format specified above
'''
).strip()


def extract_math_problems(llm_prompt: str, math_image_b64: str) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
    messages = _build_image_messages("Math Problems Image:", math_image_b64, llm_prompt)
    response = _request_vlm(messages, temperature=0.1)
    if response.status_code == 413:
        raise ValueError("Payload Too Large: The image is too large for the server to process.")
    try:
        content, usage = _parse_vlm_response(response)
        return (
            content,
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            usage.get("total_tokens"),
        )
    except Exception:
        return None, None, None, None


def extract_math_from_image(math_image_path: str, custom_prompt: Optional[str] = None) -> Dict[str, Optional[Any]]:
    if not os.path.exists(math_image_path):
        raise FileNotFoundError(f"Math problems image not found: {math_image_path}")
    math_image_b64 = encode_image(math_image_path)
    prompt = custom_prompt if custom_prompt else _DEFAULT_MATH_EXTRACTION_PROMPT
    content, prompt_tokens, completion_tokens, total_tokens = extract_math_problems(prompt, math_image_b64)
    return {
        "math_image": math_image_path,
        "extracted_problems": content,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def parse_math_result(api_response: Optional[str]) -> Optional[List[str]]:
    try:
        if not api_response:
            return None
        import re as _re
        json_match = _re.search(r"\{.*\}", api_response, _re.DOTALL)
        if json_match:
            json_str = json_match.group()
            parsed = json.loads(json_str)
            if "math_problems" in parsed:
                v = parsed["math_problems"]
                if isinstance(v, list):
                    return [str(x) for x in v]
        return None
    except Exception:
        return None


def save_math_extraction_result(result: Dict[str, Optional[Any]], math_problems: Optional[List[str]], output_file: str) -> None:
    save_data = {
        "raw_api_response": result.get("extracted_problems"),
        "parsed_math_problems": math_problems,
        "extraction_success": bool(math_problems is not None),
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    if math_problems is not None:
        csv_file = output_file.replace(".json", ".csv")
        os.makedirs(os.path.dirname(csv_file) or ".", exist_ok=True)
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Problem"])
            for prob in math_problems:
                writer.writerow([prob])


# ---- Sudoku grid extraction from image ----

_SUDOKU_PROMPT_4X4 = (
'''
You are given a Sudoku puzzle image. Please analyze the image and extract the numbers from each cell in the 4x4 grid.

Instructions:
1. Carefully examine the Sudoku grid in the image
2. For each cell in the 4x4 grid, identify if there is a number present
3. If a cell contains a number, record that number (1-4)
4. If a cell is empty, record it as 0
5. Return the result as a 4x4 matrix in JSON format

Output format:
Return ONLY a JSON object with the following structure:
{
  "sudoku_grid": [
    [row1_col1, row1_col2, ..., row1_col4],
    [row2_col1, row2_col2, ..., row2_col4],
    ...
    [row4_col1, row4_col2, ..., row4_col4]
  ]
}

Where each number represents:
- 1-4: The digit found in that cell
- 0: Empty cell

Example output:
{
  "sudoku_grid": [
    [2, 4, 1, 3],
    [3, 1, 2, 4],
    [0, 2, 3, 1],
    [1, 0, 0, 0],
  ]
}

Please be very careful to:
- Read the numbers accurately
- Maintain the correct row and column positions
- Use 0 for empty cells
- Return only the JSON format specified above
'''
).strip()

_SUDOKU_PROMPT_9X9 = (
'''
You are given a Sudoku puzzle image. Please analyze the image and extract the numbers from each cell in the 9x9 grid.

Instructions:
1. Carefully examine the Sudoku grid in the image
2. For each cell in the 9x9 grid, identify if there is a number present
3. If a cell contains a number, record that number (1-9)
4. If a cell is empty, record it as 0
5. Return the result as a 9x9 matrix in JSON format

Output format:
Return ONLY a JSON object with the following structure:
{
  "sudoku_grid": [
    [row1_col1, row1_col2, ..., row1_col9],
    [row2_col1, row2_col2, ..., row2_col9],
    ...
    [row9_col1, row9_col2, ..., row9_col9]
  ]
}

Where each number represents:
- 1-9: The digit found in that cell
- 0: Empty cell

Example output:
{
  "sudoku_grid": [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
  ]
}

Please be very careful to:
- Read the numbers accurately
- Maintain the correct row and column positions
- Use 0 for empty cells
- Return only the JSON format specified above
'''
).strip()


def extract_sudoku_numbers(llm_prompt: str, sudoku_image_b64: str) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
    messages = _build_image_messages("Sudoku Image:", sudoku_image_b64, llm_prompt)
    response = _request_vlm(messages, temperature=0.1)
    if response.status_code == 413:
        raise ValueError("Payload Too Large: The image is too large for the server to process.")
    try:
        content, usage = _parse_vlm_response(response)
        return (
            content,
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            usage.get("total_tokens"),
        )
    except Exception:
        return None, None, None, None


def extract_sudoku_from_image(sudoku_image_path: str, base: int = 3, custom_prompt: Optional[str] = None) -> Dict[str, Optional[Any]]:
    if not os.path.exists(sudoku_image_path):
        raise FileNotFoundError(f"Sudoku image not found: {sudoku_image_path}")
    sudoku_b64 = encode_image(sudoku_image_path)
    if custom_prompt is not None:
        prompt = custom_prompt
    else:
        if base == 2:
            prompt = _SUDOKU_PROMPT_4X4
        elif base == 3:
            prompt = _SUDOKU_PROMPT_9X9
        else:
            raise ValueError("Invalid base size. Must be 2 or 3.")
    content, pt, ct, tt = extract_sudoku_numbers(prompt, sudoku_b64)
    return {
        "sudoku_image": sudoku_image_path,
        "extracted_numbers": content,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": tt,
    }


def parse_sudoku_result(api_response: Optional[str]) -> Optional[List[List[int]]]:
    """Parse sudoku_grid field from API response."""
    if not api_response:
        return None
    try:
        match = re.search(r"\{.*\}", api_response, re.DOTALL)
        if not match:
            return None
        grid = json.loads(match.group()).get("sudoku_grid")
        if not isinstance(grid, list):
            return None
        parsed = [[int(x) for x in row] for row in grid]
        n = len(parsed)
        if n in (4, 9) and all(len(r) == n for r in parsed):
            return parsed
    except Exception:
        pass
    return None


def save_sudoku_extraction_result(result: Dict, sudoku_grid: Optional[List[List[int]]], output_file: str) -> None:
    """Save sudoku extraction result."""
    save_data = {
        "raw_api_response": result.get("extracted_content"),
        "parsed_sudoku_grid": sudoku_grid,
        "extraction_success": sudoku_grid is not None,
    }
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    if sudoku_grid:
        csv_file = output_file.replace(".json", ".csv")
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for row in sudoku_grid:
                writer.writerow(row)


def parse_vlm_score(vlm_response: str) -> Optional[float]:
    """Extract numeric score from VLM response."""
    if not vlm_response:
        return None
    
    match = re.search(r"<answer>(.*?)</answer>", vlm_response, re.DOTALL | re.IGNORECASE)
    if match:
        num_match = re.search(r"(\d+\.?\d*)", match.group(1))
        if num_match:
            return float(num_match.group(1))
    
    for pattern in [r"[Ff]inal\s+[Ss]core\s*[:=]\s*(\d+\.?\d*)", r"[Ss]core\s*[:=]\s*(\d+\.?\d*)"]:
        match = re.search(pattern, vlm_response)
        if match:
            return float(match.group(1))
    
    return None


# ---- Pass@k and unified results ----

def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    """Standard pass@k: probability at least one success in k draws (without replacement).

    Uses the canonical formula: 1 - C(n - c, k) / C(n, k), with stable log-space computation.
    n: total samples generated for this task, c: number of correct samples among n, k: draws.
    """
    if k <= 0 or n <= 0 or c <= 0:
        return 0.0
    if k > n:
        k = n
    if n - c < k:
        return 1.0
    import math
    log_ratio = 0.0
    for i in range(k):
        log_ratio += math.log(n - c - i) - math.log(n - i)

    ratio = math.exp(log_ratio)
    if ratio < 0.0:
        ratio = 0.0
    if ratio > 1.0:
        ratio = 1.0

    return 1.0 - ratio


def infer_model_from_path(video_path: str) -> str:
    """Infer model name from video path or filename."""
    filename = os.path.basename(video_path)
    match = re.search(r'__model_([^_]+(?:_[^_]+)*)__', filename)
    if match:
        return match.group(1)

    parts = video_path.split(os.sep)
    for i, part in enumerate(parts):
        if 'video' in part.lower() or 'output' in part.lower():
            if i + 1 < len(parts):
                return parts[i + 1]
    return os.path.basename(os.path.dirname(video_path))


def infer_gt_index_from_path(video_path: str, base_name: Optional[str] = None) -> str:
    """Infer GT index from filename (prefer two-digit patterns)."""
    base_name = base_name or os.path.basename(video_path)
    
    tokens = re.findall(r"(?<!\d)\d{2}(?!\d)", base_name)
    if tokens:
        return tokens[0]
    
    match = re.search(r"(\d+)", base_name)
    return match.group(1).zfill(2) if match else os.path.splitext(base_name)[0]


def infer_k_values_from_folders(predictions_dir: str) -> Dict[str, Dict[str, int]]:
    """Count predictions per GT per model (only .png files with __model_)."""
    from collections import defaultdict
    k_counts = defaultdict(lambda: defaultdict(int))
    
    for root, _, files in os.walk(predictions_dir):
        rel_path = os.path.relpath(root, predictions_dir)
        subdir = "" if rel_path == "." else rel_path.replace(os.sep, '_')
        
        for fname in files:
            if '__model_' in fname and fname.lower().endswith('.png'):
                match = re.match(r"(\d{2})", fname)
                if match:
                    idx = match.group(1)
                    gt_index = f"{subdir}_{idx}" if subdir else idx
                    
                    model_match = re.search(r'__model_([^_]+(?:_[^_]+)*)__', fname)
                    if model_match:
                        model = model_match.group(1)
                        k_counts[model][gt_index] += 1
    
    return {model: dict(gts) for model, gts in k_counts.items()}


def build_unified_results(results: Dict[str, Dict], score_key: str = "score", 
                         threshold: float = 0.5, predictions_dir: Optional[str] = None, k_budget: int = 5) -> Dict:
    """Build unified result format with pass@k calculation."""
    from collections import defaultdict
    
    k_values_per_gt_per_model = {}
    invalid_k_gts = []
    if predictions_dir and os.path.exists(predictions_dir):
        k_values_per_gt_per_model = infer_k_values_from_folders(predictions_dir)
        for model, gt_dict in k_values_per_gt_per_model.items():
            for gt_index, k in gt_dict.items():
                if k != 5:
                    invalid_k_gts.append({"model": model, "gt_index": gt_index, "k": k})
    
    per_model = defaultdict(list)
    per_gt = defaultdict(list)
    all_scores = []
    
    for video_path, result in results.items():
        if "error" in result:
            continue
        
        score = result.get(score_key, 0.0)
        gt_index = result.get("gt_index", infer_gt_index_from_path(video_path))
        model = result.get("model", infer_model_from_path(video_path))
        
        result.update({"gt_index": gt_index, "model": model, "passed": score >= threshold})
        
        per_model[model].append(score)
        per_gt[gt_index].append(score)
        all_scores.append(score)
    
    per_model_results = {}
    all_model_pass_list = []
    
    for model in per_model.keys():
        model_by_gt = defaultdict(list)
        for video_path, result in results.items():
            if result.get("model") == model and "error" not in result:
                model_by_gt[result["gt_index"]].append({
                    "path": video_path,
                    "score": result[score_key],
                    "passed": result["passed"]
                })
        
        per_gt_pass_list = []
        for gt_index, gt_list in model_by_gt.items():
            n_gt = len(gt_list)
            c_gt = sum(1 for e in gt_list if e["passed"])
            per_gt_pass_list.append(calculate_pass_at_k(n_gt, c_gt, k_budget))

        model_overall_pass = float(np.mean(per_gt_pass_list)) if per_gt_pass_list else 0.0
        all_model_pass_list.append(model_overall_pass)
        
        per_model_results[model] = {
            "mean_score": float(np.mean(per_model[model])),
            "count": len(per_model[model]),
            "num_gt": len(model_by_gt),
            "pass_at_k": model_overall_pass
        }
    
    aggregate = {
        "num_videos": len([r for r in results.values() if "error" not in r]),
        "num_gt": len(per_gt),
        "num_models": len(per_model),
        "mean_score": float(np.mean(all_scores)) if all_scores else 0.0,
        "pass_at_k": float(np.mean(all_model_pass_list)) if all_model_pass_list else 0.0,
        "threshold": threshold,
    }
    
    if k_values_per_gt_per_model:
        aggregate["k_values_per_gt_per_model"] = k_values_per_gt_per_model
        aggregate["invalid_k_gts"] = invalid_k_gts
    
    model_summary = [
        {
            "model": model,
            "mean_score": stats["mean_score"],
            "pass_at_k": stats["pass_at_k"],
            "count": stats["count"],
            "num_gt": stats["num_gt"]
        }
        for model, stats in sorted(per_model_results.items(), key=lambda x: x[1]["pass_at_k"], reverse=True)
    ]
    
    return {
        "model_summary": model_summary,
        "aggregate": aggregate,
        "results": results,
    }


def get_eval_directories(task_name: str) -> Dict[str, str]:
    """Get standardized evaluation directory paths for a task.
    
    Args:
        task_name: Name of the task (e.g., "Temperature", "Shape_fit")
    
    Returns:
        Dictionary with keys: vb_root, eval_root, videos_dir, predictions_dir, 
        eval_results_dir, gt_dir, input_dir
    """
    vb_root = os.path.relpath(os.path.join(os.path.dirname(__file__), os.pardir), start=os.getcwd())
    eval_root = os.path.join(vb_root, "evaluations", task_name)
    
    return {
        "vb_root": vb_root,
        "eval_root": eval_root,
        "videos_dir": os.path.join(eval_root, "video_outputs"),
        "predictions_dir": os.path.join(eval_root, "predictions"),
        "eval_results_dir": os.path.join(eval_root, "eval_results"),
        "gt_dir": os.path.join(eval_root, "GT"),
        "input_dir": os.path.join(eval_root, "inputs"),
    }


def iter_video_files_recursive(videos_root: str) -> List[Tuple[str, str]]:
    """Recursively walk directory tree and return (abs_path, rel_path) tuples for videos."""
    files: List[Tuple[str, str]] = []
    for root, _, names in os.walk(videos_root):
        for name in sorted(names):
            if os.path.splitext(name)[1].lower() in VIDEO_EXTS:
                abs_path = os.path.join(root, name)
                rel_path = os.path.relpath(abs_path, videos_root)
                files.append((abs_path, rel_path))
    return files


def infer_index_and_group(rel_path: str) -> Tuple[Optional[str], Optional[str]]:
    """Infer numeric index and group name from relative path like 'group/00.mp4'.
    
    Returns:
        Tuple of (index, group) where index is zero-padded to 2 digits
    """
    parts = rel_path.replace("\\", "/").split("/")
    if len(parts) >= 2:
        group = parts[-2]
        match = re.search(r"(\d{2,4})", parts[-1])
        if match:
            return match.group(1).zfill(2), group
    match = re.search(r"(\d+)", rel_path)
    return (match.group(1).zfill(2) if match else None), None


def load_csv_column(path: str, skip_header: bool = True) -> List[str]:
    """Load first column from CSV file as list of strings."""
    result: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        if skip_header:
            next(reader, None)
        for row in reader:
            if row:
                result.append(row[0])
    return result


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB array (single pixel or averaged color) to LAB color space."""
    from skimage import color as skcolor
    rgb_normalized = (rgb / 255.0).reshape(1, 1, 3)
    lab = skcolor.rgb2lab(rgb_normalized)
    return lab[0, 0]


def compute_delta_e(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """Compute Delta E (Euclidean distance in LAB space) between two LAB colors."""
    return float(np.sqrt(np.sum((lab1 - lab2) ** 2)))


def crop_black_white_border(
    image: np.ndarray,
    black_threshold: int = 30,
    white_threshold: int = 225
) -> np.ndarray:
    """Crop borders by detecting non-black and non-white content."""
    non_black = np.any(image > black_threshold, axis=2)
    non_white = np.any(image < white_threshold, axis=2)
    valid = non_black & non_white
    rows = np.any(valid, axis=1)
    cols = np.any(valid, axis=0)
    if not np.any(rows) or not np.any(cols):
        return image
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return image[r0:r1+1, c0:c1+1]


def load_image_rgb(path: str) -> np.ndarray:
    """Load image as RGB numpy array."""
    import cv2
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
