# V-ReasonBench

[![arXiv](https://img.shields.io/badge/arXiv-2511.16668-b31b1b.svg)](https://arxiv.org/abs/2511.16668)
[![Website](https://img.shields.io/badge/🌐_V--ReasonBench-Website-green.svg)](https://oahzxl.github.io/VReasonBench/)

A comprehensive benchmark for evaluating video generation models across **four reasoning dimensions**: structured problem-solving, spatial cognition, pattern-based inference, and physical dynamics.

<p align="center">
  <img src="assets/pipeline.png" alt="V-ReasonBench Pipeline" width="800">
</p>

**Key Features:**
- 🎯 **13 reasoning tasks** spanning 4 core dimensions
- 📊 **Pass@5 evaluation** with reproducible, answer-verifiable metrics
- 🔧 **Unified evaluation framework** with automated scoring
- 📁 **Standardized dataset** with clear input-output pairs


## 📋 TODOs

- [x] Release paper
- [x] Release dataset and eval code
- [ ] Release data generation code


## 🚀 Quick Start

### Prerequisites

**Python Dependencies:**
```
pip install -r requirements.txt
```

**Download SAM 2 checkpoint:**
```bash
mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -O checkpoints/sam2.1_hiera_large.pt
```

### API Configuration

Some tasks require VLM API for evaluation. Set these environment variables:

```bash
export VLM_API_KEY="your_api_key_here"
export VLM_API_URL="your_api_url_here"
export VLM_MODEL="gemini-2.5-pro"  # Optional, default value
```

**Tasks requiring VLM API**: code, math, shape_fit, sudoku, temperature, color_connect

### Run Evaluation

Place all generated videos in a directory and evaluate:

```bash
# Single task
python evaluate.py --generated_videos ./my_videos --task block_slide

# Multiple tasks
python evaluate.py --generated_videos ./my_videos --task math code sudoku
```

**Supported tasks:** `math`, `code`, `sudoku`, `tic_tac_toe`, `shape_fit`, `visual_symmetry`, `color_connect`, `sequence_completion`, `visual_analogy`, `rule_follow`, `temperature`, `communicating_vessels`, `block_slide`

**Video naming format:** `<input_name>_<model>_seed<N>.mp4`
- Example: `shape_fit_00_model1_seed0.mp4`
- Each (model, input) pair should have 5 seeds (seed0-seed4) for Pass@5 evaluation

Results are saved to `evaluations/<TaskName>/eval_results/<task_name>_eval.json`


## 📂 Dataset

The `dataset/` folder contains input images for all benchmark tasks in a flat structure.

**Naming Format:**

*Without subtype:* `<task_name>_<index>.png`
- Example: `shape_fit_00.png`, `visual_analogy_resize_05.png`

*With subtype:* `<task_name>_<subtype>_<index>.png`
- Example: `tic_tac_toe_3_05.png`, `math_level1_2_07.png`


## 🎬 Video Generation

To generate videos for evaluation, use inputs from `dataset/` and prompts from `prompts.txt`.

**Workflow:**
1. Pick an input image from `dataset/` (e.g., `shape_fit_00.png`)
2. Get the corresponding task prompt from `prompts.txt`
3. Generate 5 videos per input with different seeds (seed0-seed4)
4. Name outputs: `<input_name>_<model>_seed<N>.mp4`

**Examples:**
- Input: `shape_fit_00.png` → Outputs: `shape_fit_00_model1_seed0.mp4`, `shape_fit_00_model1_seed1.mp4`, ...
- Input: `tic_tac_toe_3_05.png` → Outputs: `tic_tac_toe_3_05_model1_seed0.mp4`, ...

**Task prompts in `prompts.txt`:**
- 10 reasoning task prompts (shape_fit, code, math, etc.)
- 2 sudoku prompts (4x4, 9x9)
- 4 visual symmetry prompts (vertical, horizontal, rotational, diagonal)
- 10 temperature scenario prompts (different ice melting conditions)


## 🎯 Supported Tasks

### Structured Problem-Solving

| Task | Description | GT Format | Key Metrics |
|------|-------------|-----------|-------------|
| **arithmetic operation** | Mathematical expression solving | `GT/<level>/<idx>.csv` | Problem preservation + answer accuracy |
| **code execution** | Code execution and output | `GT/<difficulty>/<idx>.csv` | Problem preservation + execution correctness |
| **sudoku** | Sudoku puzzle solving (4×4, 9×9) | `GT/<idx>.csv` | Cell-by-cell grid accuracy |
| **tic_tac_toe** | Game state progression | `GT/<idx>.png` | Grid cell comparison |

### Spatial Cognition

| Task | Description | GT Format | Key Metrics |
|------|-------------|-----------|-------------|
| **shape fitting** | Shape fitting puzzle solving | Inputs only | VLM-based hole filling accuracy |
| **visual symmetry** | Symmetry completion | `GT/<type>/single/<idx>.png` | Delta-E color accuracy |
| **color_connect** | Color matching and connection | `GT/<idx>.png` | VLM-based connection accuracy |

### Pattern-based Inference

| Task | Description | GT Format | Key Metrics |
|------|-------------|-----------|-------------|
| **sequence completion** | Sequence pattern completion | `GT/<idx>.png` + masks | Shape/background accuracy |
| **analogy solving** | Visual transformation understanding | `GT/<concept>/<idx>.png` | IoU with SAM segmentation |
| **rule following** | Pattern completion following rules | `GT/<idx>.png` | Cell-by-cell grid accuracy |

### Physical Dynamics

| Task | Description | GT Format | Key Metrics |
|------|-------------|-----------|-------------|
| **temperature** | Ice melting under different conditions | `inputs/<idx>.png` | VLM physical reasoning score |
| **lever balance** | Lever balance physics | `GT/<idx>.csv`| Mask-specific pixel accuracy |
| **communicating vessels** | Fluid dynamics | `GT/<idx>.csv` | Mask-specific pixel accuracy |
| **block_slide** | Block sliding puzzle | `GT/<idx>_gt.png` + masks | Shape/background accuracy |


## 📊 Evaluation Details

### Directory Structure

```
evaluations/
  <TaskName>/
    GT/              # Ground truth (images/CSVs)
    inputs/          # Initial state inputs
    predictions/     # Auto-generated: extracted frames
    eval_results/    # Auto-generated: JSON results
```

### Output Format

Results are saved to `evaluations/<TaskName>/eval_results/<task_name>_eval.json`:

```json
{
  "model_summary": [
    {
      "model": "model_name",
      "pass_at_k": 0.92,
      "mean_score": 0.85,
      "count": 50
    }
  ],
  "aggregate": {
    "num_videos": 300,
    "num_gt": 10,
    "num_models": 6,
    "mean_score": 0.61,
    "pass_at_k": 0.18,
    "threshold": 0.95,
  },
  "results": {
    "/path/to/video.mp4": {
      "score": 0.85,
      "gt_index": "01",
      "model": "model_name",
      "passed": true
    }
  }
}
```

### Metrics

**Pass@k:** Probability that at least one of k attempts succeeds (averaged across all GT instances)

**Calculation:**
1. For each (model, GT) pair, check if any of the k predictions pass (score ≥ threshold)
2. Average success rate across all GTs for each model


## 📝 Citation

If you find V-ReasonBench useful for your research, please cite:

```bibtex
@misc{luo2025vreasonbenchunifiedreasoningbenchmark,
      title={V-ReasonBench: Toward Unified Reasoning Benchmark Suite for Video Generation Models}, 
      author={Yang Luo and Xuanlei Zhao and Baijiong Lin and Lingting Zhu and Liyao Tang and Yuqi Liu and Ying-Cong Chen and Shengju Qian and Xin Wang and Yang You},
      year={2025},
      eprint={2511.16668},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.16668}, 
}
```
