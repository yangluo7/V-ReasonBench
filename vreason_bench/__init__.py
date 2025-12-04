import importlib
import os
import re
from typing import Dict, List, Optional
import numpy as np
from .utils import extract_last_frame


class VReasonBench(object):
    def __init__(self, device=None):
        self.device = device
        self.base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluations")

    def full_task_list(self) -> List[str]:
        """Return list of all tasks."""
        return [
            "block_slide", "code", "color_connect", "communicating_vessels",
            "lever_balance", "math", "rule_follow", "sequence_completion",
            "shape_fit", "sudoku", "temperature", "tic_tac_toe",
            "visual_analogy", "visual_symmetry",
        ]
    
    def _get_task_folder_name(self, task: str) -> str:
        """Convert task name to folder name (e.g., 'block_slide' -> 'Block_slide')."""
        if not task:
            return task
        return task[0].upper() + task[1:]
    
    def _parse_video_filename(self, filename: str, task_list: List[str]) -> Optional[Dict]:
        parts = filename.rsplit('.', 1)[0].split('_')
        if len(parts) < 4:
            return None
        
        for task in task_list:
            task_words = task.split('_')
            if len(parts) >= len(task_words) + 3:
                if '_'.join(parts[:len(task_words)]) == task:
                    model = parts[len(task_words)]
                    remaining = parts[len(task_words) + 1:]
                    if len(remaining) >= 2:
                        seed = remaining[-1]
                        gt_index = '_'.join(remaining[:-1])
                        return {
                            "task": task,
                            "model": model,
                            "gt_index": gt_index,
                            "seed": seed
                        }
        return None
    
    def distribute_videos(self, video_dir: str, task_list: Optional[List[str]] = None) -> Dict:
        if task_list is None:
            task_list = self.full_task_list()
        
        if not os.path.exists(video_dir):
            return {"error": f"Video directory not found: {video_dir}"}
        
        video_files = [f for f in os.listdir(video_dir) 
                      if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        processed = {}
        skipped = []
        seed_counts = {}
        
        for video_file in video_files:
            parsed = self._parse_video_filename(video_file, task_list)
            
            if not parsed:
                reason = "invalid_format_need_seed" if video_file.rsplit('.', 1)[0].count('_') < 3 else "task_not_found"
                skipped.append({"file": video_file, "reason": reason})
                continue
            
            key = (parsed["task"], parsed["model"], parsed["gt_index"])
            seed_counts.setdefault(key, set()).add(parsed["seed"])
            
            task_folder = self._get_task_folder_name(parsed["task"])
            base_predictions_dir = os.path.join(self.base_dir, task_folder, "predictions")
            
            group = None
            idx_part = parsed["gt_index"]
            m = re.match(r"^(?P<group>.+)_(?P<idx>\d{2,4})$", idx_part)
            if m:
                group = m.group("group")
                true_idx = m.group("idx")
            else:
                true_idx = idx_part
            
            predictions_dir = base_predictions_dir if not group else os.path.join(base_predictions_dir, group)
            os.makedirs(predictions_dir, exist_ok=True)
            
            dest_filename = f"{parsed['model']}_{true_idx}_{parsed['seed']}.png"
            dest_path = os.path.join(predictions_dir, dest_filename)
            
            try:
                if os.path.exists(dest_path) or extract_last_frame(os.path.join(video_dir, video_file), dest_path):
                    processed.setdefault(parsed["task"], []).append({
                        "source": video_file,
                        "frame_saved": dest_path,
                        "model": parsed["model"]
                    })
            except Exception as e:
                print(f" Error extracting frame from: {video_file}")
                skipped.append({"file": video_file, "reason": str(e)})
        
        k_warnings = [
            {
                "task": task, "model": model, "gt_index": gt_idx,
                "k": len(seeds), "expected": 5, "seeds": sorted(seeds)
            }
            for (task, model, gt_idx), seeds in seed_counts.items()
            if len(seeds) != 5
        ]
        
        return {
            "processed": processed,
            "skipped": skipped,
            "k_warnings": k_warnings,
            "summary": {
                "total_videos": len(video_files),
                "processed_count": sum(len(v) for v in processed.values()),
                "skipped_count": len(skipped),
                "tasks_affected": len(processed),
                "k_warnings_count": len(k_warnings)
            }
        }
    
    def evaluate(
        self,
        name: str,
        task_list: Optional[List[str]] = None,
        mode: str = 'vreason_bench_standard',
        video_dir: Optional[str] = None,
    ) -> Dict:
        """
        Evaluate tasks and present model scores.
        
        Detailed results are saved in evaluations/$TASK/ by each task.
        This method aggregates and presents model scores across tasks.
        
        Args:
            name: Evaluation name
            task_list: Tasks to evaluate (default: all tasks)
            mode: Evaluation mode
            video_dir: Directory containing generated videos
            
        Returns:
            Dict with model scores per task and overall aggregation
        """
        if task_list is None:
            task_list = self.full_task_list()

        print(f"Processing videos from {video_dir}...")
        dist_result = self.distribute_videos(video_dir, task_list)
        if "error" in dist_result:
            print(f"Error: {dist_result['error']}")
        else:
            summary = dist_result["summary"]
            print(f"Processed {summary['processed_count']}/{summary['total_videos']} videos")
            print(f"   Frames extracted to predictions/ folders")
            print(f"   Tasks affected: {summary['tasks_affected']}")
            
            if summary['k_warnings_count'] > 0:
                print(f"\n   WARNING: {summary['k_warnings_count']} (model, GT) pair(s) with k != 5")
                for item in dist_result['k_warnings'][:5]:
                    print(f"      - Task: {item['task']}, Model: {item['model']}, GT: {item['gt_index']}")
                    print(f"        Expected 5 seeds, found {item['k']}: {item['seeds']}")
                if summary['k_warnings_count'] > 5:
                    print(f"      ... +{summary['k_warnings_count'] - 5} more")

        task_results = {}
        for task in task_list:
            try:
                task_module = importlib.import_module(f'vreason_bench.{task}')
                evaluate_func = getattr(task_module, f'compute_{task}')
                task_results[task] = evaluate_func(name, mode=mode)
            except Exception as e:
                print(f"Task '{task}' failed: {e}")
                task_results[task] = {"error": str(e)}

        model_task_scores = {}
        
        for task, result in task_results.items():
            if "error" in result:
                continue
            
            for model_entry in result.get("model_summary", []):
                model = model_entry.get("model")
                if model:
                    if model not in model_task_scores:
                        model_task_scores[model] = {}
                    model_task_scores[model][task] = {
                        "mean_score": float(model_entry.get("mean_score", 0.0)),
                        "pass_at_k": float(model_entry.get("pass_at_k", 0.0)),
                    }
        
        model_summary = []
        for model in sorted(model_task_scores.keys()):
            scores = [v["mean_score"] for v in model_task_scores[model].values()]
            pass_at_k_vals = [v["pass_at_k"] for v in model_task_scores[model].values()]
            model_summary.append({
                "model": model,
                "pass_at_k": float(np.mean(pass_at_k_vals)) if pass_at_k_vals else 0.0,
                "mean_score": float(np.mean(scores)) if scores else 0.0,
                "num_tasks": len(scores),
                "per_task": model_task_scores[model],
            })
        
        model_summary.sort(key=lambda x: x["pass_at_k"], reverse=True)
        
        return {
            "model_summary": model_summary,
            "num_tasks_evaluated": len([r for r in task_results.values() if "error" not in r]),
            "num_tasks_failed": sum(1 for r in task_results.values() if "error" in r),
        }