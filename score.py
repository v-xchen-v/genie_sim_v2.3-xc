#!/usr/bin/env python3
import os
import json
import glob
import argparse
from collections import defaultdict, OrderedDict

def load_runs(task_dir: str):
    """Yield (filepath, record) for each JSON run file in a task folder."""
    pattern = os.path.join(task_dir, "evaluate_ret_*.json")
    for fp in sorted(glob.glob(pattern)):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Files are expected to be a list; keep only dict items with 'result'
            if isinstance(data, list):
                for rec in data:
                    if isinstance(rec, dict) and "result" in rec:
                        yield fp, rec
            elif isinstance(data, dict) and "result" in data:
                yield fp, data
        except Exception as e:
            print(f"[WARN] Failed to read {fp}: {e}")

def aggregate_task(task_name: str, task_dir: str):
    """
    Aggregate one task folder:
      - average per step in STEPS (if present) across all runs
      - run count
      - mean of the step-averages as a single scalar (avg_steps_mean)
    Returns dict with summary and per-run list.
    """
    step_scores_accum = defaultdict(list)
    per_run = []

    for fp, rec in load_runs(task_dir):
        result = rec.get("result", {})
        scores = result.get("scores", {})
        steps = scores.get("STEPS", {})
        run_steps_out = {}

        if isinstance(steps, dict):
            for step_key, val in steps.items():
                try:
                    v = float(val)
                    step_scores_accum[step_key].append(v)
                    run_steps_out[step_key] = v
                except Exception:
                    pass

        per_run.append({
            "file": os.path.basename(fp),
            "steps": run_steps_out,
            "code": result.get("code"),
            "msg": result.get("msg", ""),
            "start_time": rec.get("start_time"),
            "end_time": rec.get("end_time"),
        })

    avg_steps = {k: (sum(v)/len(v)) for k, v in step_scores_accum.items()} if step_scores_accum else {}
    avg_steps_mean = (sum(avg_steps.values())/len(avg_steps)) if avg_steps else None

    task_summary = {
        "task_name": task_name,
        "num_runs": len(per_run),
        "avg_steps": avg_steps,           # dict: STEP -> average across runs
        "avg_steps_mean": avg_steps_mean, # scalar: mean of per-step averages for this task
        "per_run": per_run,
    }
    return task_summary

def scan_all(base_dir: str):
    """Scan all task subfolders under base_dir and aggregate summaries."""
    tasks = []
    for entry in sorted(os.listdir(base_dir)):
        task_dir = os.path.join(base_dir, entry)
        if os.path.isdir(task_dir):
            # Only include folders that contain evaluate_ret_*.json
            if glob.glob(os.path.join(task_dir, "evaluate_ret_*.json")):
                tasks.append((entry, task_dir))

    summaries = []
    for task_name, task_dir in tasks:
        summaries.append(aggregate_task(task_name, task_dir))

    # Determine global set of step keys
    all_step_keys = set()
    for s in summaries:
        all_step_keys.update(s["avg_steps"].keys())
    all_step_keys = sorted(all_step_keys)

    # Overall per-step averages pooled across all runs (runs-weighted)
    pooled = {k: [] for k in all_step_keys}
    for s in summaries:
        for run in s["per_run"]:
            for k in run["steps"]:
                if k in pooled:
                    pooled[k].append(run["steps"][k])

    overall_per_step_runs_weighted = {k: (sum(v)/len(v) if v else None) for k, v in pooled.items()}

    # Overall scalar: mean of each task's avg_steps_mean (task-mean)
    task_means = [s["avg_steps_mean"] for s in summaries if s["avg_steps_mean"] is not None]
    overall_task_avg_of_task_means = (sum(task_means)/len(task_means)) if task_means else None

    # Another scalar: mean of all pooled step values (runs-weighted across steps)
    all_values = []
    for vals in pooled.values():
        all_values.extend(vals)
    overall_runs_weighted_scalar = (sum(all_values)/len(all_values)) if all_values else None

    return summaries, all_step_keys, overall_per_step_runs_weighted, overall_task_avg_of_task_means, overall_runs_weighted_scalar

def maybe_write_csv(csv_path: str, summaries, all_step_keys, overall_per_step_runs_weighted, overall_task_avg_of_task_means, overall_runs_weighted_scalar):
    import csv
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    # Per-task summary sheet with dynamic step columns
    headers = ["task_name", "num_runs", "avg_steps_mean"] + all_step_keys
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for s in summaries:
            row = [s["task_name"], s["num_runs"], s["avg_steps_mean"]]
            for k in all_step_keys:
                row.append(s["avg_steps"].get(k))
            writer.writerow(row)
        writer.writerow([])
        # Overall lines
        writer.writerow(["OVERALL_task_avg_of_task_means", overall_task_avg_of_task_means])
        writer.writerow(["OVERALL_runs_weighted_scalar_mean_of_all_steps", overall_runs_weighted_scalar])
        # Per-step overall (runs-weighted)
        writer.writerow([])
        writer.writerow(["OVERALL_per_step_runs_weighted"])
        writer.writerow(["step_key", "avg"])
        for k in all_step_keys:
            writer.writerow([k, overall_per_step_runs_weighted.get(k)])

def main():
    parser = argparse.ArgumentParser(description="Aggregate per-step evaluation scores across tasks and runs (no E2E).")
    parser.add_argument("--base_dir", default="source/geniesim/benchmark/output", help="Base directory containing <task_name> subfolders.")
    parser.add_argument("--csv", default=None, help="Optional path to write a CSV summary.")
    parser.add_argument("--json_out", default=None, help="Optional path to write a JSON summary.")
    args = parser.parse_args()

    (summaries,
     all_step_keys,
     overall_per_step_runs_weighted,
     overall_task_avg_of_task_means,
     overall_runs_weighted_scalar) = scan_all(args.base_dir)

    # Pretty print to console
    print("==== Per-task summaries (per-step averages) ====")
    for s in summaries:
        print(f"- {s['task_name']}: runs={s['num_runs']}, avg_steps_mean={s['avg_steps_mean']}")
        if s["avg_steps"]:
            steps_str = ", ".join(f"{k}={s['avg_steps'][k]:.4f}" for k in sorted(s["avg_steps"].keys()))
            print(f"  steps_avg: {steps_str}")
    print("\n==== Overall ====")
    print(f"Task-mean scalar avg (mean of each task's avg_steps_mean): {overall_task_avg_of_task_means}")
    print(f"Runs-weighted scalar avg (mean of all step values across all runs): {overall_runs_weighted_scalar}")
    print("Per-step overall (runs-weighted):")
    for k in all_step_keys:
        v = overall_per_step_runs_weighted.get(k)
        print(f"  {k}: {v}")

    if args.csv:
        maybe_write_csv(args.csv, summaries, all_step_keys, overall_per_step_runs_weighted, overall_task_avg_of_task_means, overall_runs_weighted_scalar)
        print(f"\nWrote CSV summary to: {args.csv}")

    if args.json_out:
        out = {
            "summaries": summaries,  # includes per-run details
            # "overall_per_step_runs_weighted": overall_per_step_runs_weighted,
            "overall_task_avg_of_task_means": overall_task_avg_of_task_means,
            "overall_runs_weighted_scalar": overall_runs_weighted_scalar,
            "all_step_keys": all_step_keys,
        }
        import os
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Wrote JSON summary to: {args.json_out}")

if __name__ == "__main__":
    main()

    # overall_task_avg_of_task_means: mean of each task’s avg_steps_mean (treats tasks equally).
    #   - overall_per_step_runs_weighted: pooled per-step averages across all runs of all tasks.
    #   - overall_runs_weighted_scalar: mean of all step values pooled across all runs (treats runs equally).

