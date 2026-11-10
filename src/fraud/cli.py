"""Command-line entry point for experiments."""

import argparse
from pathlib import Path

from fraud.train import new_run_id, run_baseline
from fraud.tune import run_tuning


def main() -> None:
    parser = argparse.ArgumentParser(description="S6E11 LightGBM experiments")
    parser.add_argument("command", choices=["baseline", "tune"])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--drop-feature", action="append", default=[], help="Exclude a feature for ablation")
    parser.add_argument("--importance-sample-size", type=int, default=20000)
    parser.add_argument("--trials", type=int, default=8, help="Number of 5-fold candidates for tune")
    parser.add_argument("--params-from", type=Path, default=None, help="Use params from a prior summary.json")
    args = parser.parse_args()
    if args.n_jobs < 1 or args.importance_sample_size < 100 or args.trials < 1:
        parser.error("--n-jobs and --trials must be >=1; --importance-sample-size must be >=100")
    if args.command == "tune" and (args.drop_feature or args.params_from is not None):
        parser.error("--drop-feature and --params-from are only available for baseline")
    run_id = args.run_id or new_run_id(args.command)
    if args.command == "tune":
        result = run_tuning(args.data_dir, run_id, args.n_jobs, args.trials, args.importance_sample_size)
    else:
        result = run_baseline(
            args.data_dir, run_id, args.n_jobs, args.drop_feature, args.importance_sample_size,
            args.params_from,
        )
    print(f"OOF ROC AUC: {result['oof_roc_auc']:.6f}")
    print(f"Report: experiments/{run_id}/summary.json")
    print(f"Submission: artifacts/{run_id}/submission.csv")


if __name__ == "__main__":
    main()
