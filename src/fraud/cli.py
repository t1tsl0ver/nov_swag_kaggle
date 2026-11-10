"""Command-line entry point for experiments."""

import argparse
from pathlib import Path

from fraud.feature_engineering import FEATURE_GROUPS
from fraud.features import FEATURE_SETS
from fraud.train import new_run_id, run_baseline
from fraud.tune import run_tuning


def main() -> None:
    parser = argparse.ArgumentParser(description="S6E11 LightGBM experiments")
    parser.add_argument("command", choices=["baseline", "tune"])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--drop-feature", action="append", default=[], help="Exclude a feature for ablation")
    parser.add_argument("--feature-set", choices=FEATURE_SETS, default="all")
    parser.add_argument("--categorical-encoding", choices=["native", "onehot"], default="native")
    parser.add_argument("--feature-group", action="append", choices=FEATURE_GROUPS, default=[])
    parser.add_argument("--compare-to", type=Path, default=None, help="Baseline summary.json to compare OOF AUC")
    parser.add_argument("--importance-sample-size", type=int, default=20000)
    parser.add_argument("--trials", type=int, default=8, help="Number of 5-fold candidates for tune")
    parser.add_argument("--params-from", type=Path, default=None, help="Use params from a prior summary.json")
    parser.add_argument("--num-leaves", type=int, default=None, help="Override LightGBM leaves per tree for baseline")
    args = parser.parse_args()
    if args.n_jobs < 1 or args.importance_sample_size < 100 or args.trials < 1:
        parser.error("--n-jobs and --trials must be >=1; --importance-sample-size must be >=100")
    if args.num_leaves is not None and args.num_leaves < 2:
        parser.error("--num-leaves must be >=2")
    if args.command == "tune" and (
        args.drop_feature or args.params_from is not None or args.feature_set != "all"
        or args.compare_to or args.num_leaves is not None
    ):
        parser.error("Feature ablation and comparison options are only available for baseline")
    run_id = args.run_id or new_run_id(args.command)
    if args.command == "tune":
        result = run_tuning(
            args.data_dir, run_id, args.n_jobs, args.trials, args.importance_sample_size,
            args.categorical_encoding, args.feature_group,
        )
    else:
        dropped_features = list(dict.fromkeys([*FEATURE_SETS[args.feature_set], *args.drop_feature]))
        result = run_baseline(
            args.data_dir, run_id, args.n_jobs, dropped_features, args.importance_sample_size,
            args.params_from, args.feature_set, args.compare_to, args.categorical_encoding,
            args.feature_group, args.num_leaves,
        )
    print(f"OOF ROC AUC: {result['oof_roc_auc']:.6f}")
    print(f"Report: experiments/{run_id}/summary.json")
    print(f"Submission: artifacts/{run_id}/submission.csv")


if __name__ == "__main__":
    main()
