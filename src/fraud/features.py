"""Named feature sets for controlled ablation experiments."""

# Baseline permutation importance had a slightly negative mean AUC drop for
# these five features. This is a hypothesis to test, not proof of harm.
FEATURE_SETS = {
    "all": (),
    "weak_ablation_v1": (
        "distance_from_home_km",
        "time_of_day_hour",
        "device_type",
        "account_balance_usd",
        "channel",
    ),
}
