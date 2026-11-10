"""Optional feature transformations, applied identically to train and test."""

import re

import numpy as np
import pandas as pd

from fraud.data import CompetitionData

FEATURE_GROUPS = (
    "numeric_cat_low", "numeric_cat_all", "decimal_digits", "integer_digits",
    "roundness", "domain_v1", "risk_flag_count", "risk_x_velocity",
    "auth_no_auth", "focused_v1",
)

# Only decimal positions with useful variation in train are included. One test
# amount has thousandths, one test velocity has hundredths, and one train risk
# value has hundredths; those nearly constant positions are omitted.
DECIMAL_PLACES = {
    "amount_usd": 2,
    "hours_since_last_txn": 2,
    "distance_from_home_km": 2,
    "account_balance_usd": 2,
    "velocity_score": 1,
    "merchant_risk_score": 1,
}


def _scaled_values(series: pd.Series, places: int) -> tuple[np.ndarray, np.ndarray]:
    values = series.to_numpy(dtype=np.float64)
    valid = np.isfinite(values)
    scaled = np.rint(np.where(valid, np.abs(values), 0) * 10**places).astype(np.int64)
    return scaled, valid


def apply_categorical_encoding(data: CompetitionData, encoding: str) -> None:
    """Add one-hot copies while retaining the original categorical columns."""
    if encoding == "native":
        return
    if encoding != "onehot":
        raise ValueError(f"Unknown categorical encoding: {encoding}")

    columns = list(data.x_train.select_dtypes(include="category").columns)
    if not columns:
        return
    train_dummies = pd.get_dummies(data.x_train[columns], dtype=np.uint8)
    test_dummies = pd.get_dummies(data.x_test[columns], dtype=np.uint8)
    test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=0)
    names = [
        f"oh_{index:03d}__{re.sub(r'[^A-Za-z0-9_]+', '_', column).strip('_')}"
        for index, column in enumerate(train_dummies.columns)
    ]
    train_dummies.columns = names
    test_dummies.columns = names
    data.x_train = pd.concat([data.x_train, train_dummies], axis=1)
    data.x_test = pd.concat([data.x_test, test_dummies], axis=1)


def apply_feature_groups(data: CompetitionData, groups: list[str]) -> None:
    """Add optional, label-free features to both train and test."""
    unknown = set(groups) - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"Unknown feature groups: {sorted(unknown)}")
    source_numeric_columns = [
        column
        for column in data.x_train.columns
        if not column.startswith("oh_")
        and pd.api.types.is_numeric_dtype(data.x_train[column])
        and not pd.api.types.is_bool_dtype(data.x_train[column])
    ]
    if any(group in groups for group in ("numeric_cat_low", "numeric_cat_all")):
        numeric_columns = source_numeric_columns
        if "numeric_cat_all" not in groups:
            numeric_columns = [
                column for column in numeric_columns if data.x_train[column].nunique(dropna=True) <= 1000
            ]
        for column in numeric_columns:
            categories = pd.Index(
                pd.concat([data.x_train[column], data.x_test[column]], ignore_index=True).dropna().unique()
            )
            dtype = pd.CategoricalDtype(categories=categories)
            name = f"{column}__exact_category"
            data.x_train[name] = data.x_train[column].astype(dtype)
            data.x_test[name] = data.x_test[column].astype(dtype)

    if "decimal_digits" in groups:
        for column, places in DECIMAL_PLACES.items():
            if column not in data.x_train:
                continue
            for frame in (data.x_train, data.x_test):
                scaled, valid = _scaled_values(frame[column], places)
                for place in range(1, places + 1):
                    codes = ((scaled // 10 ** (places - place)) % 10).astype(np.int8)
                    codes[~valid] = -1
                    frame[f"{column}__d{place}"] = pd.Categorical.from_codes(
                        codes, categories=range(10)
                    )

    if "integer_digits" in groups:
        for column in source_numeric_columns:
            arrays = []
            for frame in (data.x_train, data.x_test):
                values = frame[column].to_numpy(dtype=np.float64)
                valid = np.isfinite(values)
                whole = np.where(valid, np.floor(np.abs(values)), 0).astype(np.int64)
                arrays.append((frame, whole, valid))
            width = len(str(max(int(whole.max()) for _, whole, _ in arrays)))
            for frame, whole, valid in arrays:
                for place in range(width):
                    codes = ((whole // 10**place) % 10).astype(np.int8)
                    codes[~valid] = -1
                    frame[f"{column}__i{place}"] = pd.Categorical.from_codes(
                        codes, categories=range(10)
                    )

    if "roundness" in groups:
        for column in source_numeric_columns:
            places = DECIMAL_PLACES.get(column, 0)
            arrays = [(frame, *_scaled_values(frame[column], places)) for frame in (data.x_train, data.x_test)]
            width = len(str(max(int(scaled.max()) for _, scaled, _ in arrays)))
            for frame, scaled, valid in arrays:
                last_digit = (scaled % 10).astype(np.int8)
                last_digit[~valid] = -1
                frame[f"{column}__last_digit"] = pd.Categorical.from_codes(
                    last_digit, categories=range(10)
                )
                frame[f"{column}__ends_0_or_5"] = np.isin(last_digit, [0, 5]).astype(np.uint8)
                trailing_zeros = np.zeros(len(scaled), dtype=np.uint8)
                nonzero = (scaled != 0) & valid
                for power in range(1, width + 1):
                    trailing_zeros += ((scaled % 10**power == 0) & nonzero).astype(np.uint8)
                trailing_zeros[(scaled == 0) & valid] = width
                frame[f"{column}__trailing_zeros"] = trailing_zeros

    if "domain_v1" in groups:
        required = {
            "amount_usd", "account_balance_usd", "hours_since_last_txn",
            "distance_from_home_km", "merchant_risk_score", "velocity_score",
            "txn_count_last_24h", "is_foreign_transaction", "card_age_months",
            "customer_age", "time_of_day_hour", "day_of_week",
            "is_new_merchant", "used_vpn", "ip_country_mismatch",
            "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
        }
        missing = required - set(data.x_train.columns)
        if missing:
            raise ValueError(f"domain_v1 requires source features: {sorted(missing)}")
        for frame in (data.x_train, data.x_test):
            amount = frame["amount_usd"]
            balance = frame["account_balance_usd"]
            hours = frame["hours_since_last_txn"]
            distance = frame["distance_from_home_km"]
            risk = frame["merchant_risk_score"]
            velocity = frame["velocity_score"]
            transactions = frame["txn_count_last_24h"]
            frame["amount_to_balance"] = (amount / (balance + 1)).astype(np.float32)
            frame["amount_per_hour"] = (amount / (hours + 1)).astype(np.float32)
            frame["transactions_per_hour"] = (transactions / (hours + 1)).astype(np.float32)
            frame["risk_x_velocity"] = (risk * velocity).astype(np.float32)
            frame["risk_x_amount"] = (risk * amount).astype(np.float32)
            frame["distance_if_foreign"] = (
                distance * frame["is_foreign_transaction"].astype(np.int8)
            ).astype(np.float32)
            frame["card_years_to_customer_age"] = (
                frame["card_age_months"] / (12 * frame["customer_age"])
            ).astype(np.float32)
            for source, name in (
                (amount, "log_amount"),
                (balance, "log_balance"),
                (distance, "log_distance"),
            ):
                frame[name] = np.log1p(source.clip(lower=0)).astype(np.float32)
            for source, period, name in (
                (frame["time_of_day_hour"], 24, "hour"),
                (frame["day_of_week"], 7, "weekday"),
            ):
                phase = 2 * np.pi * source / period
                frame[f"{name}_sin"] = np.sin(phase).astype(np.float32)
                frame[f"{name}_cos"] = np.cos(phase).astype(np.float32)
            flags = [
                "is_foreign_transaction",
                "is_new_merchant",
                "used_vpn",
                "ip_country_mismatch",
                "billing_shipping_mismatch",
                "is_ai_generated_scam_attempt",
            ]
            frame["risk_flag_count"] = frame[flags].sum(axis=1).astype(np.uint8)

    focused = "focused_v1" in groups
    if focused or any(group in groups for group in ("risk_flag_count", "risk_x_velocity", "auth_no_auth")):
        for frame in (data.x_train, data.x_test):
            if focused or "risk_flag_count" in groups:
                flags = [
                    "is_foreign_transaction", "is_new_merchant", "used_vpn",
                    "ip_country_mismatch", "billing_shipping_mismatch",
                    "is_ai_generated_scam_attempt",
                ]
                frame["risk_flag_count"] = frame[flags].sum(axis=1).astype(np.uint8)
            if focused or "risk_x_velocity" in groups:
                frame["risk_x_velocity"] = (
                    frame["merchant_risk_score"] * frame["velocity_score"]
                ).astype(np.float32)
            if focused or "auth_no_auth" in groups:
                frame["auth_no_auth"] = frame["auth_method"].eq("No Authentication").astype(np.uint8)
