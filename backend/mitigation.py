"""
FairMind AI - Bias Mitigation Module
======================================
Applies pre-processing bias corrections to datasets.
All free, no paid APIs.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")


def reweigh_dataset(df: pd.DataFrame, protected_attr: str, target_col: str) -> pd.DataFrame:
    """
    Reweighing (Kamiran & Calders 2012):
    Assigns sample weights to balance protected groups vs target outcome.
    Favored+Positive → downweight. Disadvantaged+Positive → upweight.
    Returns df with added 'sample_weight' column.
    """
    df = df.copy()
    n = len(df)
    groups = df[protected_attr].unique()
    outcomes = df[target_col].unique()

    weights = np.ones(n)

    for group in groups:
        for outcome in outcomes:
            mask = (df[protected_attr] == group) & (df[target_col] == outcome)
            n_group = (df[protected_attr] == group).sum()
            n_outcome = (df[target_col] == outcome).sum()
            n_group_outcome = mask.sum()

            if n_group_outcome == 0:
                continue

            expected = (n_group * n_outcome) / n
            observed = n_group_outcome
            weight = expected / observed

            weights[mask.values] = weight

    df["sample_weight"] = weights
    return df


def uniform_label_suppressor(df: pd.DataFrame, protected_attr: str, y_pred_col: str,
                               pos_label=1) -> pd.DataFrame:
    """
    Threshold adjustment per group.
    Sets group-specific decision thresholds to equalise positive rates
    across ALL groups toward the overall mean rate.
    Works on predictions without needing model retraining.

    Bug fixes applied:
      1. Removed the *1.2 dead-zone — ALL groups above overall rate are now adjusted
      2. Tiny groups (n<5) are skipped to avoid degenerate flips
      3. Equalise toward MEAN of min/max group rates for maximum DPD reduction
    """
    df = df.copy()
    groups = df[protected_attr].unique()

    # Compute per-group positive rates
    group_rates = {}
    for g in groups:
        mask = df[protected_attr] == g
        if mask.sum() < 5:          # skip tiny groups (e.g. 'Other' with 1 row)
            continue
        group_rates[g] = (df.loc[mask, y_pred_col] == pos_label).mean()

    if len(group_rates) < 2:
        # Nothing meaningful to equalise
        df["mitigated_prediction"] = df[y_pred_col].copy()
        return df

    # Target rate: midpoint between min and max group rate
    # This maximises DPD reduction symmetrically
    min_rate = min(group_rates.values())
    max_rate = max(group_rates.values())
    target_rate = (min_rate + max_rate) / 2.0

    adjusted = df[y_pred_col].copy()

    for group, group_rate in group_rates.items():
        mask = df[protected_attr] == group
        n_group = mask.sum()
        diff = group_rate - target_rate

        if abs(diff) < 0.001:       # already at target — skip
            continue

        if diff < 0:
            # Group is below target → flip some negatives → positive
            neg_mask = mask & (df[y_pred_col] != pos_label)
            n_to_flip = max(1, round(abs(diff) * n_group))
            n_available = int(neg_mask.sum())
            n_to_flip = min(n_to_flip, n_available)
            if n_to_flip > 0:
                flip_idx = df[neg_mask].sample(n=n_to_flip, random_state=42).index
                adjusted[flip_idx] = pos_label
        else:
            # Group is above target → flip some positives → negative
            pos_mask = mask & (df[y_pred_col] == pos_label)
            n_to_flip = max(1, round(diff * n_group))
            n_available = int(pos_mask.sum())
            n_to_flip = min(n_to_flip, n_available)
            if n_to_flip > 0:
                flip_idx = df[pos_mask].sample(n=n_to_flip, random_state=42).index
                adjusted[flip_idx] = 0 if pos_label == 1 else 1

    df["mitigated_prediction"] = adjusted
    return df


def train_fair_model(df: pd.DataFrame,
                     feature_cols: list,
                     target_col: str,
                     protected_attr: str,
                     model_type: str = "logistic") -> dict:
    """
    Trains a standard model and a reweighed fair model, returns both
    with before/after metrics for comparison.
    """
    df = df.copy().dropna(subset=feature_cols + [target_col])

    # Encode categoricals
    X = pd.get_dummies(df[feature_cols], drop_first=True)
    y = df[target_col]

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.3, random_state=42, stratify=y if y.nunique() == 2 else None
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Standard model
    if model_type == "logistic":
        base_model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        base_model = RandomForestClassifier(n_estimators=100, random_state=42)

    base_model.fit(X_train_s, y_train)
    base_preds = base_model.predict(X_test_s)
    base_acc = accuracy_score(y_test, base_preds)

    # Fair model (reweighed)
    df_train_rw = reweigh_dataset(df_train, protected_attr, target_col)
    weights = df_train_rw["sample_weight"].values

    fair_model = LogisticRegression(max_iter=1000, random_state=42)
    fair_model.fit(X_train_s, y_train, sample_weight=weights)
    fair_preds = fair_model.predict(X_test_s)
    fair_acc = accuracy_score(y_test, fair_preds)

    # Compare fairness before/after
    from backend.bias_engine import demographic_parity_difference, disparate_impact_ratio

    df_test_results = df_test.copy()
    df_test_results["base_pred"] = base_preds
    df_test_results["fair_pred"] = fair_preds

    if protected_attr in df_test_results.columns:
        base_dpd, _ = demographic_parity_difference(
            pd.Series(base_preds), df_test_results[protected_attr])
        fair_dpd, _ = demographic_parity_difference(
            pd.Series(fair_preds), df_test_results[protected_attr])
        base_di, _ = disparate_impact_ratio(
            pd.Series(base_preds), df_test_results[protected_attr])
        fair_di, _ = disparate_impact_ratio(
            pd.Series(fair_preds), df_test_results[protected_attr])
    else:
        base_dpd = fair_dpd = base_di = fair_di = None

    return {
        "base_model_accuracy": round(base_acc, 4),
        "fair_model_accuracy": round(fair_acc, 4),
        "accuracy_tradeoff": round(base_acc - fair_acc, 4),
        "base_dpd": round(base_dpd, 4) if base_dpd is not None else None,
        "fair_dpd": round(fair_dpd, 4) if fair_dpd is not None else None,
        "dpd_improvement": round(base_dpd - fair_dpd, 4) if (base_dpd and fair_dpd) else None,
        "base_di": round(base_di, 4) if base_di is not None else None,
        "fair_di": round(fair_di, 4) if fair_di is not None else None,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "model_type": model_type,
        "protected_attribute": protected_attr,
        "message": (
            f"Reweighed model: DPD reduced from {base_dpd:.3f} to {fair_dpd:.3f} "
            f"({(base_dpd-fair_dpd)/base_dpd*100:.1f}% improvement) "
            f"with only {(base_acc-fair_acc)*100:.1f}% accuracy tradeoff."
            if (base_dpd and fair_dpd and base_dpd > 0) else
            "Mitigation complete. Compare metrics above."
        )
    }
