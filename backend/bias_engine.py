"""
FairMind AI - Bias Detection Engine
====================================
Unique Feature: INTERSECTIONAL bias detection across multiple attribute
combinations (e.g. gender × race × age) — not just single attributes.
This is what real-world discrimination looks like.

All free, no paid APIs.
"""

import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.metrics import confusion_matrix
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# FAIRNESS METRICS
# ─────────────────────────────────────────────────────────────────────────────

def demographic_parity_difference(y_pred, sensitive, pos_label=1):
    """
    Difference in positive prediction rate between groups.
    Ideal = 0. Negative = disadvantaged group has lower rate.
    """
    groups = sensitive.unique()
    if len(groups) < 2:
        return 0.0
    rates = {}
    for g in groups:
        mask = sensitive == g
        if mask.sum() == 0:
            rates[g] = 0.0
        else:
            rates[g] = (y_pred[mask] == pos_label).mean()
    vals = list(rates.values())
    return float(max(vals) - min(vals)), rates


def equalized_odds_difference(y_true, y_pred, sensitive, pos_label=1):
    """
    Max difference in TPR and FPR across groups.
    Ideal = 0.
    """
    groups = sensitive.unique()
    tprs, fprs = {}, {}
    for g in groups:
        mask = sensitive == g
        yt, yp = y_true[mask], y_pred[mask]
        tp = ((yt == pos_label) & (yp == pos_label)).sum()
        fn = ((yt == pos_label) & (yp != pos_label)).sum()
        fp = ((yt != pos_label) & (yp == pos_label)).sum()
        tn = ((yt != pos_label) & (yp != pos_label)).sum()
        tprs[g] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fprs[g] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr_vals = list(tprs.values())
    fpr_vals = list(fprs.values())
    eod = float(max(abs(tpr_vals[i] - tpr_vals[j])
                    for i in range(len(tpr_vals))
                    for j in range(i + 1, len(tpr_vals)))) if len(tpr_vals) > 1 else 0.0
    return eod, tprs, fprs


def disparate_impact_ratio(y_pred, sensitive, pos_label=1):
    """
    Ratio of positive rate between least and most favored group.
    Ideal = 1.0. < 0.8 = legal concern (80% rule).
    """
    groups = sensitive.unique()
    rates = {}
    for g in groups:
        mask = sensitive == g
        if mask.sum() == 0:
            rates[g] = 0.0
        else:
            rates[g] = (y_pred[mask] == pos_label).mean()
    vals = [v for v in rates.values() if v > 0]
    if not vals or max(rates.values()) == 0:
        return 1.0, rates
    di = float(min(vals) / max(rates.values()))
    return di, rates


def false_positive_rate_difference(y_true, y_pred, sensitive, pos_label=1):
    fprs = {}
    for g in sensitive.unique():
        mask = sensitive == g
        yt, yp = y_true[mask], y_pred[mask]
        neg_mask = yt != pos_label
        if neg_mask.sum() == 0:
            fprs[g] = 0.0
        else:
            fprs[g] = float((yp[neg_mask] == pos_label).mean())
    vals = list(fprs.values())
    return float(max(vals) - min(vals)) if vals else 0.0, fprs


def false_negative_rate_difference(y_true, y_pred, sensitive, pos_label=1):
    fnrs = {}
    for g in sensitive.unique():
        mask = sensitive == g
        yt, yp = y_true[mask], y_pred[mask]
        pos_mask = yt == pos_label
        if pos_mask.sum() == 0:
            fnrs[g] = 0.0
        else:
            fnrs[g] = float((yp[pos_mask] != pos_label).mean())
    vals = list(fnrs.values())
    return float(max(vals) - min(vals)) if vals else 0.0, fnrs


def individual_consistency_score(df, feature_cols, y_pred, k=5):
    """
    Measures if similar individuals get similar predictions.
    Score = 1 - (average disagreement with k nearest neighbours).
    Ideal = 1.0
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols or len(df) < k + 1:
        return None

    X = StandardScaler().fit_transform(df[numeric_cols].fillna(0))
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(df))).fit(X)
    distances, indices = nn.kneighbors(X)

    y_arr = np.array(y_pred)
    disagreements = []
    for i, neighbors in enumerate(indices):
        neighbors = neighbors[1:]  # exclude self
        disagreements.append((y_arr[neighbors] != y_arr[i]).mean())
    return float(1 - np.mean(disagreements))


# ─────────────────────────────────────────────────────────────────────────────
# ★ UNIQUE FEATURE: INTERSECTIONAL BIAS SCANNER ★
# ─────────────────────────────────────────────────────────────────────────────

def intersectional_bias_scan(df, protected_attrs, y_pred_col, y_true_col=None, pos_label=1,
                             min_group_size=5, max_combos=1000):
    """
    THE UNIQUE FEATURE:
    Scans ALL pairwise (and triple) combinations of protected attributes
    to find compounding discrimination that single-attribute tools miss.

    Example: A model may be fair on 'gender' alone and fair on 'race' alone,
    but massively biased against 'Black women' as a combined group.

    Returns a ranked list of intersectional groups by disparity severity.
    """
    results = []
    y_pred = df[y_pred_col]

    # EC-6: Guard against intersectional explosion
    total_combos = 1
    for attr in protected_attrs:
        if attr in df.columns:
            total_combos *= df[attr].nunique()
    if total_combos > max_combos:
        # Return only single-attribute results to avoid freezing
        for attr in protected_attrs:
            if attr not in df.columns:
                continue
            dpd, rates = demographic_parity_difference(y_pred, df[attr], pos_label)
            for group, rate in rates.items():
                group_size = int((df[attr] == group).sum())
                if group_size < min_group_size:  # EC-1
                    continue
                results.append({
                    "type": "single",
                    "attributes": [attr],
                    "group": f"{attr}={group}",
                    "positive_rate": round(rate, 4),
                    "dpd": round(dpd, 4),
                    "group_size": group_size,
                })
        results.sort(key=lambda x: abs(x.get("disparity_ratio", 1.0) - 1.0), reverse=True)
        return results

    # Single attributes
    for attr in protected_attrs:
        if attr not in df.columns:
            continue
        dpd, rates = demographic_parity_difference(y_pred, df[attr], pos_label)
        for group, rate in rates.items():
            group_size = int((df[attr] == group).sum())
            if group_size < min_group_size:   # EC-1: skip tiny groups
                continue
            results.append({
                "type": "single",
                "attributes": [attr],
                "group": f"{attr}={group}",
                "positive_rate": round(rate, 4),
                "dpd": round(dpd, 4),
                "group_size": group_size,
            })

    # Pairwise intersections ★
    for attr1, attr2 in combinations(protected_attrs, 2):
        if attr1 not in df.columns or attr2 not in df.columns:
            continue
        # Create combined attribute
        combined = df[attr1].astype(str) + " × " + df[attr2].astype(str)
        dpd, rates = demographic_parity_difference(y_pred, combined, pos_label)
        # Calculate population-level positive rate for reference
        overall_rate = (y_pred == pos_label).mean()
        for group, rate in rates.items():
            # Disparity ratio vs overall
            disparity_ratio = rate / overall_rate if overall_rate > 0 else 1.0
            results.append({
                "type": "intersectional_2way",
                "attributes": [attr1, attr2],
                "group": group,
                "positive_rate": round(rate, 4),
                "disparity_ratio": round(disparity_ratio, 4),
                "dpd": round(dpd, 4),
                "group_size": int((combined == group).sum()),
                "overall_rate": round(float(overall_rate), 4),
            })

    # Triple intersections (if 3+ protected attrs given) ★★
    if len(protected_attrs) >= 3:
        for attr1, attr2, attr3 in combinations(protected_attrs, 3):
            if not all(a in df.columns for a in [attr1, attr2, attr3]):
                continue
            combined = (df[attr1].astype(str) + " × " +
                        df[attr2].astype(str) + " × " +
                        df[attr3].astype(str))
            dpd, rates = demographic_parity_difference(y_pred, combined, pos_label)
            overall_rate = (y_pred == pos_label).mean()
            for group, rate in rates.items():
                disparity_ratio = rate / overall_rate if overall_rate > 0 else 1.0
                group_size = int((combined == group).sum())
                if group_size < 5:  # skip tiny groups
                    continue
                results.append({
                    "type": "intersectional_3way",
                    "attributes": [attr1, attr2, attr3],
                    "group": group,
                    "positive_rate": round(rate, 4),
                    "disparity_ratio": round(disparity_ratio, 4),
                    "dpd": round(dpd, 4),
                    "group_size": group_size,
                    "overall_rate": round(float(overall_rate), 4),
                })

    # Sort by most extreme disparity
    results.sort(key=lambda x: abs(x.get("disparity_ratio", 1.0) - 1.0), reverse=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# ★ UNIQUE FEATURE 2: COUNTERFACTUAL FAIRNESS PROBE ★
# ─────────────────────────────────────────────────────────────────────────────

def counterfactual_fairness_probe(df, protected_attr, y_pred_col, model=None):
    """
    For each individual, flips their protected attribute and checks
    if the prediction changes. If yes → counterfactually UNFAIR.

    Returns: % of individuals whose prediction would change if only
    their protected attribute changed (everything else identical).
    """
    if protected_attr not in df.columns:
        return None

    groups = df[protected_attr].unique()
    if len(groups) != 2:
        return None  # Only works cleanly for binary attributes

    g0, g1 = groups[0], groups[1]
    changed = 0
    total = 0

    if model is not None:
        # Use actual model to flip predictions
        feature_cols = [c for c in df.columns
                        if c != protected_attr and
                        df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
        if not feature_cols:
            return None

        df_flipped = df.copy()
        df_flipped[protected_attr] = df_flipped[protected_attr].apply(
            lambda x: g1 if x == g0 else g0
        )
        original_preds = np.array(df[y_pred_col])
        try:
            flipped_preds = model.predict(df_flipped[feature_cols].fillna(0))
            changed = (original_preds != flipped_preds).sum()
            total = len(original_preds)
        except Exception:
            return None
    else:
        # Statistical proxy: compare prediction rate difference
        r0 = (df[df[protected_attr] == g0][y_pred_col]).mean()
        r1 = (df[df[protected_attr] == g1][y_pred_col]).mean()
        # Estimate: if the attribute flip is associated with prediction change
        return {
            "method": "statistical_proxy",
            "group_0": str(g0),
            "group_1": str(g1),
            "rate_group_0": round(float(r0), 4),
            "rate_group_1": round(float(r1), 4),
            "estimated_flip_rate": round(abs(float(r0) - float(r1)), 4),
            "interpretation": (
                "HIGH counterfactual unfairness - attribute strongly affects outcome"
                if abs(r0 - r1) > 0.1
                else "LOW counterfactual unfairness - attribute has minimal direct effect"
            )
        }

    return {
        "method": "model_based",
        "individuals_tested": total,
        "predictions_changed": int(changed),
        "flip_rate": round(changed / total, 4) if total > 0 else 0.0,
        "interpretation": (
            f"{changed}/{total} individuals ({100*changed/total:.1f}%) would get a "
            f"different prediction if only their '{protected_attr}' changed."
        )
    }


# ─────────────────────────────────────────────────────────────────────────────
# ★ UNIQUE FEATURE 3: BIAS TIMELINE / DRIFT DETECTION ★
# ─────────────────────────────────────────────────────────────────────────────

def bias_drift_analysis(df, protected_attr, y_pred_col, time_col=None, pos_label=1):
    """
    If a time column is present, shows how bias has CHANGED over time.
    Catches models that become MORE biased as new data arrives.
    """
    if time_col is None or time_col not in df.columns:
        return None

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.dropna(subset=[time_col])

    if df[time_col].nunique() < 2:
        return None

    # Create time periods (quarters or equal bins)
    df["_period"] = pd.qcut(df[time_col].astype(np.int64), q=min(4, df[time_col].nunique()),
                             duplicates='drop', labels=False)

    timeline = []
    for period in sorted(df["_period"].unique()):
        period_df = df[df["_period"] == period]
        dpd, rates = demographic_parity_difference(
            period_df[y_pred_col], period_df[protected_attr], pos_label
        )
        timeline.append({
            "period": int(period),
            "period_label": f"Period {period + 1}",
            "dpd": round(dpd, 4),
            "group_rates": {str(k): round(v, 4) for k, v in rates.items()},
            "n_samples": len(period_df),
        })

    # Calculate drift direction
    dpds = [t["dpd"] for t in timeline]
    if len(dpds) >= 2:
        trend = "INCREASING (getting worse)" if dpds[-1] > dpds[0] else "DECREASING (improving)"
    else:
        trend = "STABLE"

    return {
        "timeline": timeline,
        "bias_trend": trend,
        "max_dpd": max(dpds),
        "min_dpd": min(dpds),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SEVERITY SCORING & PLAIN ENGLISH EXPLANATION (LOCAL LLM-FREE)
# ─────────────────────────────────────────────────────────────────────────────

def severity_score(dpd, di_ratio, eod):
    """
    Composite bias severity: 0 (no bias) → 100 (extreme bias)
    """
    # Normalize each metric to 0-100
    dpd_score = min(dpd * 200, 100)          # DPD of 0.5 → score 100
    di_score = min(abs(1 - di_ratio) * 100, 100) if di_ratio is not None else 0
    eod_score = min(eod * 200, 100)           # EOD of 0.5 → score 100
    return round((dpd_score * 0.4 + di_score * 0.35 + eod_score * 0.25), 1)


def generate_plain_explanation(results: dict) -> str:
    """
    Generates a plain-English bias explanation without any external API.
    Uses rule-based expert system. Clear enough for non-technical judges.
    """
    lines = []
    score = results.get("severity_score", 0)
    dpd = results.get("demographic_parity_difference", 0)
    di = results.get("disparate_impact_ratio", 1.0)
    eod = results.get("equalized_odds_difference", 0)
    attr = results.get("primary_attribute", "the protected attribute")
    domain = results.get("domain", "general")
    intersectional = results.get("intersectional_findings", [])

    # Severity label
    if score < 15:
        severity = "LOW"
        emoji = "✅"
    elif score < 40:
        severity = "MEDIUM"
        emoji = "⚠️"
    elif score < 65:
        severity = "HIGH"
        emoji = "🔴"
    else:
        severity = "CRITICAL"
        emoji = "🚨"

    lines.append(f"{emoji} BIAS SEVERITY: {severity} (Score: {score}/100)")
    lines.append("")
    lines.append("WHAT THIS MEANS IN PLAIN ENGLISH:")
    lines.append("─" * 50)

    # DPD explanation
    if dpd < 0.05:
        lines.append(f"✅ Demographic Parity: The model gives similar positive outcomes "
                     f"to different groups of '{attr}'. Difference is only {dpd:.1%} — acceptable.")
    elif dpd < 0.15:
        lines.append(f"⚠️  Demographic Parity: There is a {dpd:.1%} gap in positive outcomes "
                     f"between groups of '{attr}'. This is worth investigating.")
    else:
        lines.append(f"🔴 Demographic Parity: MAJOR CONCERN. One group of '{attr}' receives "
                     f"positive outcomes {dpd:.1%} more often than another. "
                     f"In a {domain} context, this means systematic discrimination.")

    # DI explanation (80% rule)
    if di is not None:
        if di < 0.8:
            lines.append(f"🔴 Disparate Impact: Ratio = {di:.2f} (below the legal 80% threshold). "
                         f"The least-favored group receives only {di:.0%} the positive outcomes "
                         f"of the most-favored group. This would likely FAIL a legal audit.")
        elif di < 0.9:
            lines.append(f"⚠️  Disparate Impact: Ratio = {di:.2f}. Approaching concerning levels. "
                         f"Monitor closely.")
        else:
            lines.append(f"✅ Disparate Impact: Ratio = {di:.2f} — above the 80% legal threshold.")

    # EOD
    if eod > 0.1:
        lines.append(f"🔴 Equalized Odds: {eod:.1%} difference in error rates between groups. "
                     f"Some groups are being incorrectly classified at a much higher rate.")

    # ★ Intersectional findings
    if intersectional:
        lines.append("")
        lines.append("★ INTERSECTIONAL BIAS (UNIQUE FINDING):")
        lines.append("─" * 50)
        lines.append("Standard tools check one attribute at a time. We checked COMBINATIONS.")
        lines.append("")
        worst = intersectional[:3]  # top 3 most biased intersectional groups
        for finding in worst:
            group = finding.get("group", "")
            ratio = finding.get("disparity_ratio", 1.0)
            size = finding.get("group_size", 0)
            ptype = finding.get("type", "")
            if "intersectional" in ptype and abs(ratio - 1.0) > 0.05:
                direction = "MORE" if ratio > 1.0 else "LESS"
                lines.append(f"  • '{group}' (n={size}) receives positive outcomes "
                              f"{abs(ratio - 1.0):.0%} {direction} than average. "
                              f"{'⚠️ Concern.' if abs(ratio-1.0) > 0.2 else 'Minor.'}")

    # Recommendations
    lines.append("")
    lines.append("RECOMMENDED ACTIONS:")
    lines.append("─" * 50)
    if score < 15:
        lines.append("1. Model appears fair. Continue monitoring for bias drift.")
        lines.append("2. Document this audit result for compliance records.")
    elif score < 40:
        lines.append("1. Investigate data collection: are all groups equally represented?")
        lines.append("2. Apply reweighing to balance group representation in training data.")
        lines.append("3. Re-audit after retraining.")
    elif score < 65:
        lines.append("1. DO NOT deploy this model without intervention.")
        lines.append("2. Apply reweighing + adversarial debiasing before retraining.")
        lines.append("3. Conduct a manual review of rejected cases from disadvantaged groups.")
        lines.append("4. Document mitigation steps for legal compliance.")
    else:
        lines.append("1. HALT deployment immediately.")
        lines.append("2. The training data likely contains historical discrimination.")
        lines.append("3. Reconstruct training dataset with equity-aware sampling.")
        lines.append("4. Engage a fairness expert before redeployment.")
        lines.append("5. This level of bias may violate GDPR Article 22 / EU AI Act.")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AUDIT FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_full_audit(df: pd.DataFrame,
                   protected_attrs: list,
                   y_pred_col: str,
                   y_true_col: Optional[str] = None,
                   domain: str = "general",
                   time_col: Optional[str] = None,
                   pos_label=1,
                   max_intersect_combos: int = 1000,
                   min_group_size: int = 5) -> dict:
    """
    Full bias audit pipeline. Returns complete results dict.
    """
    results = {
        "domain": domain,
        "n_samples": len(df),
        "protected_attributes": protected_attrs,
        "prediction_column": y_pred_col,
        "metrics": {},
        "intersectional_findings": [],
        "counterfactual_probes": [],
        "drift_analysis": None,
        "severity_score": 0,
        "plain_explanation": "",
    }

    if y_pred_col not in df.columns:
        results["error"] = f"Column '{y_pred_col}' not found in dataset."
        return results

    # Primary attribute for headline metrics
    primary_attr = protected_attrs[0] if protected_attrs else None

    # ── Per-attribute metrics ──
    for attr in protected_attrs:
        if attr not in df.columns:
            continue

        attr_results = {}

        # EC-1: Filter out tiny groups before computing metrics
        valid_groups = [g for g in df[attr].unique()
                        if (df[attr] == g).sum() >= min_group_size]
        if len(valid_groups) < 2:
            # Not enough valid groups — skip this attribute
            attr_results["skipped"] = (
                f"Fewer than 2 groups in '{attr}' have >= {min_group_size} members. "
                "Metrics cannot be calculated."
            )
            results["metrics"][attr] = attr_results
            continue

        # Work only with rows belonging to valid groups
        df_attr = df[df[attr].isin(valid_groups)].copy()

        dpd, rates = demographic_parity_difference(df_attr[y_pred_col], df_attr[attr], pos_label)
        attr_results["demographic_parity_difference"] = round(dpd, 4)
        attr_results["positive_rates_by_group"] = {str(k): round(v, 4) for k, v in rates.items()}

        di, di_rates = disparate_impact_ratio(df_attr[y_pred_col], df_attr[attr], pos_label)
        attr_results["disparate_impact_ratio"] = round(di, 4)

        if y_true_col and y_true_col in df_attr.columns:
            eod, tprs, fprs = equalized_odds_difference(
                df_attr[y_true_col], df_attr[y_pred_col], df_attr[attr], pos_label)
            attr_results["equalized_odds_difference"] = round(eod, 4)
            attr_results["true_positive_rates"] = {str(k): round(v, 4) for k, v in tprs.items()}
            attr_results["false_positive_rates"] = {str(k): round(v, 4) for k, v in fprs.items()}

            fprd, fprs2 = false_positive_rate_difference(
                df_attr[y_true_col], df_attr[y_pred_col], df_attr[attr], pos_label)
            attr_results["false_positive_rate_difference"] = round(fprd, 4)

            fnrd, fnrs = false_negative_rate_difference(
                df_attr[y_true_col], df_attr[y_pred_col], df_attr[attr], pos_label)
            attr_results["false_negative_rate_difference"] = round(fnrd, 4)
        else:
            eod = 0.0
            attr_results["equalized_odds_difference"] = None

        # Group sizes — show all groups including skipped ones
        attr_results["group_sizes"] = {str(k): int(v) for k, v in df[attr].value_counts().items()}
        attr_results["skipped_groups"] = [
            str(g) for g in df[attr].unique()
            if (df[attr] == g).sum() < min_group_size
        ]

        results["metrics"][attr] = attr_results

        # Counterfactual probe for each binary attribute
        cf = counterfactual_fairness_probe(df_attr, attr, y_pred_col)
        if cf:
            cf["attribute"] = attr
            results["counterfactual_probes"].append(cf)

    # ★ Intersectional scan
    if len(protected_attrs) >= 2:
        results["intersectional_findings"] = intersectional_bias_scan(
            df, protected_attrs, y_pred_col, y_true_col, pos_label,
            min_group_size=min_group_size,
            max_combos=max_intersect_combos,
        )

    # ★ Bias drift analysis
    if time_col:
        results["drift_analysis"] = bias_drift_analysis(
            df, primary_attr, y_pred_col, time_col, pos_label
        )

    # ── Overall severity ──
    if primary_attr and primary_attr in results["metrics"]:
        m = results["metrics"][primary_attr]
        dpd_val = m.get("demographic_parity_difference", 0) or 0
        di_val = m.get("disparate_impact_ratio", 1.0) or 1.0
        eod_val = m.get("equalized_odds_difference", 0) or 0
        score = severity_score(dpd_val, di_val, eod_val)
        results["severity_score"] = score
        results["demographic_parity_difference"] = dpd_val
        results["disparate_impact_ratio"] = di_val
        results["equalized_odds_difference"] = eod_val
        results["primary_attribute"] = primary_attr
    else:
        results["severity_score"] = 0

    # ── Plain English explanation ──
    results["plain_explanation"] = generate_plain_explanation(results)

    # ── Individual fairness ──
    feature_cols = [c for c in df.columns
                    if c not in protected_attrs + [y_pred_col, y_true_col or "__"]
                    and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
    ics = individual_consistency_score(df, feature_cols, df[y_pred_col])
    results["individual_consistency_score"] = round(ics, 4) if ics is not None else None

    return results
