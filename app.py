"""
FairMind AI - Flask Backend
============================
Run: python app.py
All free, no paid APIs required.

Edge-case fixes (v9):
  EC-1  Small group warning  — skip groups <5 AND warn user in response
  EC-2  Non-prediction column — detect raw-data columns, warn user
  EC-3  Binary-only mitigation — validate binary BEFORE mitigating; reject gracefully
  EC-4  Drift batch-mode warning — explain clearly when no real time column found
  EC-5  Large dataset warning  — warn user when CSV is truncated to 5,000 rows
  EC-6  Intersectional explosion guard — cap combinations, warn user
  EC-7  String prediction column — encode consistently in ALL endpoints
  EC-8  API reliability         — already handled in drift_engine; kept
  EC-9  Session expiry          — clear error + user-friendly message
  EC-10 Single-group dataset    — detect & return clear error before crashing
"""

import os
import sys
import json
import uuid
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np

from backend.bias_engine import run_full_audit
from backend.mitigation import reweigh_dataset, uniform_label_suppressor, train_fair_model
from backend.report_generator import generate_pdf_report
from backend.drift_engine import run_drift_audit, build_gemini_prompt, call_gemini_api

app = Flask(__name__,
            template_folder="frontend/templates",
            static_folder="frontend/static")
CORS(app)

UPLOAD_FOLDER = "data/uploads"
REPORT_FOLDER = "data/reports"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

sessions = {}

# Global audit history (persists across sessions per server run)
audit_history = []
MAX_HISTORY = 50  # keep last 50 audits

MAX_ROWS             = 5_000
MIN_GROUP_SIZE       = 5
MAX_INTERSECT_COMBOS = 1_000


# =============================================================================
# SHARED HELPER FUNCTIONS
# =============================================================================

def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def encode_binary_col(df, col):
    """
    EC-3 + EC-7: Encode a string/object prediction column to 0/1.
    Raises ValueError with a clear user-facing message if column is not binary.
    Returns (df, pos_label, warning_or_None).
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in dataset.")

    # Check if it's a text/string column (object or StringDtype in pandas >= 1.0)
    is_text_col = (
        df[col].dtype == object or
        str(df[col].dtype) == "string" or
        hasattr(df[col], "str")
    )
    unique_num = df[col].dropna().nunique()

    if not is_text_col:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) > 2:
            raise ValueError(
                f"Column '{col}' has {len(unique_vals)} unique values "
                f"(e.g. {list(unique_vals[:5])}). "
                "FairMind only supports binary prediction columns (exactly 2 values, e.g. 0/1). "
                "Please select the correct prediction column, or use a column with only two outcomes."
            )
        return df, 1, None

    # Coerce to plain Python str to avoid pandas StringDtype issues
    df = df.copy()
    df[col] = df[col].astype(str)
    unique_vals = df[col].replace("nan", pd.NA).dropna().unique()
    if len(unique_vals) > 2:
        raise ValueError(
            f"Column '{col}' has {len(unique_vals)} unique values "
            f"(e.g. {list(unique_vals[:5])}{'...' if len(unique_vals) > 5 else ''}). "
            "FairMind only supports binary prediction columns (exactly 2 values, "
            "e.g. 0/1, Yes/No, Approved/Rejected). "
            "Please select the correct column."
        )
    if len(unique_vals) < 2:
        raise ValueError(
            f"Column '{col}' has only 1 unique value ({list(unique_vals)}). "
            "A prediction column must have at least 2 distinct outcomes."
        )

    v0, v1 = str(unique_vals[0]), str(unique_vals[1])
    mapping = {v0: 0, v1: 1}
    df[col] = df[col].map(mapping)
    # Any unmapped values become NaN — fill with 0 (should not happen for clean binary)
    df[col] = df[col].fillna(0).astype(int)
    warning = (
        f"Column '{col}' contained text values {[v0, v1]} — "
        f"automatically encoded: {v0}→0, {v1}→1. Positive label set to 1."
    )
    return df, 1, warning


def check_single_group(df, attrs):
    """EC-10: Return list of attributes with only 1 unique value."""
    return [attr for attr in attrs if attr in df.columns and df[attr].nunique() <= 1]


def check_small_groups(df, attrs):
    """EC-1: Return dict of {attr: [small_group_names]} for groups below MIN_GROUP_SIZE."""
    out = {}
    for attr in attrs:
        if attr not in df.columns:
            continue
        small = [str(g) for g, cnt in df[attr].value_counts().items() if cnt < MIN_GROUP_SIZE]
        if small:
            out[attr] = small
    return out


def detect_raw_data_column(df, col):
    """
    EC-2: Heuristic to warn when user may have selected a raw measurement
    column instead of a model prediction/decision column.
    """
    if col not in df.columns:
        return None

    col_lower = col.lower()
    raw_keywords = [
        "bmi", "glucose", "blood_pressure", "bp", "cholesterol",
        "heart_rate", "age", "salary", "score", "temperature",
        "hypertension", "diabetes", "smoker", "smoking",
        "ever_married", "residence", "work_type", "avg_glucose",
    ]
    name_looks_raw = any(kw in col_lower for kw in raw_keywords)

    if df[col].dtype in [np.int64, np.int32, np.float64, np.float32]:
        vals = df[col].dropna().unique()
        unique_ratio = df[col].nunique() / max(len(df), 1)

        if len(vals) == 2 and name_looks_raw:
            return (
                f"Warning: Column '{col}' looks like a raw clinical/demographic measurement "
                "(its name matches medical or demographic keywords). "
                "If this is not an AI model's output, bias metrics will reflect natural "
                "population differences rather than algorithmic discrimination. "
                "Please verify you selected the correct prediction column."
            )
        if len(vals) > 2 and unique_ratio > 0.05:
            return (
                f"Warning: Column '{col}' has {df[col].nunique()} unique values. "
                "This looks like a continuous measurement, not a binary AI prediction. "
                "Bias metrics require a binary prediction column (0/1 or two text values)."
            )
    return None


def count_intersectional_combos(df, attrs):
    """EC-6: Estimate number of intersectional group combinations."""
    total = 1
    for attr in attrs:
        if attr in df.columns:
            total *= df[attr].nunique()
    return total


def get_csv_total_rows(filepath):
    """EC-5: Fast line count."""
    try:
        with open(filepath, "rb") as f:
            return sum(1 for _ in f) - 1
    except Exception:
        return 0


def session_expired_error():
    return jsonify({
        "error": (
            "Session expired or not found. "
            "This usually happens when the server restarted or your session timed out. "
            "Please re-upload your dataset and try again."
        )
    }), 400


# =============================================================================
# ROUTES
# =============================================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if not f.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files supported"}), 400

    session_id = str(uuid.uuid4())
    filepath = os.path.join(UPLOAD_FOLDER, f"{session_id}.csv")
    f.save(filepath)

    try:
        total_rows = get_csv_total_rows(filepath)   # EC-5
        df = pd.read_csv(filepath, nrows=MAX_ROWS)
        df.columns = df.columns.str.strip()

        sessions[session_id] = {
            "filepath":   filepath,
            "columns":    list(df.columns),
            "shape":      df.shape,
            "total_rows": total_rows,
            "dtypes":     {col: str(dtype) for col, dtype in df.dtypes.items()},
        }

        suggestions = suggest_protected_attrs(df)

        # EC-5: Truncation warning
        truncation_warning = None
        if total_rows > MAX_ROWS:
            truncation_warning = (
                f"Your dataset has {total_rows:,} rows, but FairMind analysed only the first "
                f"{MAX_ROWS:,} rows for performance. Results reflect a partial sample. "
                "Consider pre-sampling your CSV before uploading for a full analysis."
            )

        return jsonify({
            "session_id":         session_id,
            "columns":            list(df.columns),
            "shape":              {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
            "total_rows":         total_rows,
            "preview":            df.head(5).fillna("").to_dict(orient="records"),
            "suggestions":        suggestions,
            "dtypes":             {col: str(dtype) for col, dtype in df.dtypes.items()},
            "truncation_warning": truncation_warning,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def suggest_protected_attrs(df):
    protected_keywords = [
        "gender", "sex", "race", "ethnicity", "age", "religion",
        "nationality", "disability", "marital", "income", "caste"
    ]
    target_keywords = [
        "loan", "hired", "approved", "accepted", "outcome", "label",
        "target", "decision", "result", "prediction", "class", "y"
    ]
    cols_lower = {col: col.lower() for col in df.columns}
    protected_suggestions, target_suggestions = [], []
    for col, col_lower in cols_lower.items():
        if any(kw in col_lower for kw in protected_keywords):
            protected_suggestions.append(col)
        if any(kw in col_lower for kw in target_keywords):
            target_suggestions.append(col)
    return {"protected_attributes": protected_suggestions, "target_column": target_suggestions[:1]}


@app.route("/api/audit", methods=["POST"])
def run_audit():
    body = request.json
    if not body:
        return jsonify({"error": "No JSON body"}), 400

    session_id = body.get("session_id")
    if session_id not in sessions:
        return session_expired_error()   # EC-9

    session  = sessions[session_id]
    filepath = session["filepath"]

    try:
        total_rows     = session.get("total_rows", 0)
        df             = pd.read_csv(filepath, nrows=MAX_ROWS)
        df.columns     = df.columns.str.strip()

        protected_attrs = body.get("protected_attrs", [])
        y_pred_col      = body.get("y_pred_col", "")
        y_true_col      = body.get("y_true_col", None) or None
        domain          = body.get("domain", "general")
        time_col        = body.get("time_col", None) or None
        pos_label       = body.get("pos_label", 1)

        if not protected_attrs:
            return jsonify({"error": "Please select at least one protected attribute."}), 400
        if not y_pred_col:
            return jsonify({"error": "Please select the prediction/outcome column."}), 400

        # EC-10: Single-group attributes
        bad_attrs = check_single_group(df, protected_attrs)
        if bad_attrs:
            return jsonify({
                "error": (
                    f"The following protected attribute columns have only ONE unique value: "
                    f"{bad_attrs}. FairMind needs at least 2 groups to compare for bias. "
                    "Please check you selected the correct columns."
                )
            }), 400

        # EC-7 + EC-3: Consistent encoding
        encode_warning = None
        try:
            df, pos_label, encode_warning = encode_binary_col(df, y_pred_col)
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 400

        # EC-2: Raw column heuristic
        raw_col_warning = detect_raw_data_column(df, y_pred_col)

        # EC-1: Small group warnings
        small_group_warnings = check_small_groups(df, protected_attrs)

        # EC-6: Intersectional explosion guard
        intersect_warning = None
        if len(protected_attrs) >= 2:
            combos = count_intersectional_combos(df, protected_attrs)
            if combos > MAX_INTERSECT_COMBOS:
                intersect_warning = (
                    f"Intersectional analysis skipped: your {len(protected_attrs)} selected "
                    f"attributes would produce ~{combos:,} group combinations, which would "
                    f"freeze the browser. FairMind capped this at {MAX_INTERSECT_COMBOS:,}. "
                    "Consider selecting fewer attributes or attributes with fewer unique values."
                )

        results = run_full_audit(
            df=df,
            protected_attrs=protected_attrs,
            y_pred_col=y_pred_col,
            y_true_col=y_true_col,
            domain=domain,
            time_col=time_col,
            pos_label=pos_label,
            max_intersect_combos=MAX_INTERSECT_COMBOS,
            min_group_size=MIN_GROUP_SIZE,
        )

        # Collect all user-facing warnings
        user_warnings = []
        if total_rows > MAX_ROWS:
            user_warnings.append(
                f"Dataset truncated: only {MAX_ROWS:,} of {total_rows:,} rows were analysed. "
                "Results reflect a partial sample."
            )
        if encode_warning:
            user_warnings.append(encode_warning)
        if raw_col_warning:
            user_warnings.append(raw_col_warning)
        for attr, groups in small_group_warnings.items():
            user_warnings.append(
                f"Attribute '{attr}': groups {groups} have fewer than {MIN_GROUP_SIZE} members "
                "and were excluded from analysis. Results for this attribute may be incomplete."
            )
        if intersect_warning:
            user_warnings.append(intersect_warning)

        results["user_warnings"] = user_warnings

        sessions[session_id]["last_audit"] = results
        sessions[session_id]["df_path"]    = filepath

        # ── Feature 1: Save to audit history ──────────────────────────────
        sev_score = results.get("severity_score", 0) or 0
        dpd_val   = results.get("demographic_parity_difference")
        di_val    = results.get("disparate_impact_ratio")

        # Compute severity label from score
        if sev_score < 15:
            sev_label = "LOW"
        elif sev_score < 40:
            sev_label = "MEDIUM"
        elif sev_score < 65:
            sev_label = "HIGH"
        else:
            sev_label = "CRITICAL"

        snapshot = {
            "timestamp":        datetime.utcnow().isoformat() + "Z",
            "session_id":       session_id,
            "protected_attrs":  protected_attrs,
            "y_pred_col":       y_pred_col,
            "domain":           domain,
            "n_rows":           int(df.shape[0]),
            "overall_severity": int(sev_score),
            "severity_label":   sev_label,
            "dpd":  round(float(dpd_val), 4) if dpd_val is not None else None,
            "di":   round(float(di_val),  4) if di_val  is not None else None,
        }
        audit_history.append(snapshot)
        if len(audit_history) > MAX_HISTORY:
            audit_history.pop(0)

        return jsonify(make_serializable(results))

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/mitigate", methods=["POST"])
def mitigate():
    body = request.json
    session_id = body.get("session_id")

    if session_id not in sessions:
        return session_expired_error()   # EC-9

    try:
        filepath       = sessions[session_id]["filepath"]
        df             = pd.read_csv(filepath, nrows=MAX_ROWS)
        df.columns     = df.columns.str.strip()

        protected_attr = body.get("protected_attr")
        y_pred_col     = body.get("y_pred_col")
        method         = body.get("method", "threshold_adjustment")
        pos_label      = body.get("pos_label", 1)

        if not protected_attr or not y_pred_col:
            return jsonify({"error": "protected_attr and y_pred_col are required."}), 400

        # EC-7 + EC-3: Encode & validate binary
        encode_warning = None
        try:
            df, pos_label, encode_warning = encode_binary_col(df, y_pred_col)
        except ValueError as ve:
            return jsonify({
                "error": str(ve),
                "hint": (
                    "Mitigation requires a binary prediction column with exactly 2 distinct values "
                    "(e.g. 0/1, Yes/No, Approved/Rejected). "
                    "If your column has more than 2 values, mitigation is not supported."
                )
            }), 400

        # EC-2: Raw column warning
        raw_col_warning = detect_raw_data_column(df, y_pred_col)

        # EC-10: Single group
        if protected_attr in df.columns and df[protected_attr].nunique() <= 1:
            return jsonify({
                "error": (
                    f"Protected attribute '{protected_attr}' has only one unique group. "
                    "Mitigation requires at least 2 groups to compare."
                )
            }), 400

        # EC-1: Small groups (warn but don't block — uniform_label_suppressor skips them internally)
        small_groups = [
            str(g) for g, cnt in df[protected_attr].value_counts().items()
            if cnt < MIN_GROUP_SIZE
        ] if protected_attr in df.columns else []

        if method == "threshold_adjustment":
            from backend.bias_engine import demographic_parity_difference, disparate_impact_ratio

            df_mitigated = uniform_label_suppressor(df, protected_attr, y_pred_col, pos_label)

            original_dpd, orig_rates = demographic_parity_difference(
                df[y_pred_col], df[protected_attr], pos_label)
            new_dpd, new_rates = demographic_parity_difference(
                df_mitigated["mitigated_prediction"], df[protected_attr], pos_label)
            original_di, _ = disparate_impact_ratio(df[y_pred_col], df[protected_attr], pos_label)
            new_di, _      = disparate_impact_ratio(
                df_mitigated["mitigated_prediction"], df[protected_attr], pos_label)

            original_dpd = float(original_dpd)
            new_dpd      = float(new_dpd)
            original_di  = float(original_di)
            new_di       = float(new_di)
            n_changed    = int((df[y_pred_col] != df_mitigated["mitigated_prediction"]).sum())

            dpd_imp_pct = round((original_dpd - new_dpd) / original_dpd * 100, 1) \
                          if original_dpd > 0.001 else 0.0

            if dpd_imp_pct > 0:
                msg = (f"Reduced DPD by {dpd_imp_pct}% ({original_dpd:.3f} → {new_dpd:.3f}). "
                       f"Changed {n_changed} predictions.")
            elif dpd_imp_pct == 0 and n_changed > 0:
                msg = (f"DI improved ({original_di:.3f} → {new_di:.3f}) with {n_changed} "
                       f"predictions adjusted. DPD gap was already minimal ({original_dpd:.3f}).")
            else:
                msg = (f"Bias gap between groups is very small (DPD={original_dpd:.3f}). "
                       "No significant adjustment needed — model is already near-fair.")

            user_warnings = []
            if encode_warning:
                user_warnings.append(encode_warning)
            if raw_col_warning:
                user_warnings.append(raw_col_warning)
            if small_groups:
                user_warnings.append(
                    f"Groups {small_groups} in '{protected_attr}' have fewer than "
                    f"{MIN_GROUP_SIZE} members and were excluded from threshold adjustment."
                )

            return jsonify({
                "method":      "Threshold Adjustment (Post-processing)",
                "description": "Adjusted decision thresholds per group to equalise positive rates without model retraining.",
                "before": {
                    "dpd": round(original_dpd, 4),
                    "di":  round(original_di,  4),
                    "group_rates": {str(k): round(float(v), 4) for k, v in orig_rates.items()},
                },
                "after": {
                    "dpd": round(new_dpd, 4),
                    "di":  round(new_di,  4),
                    "group_rates": {str(k): round(float(v), 4) for k, v in new_rates.items()},
                },
                "dpd_improvement_pct":   dpd_imp_pct,
                "di_improvement":        round(new_di - original_di, 4),
                "n_predictions_changed": n_changed,
                "already_fair":          original_dpd < 0.05,
                "message":               msg,
                "user_warnings":         user_warnings,
            })

        return jsonify({"error": f"Unknown method: {method}"}), 400

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/report", methods=["POST"])
def download_report():
    body = request.json
    session_id = body.get("session_id")
    gemini_key = body.get("api_key", "").strip()  # optional Gemini key

    if session_id not in sessions:
        return session_expired_error()   # EC-9

    if "last_audit" not in sessions[session_id]:
        return jsonify({"error": "No audit results found. Run audit first."}), 400

    audit_results = sessions[session_id]["last_audit"]
    report_path   = os.path.join(REPORT_FOLDER, f"fairmind_report_{session_id}.pdf")

    # ── Optionally enrich report with Gemini AI narrative ─────────────────
    gemini_narrative = None
    if gemini_key:
        try:
            prompt = _build_report_gemini_prompt(audit_results)
            raw    = call_gemini_api(prompt, gemini_key)
            if raw and not raw.startswith("ERROR"):
                # Strip model prefix if present
                if raw.startswith("[Model: "):
                    raw = raw[raw.index("]")+2:].strip()
                gemini_narrative = raw
                audit_results = dict(audit_results)
                audit_results["gemini_narrative"] = gemini_narrative
        except Exception:
            pass  # Silently fall back to standard report if Gemini fails

    try:
        actual_path = generate_pdf_report(audit_results, report_path)
        return send_file(actual_path, as_attachment=True,
                         download_name="fairmind_bias_report.pdf",
                         mimetype="application/pdf")
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def _build_report_gemini_prompt(audit_results):
    """Build a Gemini prompt for generating the PDF executive narrative."""
    dpd   = audit_results.get("demographic_parity_difference", "N/A")
    di    = audit_results.get("disparate_impact_ratio", "N/A")
    score = audit_results.get("severity_score", 0)
    attrs = audit_results.get("protected_attributes", [])
    domain = audit_results.get("domain", "general")
    return f"""You are an AI fairness compliance expert writing an executive summary for a bias audit report.

Audit data:
- Domain: {domain}
- Protected attributes analysed: {', '.join(attrs) if attrs else 'N/A'}
- Severity score: {score}/100
- Demographic Parity Difference (DPD): {dpd} (threshold < 0.10)
- Disparate Impact Ratio (DI): {di} (threshold ≥ 0.80)

Write a 3-paragraph professional executive summary for a PDF compliance report:
1. Overall fairness verdict and what the scores mean in plain English
2. Key risks and which regulatory frameworks (EU AI Act, EEOC, GDPR) are at risk
3. Top 3 recommended remediation actions

Keep it under 250 words. Be specific, professional, and actionable. Do NOT use markdown."""


@app.route("/api/drift-audit", methods=["POST"])
def drift_audit():
    body = request.json
    if not body:
        return jsonify({"error": "No JSON body"}), 400

    session_id = body.get("session_id")
    if session_id not in sessions:
        return session_expired_error()   # EC-9

    try:
        filepath       = sessions[session_id]["filepath"]
        df             = pd.read_csv(filepath, nrows=MAX_ROWS)
        df.columns     = df.columns.str.strip()

        protected_attr = body.get("protected_attr", "")
        y_pred_col     = body.get("y_pred_col", "")
        time_col       = body.get("time_col", "")
        n_periods      = int(body.get("n_periods", 5))
        pos_label      = body.get("pos_label", 1)

        for field, value in [("protected_attr", protected_attr),
                              ("y_pred_col",     y_pred_col),
                              ("time_col",       time_col)]:
            if not value:
                return jsonify({"error": f"'{field}' is required."}), 400

        # EC-10
        if protected_attr in df.columns and df[protected_attr].nunique() <= 1:
            return jsonify({
                "error": (
                    f"Protected attribute '{protected_attr}' has only one unique group. "
                    "Drift analysis requires at least 2 groups."
                )
            }), 400

        # EC-7
        encode_warning = None
        try:
            df, pos_label, encode_warning = encode_binary_col(df, y_pred_col)
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 400

        # EC-2
        raw_col_warning = detect_raw_data_column(df, y_pred_col)

        # EC-1
        small_groups = (
            [str(g) for g, cnt in df[protected_attr].value_counts().items()
             if cnt < MIN_GROUP_SIZE]
            if protected_attr in df.columns else []
        )

        result = run_drift_audit(
            df=df,
            protected_attr=protected_attr,
            y_pred_col=y_pred_col,
            time_col=time_col,
            n_periods=n_periods,
            pos_label=pos_label,
        )

        if "error" in result:
            return jsonify(result), 400

        # EC-4: Enhanced batch-mode explanation
        if result.get("time_mode") == "BATCH_SEGMENT":
            result["batch_mode_explanation"] = (
                f"BATCH SEGMENT MODE: Column '{time_col}' does not contain real date/time values. "
                "FairMind divided the data into equal segments ordered by that column's values. "
                "This is NOT true temporal drift — the 'periods' are arbitrary data segments, "
                "not time windows. Do NOT present these results as time-based findings. "
                "For genuine temporal analysis, use a column containing actual dates or timestamps."
            )

        user_warnings = []
        if encode_warning:
            user_warnings.append(encode_warning)
        if raw_col_warning:
            user_warnings.append(raw_col_warning)
        if small_groups:
            user_warnings.append(
                f"Groups {small_groups} in '{protected_attr}' have fewer than "
                f"{MIN_GROUP_SIZE} members and may produce unreliable drift metrics."
            )
        result["user_warnings"] = user_warnings

        sessions[session_id]["last_drift"]  = result
        sessions[session_id]["drift_domain"] = body.get("domain", "general")

        return jsonify(make_serializable(result))

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/drift-ai-report", methods=["POST"])
def drift_ai_report():
    body = request.json
    if not body:
        return jsonify({"error": "No JSON body"}), 400

    session_id = body.get("session_id")
    api_key    = body.get("api_key", "").strip()

    if session_id not in sessions:
        return session_expired_error()   # EC-9

    if not api_key:
        return jsonify({"error": "API key is required. Get one free at aistudio.google.com"}), 400

    if "last_drift" not in sessions[session_id]:
        return jsonify({"error": "No drift analysis found. Run drift audit first."}), 400

    try:
        drift_result = sessions[session_id]["last_drift"]
        domain       = sessions[session_id].get("drift_domain", "general")

        prompt = build_gemini_prompt(drift_result, domain)
        raw    = call_gemini_api(prompt, api_key)   # EC-8: already has multi-provider retry

        model_used = "ai (free tier)"
        report     = raw
        if raw.startswith("[Model: "):
            end        = raw.index("]")
            model_used = raw[8:end] + " (free tier)"
            report     = raw[end+2:].strip()

        return jsonify({
            "report":        report,
            "model_used":    model_used,
            "prompt_tokens": len(prompt.split()),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# DEMO DATASETS
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/demo-datasets", methods=["GET"])
def list_demo_datasets():
    return jsonify([
        {
            "id": "compas",
            "name": "COMPAS Recidivism Dataset",
            "description": "Criminal justice AI — predicts recidivism risk. Famous for racial bias.",
            "rows": 3000,
            "protected_attrs": ["race", "sex"],
            "prediction_col": "two_year_recid",
        },
        {
            "id": "adult_income",
            "name": "Adult Income Census",
            "description": "Predicts whether income >$50K. Classic hiring/lending bias benchmark.",
            "rows": 3000,
            "protected_attrs": ["sex", "race"],
            "prediction_col": "income_label",
        },
        {
            "id": "synthetic_hr",
            "name": "Synthetic HR Dataset",
            "description": "Simulated hiring decisions with injected gender and age bias.",
            "rows": 2000,
            "protected_attrs": ["gender", "age_group"],
            "prediction_col": "hired",
        },
    ])


@app.route("/api/load-demo/<dataset_id>", methods=["GET"])
def load_demo_dataset(dataset_id):
    session_id = str(uuid.uuid4())

    generators = {
        "synthetic_hr":  generate_synthetic_hr,
        "compas":        generate_synthetic_compas,
        "adult_income":  generate_synthetic_adult,
    }
    if dataset_id not in generators:
        return jsonify({"error": "Unknown dataset"}), 404

    df = generators[dataset_id]()
    filepath = os.path.join(UPLOAD_FOLDER, f"{session_id}.csv")
    df.to_csv(filepath, index=False)

    sessions[session_id] = {
        "filepath":   filepath,
        "columns":    list(df.columns),
        "shape":      df.shape,
        "total_rows": len(df),
    }

    return jsonify({
        "session_id": session_id,
        "columns":    list(df.columns),
        "shape":      {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "total_rows": len(df),
        "preview":    df.head(5).fillna("").to_dict(orient="records"),
        "suggestions": suggest_protected_attrs(df),
        "dataset_id": dataset_id,
    })


# ─────────────────────────────────────────────────────────────────────────────
# DEMO DATA GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_hr():
    np.random.seed(42)
    n = 2000
    genders    = np.random.choice(["Male", "Female", "Non-binary"], n, p=[0.48, 0.48, 0.04])
    age_groups = np.random.choice(["18-30", "31-45", "46-60", "60+"], n, p=[0.3, 0.4, 0.2, 0.1])
    experience = np.random.randint(0, 20, n)
    education  = np.random.choice(["High School", "Bachelor", "Master", "PhD"], n,
                                   p=[0.2, 0.4, 0.3, 0.1])
    skills_score = np.random.randint(40, 100, n)
    merit = (experience * 2 + skills_score +
             np.where(education == "PhD", 20,
             np.where(education == "Master", 10,
             np.where(education == "Bachelor", 5, 0))))
    bias = (np.where(genders == "Female", -10, 0) +
            np.where(genders == "Non-binary", -15, 0) +
            np.where(age_groups == "46-60", -8, 0) +
            np.where(age_groups == "60+", -20, 0))
    race = np.random.choice(["White", "Black", "Hispanic", "Asian", "Other"],
                             n, p=[0.6, 0.13, 0.18, 0.06, 0.03])
    race_bias = np.where(race == "Black", -6, np.where(race == "Hispanic", -4, 0))
    score2 = merit + bias + race_bias + np.random.normal(0, 8, n)
    hired = (score2 > np.percentile(score2, 50)).astype(int)
    return pd.DataFrame({
        "gender": genders, "age_group": age_groups, "race": race,
        "experience_years": experience, "education": education,
        "skills_score": skills_score, "hired": hired,
    })


def generate_synthetic_compas():
    np.random.seed(123)
    n = 3000
    race = np.random.choice(["White", "Black", "Hispanic", "Other"], n, p=[0.4, 0.4, 0.15, 0.05])
    sex  = np.random.choice(["Male", "Female"], n, p=[0.8, 0.2])
    age  = np.random.randint(18, 70, n)
    priors = np.random.poisson(2, n)
    charge_degree = np.random.choice(["Felony", "Misdemeanor"], n, p=[0.4, 0.6])
    base_risk  = (priors * 0.1 + np.where(charge_degree == "Felony", 0.3, 0) +
                  np.where(age < 25, 0.2, 0))
    race_bias  = np.where(race == "Black", 0.25, np.where(race == "Hispanic", 0.1, 0))
    risk_score = base_risk + race_bias + np.random.normal(0, 0.1, n)
    recidivism = (risk_score > 0.5).astype(int)
    return pd.DataFrame({
        "race": race, "sex": sex, "age": age,
        "priors_count": priors, "charge_degree": charge_degree,
        "two_year_recid": recidivism,
    })


def generate_synthetic_adult():
    np.random.seed(456)
    n = 3000
    sex   = np.random.choice(["Male", "Female"], n, p=[0.67, 0.33])
    race  = np.random.choice(["White", "Black", "Asian", "Other"], n, p=[0.6, 0.2, 0.1, 0.1])
    age   = np.random.randint(18, 70, n)
    education = np.random.choice(
        ["HS-grad", "Some-college", "Bachelors", "Masters", "Doctorate"],
        n, p=[0.3, 0.25, 0.25, 0.15, 0.05])
    hours_per_week = np.random.randint(20, 80, n)
    edu_score = {"HS-grad": 0, "Some-college": 1, "Bachelors": 2, "Masters": 3, "Doctorate": 4}
    base = np.array([edu_score[e] for e in education]) * 0.15 + hours_per_week * 0.005 + age * 0.005
    bias = np.where(sex == "Female", -0.3, 0) + np.where(race == "Black", -0.2, 0)
    score = base + bias + np.random.normal(0, 0.2, n)
    income_label = (score > np.percentile(score, 75)).astype(int)
    return pd.DataFrame({
        "age": age, "sex": sex, "race": race,
        "education": education, "hours_per_week": hours_per_week,
        "income_label": income_label,
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": "1.1.0", "name": "FairMind AI"})


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 1: AUDIT HISTORY
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/audit-history", methods=["GET"])
def get_audit_history():
    """Return the last N audit snapshots for the timeline view."""
    return jsonify(make_serializable(list(reversed(audit_history))))


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 2: REGULATORY SCORECARD
# ─────────────────────────────────────────────────────────────────────────────

REGULATIONS = [
    {
        "id": "eu_ai_act",
        "name": "EU AI Act (2024)",
        "article": "Article 10 & Annex III",
        "threshold_metric": "dpd",
        "threshold_value": 0.10,
        "operator": "lt",  # metric must be < threshold to PASS
        "description": "High-risk AI systems must not produce discriminatory outcomes. DPD must stay below 0.10.",
        "consequence": "Fines up to €30M or 6% of global annual turnover.",
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689",
    },
    {
        "id": "eeoc_80",
        "name": "US EEOC 80% Rule",
        "article": "29 CFR §1607 — Uniform Guidelines",
        "threshold_metric": "di",
        "threshold_value": 0.80,
        "operator": "gt",  # metric must be > threshold to PASS
        "description": "Selection rate for any protected group must be ≥80% of the highest-selected group.",
        "consequence": "Adverse impact liability; EEOC enforcement action or private lawsuit.",
        "url": "https://www.eeoc.gov/laws/guidance/questions-and-answers-clarify-and-provide-common-interpretation-uniform-guidelines",
    },
    {
        "id": "gdpr_22",
        "name": "GDPR Article 22",
        "article": "Article 22 — Automated Decision-Making",
        "threshold_metric": "dpd",
        "threshold_value": 0.05,
        "operator": "lt",
        "description": "Automated decisions must not discriminate. Strong fairness standard applies to EU data subjects.",
        "consequence": "Fines up to €20M or 4% of global annual turnover.",
        "url": "https://gdpr-info.eu/art-22-gdpr/",
    },
    {
        "id": "india_it_act",
        "name": "India IT Act / DPDP 2023",
        "article": "Digital Personal Data Protection Act §§ 4-8",
        "threshold_metric": "dpd",
        "threshold_value": 0.10,
        "operator": "lt",
        "description": "Data fiduciaries must ensure automated processing does not result in discriminatory profiling.",
        "consequence": "Penalties up to ₹250 crore (~USD 30M) per violation.",
        "url": "https://www.meity.gov.in/data-protection-framework",
    },
    {
        "id": "canada_aida",
        "name": "Canada AIDA (Bill C-27)",
        "article": "Artificial Intelligence and Data Act — Part 3",
        "threshold_metric": "di",
        "threshold_value": 0.80,
        "operator": "gt",
        "description": "High-impact AI systems must identify and mitigate biased outputs harming individuals.",
        "consequence": "Administrative monetary penalties up to CAD $25M.",
        "url": "https://ised-isde.canada.ca/site/innovation-better-canada/en/artificial-intelligence-and-data-act",
    },
]


@app.route("/api/scorecard", methods=["POST"])
def regulatory_scorecard():
    """
    Feature 2: Generate a regulatory compliance scorecard.
    Expects the same audit results that /api/audit returns.
    """
    body = request.json
    session_id = body.get("session_id")

    if session_id not in sessions:
        return session_expired_error()

    if "last_audit" not in sessions[session_id]:
        return jsonify({"error": "No audit results found. Run audit first."}), 400

    results = sessions[session_id]["last_audit"]

    # Extract worst-case DPD and DI across all protected attributes
    per_attr = results.get("per_attribute", {})
    all_dpd, all_di = [], []
    for attr_data in per_attr.values():
        dpd = attr_data.get("demographic_parity_difference")
        di  = attr_data.get("disparate_impact_ratio")
        if dpd is not None:
            all_dpd.append(float(dpd))
        if di is not None:
            all_di.append(float(di))

    worst_dpd = max(all_dpd) if all_dpd else None
    worst_di  = min(all_di)  if all_di  else None

    scorecard = []
    for reg in REGULATIONS:
        metric = reg["threshold_metric"]
        thresh = reg["threshold_value"]
        op     = reg["operator"]

        # Get actual value
        if metric == "dpd":
            actual = worst_dpd
        else:
            actual = worst_di

        if actual is None:
            status = "UNKNOWN"
            verdict = "Metric not available for this dataset."
        elif op == "lt":
            status = "PASS" if actual < thresh else "FAIL"
            direction = "below" if op == "lt" else "above"
            verdict = (
                f"{'✅ Compliant' if status == 'PASS' else '❌ Non-compliant'}: "
                f"{'DPD' if metric == 'dpd' else 'DI'} = {actual:.4f} "
                f"({'below' if actual < thresh else 'above'} threshold of {thresh})."
            )
        else:  # gt
            status = "PASS" if actual > thresh else "FAIL"
            verdict = (
                f"{'✅ Compliant' if status == 'PASS' else '❌ Non-compliant'}: "
                f"{'DPD' if metric == 'dpd' else 'DI'} = {actual:.4f} "
                f"({'above' if actual > thresh else 'below'} minimum of {thresh})."
            )

        # Plain-English verdict for judges
        if status == "FAIL":
            plain_verdict = (
                f"This model would likely violate {reg['name']} {reg['article']} "
                f"in a regulated deployment. "
                f"{reg['consequence']}"
            )
        else:
            plain_verdict = (
                f"This model appears compliant with {reg['name']} {reg['article']} "
                f"based on current metrics. Continue monitoring."
            )

        scorecard.append({
            "regulation":    reg["name"],
            "article":       reg["article"],
            "status":        status,
            "metric_used":   metric.upper(),
            "actual_value":  round(actual, 4) if actual is not None else None,
            "threshold":     thresh,
            "operator":      op,
            "verdict":       verdict,
            "plain_verdict": plain_verdict,
            "consequence":   reg["consequence"],
            "description":   reg["description"],
            "url":           reg["url"],
        })

    n_pass = sum(1 for s in scorecard if s["status"] == "PASS")
    n_fail = sum(1 for s in scorecard if s["status"] == "FAIL")
    overall = "ALL CLEAR" if n_fail == 0 else ("HIGH RISK" if n_fail >= 3 else "PARTIAL RISK")

    return jsonify({
        "scorecard":      scorecard,
        "summary": {
            "total":    len(scorecard),
            "passing":  n_pass,
            "failing":  n_fail,
            "overall":  overall,
            "worst_dpd": round(worst_dpd, 4) if worst_dpd is not None else None,
            "worst_di":  round(worst_di,  4) if worst_di  is not None else None,
        }
    })


if __name__ == "__main__":
    print("\n" + "═"*55)
    print("  🧠 FairMind AI - Bias Detection Platform (v9)")
    print("  Running at: http://localhost:5000")
    print("  No paid APIs required.")
    print("═"*55 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
