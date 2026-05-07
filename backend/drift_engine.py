"""
FairMind AI — Temporal Bias Drift Detection Engine
====================================================
★ NEW FEATURE: Paper-level / Google-level

Detects whether AI model bias is WORSENING, IMPROVING, or STABLE over time.
Generates AI-powered regulatory compliance reports via Google Gemini (free tier).

Why this matters:
  - AI bias is not static. Models trained on historical data degrade as
    real-world distributions shift. A model that was 'fair' in Q1 may be
    discriminating by Q4 — with no one noticing.
  - EU AI Act Article 9 explicitly requires continuous bias monitoring.
  - This is the ONLY free, no-code tool that implements this.

Free API: Google Gemini 1.5 Flash (1,500 req/day, no credit card needed)
Get key:  https://aistudio.google.com/app/apikey
"""

import json
import warnings
import urllib.request
import urllib.error

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Import from existing bias engine ─────────────────────────────────────────
from backend.bias_engine import (
    demographic_parity_difference,
    disparate_impact_ratio,
    severity_score,
)


# =============================================================================
# PERIOD SPLITTING
# =============================================================================

def split_into_periods(df: pd.DataFrame, time_col: str, n_periods: int = 5):
    """
    Split dataset into chronological time periods.
    Returns list of (label, sub_dataframe) tuples.
    Tries datetime parsing first, falls back to equal-size bins.
    """
    df = df.copy()

    # ── Try datetime parsing ──────────────────────────────────────────────────
    try:
        df["_ts"] = pd.to_datetime(df[time_col], errors="coerce")
        n_valid = df["_ts"].notna().sum()

        if n_valid >= max(20, len(df) * 0.5):
            df = df.dropna(subset=["_ts"]).sort_values("_ts")
            time_range_days = (df["_ts"].max() - df["_ts"].min()).days

            # Choose granularity based on time range
            if time_range_days > 365 * 2:
                df["_period_key"] = df["_ts"].dt.to_period("Q").astype(str)
            elif time_range_days > 180:
                df["_period_key"] = df["_ts"].dt.to_period("M").astype(str)
            elif time_range_days > 30:
                df["_period_key"] = df["_ts"].dt.to_period("W").astype(str)
            else:
                df["_period_key"] = df["_ts"].dt.to_period("D").astype(str)

            unique_keys = sorted(df["_period_key"].unique())

            # If too many periods, bin them
            if len(unique_keys) > n_periods:
                chunk = max(1, len(unique_keys) // n_periods)
                key_to_bin = {k: f"Period {i // chunk + 1}" for i, k in enumerate(unique_keys)}
                df["_period_key"] = df["_period_key"].map(key_to_bin)

            periods = []
            for key in sorted(df["_period_key"].unique()):
                sub = df[df["_period_key"] == key].copy()
                if len(sub) >= 10:
                    periods.append((str(key), sub))

            if len(periods) >= 2:
                return periods[:8]

    except Exception:
        pass

    # ── Fallback: numeric or ordinal column → equal-size bins ────────────────
    try:
        df = df.copy()
        df["_numeric"] = pd.to_numeric(df[time_col], errors="coerce")
        df = df.dropna(subset=["_numeric"]).sort_values("_numeric")
        q = min(n_periods, len(df) // 10)
        if q < 2:
            q = 2
        df["_bin"] = pd.qcut(df["_numeric"], q=q, duplicates="drop", labels=False)
        periods = []
        for b in sorted(df["_bin"].unique()):
            sub = df[df["_bin"] == b].copy()
            if len(sub) >= 10:
                lo = sub["_numeric"].min()
                hi = sub["_numeric"].max()
                periods.append((f"{time_col}: {lo:.0f}–{hi:.0f}", sub))
        if len(periods) >= 2:
            return periods[:8]
    except Exception:
        pass

    # ── Last resort: split by row position ───────────────────────────────────
    q = min(n_periods, len(df) // 10)
    if q < 2:
        return []
    chunk = len(df) // q
    return [(f"Segment {i + 1}", df.iloc[i * chunk:(i + 1) * chunk].copy())
            for i in range(q)]


# =============================================================================
# CORE DRIFT AUDIT
# =============================================================================

def run_drift_audit(
    df: pd.DataFrame,
    protected_attr: str,
    y_pred_col: str,
    time_col: str,
    n_periods: int = 5,
    pos_label=1,
) -> dict:
    """
    Main drift analysis function.

    Runs all bias metrics across each time period and returns:
      - timeline         : per-period bias metrics
      - trend            : WORSENING / IMPROVING / STABLE
      - alerts           : list of detected anomalies
      - regulatory_flags : EU AI Act / GDPR / EEOC compliance flags
      - summary stats    : avg, min, max DPD across periods
    """
    # ── Validation ────────────────────────────────────────────────────────────
    missing = [c for c in [protected_attr, y_pred_col, time_col] if c not in df.columns]
    if missing:
        return {"error": f"Columns not found: {missing}"}

    period_dfs = split_into_periods(df, time_col, n_periods)
    if len(period_dfs) < 2:
        return {
            "error": (
                "Could not split data into at least 2 time periods. "
                "Make sure the time column has varied values and at least 20 rows per period."
            )
        }

    # ── Per-period metrics ────────────────────────────────────────────────────
    timeline = []
    for label, period_df in period_dfs:
        dpd, rates = demographic_parity_difference(
            period_df[y_pred_col], period_df[protected_attr], pos_label
        )
        di, _ = disparate_impact_ratio(
            period_df[y_pred_col], period_df[protected_attr], pos_label
        )
        sev = severity_score(float(dpd), float(di), 0.0)

        group_rates = {str(k): round(float(v), 4) for k, v in rates.items()}
        legal_ok = bool(float(di) >= 0.8 and float(dpd) < 0.1)

        timeline.append({
            "period_label": label,
            "n_samples": int(len(period_df)),
            "dpd": round(float(dpd), 4),
            "di": round(float(di), 4),
            "severity_score": round(float(sev), 1),
            "group_rates": group_rates,
            "legal_compliant": legal_ok,
        })

    # ── Trend classification (linear regression slope on DPD) ─────────────────
    dpds = [t["dpd"] for t in timeline]
    dis  = [t["di"]  for t in timeline]
    sevs = [t["severity_score"] for t in timeline]

    x = np.arange(len(dpds), dtype=float)
    slope = float(np.polyfit(x, dpds, 1)[0]) if len(dpds) > 1 else 0.0

    if abs(slope) < 0.005:
        trend, trend_emoji, trend_color = "STABLE",    "➡️", "blue"
    elif slope > 0:
        trend, trend_emoji, trend_color = "WORSENING", "📈", "red"
    else:
        trend, trend_emoji, trend_color = "IMPROVING", "📉", "green"

    # ── Alert detection ───────────────────────────────────────────────────────
    alerts = []

    # Sudden spike between consecutive periods
    for i in range(1, len(dpds)):
        delta = dpds[i] - dpds[i - 1]
        if abs(delta) > 0.08:
            direction = "surge" if delta > 0 else "drop"
            sev_label = "CRITICAL" if abs(delta) > 0.20 else "HIGH" if abs(delta) > 0.12 else "MEDIUM"
            alerts.append({
                "type": "SUDDEN_CHANGE",
                "severity": sev_label,
                "period": timeline[i]["period_label"],
                "message": (
                    f"Bias {direction} of {abs(delta):.3f} detected between "
                    f"'{timeline[i-1]['period_label']}' and '{timeline[i]['period_label']}'. "
                    f"This rapid change warrants immediate investigation."
                ),
            })

    # Legal threshold violations
    for t in timeline:
        if not t["legal_compliant"]:
            alerts.append({
                "type": "LEGAL_VIOLATION",
                "severity": "CRITICAL",
                "period": t["period_label"],
                "message": (
                    f"[{t['period_label']}] Disparate Impact = {t['di']:.3f} (threshold: ≥0.80). "
                    f"DPD = {t['dpd']:.3f} (threshold: <0.10). "
                    f"This period fails EEOC 80% Rule and EU AI Act standards."
                ),
            })

    # Progressive degradation (every period worse than last)
    if trend == "WORSENING" and len(dpds) >= 3:
        if all(dpds[i] > dpds[i - 1] for i in range(1, len(dpds))):
            alerts.append({
                "type": "PROGRESSIVE_DEGRADATION",
                "severity": "CRITICAL",
                "period": "All periods",
                "message": (
                    f"Bias has increased every single period without exception "
                    f"(DPD: {dpds[0]:.3f} → {dpds[-1]:.3f}). "
                    f"This is systematic model degradation — likely caused by data drift or "
                    f"a feedback loop. Immediate retraining required."
                ),
            })

    # ── Regulatory flags ──────────────────────────────────────────────────────
    n_noncompliant = sum(1 for t in timeline if not t["legal_compliant"])
    regulatory_flags = {
        "eu_ai_act_article9":             trend == "WORSENING",
        "gdpr_article22":                 any(t["dpd"] > 0.20 for t in timeline),
        "eeoc_80_rule_violations":        n_noncompliant,
        "continuous_monitoring_required": (
            trend in ("WORSENING", "STABLE") and n_noncompliant > 0
        ),
    }

    # ── Summary ───────────────────────────────────────────────────────────────
    worst_period = timeline[dpds.index(max(dpds))]
    best_period  = timeline[dpds.index(min(dpds))]
    total_change = dpds[-1] - dpds[0]

    # ── Detect if we used real datetime or batch segments (EC-4) ─────────────
    # Check if time_col can be parsed as datetime for at least 50% of rows
    time_mode = "BATCH_SEGMENT"
    try:
        ts_test = pd.to_datetime(df[time_col], errors="coerce")
        if ts_test.notna().sum() >= max(20, len(df) * 0.5):
            time_mode = "TRUE_TEMPORAL"
    except Exception:
        pass

    return {
        "protected_attribute": protected_attr,
        "prediction_column":   y_pred_col,
        "time_column":         time_col,
        "time_mode":           time_mode,          # EC-4
        "n_periods":           len(timeline),
        "timeline":            timeline,
        "trend":               trend,
        "trend_emoji":         trend_emoji,
        "trend_color":         trend_color,
        "trend_slope":         round(slope, 5),
        "total_dpd_change":    round(total_change, 4),
        "worst_period":        worst_period,
        "best_period":         best_period,
        "alerts":              alerts,
        "regulatory_flags":    regulatory_flags,
        "summary": {
            "avg_dpd":      round(float(np.mean(dpds)), 4),
            "max_dpd":      round(float(max(dpds)),     4),
            "min_dpd":      round(float(min(dpds)),     4),
            "avg_di":       round(float(np.mean(dis)),  4),
            "avg_severity": round(float(np.mean(sevs)), 1),
        },
    }


# =============================================================================
# GEMINI API INTEGRATION
# =============================================================================

def build_gemini_prompt(drift_result: dict, domain: str = "general") -> str:
    """
    Build a structured prompt that instructs Gemini to write a
    regulatory-quality bias drift analysis report in plain English.
    """
    tl = drift_result.get("timeline", [])
    timeline_text = "\n".join(
        f"  {t['period_label']}: DPD={t['dpd']:.4f} | "
        f"DisparateImpact={t['di']:.4f} | "
        f"Severity={t['severity_score']}/100 | "
        f"Legal={'PASS' if t['legal_compliant'] else 'FAIL'} | "
        f"n={t['n_samples']}"
        for t in tl
    )

    alerts_text = "\n".join(
        f"  [{a['severity']}] {a['message']}"
        for a in drift_result.get("alerts", [])
    ) or "  No critical alerts detected."

    reg = drift_result.get("regulatory_flags", {})
    summary = drift_result.get("summary", {})
    worst = drift_result.get("worst_period", {})
    best  = drift_result.get("best_period",  {})

    return f"""You are a senior AI ethics and regulatory compliance auditor.
A company has submitted their AI model's bias drift data for review.
Write a professional regulatory report based on the data below.

═══════════════════════════════════════
BIAS DRIFT ANALYSIS — SUBMITTED DATA
═══════════════════════════════════════
Protected Attribute : {drift_result.get('protected_attribute', 'unknown')}
Prediction Column   : {drift_result.get('prediction_column', 'unknown')}
Domain              : {domain}
Number of Periods   : {drift_result.get('n_periods', 0)}

TREND: {drift_result.get('trend', 'UNKNOWN')} {drift_result.get('trend_emoji', '')}
  Slope (DPD per period): {drift_result.get('trend_slope', 0):+.5f}
  Total DPD Change      : {drift_result.get('total_dpd_change', 0):+.4f}
  Average DPD           : {summary.get('avg_dpd', 0):.4f}
  Best period  : {best.get('period_label', 'N/A')} (DPD={best.get('dpd', 0):.4f})
  Worst period : {worst.get('period_label', 'N/A')} (DPD={worst.get('dpd', 0):.4f})

TIMELINE:
{timeline_text}

ALERTS DETECTED:
{alerts_text}

REGULATORY FLAGS:
  EU AI Act Art. 9 (continuous monitoring violation) : {'YES ⚠️' if reg.get('eu_ai_act_article9') else 'No'}
  GDPR Article 22 (automated decision concern)      : {'YES ⚠️' if reg.get('gdpr_article22') else 'No'}
  EEOC 80% Rule violations                          : {reg.get('eeoc_80_rule_violations', 0)} period(s)
  Continuous monitoring required                    : {'YES' if reg.get('continuous_monitoring_required') else 'No'}
═══════════════════════════════════════

Write a structured report with EXACTLY these 5 sections using these headers:

## 1. EXECUTIVE SUMMARY
(2–3 sentences. State what happened to bias over time in plain English.
 No jargon. Start with the most important finding.)

## 2. TREND ANALYSIS
(Explain what the numbers actually mean. Is this model becoming more discriminatory?
 Reference specific periods and DPD values. Explain the slope and what it implies.)

## 3. ROOT CAUSE HYPOTHESES
(List 2–3 likely explanations for this trend. Consider: data drift,
 feedback loops, population shifts, seasonal effects, model degradation.
 Be specific to the domain: {domain}.)

## 4. LEGAL & COMPLIANCE ASSESSMENT
(State which regulations are triggered and what the consequences could be.
 Cite the specific DPD and DI values that trigger each regulation.
 Be direct — if this is a legal problem, say so clearly.)

## 5. URGENT RECOMMENDATIONS
(List 3–4 concrete, prioritized actions the organization must take.
 Be specific — not "monitor the model" but "re-audit every quarter and
 halt deployment if DPD exceeds 0.15". Include timelines where possible.)

Write for a compliance officer — not a data scientist.
Use plain English. Be direct about severity. Cite the actual numbers."""


def call_gemini_api(prompt: str, api_key: str) -> str:
    """
    Smart AI report generator. Auto-detects key type by prefix:
      gsk_...  -> Groq        (free, works in India, 14,400 req/day)
      sk-or-.. -> OpenRouter  (free tier, works in India)
      AIza...  -> Gemini      (Google AI Studio)

    Get free Groq key:       https://console.groq.com
    Get free OpenRouter key: https://openrouter.ai/keys
    """
    api_key = api_key.strip()

    # Detect provider by key prefix
    is_groq        = api_key.startswith("gsk_")
    is_openrouter  = api_key.startswith("sk-or-")
    is_gemini      = api_key.startswith("AIza")

    if not any([is_groq, is_openrouter, is_gemini]):
        return (
            "Unrecognised API key format.\n\n"
            "FairMind supports these FREE providers:\n"
            "- Groq key       starts with: gsk_    -> https://console.groq.com\n"
            "- OpenRouter key starts with: sk-or-  -> https://openrouter.ai/keys\n"
            "- Gemini key     starts with: AIza    -> https://aistudio.google.com/app/api-keys\n\n"
            "Make sure you copied the full key without extra spaces."
        )

    def _call_openai_compat(url, key, model, prompt_text, extra_headers=None):
        """Call any OpenAI-compatible endpoint."""
        payload = json.dumps({
            "model":       model,
            "messages":    [{"role": "user", "content": prompt_text}],
            "temperature": 0.35,
            "max_tokens":  1400,
        }).encode("utf-8")
        headers = {
            "Content-Type":  "application/json",
            "Authorization": "Bearer " + key,
        }
        if extra_headers:
            headers.update(extra_headers)
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=40) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"]

    # ── GROQ ──────────────────────────────────────────────────────────────────
    if is_groq:
        groq_models = [
            "llama-3.3-70b-versatile",
            "llama3-70b-8192",
            "gemma2-9b-it",
            "llama3-8b-8192",
        ]
        last_err = ""
        for attempt in range(2):          # retry once on 503
            for model in groq_models:
                try:
                    text = _call_openai_compat(
                        "https://api.groq.com/openai/v1/chat/completions",
                        api_key, model, prompt
                    )
                    return "[Model: " + model + " via Groq]\n\n" + text
                except urllib.error.HTTPError as e:
                    body = e.read().decode("utf-8", errors="replace")
                    try:
                        msg = json.loads(body).get("error", {}).get("message", body[:150])
                    except Exception:
                        msg = body[:150]
                    last_err = "Groq " + str(e.code) + " (" + model + "): " + msg
                    if e.code == 401:
                        return (
                            "Invalid Groq API key (401).\n\n"
                            "Please check your key at https://console.groq.com\n"
                            "The key should start with gsk_ and be copied in full."
                        )
                    if e.code == 429:
                        import time; time.sleep(2)
                    continue
                except Exception as e:
                    last_err = "Groq error (" + model + "): " + str(e)
                    continue
        return (
            "All Groq models temporarily unavailable.\n\n"
            "Last error: " + last_err + "\n\n"
            "Groq's free servers get busy — this is temporary. Options:\n"
            "1. Wait 1-2 minutes and try again\n"
            "2. Get a FREE OpenRouter key (also works in India):\n"
            "   -> https://openrouter.ai/keys  (sign in with Google)\n"
            "   -> Key starts with sk-or-\n"
            "   -> Paste that key here instead\n"
            "3. The drift analysis above is complete without the AI report."
        )

    # ── OPENROUTER ────────────────────────────────────────────────────────────
    if is_openrouter:
        # ── Step 1: Fetch live free model list from OpenRouter ────────────────
        # This avoids hardcoding models that may be removed at any time
        live_free_models = []
        try:
            models_req = urllib.request.Request(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": "Bearer " + api_key},
                method="GET"
            )
            with urllib.request.urlopen(models_req, timeout=10) as resp:
                models_data = json.loads(resp.read().decode("utf-8"))
            for m in models_data.get("data", []):
                pricing = m.get("pricing", {})
                try:
                    p_cost = float(pricing.get("prompt", 1))
                    c_cost = float(pricing.get("completion", 1))
                    if p_cost == 0 and c_cost == 0:
                        live_free_models.append(m["id"])
                except Exception:
                    continue
        except Exception:
            pass

        # ── Step 2: Preferred models first, then any other free model ─────────
        preferred = [
            "meta-llama/llama-3.3-70b-instruct:free",
            "meta-llama/llama-3.1-8b-instruct:free",
            "mistralai/mistral-7b-instruct:free",
            "deepseek/deepseek-r1:free",
            "deepseek/deepseek-chat:free",
            "microsoft/phi-3-mini-128k-instruct:free",
        ]
        # Put preferred models first if they are in the live list,
        # then append any remaining live free models
        if live_free_models:
            ordered = [m for m in preferred if m in live_free_models]
            ordered += [m for m in live_free_models if m not in ordered]
            or_models = ordered[:6]   # try up to 6
        else:
            # Fallback: use preferred list as-is if we couldn't fetch live list
            or_models = preferred

        # ── Step 3: Try each model in order ───────────────────────────────────
        last_err = ""
        for model in or_models:
            try:
                text = _call_openai_compat(
                    "https://openrouter.ai/api/v1/chat/completions",
                    api_key, model, prompt,
                    extra_headers={"HTTP-Referer": "https://fairmind.ai"}
                )
                return "[Model: " + model + " via OpenRouter]\n\n" + text
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                try:
                    msg = json.loads(body).get("error", {}).get("message", body[:150])
                except Exception:
                    msg = body[:150]
                last_err = "OpenRouter " + str(e.code) + " (" + model + "): " + msg
                if e.code == 401:
                    return "Invalid OpenRouter key (401). Check https://openrouter.ai/keys"
                continue
            except Exception as ex:
                last_err = "OpenRouter error (" + model + "): " + str(ex)
                continue
        return "OpenRouter failed after trying " + str(len(or_models)) + " models. Last error: " + last_err

    # ── GEMINI ────────────────────────────────────────────────────────────────
    if is_gemini:
        gemini_models = [
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash",
            "gemini-1.0-pro",
        ]
        gemini_payload_obj = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.35, "maxOutputTokens": 1400, "topP": 0.85},
        }
        last_err = ""
        for model in gemini_models:
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                + model + ":generateContent?key=" + api_key
            )
            payload = json.dumps(gemini_payload_obj).encode("utf-8")
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"}, method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=40) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                return "[Model: " + model + " via Gemini]\n\n" + text
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                try:
                    msg = json.loads(body).get("error", {}).get("message", body[:150])
                except Exception:
                    msg = body[:150]
                last_err = "Gemini " + str(e.code) + " (" + model + "): " + msg
                if e.code == 401:
                    return "Invalid Gemini API key. Get one free at https://aistudio.google.com/app/api-keys"
                continue
            except Exception as e:
                last_err = str(e)
                continue
        return (
            "Gemini quota exceeded or unavailable.\n"
            "Last error: " + last_err + "\n\n"
            "Try a free Groq key instead: https://console.groq.com (key starts with gsk_)"
        )

    return "Unknown error — no provider matched."

