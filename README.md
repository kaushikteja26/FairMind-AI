# 🧠 FairMind AI
### Bias Detection & Fairness Auditing Platform
**Google Solution Challenge 2026**

> The world's first open-source **intersectional bias detection** platform.
> Audit any AI dataset or model in under 3 minutes. 100% free, no paid APIs.

---

## ⚡ Unique Features

| Feature | FairMind AI | IBM AIF360 | Google What-If | Microsoft Fairlearn |
|---|---|---|---|---|
| Web UI | ✅ | ❌ CLI only | ❌ Notebook only | ❌ |
| Intersectional bias (multi-attr) | ✅ **★** | ❌ | ❌ | ❌ |
| Counterfactual fairness probe | ✅ **★** | ❌ | Partial | ❌ |
| Plain-English explanation | ✅ **★** | ❌ | ❌ | ❌ |
| One-click mitigation | ✅ | Partial | ❌ | Partial |
| PDF compliance report | ✅ | ❌ | ❌ | ❌ |
| **Regulatory Scorecard (5 laws)** | ✅ **★ NEW** | ❌ | ❌ | ❌ |
| **Audit History Timeline** | ✅ **★ NEW** | ❌ | ❌ | ❌ |
| No paid APIs | ✅ | ✅ | ✅ | ✅ |

---

## 🆕 New Features (v10)

### ⚖️ Regulatory Compliance Scorecard
After every audit, a colour-coded scorecard maps your results to **5 real-world AI regulations**:
- **EU AI Act (2024)** — Article 10 & Annex III
- **US EEOC 80% Rule** — 29 CFR §1607
- **GDPR Article 22** — Automated Decision-Making
- **India DPDP Act 2023** — §§ 4-8
- **Canada AIDA Bill C-27** — Part 3

Each regulation gets a ✅ PASS / ❌ FAIL verdict with:
- The exact metric value vs. the legal threshold
- Plain-English verdict: *"This model would violate EU AI Act Article 10 in a high-risk deployment. Fines up to €30M."*
- Direct link to the regulation text

### 📜 Audit History & Fairness Timeline
Every audit is automatically saved and plotted on a **longitudinal timeline**:
- Severity score trend across all runs
- DPD trend line (improving / stable / worsening)
- Colour-coded audit cards with timestamps, row counts, and domains
- Step 7 in the navigation bar

---

## 🚀 Run Locally (5 minutes)

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/fairmind-ai.git
cd fairmind-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python app.py

# 4. Open browser
# http://localhost:5000
```

---

## 📁 Project Structure

```
fairmind/
├── app.py                    # Flask backend + all API endpoints
├── requirements.txt
├── README.md
├── backend/
│   ├── bias_engine.py        # ★ Core bias detection (12+ metrics + intersectional)
│   ├── mitigation.py         # Reweighing + threshold adjustment algorithms
│   └── report_generator.py   # PDF compliance report generator
├── frontend/
│   └── templates/
│       └── index.html        # Full single-file React-free UI
└── data/
    ├── uploads/              # Uploaded datasets (auto-cleaned)
    └── reports/              # Generated PDF reports
```

---

## 🔬 How Intersectional Bias Detection Works

Standard tools check:
- Is the model biased against **women**? (single attribute)
- Is the model biased against **Black people**? (single attribute)

FairMind AI also checks:
- Is the model biased against **Black women**? (2-way intersection)
- Is the model biased against **young Black men**? (3-way intersection)
- Is the model biased against **Hispanic women over 45**? (3-way intersection)

This is what **real-world discrimination looks like** — and no other open-source tool catches it automatically.

---

## 📊 Fairness Metrics Computed

| Metric | Ideal | Legal Threshold |
|---|---|---|
| Demographic Parity Difference | 0.000 | < 0.10 |
| Disparate Impact Ratio | 1.000 | ≥ 0.80 (80% rule) |
| Equalized Odds Difference | 0.000 | < 0.10 |
| False Positive Rate Difference | 0.000 | < 0.10 |
| False Negative Rate Difference | 0.000 | < 0.10 |
| Individual Consistency Score | 1.000 | > 0.90 |
| Counterfactual Flip Rate | 0.000 | < 0.05 |

---

## 🌐 Deploy to GitHub Pages / Render / Railway

```bash
# Using gunicorn for production
gunicorn app:app --bind 0.0.0.0:$PORT

# Or using Railway (free tier):
# 1. Push to GitHub
# 2. Connect repo to railway.app
# 3. Set start command: gunicorn app:app
```

---

## 🛠️ Tech Stack (100% Free)

- **Backend**: Python + Flask + scikit-learn + pandas
- **Bias Engine**: Custom intersectional scanner + AIF360-compatible metrics
- **Frontend**: Pure HTML/CSS/JS + Chart.js (no build step needed)
- **Reports**: ReportLab (free PDF generation)
- **Hosting**: GitHub Pages / Railway / Render (all free tiers)

---

## 📄 License
MIT — free to use, modify, and deploy.

---

*Built for Google Solution Challenge 2026 | Team FairMind AI*

### Bias Detection & Fairness Auditing Platform
**Google Solution Challenge 2026**

> The world's first open-source **intersectional bias detection** platform.
> Audit any AI dataset or model in under 3 minutes. 100% free, no paid APIs.

---

## ⚡ Unique Features

| Feature | FairMind AI | IBM AIF360 | Google What-If | Microsoft Fairlearn |
|---|---|---|---|---|
| Web UI | ✅ | ❌ CLI only | ❌ Notebook only | ❌ |
| Intersectional bias (multi-attr) | ✅ **★** | ❌ | ❌ | ❌ |
| Counterfactual fairness probe | ✅ **★** | ❌ | Partial | ❌ |
| Plain-English explanation | ✅ **★** | ❌ | ❌ | ❌ |
| One-click mitigation | ✅ | Partial | ❌ | Partial |
| PDF compliance report | ✅ | ❌ | ❌ | ❌ |
| No paid APIs | ✅ | ✅ | ✅ | ✅ |

---

## 🚀 Run Locally (5 minutes)

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/fairmind-ai.git
cd fairmind-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python app.py

# 4. Open browser
# http://localhost:5000
```

---

## 📁 Project Structure

```
fairmind/
├── app.py                    # Flask backend + all API endpoints
├── requirements.txt
├── README.md
├── backend/
│   ├── bias_engine.py        # ★ Core bias detection (12+ metrics + intersectional)
│   ├── mitigation.py         # Reweighing + threshold adjustment algorithms
│   └── report_generator.py   # PDF compliance report generator
├── frontend/
│   └── templates/
│       └── index.html        # Full single-file React-free UI
└── data/
    ├── uploads/              # Uploaded datasets (auto-cleaned)
    └── reports/              # Generated PDF reports
```

---

## 🔬 How Intersectional Bias Detection Works

Standard tools check:
- Is the model biased against **women**? (single attribute)
- Is the model biased against **Black people**? (single attribute)

FairMind AI also checks:
- Is the model biased against **Black women**? (2-way intersection)
- Is the model biased against **young Black men**? (3-way intersection)
- Is the model biased against **Hispanic women over 45**? (3-way intersection)

This is what **real-world discrimination looks like** — and no other open-source tool catches it automatically.

---

## 📊 Fairness Metrics Computed

| Metric | Ideal | Legal Threshold |
|---|---|---|
| Demographic Parity Difference | 0.000 | < 0.10 |
| Disparate Impact Ratio | 1.000 | ≥ 0.80 (80% rule) |
| Equalized Odds Difference | 0.000 | < 0.10 |
| False Positive Rate Difference | 0.000 | < 0.10 |
| False Negative Rate Difference | 0.000 | < 0.10 |
| Individual Consistency Score | 1.000 | > 0.90 |
| Counterfactual Flip Rate | 0.000 | < 0.05 |

---

## 🌐 Deploy to GitHub Pages / Render / Railway

```bash
# Using gunicorn for production
gunicorn app:app --bind 0.0.0.0:$PORT

# Or using Railway (free tier):
# 1. Push to GitHub
# 2. Connect repo to railway.app
# 3. Set start command: gunicorn app:app
```

---

## 🛠️ Tech Stack (100% Free)

- **Backend**: Python + Flask + scikit-learn + pandas
- **Bias Engine**: Custom intersectional scanner + AIF360-compatible metrics
- **Frontend**: Pure HTML/CSS/JS + Chart.js (no build step needed)
- **Reports**: ReportLab (free PDF generation)
- **Hosting**: GitHub Pages / Railway / Render (all free tiers)

---

## 📄 License
MIT — free to use, modify, and deploy.

---

*Built for Google Solution Challenge 2026 | Team FairMind AI*
