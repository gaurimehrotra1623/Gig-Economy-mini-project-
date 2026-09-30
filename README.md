# Gig-Economy Income Volatility & Credit Access

Independent research project exploring whether **decomposing gig-worker income volatility
into a predictable (seasonal) component and a true-risk (residual) component** improves
credit-risk assessment, compared to models that rely on average income or raw variance alone.

Traditional credit scoring assumes stable salaried income. Gig workers' income swings
week to week — but a lot of that swing is structurally predictable (festivals, monsoon,
weekly demand cycles), not genuinely risky. This project tests whether separating the two
produces a better risk signal.

## Hypothesis

Models using **decomposed residual volatility** (via STL — Seasonal-Trend decomposition
using Loess) as a risk feature will show better discriminative power (e.g. higher AUC-ROC)
than baseline models using only average income and raw income variance.

## Repo Structure

```
gig_credit_risk/
├── src/
│   ├── simulate_income.py   # Phase 2 — synthetic weekly income generator,
│   │                          calibrated against real published volatility stats
│   └── decompose.py         # Phase 4 — STL decomposition + feature engineering
│                               (trend, seasonal amplitude, residual std/CV)
├── notebooks/
│   └── STL_Gig_Income_Mini_Project.ipynb   # standalone Colab demo of STL on one worker
├── data/                    # generated CSVs (simulated, not real)
├── outputs/                 # plots, one-pager, model results
├── requirements.txt
└── README.md
```

## Why Simulated Data

Real individual-level gig-income + repayment data isn't publicly available — it's held by
fintechs like Karmalife, KreditBee, and Jai Kisan. So this project uses a **synthetic income
simulator calibrated against real published aggregate statistics**:

- JPMorgan Chase Institute — platform-worker income volatility benchmarks
- Fairwork India Ratings 2024 — platform earnings patterns
- NITI Aayog — *India's Booming Gig and Platform Economy* (2022)

This is stated explicitly rather than implied — the simulation reproduces realistic
*patterns*, not real individuals' data.

## Status / Roadmap

| Phase | Status |
|---|---|
| 1. Literature review | Done |
| 2. Income simulation | Done — `simulate_income.py` |
| 3. Risk-label proxy | Not started |
| 4. STL feature engineering | Done — `decompose.py` |
| 5. Modeling (baseline vs. decomposed) | Not started |
| 6. Evaluation | Not started |
| 7. Writeup | Not started |

## Key Early Result

Running the decomposition on three simulated worker archetypes (steady / seasonal /
volatile) shows the core hypothesis holding up even at this stage:

| Archetype | Raw CV | Residual CV |
|---|---|---|
| Steady | 0.078 | 0.022 |
| Seasonal | 0.112 | 0.023 |
| Volatile | 0.162 | 0.042 |

`raw_cv` alone would rank "seasonal" workers as riskier than "steady" ones. `residual_cv`
(the true-risk signal after removing trend/seasonality) shows they're nearly identical —
while "volatile" workers, whose swings aren't explained by the calendar, stand apart
clearly. This is the central distinction the project is testing.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
mkdir -p data outputs
```

## Run

```bash
python src/simulate_income.py    # generates data/simulated_income.csv
python src/decompose.py          # generates data/worker_features.csv
```

Or open `notebooks/STL_Gig_Income_Mini_Project.ipynb` in Google Colab for a standalone,
visual walkthrough of the decomposition on a single worker.

## Known Limitations (tracked honestly, not hidden)

- **STL needs ~2 full seasonal cycles** of history to work reliably (104 weeks for an
  annual cycle on weekly data). Real gig workers often don't have that much tenure on one
  platform.
- **STL's LOESS smoothing isn't strictly causal** — estimates for a given week can be
  influenced by later data. Fine for retrospective analysis; a live scoring system would
  need a causal/online variant to avoid leakage.
- **Single fixed seasonal period assumption** — real income likely has overlapping weekly +
  monthly + annual cycles; vanilla STL only models one (MSTL handles multiple, not yet used
  here).
- No real repayment/default data exists to validate against — the risk-label proxy (Phase 3)
  will itself be an assumption, and will be documented as such.

## References

- Oyeyemi, O. O. — *Credit Access in the Gig Economy: Rethinking Creditworthiness in a
  Post-Employment Financial System* — https://ssrn.com/abstract=5336043
- Anekadhana, N. (2025) — *The Labor Credit Scoring Model for the Gig Economy in Developing
  Countries* (preprint) — https://doi.org/10.21203/rs.3.rs-7198891/v1
- NITI Aayog (2022) — *India's Booming Gig and Platform Economy*
- Fairwork India Ratings 2024
- JPMorgan Chase Institute — platform income volatility research
- CGAP — *Gig Platforms and Financial Inclusion*

---
*Independent research project — Gauri Mehrotra, BTech CS + AI (Finance Minor), Rishihood
University / Newton School of Technology. Built ahead of MBA applications.*