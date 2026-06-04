# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ulaanbaatar fire risk prediction system for the Emergency Management Agency (ОБЕГ). Predicts monthly fire risk per khoroo (sub-district) using XGBoost, then renders a self-contained interactive HTML dashboard. The project is in the `model_ai/` subdirectory; all scripts use relative paths and must be run from `model_ai/`.

## Setup

```bash
pip install -r model_ai/requirements.txt
```

Dependencies: `pandas`, `numpy`, `xgboost`, `scikit-learn`, `openpyxl`, `pyarrow`.

## Pipeline (run in order from `model_ai/`)

```bash
python scripts/01_population.py       # Parse Excel → pop_district.csv, pop_khoroo.csv
python scripts/02_train_monthly.py    # Build panel, engineer features, train 3 XGBoost models (~1 min)
python scripts/03_dashboard_data.py   # Scored panel → dashboard_data.json
python scripts/04_build_dashboard.py  # Inject JSON into HTML template → outputs/ub_fire_risk_dashboard.html
```

The final `outputs/ub_fire_risk_dashboard.html` is fully self-contained — no server needed, open directly in a browser.

## Architecture

**Three separate XGBoost binary classifiers** — one per fire cause:
- `ilgal` (open flame) — moderate signal, behavior-dependent
- `tsakhilgaan` (electrical) — weakest spatial pattern
- `yandan` (chimney/flue) — strongest seasonal signal (peaks in winter)

**Panel structure:** 204 khoroos × 132 months (2015–2025) = ~26,928 rows. Missing khoroo-month combinations become negative examples, which naturally handles class imbalance alongside `scale_pos_weight`.

**Feature engineering** (in `02_train_monthly.py::add_features`): all features use only past data (no leakage):
- Lags: 1/2/3/12 months
- Rolling sums: 3/6/12 months (shifted by 1)
- Cumulative mean per khoroo
- Seasonal mean per khoroo-month pair (`seasmonth`)
- Time: sin/cos of month, year index, log population

**Train/test split:** 2015–2023 train, 2024–2025 test (temporal order preserved).

**Data flow:**
```
fires_clean.csv + pop_khoroo.csv → panel → panel_scored.parquet → dashboard_data.json → HTML
```

**Dashboard:** `dashboard_template.html` contains a vanilla JS/HTML app. `04_build_dashboard.py` injects the JSON blob by replacing the `__DATA__` placeholder, producing a single portable file.

## Key Files

| File | Role |
|------|------|
| `data/fires_clean.csv` | Source fire incident data (2014–2025, ~24,960 calls) |
| `data/Суурин_хүн_амын_тоо.xlsx` | NSO population by district/khoroo |
| `data/panel_scored.parquet` | Full panel with risk scores (output of step 2) |
| `data/dashboard_data.json` | 2025 risk scores formatted for the dashboard |
| `models/model_m_{cause}.json` | Saved XGBoost models (XGBoost native format) |
| `models/metrics_monthly.json` | ROC-AUC, PR-AUC, recall@K per cause |
| `scripts/_exp_weekly_*.py` | Experimental weekly-granularity variants — not part of the main pipeline |

## Model Performance (test: 2024–2025)

| Cause | ROC-AUC | PR-AUC |
|-------|---------|--------|
| Chimney/flue | 0.84 | 0.31 |
| Open flame | 0.65 | 0.47 |
| Electrical | 0.62 | 0.32 |

## Planned Improvements

Priority data additions to improve accuracy:
1. **Khoroo boundary GeoJSON** — unlocks ger density, OSM land-use features
2. **Weather data** — monthly temp/wind/humidity via Open-Meteo Archive API (free, no key)
3. **Socioeconomic data** — income/unemployment per khoroo from NSO
