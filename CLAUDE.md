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
python scripts/01_population.py          # Parse Excel → pop_district.csv, pop_khoroo.csv
python scripts/02_train_monthly.py       # Build panel, engineer features, train 3 XGBoost models (~1 min)
python scripts/03_dashboard_data.py      # Scored panel → dashboard_data.json (2025 historical)
python scripts/05_forecast_next_month.py # Predict next month → forecast_data.json (run monthly)
python scripts/04_build_dashboard.py     # Inject both JSONs → outputs/ub_fire_risk_dashboard.html
```

The final `outputs/ub_fire_risk_dashboard.html` is fully self-contained — no server needed, open directly in a browser.

**Monthly refresh:** only steps 05 → 04 need to be re-run each month to update the forecast.

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



Сайхан асуулт. Системийн одоогийн түвшин, ОБЕГ-н ажлын онцлогийг харгалзан бодит үнэ цэнтэй 3 давхарга санал
  болгоно.

  ---
  1. Яаралтай нэмж болох (хялбар, их үр дүн)

  📊 Сарын автомат тайлан (PDF)

  Сар бүр загвар ажиллаад дараах PDF гарна:
  - Өндөр эрсдэлт TOP-20 хороо
  - Өмнөх сартай харьцуулсан өөрчлөлт (↑↓)
  - Дүүрэг бүрийн хураангуй
  - Удирдлагад илгээх, архивд хадгалах

  Яагаад: Одоо алба хаагчид гараар тайлан бичдэг → автоматжуулна

  ---
  📈 Эрсдэлийн чиг хандлага (Trend)

  Хороо бүрд ↑ нэмэгдэж байна / ↓ буурч байна / → тогтвортой зэрэглэл харуулна. Өмнөх 3 сарын дундаж vs одоогийн
  харьцуулалт.

  Яагаад: "Энэ хороо өнгөрсөн сараас 15% нэмэгдсэн" гэдэг мэдээлэл илүү ухааллаг

  ---
  📱 Мобайл дасан зохицол

  Одоо 1400px өргөн. Алба хаагчид талбайд утас ашигладаг → responsive layout

  ---
  2. Дунд хугацаа (өндөр нөлөө)

  🏠 Гэр хорооллын нягтрал дата

  Хамгийн чухал feature — одоо байгаа загварт алга.

  OpenStreetMap-аас гэр/орон сууц харьцааг хороо бүрт тооцвол яндан болон ил галын загварын нарийвчлал мэдэгдэхүйц
  нэмэгдэнэ. УБ-ын гал түймрийн 70%+ гэр хороолол дээр гардаг.

  Нэмэх feature:
  - ger_ratio: гэрийн % (OpenStreetMap building tags)
  - building_density: 1 км²-т барилгын тоо

  ---
  🌤 Бодит цаг агаарын урьдчилсан мэдээ

  Одоо климатологийн дундаж ашигладаг. Open-Meteo-н forecast API (үнэгүй, key шаардлагагүй) ашиглаж ирэх 16 хоногийн
  бодит мэдээ авна → сарын таамаглал илүү нарийн болно.

  # Open-Meteo forecast URL (үнэгүй)
  https://api.open-meteo.com/v1/forecast?latitude=47.92&longitude=106.92&...

  ---
  📋 Сургалт/шалгалтын бүртгэл

  Дашбордод хялбар form нэмж:
  - Аль хороонд хэзээ сургалт явуулсан бүртгэх
  - Хамрагдсан өрх тоо
  - Дараагийн сургалтын огноо
  → Загвар дараагийн жилд "энэ хороо сургалт авсан" feature ашиглаж болно

  ---
  3. Стратегийн (game-changer)

  🔄 Feedback loop — Бодит гал + таамаглал харьцуулалт

  Загвар таамаглал хийсний дараа бодит гал гарсан эсэхийг харьцуулан:
  - "Таамаглаж байсан 30 хороогоос хэдэнд нь бодитоор гал гарсан бэ?"
  - Алба хаагчид загварт итгэх итгэлийг нэмэгдүүлнэ
  - Загварыг тасралтгүй сайжруулах боломж

  ---
  🤝 105 Дуудлагын системтэй холболт

  Одоо fires_clean.csv-г гараар шинэчилдэг. Хэрэв 105-н dispatch системтэй API холболт хийвэл:
  - Бодит цагийн дата → загвар автоматаар шинэчлэгдэнэ
  - Алба хаагчид гараар дата оруулах шаардлагагүй болно

  ---
  Тэргүүлэх дараалал

  ┌─────┬────────────────────────────────────┬───────────────┬────────────┐
  │  #  │              Юу хийх               │    Хугацаа    │   Нөлөө    │
  ├─────┼────────────────────────────────────┼───────────────┼────────────┤
  │ 1   │ Гэр хорооллын нягтрал              │ 1 долоо хоног │ ⭐⭐⭐⭐⭐ │
  ├─────┼────────────────────────────────────┼───────────────┼────────────┤
  │ 2   │ Сарын PDF тайлан                   │ 2-3 хоног     │ ⭐⭐⭐⭐   │
  ├─────┼────────────────────────────────────┼───────────────┼────────────┤
  │ 3   │ Бодит цаг агаарын урьдчилсан мэдээ │ 1 хоног       │ ⭐⭐⭐⭐   │
  ├─────┼────────────────────────────────────┼───────────────┼────────────┤
  │ 4   │ Trend харуулах                     │ 1 хоног       │ ⭐⭐⭐     │
  ├─────┼────────────────────────────────────┼───────────────┼────────────┤
  │ 5   │ Сургалт бүртгэл                    │ 3-4 хоног     │ ⭐⭐⭐     │
  ├─────┼────────────────────────────────────┼───────────────┼────────────┤
  │ 6   │ 105 холболт                        │ Урт хугацаа   │ ⭐⭐⭐⭐⭐ │
  └─────┴────────────────────────────────────┴───────────────┴────────────┘

  ---
  Яаж эхлэх вэ? Гэр хорооллын нягтрал + PDF тайлан хоёрыг эхлүүлвэл хамгийн хурдан мэдэгдэхүйц өөрчлөлт гарна. Аль
  нэгийг нь хийж өгөхүү?