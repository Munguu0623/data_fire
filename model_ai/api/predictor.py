"""
XGBoost загваруудыг ачаалж, /predict endpoint-д зориулан on-demand
feature тооцоолол хийж таамаглал гаргана.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.features import CAUSE_MN, TSAGAAN_SAR, WEATHER_FEATS, CALENDAR_FEATS, feats_for

_BASE = Path(__file__).parent.parent

models: dict[str, xgb.Booster] = {}
_panel: pd.DataFrame | None = None
_weather: pd.DataFrame | None = None


def _load_booster(path: Path) -> xgb.Booster:
    """Загвар JSON дахь attributes-ийг засаж Booster ачаална.

    XGBoost 3.2+ дээр scikit_learn attribute JSON string байх ёстой
    боловч зарим хадгалалтад nested object хэлбэрээр бичигддэг.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    attrs = data.get("learner", {}).get("attributes", {})
    for k, v in list(attrs.items()):
        if not isinstance(v, str):
            attrs[k] = json.dumps(v, ensure_ascii=False)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(data, tmp, ensure_ascii=False)
        tmp_path = tmp.name
    try:
        b = xgb.Booster()
        b.load_model(tmp_path)
    finally:
        os.unlink(tmp_path)
    return b


def startup() -> None:
    global models, _panel, _weather
    for key in CAUSE_MN:
        models[key] = _load_booster(_BASE / "models" / f"model_m_{key}.json")
    _panel  = pd.read_parquet(_BASE / "data" / "panel_scored.parquet")
    _weather = pd.read_csv(_BASE / "data" / "weather_monthly.csv")


def predict(requests: list[dict]) -> list[dict]:
    """
    requests: [{"district": str, "khoroo": int, "year": int, "month": int}]
    Буцаах: [{"district","khoroo","year","month","risk_ilgal","risk_tsakhilgaan","risk_yandan"}]
    """
    results = []
    for req in requests:
        row = _compute_features(req["district"], req["khoroo"], req["year"], req["month"])
        risks = {}
        for key in CAUSE_MN:
            X = pd.DataFrame([{f: row.get(f, 0.0) for f in feats_for(key)}])
            dm = xgb.DMatrix(X)
            risks[f"risk_{key}"] = round(float(models[key].predict(dm)[0]), 4)
        results.append({**req, **risks})
    return results


def _compute_features(district: str, khoroo: int, year: int, month: int) -> dict:
    """Нэг хороо-сарын feature векторыг тооцоолно."""
    kh = _panel[(_panel["district"] == district) & (_panel["khoroo"] == khoroo)].copy()
    target_t = year * 12 + month

    row: dict = {
        "month":            month,
        "year":             year,
        "month_sin":        float(np.sin(2 * np.pi * month / 12)),
        "month_cos":        float(np.cos(2 * np.pi * month / 12)),
        "year_idx":         year - 2015,
        "is_spring":        int(month in [3, 4, 5]),
        "is_heating_season": int(month in [10, 11, 12, 1, 2, 3]),
        "tsagaan_sar":      int(TSAGAAN_SAR.get(year, -1) == month),
    }

    if len(kh) > 0:
        last = kh.sort_values(["year", "month"]).iloc[-1]
        row["log_pop"]    = float(last.get("log_pop", 0))
        row["n_stations"] = int(last.get("n_stations", 1))
    else:
        row["log_pop"] = 0.0
        row["n_stations"] = 1

    # Цаг агаарын feature-ууд
    wx = _weather[(_weather["year"] == year) & (_weather["month"] == month)]
    if wx.empty:
        wx = _weather[(_weather["year"] == year - 1) & (_weather["month"] == month)]
    if not wx.empty:
        for col in wx.columns:
            if col not in ("year", "month"):
                row[col] = float(wx.iloc[0][col])

    # Шалтгаан бүрт lag / rolling / seasonal feature-ууд
    for key in CAUSE_MN:
        n_col    = f"n_{key}"
        fire_col = f"fire_{key}"
        if n_col not in kh.columns:
            _zero_lag_feats(row, key)
            continue

        kh["_t"] = kh["year"] * 12 + kh["month"]
        hist = kh[kh["_t"] < target_t].sort_values("_t")
        vals = hist[n_col].values.astype(float)

        for i, lag in enumerate([1, 2, 3, 12]):
            row[f"{key}_lag{lag}"] = float(vals[-lag]) if len(vals) >= lag else 0.0
        for win in [3, 6, 12]:
            row[f"{key}_roll{win}"] = float(vals[-win:].sum()) if len(vals) >= 1 else 0.0
        row[f"{key}_cummean"] = float(vals.mean()) if len(vals) > 0 else 0.0

        if fire_col in kh.columns:
            same = kh[(kh["month"] == month) & (kh["_t"] < target_t)]
            row[f"{key}_seasmonth"] = float(same[fire_col].mean()) if len(same) > 0 else 0.0
        else:
            row[f"{key}_seasmonth"] = 0.0

    return row


def _zero_lag_feats(row: dict, key: str) -> None:
    for lag in [1, 2, 3, 12]:
        row[f"{key}_lag{lag}"] = 0.0
    for win in [3, 6, 12]:
        row[f"{key}_roll{win}"] = 0.0
    row[f"{key}_cummean"]  = 0.0
    row[f"{key}_seasmonth"] = 0.0
