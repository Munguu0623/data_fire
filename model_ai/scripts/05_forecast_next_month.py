"""
АЛХАМ 5 — Дараагийн сарын урьдчилсан таамаглал
================================================
Өнөөдрийн огноог үндэслэн дараагийн сарын хороо бүрийн
гал түймрийн эрсдэлийг 3 загвараар тооцоолно.

Цаг агаарын прогноз:
  - тухайн сарын бодит дата байвал ашиглана
  - байхгүй бол климатологийн дундаж (2014–2025 ижил сарын дундаж)

Lag feature-ууд:
  - lag12  : өмнөх жилийн ижил сарын бодит дата
  - seasmonth : түүхэн ижил сарын дундаж (хамгийн найдвартай)
  - lag1/2/3, roll3/6 : тухайн жилд дата байхгүй тул 0
  - roll12  : өнгөрсөн жилийн 12 сарын нийлбэр

Гаралт: data/forecast_data.json
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import xgboost as xgb
import json
from datetime import date

from core.features import CAUSE_MN, TSAGAAN_SAR, WEATHER_FEATS, feats_for


def prev_ym(yr, mo, k):
    """(yr, mo)-с k сар өмнөх (year, month) буцаана."""
    total = yr * 12 + (mo - 1) - k
    return total // 12, (total % 12) + 1


def get_n(hist_k, key, yr, mo):
    """Хороо K-н n_{key} утгыг (yr, mo)-с хайна; олдохгүй бол 0."""
    row = hist_k[(hist_k['year'] == yr) & (hist_k['month'] == mo)]
    return int(row[f'n_{key}'].values[0]) if not row.empty else 0


def build_forecast_rows(panel, khoroos, weather, stations_map,
                        target_year, target_month, ger_map=None, mean_ger=0.5):
    """Хороо бүрт нэг таамаглалын мөр үүсгэнэ."""

    # Цаг агаарын дата: бодит байвал ашиглана, эс бөгөөс климатологийн дундаж
    wx_clim = (weather[weather['month'] == target_month]
               .drop(columns=['year', 'month']).mean())
    wx_actual = weather[
        (weather['year'] == target_year) & (weather['month'] == target_month)
    ]
    wx_row = (wx_actual.drop(columns=['year', 'month']).iloc[0]
              if not wx_actual.empty else wx_clim)

    all_wx_cols = list(wx_clim.index)
    rows = []

    for _, kh in khoroos.iterrows():
        district = kh['district']
        khoroo   = kh['khoroo']
        pop      = kh.get('pop', 0)

        hist_k = panel[(panel['district'] == district) & (panel['khoroo'] == khoroo)].copy()

        r = {
            'district':          district,
            'khoroo':            khoroo,
            'pop':               int(pop) if pd.notna(pop) else 0,
            'log_pop':           float(np.log1p(pop if pd.notna(pop) else 0)),
            'year_idx':          target_year - 2015,
            'month':             target_month,
            'month_sin':         float(np.sin(2 * np.pi * target_month / 12)),
            'month_cos':         float(np.cos(2 * np.pi * target_month / 12)),
            'is_spring':         int(target_month in [3, 4, 5]),
            'is_heating_season': int(target_month in [10, 11, 12, 1, 2, 3]),
            'tsagaan_sar':       int(TSAGAAN_SAR.get(target_year, -1) == target_month),
            'n_stations':        int(stations_map.get(district, 1)),
            'ger_ratio':         float((ger_map or {}).get(district, mean_ger)),
        }

        for col in all_wx_cols:
            r[col] = float(wx_row[col])

        for key in CAUSE_MN:
            # Lag 1/2/3/12
            for lag in [1, 2, 3, 12]:
                py, pm = prev_ym(target_year, target_month, lag)
                r[f'{key}_lag{lag}'] = get_n(hist_k, key, py, pm)

            # Rolling sum: lag1..lag_n нийлбэр
            r[f'{key}_roll3']  = sum(get_n(hist_k, key, *prev_ym(target_year, target_month, k))
                                     for k in range(1, 4))
            r[f'{key}_roll6']  = sum(get_n(hist_k, key, *prev_ym(target_year, target_month, k))
                                     for k in range(1, 7))
            r[f'{key}_roll12'] = sum(get_n(hist_k, key, *prev_ym(target_year, target_month, k))
                                     for k in range(1, 13))

            # Cummean: өнгөрсөн бүх сарын дундаж (lag1 = shift(1) → excludes current)
            all_n = hist_k[f'n_{key}'].values
            r[f'{key}_cummean'] = float(np.mean(all_n)) if len(all_n) > 0 else 0.0

            # Seasmonth: ижил сарын түүхэн дундаж (хамгийн найдвартай feature)
            same_m = hist_k[hist_k['month'] == target_month][f'fire_{key}'].values
            r[f'{key}_seasmonth'] = float(np.mean(same_m)) if len(same_m) > 0 else 0.0

        rows.append(r)

    return pd.DataFrame(rows)


def main():
    today = date.today()
    if today.month == 12:
        target_year, target_month = today.year + 1, 1
    else:
        target_year, target_month = today.year, today.month + 1

    MN_MONTHS = ['1-р','2-р','3-р','4-р','5-р','6-р','7-р','8-р','9-р','10-р','11-р','12-р']
    print(f'Таамаглалын хугацаа: {target_year} оны {MN_MONTHS[target_month-1]} сар')

    panel    = pd.read_parquet('./data/panel_scored.parquet')
    pop_kh   = pd.read_csv('./data/pop_khoroo.csv')
    khoroos  = pop_kh[['district', 'khoroo', 'pop']].drop_duplicates()
    weather  = pd.read_csv('./data/weather_monthly.csv')
    stations = pd.read_csv('./data/stations_district.csv')[['district', 'n_stations']]
    stations_map = dict(zip(stations['district'], stations['n_stations']))

    ger = pd.read_csv('./data/ger_density.csv')[['district', 'ger_ratio']]
    ger_map = dict(zip(ger['district'], ger['ger_ratio']))
    mean_ger = ger['ger_ratio'].mean()

    forecast = build_forecast_rows(panel, khoroos, weather, stations_map,
                                   target_year, target_month, ger_map, mean_ger)

    print('Загварууд ачааллаж байна...')
    for key in CAUSE_MN:
        mdl = xgb.XGBClassifier()
        mdl.load_model(f'./models/model_m_{key}.json')
        feats = feats_for(key)
        forecast[f'risk_{key}'] = mdl.predict_proba(forecast[feats])[:, 1]

    records = []
    for _, r in forecast.iterrows():
        records.append({
            'district':         r['district'],
            'khoroo':           int(r['khoroo']),
            'pop':              r['pop'],
            'risk_ilgal':       round(float(r['risk_ilgal']),       4),
            'risk_tsakhilgaan': round(float(r['risk_tsakhilgaan']), 4),
            'risk_yandan':      round(float(r['risk_yandan']),      4),
        })

    with open('./models/metrics_monthly.json') as f:
        metrics = json.load(f)

    out = {
        'records':   records,
        'districts': sorted(forecast['district'].unique().tolist()),
        'metrics':   metrics['metrics'],
        'meta': {
            'n_khoroo':       len(records),
            'forecast_year':  target_year,
            'forecast_month': target_month,
            'generated_date': str(today),
            'is_forecast':    True,
        },
    }

    with open('./data/forecast_data.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False)

    print(f'\n✓ forecast_data.json — {len(records)} хороо')

    for label, key in [('Ил гал', 'ilgal'), ('Яндан/цонолт', 'yandan'), ('Цахилгаан', 'tsakhilgaan')]:
        top5 = sorted(records, key=lambda r: r[f'risk_{key}'], reverse=True)[:5]
        print(f'\n{label} — дээд 5 хороо:')
        for r in top5:
            print(f"  {r['district']} {r['khoroo']}-р хороо: {r[f'risk_{key}']*100:.0f}%")


if __name__ == '__main__':
    main()
