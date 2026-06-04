"""
АЛХАМ 2 — Сарын загваруудыг эцэслэх, хадгалах
==============================================
- Хороо × сар panel дээр 3 шалтгааны XGBoost
- Шинэ feature-ууд: цаг агаар (weather_monthly.csv) + ОБЕГ ангиуд (stations_district.csv)
                     + календарийн (is_spring, is_heating_season, tsagaan_sar)
- Walk-forward CV (3 цонх): загварын найдвартай байдлыг тестийн хугацаанаас хамааралгүй шалгана
- Гаралт: model_m_*.json, panel_scored.parquet, metrics_monthly.json
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
import json

CAUSE_MN = {'ilgal': 'Ил гал', 'tsakhilgaan': 'Цахилгаан', 'yandan': 'Яндан/цонолт'}
TRAIN_END, TEST_START = 2023, 2024

# Цагаан сарын огноо — {жил: сар}
TSAGAAN_SAR = {
    2015: 2, 2016: 2, 2017: 1, 2018: 2, 2019: 2,
    2020: 1, 2021: 2, 2022: 2, 2023: 1, 2024: 2, 2025: 1,
}

# Шалтгаан бүрт хамааралтай цаг агаарын feature-ууд
WEATHER_FEATS = {
    'ilgal':       ['temp_mean', 'wind_max_mean', 'dry_days', 'fire_wx_days'],
    'tsakhilgaan': ['temp_mean', 'precip_sum', 'wind_max_mean'],
    'yandan':      ['temp_min_mean', 'cold_days_n25', 'cold_days_n35', 'hdd'],
}
CALENDAR_FEATS = ['is_spring', 'is_heating_season', 'tsagaan_sar']


def build_monthly_panel():
    fires = pd.read_csv('./data/fires_clean.csv', parse_dates=['datetime'])
    ub = fires[fires['is_ub'] == True].dropna(subset=['horoo']).copy()
    ub['horoo'] = ub['horoo'].astype(int)
    ub['year']  = ub['datetime'].dt.year
    ub['month'] = ub['datetime'].dt.month
    ub = ub[(ub['year'] >= 2015) & (ub['year'] <= 2025)]

    pop_kh  = pd.read_csv('./data/pop_khoroo.csv')
    khoroos = pop_kh[['district', 'khoroo']].drop_duplicates()
    months  = pd.DataFrame([(y, m) for y in range(2015, 2026) for m in range(1, 13)],
                            columns=['year', 'month'])
    panel = khoroos.merge(months, how='cross')

    for key, mn in CAUSE_MN.items():
        sub = ub[ub['cause'] == mn]
        cnt = (sub.groupby(['district', 'horoo', 'year', 'month']).size()
                  .reset_index(name=f'n_{key}').rename(columns={'horoo': 'khoroo'}))
        panel = panel.merge(cnt, on=['district', 'khoroo', 'year', 'month'], how='left')
        panel[f'n_{key}']    = panel[f'n_{key}'].fillna(0).astype(int)
        panel[f'fire_{key}'] = (panel[f'n_{key}'] > 0).astype(int)

    panel = panel.merge(pop_kh, on=['district', 'khoroo'], how='left')
    panel['log_pop'] = np.log1p(panel['pop'].fillna(panel['pop'].median()))

    # Цаг агаар (нийслэлийн дундаж, хороо бүрт ижил)
    weather = pd.read_csv('./data/weather_monthly.csv')
    panel = panel.merge(weather, on=['year', 'month'], how='left')

    # ОБЕГ ангиуд — дүүргийн ангийн тоо
    stations = pd.read_csv('./data/stations_district.csv')[['district', 'n_stations']]
    panel = panel.merge(stations, on='district', how='left')
    panel['n_stations'] = panel['n_stations'].fillna(1).astype(int)

    # Календарийн feature-ууд
    panel['is_spring']         = panel['month'].isin([3, 4, 5]).astype(int)
    panel['is_heating_season'] = panel['month'].isin([10, 11, 12, 1, 2, 3]).astype(int)
    panel['tsagaan_sar']       = panel.apply(
        lambda r: int(TSAGAAN_SAR.get(int(r['year']), -1) == int(r['month'])), axis=1)

    return panel.sort_values(['district', 'khoroo', 'year', 'month']).reset_index(drop=True)


def add_features(df, key):
    df = df.sort_values(['district', 'khoroo', 'year', 'month'])
    g = df.groupby(['district', 'khoroo'])[f'n_{key}']
    for lag in [1, 2, 3, 12]:
        df[f'{key}_lag{lag}'] = g.shift(lag).fillna(0)
    sh = g.shift(1)
    for win in [3, 6, 12]:
        df[f'{key}_roll{win}'] = (sh.groupby([df['district'], df['khoroo']])
            .rolling(win, min_periods=1).sum().reset_index(level=[0, 1], drop=True).fillna(0))
    df[f'{key}_cummean'] = (sh.groupby([df['district'], df['khoroo']])
        .expanding().mean().reset_index(level=[0, 1], drop=True).fillna(0))
    df = df.sort_values(['district', 'khoroo', 'month', 'year'])
    gm = df.groupby(['district', 'khoroo', 'month'])[f'fire_{key}']
    df[f'{key}_seasmonth'] = (gm.apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=[0, 1, 2], drop=True).fillna(0))
    return df.sort_values(['district', 'khoroo', 'year', 'month']).reset_index(drop=True)


def feats_for(key):
    base = ['month_sin', 'month_cos', 'month', 'year_idx', 'log_pop', 'n_stations']
    lag  = [f'{key}_lag1', f'{key}_lag2', f'{key}_lag3', f'{key}_lag12',
            f'{key}_roll3', f'{key}_roll6', f'{key}_roll12',
            f'{key}_cummean', f'{key}_seasmonth']
    return base + CALENDAR_FEATS + lag + WEATHER_FEATS.get(key, [])


def recall_at_k(y, s, g, k):
    d = pd.DataFrame({'y': y, 's': s, 'g': g})
    hit, fire = 0, 0
    for _, grp in d.groupby('g'):
        hit += grp.nlargest(k, 's')['y'].sum()
        fire += grp['y'].sum()
    return hit / fire if fire else np.nan


def _fit_model(Xtr, ytr):
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    m = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        min_child_weight=5, eval_metric='aucpr', n_jobs=-1, random_state=42,
    )
    m.fit(Xtr, ytr, verbose=False)
    return m


def walk_forward_cv(panel):
    """Walk-forward CV — 3 цонх × 3 загвар = 9 загвар сургана."""
    windows = [(2020, 2021), (2021, 2022), (2022, 2023)]
    rows = []
    for train_end, test_year in windows:
        tr = panel[panel['year'] <= train_end]
        te = panel[panel['year'] == test_year]
        for key, mn in CAUSE_MN.items():
            feats = feats_for(key)
            model = _fit_model(tr[feats], tr[f'fire_{key}'])
            proba = model.predict_proba(te[feats])[:, 1]
            yte   = te[f'fire_{key}']
            rows.append({
                'test_year': test_year,
                'cause':     mn,
                'roc_auc':   round(float(roc_auc_score(yte, proba)), 3),
                'pr_auc':    round(float(average_precision_score(yte, proba)), 3),
                'baseline':  round(float(yte.mean()), 3),
            })
    return rows


def main():
    panel = build_monthly_panel()
    panel['month_sin'] = np.sin(2 * np.pi * panel['month'] / 12)
    panel['month_cos'] = np.cos(2 * np.pi * panel['month'] / 12)
    panel['year_idx']  = panel['year'] - 2015

    for key in CAUSE_MN:
        panel = add_features(panel, key)

    # ── Walk-forward CV ──────────────────────────────────────────────────────
    print('Walk-forward CV ажиллаж байна (3 цонх × 3 загвар)...')
    cv_rows = walk_forward_cv(panel)
    cv_df   = pd.DataFrame(cv_rows)

    print('\n=== Walk-Forward Cross-Validation ===')
    cv_summary = {}
    for mn in CAUSE_MN.values():
        sub = cv_df[cv_df['cause'] == mn]
        for _, r in sub.iterrows():
            print(f"  [{r['test_year']}] {mn:20s}  ROC-AUC={r['roc_auc']:.3f}  PR-AUC={r['pr_auc']:.3f}  (суурь {r['baseline']*100:.0f}%)")
        avg_roc = round(sub['roc_auc'].mean(), 3)
        avg_pr  = round(sub['pr_auc'].mean(), 3)
        print(f"  [Дундаж]  {mn:20s}  ROC-AUC={avg_roc:.3f}  PR-AUC={avg_pr:.3f}\n")
        cv_summary[mn] = {
            'cv_roc_auc_mean': avg_roc,
            'cv_pr_auc_mean':  avg_pr,
            'cv_windows': sub[['test_year', 'roc_auc', 'pr_auc']].to_dict('records'),
        }

    # ── Эцсийн загвар (2015-2023 → 2024-2025) ───────────────────────────────
    print('Эцсийн загварууд сургагдаж байна...')
    metrics, all_imp = [], {}
    for key, mn in CAUSE_MN.items():
        feats = feats_for(key)
        tr = panel[panel['year'] <= TRAIN_END]
        te = panel[panel['year'] >= TEST_START]
        Xtr, ytr = tr[feats], tr[f'fire_{key}']
        Xte, yte = te[feats], te[f'fire_{key}']

        model = _fit_model(Xtr, ytr)
        model.save_model(f'./models/model_m_{key}.json')

        proba = model.predict_proba(Xte)[:, 1]
        grp   = (te['year'].astype(str) + '-' + te['month'].astype(str)).values
        metrics.append({
            'cause':        mn,
            'baseline':     float(yte.mean()),
            'pr_auc':       float(average_precision_score(yte, proba)),
            'roc_auc':      float(roc_auc_score(yte, proba)),
            'recall_top30': float(recall_at_k(yte.values, proba, grp, 30)),
            'recall_top50': float(recall_at_k(yte.values, proba, grp, 50)),
        })
        all_imp[mn] = (pd.Series(model.feature_importances_, index=feats)
                       .sort_values(ascending=False).head(8).round(3).to_dict())

        panel[f'risk_{key}'] = model.predict_proba(panel[feats])[:, 1]

    panel.to_parquet('./data/panel_scored.parquet', index=False)
    with open('./models/metrics_monthly.json', 'w', encoding='utf-8') as f:
        json.dump({'metrics': metrics, 'importance': all_imp,
                   'cv_summary': cv_summary}, f, ensure_ascii=False, indent=2)

    print('\n✓ Загварууд хадгалагдлаа (model_m_*.json)')
    print('✓ panel_scored.parquet — эрсдэлийн оноотой')

    print('\n=== Эцсийн үр дүн (тест: 2024-2025) ===')
    for m in metrics:
        print(f"\n▶ {m['cause']} (суурь {m['baseline']*100:.0f}%)")
        print(f"  PR-AUC {m['pr_auc']:.3f} | ROC-AUC {m['roc_auc']:.3f}")
        print(f"  Дээд 30 хороонд {m['recall_top30']*100:.0f}%, дээд 50-д {m['recall_top50']*100:.0f}% хамрана")

    # Шинэ feature-уудын importance шалгах
    new_feats = set(CALENDAR_FEATS) | {'n_stations'}
    for v in WEATHER_FEATS.values():
        new_feats.update(v)
    print('\n=== Шинэ feature-уудын importance ===')
    for mn, imp in all_imp.items():
        hits = {k: v for k, v in imp.items() if k in new_feats}
        if hits:
            print(f"  {mn}: {hits}")


if __name__ == '__main__':
    main()
