"""
Хуваалцсан feature инженерийн модуль.
02_train_monthly.py (сургалт) болон api/predictor.py (serving) хоёулаа
энэ файлаас импортолно — train/serve тохирол алдагдахгүй.
"""
import pandas as pd
import numpy as np

CAUSE_MN = {'ilgal': 'Ил гал', 'tsakhilgaan': 'Цахилгаан', 'yandan': 'Яндан/цонолт'}

TSAGAAN_SAR = {
    2015: 2, 2016: 2, 2017: 1, 2018: 2, 2019: 2,
    2020: 1, 2021: 2, 2022: 2, 2023: 1, 2024: 2, 2025: 1,
    2026: 2,
}

WEATHER_FEATS = {
    'ilgal':       ['temp_mean', 'wind_max_mean', 'dry_days', 'fire_wx_days'],
    'tsakhilgaan': ['temp_mean', 'precip_sum', 'wind_max_mean'],
    'yandan':      ['temp_min_mean', 'cold_days_n25', 'cold_days_n35', 'hdd'],
}

CALENDAR_FEATS = ['is_spring', 'is_heating_season', 'tsagaan_sar']


def add_features(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Panel дээр lag/rolling/seasonal feature-ууд нэмнэ (vectorized)."""
    df = df.sort_values(['district', 'khoroo', 'year', 'month'])
    g = df.groupby(['district', 'khoroo'])[f'n_{key}']
    for lag in [1, 2, 3, 12]:
        df[f'{key}_lag{lag}'] = g.shift(lag).fillna(0)
    sh = g.shift(1)
    for win in [3, 6, 12]:
        df[f'{key}_roll{win}'] = (
            sh.groupby([df['district'], df['khoroo']])
            .rolling(win, min_periods=1).sum()
            .reset_index(level=[0, 1], drop=True).fillna(0)
        )
    df[f'{key}_cummean'] = (
        sh.groupby([df['district'], df['khoroo']])
        .expanding().mean()
        .reset_index(level=[0, 1], drop=True).fillna(0)
    )
    df = df.sort_values(['district', 'khoroo', 'month', 'year'])
    gm = df.groupby(['district', 'khoroo', 'month'])[f'fire_{key}']
    df[f'{key}_seasmonth'] = (
        gm.apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=[0, 1, 2], drop=True).fillna(0)
    )
    return df.sort_values(['district', 'khoroo', 'year', 'month']).reset_index(drop=True)


GER_FEATS = ['ger_ratio']


def feats_for(key: str) -> list[str]:
    """Загвар бүрт хэрэглэгдэх feature-уудын жагсаалт."""
    base = ['month_sin', 'month_cos', 'month', 'year_idx', 'log_pop', 'n_stations']
    lag  = [
        f'{key}_lag1', f'{key}_lag2', f'{key}_lag3', f'{key}_lag12',
        f'{key}_roll3', f'{key}_roll6', f'{key}_roll12',
        f'{key}_cummean', f'{key}_seasmonth',
    ]
    return base + GER_FEATS + CALENDAR_FEATS + lag + WEATHER_FEATS.get(key, [])
