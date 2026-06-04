"""
АЛХАМ 3 — Feature engineering
=============================
Хамгийн хүчтэй predictor: ТҮҮХЭН ЗАГВАР (gadny data shaardakhgüi).
  - Lag: өнгөрсөн N долоо хоногийн галын тоо
  - Rolling: сүүлийн 4/8/12 долоо хоногийн нийлбэр
  - Улирлын: тухайн хорооны тухайн сар дахь түүхэн дундаж
  - Жилийн өмнөх: 52 долоо хоногийн өмнөх ижил үе
  - Хүн ам, тренд

ЧУХАЛ: Бүх lag/rolling нь ЗӨВХӨН ӨНГӨРСӨН датаг ашиглана (leakage-аас сэргийлж).

Гаралт: features.parquet
"""
import pandas as pd
import numpy as np

CAUSE_KEYS = ['ilgal', 'tsakhilgaan', 'yandan']

def add_time_features(df):
    """Цаг хугацааны мөчлөгийн features (sin/cos encoding)."""
    df['week_sin'] = np.sin(2 * np.pi * df['iso_week'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['iso_week'] / 52)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['year_idx'] = df['iso_year'] - df['iso_year'].min()  # тренд
    return df

def add_lag_features(df, target_col, prefix):
    """
    Хороо тус бүрд цаг хугацааны дарааллаар lag/rolling features.
    Бүгд shift(1)-ээр эхэлдэг тул одоогийн долоо хоногийн утга ОРОХГҮЙ (leakage үгүй).
    """
    df = df.sort_values(['district', 'khoroo', 'iso_year', 'iso_week'])
    g = df.groupby(['district', 'khoroo'])[target_col]

    # Сүүлийн долоо хоногуудын lag
    for lag in [1, 2, 4]:
        df[f'{prefix}_lag{lag}'] = g.shift(lag).fillna(0)

    # Rolling нийлбэр (өнгөрсөн цонх) — shift(1) дараа rolling
    shifted = g.shift(1)
    for win in [4, 8, 12, 26, 52]:
        df[f'{prefix}_roll{win}'] = (
            shifted.groupby([df['district'], df['khoroo']])
                   .rolling(win, min_periods=1).sum()
                   .reset_index(level=[0, 1], drop=True)
                   .fillna(0)
        )

    # Жилийн өмнөх ижил долоо хоног (52 долоо хоногийн өмнө)
    df[f'{prefix}_lag52'] = g.shift(52).fillna(0)

    # Хорооны нийт түүхэн дундаж (expanding, өнгөрсөн л)
    df[f'{prefix}_cummean'] = (
        shifted.groupby([df['district'], df['khoroo']])
               .expanding().mean()
               .reset_index(level=[0, 1], drop=True)
               .fillna(0)
    )
    return df

def add_seasonal_baseline(df, target_col, prefix):
    """
    Тухайн хорооны тухайн ISO долоо хоногийн түүхэн дундаж эрсдэл.
    Зөвхөн өмнөх жилүүдийн датаг ашиглахын тулд expanding-аар тооцно.
    """
    df = df.sort_values(['district', 'khoroo', 'iso_week', 'iso_year'])
    g = df.groupby(['district', 'khoroo', 'iso_week'])[target_col]
    df[f'{prefix}_seasweek'] = (
        g.apply(lambda s: s.shift(1).expanding().mean())
         .reset_index(level=[0, 1, 2], drop=True)
         .fillna(0)
    )
    return df

def main():
    df = pd.read_parquet('./data/panel.parquet')

    # Хүн ам нэмэх (хороо түвшин 2025, статик)
    pop_kh = pd.read_csv('./data/pop_khoroo.csv')
    df = df.merge(pop_kh, on=['district', 'khoroo'], how='left')
    df['pop'] = df['pop'].fillna(df['pop'].median())
    df['log_pop'] = np.log1p(df['pop'])

    # Дүүргийн жилийн хүн ам (оноор өөрчлөгддөг тренд)
    pop_d = pd.read_csv('./data/pop_district.csv')
    pop_d = pop_d.rename(columns={'year': 'iso_year', 'pop': 'district_pop'})
    df = df.merge(pop_d, on=['district', 'iso_year'], how='left')
    df['district_pop'] = df.groupby('district')['district_pop'].ffill().bfill()
    df['log_district_pop'] = np.log1p(df['district_pop'])

    df = add_time_features(df)

    # Шалтгаан бүрд lag + seasonal features
    for key in CAUSE_KEYS:
        df = add_lag_features(df, f'n_{key}', key)
        df = add_seasonal_baseline(df, f'fire_{key}', key)

    df = df.sort_values(['district', 'khoroo', 'iso_year', 'iso_week']).reset_index(drop=True)
    df.to_parquet('./data/features.parquet', index=False)

    n_feat = len([c for c in df.columns if any(k in c for k in CAUSE_KEYS)
                  and ('lag' in c or 'roll' in c or 'cummean' in c or 'seasweek' in c)])
    print('✓ features.parquet:', df.shape)
    print(f'  Lag/rolling/seasonal features: {n_feat}')
    print(f'  Цаг хугацааны features: week_sin/cos, month_sin/cos, year_idx')
    print(f'  Хүн ам: log_pop, log_district_pop')
    print('\nЖишээ багана:')
    print([c for c in df.columns if 'ilgal' in c])

if __name__ == '__main__':
    main()
