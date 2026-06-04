"""
АЛХАМ 0a — Цаг агаарын дата боловсруулах (water.json)
======================================================
Гаралт: data/weather_monthly.csv

Баганууд:
  temp_mean, temp_min_mean, temp_max_mean  — температур
  precip_sum, snowfall_sum                 — хур тунадас, цас
  wind_max_mean                            — салхи
  cold_days_n25, cold_days_n35             — хүйтний экстрем (яндан feature)
  dry_days, fire_wx_days                   — хуурай/галын цаг агаар (ил гал feature)
  hdd                                      — халаалтын градус-өдөр (яндан feature)
"""
import json
import numpy as np
import pandas as pd


def main():
    with open('./data/water.json', encoding='utf-8') as f:
        w = json.load(f)

    daily = pd.DataFrame({
        'date':      pd.to_datetime(w['daily']['time']),
        'temp_max':  w['daily']['temperature_2m_max'],
        'temp_min':  w['daily']['temperature_2m_min'],
        'temp_mean': w['daily']['temperature_2m_mean'],
        'precip':    w['daily']['precipitation_sum'],
        'snowfall':  w['daily']['snowfall_sum'],
        'wind_max':  w['daily']['windspeed_10m_max'],
    }).dropna(subset=['temp_mean'])

    daily['year']  = daily['date'].dt.year
    daily['month'] = daily['date'].dt.month
    daily['hdd_day'] = np.maximum(0.0, 18.0 - daily['temp_mean'])

    # Хоногийн бинар тугнууд — сараар нийлбэрлэнэ
    daily['flag_cold25']  = (daily['temp_min'] < -25).astype(int)
    daily['flag_cold35']  = (daily['temp_min'] < -35).astype(int)
    daily['flag_dry']     = (daily['precip'] < 0.5).astype(int)
    # Галын цаг агаарын нөхцөл: дулаан + салхитай + хуурай
    daily['flag_fire_wx'] = (
        (daily['temp_mean'] > 5) &
        (daily['wind_max'] > 15) &
        (daily['precip'] < 0.5)
    ).astype(int)

    weather = daily.groupby(['year', 'month']).agg(
        temp_mean     =('temp_mean',     'mean'),
        temp_min_mean =('temp_min',      'mean'),
        temp_max_mean =('temp_max',      'mean'),
        precip_sum    =('precip',        'sum'),
        snowfall_sum  =('snowfall',      'sum'),
        wind_max_mean =('wind_max',      'mean'),
        cold_days_n25 =('flag_cold25',   'sum'),
        cold_days_n35 =('flag_cold35',   'sum'),
        dry_days      =('flag_dry',      'sum'),
        fire_wx_days  =('flag_fire_wx',  'sum'),
        hdd           =('hdd_day',       'sum'),
    ).reset_index().round(3)

    weather.to_csv('./data/weather_monthly.csv', index=False)

    print(f'✓ weather_monthly.csv: {weather.shape}  [{weather.year.min()}-{weather.year.max()}]')
    print('\nЖишээ — 1-р сар (яндан):')
    cols_y = ['year', 'temp_mean', 'temp_min_mean', 'cold_days_n25', 'cold_days_n35', 'hdd']
    print(weather[weather['month'] == 1][cols_y].tail(6).to_string(index=False))
    print('\nЖишээ — 4-р сар (ил гал):')
    cols_i = ['year', 'temp_mean', 'wind_max_mean', 'dry_days', 'fire_wx_days']
    print(weather[weather['month'] == 4][cols_i].tail(6).to_string(index=False))


if __name__ == '__main__':
    main()
