"""
АЛХАМ 6 — Дашбордын дата бэлдэх
================================
panel_scored.parquet → dashboard_data.json
ОБ-ын хэрэглээнд: сар сонгоход хороо бүрийн эрсдэлийн зэрэг, шалтгаанаар.
"""
import pandas as pd
import numpy as np
import json

CAUSE_MN = {'ilgal': 'Ил гал', 'tsakhilgaan': 'Цахилгаан', 'yandan': 'Яндан/цонолт'}

def main():
    p = pd.read_parquet('./data/panel_scored.parquet')

    # Дашбордод 2025 оны таамаглал (хамгийн сүүлийн жил) харуулна
    latest = p[p['year'] == 2025].copy()

    # Хороо бүрийн эрсдэлийн оноог сараар бүлэглэх
    records = []
    for _, r in latest.iterrows():
        records.append({
            'district': r['district'],
            'khoroo': int(r['khoroo']),
            'month': int(r['month']),
            'pop': int(r['pop']) if pd.notna(r['pop']) else 0,
            'risk_ilgal': round(float(r['risk_ilgal']), 4),
            'risk_tsakhilgaan': round(float(r['risk_tsakhilgaan']), 4),
            'risk_yandan': round(float(r['risk_yandan']), 4),
            'n_ilgal': int(r['n_ilgal']),
            'n_tsakhilgaan': int(r['n_tsakhilgaan']),
            'n_yandan': int(r['n_yandan']),
        })

    # Метрикийг оруулах
    with open('./models/metrics_monthly.json') as f:
        metrics = json.load(f)

    # Дүүргийн нэгтгэл (хорооны эрсдэлийн дундаж)
    dist_summary = (latest.groupby(['district', 'month'])
        [['risk_ilgal', 'risk_tsakhilgaan', 'risk_yandan']]
        .mean().round(4).reset_index())

    out = {
        'records': records,
        'districts': sorted(latest['district'].unique().tolist()),
        'metrics': metrics['metrics'],
        'importance': metrics['importance'],
        'meta': {
            'n_khoroo': int(latest[['district', 'khoroo']].drop_duplicates().shape[0]),
            'year': 2025,
            'train_period': '2015-2023',
            'test_period': '2024-2025',
        }
    }
    with open('./data/dashboard_data.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False)

    print('✓ dashboard_data.json')
    print(f'  Бичлэг: {len(records):,} (хороо × сар)')
    print(f'  Дүүрэг: {len(out["districts"])}')
    # Жишээ: 3-р сарын хамгийн эрсдэлтэй 5 хороо (ил гал)
    mar = latest[latest['month'] == 3].nlargest(5, 'risk_ilgal')
    print('\nЖишээ — 3-р сар ил галын эрсдэлт дээд 5 хороо:')
    for _, r in mar.iterrows():
        print(f"  {r['district']} {int(r['khoroo'])}-р хороо: {r['risk_ilgal']:.2f}")

if __name__ == '__main__':
    main()
