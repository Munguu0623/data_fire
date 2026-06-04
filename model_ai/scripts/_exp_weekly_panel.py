"""
АЛХАМ 2 — Хороо × долоо хоног panel байгуулах
==============================================
Гол санаа: бүх хороо × бүх долоо хоног (2015-2025) бүхий бүтэн хүснэгт.
Гал гараагүй нүднүүд автоматаар "сөрөг жишээ" болж XGBoost-д хэрэгтэй тэнцвэр үүснэ.

Гурван гол шалтгаан тус бүрд галын тоо болон гарсан/үгүй (binary) баганатай.

Гаралт: panel.parquet  (хороо × долоо хоног, шалтгаан бүрийн галын тоо)
"""
import pandas as pd
import numpy as np

FIRES = './data/fires_clean.csv'
CAUSES = ['Ил гал', 'Цахилгаан', 'Яндан/цонолт']  # гол 3 шалтгаан
CAUSE_KEY = {'Ил гал': 'ilgal', 'Цахилгаан': 'tsakhilgaan', 'Яндан/цонолт': 'yandan'}
START_YEAR, END_YEAR = 2015, 2025

def load_fires():
    df = pd.read_csv(FIRES, parse_dates=['datetime'])
    ub = df[df['is_ub'] == True].dropna(subset=['horoo']).copy()
    ub['horoo'] = ub['horoo'].astype(int)
    # ISO долоо хоног (Даваа гарагаас эхэлдэг, олон улсын стандарт)
    iso = ub['datetime'].dt.isocalendar()
    ub['iso_year'] = iso['year'].astype(int)
    ub['iso_week'] = iso['week'].astype(int)
    # 2015-2025 он
    ub = ub[(ub['iso_year'] >= START_YEAR) & (ub['iso_year'] <= END_YEAR)]
    return ub

def build_week_index():
    """Бүх (iso_year, iso_week) хослолыг жинхэнэ календарь дээр үндэслэн үүсгэх."""
    dates = pd.date_range(f'{START_YEAR}-01-01', f'{END_YEAR}-12-31', freq='D')
    iso = dates.isocalendar()
    wk = pd.DataFrame({'iso_year': iso['year'].values,
                       'iso_week': iso['week'].values}).drop_duplicates()
    # Зөвхөн манай он мужид
    wk = wk[(wk['iso_year'] >= START_YEAR) & (wk['iso_year'] <= END_YEAR)]
    # Долоо хоног бүрийн төлөөлөх огноо (даваа) — сар/улирал гаргахад
    wk['week_start'] = wk.apply(
        lambda r: pd.Timestamp.fromisocalendar(int(r['iso_year']), int(r['iso_week']), 1),
        axis=1)
    return wk.sort_values(['iso_year', 'iso_week']).reset_index(drop=True)

def khoroo_index():
    """Бүх хүчинтэй (district, khoroo) хослол — хүн амын датанаас."""
    pk = pd.read_csv('./data/pop_khoroo.csv')
    return pk[['district', 'khoroo']].drop_duplicates().reset_index(drop=True)

def main():
    fires = load_fires()
    weeks = build_week_index()
    khoroos = khoroo_index()

    print(f'Хороо: {len(khoroos)} | Долоо хоног: {len(weeks)}')

    # Бүрэн panel grid: хороо × долоо хоног (cross join)
    panel = khoroos.merge(weeks, how='cross')
    print(f'Panel хэмжээ: {len(panel):,} мөр')

    # Шалтгаан бүрийн галын тоог тоолж нэгтгэх
    for cause in CAUSES:
        key = CAUSE_KEY[cause]
        sub = fires[fires['cause'] == cause]
        cnt = (sub.groupby(['district', 'horoo', 'iso_year', 'iso_week'])
                  .size().reset_index(name=f'n_{key}'))
        cnt = cnt.rename(columns={'horoo': 'khoroo'})
        panel = panel.merge(cnt, on=['district', 'khoroo', 'iso_year', 'iso_week'],
                            how='left')
        panel[f'n_{key}'] = panel[f'n_{key}'].fillna(0).astype(int)
        # Binary: гал гарсан эсэх
        panel[f'fire_{key}'] = (panel[f'n_{key}'] > 0).astype(int)

    # Нийт гал (бүх шалтгаан, лавлагаанд)
    all_cnt = (fires.groupby(['district', 'horoo', 'iso_year', 'iso_week'])
                  .size().reset_index(name='n_all').rename(columns={'horoo': 'khoroo'}))
    panel = panel.merge(all_cnt, on=['district', 'khoroo', 'iso_year', 'iso_week'],
                        how='left')
    panel['n_all'] = panel['n_all'].fillna(0).astype(int)

    # Сар, улирал (долоо хоногийн эхлэх огнооноос)
    panel['month'] = panel['week_start'].dt.month
    panel['season'] = panel['month'].map(lambda m:
        'Өвөл' if m in (12, 1, 2) else 'Хавар' if m in (3, 4, 5)
        else 'Зун' if m in (6, 7, 8) else 'Намар')

    panel = panel.sort_values(['district', 'khoroo', 'iso_year', 'iso_week']).reset_index(drop=True)
    panel.to_parquet('./data/panel.parquet', index=False)

    # Тойм статистик
    print('\n=== Panel тойм ===')
    print('Нийт мөр:', f'{len(panel):,}')
    for cause in CAUSES:
        key = CAUSE_KEY[cause]
        pos = panel[f'fire_{key}'].sum()
        rate = pos / len(panel) * 100
        print(f'{cause:14s}: гал гарсан долоо хоног-хороо = {pos:,} ({rate:.2f}%)')

if __name__ == '__main__':
    main()
