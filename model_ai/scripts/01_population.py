"""
АЛХАМ 1 — Хүн амын датаг цэвэрлэх
==================================
Гаралт:
  - pop_district.csv : дүүрэг × жил (1990-2025) хүн амын тоо
  - pop_khoroo.csv   : дүүрэг × хороо хүн амын тоо (зөвхөн 2025)

Тэмдэглэл: Excel-д зөвхөн 2025 онд хороогоор задлагдсан дата байгаа.
Бусад жил дүүргийн түвшний дүнтэй.
"""
import pandas as pd
import re

SRC = './data/Суурин_хүн_амын_тоо.xlsx'

DISTRICTS = ['Багануур','Багахангай','Баянгол','Баянзүрх','Налайх',
             'Сонгинохайрхан','Сүхбаатар','Хан-Уул','Чингэлтэй']

# Дүүрэг бүрийн хорооны тоо (Excel-д гарч буй дарааллаар) — Wikipedia/NSO лавлагаа
# Файлын дотоод дараалал нь дүүргийн толгойн дараалалтай ижил болохыг шалгасан
KHOROO_ORDER = [
    ('Багануур', 5), ('Багахангай', 2), ('Баянгол', 34), ('Баянзүрх', 43),
    ('Налайх', 8), ('Сонгинохайрхан', 43), ('Сүхбаатар', 20),
    ('Хан-Уул', 25), ('Чингэлтэй', 24),
]

def load_raw():
    df = pd.read_excel(SRC, sheet_name=0, header=0)
    df.columns = ['no', 'year', 'name', 'name2', 'value']
    df['name'] = df['name'].astype(str).str.strip()
    return df

def build_district(df):
    """Дүүрэг × жил хүн амын хүснэгт (1990-2025)."""
    dd = df[df['name'].isin(DISTRICTS)][['year', 'name', 'value']].copy()
    dd = dd.rename(columns={'name': 'district', 'value': 'pop'})
    dd['year'] = dd['year'].astype(int)
    dd['pop'] = dd['pop'].astype(int)
    return dd.sort_values(['year', 'district']).reset_index(drop=True)

def build_khoroo(df):
    """Хороо × хүн ам (зөвхөн 2025). Лавлагааны хорооны тоогоор дараалан таслана."""
    y = df[df['year'] == 2025].reset_index(drop=True)
    kh = y[y['name'].str.contains('хороо', na=False)].reset_index(drop=True)
    kh['knum'] = kh['name'].str.extract(r'(\d+)').astype(int)

    assert len(kh) == sum(c for _, c in KHOROO_ORDER), \
        f"Хорооны мөрийн тоо таарахгүй: {len(kh)}"

    rows, idx = [], 0
    for dist, cnt in KHOROO_ORDER:
        block = kh.iloc[idx:idx + cnt]
        for _, r in block.iterrows():
            rows.append({'district': dist, 'khoroo': int(r['knum']),
                         'pop': int(r['value'])})
        idx += cnt
    out = pd.DataFrame(rows)

    # Баталгаажуулалт: хорооны нийлбэр = дүүргийн толгойн утга
    hdr = y[y['name'].isin(DISTRICTS)].set_index('name')['value']
    for dist, _ in KHOROO_ORDER:
        s = out[out['district'] == dist]['pop'].sum()
        assert s == hdr[dist], f"{dist}: {s} != {hdr[dist]}"
    return out

if __name__ == '__main__':
    df = load_raw()
    pd_district = build_district(df)
    pd_khoroo = build_khoroo(df)

    pd_district.to_csv('./data/pop_district.csv', index=False)
    pd_khoroo.to_csv('./data/pop_khoroo.csv', index=False)

    print('✓ pop_district.csv:', pd_district.shape,
          '| жил', pd_district['year'].min(), '-', pd_district['year'].max())
    print('✓ pop_khoroo.csv:  ', pd_khoroo.shape, '| нийт хороо', len(pd_khoroo))
    print('\nДүүргийн хүн ам (2025):')
    print(pd_district[pd_district['year'] == 2025].to_string(index=False))
