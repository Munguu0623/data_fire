"""
АЛХАМ 0b — ОБЕГ ангиудын дата боловсруулах
===========================================
УБ-ОБЕГ-ангиуд.xlsx → data/stations_district.csv

Дүүрэг бүрийн гал унтраах ангийн тоо (статик загварын feature).
"""
import pandas as pd

# Товчлол → бүтэн нэр (fires_clean.csv дахь district нэртэй таарах ёстой)
DIST_MAP = {
    'ЧД':       'Чингэлтэй',
    'СХД':      'Сонгинохайрхан',
    'ХУД':      'Хан-Уул',
    'БГД':      'Баянгол',
    'СБД':      'Сүхбаатар',
    'БЗД':      'Баянзүрх',
    'Багануур':  'Багануур',
    'Багахангай':'Багахангай',
    'Налайх':    'Налайх',
}


def parse_coord(s):
    if pd.isna(s):
        return None, None
    parts = str(s).split(',')
    if len(parts) == 2:
        try:
            return float(parts[0].strip()), float(parts[1].strip())
        except ValueError:
            pass
    return None, None


def main():
    df = pd.read_excel('./data/УБ-ОБЕГ-ангиуд.xlsx', sheet_name='Sheet1')
    df = df[['Харьяалах дүүрэг', 'Анги', 'Координат']].copy()
    df.columns = ['dist_raw', 'name', 'coord']
    df = df.dropna(subset=['dist_raw', 'name'])

    df['district'] = df['dist_raw'].str.strip().map(DIST_MAP)
    df = df[df['district'].notna()].copy()

    coord_parsed = df['coord'].apply(lambda x: pd.Series(parse_coord(x), index=['lat', 'lon']))
    df = pd.concat([df[['district', 'name']], coord_parsed], axis=1)

    counts  = df.groupby('district').size().reset_index(name='n_stations')
    avg_pos = (df.dropna(subset=['lat'])
                 .groupby('district')[['lat', 'lon']].mean()
                 .round(5).reset_index())

    result = counts.merge(avg_pos, on='district', how='left')
    result.to_csv('./data/stations_district.csv', index=False)

    print('✓ stations_district.csv:')
    print(result.to_string(index=False))
    print(f'\nНийт анги: {counts["n_stations"].sum()}  |  Дүүрэг: {len(counts)}')


if __name__ == '__main__':
    main()
