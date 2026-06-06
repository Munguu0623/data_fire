"""
АЛХАМ 0b — Гэр хорооллын нягтрал (OpenStreetMap Overpass API)
=============================================================
Дүүрэг бүрийн гэрийн барилгын нягтралыг OSM-аас тооцоолно.
Амжилттай бол data/ger_density.csv-г шинэчилнэ.
Алдаа гарвал одоогийн файлыг хэвээр үлдээнэ.

Шаардлага: pip install requests
Ажиллуулах: model_ai/ дотроос
"""
import requests
import pandas as pd
import time
import sys

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
HEADERS = {"User-Agent": "UB-fire-risk-dashboard/1.0"}

# Дүүргийн Монгол нэр → OSM-н хайлтын нэрс (алдааны тэсвэртэй)
DISTRICT_OSM = {
    'Баянгол':         ['Баянгол дүүрэг', 'Bayangol düüreg', 'Bayangol'],
    'Баянзүрх':       ['Баянзүрх дүүрэг', 'Bayanzürkh', 'Bayanzurkh'],
    'Сонгинохайрхан': ['Сонгинохайрхан дүүрэг', 'Songinokhairkhan'],
    'Сүхбаатар':      ['Сүхбаатар дүүрэг', 'Sükhbaatar', 'Sukhbaatar'],
    'Чингэлтэй':      ['Чингэлтэй дүүрэг', 'Chingeltei'],
    'Хан-Уул':        ['Хан-Уул дүүрэг', 'Khan-Uul', 'Khanuu'],
    'Налайх':         ['Налайх дүүрэг', 'Nalaikh'],
    'Багануур':       ['Багануур дүүрэг', 'Baganuur'],
    'Багахангай':     ['Багахангай дүүрэг', 'Bagakhangai'],
}

# Overpass хариугүй болсон үеийн fallback утгууд
FALLBACK = {
    'Баянгол': 0.42, 'Баянзүрх': 0.60, 'Сонгинохайрхан': 0.72,
    'Сүхбаатар': 0.25, 'Чингэлтэй': 0.30, 'Хан-Уул': 0.40,
    'Налайх': 0.75, 'Багануур': 0.62, 'Багахангай': 0.82,
}


def overpass_count(area_name, building_filter):
    q = f"""
[out:json][timeout:60];
area["name"="{area_name}"]["admin_level"="5"]->.d;
(
  way(area.d){building_filter};
  node(area.d){building_filter};
);
out count;
"""
    r = requests.post(OVERPASS_URL, data={"data": q}, headers=HEADERS, timeout=70)
    r.raise_for_status()
    data = r.json()
    return int(data["elements"][0]["tags"].get("total", 0))


def get_ger_ratio(district):
    for osm_name in DISTRICT_OSM.get(district, []):
        try:
            n_ger   = overpass_count(osm_name, '["building"~"^(ger|yurt)$"]')
            n_total = overpass_count(osm_name, '["building"]')
            if n_total > 10:
                ratio = round(n_ger / n_total, 4)
                print(f"  {district}: {n_ger} гэр / {n_total} нийт = {ratio:.2%}  [{osm_name}]")
                return ratio, 'osm'
        except Exception as e:
            print(f"  {district} [{osm_name}] алдаа: {e}", file=sys.stderr)
        time.sleep(2)  # rate limit
    # Fallback
    fb = FALLBACK.get(district, 0.50)
    print(f"  {district}: OSM олдсонгүй → fallback {fb:.0%}")
    return fb, 'fallback'


def main():
    existing = pd.read_csv('./data/ger_density.csv')
    districts = existing['district'].tolist()

    rows = []
    for d in districts:
        print(f"Хайж байна: {d} ...")
        ratio, src = get_ger_ratio(d)
        rows.append({'district': d, 'ger_ratio': ratio, 'source': src})
        time.sleep(1)

    df = pd.DataFrame(rows)
    df.to_csv('./data/ger_density.csv', index=False)
    print('\n✓ data/ger_density.csv шинэчлэгдлээ')
    print(df.sort_values('ger_ratio', ascending=False).to_string(index=False))


if __name__ == '__main__':
    main()
