"""
OSM-аас гэрийн барилгын дата татах
Strategy:
  1. Бүх UB-ийн ger+yurt барилгыг нэг query-оор татна (center coordinates-тай)
  2. Дүүрэг бүрийн bbox-оор тоолно
  3. Нийт барилгын тоог дүүрэг бүрт тус тусад нь count query-оор авна
"""
import requests, time, json, sys
import pandas as pd

PRIMARY   = "https://overpass.openstreetmap.fr/api/interpreter"
FALLBACK  = "https://overpass-api.de/api/interpreter"
HEADERS   = {"User-Agent": "UB-fire-risk/1.0"}

# Approximate district bboxes [S, W, N, E]
DISTRICT_BBOX = {
    'Баянгол':         (47.855, 106.785, 47.935, 106.920),
    'Баянзүрх':       (47.865, 106.910, 47.975, 107.180),
    'Сонгинохайрхан': (47.890, 106.645, 48.055, 106.900),
    'Сүхбаатар':      (47.895, 106.860, 47.965, 107.010),
    'Чингэлтэй':      (47.885, 106.840, 47.965, 106.965),
    'Хан-Уул':        (47.820, 106.855, 47.910, 107.040),
    'Налайх':         (47.680, 107.180, 47.830, 107.400),
    'Багануур':       (47.665, 108.190, 47.775, 108.380),
    'Багахангай':     (47.800, 106.300, 47.910, 106.500),
}

FALLBACK_RATIO = {
    'Баянгол':0.42,'Баянзүрх':0.60,'Сонгинохайрхан':0.72,
    'Сүхбаатар':0.25,'Чингэлтэй':0.30,'Хан-Уул':0.40,
    'Налайх':0.75,'Багануур':0.62,'Багахангай':0.82,
}


def query(q, timeout=90):
    for ep in [PRIMARY, FALLBACK]:
        try:
            r = requests.post(ep, data={"data": q}, headers=HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            print(f"  {ep}: {e}", file=sys.stderr)
        time.sleep(2)
    return None


def get_all_ger_buildings():
    """UB нутаг дэвсгэрийн бүх ger+yurt барилгыг center-тай авна (1 query)."""
    q = """
[out:json][timeout:120];
(
  way[building~"^(ger|yurt)$"](47.65,106.30,48.10,107.50);
  way[building~"^(ger|yurt)$"](47.65,108.15,47.80,108.40);
);
out center;
"""
    print("Гэрийн барилгуудыг татаж байна (1 query)...", flush=True)
    data = query(q, timeout=120)
    if not data:
        return None
    centers = []
    for el in data.get("elements", []):
        c = el.get("center", {})
        lat = c.get("lat") or el.get("lat")
        lon = c.get("lon") or el.get("lon")
        if lat and lon:
            centers.append((lat, lon))
    print(f"  ✓ {len(centers)} гэрийн барилга олдлоо")
    return centers


def count_all_buildings(district, bbox):
    """Дүүргийн нийт барилгын тоог count query-оор авна."""
    S, W, N, E = bbox
    q = f"""
[out:json][timeout:45];
way[building]({S},{W},{N},{E});
out count;
"""
    data = query(q, timeout=50)
    if data and data.get("elements"):
        return int(data["elements"][0]["tags"].get("total", 0))
    return None


def assign_to_district(lat, lon):
    for d, (S, W, N, E) in DISTRICT_BBOX.items():
        if S <= lat <= N and W <= lon <= E:
            return d
    return None


def main():
    ger_centers = get_all_ger_buildings()

    ger_counts = {d: 0 for d in DISTRICT_BBOX}
    if ger_centers:
        for lat, lon in ger_centers:
            d = assign_to_district(lat, lon)
            if d:
                ger_counts[d] += 1
        print(f"  Дүүргүүдэд хуваарилсан: {sum(ger_counts.values())}")
    else:
        print("  Гэрийн барилга олдсонгүй — бүгдэд fallback ашиглана")

    rows = []
    for district, bbox in DISTRICT_BBOX.items():
        n_ger = ger_counts.get(district, 0)

        if ger_centers is not None:
            print(f"\n{district} нийт барилга тоолж байна...", flush=True)
            n_all = count_all_buildings(district, bbox)
            time.sleep(3)
        else:
            n_all = None

        if n_all and n_all > 50:
            ratio = round(n_ger / n_all, 4)
            src = 'osm'
            print(f"  ✓ {n_ger} гэр / {n_all} нийт = {ratio:.1%}")
        else:
            ratio = FALLBACK_RATIO.get(district, 0.50)
            src = 'fallback'
            if n_all is not None:
                print(f"  ✗ Нийт барилга олдсонгүй → fallback {ratio:.0%}")
            else:
                print(f"  Fallback: {district} {ratio:.0%}")

        rows.append({'district': district, 'ger_ratio': ratio, 'source': src})

    df = pd.DataFrame(rows)
    df.to_csv('./data/ger_density.csv', index=False)
    print("\n✓ ger_density.csv шинэчлэгдлээ")
    print(df.sort_values('ger_ratio', ascending=False).to_string(index=False))


if __name__ == '__main__':
    main()
