"""
04 — Дашбордын HTML үүсгэх
==========================
dashboard_data.json (түүхэн 2025) болон forecast_data.json (дараагийн сар)-ийг
template дотор суулгаж бие даасан HTML гаргана.
"""
import json, os


def main():
    tpl  = open('./scripts/dashboard_template.html', encoding='utf-8').read()
    data = json.loads(open('./data/dashboard_data.json', encoding='utf-8').read())

    fc_path = './data/forecast_data.json'
    fc = json.loads(open(fc_path, encoding='utf-8').read()) if os.path.exists(fc_path) else None

    data_str = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
    fc_str   = json.dumps(fc,   ensure_ascii=False, separators=(',', ':'))

    out = tpl.replace('__DATA__', data_str).replace('__FORECAST__', fc_str)
    open('./outputs/ub_fire_risk_dashboard.html', 'w', encoding='utf-8').write(out)
    print('✓ outputs/ub_fire_risk_dashboard.html үүслээ:', len(out), 'тэмдэгт')
    if fc:
        mn = fc['meta']['forecast_month']
        yr = fc['meta']['forecast_year']
        print(f'  Прогноз: {yr} оны {mn}-р сар нэмэгдлээ')


if __name__ == '__main__':
    main()
