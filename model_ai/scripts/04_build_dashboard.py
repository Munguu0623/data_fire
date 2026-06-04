"""
04 — Дашбордын HTML үүсгэх
==========================
dashboard_data.json-ийг template дотор суулгаж бие даасан HTML гаргана.
"""
import json

def main():
    tpl = open('./scripts/dashboard_template.html', encoding='utf-8').read()
    data = open('./data/dashboard_data.json', encoding='utf-8').read()
    # compact
    data = json.dumps(json.loads(data), ensure_ascii=False, separators=(',', ':'))
    out = tpl.replace('__DATA__', data)
    open('./outputs/ub_fire_risk_dashboard.html', 'w', encoding='utf-8').write(out)
    print('✓ outputs/ub_fire_risk_dashboard.html үүслээ:', len(out), 'тэмдэгт')

if __name__ == '__main__':
    main()
