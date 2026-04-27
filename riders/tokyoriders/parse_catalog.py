import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

d = json.load(open('cache/estat_catalog.json', encoding='utf-8'))
items = d['GET_DATA_CATALOG']['DATA_CATALOG_LIST_INF']['DATA_CATALOG_INF']
if isinstance(items, dict): items = [items]

for it in items:
    ds = it['DATASET']
    t = ds['TITLE']
    print(f"[{ds['TITLE'].get('SURVEY_DATE','?')}] id={it['@id']}  {t['NAME']}")
    res = it['RESOURCES']['RESOURCE']
    if isinstance(res, dict): res = [res]
    for r in res:
        name = r['TITLE']['NAME']
        flag = '  <-- TIME' if '時間帯' in name else ''
        print(f"    {r['@id']} {r['FORMAT']}  {name}{flag}")
    print()
