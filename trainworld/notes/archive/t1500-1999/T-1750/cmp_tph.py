import json, sys
C = sys.argv[1]
j = json.load(open('C:/Users/anita/projects/maps/trainworld/data/work/realnet/'+C+'_real.json', encoding='utf-8'))
meta = json.load(open('C:/Users/anita/projects/maps/trainworld/data/work/realnet/tyo_routes.json', encoding='utf-8'))
d = json.load(open('C:/Users/anita/projects/maps/artifacts/trainworld/t1750/'+C+'_dump.json', encoding='utf-8'))
by = {}
for l in j['lines']:
    by.setdefault(l['name'], []).append(l)
out = []
for l in sorted(d['lines'], key=lambda l: -(l['dm']['ridersPerDay'] if l['dm'] else 0))[:40]:
    c = by.get(l['name'], [])
    best = min(c, key=lambda x: abs(len(x['stops']) - len(l['stops']))) if c else None
    pk = meta.get(best['route'], {}).get('peak') if best else None
    out.append(f"{l['name']} n={len(l['stops'])} save tph={[round(x,1) for x in l['saveTph'].values()] if isinstance(l['saveTph'], dict) else l['saveTph']} solved={[round(x,1) for x in l['dm']['tph']]} | json n={len(best['stops']) if best else None} tph={[round(x,1) for x in best['tph']] if best else None} peak(csv)={pk} cars={best['cars'] if best else None} trips_day={best['trips_day'] if best else None}")
sys.stdout.buffer.write(('\n'.join(out) + '\n').encode('utf-8'))
