"""T-1005: the rule over every registered zone pair. Writes candidates.json and prints a table.
    classify.py [hull] [border] [fill]"""
import sys, json, os
sys.stdout.reconfigure(encoding='utf-8')
D = os.path.dirname(os.path.abspath(__file__))
TW = 'C:/Users/anita/projects/maps/trainworld'
Z = json.load(open(f'{D}/zones.json', encoding='utf-8'))
P = json.load(open(f'{D}/pairs.json', encoding='utf-8'))
plan = {r['id']: r for r in json.load(open(f'{TW}/pipeline/world_plan.json', encoding='utf-8'))['metros']}
names = json.load(open(f'{TW}/app/src/game/zone-names.json', encoding='utf-8'))
HULL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
BORDER = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
FILL = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
EUROPE_CAP = 1.5e6
CAPS = {'CN': 2.5e6, 'TW': 2.5e6}

def nm(i):
    v = names.get(i)
    return v['name'] if isinstance(v, dict) and v.get('name') else Z[i]['name']

def nunits(i):
    r = plan.get(i)
    if not r or not r.get('units'):
        return None
    return sum(len(v) for v in r['units'].values())

def stands(z):
    if z['country'] in CAPS:
        return z['pop'] >= 1e6
    return z['pop'] / 5e5 + z['area_km2'] / 1e4 >= 1

out = []
for p in P:
    a, b = Z[p['a']], Z[p['b']]
    if not (p['b_in_a_hull'] >= HULL and p['share_of_b_land'] >= BORDER):
        continue
    cap = CAPS.get(a['country'], EUROPE_CAP)
    merged = (a['pop'] or 0) + (b['pop'] or 0)
    shape = 'horseshoe' if p['b_fills_a_concavity'] >= FILL else 'chain'
    if a['country'] != b['country']:
        verdict = 'leave: a national border'
    elif a['kind'] == 'city' and b['kind'] == 'city':
        verdict = 'leave: two metros'
    elif b['kind'] == 'city':
        verdict = 'leave: wraps a metro' if (nunits(a['id']) or 1) == 1 else 'regroup the wrapper: wraps a metro'
    elif a['kind'] == 'city':
        verdict = 'leave: a pocket in a metro (the metro could take it)'
    elif shape == 'chain':
        verdict = 'split the wrapper: it wraps more than this zone'
    elif merged <= cap:
        verdict = 'merge'
    else:
        verdict = f'regroup the two: merged {merged / 1e6:.2f}M is over the cap'
    out.append(dict(a=a['id'], a_name=nm(a['id']), a_kind=a['kind'], a_units=nunits(a['id']), a_pop=a['pop'], a_km2=round(a['area_km2']),
                    a_solidity=round(a['solidity'], 2), b=b['id'], b_name=nm(b['id']), b_kind=b['kind'], b_units=nunits(b['id']),
                    b_pop=b['pop'], b_km2=round(b['area_km2']), country=a['country'],
                    in_hull=round(p['b_in_a_hull'], 2), border=round(p['share_of_b_land'], 2), fill=round(p['b_fills_a_concavity'], 2),
                    shape=shape, merged_pop=merged, merged_km2=round(a['area_km2'] + b['area_km2']), cap=cap,
                    a_stands=stands(a), b_stands=stands(b), verdict=verdict,
                    a_status=plan.get(a['id'], {}).get('status'), b_status=plan.get(b['id'], {}).get('status')))
json.dump(out, open(f'{D}/candidates.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
from collections import Counter
print(len(out), 'wrap pairs;', Counter(o['verdict'].split(':')[0] for o in out))
order = ['merge', 'regroup the two', 'regroup the wrapper', 'split the wrapper', 'leave']
for v in order:
    sel = [o for o in out if o['verdict'].startswith(v)]
    print(f'\n## {v} ({len(sel)})')
    for o in sorted(sel, key=lambda o: (o['country'], -o['in_hull'] * o['border'])):
        print(f"{o['country']} {o['a']} {o['a_name']} ({o['a_kind']}, {o['a_units']}u, {o['a_pop'] / 1e6:.2f}M, {o['a_km2']:,} km2) "
              f"wraps {o['b']} {o['b_name']} ({o['b_kind']}, {o['b_units']}u, {o['b_pop'] / 1e6:.2f}M, {o['b_km2']:,} km2): "
              f"hull {o['in_hull']}, border {o['border']}, fill {o['fill']}; merged {o['merged_pop'] / 1e6:.2f}M"
              f"{'' if o['b_stands'] else ', B under the size rule'}; {o['verdict']}")
