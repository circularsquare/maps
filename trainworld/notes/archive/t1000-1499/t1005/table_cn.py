"""T-1005: the China and Taiwan retrofit table for the note."""
import os, sys, json, collections
sys.stdout.reconfigure(encoding='utf-8')
D = os.path.dirname(os.path.abspath(__file__))
TW = 'C:/Users/anita/projects/maps/trainworld'
plan = {r['id']: r for r in json.load(open(f'{TW}/pipeline/world_plan.json', encoding='utf-8'))['metros']}
names = json.load(open(f'{TW}/app/src/game/zone-names.json', encoding='utf-8'))
C = json.load(open(f'{D}/candidates.json', encoding='utf-8'))
def nm(i):
    v = names.get(i)
    return v['name'] if isinstance(v, dict) and v.get('name') else plan[i]['name']
M = lambda x: f'{x / 1e6:.2f}M'
S2 = {tuple(s['rows']): s for iso in ('CHN',) for s in json.load(open(f'{D}/split2_{iso}.json', encoding='utf-8'))}
RF = [r for iso in ('CHN', 'TWN') for r in json.load(open(f'{D}/retrofit_{iso}.json', encoding='utf-8'))]
rows_out = []
cnt = collections.Counter(); rebuilt = collections.defaultdict(list)
for r in RF:
    key = tuple(r['rows'])
    pair = [c for c in C if c['a'] in key and c['b'] in key]
    hb = '; '.join(f"{c['in_hull']:.2f}/{c['border']:.2f}" for c in pair)
    now = ' + '.join(f"{nm(z)} {z} {M(r['before'][z])}" for z in key)
    sr = plan[key[0]].get('solve_region') or 'taiwan'
    st = '/'.join(sorted({plan[z].get('status') for z in key}))
    s = S2.get(key)
    if r['how'] == 'merge':
        n = r['new'][0]
        out = f"merge: {M(n['people'])}, {n['km2']:,} km²"; v = 'merge'
    elif s and not s.get('skipped') and not s.get('none'):
        gain = s['new_min_fill'] - s['old_min_fill']
        new = ' / '.join(f"{M(g['people'])} {', '.join(g['units'][:4])}{', ...' if len(g['units']) > 4 else ''}" for g in s['new'])
        if gain < 0.05:
            out = f"leave: no better split (fill {s['old_min_fill']:.2f} to {s['new_min_fill']:.2f})"; v = 'leave'
        else:
            out = f"split: {new} (fill {s['old_min_fill']:.2f} to {s['new_min_fill']:.2f})"; v = 'split'
    else:
        new = ' / '.join(f"{M(n['people'])} {', '.join(n['units'][:4])}{', ...' if len(n['units']) > 4 else ''}" for n in r['new'])
        out = f"regroup ({len(key)} rows to {len(r['new'])}, the planner with the guard): {new}"; v = 'regroup'
        if key == ('cea', 'smz'):
            out += '; 74 units, by hand'
    cnt[v] += 1
    if v != 'leave':
        rebuilt[sr] += list(key)
    rows_out.append((sr, f"| {now} | {sr}, {st} | {hb} | {out} |"))
print('| Rows now | Solve region, status | In hull / border | The rule gives |')
print('|---|---|---|---|')
for sr, line in sorted(rows_out):
    print(line)
print()
print(dict(cnt), 'rows rebuilt', sum(len(v) for v in rebuilt.values()))
print({k: len(v) for k, v in sorted(rebuilt.items())})
st = collections.Counter(plan[z].get('status') for v in rebuilt.values() for z in v)
print(dict(st))
