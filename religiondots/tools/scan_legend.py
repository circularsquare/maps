"""What the default legend actually draws, branch by branch, with world dots.

Default view: all countries, scope null, depth 2, `unaffiliated` hidden. That
draws every root's children, plus any root or branch carrying dots of its own
(spec §6.6, the `unspecified` row).

Prints, for each root: its level-2 rows in the order the panel sorts them, with
each row's share of the ROOT's total — which is the denominator spec-side folding
should use, per Anita: "if we're looking at judaism, relative to just total jew
count".
"""
import json, collections

ROOT = 'C:/Users/anita/projects/maps/religiondots'
NODES = json.load(open(f'{ROOT}/taxonomy/religions.json', encoding='utf-8'))['nodes']
COUNTS = json.load(open(f'{ROOT}/data/processed/counts.json', encoding='utf-8'))

label = {n['id']: n['label'] for n in NODES}
kids = collections.defaultdict(list)
order = {n['id']: i for i, n in enumerate(NODES)}
for n in NODES:
    if '.' in n['id']:
        kids[n['id'].rsplit('.', 1)[0]].append(n['id'])
roots = [n['id'] for n in NODES if '.' not in n['id']]

own, ring = collections.Counter(), collections.Counter()
for c in COUNTS['countries'].values():
    for nid, v in c['dots'].items():
        own[nid] += v
    for nid in c.get('rings', {}):
        ring[nid] += 1

def total(nid):
    return own[nid] + sum(total(k) for k in kids[nid])

def present(nid):                      # spec §6.2 presence pruning
    return total(nid) > 0 or ring[nid] or any(present(k) for k in kids[nid])

THRESH = 1e-4
for r in sorted(roots, key=lambda x: -total(x)):
    if r == 'unaffiliated' or not present(r):
        continue
    t = total(r)
    ks = [k for k in kids[r] if present(k)]
    if not ks and not own[r]:
        continue
    print(f'\n{"="*74}\n{label[r].upper()}   {t:,} dots   ({len(ks)} branch rows)')
    rows = []
    if own[r] and ks:
        rows.append(('unspecified', own[r], r, True))
    rows += [(label[k], total(k), k, False) for k in ks]
    tiny = []
    for lab, v, nid, is_unspec in rows:
        share = v / t if t else 0
        flag = ''
        if t and share < THRESH:
            flag = '  <-- under 1e-4'
            tiny.append((lab, v, nid))
        rr = f' ring:{ring[nid]}' if ring[nid] and not v else ''
        print(f'   {lab:<34} {v:>9,}  {share:>8.4%}{rr}{flag}')
    if len(tiny) >= 2:
        s = sum(v for _, v, _ in tiny)
        print(f'   >>> {len(tiny)} rows under 1e-4, {s:,} dots ({s/t:.4%}) — foldable')
