"""Scan the taxonomy for nodes that are the same thing at two depths.

Three shapes, and they are not the same problem:

  IDENTITY      a branch with exactly one child, where the child is the branch.
                Latin Catholic / Catholic Church (spec §2.5) is the case that
                started this. Two rows, two colours, one body of people. The fix
                is in the source mapping, not the viewer.

  NEAR-IDENTITY a branch whose own dots plus one child are >99% one of the two —
                i.e. the split exists but nothing is on one side of it.

  THIN SPLIT    a branch whose child is real but tiny against the branch's own
                dots. Not an identity; a candidate for §-other folding.

Counts are the world total across every built country, which is the view the
default legend draws.
"""
import json, collections, sys

ROOT = 'C:/Users/anita/projects/maps/religiondots'
NODES = json.load(open(f'{ROOT}/taxonomy/religions.json', encoding='utf-8'))['nodes']
COUNTS = json.load(open(f'{ROOT}/data/processed/counts.json', encoding='utf-8'))

label = {n['id']: n['label'] for n in NODES}
kids = collections.defaultdict(list)
for n in NODES:
    if '.' in n['id']:
        kids[n['id'].rsplit('.', 1)[0]].append(n['id'])

# own dots, world-wide, and by which countries put them there
own = collections.Counter()
where = collections.defaultdict(set)
for cc, c in COUNTS['countries'].items():
    for nid, v in c['dots'].items():
        own[nid] += v
        where[nid].add(cc)

def total(nid):
    return own[nid] + sum(total(k) for k in kids[nid])

def srcs(nid):
    return ','.join(sorted(where[nid])) or '-'

print('=' * 78)
print('IDENTITIES — one child, and the branch is the child')
print('=' * 78)
for nid in sorted(kids, key=lambda x: -total(x)):
    ks = kids[nid]
    if len(ks) != 1:
        continue
    k = ks[0]
    print(f'\n  {nid}   {total(nid):,} total')
    print(f'    branch own  {own[nid]:>12,}   [{srcs(nid)}]')
    print(f'    {label[k]:<26} {total(k):>12,}   [{srcs(k)}]   {k}')

print()
print('=' * 78)
print('THIN SPLITS — branch has own dots and children, one side under 2%')
print('=' * 78)
rows = []
for nid in sorted(kids, key=lambda x: -total(x)):
    ks = kids[nid]
    if len(ks) < 1 or not own[nid]:
        continue
    t = total(nid)
    if not t:
        continue
    kid_tot = sum(total(k) for k in ks)
    small = min(own[nid], kid_tot) / t
    if small < 0.02:
        rows.append((small, nid, t, own[nid], kid_tot, ks))
for small, nid, t, o, kt, ks in sorted(rows, key=lambda r: -r[2]):
    side = 'branch own' if o < kt else 'all children'
    print(f'\n  {nid}   {t:,} total   minority {side} {small:.4%}')
    print(f'    unspecified (branch own)   {o:>12,}   [{srcs(nid)}]')
    for k in sorted(ks, key=lambda x: -total(x)):
        print(f'    {label[k]:<26} {total(k):>12,}   [{srcs(k)}]')
