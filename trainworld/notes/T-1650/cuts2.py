"""T-1650: parts of places (zone packs) with few market riders crossing, each part at most `cap`
of the network's stations. Greedy: merge the two parts whose crossing is the largest share of the
smaller one's rail, while the merged part's stations stay under the cap.

python cuts2.py <dump.json> <cap fractions, comma> <out.json>
"""
import json, sys, math, collections

dump = json.load(open(sys.argv[1]))
caps = [float(c) for c in sys.argv[2].split(',')]
out = sys.argv[3]
zones = dump['zones']
place = [z[0] for z in zones]
zst = collections.defaultdict(set)
for z, acc in enumerate(dump['access']):
    zst[place[z]].update(acc)
all_st = set().union(*zst.values())
S = len(all_st)

pairs = []  # (place a, place b, rail, rail km)
for key in ('long', 'com', 'ev'):
    for a, b, _, r in dump[key]:
        if b is None or b >= len(zones):
            continue
        km = math.hypot(zones[a][1] - zones[b][1], zones[a][2] - zones[b][2]) / 1000
        pairs.append((place[a], place[b], r, r * km, key))
rail = sum(p[2] for p in pairs)
rkm = sum(p[3] for p in pairs)
agg = collections.defaultdict(lambda: [0., 0.])
for a, b, r, rk, kind in pairs:
    k = (a, b, kind if a != b else 'own')
    agg[k][0] += r
    agg[k][1] += rk
pairs = [(a, b, v[0], v[1], kind) for (a, b, kind), v in agg.items()]
print(f'market rail {rail:.0f}, rail km {rkm:.0f}, stations {S}, places {len(zst)}')

res = {}
for cap in caps:
    part = {p: p for p in zst}
    members = {p: {p} for p in zst}
    sts = {p: set(zst[p]) for p in zst}
    while True:
        own = collections.Counter()
        cr = collections.Counter()
        for a, b, r, _, _ in pairs:
            pa, pb = part[a], part[b]
            if pa == pb:
                own[pa] += r
            else:
                cr[(min(pa, pb), max(pa, pb))] += r
        touch = collections.Counter()
        for (x, y), r in cr.items():
            touch[x] += r
            touch[y] += r
        best, bs = None, 0
        for (x, y), r in cr.items():
            if len(sts[x] | sts[y]) > cap * S:
                continue
            s = r / max(1., min(own[x] + touch[x], own[y] + touch[y]))
            if s > bs:
                best, bs = (x, y), s
        if best is None:
            break
        x, y = best
        for p in members[y]:
            part[p] = x
        members[x] |= members.pop(y)
        sts[x] |= sts.pop(y)
    # report
    own = collections.Counter(); ownkm = collections.Counter(); cross = collections.Counter(); crosskm = collections.Counter()
    xk = collections.Counter()
    for a, b, r, rk, kind in pairs:
        pa, pb = part[a], part[b]
        if pa == pb:
            own[pa] += r; ownkm[pa] += rk
        else:
            cross[pa] += r; cross[pb] += r; crosskm[pa] += rk; crosskm[pb] += rk; xk[kind] += r
    xr = sum(cross.values()) / 2
    xkm = sum(crosskm.values()) / 2
    big = sorted(members, key=lambda p: -len(sts[p]))
    print(f'\ncap {cap:.2f}: {len(members)} parts, crossing {xr:.0f} trips ({100 * xr / rail:.1f}%), {100 * xkm / rkm:.1f}% of rail km; by kind {dict((k, round(v)) for k, v in xk.items())}')
    rows = []
    for p in big:
        n = len(sts[p])
        share = cross[p] / max(1., own[p] + cross[p])
        sharekm = crosskm[p] / max(1., ownkm[p] + crosskm[p])
        rows.append({'places': sorted(members[p]), 'stations': n, 'own': own[p], 'cross': cross[p], 'cross_share': share, 'cross_km_share': sharekm})
        if n >= 0.01 * S:
            print(f'  {n:5d} st ({100 * n / S:4.1f}%)  own {own[p]:8.0f}  crossing {cross[p]:7.0f} ({100 * share:4.1f}% of its rail, {100 * sharekm:4.1f}% of its rail km)  {len(members[p])} places: {",".join(sorted(members[p])[:16])}')
    res[str(cap)] = {'parts': rows, 'crossing': xr, 'crossing_km': xkm, 'rail': rail, 'rail_km': rkm, 'S': S}
json.dump(res, open(out, 'w'))
