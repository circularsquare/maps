"""T-2621: each coarse mode against the exact pass: time per pass by part, and boardings by line.

python cmp.py <dump.json> [<t756 lines json with every population's lineBoard>]

The dump is t2621.rs's. Lines are compared over the periods dumped (added up); "market" is the
market's boardings, "total" adds every other population's from the lines file (its lineBoard less
its own market share is not known, so the total is the file's lineBoard, a whole day of everyone).
"""
import json, sys, math

d = json.load(open(sys.argv[1]))
tot = json.load(open(sys.argv[2]))['lineBoard'] if len(sys.argv) > 2 else None
modes = d['modes']
ex = modes[0]
assert ex['mode'] in ('exact', 'searched')


def lines(m, key='line_all'):
    out = None
    for p in m['periods']:
        v = p[key]
        out = list(v) if out is None else [a + b for a, b in zip(out, v)]
    return out


def tsum(m):
    keys = {'solve': None}
    t = [0.0] * 24
    solve = 0.0
    for p in m['periods']:
        solve += p['solve_ms']
        for i, x in enumerate(p['tm']):
            t[i] += x
    return solve, t


# pairs by distance (exact pass's rail, the first period dumped)
pairs = d['pairs']
pr = ex['periods'][0]['pair_rail']
bands = [0, 25, 50, 100, 150, 250, 500, 1000, 1e9]
cnt = [0] * (len(bands) - 1)
rail = [0.0] * (len(bands) - 1)
for (a, b, km), r in zip(pairs, pr):
    i = next(i for i in range(len(bands) - 1) if km < bands[i + 1])
    cnt[i] += 1
    rail[i] += r
print(f"{d['net']}: {d['stations']} stations, {len(pairs)} longer-trip pairs")
print('  km band       pairs      rail trips (period %d)' % ex['periods'][0]['q'])
for i in range(len(cnt)):
    print(f'  {bands[i]:>5.0f}-{bands[i+1]:<6.0f} {cnt[i]:9d} ({100*cnt[i]/max(1,len(pairs)):4.1f}%) {rail[i]:10.0f} ({100*rail[i]/max(1e-9,sum(rail)):4.1f}%)')

ref = lines(ex)
refl = [a - c - e for a, c, e in zip(ref, lines(ex, 'line_com'), lines(ex, 'line_ev'))]
s0, t0 = tsum(ex)
print(f"\n{'mode':52s} {'solve s':>8s} {'x':>5s}  rail  | market boardings by line: lines >=1000 within 2%/5%/10%, worst | total riders: worst, lines past 1%/2%")
for m in modes:
    s, t = tsum(m)
    v = lines(m)
    r0 = sum(p['rail'] + p['c_rail'] + p['e_rail'] for p in ex['periods'])
    r1 = sum(p['rail'] + p['c_rail'] + p['e_rail'] for p in m['periods'])
    big = [(l, a, b) for l, (a, b) in enumerate(zip(ref, v)) if a >= 1000]
    rel = [abs(b - a) / a for _, a, b in big]
    w = max(big, key=lambda x: abs(x[2] - x[1]) / x[1]) if big else None
    line = f"{m['mode']:52s} {s/1e3:8.2f} {s0/max(s,1):5.2f} {100*(r1-r0)/r0:+5.1f}% | {sum(x<=.02 for x in rel)}/{sum(x<=.05 for x in rel)}/{sum(x<=.1 for x in rel)} of {len(rel)}, worst line {w[0]} {w[1]:.0f}->{w[2]:.0f} ({100*(w[2]-w[1])/w[1]:+.1f}%)" if big else ''
    if tot:
        # the dump's periods are a part of the day: scale the change to the day by the exact market's share
        days = [abs(b - a) for a, b in zip(ref, v)]
        k = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
        rels = [(dd * k) / tt for dd, tt in zip(days, tot) if tt >= 1000]
        line += f" | worst {100*max(rels):.1f}%, past 1% {sum(x>.01 for x in rels)}, past 2% {sum(x>.02 for x in rels)} of {len(rels)}"
    print(line)
    print('    ms: skims %.0f, searches %.0f (%d full, %d bounded, %.1fM pops), longer %.0f (set %.0f logsum %.0f choice %.0f spread %.0f; far %d near %d), commutes %.0f, everyday %.0f, assign %.0f, again %.0f, needs %.0f, hubs %d' % (
        t[0], t[1], t[10], t[11], t[12] / 1e6, t[3], t[16], t[17], t[18], t[19], t[13], t[14], t[4], t[5], t[6] - t[7], t[7], t[8], m.get('hubs', 0)))
