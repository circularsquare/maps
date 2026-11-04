"""T-1222 stage 0: bias, noise, memory and time per recipe, from the JSON lines of
`everyday::lod_tests::t1222_probe` (notes/T-1206.md section 9): python analyse.py <jsonl globs>.

Bias: per zone, the recipe's rail (commutes and everyday, a day) averaged over its draws against
fine's averaged over its own draws (fine's draw 0 is the game's, but one draw of fine has noise of
its own as big as the bound). Noise: each draw against fine's mean over draws, the recipe's beside
fine's own draws'."""
import json, sys, glob, math, statistics as st, collections

rows = []
for p in sys.argv[1:]:
    for f in glob.glob(p):
        rows += [json.loads(l) for l in open(f)]
METROS = {'wuh', 'chg', 'gua', 'sha'}


def kind(r):
    if r['zone'] == 'nyc':
        return 'nyc'
    if r['zone'] in METROS:
        return 'metro'
    return 'town' if r['dense_share'] >= 0.15 else 'country'


def rail(r):
    per = r['figures']['periods']
    return sum(p[1] + p[4] for p in per)


def ev_rail(r):
    return sum(p[4] for p in r['figures']['periods'])


def trips(r):
    return sum(p[0] + p[3] for p in r['figures']['periods'])


runs = collections.defaultdict(list)
for r in rows:
    runs[(r['zone'], r['recipe'])].append(r)
zones = sorted({r['zone'] for r in rows if r['recipe'] == 'fine'}, key=lambda z: (['metro', 'town', 'country', 'nyc'].index(kind(runs[(z, 'fine')][0])), z))
print('zones:', ', '.join(f"{z} {kind(runs[(z, 'fine')][0])} {runs[(z, 'fine')][0]['pop'] / 1e6:.2f}M dense {runs[(z, 'fine')][0]['dense_share']:.2f} ({len(runs[(z, 'fine')])} fine draws)" for z in zones))
groups = ['metro', 'town', 'country', 'nyc']
fine_mean = {}
for z in zones:
    fr = runs[(z, 'fine')]
    n = len(fr)
    st_mean = [sum(v) / n for v in zip(*(r['figures']['stations'] for r in fr))]
    fine_mean[z] = dict(rail=st.mean(rail(r) for r in fr), rail_sd=st.stdev(rail(r) for r in fr) if n > 1 else 0, n=n,
                        ev=st.mean(ev_rail(r) for r in fr), share=st.mean(rail(r) / trips(r) for r in fr), stations=st_mean)


def fmt(v, se):
    return f'{100 * v:+.3f}% (se {100 * se:.3f})'


recipes = list(dict.fromkeys(r['recipe'] for r in rows))
for rec in recipes:
    print(f'\n== {rec}')
    by = collections.defaultdict(list)
    noise = collections.defaultdict(lambda: [0.0, 0.0, 0, 0])
    for z in zones:
        rs = runs.get((z, rec), [])
        f = fine_mean[z]
        if not rs or f['n'] < 2:
            continue
        c = [rail(r) for r in rs]
        cm = st.mean(c)
        b = cm / f['rail'] - 1
        var_c = (st.variance(c) / len(c)) if len(c) > 1 else 0
        se = math.sqrt(var_c / f['rail'] ** 2 + (f['rail_sd'] ** 2 / f['n']) * cm ** 2 / f['rail'] ** 4)
        ev = st.mean(ev_rail(r) for r in rs) / f['ev'] - 1
        k = kind(rs[0])
        by[k].append((b, se, ev))
        if k != 'nyc':
            by['china'].append((b, se, ev))
        for r in rs:
            share = abs(100 * (rail(r) / trips(r) - f['share']))
            big = [(a, x) for a, x in zip(f['stations'], r['figures']['stations']) if a >= 1000]
            dev = [abs(x / a - 1) for a, x in big]
            w = noise[k]
            noise[k] = [max(w[0], share), max([w[1]] + dev), w[2] + sum(d > 0.1 for d in dev), w[3] + len(big)]
        print(f'  {z:4} {k:7} rail {fmt(b, se)}  everyday {100 * ev:+.2f}%  per draw sd {100 * (st.stdev(c) / cm if len(c) > 1 else 0):.2f}%  ({len(rs)} draws)')
    for k in groups + ['china']:
        v = by[k]
        if v:
            mb = st.mean(x[0] for x in v)
            se = math.sqrt(sum(x[1] ** 2 for x in v)) / len(v)
            print(f'    {k:8} bias {fmt(mb, se)} over {len(v)} zones; everyday {100 * st.mean(x[2] for x in v):+.3f}%')
    # the network's total: the China zones' rail summed, recipe draws' mean against fine's
    cz = [z for z in zones if z != 'nyc' and runs.get((z, rec))]
    if cz:
        tot_c = sum(st.mean(rail(r) for r in runs[(z, rec)]) for z in cz)
        tot_f = sum(fine_mean[z]['rail'] for z in cz)
        print(f'    China total rail {100 * (tot_c / tot_f - 1):+.3f}% over {len(cz)} zones')
    print('  noise against fine\'s mean: worst train share points, worst station 1000+, stations past 10% / station-draws')
    for k in groups:
        if k in noise:
            w = noise[k]
            print(f'    {k:8} {w[0]:.2f} pt, {100 * w[1]:.1f}%, {w[2]} / {w[3]}')

print('\n== memory (MB natively, held after the day / peak) and time (s a day, everyday opening ms), salt 0')
for z in zones:
    f = runs[(z, 'fine')][0]
    line = f"  {z:4} {kind(f):7} fine {f['mb']['held']:6.1f} / {f['mb']['peak']:6.1f}, {f['ms']['day'] / 1000:6.2f} s, {f['ms']['everyday_open']:5.0f} ms"
    for rec in recipes[1:]:
        c = runs.get((z, rec))
        if c:
            c = c[0]
            line += f" | {rec} {c['mb']['held']:5.1f} / {c['mb']['peak']:5.1f} {c['ms']['day'] / 1000:5.2f}s"
    print(line)
if any('parts' in r for r in rows):
    print('\n== held by part (MB, salt 0)')
    for z in zones:
        for rec in recipes:
            c = runs.get((z, rec))
            if c and 'parts' in c[0]:
                print(f'  {z:4} {rec:7}', ', '.join(f'{k} {v:.1f}' for k, v in c[0]['parts'].items()))
