"""Variants of the T-509 track-model network JSON (data/work/realnet/<city>_real_track.json) for the native calib bin.
G  = the game's service as solved (tph after capacity cuts, run times with the capacity holds baked in, cars)
X  = G with the flat-crossing cuts removed: tph the requested ones (<city>_real.json), run times the delay-free (low level) ones
C15 = cars x1.5 (crush 160 -> 240 a car)
usage: make_variants.py <city>"""
import json, sys, copy, os
city = sys.argv[1]
W = 'C:/Users/anita/projects/maps/trainworld/data/work/realnet'
S = 'C:/Users/anita/AppData/Local/Temp/claude/c--Users-anita-projects-maps/f9a7f88d-0531-4963-9df6-6771326e513f/scratchpad/t1750/var'
os.makedirs(S, exist_ok=True)
trk = json.load(open(f'{W}/{city}_real_track.json', encoding='utf-8'))
req = json.load(open(f'{W}/{city}_real.json', encoding='utf-8'))
dump = json.load(open(f'C:/Users/anita/projects/maps/artifacts/trainworld/t1750/{city}_dump.json', encoding='utf-8'))
d = trk['demand']
print('track lines', len(trk['lines']), 'line_n', len(d['line_n']), 'req lines', len(req['lines']), 'dump lines', len(dump['lines']))
# dump lines by (name, n), in order
dl = {}
for l in dump['lines']:
    if l['dm'] and l['stat']['status'] == 'running':
        dl.setdefault((l['name'], len(l['stops'])), []).append(l)
rl = {}
for l in req['lines']:
    rl.setdefault((l['name'], len(l['stops'])), []).append(l)
offs = [0]
for n in d['line_n']: offs.append(offs[-1] + n)
def hops(fs):
    return [fs[j + 1] - fs[j] for j in range(len(fs) - 1)]
variants = {k: copy.deepcopy(d) for k in ('G', 'X', 'R')}
matched = 0
for li, meta in enumerate(trk['lines']):
    n = d['line_n'][li]
    key = (meta['name'], n)
    if not dl.get(key) or not rl.get(key):
        continue
    g = dl[key].pop(0); r = rl[key].pop(0)
    matched += 1
    for vname, use_low in (('G', False), ('X', True)):
        v = variants[vname]
        for lev in range(3):
            fs = g['stat']['fromStartByLevel'][2 if use_low else lev]
            h = hops(fs)
            for run in range(2):
                for j in range(n - 1):
                    k = j if run == 0 else n - 1 - j
                    hop = h[j] if run == 0 else h[n - 2 - j]
                    v['times'][6 * offs[li] + (run * 3 + lev) * n + k] = hop
        if vname == 'G':
            for lev in range(3): v['tph'][3 * li + lev] = g['dm']['tph'][lev]
        else:
            for lev in range(3): v['tph'][3 * li + lev] = r['tph'][lev]
        v['cars'][li] = g['dm']['cars']
    # R: the game's service as solved, run times from the real timetable (NYC: GTFS medians; Tokyo: the synthetic ones)
    v = variants['R']
    for lev in range(3):
        v['tph'][3 * li + lev] = g['dm']['tph'][lev]
        for run in range(2):
            src = r['t0'] if run == 0 else r['t1']
            for k in range(n):
                v['times'][6 * offs[li] + (run * 3 + lev) * n + k] = src[lev][k]
    v['cars'][li] = g['dm']['cars']
print('matched', matched, 'of', len(trk['lines']))
for vname, v in variants.items():
    for cm in (1.0, 1.5):
        w = copy.deepcopy(trk); w['demand'] = copy.deepcopy(v)
        if cm != 1.0: w['demand']['cars'] = [min(20.0, round(c * cm)) if False else c * cm for c in w['demand']['cars']]
        json.dump(w, open(f'{S}/{city}_{vname}{"_c15" if cm != 1.0 else ""}.json', 'w'))
# the unmodified base as well, cars x1.5
b = copy.deepcopy(trk); json.dump(b, open(f'{S}/{city}_B.json', 'w'))
b['demand']['cars'] = [c * 1.5 for c in b['demand']['cars']]; json.dump(b, open(f'{S}/{city}_B_c15.json', 'w'))
