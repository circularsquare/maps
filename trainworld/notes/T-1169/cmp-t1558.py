import json, sys, collections
S = sys.argv[1].replace(chr(92), '/').rstrip('/') + '/'  # the folder holding before-<save>.json and after-<save>.json (artifacts/trainworld/t1558)
MB = 1 << 20
for k in ['nyc', 'china', 'japan']:
    b = json.load(open(S + f'before-{k}.json'))
    a = json.load(open(S + f'after-{k}.json'))
    nb, na = b['nums'], a['nums']
    same = len(nb) == len(na) and all(x == y for x, y in zip(nb, na))
    print(k, 'numbers', len(nb), len(na), 'identical:', same, 'hash', b['sig']['hash'], a['sig']['hash'], 'modes', b.get('modes') == a.get('modes'), 'secs', b['finalSecs'], a['finalSecs'])
    for nm, d in (('before', b), ('after', a)):
        for m in d['memory']:
            if not m['aux']:
                print('   ', nm, 'worker', m['worker'], 'held MB', m['heap'], 'peak', m['solvePeak'], 'wasm', m['wasm'])
        # tag breakdown for the first solving worker with most periods
    def tags(d):
        best = None
        for rep in d['diag']['full'].values():
            if rep['kind'] == 'done' and not rep['aux'] and (best is None or rep['live'] > best['live']):
                best = rep
        return best
    tb, ta = tags(b), tags(a)
    def cat(rep):
        c = collections.Counter(); p = collections.Counter()
        for name, live, mx, ap in rep['tags']:
            base = name.split('@')[0]
            c[base] += live; p[base] += ap
        return c, p
    cb, pb = cat(tb); ca, pa = cat(ta)
    print('   worker with most live: before', tb['mine'], round(tb['live'] / MB), round(tb['peak'] / MB), 'after', ta['mine'], round(ta['live'] / MB), round(ta['peak'] / MB))
    for key in sorted(set(cb) | set(ca), key=lambda x: -max(cb[x], ca[x], pb[x], pa[x])):
        if max(cb[key], ca[key], pb[key], pa[key]) < 5 * MB: continue
        if abs(cb[key] - ca[key]) < MB and abs(pb[key] - pa[key]) < MB: continue
        print(f'      {key:42} held {cb[key]/MB:7.1f} -> {ca[key]/MB:7.1f}   at peak {pb[key]/MB:7.1f} -> {pa[key]/MB:7.1f}')
