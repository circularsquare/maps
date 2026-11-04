import json, sys
C = sys.argv[1]
want = sys.argv[2:]   # "id:label"
W = 'C:/Users/anita/projects/maps/trainworld/data/work/realnet'
d = json.load(open(f'C:/Users/anita/projects/maps/artifacts/trainworld/t1750/{C}_dump.json', encoding='utf-8'))
j = json.load(open(f'{W}/{C}_real.json', encoding='utf-8'))
rl = {}
for l in j['lines']:
    rl.setdefault((l['name'], len(l['stops'])), []).append(l)
byid = {}
for l in d['lines']:
    if not l['dm']: continue
    c = rl.get((l['name'], len(l['stops'])))
    r = c.pop(0) if c else None
    byid[str(l['id'])] = (l, r)
out = []
for w in want:
    i, lab = w.split(':', 1)
    l, r = byid[i]
    st = l['stat']; dm = l['dm']; n = len(l['stops'])
    fs = st['fromStartByLevel']
    hi, lo = fs[0][-1] / 60, fs[2][-1] / 60
    gt = sum(r['t0'][0]) / 60 if r else 0
    km = st['km']
    out.append(f"| {lab} | {n} | {km:.1f} | {hi:.0f} | {lo:.0f} | {km/(hi/60):.0f} | {max(s['reached'] for s in st['stretches']):.0f} | {st['dwellS']*(n-2)/fs[0][-1]*100:.0f}% | {r['tph'][0] if r else 0:.1f} | {dm['tph'][0]:.1f} | {dm['cars']} | {gt:.0f} |")
sys.stdout.buffer.write(('\n'.join(out) + '\n').encode('utf-8'))
