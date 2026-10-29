import json, sys
C = sys.argv[1]
W = 'C:/Users/anita/projects/maps/trainworld/data/work/realnet'
j = json.load(open(f'{W}/{C}_real.json', encoding='utf-8'))
d = json.load(open(f'C:/Users/anita/projects/maps/artifacts/trainworld/t1750/{C}_dump.json', encoding='utf-8'))
rl = {}
for l in j['lines']:
    rl.setdefault((l['name'], len(l['stops'])), []).append(l)
tot = 0; cutr = 0; sum_req = sum_sol = 0; trainkm_req = trainkm_sol = 0
rows = []
for l in d['lines']:
    dm = l['dm']
    if not dm or l['stat']['status'] != 'running': continue
    c = rl.get((l['name'], len(l['stops'])))
    if not c: continue
    r = c.pop(0)
    req, sol = r['tph'][0], dm['tph'][0]
    km = l['stat']['km']
    tot += dm['ridersPerDay']
    sum_req += req; sum_sol += sol
    trainkm_req += req * km; trainkm_sol += sol * km
    if sol < 0.8 * req:
        cutr += dm['ridersPerDay']
        rows.append((l['name'], len(l['stops']), round(req, 1), round(sol, 1), round(l['stat']['delayS'][0] / 120, 1), dm['ridersPerDay']))
print(f'{C}: lines cut by more than 20% at the high level: {len(rows)} of {len(d["lines"])}, carrying {100*cutr/tot:.0f}% of line riders')
print(f'high-level tph sum requested {sum_req:.0f} solved {sum_sol:.0f} ({sum_sol/sum_req:.2f}); train-km/h requested {trainkm_req:.0f} solved {trainkm_sol:.0f} ({trainkm_sol/trainkm_req:.2f})')
for r in sorted(rows, key=lambda r: -r[5]):
    sys.stdout.buffer.write((str(r) + '\n').encode('utf-8'))
