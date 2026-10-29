import json, sys
city = sys.argv[1]
j = json.load(open(f'C:/Users/anita/projects/maps/trainworld/data/work/realnet/{city}_real.json', encoding='utf-8'))
d = json.load(open(f'C:/Users/anita/projects/maps/artifacts/trainworld/t1750/{city}_dump.json', encoding='utf-8'))
by = {}
for l in j['lines']:
    by.setdefault(l['name'], []).append(l)
out = []
tot_j = tot_g = 0
for l in sorted(d['lines'], key=lambda l: -(l['dm']['ridersPerDay'] if l['dm'] else 0))[:int(sys.argv[2])]:
    c = by.get(l['name'], [])
    if not c: continue
    best = min(c, key=lambda x: abs(len(x['stops']) - len(l['stops'])))
    if len(best['stops']) != len(l['stops']): continue
    tj = [sum(best['t0'][k]) / 60 for k in range(3)]
    tg = [x[-1] / 60 for x in l['stat']['fromStartByLevel']]
    dl = [x / 120 for x in l['stat']['delayS']]  # half the round-trip delay, min (one way)
    out.append(f"{l['name']:12s} n={len(l['stops'])} json_min={[round(x,1) for x in tj]} game_min={[round(x,1) for x in tg]} game/json(high)={tg[0]/tj[0]:.2f} oneway_delay_min={[round(x,1) for x in dl]}")
    tot_j += tj[0]; tot_g += tg[0]
out.append(f'total json {tot_j:.0f} game {tot_g:.0f} ratio {tot_g/tot_j:.2f}')
sys.stdout.buffer.write(('\n'.join(out) + '\n').encode('utf-8'))
