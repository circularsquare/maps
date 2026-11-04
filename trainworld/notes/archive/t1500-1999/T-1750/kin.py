"""Kinematic run time of a line from its stop distances and stretch speed limits (the sim's traction bands),
to size what a lower top speed or longer dwell does. usage: kin.py <city> [cap_kmh] [dwell_s] [n lines]"""
import json, sys
A1, A2, BR, V1 = 1.0, 0.5, 0.9, 60 / 3.6

def dist_acc(v):  return v * v / (2 * A1) if v <= V1 else V1 * V1 / (2 * A1) + (v * v - V1 * V1) / (2 * A2)
def t_acc(v):     return v / A1 if v <= V1 else V1 / A1 + (v - V1) / A2

def hop_time(D, vmax):
    if dist_acc(vmax) + vmax * vmax / (2 * BR) <= D:
        return t_acc(vmax) + vmax / BR + (D - dist_acc(vmax) - vmax * vmax / (2 * BR)) / vmax
    lo, hi = 0.0, vmax
    for _ in range(60):
        m = (lo + hi) / 2
        if dist_acc(m) + m * m / (2 * BR) > D: hi = m
        else: lo = m
    return t_acc(lo) + lo / BR

def line_time(l, cap=None, dwell=None, level=0):
    st = l['stat']; s = st['stopS'][0]; n = len(s)
    dw = st['dwellS'] if dwell is None else dwell
    t = 0.0
    for k in range(n - 1):
        D = s[k + 1] - s[k]
        lim = st['stretches'][k]['limit']
        if cap: lim = min(lim, cap)
        t += hop_time(D, lim / 3.6)
    return t + dw * (n - 2)

if __name__ == '__main__':
    city = sys.argv[1]
    cap = float(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != '-' else None
    dwell = float(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != '-' else None
    N = int(sys.argv[4]) if len(sys.argv) > 4 else 15
    d = json.load(open(f'C:/Users/anita/projects/maps/artifacts/trainworld/t1750/{city}_dump.json', encoding='utf-8'))
    rows = sorted([l for l in d['lines'] if l['dm'] and l['stat']['status'] == 'running'], key=lambda l: -l['dm']['ridersPerDay'])[:N]
    out = []
    for l in rows:
        low = l['stat']['fromStartByLevel'][2][-1] / 60
        k0 = line_time(l) / 60
        k1 = line_time(l, cap, dwell) / 60
        out.append(f"{l['name']:14s} n={len(l['stops']):2d} game(low,no-delay)={low:6.1f} kin={k0:6.1f} ({k0/low:.2f})  kin(cap={cap},dwell={dwell})={k1:6.1f} ({k1/k0:.2f} of kin)")
    sys.stdout.buffer.write(('\n'.join(out) + '\n').encode('utf-8'))
