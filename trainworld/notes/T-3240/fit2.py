"""T-3254 second grid: RU and TR without the pull (their no-pull trips already over the published), CN apart; pairs at or
under max(cap, no pull); 300 km+ at or under the ceilings for DE PL FR ES IT; 100-300 km at most 10% up in DE FR.
Score: sum of log(model / cap)^2 over the fitted pairs outside China."""
import itertools, math
from sc2 import scenario, pair, nat, T, OVER, FIT, NAT

GAM = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
DAMP = [0., 0.15, 0.3, 0.45, 0.6, 0.8]
PH = [(300, 500), (300, 600), (300, 700)]
CG = {"RU": 0., "TR": 0.}
SKIP = {"mos", "ist", "bei", "jnj"}
pairs = [p for p in OVER + FIT if p[0] not in SKIP]
base = {(x, y): pair(T, x, y) for x, y, _ in pairs}
nat0 = {c: nat(T, c) for c in NAT}
out = []
for gam, dmp, ph in itertools.product(GAM, DAMP, PH):
    M = scenario(gamma=gam, phase=list(ph), damp=dmp, cg=CG)
    viol, score, fit = [], 0., {}
    for x, y, cap in pairs:
        m = pair(M, x, y)
        fit[x + "-" + y[:3]] = m
        lim = max(cap, base[(x, y)])
        if m > lim * 1.001:
            viol.append(f"{x}-{y[:3]} {m / lim:.2f}")
    for x, y, cap in FIT:
        if x not in SKIP:
            score += math.log(fit[x + "-" + y[:3]] / cap) ** 2
    nn = {}
    for c in ["DE", "PL", "FR", "ES", "IT"]:
        m1, m3, _ = nat(M, c)
        b1, b3, _ = nat0[c]
        nn[c] = (round(m1, 4), round(m3, 4))
        c3 = NAT[c][1]
        if m3 > max(c3, b3) * 1.001:
            viol.append(f"{c}300 {m3 / c3:.2f}")
        if c in ("DE", "FR") and m1 > b1 * 1.10:
            viol.append(f"{c}100 {m1 / b1:.2f}")
    out.append((len(viol), score, gam, dmp, ph, viol, fit, nn))
    print(f"g {gam} damp {dmp} ph {ph}: score {score:.2f}, {len(viol)} over: {' '.join(viol)}", flush=True)
print("\nbest:")
for r in sorted(out, key=lambda r: (r[0], r[1]))[:12]:
    print(r[:6])
    print("   ", {k: round(v) for k, v in r[6].items()}, r[7])


