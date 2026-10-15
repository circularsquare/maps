"""T-362: a two-part kernel (the NHTS ground curve plus a long-haul tail with destinations weighted
towards dense cells), checked against NHTS all-mode bands and flights segment pairs.
Kernel: f(d) = on(d) * [ max(d,25)^-A + k * max(d,D1)^-G * w ]  where w = (cellpop/1e5)^TH on the tail only."""
import json, math, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "3")
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from world_gravity import METROS, unit, R_KM, load_r5

HERE = Path(__file__).resolve().parents[3] / "artifacts/trainworld/t362"
EDGES = np.array([25, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250, 300, 400, 500, 700, 1000, 1500, 2500, 5000, 20040.0])
NH = json.load(open(r"C:\Users\anita\projects\maps\trainworld\research\intercity\nhts_long.json"))
e = NH["edges_km"]
nh_all = np.array(NH["northeast"]["noncommute"]["rate"])[e.index(25):]  # bands 25..5000
LB = np.exp(np.linspace(np.log(25), np.log(20040), 400))  # log bins for distance histograms


def dist(U, u):
    return 2 * R_KM * np.arcsin(np.clip(np.linalg.norm(U - u, axis=1) / 2, 0, 1))


def main():
    lon, lat, pop = load_r5()
    U = unit(lon, lat)
    members = [np.nonzero(dist(U, unit(np.array(m[1]), np.array(m[2]))) <= m[3])[0] for m in METROS]
    P = json.load(open(HERE / "pairs.json"))
    air = {(r[0], r[1]): r for r in P}
    names = [m[0] for m in METROS]
    ne = [names.index(n) for n in ["New York", "Philadelphia", "Boston", "Washington", "Baltimore"]]
    # destination weight per r5 cell for the tail needs the cell's own pop; histogram Hu only holds pop*log(pop),
    # so for the tail use exact sums per origin (cheap enough for the few parameter sets below)
    import itertools
    results = []
    for A, k, G, D1, TH in [(2.37, 0, 1, 500, 0)] + [(2.37, k, G, D1, TH) for k, G, D1, TH in itertools.product([1e-5, 3e-5, 1e-4], [0.5, 1.0], [300], [0, 0.5])]:
        res = run(U, pop, members, A, k, G, D1, TH, ne, names, air)
        results.append(res)
        print(json.dumps(res))


def kern(d, A, k, G, D1, w):
    on = np.clip((d - 25) / 25, 0, 1)
    return on * (np.maximum(d, 25) ** -A + k * np.maximum(d, D1) ** -G * w)


def run(U, pop, members, A, k, G, D1, TH, ne, names, air):
    w = (np.maximum(pop, 1) / 1e5) ** TH
    rate = 0.107
    n = len(members)
    # per origin cell: normaliser and band shares (NE only), and pair sums
    band = np.zeros(len(EDGES) - 1); wsum = 0
    Nn = []
    for I, idx in enumerate(members):
        N = np.empty(len(idx))
        for t, i in enumerate(idx):
            d = dist(U, U[i])
            f = kern(d, A, k, G, D1, w) * pop
            N[t] = f.sum()
            if I in ne:
                band += np.bincount(np.clip(np.searchsorted(EDGES, d, side="right") - 1, 0, len(EDGES) - 2), weights=f, minlength=len(EDGES) - 1) / N[t] * pop[i]
                wsum += pop[i]
        Nn.append(N)
    band = rate * band / wsum
    T = np.zeros((n, n))
    for I in range(n):
        oi = members[I]
        for J in range(n):
            if I != J:
                oj = members[J]
                dij = 2 * R_KM * np.arcsin(np.clip(np.linalg.norm(U[oi][:, None, :] - U[oj][None, :, :], axis=2) / 2, 0, 1))
                T[I, J] = (rate * pop[oi] / Nn[I] * (kern(dij, A, k, G, D1, w[oj][None, :]) * pop[oj][None, :]).sum(axis=1)).sum()
    ratios = {}
    for (a, b), r in air.items():
        I, J = names.index(a), names.index(b)
        r.append(T[I, J] + T[J, I])
    dom = [x[4] / x[-1] for x in air.values() if x[2] >= 1000 and x[4] > 0 and x[5] == x[6]]
    intl = [x[4] / x[-1] for x in air.values() if x[2] >= 1000 and x[4] > 0 and x[5] != x[6]]
    pick = {}
    for a, b in [("New York", "Philadelphia"), ("New York", "Boston"), ("New York", "Washington"), ("Boston", "Washington"), ("London", "New York"), ("New York", "Toronto"), ("London", "Paris"), ("Tokyo", "Osaka"), ("Tokyo", "Fukuoka")]:
        key = (a, b) if (a, b) in air else (b, a)
        pick[f"{a}-{b}"] = round(air[key][-1])
    for x in air.values():
        x.pop()
    nb = nh_all  # 25..5000 in NHTS bands; model bands EDGES[:-1] are the same up to 5000
    return {"A": A, "k": k, "G": G, "D1": D1, "TH": TH,
            "nhts_vs_model_>=500": [round(float(nb[i]), 5) for i in range(13, 18)], "model": [round(float(band[i]), 5) for i in range(13, 19)],
            "share_ge_1000": round(float(band[15:].sum() / band.sum()), 4),
            "air/model dom median": round(float(np.median(dom)), 2), "intl median": round(float(np.median(intl)), 2), "pairs": pick}


if __name__ == "__main__":
    main()
