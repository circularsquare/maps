"""T-362 experiment: the Northeast market's formula (rate 0.107, d^-2.37, onset 25-50 km, floor 25 km)
run over the whole world on Kontur r5, metro to metro, against flights segment passengers.
Usage: python world_gravity.py  (writes cache r5.npz beside this script)"""
import gzip, json, math, os, shutil, sqlite3, sys
os.environ.setdefault("OMP_NUM_THREADS", "3")
from pathlib import Path
import numpy as np
import h3

HERE = Path(__file__).resolve().parents[3] / "artifacts/trainworld/t362"  # outputs and caches
GZ = Path(r"C:\Users\anita\projects\maps\trainworld\data\raw\kontur\kontur_population_20231101_r6.gpkg.gz")
GPKG = HERE / "k6.gpkg"
CACHE = HERE / "r5.npz"
ROUTES = Path(r"C:\Users\anita\projects\maps\flights\routes_fr24.json")
R_KM = 6371.0088
RATE, A, D0, ON0, ON1 = 0.107, 2.37, 25.0, 25.0, 50.0

METROS = [  # name, lon, lat, radius km, airports, country
    ("New York", -73.98, 40.75, 50, "JFK EWR LGA", "US"), ("Philadelphia", -75.16, 39.95, 35, "PHL", "US"),
    ("Boston", -71.06, 42.36, 40, "BOS", "US"), ("Washington", -77.04, 38.90, 30, "DCA IAD", "US"),
    ("Baltimore", -76.61, 39.29, 20, "BWI", "US"), ("Chicago", -87.65, 41.88, 45, "ORD MDW", "US"),
    ("Los Angeles", -118.25, 34.05, 55, "LAX BUR LGB SNA ONT", "US"), ("San Francisco", -122.27, 37.6, 50, "SFO OAK SJC", "US"),
    ("Atlanta", -84.39, 33.75, 40, "ATL", "US"), ("Miami", -80.2, 25.9, 40, "MIA FLL", "US"),
    ("Toronto", -79.38, 43.70, 45, "YYZ YTZ", "CA"), ("Montreal", -73.57, 45.50, 35, "YUL", "CA"),
    ("Mexico City", -99.13, 19.43, 40, "MEX", "MX"), ("Sao Paulo", -46.63, -23.55, 45, "GRU CGH", "BR"),
    ("London", -0.12, 51.51, 50, "LHR LGW LCY STN LTN", "GB"), ("Manchester", -2.24, 53.48, 35, "MAN", "GB"),
    ("Edinburgh", -3.19, 55.95, 25, "EDI", "GB"), ("Glasgow", -4.25, 55.86, 25, "GLA", "GB"),
    ("Paris", 2.35, 48.86, 45, "CDG ORY", "FR"), ("Amsterdam", 4.90, 52.37, 30, "AMS", "NL"),
    ("Brussels", 4.35, 50.85, 30, "BRU", "BE"), ("Frankfurt", 8.68, 50.11, 35, "FRA", "DE"),
    ("Berlin", 13.40, 52.52, 35, "BER SXF", "DE"), ("Munich", 11.58, 48.14, 35, "MUC", "DE"),
    ("Madrid", -3.70, 40.42, 35, "MAD", "ES"), ("Barcelona", 2.17, 41.39, 30, "BCN", "ES"),
    ("Rome", 12.50, 41.90, 30, "FCO CIA", "IT"), ("Milan", 9.19, 45.46, 35, "MXP LIN BGY", "IT"),
    ("Dublin", -6.26, 53.35, 25, "DUB", "IE"), ("Istanbul", 28.98, 41.01, 45, "IST SAW", "TR"),
    ("Dubai", 55.27, 25.20, 35, "DXB DWC", "AE"), ("Mumbai", 72.88, 19.08, 40, "BOM", "IN"),
    ("Delhi", 77.21, 28.61, 45, "DEL", "IN"), ("Bangalore", 77.59, 12.97, 35, "BLR", "IN"),
    ("Singapore", 103.82, 1.35, 25, "SIN", "SG"), ("Hong Kong", 114.17, 22.32, 25, "HKG", "HK"),
    ("Shanghai", 121.47, 31.23, 50, "PVG SHA", "CN"), ("Beijing", 116.40, 39.90, 50, "PEK PKX", "CN"),
    ("Guangzhou", 113.26, 23.13, 40, "CAN", "CN"), ("Chengdu", 104.07, 30.67, 40, "CTU TFU", "CN"),
    ("Seoul", 126.98, 37.57, 45, "ICN GMP", "KR"), ("Busan", 129.08, 35.18, 30, "PUS", "KR"),
    ("Tokyo", 139.69, 35.69, 50, "HND NRT", "JP"), ("Osaka", 135.50, 34.69, 40, "ITM KIX UKB", "JP"),
    ("Fukuoka", 130.40, 33.59, 30, "FUK", "JP"), ("Sapporo", 141.35, 43.06, 30, "CTS", "JP"),
    ("Sydney", 151.21, -33.87, 40, "SYD", "AU"), ("Melbourne", 144.96, -37.81, 40, "MEL", "AU"),
]


def unit(lon, lat):
    lon, lat = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def load_r5():
    if CACHE.exists():
        z = np.load(CACHE)
        return z["lon"], z["lat"], z["pop"]
    if not GPKG.exists():
        with gzip.open(GZ, "rb") as f, open(GPKG, "wb") as g:
            shutil.copyfileobj(f, g, 1 << 24)
    rows = sqlite3.connect(GPKG).execute("select h3, population from population where population > 0").fetchall()
    acc = {}
    for c, p in rows:
        par = h3.cell_to_parent(c, 5)
        lat, lon = h3.cell_to_latlng(c)
        a = acc.setdefault(par, [0.0, 0.0, 0.0])
        a[0] += p * lon; a[1] += p * lat; a[2] += p
    v = np.array(list(acc.values()))
    lon, lat, pop = v[:, 0] / v[:, 2], v[:, 1] / v[:, 2], v[:, 2]
    np.savez(CACHE, lon=lon, lat=lat, pop=pop)
    GPKG.unlink()
    return lon, lat, pop


def g(d):
    on = np.clip((d - ON0) / (ON1 - ON0), 0, 1)
    return on * np.maximum(d, D0) ** -A


def main():
    lon, lat, pop = load_r5()
    print(f"r5 cells {len(pop):,}, people {pop.sum()/1e9:.2f} bn")
    U = unit(lon, lat)
    members, norm = [], []
    for name, mlon, mlat, rad, _, _ in METROS:
        u = unit(np.array(mlon), np.array(mlat))
        d = 2 * R_KM * np.arcsin(np.clip(np.linalg.norm(U - u, axis=1) / 2, 0, 1))
        idx = np.nonzero(d <= rad)[0]
        members.append(idx)
        N = np.empty(len(idx))
        for t, i in enumerate(idx):
            di = 2 * R_KM * np.arcsin(np.clip(np.linalg.norm(U - U[i], axis=1) / 2, 0, 1))
            N[t] = (pop * g(di)).sum()
        norm.append(N)
        print(f"{name}: {len(idx)} cells, {pop[idx].sum()/1e6:.1f} M people", flush=True)
    n = len(METROS)
    T = np.zeros((n, n))
    for I in range(n):
        oi = members[I]
        for J in range(n):
            if I == J:
                continue
            oj = members[J]
            dij = 2 * R_KM * np.arcsin(np.clip(np.linalg.norm(U[oi][:, None, :] - U[oj][None, :, :], axis=2) / 2, 0, 1))
            T[I, J] = (RATE * pop[oi] / norm[I] * (g(dij) * pop[oj][None, :]).sum(axis=1)).sum()
    # flights: segment passengers a year, both directions
    fr = json.load(open(ROUTES, encoding="utf-8"))
    ap = {}
    for I, m in enumerate(METROS):
        for c in m[4].split():
            ap[c] = I
    F = np.zeros((n, n))
    for a, b, pax, *_ in fr["routes"]:
        if a in ap and b in ap and ap[a] != ap[b]:
            F[ap[a], ap[b]] += pax; F[ap[b], ap[a]] += pax
    out = []
    for I in range(n):
        for J in range(I + 1, n):
            mi, mj = METROS[I], METROS[J]
            d = 2 * R_KM * math.asin(min(1, float(np.linalg.norm(unit(np.array(mi[1]), np.array(mi[2])) - unit(np.array(mj[1]), np.array(mj[2])))) / 2))
            model = T[I, J] + T[J, I]  # trips a weekday, both directions
            air = F[I, J] / 365
            out.append((mi[0], mj[0], round(d), round(model), round(air), mi[5], mj[5]))
    json.dump(out, open(HERE / "pairs.json", "w"), indent=0)
    print("wrote pairs.json", len(out))


if __name__ == "__main__":
    main()
