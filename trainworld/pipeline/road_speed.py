"""T-034: how fast the "other" mode (driving) really is, by county of residence.

    python pipeline/road_speed.py nyc

ACS 2019-2023 B08136 (aggregate minutes to work by means) over B08301 (workers by means) gives
the mean car commute in minutes per county of residence; the pack's zone gravity gives the mean
crow-fly length of those commutes (zone-centre distances, as the game measures them). Their
ratio is the crow-fly speed a car commute really makes there, door to door.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import read_pack  # noqa: E402
import realnet  # noqa: E402

RAW = realnet.RAW
B08136 = "https://www2.census.gov/programs-surveys/acs/summary_file/2023/table-based-SF/data/5YRData/acsdt5y2023-b08136.dat"
# B08136 (aggregate travel time to work, minutes): 001 total, 002 car, truck or van, 003 drove
# alone, 004 carpooled, 007 public transportation, 008 bus, 009 subway or elevated, 010 commuter
# rail and the rest of rail, 011 walked, 012 taxi, motorcycle, bicycle, other.
COLS = [f"B08136_E{i:03d}" for i in range(1, 13)]


def acs_time():
    cache = os.path.join(RAW, "acs2023_5y_b08136_counties.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, dtype={"fips": str}).set_index("fips")
    src = os.path.join(RAW, "acsdt5y2023-b08136.dat")
    rows = []
    ct = set()
    import shapefile
    rd = shapefile.Reader(dbf=open(os.path.join(RAW, "tl_2020_09_tabblock20.dbf"), "rb"))
    for rec in rd.iterRecords(fields=["COUNTYFP20", "TRACTCE20"]):
        if rec[0] == "001":
            ct.add(rec[1])
    for ch in pd.read_csv(src, sep="|", usecols=["GEO_ID"] + COLS, dtype={"GEO_ID": str}, chunksize=200_000):
        # tract rows carry negative "jam values" where an estimate is suppressed
        ch[COLS] = ch[COLS].where(ch[COLS] >= 0)
        c = ch[ch["GEO_ID"].str.startswith("0500000US")]
        c = c.assign(fips=c["GEO_ID"].str[9:14])
        rows.append(c[c["fips"].isin(realnet.COUNTIES)])
        tr = ch[ch["GEO_ID"].str.startswith("1400000US09") & ch["GEO_ID"].str[14:20].isin(ct)]
        if len(tr):
            s = tr[COLS].sum().to_frame().T
            s["fips"] = "09001"
            rows.append(s)
    d = pd.concat(rows, ignore_index=True).groupby("fips")[COLS].sum()
    d.to_csv(cache)
    return d


def county_distances():
    """Mean and quartiles of commute crow-fly length by county of residence, from the pack."""
    header, a = read_pack.load_pack("nyc")
    g = header["gravity"]
    nz = header["zones"]
    zone = a["zone"].astype(np.int64)
    gx, gy = a["zone_gx"].astype(np.int64), a["zone_gy"].astype(np.int64)
    cc = np.fromfile(os.path.join(realnet.WORK, "cell_county.u8"), np.uint8)
    hw = a["commute_home"].astype(np.float64)
    # each zone's home weight by county
    zc = np.zeros((nz, len(realnet.COUNTY_ORDER)))
    np.add.at(zc, (zone, cc), hw)
    zfrac = zc / np.maximum(zc.sum(1, keepdims=True), 1e-9)
    row, col = a["zone_row"].astype(np.float64), a["zone_col"].astype(np.float64)
    a_pow = g.get("decay_pow", 0.0)
    zk = g["zone_m"] / 1000.0
    edges = np.array([2, 5, 10, 20, 40, 1e9])
    trips = np.zeros(nz)
    dsum = np.zeros(nz)
    hist = np.zeros((nz, len(edges)))
    for s in range(0, nz, 400):
        e = min(nz, s + 400)
        d = zk * np.sqrt((gx[s:e, None] - gx[None, :]) ** 2 + (gy[s:e, None] - gy[None, :]) ** 2)
        d[d == 0] = g["intrazonal_km"]
        t = row[s:e, None] * col[None, :] * np.exp(-d / g["decay_km"]) * d ** -a_pow
        trips[s:e] = t.sum(1)
        dsum[s:e] = (t * d).sum(1)
        b = np.searchsorted(edges, d)
        for k in range(len(edges)):
            hist[s:e, k] = (t * (b == k)).sum(1)
    out = {}
    for i, f in enumerate(realnet.COUNTY_ORDER):
        w = zfrac[:, i]
        out[f] = ((w * dsum).sum() / (w * trips).sum(), (w[:, None] * hist).sum(0) / (w * trips).sum())
    return out


N_BINS = 16


def zone_sums(R_km, detour=1.3):
    """Per zone of residence: trips Tot_I, and for the road-time model
        time = 3 + 60 * (e (1/v_I + 1/v_J) + (road - 2e) / v_hwy),  road = d x detour, e = min(road/2, R)
    the sums A_I = sum_J T e, M_Ik = sum_{J in density bin k} T e, H_I = sum_J T (road - 2e);
    plus each zone's density (commute ends per km² of its cells), its bin, and its county mix."""
    header, a = read_pack.load_pack("nyc")
    g = header["gravity"]
    nz = header["zones"]
    zone = a["zone"].astype(np.int64)
    gx, gy = a["zone_gx"].astype(np.int64), a["zone_gy"].astype(np.int64)
    cc = np.fromfile(os.path.join(realnet.WORK, "cell_county.u8"), np.uint8)
    hw = a["commute_home"].astype(np.float64)
    ww = a["commute_work"].astype(np.float64)
    ends = np.bincount(zone, weights=hw * (ww.sum() / hw.sum()) + ww, minlength=nz)
    ncell = np.bincount(zone, minlength=nz)
    dens = ends / np.maximum(ncell * 0.1053, 1e-3)
    zc = np.zeros((nz, len(realnet.COUNTY_ORDER)))
    np.add.at(zc, (zone, cc), hw)
    zfrac = zc / np.maximum(zc.sum(1, keepdims=True), 1e-9)
    qs = np.quantile(np.log(np.maximum(dens, 1)), np.linspace(0, 1, N_BINS + 1))
    zbin = np.clip(np.searchsorted(qs, np.log(np.maximum(dens, 1)), side="right") - 1, 0, N_BINS - 1)
    bin_dens = np.array([np.exp(np.log(np.maximum(dens[zbin == k], 1)).mean()) for k in range(N_BINS)])
    row, col = a["zone_row"].astype(np.float64), a["zone_col"].astype(np.float64)
    a_pow = g.get("decay_pow", 0.0)
    zk = g["zone_m"] / 1000.0
    Tot, A, H, D = np.zeros(nz), np.zeros(nz), np.zeros(nz), np.zeros(nz)
    M = np.zeros((nz, N_BINS))
    for s in range(0, nz, 300):
        e_ = min(nz, s + 300)
        d = zk * np.sqrt((gx[s:e_, None] - gx[None, :]) ** 2 + (gy[s:e_, None] - gy[None, :]) ** 2)
        d[d == 0] = g["intrazonal_km"]
        t = row[s:e_, None] * col[None, :] * np.exp(-d / g["decay_km"]) * d ** -a_pow
        road = d * detour
        e = np.minimum(road / 2, R_km)
        Tot[s:e_] = t.sum(1)
        D[s:e_] = (t * d).sum(1)
        te = t * e
        A[s:e_] = te.sum(1)
        H[s:e_] = (t * (road - 2 * e)).sum(1)
        for k in range(N_BINS):
            M[s:e_, k] = te[:, zbin == k].sum(1)
    return dict(Tot=Tot, A=A, H=H, M=M, D=D, dens=dens, zbin=zbin, bin_dens=bin_dens, zfrac=zfrac)


def speed(dens, v_lo, v_hi, d50, k=1.0):
    """Road speed at a place, km/h, falling from v_hi in the countryside to v_lo where it is dense."""
    return v_lo + (v_hi - v_lo) / (1.0 + (dens / d50) ** k)


def county_times(zs, p):
    v_lo, v_hi, d50, v_hwy = p
    s_zone = 1.0 / speed(zs["dens"], v_lo, v_hi, d50)
    s_bin = 1.0 / speed(zs["bin_dens"], v_lo, v_hi, d50)
    tsum = 3.0 * zs["Tot"] + 60.0 * (s_zone * zs["A"] + zs["M"] @ s_bin + zs["H"] / v_hwy)
    w = zs["zfrac"]
    return (w.T @ tsum) / (w.T @ zs["Tot"]), (w.T @ zs["D"]) / (w.T @ zs["Tot"])


def fit():
    """Fit the road speed curve to ACS mean car commute minutes by county of residence, on the
    counties where at least 60% drive (their car commuters are nearly all commuters)."""
    from scipy.optimize import minimize
    acs = realnet.acs_counties()
    tm = acs_time()
    obs, wts, use = [], [], []
    for f in realnet.COUNTY_ORDER:
        a, t = acs.loc[f], tm.loc[f]
        car_n = a["B08301_E002"]
        comm = a["B08301_E001"] - a["B08301_E021"]
        m = t["B08136_E002"] / car_n
        obs.append(m)
        wts.append(car_n)
        use.append(car_n / comm >= 0.6 and 10 < m < 80)
    obs, wts, use = np.array(obs), np.array(wts), np.array(use)
    best = None
    for R in (2.0, 4.0, 8.0):
        zs = zone_sums(R)

        def err(p):
            if min(p) <= 1:
                return 1e9
            ct, _ = county_times(zs, p)
            return float((wts[use] * np.log(ct[use] / obs[use]) ** 2).sum() / wts[use].sum())
        r = minimize(err, [15.0, 50.0, 5000.0, 60.0], method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-2, "fatol": 1e-8})
        print(f"R {R} km: v_lo {r.x[0]:.1f}, v_hi {r.x[1]:.1f}, d50 {r.x[2]:.0f}/km², v_hwy {r.x[3]:.1f} km/h; rms log error {np.sqrt(r.fun):.3f}")
        if best is None or r.fun < best[0]:
            best = (r.fun, R, r.x, zs)
    _, R, p, zs = best
    ct, cd = county_times(zs, p)
    flat, _ = county_times(zs, [28.0, 28.0, 1.0, 28.0])
    print(f"\nbest: R {R} km, {np.round(p, 1)}")
    print(f"{'county':14} {'used':>5} {'ACS car min':>11} {'model':>7} {'flat 28':>8} {'mean km':>8}")
    for i, f in enumerate(realnet.COUNTY_ORDER):
        print(f"{realnet.COUNTIES[f]:14} {'yes' if use[i] else '':>5} {obs[i]:11.1f} {ct[i]:7.1f} {flat[i]:8.1f} {cd[i]:8.1f}")
    for dd in (500, 2000, 5000, 10000, 20000, 50000, 100000):
        print(f"  density {dd:>7}/km²: {speed(dd, *p[:3]):.1f} km/h")
    json.dump({"R_km": R, "v_lo": p[0], "v_hi": p[1], "d50": p[2], "v_hwy": p[3]}, open(os.path.join(realnet.WORK, "road_speed_fit.json"), "w"))


def main():
    if "--fit" in sys.argv:
        return fit()
    acs = realnet.acs_counties()
    tm = acs_time()
    dist = county_distances()
    rows = []
    for f in realnet.COUNTY_ORDER:
        a, t = acs.loc[f], tm.loc[f]
        car_n = a["B08301_E002"]
        comm = a["B08301_E001"] - a["B08301_E021"]
        car_min = t["B08136_E002"] / car_n
        d, h = dist[f]
        rows.append((realnet.COUNTIES[f], comm, car_n / comm, car_min, d, d / (car_min / 60), h))
    df = pd.DataFrame(rows, columns=["county", "commuters", "car_share", "car_min", "mean_km", "crowfly_kmh", "hist"])
    pd.set_option("display.width", 200)
    print(df.drop(columns="hist").sort_values("crowfly_kmh").to_string(index=False, formatters={
        "commuters": "{:,.0f}".format, "car_share": "{:.0%}".format, "car_min": "{:.1f}".format,
        "mean_km": "{:.1f}".format, "crowfly_kmh": "{:.1f}".format}))
    df.drop(columns="hist").to_csv(os.path.join(realnet.WORK, "road_speed_by_county.csv"), index=False)


if __name__ == "__main__":
    main()
