"""T-159: Philadelphia work flows across the Delaware, model gravity against LODES OD and ACS.

    ..\\venv\\Scripts\\python.exe trainworld/notes/t159_flows.py [OUTDIR]

Read-only on the pack. Groups: the 11 MSA counties, Center City (Vine to South St, river to
river), Camden city (LODES crosswalk place). Writes a cache (zone shares, LODES classes) to OUTDIR
(default: a t159 folder in the system temp dir) for t159_variants.py.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import sys
import tempfile

import h3
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import read_pack  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw")
WORK = os.path.join(ROOT, "data", "work")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(tempfile.gettempdir(), "t159")
os.makedirs(OUT, exist_ok=True)

COUNTIES = {"42017": "Bucks", "42029": "Chester", "42045": "Delaware", "42091": "Montgomery",
            "42101": "Philadelphia", "34005": "Burlington", "34007": "Camden", "34015": "Gloucester",
            "34033": "Salem", "10003": "New Castle", "24015": "Cecil"}
CF = list(COUNTIES)
CIDX = {f: i for i, f in enumerate(CF)}
NC = len(CF)
# Center City: Vine St to South St, Schuylkill to Delaware (lat/lon box)
CC_BOX = (39.9415, 39.9585, -75.1795, -75.1390)
GROUPS = [COUNTIES[f] for f in CF] + ["Center City", "Camden city"]
G = len(GROUPS)
GCC, GCAM = NC, NC + 1


def in_cc(lat, lon):
    return (lat >= CC_BOX[0]) & (lat <= CC_BOX[1]) & (lon >= CC_BOX[2]) & (lon <= CC_BOX[3])


def blocks():
    b = pd.read_csv(os.path.join(WORK, "phl.blocks.csv.gz"), dtype={"geoid": str},
                    usecols=["geoid", "lat", "lon", "jobs", "rac", "pop"])
    b["geoid"] = b["geoid"].str.zfill(15)
    xw = pd.read_csv(os.path.join(RAW, "nj_xwalk.csv.gz"), dtype=str, usecols=["tabblk2020", "stplcname"])
    camden = set(xw.loc[xw["stplcname"].fillna("").str.startswith("Camden city"), "tabblk2020"])
    b["county"] = b["geoid"].str[:5].map(CIDX)
    b["cc"] = in_cc(b["lat"].to_numpy(), b["lon"].to_numpy())
    b["cam"] = b["geoid"].isin(camden)
    return b, camden


def block_class(county, cc, cam):
    """One integer per (county, Center City flag, Camden city flag)."""
    return county * 4 + cc.astype(int) * 2 + cam.astype(int)


def class_to_groups():
    """Indicator matrix: class -> groups it belongs to."""
    M = np.zeros((NC * 4, G))
    for c in range(NC * 4):
        M[c, c // 4] = 1
        if c & 2:
            M[c, GCC] = 1
        if c & 1:
            M[c, GCAM] = 1
    return M


def lodes():
    """LODES OD 2023 S000 between blocks of the 11 counties, by (home class, work class)."""
    b, camden = blocks()
    bi = b.set_index("geoid")
    cls = pd.Series(block_class(b["county"].to_numpy(), b["cc"].to_numpy(), b["cam"].to_numpy()), index=b["geoid"])
    files = ["pa_od_main", "pa_od_aux", "nj_od_main", "nj_od_aux"]
    F = np.zeros((NC * 4, NC * 4))
    pairs = []  # cross-state block pairs for distance analysis
    for f in files:
        d = pd.read_csv(os.path.join(RAW, f"{f}_JT00_2023.csv.gz"), dtype={"w_geocode": str, "h_geocode": str},
                        usecols=["w_geocode", "h_geocode", "S000"])
        d = d[d["w_geocode"].str[:5].isin(CIDX) & d["h_geocode"].str[:5].isin(CIDX)]
        hc = cls.reindex(d["h_geocode"]).to_numpy()
        wc = cls.reindex(d["w_geocode"]).to_numpy()
        ok = ~(np.isnan(hc) | np.isnan(wc))
        miss = d["S000"].to_numpy()[~ok].sum()
        np.add.at(F, (hc[ok].astype(int), wc[ok].astype(int)), d["S000"].to_numpy()[ok])
        print(f"  {f}: {len(d):,} rows in region, {d['S000'].sum():,} jobs, {miss:,} not matched to pack blocks")
        hs, ws = d["h_geocode"].str[:2], d["w_geocode"].str[:2]
        x = d[((hs == "42") & (ws == "34")) | ((hs == "34") & (ws == "42"))]
        pairs.append(x)
    # crow-fly km of the PA<->NJ jobs
    x = pd.concat(pairs)
    lat = np.radians(bi["lat"]); lon = np.radians(bi["lon"])
    h_lat, h_lon = lat.reindex(x["h_geocode"]).to_numpy(), lon.reindex(x["h_geocode"]).to_numpy()
    w_lat, w_lon = lat.reindex(x["w_geocode"]).to_numpy(), lon.reindex(x["w_geocode"]).to_numpy()
    km = 6371.0 * 2 * np.arcsin(np.sqrt(np.sin((w_lat - h_lat) / 2) ** 2 + np.cos(h_lat) * np.cos(w_lat) * np.sin((w_lon - h_lon) / 2) ** 2))
    pd.DataFrame({"h": x["h_geocode"].str[:5].to_numpy(), "w": x["w_geocode"].str[:5].to_numpy(), "km": km, "n": x["S000"].to_numpy()}).to_csv(
        os.path.join(OUT, "lodes_cross_pairs.csv.gz"), index=False)
    # residents (RAC) and jobs (WAC) per class, all destinations
    rac = np.bincount(cls.to_numpy().astype(int), weights=b["rac"].to_numpy(), minlength=NC * 4)
    return F, rac, b, camden


def pack_cells(b):
    header, a = read_pack.load_pack("phl")
    ids = a["h3"]
    ll = np.array([h3.cell_to_latlng(h3.int_to_str(int(i))) for i in ids])
    tree = cKDTree(np.c_[b["lat"].to_numpy(), b["lon"].to_numpy() * np.cos(np.radians(39.95))])
    _, k = tree.query(np.c_[ll[:, 0], ll[:, 1] * np.cos(np.radians(39.95))])
    nb = b.iloc[k]
    county = nb["county"].to_numpy().astype(int)
    cc = in_cc(ll[:, 0], ll[:, 1])
    cam = nb["cam"].to_numpy()
    return header, a, ll, block_class(county, cc, cam)


def zone_trips(header, a, rows=None):
    """Yields (start, end, T block) of the shipped zone gravity."""
    g = header["gravity"]
    gx, gy = a["zone_gx"].astype(np.int64), a["zone_gy"].astype(np.int64)
    row, col = a["zone_row"].astype(np.float64), a["zone_col"].astype(np.float64)
    zk = g["zone_m"] / 1000.0
    nz = header["zones"]
    for s in range(0, nz, 400):
        e = min(nz, s + 400)
        d = zk * np.sqrt((gx[s:e, None] - gx[None, :]) ** 2 + (gy[s:e, None] - gy[None, :]) ** 2)
        d[d == 0] = g["intrazonal_km"]
        yield s, e, row[s:e, None] * col[None, :] * np.exp(-d / g["decay_km"]) * d ** -g["decay_pow"], d


def main():
    print("LODES OD 2023:")
    L, rac, b, camden = lodes()
    C2G = class_to_groups()
    LG = C2G.T @ L @ C2G
    racG = C2G.T @ rac
    header, a, ll, ccls = pack_cells(b)
    nz = header["zones"]
    zone = a["zone"].astype(np.int64)
    zh = np.zeros((nz, NC * 4))
    zw = np.zeros((nz, NC * 4))
    np.add.at(zh, (zone, ccls), a["commute_home"].astype(np.float64))
    np.add.at(zw, (zone, ccls), a["commute_work"].astype(np.float64))
    np.save(os.path.join(OUT, "zh.npy"), zh)
    np.save(os.path.join(OUT, "zw.npy"), zw)
    zhs = zh / np.maximum(zh.sum(1, keepdims=True), 1e-12)
    zws = zw / np.maximum(zw.sum(1, keepdims=True), 1e-12)
    M = np.zeros((NC * 4, NC * 4))
    # model Philadelphia -> Camden County trips by zone distance
    ph, cm = CIDX["42101"], CIDX["34007"]
    hph = zhs[:, ph * 4:ph * 4 + 4].sum(1)
    wcm = zws[:, cm * 4:cm * 4 + 4].sum(1)
    hcm = zhs[:, cm * 4:cm * 4 + 4].sum(1)
    wph = zws[:, ph * 4:ph * 4 + 4].sum(1)
    bins = np.array([0, 1.5, 3, 5, 8, 12, 20, 40, 200])
    hist_pc = np.zeros(len(bins) - 1)
    hist_cp = np.zeros(len(bins) - 1)
    for s, e, T, d in zone_trips(header, a):
        M += zhs[s:e].T @ T @ zws
        x = hph[s:e, None] * T * wcm[None, :]
        hist_pc += np.bincount(np.digitize(d.ravel(), bins) - 1, weights=x.ravel(), minlength=len(bins) - 1)[:len(bins) - 1]
        x = hcm[s:e, None] * T * wph[None, :]
        hist_cp += np.bincount(np.digitize(d.ravel(), bins) - 1, weights=x.ravel(), minlength=len(bins) - 1)[:len(bins) - 1]
    np.save(os.path.join(OUT, "model_class_flows.npy"), M)
    np.save(os.path.join(OUT, "lodes_class_flows.npy"), L)
    MG = C2G.T @ M @ C2G
    homeM = MG[:, :NC].sum(1)  # model rows over the 11 counties
    homeL = LG[:, :NC].sum(1)  # LODES in-region rows

    # ACS 2016-2020 county flows
    acs = pd.read_excel(os.path.join(RAW, "acsflows", "table1.xlsx"), header=None, skiprows=8, dtype=str).iloc[:, :10]
    acs.columns = ["hs", "hc", "hsn", "hcn", "ws", "wc", "wsn", "wcn", "n", "moe"]
    acs = acs.dropna(subset=["hs", "ws"])
    acs["h"] = acs["hs"].str.zfill(2) + acs["hc"].str.zfill(3)
    acs["w"] = acs["ws"].str[-2:] + acs["wc"].str.zfill(3)
    acs["n"] = pd.to_numeric(acs["n"], errors="coerce")
    acs["moe"] = pd.to_numeric(acs["moe"], errors="coerce")
    acs = acs[acs["h"].isin(CIDX)]
    acs[["h", "w", "n", "moe"]].to_csv(os.path.join(RAW, "acsflows", "phl_county_flows.csv"), index=False)
    A = np.zeros((NC, NC)); Amoe = np.zeros((NC, NC))
    acs_tot = acs.groupby("h")["n"].sum()
    for r in acs.itertuples():
        if r.w in CIDX:
            A[CIDX[r.h], CIDX[r.w]] += r.n
            Amoe[CIDX[r.h], CIDX[r.w]] = r.moe
    homeA = A.sum(1)

    scale = header["totals"]["commute_work"] / rac.sum()
    print(f"\nmodel trips {M.sum():,.0f}; LODES in-region jobs {L.sum():,.0f}; RAC jobs {rac.sum():,.0f}; ratio commute/RAC {scale:.3f}")

    def row(hg, wg, label):
        m, l = MG[hg, wg], LG[hg, wg]
        s = f"| {label} | {m:,.0f} | {m / homeM[hg]:.2%} | {l:,.0f} | {l / homeL[hg]:.2%} | {m / homeM[hg] / (l / homeL[hg]):.2f}x |"
        if hg < NC and wg < NC:
            s += f" {A[hg, wg]:,.0f} ±{Amoe[hg, wg]:,.0f} | {A[hg, wg] / homeA[hg]:.2%} | {m / homeM[hg] / (A[hg, wg] / homeA[hg]):.2f}x |"
        else:
            s += " | | |"
        print(s)

    P, CM, BU, GL, DE, MO, BK, CH = (CIDX[f] for f in ["42101", "34007", "34005", "34015", "42045", "42091", "42017", "42029"])
    print("\n| home -> work | model trips | model share of home group | LODES jobs | LODES share | model/LODES | ACS workers | ACS share | model/ACS |")
    print("|---|---|---|---|---|---|---|---|---|")
    for hg, wg, lab in [
        (GCC, CM, "Center City -> Camden County"), (GCC, GCAM, "Center City -> Camden city"),
        (GCC, BU, "Center City -> Burlington"), (GCC, GL, "Center City -> Gloucester"),
        (GCC, GCC, "Center City -> Center City"), (GCC, P, "Center City -> Philadelphia County"),
        (P, CM, "Philadelphia -> Camden County"), (P, GCAM, "Philadelphia -> Camden city"),
        (P, BU, "Philadelphia -> Burlington"), (P, GL, "Philadelphia -> Gloucester"),
        (P, DE, "Philadelphia -> Delaware County"), (P, MO, "Philadelphia -> Montgomery"),
        (P, BK, "Philadelphia -> Bucks"), (P, P, "Philadelphia -> Philadelphia"),
        (CM, P, "Camden County -> Philadelphia"), (CM, GCC, "Camden County -> Center City"),
        (GCAM, GCC, "Camden city -> Center City"), (GCAM, P, "Camden city -> Philadelphia"),
        (CM, CM, "Camden County -> Camden County"),
        (BU, P, "Burlington -> Philadelphia"), (GL, P, "Gloucester -> Philadelphia"),
        (DE, P, "Delaware County -> Philadelphia"), (MO, P, "Montgomery -> Philadelphia"),
    ]:
        row(hg, wg, lab)

    # state-level blocks
    pa = [CIDX[f] for f in CF if f.startswith("42")]
    nj = [CIDX[f] for f in CF if f.startswith("34")]
    def blk(X, h, w):
        return X[np.ix_(h, w)].sum()
    print("\nPA5 <-> NJ4 (model trips / LODES jobs / ACS workers):")
    for lab, h, w in [("PA->NJ", pa, nj), ("NJ->PA", nj, pa), ("NJ->NJ", nj, nj), ("PA->PA", pa, pa)]:
        print(f"  {lab}: model {blk(MG, h, w):,.0f} ({blk(MG, h, w) / MG[h][:, :NC].sum():.2%}), "
              f"LODES {blk(LG, h, w):,.0f} ({blk(LG, h, w) / LG[h][:, :NC].sum():.2%}), "
              f"ACS {blk(A, h, w):,.0f} ({blk(A, h, w) / A[h].sum():.2%})")
    print("\nown-county share (model / LODES / ACS):")
    for i, f in enumerate(CF):
        print(f"  {COUNTIES[f]}: {MG[i, i] / homeM[i]:.1%} / {LG[i, i] / homeL[i]:.1%} / {A[i, i] / homeA[i]:.1%}")
    print(f"  all: {np.trace(MG[:NC, :NC]) / MG[:NC, :NC].sum():.1%} / {np.trace(LG[:NC, :NC]) / LG[:NC, :NC].sum():.1%} / {np.trace(A) / A.sum():.1%}")
    print(f"\nhome totals: Center City model {homeM[GCC]:,.0f}, LODES in-region {homeL[GCC]:,.0f} (RAC {racG[GCC]:,.0f}); "
          f"Philadelphia model {homeM[P]:,.0f}, LODES {homeL[P]:,.0f}, ACS {homeA[P]:,.0f}")
    print(f"work totals: Camden County model {MG[:NC, CM].sum():,.0f}, LODES {LG[:NC, CM].sum():,.0f}; Camden city model {MG[:NC, GCAM].sum():,.0f}, LODES {LG[:NC, GCAM].sum():,.0f}")
    print("\nmodel Philadelphia -> Camden County by zone-centre km:", {f"{bins[i]}-{bins[i+1]}": round(hist_pc[i]) for i in range(len(hist_pc))})
    print("model Camden County -> Philadelphia by zone-centre km:", {f"{bins[i]}-{bins[i+1]}": round(hist_cp[i]) for i in range(len(hist_cp))})
    x = pd.read_csv(os.path.join(OUT, "lodes_cross_pairs.csv.gz"), dtype={"h": str, "w": str})
    for lab, hh, ww in [("Philadelphia -> Camden County", "42101", "34007"), ("Camden County -> Philadelphia", "34007", "42101")]:
        y = x[(x["h"] == hh) & (x["w"] == ww)]
        hcount = np.bincount(np.digitize(y["km"], bins) - 1, weights=y["n"], minlength=len(bins) - 1)[:len(bins) - 1]
        print(f"LODES {lab} by block crow-fly km:", {f"{bins[i]}-{bins[i+1]}": round(hcount[i]) for i in range(len(hcount))})

    # zones holding cells of both states
    st = np.zeros((nz, 2))
    for c in range(NC * 4):
        s = 0 if CF[c // 4].startswith("42") else (1 if CF[c // 4].startswith("34") else None)
        if s is not None:
            st[:, s] += zh[:, c] + zw[:, c]
    both = (st[:, 0] > 0) & (st[:, 1] > 0)
    print(f"\nzones with PA and NJ cells: {both.sum()}; their commute weight {st[both].sum():,.0f}")
    np.save(os.path.join(OUT, "classes_meta.npy"), np.array([NC]))


if __name__ == "__main__":
    main()
