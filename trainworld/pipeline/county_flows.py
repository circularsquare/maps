"""T-034 check: where each county's residents work, in the model's gravity, in LODES OD (what the
decay is fitted to, T-006) and in the ACS county-to-county flows (2016-2020, a survey of where
people say they work).

    python pipeline/county_flows.py

ACS counts every worker 16+, those working from home included (at their home county); LODES and
the model count jobs, the model less those worked from home. Shares are compared, not totals.
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

# https://www2.census.gov/programs-surveys/demo/tables/metro-micro/2020/commuting-flows-2020/table1.xlsx
# (6 MB; only its New York rows are kept, in nyc_county_flows.csv beside it)
FLOWS = os.path.join(realnet.RAW, "acsflows", "table1.xlsx")
ORDER = realnet.COUNTY_ORDER
IDX = {f: i for i, f in enumerate(ORDER)}


def acs_flows():
    cache = os.path.join(realnet.RAW, "acsflows", "nyc_county_flows.csv")
    if not os.path.exists(cache):
        d = pd.read_excel(FLOWS, header=None, skiprows=8, dtype=str)
        d = d.iloc[:, :10]
        d.columns = ["hs", "hc", "hsn", "hcn", "ws", "wc", "wsn", "wcn", "n", "moe"]
        d = d.dropna(subset=["hs", "ws"])
        d["h"] = d["hs"].str.zfill(2) + d["hc"].str.zfill(3)
        d["w"] = d["ws"].str[-2:] + d["wc"].str.zfill(3)
        d = d[d["h"].isin(IDX)]
        d["n"] = pd.to_numeric(d["n"], errors="coerce")
        d[["h", "w", "n"]].to_csv(cache, index=False)
    return pd.read_csv(cache, dtype={"h": str, "w": str})


def model_flows():
    header, a = read_pack.load_pack("nyc")
    g = header["gravity"]
    nz = header["zones"]
    zone = a["zone"].astype(np.int64)
    gx, gy = a["zone_gx"].astype(np.int64), a["zone_gy"].astype(np.int64)
    cc = np.fromfile(os.path.join(realnet.WORK, "cell_county.u8"), np.uint8)
    nc = len(ORDER)
    zh = np.zeros((nz, nc))
    zw = np.zeros((nz, nc))
    np.add.at(zh, (zone, cc), a["commute_home"].astype(np.float64))
    np.add.at(zw, (zone, cc), a["commute_work"].astype(np.float64))
    zh /= np.maximum(zh.sum(1, keepdims=True), 1e-9)
    zw /= np.maximum(zw.sum(1, keepdims=True), 1e-9)
    row, col = a["zone_row"].astype(np.float64), a["zone_col"].astype(np.float64)
    zk = g["zone_m"] / 1000.0
    F = np.zeros((nc, nc))
    for s in range(0, nz, 300):
        e = min(nz, s + 300)
        d = zk * np.sqrt((gx[s:e, None] - gx[None, :]) ** 2 + (gy[s:e, None] - gy[None, :]) ** 2)
        d[d == 0] = g["intrazonal_km"]
        t = row[s:e, None] * col[None, :] * np.exp(-d / g["decay_km"]) * d ** -g.get("decay_pow", 0.0)
        F += zh[s:e].T @ t @ zw
    return F


def lodes_flows():
    cache = os.path.join(realnet.WORK, "lodes_county_flows.npy")
    if os.path.exists(cache):
        return np.load(cache)
    import fit_decay
    od = fit_decay.lodes_od("nyc")
    h = od["h_geocode"].str[:5].map(IDX)
    w = od["w_geocode"].str[:5].map(IDX)
    F = np.zeros((len(ORDER), len(ORDER)))
    np.add.at(F, (h.to_numpy(int), w.to_numpy(int)), od["S000"].to_numpy(float))
    np.save(cache, F)
    return F


def main():
    acs = acs_flows()
    A = np.zeros((len(ORDER), len(ORDER)))
    out_of_region = np.zeros(len(ORDER))
    for r in acs.itertuples():
        if r.w in IDX:
            A[IDX[r.h], IDX[r.w]] += r.n
        else:
            out_of_region[IDX[r.h]] += r.n
    M = model_flows()
    L = lodes_flows()
    man = IDX["36061"]
    rows = []
    for i, f in enumerate(ORDER):
        rows.append((realnet.COUNTIES[f], A[i, i] / A[i].sum(), L[i, i] / L[i].sum(), M[i, i] / M[i].sum(),
                     A[i, man] / A[i].sum(), L[i, man] / L[i].sum(), M[i, man] / M[i].sum(), out_of_region[i] / (A[i].sum() + out_of_region[i])))
    df = pd.DataFrame(rows, columns=["county", "own_acs", "own_lodes", "own_model", "manh_acs", "manh_lodes", "manh_model", "acs_outside"])
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, formatters={c: "{:.1%}".format for c in df.columns[1:]}))
    tot = lambda X: np.trace(X) / X.sum()
    print(f"\nown county, all: ACS {tot(A):.1%}, LODES {tot(L):.1%}, model {tot(M):.1%}")
    print(f"to Manhattan, all: ACS {A[:, man].sum() / A.sum():.1%}, LODES {L[:, man].sum() / L.sum():.1%}, model {M[:, man].sum() / M.sum():.1%}")
    # ACS without work from home: subtract each county's home workers from its own-county cell
    acs_wfh = realnet.acs_counties()
    A2 = A.copy()
    for i, f in enumerate(ORDER):
        A2[i, i] -= acs_wfh.loc[f, "B08301_E021"] * A[i].sum() / acs_wfh.loc[f, "B08301_E001"]
    print(f"ACS less work from home (2019-2023 share applied to 2016-2020 flows): own county {tot(A2):.1%}, "
          f"to Manhattan {A2[:, man].sum() / A2.sum():.1%}")
    np.save(os.path.join(realnet.WORK, "acs_county_flows.npy"), A)
    np.save(os.path.join(realnet.WORK, "model_county_flows.npy"), M)
    df.to_csv(os.path.join(realnet.WORK, "county_flows.csv"), index=False)


if __name__ == "__main__":
    main()
