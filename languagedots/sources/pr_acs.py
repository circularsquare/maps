"""Puerto Rico, Puerto Rico Community Survey 2020-2024 5-year, language spoken at home
-> data/normalized/pr.csv and pr_split.csv.

    python sources/pr_acs.py [--fetch]

The Puerto Rico Community Survey is the American Community Survey as run in Puerto Rico: the
same questionnaire (in Spanish), the same tables, released with the ACS. So this is
sources/us_acs.py's method on Puerto Rico's rows, and it imports that script's group lists and
crosswalks rather than repeating them (sources/us.md is the record of how they were built and
checked):

  C16001  tract (~940)   English only, Spanish and 11 groups
  B16001  PUMA (~18)     the same people in 42 groups
  PUMS    PUMA           person microdata, LANP codes; Puerto Rico has its own person file
                         (csv_ppr.zip, ~18 MB), the only thing this script downloads

The two summary files are the ones sources/us_acs.py fetched into data/raw/us/ (they cover every
state and Puerto Rico); read in place. The question, of everyone aged 5 and over: does this
person speak a language other than English at home, and if so which. A person who speaks Spanish
and English at home is counted under Spanish, so "English" is English only.

WHAT THIS WRITES (the same shapes as us.csv / us_split.csv)

  pr.csv        tract x C16001 group, as published
  pr_split.csv  per PUMA and C16001 group, each LANP code's share:
                B16001(PUMA, b) / B16001(PUMA, b's C group) x PUMS(PUMA, code) / PUMS(PUMA, b).
                Where the PUMA's PUMS sample has nobody in b although B16001 has people, Puerto
                Rico's own PUMS mix of b is used, then the 50 states' (data/raw/us/
                pums_lanp_puma.csv, written by us_acs.py); `fallback` says which.

CHECKS (asserted unless said)
  1. tracts sum to Puerto Rico's row in all 13 C16001 rows, exactly
  2. tracts sum to their PUMA in every row, exactly (also proves the tract -> PUMA key)
  3. B16001 groups sum to their C16001 group at every PUMA, exactly
  4. PUMS against B16001 for Puerto Rico per B group (printed; bar 5% on groups over 100,000)
  5. Puerto Rico's PUMS has no LANP code outside us_acs.LANP_B
"""
import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "sources"))
from us_acs import (B_GROUPS, B_LABEL, C_GROUPS, ENGLISH, LANP_B, PUMS_AGG as US_PUMS_AGG,  # noqa: E402
                    RD_GEO, lanp_labels, read_sf)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAW = HERE / "data" / "raw" / "pr"
NORM = HERE / "data" / "normalized"
PUMS_ZIP = RAW / "csv_ppr_2024_5y.zip"
PUMS_URL = "https://www2.census.gov/programs-surveys/acs/data/pums/2024/5-Year/csv_ppr.zip"
PUMS_AGG = RAW / "pums_lanp_puma.csv"
ST = "72"


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    if PUMS_ZIP.exists() and PUMS_ZIP.stat().st_size > 0:
        return
    print(f"  fetching {PUMS_URL}")
    req = urllib.request.Request(PUMS_URL, headers={"User-Agent": "Mozilla/5.0"})
    tmp = PUMS_ZIP.with_suffix(".part")
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as fh:
        while chunk := r.read(1 << 22):
            fh.write(chunk)
    tmp.replace(PUMS_ZIP)


def pums_aggregate():
    """Weighted persons aged 5+ by (PUMA, LANP), Puerto Rico's person file; cached."""
    if PUMS_AGG.exists():
        return pd.read_csv(PUMS_AGG, dtype={"st": str, "puma": str, "lanp": str})
    with zipfile.ZipFile(PUMS_ZIP) as z:
        members = [n for n in z.namelist() if n.endswith(".csv")]
        assert len(members) == 1, members
        with z.open(members[0]) as fh:
            d = pd.read_csv(fh, usecols=["STATE", "PUMA", "PWGTP", "AGEP", "LANX", "LANP"],
                            dtype={"STATE": str, "PUMA": str, "LANX": str, "LANP": str})
    d = d[d["AGEP"] >= 5].copy()
    d["lanp"] = d["LANP"].fillna("")
    d.loc[d["LANX"] == "2", "lanp"] = "EN"
    if (d["lanp"] == "").any():
        raise SystemExit(f"{int((d['lanp'] == '').sum())} persons 5+ with no LANX/LANP")
    if set(d["STATE"].str.zfill(2)) != {ST}:
        raise SystemExit(f"states in the PR person file: {sorted(set(d['STATE']))}")
    agg = d.groupby(["STATE", "PUMA", "lanp"]).agg(w=("PWGTP", "sum"), n=("PWGTP", "size")).reset_index()
    agg = agg.rename(columns={"STATE": "st", "PUMA": "puma"})
    agg["st"] = agg["st"].str.zfill(2)
    agg.to_csv(PUMS_AGG, index=False)
    return pd.read_csv(PUMS_AGG, dtype={"st": str, "puma": str, "lanp": str})


def tract_puma_key(tract_ids):
    rel = pd.read_csv(RD_GEO / "tract_to_puma_2020.txt", dtype=str, encoding="utf-8-sig")
    rel = rel[rel["STATEFP"] == ST]
    key = dict(zip(rel["STATEFP"] + rel["COUNTYFP"] + rel["TRACTCE"], rel["STATEFP"] + rel["PUMA5CE"]))
    out = tract_ids.map(key)
    if out.isna().any():
        raise SystemExit(f"{int(out.isna().sum())} tracts with no PUMA: {list(tract_ids[out.isna()][:5])}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    if args.fetch:
        fetch()
    NORM.mkdir(parents=True, exist_ok=True)

    # ---- C16001 ----
    c = read_sf("acsdt5y2024-c16001.dat", "C16001", 38)
    c_cols = {2: ENGLISH, **C_GROUPS}
    st = c[c["GEO_ID"] == f"0400000US{ST}"].iloc[0]
    assert sum(st[col] for col in c_cols) == st[1], "Puerto Rico row does not add up"
    tr = c[c["GEO_ID"].str.startswith(f"1400000US{ST}")].copy()
    tr["geo_id"] = tr["GEO_ID"].str[9:]
    n_all = len(tr)
    tr = tr[tr[1] > 0].copy()
    print(f"C16001: {n_all:,} Puerto Rico tracts, {len(tr):,} with people aged 5+ "
          f"({st[1]:,} in Puerto Rico's row)")
    for col in c_cols:                                                   # check 1
        if tr[col].sum() != st[col]:
            raise SystemExit(f"check 1: tracts {tr[col].sum():,} != Puerto Rico {st[col]:,} in {c_cols[col]}")
    print("  check 1 ok: tracts sum to Puerto Rico's row in all 13 rows")
    for col, lab in c_cols.items():
        print(f"    {lab:45s} {st[col]:>10,}  {st[col] / st[1]:7.3%}")
    tr["puma"] = tract_puma_key(tr["geo_id"])

    pu_c = c[c["GEO_ID"].str.startswith(f"795P200US{ST}")].copy()
    pu_c["puma"] = pu_c["GEO_ID"].str[9:]
    pu_c = pu_c.set_index("puma")
    sums = tr.groupby("puma")[list(c_cols)].sum()                        # check 2
    if set(sums.index) != set(pu_c.index):
        raise SystemExit(f"check 2: PUMA sets differ: {sorted(set(sums.index) ^ set(pu_c.index))}")
    resid = sums - pu_c.loc[sums.index, list(c_cols)]
    moved = resid[(resid != 0).any(axis=1)]
    # As in the 50 states (sources/us.md §3), the 2020 relationship file puts a tract or two in the
    # neighbouring PUMA from the one the 2024 tables count it in: PUMAs off in equal and opposite
    # pairs. That only changes which PUMA's mix a tract is split by; bounded, asserted pairwise.
    if moved.sum().abs().max() != 0 or moved.abs().to_numpy().sum() > 2 * 500:
        raise SystemExit(f"check 2: tracts do not sum to their PUMA, and not by pairwise moves:\n{moved}")
    print(f"  check 2 ok: tracts sum exactly to {len(sums) - len(moved)} of {len(sums)} PUMAs in every "
          f"row; {list(moved.index)} differ by a move between neighbours of "
          f"{int(moved.abs().to_numpy().sum() / 2):,} people 5+ (bounded at 500)")

    long = tr.melt(id_vars=["geo_id", "puma"], value_vars=list(c_cols), var_name="col", value_name="count")
    long["source_category"] = long["col"].map(c_cols)
    long["geo_level"] = "tract"
    long = long[long["count"] > 0]
    long[["geo_id", "geo_level", "puma", "source_category", "count"]].to_csv(NORM / "pr.csv", index=False)
    print(f"  wrote pr.csv: {len(long):,} rows")

    # ---- B16001 at PUMA; check 3 ----
    b = read_sf("acsdt5y2024-b16001.dat", "B16001", 128)
    bst = b[b["GEO_ID"] == f"0400000US{ST}"].iloc[0]
    pb = b[b["GEO_ID"].str.startswith(f"795P200US{ST}")].copy()
    pb["puma"] = pb["GEO_ID"].str[9:]
    pb = pb.set_index("puma")
    for ccol, cname in C_GROUPS.items():
        bcols = [bc for bc, (_, cg) in B_GROUPS.items() if cg == cname]
        d = (pb[bcols].sum(axis=1) - pu_c.loc[pb.index, ccol]).abs().sum()
        if d:
            raise SystemExit(f"check 3: B16001 {bcols} do not sum to C16001 {cname} (abs diff {d:,})")
    print(f"  check 3 ok: the 42 B16001 groups sum to the 12 C16001 groups at all {len(pb)} PUMAs")

    # ---- PUMS; checks 4 and 5 ----
    labels = lanp_labels()
    pums = pums_aggregate()
    pums["puma"] = pums["st"] + pums["puma"].str.zfill(5)
    nonen = pums[pums["lanp"] != "EN"].copy()
    nonen["code"] = nonen["lanp"].astype(int)
    unknown = sorted(set(nonen["code"]) - set(LANP_B))
    if unknown:
        raise SystemExit(f"check 5: LANP codes with no B16001 group: {unknown}")
    nonen["bcol"] = nonen["code"].map(LANP_B)
    en_w = pums.loc[pums["lanp"] == "EN", "w"].sum()
    print(f"  PUMS: {pums['n'].sum():,} persons 5+ in {pums['puma'].nunique()} PUMAs; English only "
          f"{en_w:,.0f} against C16001's {st[2]:,} ({en_w / st[2] - 1:+.2%})")
    print("  check 4, PUMS against B16001 for Puerto Rico, per B group with anyone in either:")
    pw = nonen.groupby("bcol")["w"].sum()
    bad = []
    for bcol, lab in B_LABEL.items():
        pub, got = bst[bcol], pw.get(bcol, 0.0)
        if not pub and not got:
            continue
        r = got / pub - 1 if pub else float("inf")
        flag = "  !!" if pub > 100_000 and abs(r) > 0.05 else ""
        print(f"    {lab[:60]:60s} {pub:>10,} {got:>10,.0f} {r:+8.1%}{flag}")
        if flag:
            bad.append(lab)
    if bad:
        raise SystemExit(f"check 4: {bad}")
    print("  PUMS codes in Puerto Rico (weighted, aged 5+, not English only):")
    for code, w in nonen.groupby("code")["w"].sum().sort_values(ascending=False).items():
        print(f"    {code:>5} {labels[code][:50]:50s} {w:>10,.0f}")

    # ---- the split ----
    bl = pb[list(B_GROUPS)].reset_index().melt(id_vars="puma", var_name="bcol", value_name="b")
    bl["cgroup"] = bl["bcol"].map(lambda x: B_GROUPS[x][1])
    bl["c"] = bl.groupby(["puma", "cgroup"])["b"].transform("sum")
    bl = bl[bl["b"] > 0]
    bl["b_share"] = bl["b"] / bl["c"]

    us = pd.read_csv(US_PUMS_AGG, dtype={"st": str, "puma": str, "lanp": str})
    us = us[us["lanp"] != "EN"].copy()
    us["code"] = us["lanp"].astype(int)
    us["bcol"] = us["code"].map(LANP_B)
    gp = {k: g for k, g in nonen.groupby(["puma", "bcol"])[["code", "w"]]}
    gs = {k: g for k, g in nonen.groupby("bcol")[["code", "w"]]}
    gu = {k: g for k, g in us.groupby("bcol")[["code", "w"]]}
    gs = {k: g.groupby("code", as_index=False)["w"].sum() for k, g in gs.items()}
    gu = {k: g.groupby("code", as_index=False)["w"].sum() for k, g in gu.items()}

    rows, fb = [], {"puma": 0.0, "puerto_rico": 0.0, "states": 0.0}
    for r in bl.itertuples(index=False):
        if (r.puma, r.bcol) in gp:
            g, how = gp[(r.puma, r.bcol)].groupby("code", as_index=False)["w"].sum(), "puma"
        elif r.bcol in gs:
            g, how = gs[r.bcol], "puerto_rico"
        else:
            g, how = gu[r.bcol], "states"
        fb[how] += r.b
        w = g["w"].to_numpy(dtype=float)
        for code, s in zip(g["code"].to_numpy(), w / w.sum()):
            rows.append((r.puma, r.cgroup, B_LABEL[r.bcol], int(code), labels[int(code)],
                         r.b_share * s, how))
    sp = pd.DataFrame(rows, columns=["puma", "c_group", "b_group", "lanp", "source_category",
                                     "share", "fallback"])
    chk = sp.groupby(["puma", "c_group"])["share"].sum()
    assert (chk - 1).abs().max() < 1e-9, chk[(chk - 1).abs() >= 1e-9].head()
    need = set(map(tuple, long.loc[long["source_category"] != ENGLISH, ["puma", "source_category"]]
                   .to_numpy()))
    have = set(map(tuple, sp[["puma", "c_group"]].to_numpy()))
    if need - have:
        raise SystemExit(f"tract groups with no PUMA mix: {sorted(need - have)[:5]}")
    sp.to_csv(NORM / "pr_split.csv", index=False)
    tot_b = sum(fb.values())
    print(f"  wrote pr_split.csv: {len(sp):,} rows; people in B16001 groups whose mix came from the "
          + ", ".join(f"{k} {v:,.0f} ({v / tot_b:.2%})" for k, v in fb.items()))


if __name__ == "__main__":
    main()
