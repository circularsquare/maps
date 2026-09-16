"""DR Congo: the Enquête 1-2-3's household heads (2005 and 2012), religion by province, on COD-PS 2024.

Reads data/raw/cd/ and writes data/normalized/cd.csv. `sources/cd.md` is the record.

## WHAT THIS IS

No census of the DRC has been held since 1984, and no census or survey table of religion by
province has been published (the UNSD oracle has no DRC row; the EDS-RDC 2023-24 final report,
FR393, prints religion only nationally, Tableau 3.1 p.32; the MICS-Palu 2017-18 report asks the
head's religion, HC1A, and tabulates none; the 2012 Enquête 1-2-3 results report has no religion
table). The one open subnational religion table is the U.S. Census Bureau's tabulation of the
Institut National de la Statistique's Enquête 1-2-3 microdata on HDX (sources.md §11h):

    31,755 household heads, 2005 (11,636 households) and 2012 (20,119) pooled by USCB,
    summed at 26 provinces and 164 districts; 16 districts have no sampled household.

The 2012 household roster asks `M27 Religion pratiquée` of every member (1 Catholique,
2 Protestante, 3 Kimbanguiste, 4 Musulmane, 5 Autre Chrétiens, 6 Animiste, 7 Autre religion,
8 Sans religion); USCB counted the heads only. The person-level files sit behind the University of
Antwerp's registration form and are not read here.

§11h refused this table on 2026-09-05 ("not a census and not people"). That was before any survey
was drawn on this map. `do` (ENHOGAR-MICS6) and `hn` (ENDESA-MICS) are drawn from exactly this
shape since: the household head's religion, applied to the household. Every row is `modelled`.

## HOW A PROVINCE'S SHARES ARE BUILT

1. District shares of heads, unweighted: USCB summed raw responses and the design weights are not
   in the table.
2. Province share = the districts' shares weighted by each district's population (COD-PS 2019, the
   `Population Estimates` sheet of the same workbook, same GEO_MATCH keys), over sampled districts.
   The 2012 sample was fixed per district rather than proportional, so pooling heads straight would
   over-weight small districts: it moves Ituri's Catholic share by 9.5 points.
3. The split-half decides which answers carry their own geography (below). A failing answer is drawn
   at its national share in every province, and the carried answers are scaled so each row sums.
4. The shares are laid on COD-PS 2024 (OCHA's `drc-hpc-projection-population-2024.xlsx`, 519 health
   zones, 117,808,872 people), summed to province. `Manquant` (20 heads) stays in the table as its
   own row and is not drawn.

## THE SPLIT-HALF, WITHOUT MICRODATA

No cluster ids survive USCB's summing, so the halving is one level up: each province's sampled
districts are dealt at random into two halves (400 halvings), and the Spearman across provinces
between the halves' unweighted shares is taken per answer, median over halvings. The null deals the
districts into provinces at random, keeping each province's district count
(`stability.cluster_null`, spec §12 "WHERE THE SAMPLING UNITS NEST INSIDE THE DRAWN UNITS, THE NULL
REGROUPS THEM"), 2,000 times. Kinshasa is one district and cannot be halved; it sits out of the
rank test and is in the chi-square. Vetoes: chi-square across the 26 provinces, and no single
district holding half of an answer. `EXPECT_FLAT` pins the verdict so a changed file is loud.

Usage:
    python sources/cd.py --fetch    two GETs, ~0.9 MB
    python sources/cd.py            rebuild from data/raw/cd/
"""

import csv
import os
import re
import sys
import unicodedata

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd

RAW = os.path.join(ROOT, "data", "raw", "cd")
OUT = os.path.join(ROOT, "data", "normalized", "cd.csv")

SOURCE_ID = "cd_ins_enquete123_2005_2012_uscb"
YEAR = 2012
BASIS = "self_id"
COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")
XLSX_NAME = "democratic-republic-of-the-congo_uscb_202103.xlsx"
POP_NAME = "drc-hpc-projection-population-2024.xlsx"
DOWNLOADS = {
    XLSX_NAME: ("https://data.humdata.org/dataset/881bdaa3-6d4c-408b-88dc-aed0ecc0d119/resource/"
                "a4a72c31-af0d-4b0d-975b-9eadf6d5c103/download/"
                "democratic-republic-of-the-congo_uscb_202103.xlsx", 250_000),
    POP_NAME: ("https://data.humdata.org/dataset/d1160fa9-1d58-4f96-9df5-edbff2e80895/resource/"
               "9509b41f-efb2-4305-ac43-1e735e3efc60/download/"
               "drc-hpc-projection-population-2024.xlsx", 500_000),
}
XLSX = os.path.join(RAW, XLSX_NAME)
POP = os.path.join(RAW, POP_NAME)

# USCB column -> the survey's own label, as the workbook's data dictionary gives each column's
# "Original field name". Asserted against the dictionary in read_uscb().
CATEGORIES = [
    ("RLG_CATH", "Catholique"),
    ("RLG_PROT", "Protestant"),
    ("RLG_KIMB", "Kimbanguiste"),
    ("RLG_MUS", "Musulman"),
    ("RLG_OCHR", "Autre chrétien"),
    ("RLG_ANIM", "Animiste"),
    ("RLG_OTHR", "Autre réligion"),
    ("RLG_WOUT", "Sans religion"),
    ("RLG_NDTA", "Manquant"),
]
KEYS = [k for k, _ in CATEGORIES]
LABEL = dict(CATEGORIES)
NONRESPONSE = "RLG_NDTA"
TESTED = [k for k in KEYS if k != NONRESPONSE]

HEADS = 31_755
N_PROVINCES = 26
N_DISTRICTS = 164
N_UNSAMPLED = 16
COD_PS_2024 = 117_808_872
COD_PS_2024_ROWS = 519

S_HALVINGS = 400
CELL_CAP = 0.5
ALPHA = 0.05
# Measured 2026-09-15 (cb8b206e-cd): Animiste, 167 heads, median +0.264 against a null 95th
# percentile of +0.278, p 0.059. Every other answer p <= 0.0035.
EXPECT_FLAT = {"RLG_ANIM"}

# COD-PS 2019 (USCB's sheet) against COD-PS 2024, per province: a sanity band beside the join
# witness, which is the rank test in join(). Measured 2026-09-15: 1.179 nationally, Ituri 0.84 to
# Kongo-Central 1.98 (Lualaba 1.69, Kinshasa 1.56); the 2024 projection rebased the west, it is not
# growth. Spearman +0.902 against the best of 2,000 shuffled pairings +0.709.
POP_RATIO_BAND = (0.75, 2.10)

# EDS-RDC III 2023-24, FR393 Tableau 3.1 (PDF p.78, printed p.32): women and men 15-49, weighted,
# per cent. Printed beside the drawn national shares, never used.
EDS_2023 = {
    "Catholique": (23.7, 25.8),
    "Protestante": (27.4, 26.7),
    "Église non dénominationelle": (39.1, 33.2),
    "Autre religion chrétienne": (5.8, 6.8),
    "Animisme/religion traditionnelle": (3.0, 2.5),
    "Sans religion": (1.0, 3.5),
    "Autre": (0.1, 1.6),
}


def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.casefold())


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    for name, (url, least) in DOWNLOADS.items():
        dst = os.path.join(RAW, name)
        if os.path.exists(dst) and os.path.getsize(dst) > least:
            print(f"  have {name} ({os.path.getsize(dst):,} bytes)")
            continue
        r = requests.get(url, headers={"User-Agent": UA}, timeout=300)
        r.raise_for_status()
        if r.content[:4] != b"PK\x03\x04" or len(r.content) < least:   # §5a: a 200 is not a download
            raise SystemExit(f"{name}: starts {r.content[:16]!r}, {len(r.content):,} bytes")
        with open(dst + ".part", "wb") as fh:
            fh.write(r.content)
        os.replace(dst + ".part", dst)
        print(f"  got  {name} ({len(r.content):,} bytes)")


def read_uscb():
    """The Tribe and Religion sheet, checked, with COD-PS 2019 per unit beside it."""
    x = pd.read_excel(XLSX, sheet_name="Tribe and Religion", header=0, skiprows=[1])
    lv = pd.to_numeric(x["ADM_LEVEL"], errors="coerce")
    got = {int(k): int(v) for k, v in lv.value_counts().items()}
    if got != {0: 1, 1: N_PROVINCES, 2: N_DISTRICTS}:
        raise SystemExit(f"Tribe and Religion levels {got}, expected 1 / {N_PROVINCES} / {N_DISTRICTS}")
    x["level"] = lv.astype(int)
    cols = KEYS + ["TRB_SSIZE"]
    for c in cols:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    if int((x[cols] < 0).sum().sum()):
        raise SystemExit("a negative cell: USCB's -999 sentinel is in this release")

    unsampled = x[x["TRB_SSIZE"].isna()]
    if len(unsampled) != N_UNSAMPLED or (unsampled["level"] != 2).any():
        raise SystemExit(f"{len(unsampled)} rows with no sample, expected {N_UNSAMPLED} districts")
    if unsampled[KEYS].notna().any().any():
        raise SystemExit("a district with no sample size carries a religion count")
    s = x[x["TRB_SSIZE"].notna()]
    off = s[s[KEYS].sum(axis=1) != s["TRB_SSIZE"]]
    if len(off):
        raise SystemExit(f"{len(off)} rows where the nine answers do not sum to the sample size")
    nat = s[s["level"] == 0].iloc[0]
    if int(nat["TRB_SSIZE"]) != HEADS:
        raise SystemExit(f"national sample {int(nat['TRB_SSIZE']):,}, expected {HEADS:,}")
    prov = s[s["level"] == 1].set_index("ADM1_NAME")
    dist = s[s["level"] == 2]
    if not all(int(prov[k].sum()) == int(nat[k]) for k in KEYS):
        raise SystemExit("the provinces do not sum to the national row")
    dsum = dist.groupby("ADM1_NAME")[KEYS].sum().reindex(prov.index)
    if not (dsum == prov[KEYS]).all().all():
        raise SystemExit("a province's districts do not sum to its row")
    print(f"USCB Enquête 1-2-3 heads: {HEADS:,} over {N_PROVINCES} provinces and "
          f"{len(dist)} sampled districts ({N_UNSAMPLED} unsampled); every row partitions exactly")

    dd = pd.read_excel(XLSX, sheet_name="Data Dictionary", header=None)
    text = dd.apply(lambda r: " | ".join(str(c) for c in r if pd.notna(c)), axis=1)
    for key, label in CATEGORIES:
        row = text[text.str.startswith(key + " |")]
        if len(row) != 1 or f'Original field name: "{label}."' not in row.iloc[0]:
            raise SystemExit(f"the data dictionary does not give {key} as {label!r}")

    pe = pd.read_excel(XLSX, sheet_name="Population Estimates", header=0, skiprows=[1])
    pe["ES19_BTOTL"] = pd.to_numeric(pe["ES19_BTOTL"], errors="coerce")
    x = x.merge(pe[["GEO_MATCH", "ES19_BTOTL"]], on="GEO_MATCH", how="left", validate="1:1")
    if x["ES19_BTOTL"].isna().any() or (x["ES19_BTOTL"] <= 0).any():
        raise SystemExit("a unit with no COD-PS 2019 population in the Population Estimates sheet")
    return x


def province_shares(x):
    """District shares weighted by district population inside each province. (province x key)."""
    dist = x[(x["level"] == 2) & x["TRB_SSIZE"].notna()]
    rows, info = {}, {}
    for prov, d in dist.groupby("ADM1_NAME"):
        w = d["ES19_BTOTL"] / d["ES19_BTOTL"].sum()
        rows[prov] = {k: float((w * d[k] / d["TRB_SSIZE"]).sum()) for k in KEYS}
        alld = x[(x["level"] == 2) & (x["ADM1_NAME"] == prov)]
        info[prov] = dict(heads=int(d["TRB_SSIZE"].sum()), districts=len(d), of=len(alld),
                          unsampled_pop=float(alld.loc[alld["TRB_SSIZE"].isna(), "ES19_BTOTL"].sum()
                                              / alld["ES19_BTOTL"].sum()))
    share = pd.DataFrame(rows).T[KEYS]
    if not np.allclose(share.sum(axis=1), 1.0):
        raise SystemExit("weighted province shares do not sum to one")
    prov = x[x["level"] == 1].set_index("ADM1_NAME")
    raw = prov[KEYS].div(prov["TRB_SSIZE"], axis=0).reindex(share.index)
    diff = (share - raw).abs()
    print("\n  weighting districts by population inside each province moves the shares by at most:")
    for k in TESTED:
        p = diff[k].idxmax()
        print(f"    {LABEL[k]:<16}{100 * diff.loc[p, k]:5.1f} points ({p})")
    return share, pd.DataFrame(info).T


def stability(x):
    """Split-half on districts inside provinces, regrouping null. Returns {key: verdict dict}."""
    from scipy.stats import rankdata

    import stability as shared

    dist = x[(x["level"] == 2) & x["TRB_SSIZE"].notna()]
    nd = dist.groupby("ADM1_NAME").size()
    halvable = sorted(nd[nd >= 2].index)
    d = dist[dist["ADM1_NAME"].isin(halvable)].sort_values(["ADM1_NAME", "GEO_MATCH"])
    M = d[TESTED].to_numpy(dtype=float)
    T = (d["TRB_SSIZE"] - d[NONRESPONSE]).to_numpy(dtype=float)
    sizes = d.groupby("ADM1_NAME", sort=True).size().reindex(halvable).to_numpy()
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    D = M.shape[0]

    rng = np.random.default_rng(shared.STAB_SEED)
    P = np.zeros((S_HALVINGS, D))
    for s in range(S_HALVINGS):
        for r, n in enumerate(sizes):
            P[s, starts[r] + rng.permutation(n)[: n // 2]] = 1.0

    def median_rho(Mo, To):
        A = np.add.reduceat(P[:, :, None] * Mo[None, :, :], starts, axis=1)
        B = np.add.reduceat((1 - P)[:, :, None] * Mo[None, :, :], starts, axis=1)
        TA = np.add.reduceat(P * To[None, :], starts, axis=1)
        TB = np.add.reduceat((1 - P) * To[None, :], starts, axis=1)
        ra, rb = rankdata(A / TA[:, :, None], axis=1), rankdata(B / TB[:, :, None], axis=1)
        ra -= ra.mean(axis=1, keepdims=True)
        rb -= rb.mean(axis=1, keepdims=True)
        den = np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            rho = np.where(den > 0, (ra * rb).sum(axis=1) / den, np.nan)
        return np.nanmedian(rho, axis=0)

    obs = median_rho(M, T)
    null = shared.cluster_null(lambda perm: median_rho(M[perm], T[perm]), D, shared.STAB_PERM, rng)

    prov = x[x["level"] == 1].sort_values("ADM1_NAME")
    nat = x[x["level"] == 0].iloc[0]
    left = sorted(set(prov["ADM1_NAME"]) - set(halvable))
    print(f"\n  split-half: {S_HALVINGS} random halvings of the sampled districts inside each of "
          f"{len(halvable)} provinces (left out, one district: {', '.join(left)}); null "
          f"{shared.STAB_PERM:,} regroupings of the {D} districts")
    print(f"    {'answer':<16}{'heads':>7}{'median':>8}{'null95':>8}{'p':>8}{'chi2 p':>10}"
          f"{'top dist':>9}  verdict")
    out = {}
    for j, k in enumerate(TESTED):
        p, q95 = shared.permutation_p(obs[j], null[:, j], ALPHA)
        chi = shared.chi2_p(prov[k].to_numpy(), (prov["TRB_SSIZE"] - prov[NONRESPONSE]).to_numpy())
        top_row = dist.loc[dist[k].idxmax()]
        top = float(top_row[k] / nat[k])
        carries = p < ALPHA and np.isfinite(chi) and chi < ALPHA and top < CELL_CAP
        verdict = "own geography" if carries else "national share in every province"
        out[k] = dict(median=float(obs[j]), null95=q95, p=p, chi=chi, top=top,
                      top_where=f"{top_row['ADM2_NAME']}, {top_row['ADM1_NAME']}", carries=carries)
        print(f"    {LABEL[k]:<16}{int(nat[k]):>7,}{obs[j]:>8.3f}{q95:>8.3f}{p:>8.4f}{chi:>10.1e}"
              f"{top:>9.2f}  {verdict} (largest district: {out[k]['top_where']})")
    flat = {k for k, v in out.items() if not v["carries"]}
    if flat != EXPECT_FLAT:
        raise SystemExit(f"the answers failing the split-half are {sorted(flat)}, pinned as "
                         f"{sorted(EXPECT_FLAT)}; re-read the verdicts and update EXPECT_FLAT")
    return out


def read_pop2024():
    ps = pd.read_excel(POP, header=0)
    ps.columns = [str(c).strip() for c in ps.columns]
    for c in ("Province", "Code Province", "Code Terrtoire", "Pcode ZS", "Population 2024"):
        if c not in ps.columns:
            raise SystemExit(f"COD-PS 2024 has no column {c!r}: {list(ps.columns)[:10]}")
    ps["Population 2024"] = pd.to_numeric(ps["Population 2024"], errors="coerce")
    if len(ps) != COD_PS_2024_ROWS or int(ps["Population 2024"].sum()) != COD_PS_2024:
        raise SystemExit(f"COD-PS 2024: {len(ps)} rows, {ps['Population 2024'].sum():,.0f} people; "
                         f"expected {COD_PS_2024_ROWS} and {COD_PS_2024:,}")
    names = ps.groupby("Code Province")["Province"].nunique()
    if len(names) != N_PROVINCES or (names != 1).any():
        raise SystemExit("COD-PS 2024 does not give 26 provinces with one name each")
    g = ps.groupby(["Code Province", "Province"])["Population 2024"].sum().reset_index()
    return g.rename(columns={"Code Province": "pcode", "Province": "name", "Population 2024": "pop"})


def join(share_index, x, pop):
    """USCB province names -> COD-PS pcodes, by folded name, 1:1, with a population witness."""
    by_fold = {fold(n): r for r, n in zip(pop["pcode"], pop["name"])}
    if len(by_fold) != N_PROVINCES:
        raise SystemExit("two COD-PS province names fold to the same key")
    key = {p: by_fold.get(fold(p)) for p in share_index}
    miss = sorted(p for p, v in key.items() if v is None)
    if miss or len(set(key.values())) != N_PROVINCES:
        raise SystemExit(f"USCB provinces with no COD-PS 2024 match: {miss}")

    prov = x[x["level"] == 1].set_index("ADM1_NAME")
    p19 = prov["ES19_BTOTL"].rename(index=key)
    p24 = pop.set_index("pcode")["pop"]
    ratio = (p24 / p19.reindex(p24.index)).sort_values()
    nat = p24.sum() / p19.sum()
    print(f"\n  join witness: COD-PS 2024 over COD-PS 2019 per province (nationally {nat:.3f}); "
          f"the name join is the only thing pairing them")
    nm = pop.set_index("pcode")["name"]
    print("    " + ", ".join(f"{nm[c]} {v:.2f}" for c, v in ratio.items()))
    from stability import rho
    r_obs = rho(p19.reindex(p24.index).to_numpy(), p24.to_numpy())
    rng = np.random.default_rng(0)
    shuffled = [rho(rng.permutation(p19.reindex(p24.index).to_numpy()), p24.to_numpy())
                for _ in range(2000)]
    print(f"    Spearman 2019 against 2024 over the 26: {r_obs:+.3f}; best of 2,000 shuffled "
          f"pairings {max(shuffled):+.3f}")
    if r_obs <= max(shuffled):
        raise SystemExit("the 2019 and 2024 populations do not rank the provinces together; "
                         "the name join may have paired the wrong provinces")
    lo, hi = POP_RATIO_BAND
    out = ratio[(ratio < lo) | (ratio > hi)]
    if len(out):
        raise SystemExit(f"provinces outside the 2024/2019 band {POP_RATIO_BAND}: {out.to_dict()}")
    return key


def compose(share, flat, pop_by_prov):
    """Flat answers at their national share; carried answers scaled so every row sums to one."""
    w = pop_by_prov.reindex(share.index)
    national = share.mul(w, axis=0).sum() / w.sum()
    out = share.copy()
    carried = [k for k in TESTED if k not in flat]
    for p in out.index:
        nd = out.loc[p, NONRESPONSE]
        room_now = 1.0 - nd - sum(share.loc[p, k] for k in flat)
        room_new = 1.0 - nd - sum(national[k] for k in flat)
        for k in flat:
            out.loc[p, k] = national[k]
        for k in carried:
            out.loc[p, k] = share.loc[p, k] * room_new / room_now
    if not np.allclose(out.sum(axis=1), 1.0):
        raise SystemExit("composed shares do not sum to one")
    return out, national


def main():
    from lr import round_within_rows

    if "--fetch" in sys.argv:
        fetch()
    for p in (XLSX, POP):
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing; run: python sources/cd.py --fetch")

    x = read_uscb()
    share, info = province_shares(x)
    verdicts = stability(x)
    flat = {k for k, v in verdicts.items() if not v["carries"]}

    pop = read_pop2024()
    key = join(share.index, x, pop)
    share = share.rename(index=key)
    info = info.rename(index=key)
    p24 = pop.set_index("pcode")["pop"]
    name = pop.set_index("pcode")["name"]
    final, national = compose(share, flat, p24)

    counts = round_within_rows(final.mul(p24.reindex(final.index), axis=0))
    if not (counts.sum(axis=1) == p24.reindex(counts.index)).all():
        raise SystemExit("rounded rows do not keep the province populations")

    rows = []
    for pc in sorted(counts.index):
        i = info.loc[pc]
        base = (f"INS Enquête 1-2-3 2005 and 2012 (USCB tabulation), religion of the household head, "
                f"n={int(i['heads']):,} heads in {int(i['districts'])} of {int(i['of'])} districts; "
                f"district shares weighted by COD-PS 2019 district population")
        for k in KEYS:
            how = ("national share, this answer having failed the split-half" if k in flat
                   else "province share")
            rows.append({"geo_id": pc, "geo_level": "province", "geo_name": name[pc],
                         "source_category": LABEL[k], "count": int(counts.loc[pc, k]),
                         "basis": BASIS, "year": YEAR, "source_id": SOURCE_ID,
                         "note": f"{base}; {how}, applied to COD-PS 2024"})

    drawn = counts.drop(columns=[NONRESPONSE])
    tot = int(drawn.to_numpy().sum())
    print(f"\n  drawn: {tot:,} of COD-PS 2024's {COD_PS_2024:,}; `Manquant` "
          f"{int(counts[NONRESPONSE].sum()):,} ({100 * counts[NONRESPONSE].sum() / COD_PS_2024:.3f}%)")
    print("  national shares as drawn, beside EDS-RDC III 2023-24 (women / men 15-49, Tableau 3.1):")
    for k in TESTED:
        print(f"    {LABEL[k]:<16}{100 * drawn[k].sum() / tot:6.2f}%")
    for lab, (f, m) in EDS_2023.items():
        print(f"      EDS 2023-24  {lab:<34}{f:5.1f} / {m:.1f}")

    pct = 100 * final[TESTED].div(1 - final[NONRESPONSE], axis=0)
    pct.index = [name[c] for c in pct.index]
    pct.columns = [LABEL[k] for k in TESTED]
    pct.insert(0, "heads", [int(info.loc[c, "heads"]) for c in final.index])
    pct.insert(1, "pop24_M", [p24[c] / 1e6 for c in final.index])
    with pd.option_context("display.width", 250, "display.max_columns", 20,
                           "display.float_format", "{:.1f}".format):
        print("\n  per province, per cent of those drawn:")
        print(pct.sort_values("pop24_M", ascending=False).to_string())

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".part", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(OUT + ".part", OUT)
    print(f"\nwrote {OUT} ({len(rows):,} rows)")


if __name__ == "__main__":
    main()
