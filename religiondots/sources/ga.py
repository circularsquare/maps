"""Gabon: Gabonese citizens by province from Afrobarometer rounds 6-9, on the 2026 census count.

Reads data/raw/afrobarometer/*.sav, data/geo/ga/ga_lookup.csv (sources/ga_geo.py) and the DHS
2019-21 final report (data/raw/ga/FR371.pdf); writes data/normalized/ga.csv. `sources/ga.md` is the
record; the construction is `sources/cm.py`'s.

## WHY A SURVEY

Gabon's censuses do not publish religion. RGPL 2013's *Résultats globaux* has a section headed
for religion that prints language only; the per-province *Principaux tableaux statistiques* are in
print at Stanford, contents unknown; RGPL 2026 has published nine province totals and a national
nationality split, nothing else (sources.md §scout-2026-09-14-taiwan-belarus-gabon, and §ga-
2026-10-03). The DHS 2019-21 asks religion and prints it nationally only; its microdata need a DHS
registration, which is Anita's (asks 047, 048). The Afrobarometer asked Gabon in rounds 6-9
(2015-2021), about 1,200 adults a round, with all nine provinces in every round.

## WHO IS DRAWN

Afrobarometer interviews Gabonese citizens of 18 and over. So the dots are Gabon's 2,318,365
citizens (RGPL 2026), split by province in `sources/ga_geo.py`; the 1,200,256 foreign residents
(34.1%) are the `gap` (Anita 2026-09-15, Libya ruling). Children are drawn at adults' shares.

## WHAT IS DRAWN

Grouped to five answers, because `Christian only` swings 20.9-45.1% by round (24.2 points, among
the widest in sources.md §11ai) and takes the churches' levels with it: Roman Catholic runs
36.7, 35.5, 21.7, 22.5% by round. Each group takes the split-half (`cab.stability`) and the 1%
floor; carried groups keep each province's own share, the rest go to the residual (`ab.compose`).

The Christians are then divided at ONE NATIONAL RATIO from the DHS 2019-21 report's Tableau 3.1
(women 15-49 and men 15-59, combined at the 2026 census's sex split): Catholic, Protestant, revival
churches (*Église de réveil*) and other Christian. The ratio is of all residents, foreigners
included (14.6% of the women and 19.0% of the men are `Autres nationalités`); nothing splits the
citizens' Christians alone. Its Catholic level (29.85% of all) is the survey's pooled Catholic
level to within a point (`dhs_witness`), which is the one place the two can be compared.
The fallback is the bare `christianity` node (`taxonomy/ga2021.py` REVIEW).

Usage:
    python sources/ga.py --fetch    the DHS report (the Afrobarometer files are shared:
                                    `python sources/afrobarometer.py --fetch`)
    python sources/ga.py            rebuild data/normalized/ga.csv
"""

import os
import re
import sys
import urllib.request

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(ROOT, "tools"))

import pandas as pd

import afrobarometer as ab
import cab
import stability
import tz
from cm import gkey, key

LOOKUP = os.path.join(ROOT, "data", "geo", "ga", "ga_lookup.csv")
COD_AB = os.path.join(ROOT, "data", "raw", "ga", "gab_admin_boundaries.geojson.zip")
DHS_PDF = os.path.join(ROOT, "data", "raw", "ga", "FR371.pdf")
DHS_URL = "https://dhsprogram.com/pubs/pdf/FR371/FR371.pdf"
OUT = os.path.join(ROOT, "data", "normalized", "ga.csv")

COUNTRY = "Gabon"
ROUNDS = [6, 7, 8, 9]
RECENT = [8, 9]
SOURCE_ID = "ga_afrobarometer_2015_2021_rgpl2026"
YEARS = "2015-2021"
N_UNITS = 9
GABONESE_2026 = 2_318_365

CATEGORIES = ["Christian", "Muslim", "Traditional/ethnic religion", "None", "Other"]

# Every Gabonese answer over the four rounds, keyed through `cm.key`.
GROUP = {
    "roman catholic": "Christian", "christian only": "Christian", "evangelical": "Christian",
    "pentecostal": "Christian", "baptist": "Christian", "orthodox": "Christian",
    "church of christ": "Christian", "jehovah's witness": "Christian",
    "independent": "Christian", "presbyterian": "Christian", "protestante": "Christian",
    "alliance chretienne": "Christian", "anglican": "Christian",
    "seventh day adventist": "Christian", "eglise de reveil": "Christian", "lutheran": "Christian",
    "dutch reformed": "Christian", "zionist christian church": "Christian",
    "muslim only": "Muslim", "ismaeli": "Muslim", "shia": "Muslim", "sunni only": "Muslim",
    "traditional/ethnic religion": "Traditional/ethnic religion",
    "none": "None", "atheist": "None",
    "other": "Other", "bahai": "Other", "jewish": "Other",
}

# REGION label (through `cm.gkey`, which undoes R6's LATIN1 mojibake) -> COD-AB p-code.
NORM = {"estuaire": "GA01", "hautogooue": "GA02", "moyenogooue": "GA03", "ngounie": "GA04",
        "nyanga": "GA05", "ogooueivindo": "GA06", "ogoouelolo": "GA07", "ogooueogoouelolo": "GA07",
        "ogooueamaritime": "GA08", "ogoouemaritime": "GA08", "woleuntem": "GA09"}
LOCATION_ROUNDS = [6, 7, 9]           # R8 carries no department (every row `None`)
# The survey's department spellings (2013 departments) that COD-AB v01 spells otherwise or has
# since merged, -> COD-AB department (gkey).
DEPT_ALIAS = {"owendo": "libreville", "komoocean": "komo", "komokango": "komo", "mpassa": "passa",
              "mulundu": "mouloundou", "desplateaux": "plateaux", "douigny": "douigni",
              "tsambamagosti": "tsambamagotsi", "ogoouetlacs": "ogooueetlacs",
              "ogoouelacs": "ogooueetlacs", "djouiriagnili": "djouoriagnili",
              "abangabigne": "abangabigne"}

# What carries its own geography, asserted against the split-half; set from its output.
CARRIES = []                          # the split-half passes nothing at 9 provinces (2026-10-03)
# One province tops a failing category in both halves of every halving, with chi-square < 0.05.
STANDOUTS = {"Christian": "GA09", "None": "GA05"}     # Woleu-Ntem, Nyanga
LEVEL_GAP_MAX = 0.035

# DHS 2019-21 (EDSG-III) final report, Tableau 3.1, % of women 15-49 and of men 15-59, weighted.
# Re-read from the PDF on every run (`check_dhs`).
DHS_WOMEN = {"Catholique": 29.8, "Protestante": 9.9, "Église de réveil": 41.9,
             "Autre religion chrétienne": 4.3, "Musulmane": 8.2, "Traditionnelle/animiste": 0.4,
             "Autre religion": 0.7, "Sans religion/aucune": 4.7}
DHS_MEN = {"Catholique": 29.9, "Protestante": 8.6, "Église de réveil": 28.6,
           "Autre religion chrétienne": 3.5, "Musulmane": 15.2, "Traditionnelle/animiste": 2.2,
           "Autre religion": 1.1, "Sans religion/aucune": 10.9}
DHS_FOREIGN = (14.6, 19.0)            # `Autres nationalités` in the same table, women and men
CHRISTIAN_ROWS = ["Catholique", "Protestante", "Église de réveil", "Autre religion chrétienne"]
# RGPL 2026: 1,718,492 men of 3,518,621 (Gabonactu, 2026-08-27).
MEN_SHARE = 1_718_492 / 3_518_621
CATHOLIC_GAP_MAX = 0.03               # survey pooled Catholic vs DHS Catholic, share of all


def fetch():
    if os.path.exists(DHS_PDF) and os.path.getsize(DHS_PDF) > 1_000_000:
        print(f"  have {os.path.basename(DHS_PDF)}")
        return
    req = urllib.request.Request(DHS_URL, headers={"User-Agent": ab.UA})
    with urllib.request.urlopen(req, timeout=900) as r:
        data = r.read()
    if data[:4] != b"%PDF" or b"%%EOF" not in data[-2048:]:
        raise SystemExit("the DHS report is not a complete PDF ([[reference_pdf_truncated_at_source]])")
    with open(DHS_PDF + ".part", "wb") as fh:
        fh.write(data)
    os.replace(DHS_PDF + ".part", DHS_PDF)
    print(f"  got  {os.path.basename(DHS_PDF)} ({len(data):,} bytes)")


def check_dhs():
    """Find Tableau 3.1's religion block in the PDF and assert every transcribed figure."""
    import fitz

    if not os.path.exists(DHS_PDF):
        raise SystemExit(f"missing {DHS_PDF}; run with --fetch")
    doc = fitz.open(DHS_PDF)
    for i, page in enumerate(doc):
        lines = [s.strip() for s in page.get_text().split("\n")]
        if "Église de réveil" in lines and "Autres nationalités" in lines and "Religion" in lines:
            break
    else:
        raise SystemExit("no page of the DHS report has Tableau 3.1's religion block")
    for row in DHS_WOMEN:
        j = lines.index(row)
        got = (lines[j + 1], lines[j + 4])
        want = (f"{DHS_WOMEN[row]:.1f}".replace(".", ","), f"{DHS_MEN[row]:.1f}".replace(".", ","))
        if got != want:
            raise SystemExit(f"DHS Tableau 3.1 {row}: PDF page {i + 1} has {got}, transcribed {want}")
    j = lines.index("Autres nationalités")
    if (lines[j + 1], lines[j + 4]) != tuple(f"{v:.1f}".replace(".", ",") for v in DHS_FOREIGN):
        raise SystemExit("DHS Tableau 3.1's `Autres nationalités` row differs from the transcription")
    for d in (DHS_WOMEN, DHS_MEN):
        if abs(sum(d.values()) - 100) > 0.15:
            raise SystemExit(f"a DHS religion column sums to {sum(d.values()):.1f}")
    mix = {k: (1 - MEN_SHARE) * DHS_WOMEN[k] + MEN_SHARE * DHS_MEN[k] for k in DHS_WOMEN}
    print(f"  DHS 2019-21 Tableau 3.1 (PDF page {i + 1}) re-read and equal; women and men combined "
          f"at {MEN_SHARE:.1%} men: " + ", ".join(f"{k} {v:.2f}" for k, v in mix.items()))
    return mix


def report_card():
    """The boxes the grouping relies on must be on every pooled round's card (value labels)."""
    import pyreadstat

    watch = ["christian only", "roman catholic", "muslim only", "none",
             "traditional/ethnic religion", "other"]
    print("\n  boxes on each round's showcard (value labels, not responses):")
    missing = []
    for rnd, name, _url, relname, _wt in ab.ROUNDS:
        if rnd not in ROUNDS:
            continue
        path = os.path.join(ab.AB_DIR, name)
        try:
            _d, meta = pyreadstat.read_sav(path, metadataonly=True)
        except pyreadstat._readstat_parser.ReadstatError:
            _d, meta = pyreadstat.read_sav(path, metadataonly=True, encoding="LATIN1")
        col = next(c for c in meta.column_names if c.upper() == relname.upper())
        have = {key(v) for v in meta.variable_value_labels.get(col, {}).values()}
        gone = [w for w in watch if w not in have]
        print(f"    R{rnd}: {'every box present' if not gone else 'MISSING ' + ', '.join(gone)}")
        missing += [(rnd, w) for w in gone]
    if missing:
        raise SystemExit(f"boxes missing from a pooled round's card: {missing}")


def check_locations(df):
    """Rounds 6, 7 and 9 carry the department (`LOCATION.LEVEL.1`); it must agree with REGION."""
    import io
    import zipfile

    import geopandas as gpd

    with zipfile.ZipFile(COD_AB) as z:
        d = gpd.read_file(io.BytesIO(z.read("gab_admin2.geojson")))
    if len(d) != 48:
        raise SystemExit(f"COD-AB has {len(d)} departments, expected 48")
    dept_unit = dict(zip(d["adm2_name"].map(gkey), d["adm1_pcode"]))
    dept_unit["libreville"] = "GA01"
    print("\n  unit from REGION against the department column (LOCATION.LEVEL.1):")
    for rnd in LOCATION_ROUNDS:
        sub = df[df["round"] == rnd]
        k = sub["LOCATION.LEVEL.1"].map(gkey).map(lambda s: DEPT_ALIAS.get(s, s))
        du = k.map(dept_unit)
        if du.isna().any():
            raise SystemExit(f"R{rnd}: departments with no COD-AB match: "
                             f"{sorted(sub.loc[du.isna(), 'LOCATION.LEVEL.1'].astype(str).unique())}")
        bad = du != sub["geo_id"]
        print(f"    R{rnd}: {len(sub):,} respondents, {int(bad.sum())} disagree")
        if bad.any():
            pairs = sub[bad].assign(du=du[bad]).groupby(["geo_id", "LOCATION.LEVEL.1", "du"]).size()
            raise SystemExit(f"R{rnd}: department and REGION disagree:\n{pairs.to_string()}")


def levels_by_round(df):
    tot = df.groupby("round")["w"].sum()
    t = df.groupby(["k", "round"])["w"].sum().unstack(fill_value=0.0).div(tot, axis=1)
    print("\n  weighted share of all respondents by round (%), and the range:")
    for k in ["roman catholic", "christian only", "pentecostal", "evangelical", "baptist", "none",
              "muslim only", "traditional/ethnic religion", "other"]:
        row = t.loc[k] if k in t.index else pd.Series(0.0, index=tot.index)
        print(f"    {k:<30}" + "".join(f"{100 * row.get(r, 0):7.1f}" for r in ROUNDS)
              + f"   range {100 * (row.max() - row.min()):5.1f}")
    return t


def swap_table(df, nm):
    """Madagascar's trap: traditional and none, early rounds against late, per unit."""
    print("\n  traditional / none by unit, weighted % (playbooks/afrobarometer.md swap check):")
    for lab, rr in (("R6-R7", [6, 7]), ("R8-R9", [8, 9])):
        s = df[df["round"].isin(rr)]
        t = s.groupby(["geo_id", "category"])["w"].sum().unstack(fill_value=0.0)
        t = 100 * t.div(t.sum(axis=1), axis=0)
        print(f"    {lab:<6}" + "  ".join(
            f"{nm[u][:8]:>8} {t.loc[u].get('Traditional/ethnic religion', 0):4.1f}/{t.loc[u].get('None', 0):4.1f}"
            for u in t.index))


def dhs_witness(df, mix, drawn):
    """The survey (citizens 18+) beside the DHS (all residents 15-49/59). Asserts the one level the
    two share: Catholics as a share of everyone."""
    cath = float(df.loc[df["k"] == "roman catholic", "w"].sum() / df["w"].sum())
    print("\n  DHS 2019-21 (all residents, combined) beside the survey (citizens, pooled) and as drawn:")
    fam = {"Christian": sum(mix[k] for k in CHRISTIAN_ROWS), "Muslim": mix["Musulmane"],
           "Traditional/ethnic religion": mix["Traditionnelle/animiste"],
           "None": mix["Sans religion/aucune"], "Other": mix["Autre religion"]}
    for c in CATEGORIES:
        print(f"    {c:<30}DHS {fam[c]:6.2f}%   drawn {100 * drawn[c]:6.2f}%")
    print(f"    Catholic, share of everyone:  DHS {mix['Catholique']:.2f}%, survey pooled {100 * cath:.2f}%")
    if abs(cath - mix["Catholique"] / 100) > CATHOLIC_GAP_MAX:
        raise SystemExit("the survey's pooled Catholic level is no longer within "
                         f"{100 * CATHOLIC_GAP_MAX:.0f} points of the DHS's; the Christian split's "
                         "one check has gone, decide it again")


def compose(df, units, stand):
    """Nothing carries its own geography here (`CARRIES` is empty), so every province starts from
    one national mix, and a standout (spec §12, Honduras; `tz.standouts`) takes its own share in
    its one province.

    NOT `tz.py::compose`. Its residual rule fixes each standout at the other provinces' pooled
    share everywhere and fills what is left with the tail; Gabon's two standouts are Christian
    (Woleu-Ntem) and None (Nyanga), which are each other's complement, so in Nyanga the two
    fixed shares come to more than 100% and the residual goes negative. Here the base mix is each
    category's share pooled over the provinces where it is not a standout, renormalised; a
    province with a standout keeps that share and scales the rest of the base mix to fill. A
    province with none is drawn at the base mix."""
    by = df.groupby(["geo_id", "category"])["w"].sum().unstack(fill_value=0.0)
    by = by.reindex(index=units, columns=CATEGORIES, fill_value=0.0)
    own = by.div(by.sum(axis=1), axis=0)
    base = pd.Series({c: by.loc[[u for u in units if u != stand.get(c)], c].sum()
                      / by.loc[[u for u in units if u != stand.get(c)]].sum().sum()
                      for c in CATEGORIES})
    base = base / base.sum()
    frame = pd.DataFrame([base.to_numpy()] * len(units), index=units, columns=CATEGORIES)
    for u in units:
        mine = [c for c, su in stand.items() if su == u]
        if not mine:
            continue
        rest = [c for c in CATEGORIES if c not in mine]
        left = 1.0 - float(own.loc[u, mine].sum())
        frame.loc[u, mine] = own.loc[u, mine].to_numpy()
        frame.loc[u, rest] = (base[rest] / base[rest].sum() * left).to_numpy()
    print("\n  the base mix (each category pooled where it is not a standout): "
          + ", ".join(f"{c} {100 * v:.2f}%" for c, v in base.items()))
    for c, u in stand.items():
        print(f"    standout: {c} in {u} at {100 * own.loc[u, c]:.2f}% against {100 * base[c]:.2f}%")
    if (frame.sum(axis=1) - 1.0).abs().max() > 1e-9 or (frame < 0).any().any():
        raise SystemExit("a unit's shares do not sum to 1, or one is negative")
    return frame


def main():
    if "--fetch" in sys.argv:
        fetch()
    print("=== Afrobarometer, Gabon ===")
    mix = check_dhs()
    raw = ab.load(COUNTRY, expect_rounds=ROUNDS, regroup=True, extra=["LOCATION.LEVEL.1", "URBRUR"])
    raw["k"] = raw["category"].map(key)
    ct = pd.crosstab(raw["k"], raw["round"])
    ct["all"] = ct.sum(axis=1)
    print(f"\n  pooled: {len(raw):,} respondents; every answer as it arrives, keyed:")
    print(ct.sort_values("all", ascending=False).to_string())
    report_card()
    unmapped = sorted(set(raw["k"]) - set(GROUP))
    if unmapped:
        raise SystemExit(f"answers with no group: {unmapped}; add them to GROUP deliberately")
    df = raw.copy()
    df["raw_category"] = raw["category"]
    df["category"] = raw["k"].map(GROUP)
    ab.assert_one_wording(df, COUNTRY)
    levels_by_round(df)

    # ---- units ----
    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})
    if len(lut) != N_UNITS:
        raise SystemExit(f"{LOOKUP} has {len(lut)} units, expected {N_UNITS}; re-run ga_geo.py")
    nm = dict(zip(lut["geo_id"], lut["name"]))
    pop = lut.set_index("geo_id")["pop"].astype(float)
    if int(pop.sum()) != GABONESE_2026:
        raise SystemExit(f"ga_lookup.csv sums to {int(pop.sum()):,}, not {GABONESE_2026:,}")
    units = sorted(lut["geo_id"])
    df["geo_id"] = df["geo_raw"].map(gkey).map(NORM)
    if df["geo_id"].isna().any():
        raise SystemExit(f"REGION labels with no unit: "
                         f"{sorted(df.loc[df['geo_id'].isna(), 'geo_raw'].astype(str).unique())}")
    per_round = df.groupby("round")["geo_id"].nunique()
    if (per_round != N_UNITS).any():
        raise SystemExit(f"a pooled round does not sample all 9 provinces: {per_round.to_dict()}")
    if (df.groupby(["round", "geo_raw"])["geo_id"].nunique() > 1).any():
        raise SystemExit("one REGION label names two units in a round")
    codes = df.groupby("geo_id")["geo_code"].unique()
    if any(len(v) != 1 for v in codes):
        raise SystemExit(f"a province has more than one REGION code across rounds: {codes.to_dict()}")
    check_locations(df)
    for rnd in ROUNDS:
        print(f"\n  R{rnd}:", end="")
        ab.held_out(df[df["round"] == rnd], pop, f"{COUNTRY} R{rnd}", pop_source="Gabonese 2026")
    ab.held_out(df, pop, COUNTRY, pop_source="Gabonese 2026 (fitted)")

    nat = ab.national(df).reindex(CATEGORIES).fillna(0.0)
    print(f"\n  the survey's national shares, pooled over R6-R9 (n={len(df):,}):")
    for c in CATEGORIES:
        print(f"    {c:<30}{nat[c]:8.3%}")
    rn = df.groupby(["round", "category"])["w"].sum().unstack(fill_value=0)
    print("\n  by round (survey weighting, %):")
    print((100 * rn.div(rn.sum(axis=1), axis=0)).round(1).reindex(columns=CATEGORIES).to_string())
    swap_table(df, nm)

    # ---- quota, then the split-half ----
    dfw = df.rename(columns={"round": "wave", "category": "code"})[["wave", "geo_id", "code", "w"]]
    cab.assert_not_quota(dfw, COUNTRY, ROUNDS, unit_col="geo_id", cat_col="code")
    passed, table = cab.stability(dfw, CATEGORIES, units, f"{N_UNITS} provinces")
    carries = [c for c in passed if nat[c] >= ab.ELIGIBLE_FLOOR]
    if sorted(carries) != sorted(CARRIES):
        raise SystemExit(f"the split-half now selects {sorted(carries)}, not {sorted(CARRIES)}. "
                         "Read the table above, then edit CARRIES and the docstring deliberately.")
    failing = [c for c in CATEGORIES if c not in carries and nat[c] >= ab.ELIGIBLE_FLOOR]
    stand = tz.standouts(dfw, CATEGORIES, units, failing, table)
    if stand != STANDOUTS:
        raise SystemExit(f"standouts are now {stand}, against STANDOUTS={STANDOUTS}")

    # ---- compose ----
    frame = compose(df, units, stand)
    fam = frame.mul(pop.reindex(units), axis=0)

    # ---- level: pooled against the recent rounds, both recomposed on the Gabonese count ----
    drawn = fam.sum(axis=0) / fam.sum().sum()
    rec = df[df["round"].isin(RECENT)]
    rb = rec.groupby(["geo_id", "category"])["w"].sum().unstack(fill_value=0.0)
    rb = rb.reindex(index=units, columns=CATEGORIES, fill_value=0.0)
    recent = rb.div(rb.sum(axis=1), axis=0).mul(pop.reindex(units), axis=0).sum() / pop.sum()
    print("\n  national level: survey pool, as drawn, and rounds 8-9 alone recomposed the same way:")
    print(f"    {'category':<30}{'survey':>9}{'drawn':>9}{'R8-R9':>9}{'drawn-R8R9':>12}")
    for c in CATEGORIES:
        print(f"    {c:<30}{100 * nat[c]:8.2f}%{100 * drawn[c]:8.2f}%{100 * recent[c]:8.2f}%"
              f"{100 * (drawn[c] - recent[c]):+11.2f}")
    stale = [c for c in CATEGORIES if abs(drawn[c] - recent[c]) > LEVEL_GAP_MAX]
    if stale:
        raise SystemExit(f"the pooled level differs from rounds 8-9 by more than "
                         f"{100 * LEVEL_GAP_MAX:.1f} points for {stale}; spec §12 (Norway)")
    dhs_witness(df, mix, drawn)

    # ---- the Christians, divided at the DHS's national ratio ----
    chr_tot = sum(mix[k] for k in CHRISTIAN_ROWS)
    split = {k: mix[k] / chr_tot for k in CHRISTIAN_ROWS}
    print("\n  Christians divided at the DHS ratio: " + ", ".join(f"{k} {100 * v:.2f}%" for k, v in split.items()))
    full = fam.drop(columns=["Christian"]).copy()
    for k in CHRISTIAN_ROWS:
        full[k] = fam["Christian"] * split[k]
    cols = CHRISTIAN_ROWS + [c for c in CATEGORIES if c != "Christian"]
    counts = ab.round_within_rows(full[cols])
    if not (counts.sum(axis=1) == pop.reindex(units).round().astype("int64")).all():
        raise SystemExit("a province's drawn total is not its Gabonese count")

    # ---- write ----
    n_by = df.groupby("geo_id").size()
    basis_note = {c: (f"the survey's own share in {nm[stand[c]]}, the other provinces' pooled "
                      "share elsewhere (a standout)" if c in stand else
                      "one national mix, scaled where a standout takes its own share")
                  for c in CATEGORIES}
    for k in CHRISTIAN_ROWS:
        basis_note[k] = (basis_note["Christian"] + "; Christians divided at the DHS 2019-21 national "
                         "ratio")
    out = counts.stack().rename("count").reset_index()
    out.columns = ["geo_id", "source_category", "count"]
    out["geo_level"] = "province"
    out["geo_name"] = out["geo_id"].map(nm)
    out["basis"] = "self_id"
    out["year"] = YEARS
    out["source_id"] = SOURCE_ID
    out["note"] = out.apply(
        lambda r: ("Gabonese citizens per province (RGPL 2026, split fitted) composed with the "
                   f"province's mix from Afrobarometer rounds 6-9 pooled (n={int(n_by[r.geo_id])} "
                   f"here); {basis_note[r.source_category]}"), axis=1)
    total = int(out["count"].sum())
    if total != GABONESE_2026:
        raise SystemExit(f"drawn {total:,} against {GABONESE_2026:,} Gabonese")
    keep = ["geo_id", "geo_level", "geo_name", "source_category", "count", "basis", "year",
            "source_id", "note"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[keep].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {total:,} people, {out['source_category'].nunique()} "
          f"categories, {out['geo_id'].nunique()} units)")

    share = counts.div(counts.sum(axis=1), axis=0)
    print("\n  as drawn by province (%), with respondents:")
    for u in units:
        print(f"    {nm[u]:<16}" + ", ".join(f"{c} {100 * share.loc[u, c]:.1f}" for c in cols)
              + f"  n={int(n_by[u])}  pop {int(pop[u]):,}")
    print("  national: " + ", ".join(f"{c} {100 * counts[c].sum() / total:.2f}" for c in cols))


if __name__ == "__main__":
    main()
