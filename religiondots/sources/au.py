"""Australia — ABS Census of Population and Housing, 2021.

Reads  data/raw/au/   (three files, all free, no login — see sources/au.md)
Writes data/normalized/au.csv

Three geographies come out of one census, all `self_id`, all 2021:

  sa2     2,472 Statistical Areas Level 2 (ASGS Edition 3, 2021) x the 34
          columns of DataPack table G14 "Religious Affiliation by Sex",
          persons only. This is the map layer.
  nation  the ~148 ASCRG "religious group" (4-digit) categories — the finest
          classification level the ABS publishes outside TableBuilder.
  state   8 states/territories x 9 ASCRG "broad group" categories.

Nothing here is mapped to the religiondots taxonomy (spec.md 2.4).
`source_category` is the ABS's own published label, verbatim, taken from the
official ASCRG 2021 classification file rather than from the DataPack's
abbreviated column names. `note` carries the ASCRG code, the level in the ABS
hierarchy and the parent category, so a consumer can pick exactly one
non-overlapping level and never double count. See au.md.
"""

import csv
import io
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "au"
OUT = ROOT / "data" / "normalized" / "au.csv"

GCP_ZIP = RAW / "2021_GCP_SA2_for_AUS_short-header.zip"
ARTICLE = RAW / "Census_article_Religious_affiliation_in_Australia.xlsx"
CLASSIF = RAW / "2021_Religious_affiliation_classification.xlsx"

SOURCE_ID = "au_census_2021"
YEAR = 2021
BASIS = "self_id"  # census self-reported affiliation — every row, no exceptions

G14_INSIDE = "2021 Census GCP Statistical Area 2 for AUS/2021Census_G14_AUST_SA2.csv"
GEOG_INSIDE = "Metadata/2021Census_geog_desc_1st_2nd_3rd_release.xlsx"

STATES = [
    ("1", "New South Wales"),
    ("2", "Victoria"),
    ("3", "Queensland"),
    ("4", "South Australia"),
    ("5", "Western Australia"),
    ("6", "Tasmania"),
    ("7", "Northern Territory"),
    ("8", "Australian Capital Territory"),
]

# ---------------------------------------------------------------------------
# G14's 34 persons columns, in file order, keyed to the ASCRG.
#
#   N<code>  an ASCRG narrow group   -> label and parent read from the
#   B<code>  an ASCRG broad group       classification file, never hardcoded
#   X:<label>  not an ASCRG category at all
#
# `level` is what stops a consumer double counting. The 30 `leaf` columns
# partition the population exactly once; the 3 `group_total` columns are sums
# of leaves above them; `grand_total` is the whole population.
#
# Buddhism / Hinduism / Islam / Judaism are broad groups containing a single
# narrow group containing a single religious group, so one column serves all
# three ABS levels. They are tagged `leaf` and coded at the narrow level.
# ---------------------------------------------------------------------------
G14_COLUMNS = [
    ("Buddhism_P", "N101", "leaf"),
    ("Christianity_Anglican_P", "N201", "leaf"),
    ("Christianity_Asyrin_Apstlic_P", "N222", "leaf"),
    ("Christianity_Baptist_P", "N203", "leaf"),
    ("Christianity_Brethren_P", "N205", "leaf"),
    ("Christianity_Catholic_P", "N207", "leaf"),
    ("Christianity_Church_Christ_P", "N211", "leaf"),
    ("Christianity_Eastrn_Orthdox_P", "N223", "leaf"),
    ("Christinty_Jehvahs_Witnses_P", "N213", "leaf"),
    ("Christianity_Lattr_day_Snts_P", "N215", "leaf"),
    ("Christianity_Lutheran_P", "N217", "leaf"),
    ("Christianity_Orintal_Orthdx_P", "N221", "leaf"),
    ("Christianity_Othr_Protestnt_P", "N28", "leaf"),
    ("Christianity_Pentecostal_P", "N24", "leaf"),
    ("Christinty_Prsbytrin_Refrmd_P", "N225", "leaf"),
    ("Christianity_Salvation_Army_P", "N227", "leaf"),
    ("Christnty_Sevnth_dy_Advntst_P", "N231", "leaf"),
    ("Christianity_Uniting_Church_P", "N233", "leaf"),
    ("Christianity_Christian_nfd_P", "N200", "leaf"),
    ("Christianity_Othr_Christian_P", "N29", "leaf"),
    ("Christianity_Tot_P", "B2", "group_total"),
    ("Hinduism_P", "N301", "leaf"),
    ("Islam_P", "N401", "leaf"),
    ("Judaism_P", "N501", "leaf"),
    ("Othr_Rel_Aust_Abor_Trad_Rel_P", "N601", "leaf"),
    ("Othr_Rel_Sikhism_P", "N615", "leaf"),
    # Not an ASCRG category: a G14-only residual over the seven remaining
    # narrow groups of broad group 6 — Baha'i, Chinese Religions, Druse,
    # Japanese Religions, Nature Religions, Spiritualism, Miscellaneous
    # Religions. Those seven are only separable at nation level. See au.md.
    ("Othr_Reln_Other_reln_groups_P", "X:Other Religious Groups", "leaf"),
    ("Other_Religions_Tot_P", "B6", "group_total"),
    ("SB_OSB_NRA_NR_P", "N71", "leaf"),
    ("SB_OSB_NRA_SB_P", "N72", "leaf"),
    ("SB_OSB_NRA_OSB_P", "N73", "leaf"),
    ("SB_OSB_NRA_Tot_P", "B7", "group_total"),
    # G14 folds the two ASCRG supplementary codes into one column:
    # 000 Inadequately described + &&& Not stated.
    ("Religious_affiliation_ns_P", "X:Religious affiliation not stated", "leaf"),
    ("Tot_P", "X:Total", "grand_total"),
]

# Where an X: column sits, for the note field.
X_META = {
    "Other Religious Groups": ("603+605+607+611+613+617+69", "Other Religions"),
    "Religious affiliation not stated": ("000+&&&", "Total"),
    "Total": ("", ""),
}

# Table 4's four aggregate rows, which are not ASCRG religious groups.
TABLE4_TOTALS = {
    "Total Christian": ("group_total", "2", "Total"),
    "Total Other Religions": ("group_total", "6", "Total"),
    "Total Secular Beliefs and Other Spiritual Beliefs and No Religious Affiliation":
        ("group_total", "7", "Total"),
    "Total": ("grand_total", "", ""),
}

# Table 1's row labels -> ASCRG broad code. "Total" is handled separately.
TABLE1_BROAD = {
    "Buddhism": "1",
    "Christianity": "2",
    "Hinduism": "3",
    "Islam": "4",
    "Judaism": "5",
    "Other Religions": "6",
    "Secular Beliefs and Other Spiritual Beliefs and No Religious Affiliation": "7",
    "Inadequately described": "0",
    "Not stated": "&",
}


def note(**kv):
    return "; ".join(f"{k}={v}" for k, v in kv.items() if v not in (None, ""))


# ------------------------------------------------------------- classification

def read_classification():
    """ASCRG 2021: broad (1 char) / narrow (2-3 char) / religious group (4 char).

    Returns (broad, narrow, leaf):
      broad  {code: label}
      narrow {code: (label, broad_label)}
      leaf   {label: (code4, narrow_label, narrow_code, broad_label, broad_code)}
    """
    df = pd.read_excel(CLASSIF, sheet_name="RELP", header=None, dtype=str)
    broad, narrow, leaf = {}, {}, {}
    b_code = b_label = n_code = n_label = None
    for _, row in df.iterrows():
        c0, c1, c2, c3 = (str(row[i]).strip() if pd.notna(row[i]) else None
                          for i in range(4))
        if c0 and c1 and not c2 and not c3 and c0 != "Supplementary Codes":
            b_code, b_label = c0, c1
            broad[b_code] = b_label
            n_code = n_label = None
        elif c1 and c2 and not c3:
            n_code, n_label = c1, c2
            narrow[n_code] = (n_label, b_label)
        elif c2 and c3:
            # the sheet prefixes supplementary 4-char codes with a dagger
            code4 = c2.lstrip("†‡*� ")
            # broad group 7 carries a religious group with no narrow group
            leaf[c3] = (code4, n_label or c3, n_code or "", b_label, b_code)
    return broad, narrow, leaf


def resolve_g14(broad, narrow):
    """G14 column -> (label, level, ascrg code, parent label)."""
    out = []
    for col, key, level in G14_COLUMNS:
        if key.startswith("N"):
            code = key[1:]
            assert code in narrow, f"unknown ASCRG narrow code {code}"
            label, parent = narrow[code]
        elif key.startswith("B"):
            code = key[1:]
            assert code in broad, f"unknown ASCRG broad code {code}"
            label, parent = broad[code], "Total"
        else:
            label = key[2:]
            code, parent = X_META[label]
        out.append((col, label, level, code, parent))
    return out


# ------------------------------------------------------------- SA2, table G14

def read_sa2(spec):
    with zipfile.ZipFile(GCP_ZIP) as z:
        with z.open(G14_INSIDE) as f:
            g14 = pd.read_csv(f, dtype={"SA2_CODE_2021": str})
        with z.open(GEOG_INSIDE) as f:
            geog = pd.read_excel(io.BytesIO(f.read()),
                                 sheet_name="2021_ASGS_MAIN_Structures", dtype=str)
    sa2meta = geog[geog.ASGS_Structure == "SA2"]
    names = dict(zip(sa2meta.Census_Code_2021, sa2meta.Census_Name_2021))

    # both directions, always (spec.md 8.1)
    data_ids, meta_ids = set(g14.SA2_CODE_2021), set(names)
    print(f"[sa2] G14 rows {len(g14):,}   ASGS Ed3 SA2s {len(meta_ids):,}")
    print(f"[sa2] in data not in geography: {sorted(data_ids - meta_ids) or 'none'}")
    print(f"[sa2] in geography not in data: {sorted(meta_ids - data_ids) or 'none'}")

    missing = [c for c, *_ in spec if c not in g14.columns]
    assert not missing, f"G14 columns missing: {missing}"

    # 18 special-purpose SA2s, two per state/territory: "Migratory - Offshore -
    # Shipping" (x97979799) and "No usual address" (x99999499). Zero area, no
    # usable boundary, 52,920 people between them. They are real census units
    # and must be dropped deliberately rather than lost in a geometry join
    # (spec.md 8.1).
    pseudo = {c for c in data_ids if c[1:] in ("97979799", "99999499")}
    npseudo = sum(int(v) for c, v in zip(g14.SA2_CODE_2021, g14.Tot_P) if c in pseudo)
    print(f"[sa2] {len(pseudo)} special-purpose SA2s (Migratory-Offshore-Shipping, "
          f"No usual address) holding {npseudo:,} people "
          f"= {100 * npseudo / g14.Tot_P.sum():.3f}% -- flagged, no boundary")
    zero = set(g14.loc[g14.Tot_P == 0, "SA2_CODE_2021"])
    print(f"[sa2] {len(zero)} SA2s have zero population, {len(zero - pseudo)} of them "
          f"real and uninhabited (national parks, reservoirs, industrial estates)")

    rows = []
    for r in g14.itertuples(index=False):
        rec = r._asdict()
        code = rec["SA2_CODE_2021"]
        for col, label, level, ascrg, parent in spec:
            rows.append(dict(
                geo_id=code, geo_level="sa2", geo_name=names.get(code, ""),
                source_category=label, count=int(rec[col]), basis=BASIS, year=YEAR,
                source_id=SOURCE_ID,
                note=note(table="G14", level=level, ascrg=ascrg, parent=parent,
                          col=col,
                          flag="pseudo_sa2_no_boundary" if code in pseudo else ""),
            ))
    return rows, g14


# ------------------------------------------------------------ nation, Table 4

def sheet(name):
    return pd.read_excel(ARTICLE, sheet_name=name, header=None)


def read_nation(leaf):
    """Table 4: ASCRG religious groups (4-digit), Australia, column 2 = 2021."""
    df = sheet("Table 4")
    rows, unmatched = [], []
    for _, r in df.iterrows():
        label = str(r[0]).strip() if pd.notna(r[0]) else ""
        val = r[2]
        if not label or pd.isna(val) or not isinstance(val, (int, float)):
            continue
        if label in TABLE4_TOTALS:
            level, ascrg, parent = TABLE4_TOTALS[label]
        elif label in leaf:
            ascrg, narrow_label, _nc, _bl, _bc = leaf[label]
            level = "leaf"
            # the two supplementary codes are their own narrow group
            parent = "Total" if narrow_label == label else narrow_label
        elif label in ("Inadequately described", "Not stated"):
            level = "leaf"
            ascrg = {"Inadequately described": "0000", "Not stated": "&&&&"}[label]
            parent = "Total"
        else:
            unmatched.append(label)
            continue
        rows.append(dict(
            geo_id="AUS", geo_level="nation", geo_name="Australia",
            source_category=label, count=int(val), basis=BASIS, year=YEAR,
            source_id=SOURCE_ID,
            note=note(table="Table 4", level=level, ascrg=ascrg, parent=parent),
        ))
    assert not unmatched, f"Table 4 labels not in the ASCRG classification: {unmatched}"
    return rows, df


# ------------------------------------------------------------- state, Table 1

def read_state():
    """Table 1: ASCRG broad groups by state. 2016 block, then the 2021 block."""
    df = sheet("Table 1")
    marks = df.index[df[1].astype(str).str.strip() == "2021"]
    assert len(marks) == 1, "could not find the single 2021 block marker in Table 1"

    rows, seen, published = [], set(), {}
    for _, r in df.loc[marks[0]:].iterrows():
        label = str(r[0]).strip() if pd.notna(r[0]) else ""
        if label in seen:
            continue
        if label in TABLE1_BROAD:
            level, ascrg, parent = "broad_group", TABLE1_BROAD[label], "Total"
        elif label == "Total":
            level, ascrg, parent = "grand_total", "", ""
        else:
            continue
        seen.add(label)
        published[label] = int(r[9])          # the Total(b) column = Australia
        for i, (code, name) in enumerate(STATES, start=1):
            assert pd.notna(r[i]), f"Table 1 {label} / {name} is blank"
            rows.append(dict(
                geo_id=code, geo_level="state", geo_name=name,
                source_category=label, count=int(r[i]), basis=BASIS, year=YEAR,
                source_id=SOURCE_ID,
                note=note(table="Table 1", level=level, ascrg=ascrg, parent=parent),
            ))
    want = set(TABLE1_BROAD) | {"Total"}
    assert seen == want, f"Table 1 rows missing: {want - seen}"
    return rows, published


def read_published_narrow():
    """Table 3: ASCRG narrow groups, Australia, 2021. The reconciliation target."""
    df = sheet("Table 3")
    pub = {}
    for _, r in df.iterrows():
        label = str(r[1]).strip() if pd.notna(r[1]) else ""
        if label and label != "Total" and pd.notna(r[3]) and isinstance(r[3], (int, float)):
            pub.setdefault(label, int(r[3]))
    return pub


# ----------------------------------------------------------- reconciliation

def reconcile(spec, g14, nation_rows, pub_narrow, pub_broad):
    print("\n" + "=" * 86)
    print("RECONCILIATION - SA2 sums against the ABS published national figures")
    print("  SA2 sum: 2,472 SA2 cells of DataPack G14, added up")
    print("  published: ABS article 'Religious affiliation in Australia', Tables 1 and 3")
    print("=" * 86)

    # G14's two composite columns have no single published counterpart
    composite = {
        "Other Religious Groups": ["Baha'i", "Chinese Religions", "Druse",
                                   "Japanese Religions", "Nature Religions",
                                   "Spiritualism", "Miscellaneous Religions"],
        "Religious affiliation not stated": ["Inadequately described", "Not stated"],
    }

    print(f"{'ABS category':<66}{'SA2 sum':>12}{'published':>12}{'diff':>9}{'%':>8}")
    worst_label, worst_pct = "", 0.0
    for col, label, level, _a, _p in spec:
        s = int(g14[col].sum())
        if label in composite:
            p = sum(pub_narrow[k] for k in composite[label])
        elif label in pub_broad:                 # broad totals and "Total"
            p = pub_broad[label]
        else:
            p = pub_narrow[label]
        d, pct = s - p, 100.0 * (s - p) / p
        if abs(pct) > abs(worst_pct):
            worst_label, worst_pct = label, pct
        print(f"{label:<66}{s:>12,}{p:>12,}{d:>9,}{pct:>8.2f}")

    print(f"\nlargest discrepancy: {worst_label} at {worst_pct:+.2f}%. "
          "The ABS applies a small random adjustment to every cell of every")
    print("table independently, so no two ABS tables add up to each other exactly. "
          "The bias is downward for rare")
    print("categories because true counts of 1-2 are perturbed to 0 in many of "
          "the 2,472 SA2 cells.")

    leaves = [c for c, _l, lvl, _a, _p in spec if lvl == "leaf"]
    tot, lsum = int(g14["Tot_P"].sum()), int(g14[leaves].sum().sum())
    print(f"\nNESTING. The 30 leaf columns sum to {lsum:,} against Tot_P {tot:,} "
          f"(diff {lsum - tot:+,}).")
    print("Leaf + group_total + grand_total in the same file is the double-counting "
          "trap: use one level only.")
    for tot_col, parent in [("Christianity_Tot_P", "Christianity"),
                            ("Other_Religions_Tot_P", "Other Religions"),
                            ("SB_OSB_NRA_Tot_P",
                             "Secular Beliefs and Other Spiritual Beliefs "
                             "and No Religious Affiliation")]:
        parts = [c for c, _l, lvl, _a, par in spec if lvl == "leaf" and par == parent]
        a, b = int(g14[parts].sum().sum()), int(g14[tot_col].sum())
        print(f"  {tot_col:<24} {len(parts):>2} children {a:>12,}   parent {b:>12,}"
              f"   diff {a - b:+,}")

    ns = int(g14["Religious_affiliation_ns_P"].sum())
    print(f"\nNON-RESPONSE. Religion is the ONLY optional question on the Australian "
          f"census; s14(3) of the")
    print("Census and Statistics Act 1905 forbids compelling an answer. "
          "ABS publishes the 2021 non-response")
    print("rate as 6.9%, down from 9.1% in 2016 - that is 'Not stated' alone.")
    print(f"  published: Not stated {pub_narrow['Not stated']:,} "
          f"({100 * pub_narrow['Not stated'] / pub_broad['Total']:.2f}%)"
          f"  +  Inadequately described {pub_narrow['Inadequately described']:,} "
          f"({100 * pub_narrow['Inadequately described'] / pub_broad['Total']:.2f}%)")
    print(f"  G14 folds the two together, so at SA2 the unusable share is "
          f"{ns:,} of {tot:,} = {100 * ns / tot:.2f}%")

    nleaf = [r for r in nation_rows if "level=leaf" in r["note"]]
    nsum = sum(r["count"] for r in nleaf)
    print(f"\nNATION DETAIL. {len(nleaf)} ASCRG religious-group (4-digit) categories "
          f"sum to {nsum:,}")
    print(f"  against the published Total {pub_broad['Total']:,} "
          f"(diff {nsum - pub_broad['Total']:+,}).")


# ----------------------------------------------------------------------- main

def main():
    broad, narrow, leaf = read_classification()
    print(f"[ascrg] {len(broad)} broad groups, {len(narrow)} narrow groups, "
          f"{len(leaf)} religious groups (incl. supplementary codes)")

    spec = resolve_g14(broad, narrow)
    sa2_rows, g14 = read_sa2(spec)
    nation_rows, _t4 = read_nation(leaf)
    state_rows, pub_broad = read_state()
    pub_narrow = read_published_narrow()

    rows = sa2_rows + nation_rows + state_rows

    counts = {}
    for r in rows:
        k = (r["geo_level"], r["geo_id"], r["source_category"])
        counts[k] = counts.get(k, 0) + 1
    dupes = [k for k, v in counts.items() if v > 1]
    assert not dupes, f"duplicate (geo_level, geo_id, source_category): {dupes[:5]}"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count",
            "basis", "year", "source_id", "note"]
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {OUT}   {len(rows):,} rows")
    for lvl in ("sa2", "nation", "state"):
        sub = [r for r in rows if r["geo_level"] == lvl]
        print(f"  {lvl:<8}{len(sub):>8,} rows{len({r['geo_id'] for r in sub}):>7,} units"
              f"{len({r['source_category'] for r in sub}):>5} categories")

    reconcile(spec, g14, nation_rows, pub_narrow, pub_broad)


if __name__ == "__main__":
    main()
