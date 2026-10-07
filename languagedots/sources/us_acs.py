"""United States, ACS 2020-2024 5-year, language spoken at home -> data/normalized/us*.csv.

    python sources/us_acs.py [--fetch]

THREE TABLES OF ONE SURVEY, each the finest the Census Bureau publishes at its grain:

  C16001  tract (85,000)  English only, Spanish and 11 groups ("Other Indo-European languages")
  B16001  PUMA (2,462)    the same people in 42 groups ("Gujarati", "Nepali, Marathi, or other
                          Indic languages"); since 2016 not published for counties or tracts
  PUMS    PUMA            person microdata, LANP: 125 codes (Hmong, Navajo, Pennsylvania German)

All three are the ACS 2020-2024 5-year release (the table-based summary file and the 5-year PUMS
person file), so they are the same sample under the same weights; PUMS is a subsample of it.
The question (asked of everyone aged 5 and over, group quarters included) is "does this person
speak a language other than English at home? What is this language?" One answer per person.

WHAT THIS WRITES

  us.csv        tract x C16001 group, as published (geo_id = tract GEOID, puma = state+PUMA)
  us_split.csv  per PUMA and C16001 group, the share of each LANP code in it:
                share = B16001(PUMA, b) / B16001(PUMA, all b in the group)      [published]
                      x PUMS(PUMA, code) / PUMS(PUMA, all codes in b)           [microdata]
                Where the PUMA's PUMS sample has nobody in b although B16001 has people (a
                small group in a small sample), the state's PUMS mix of b is used, then the
                nation's; `fallback` says which.

countries/us.py multiplies the two: a tract's published group count shared out by its PUMA's
mix. So English, Spanish, Korean, Vietnamese and Arabic are tract counts as published; every
other language is a tract's group count split by a PUMA-level mix (tier `derived`).

THE CROSSWALKS. LANP -> B16001 group and B16001 -> C16001 group are not published as a table.
B -> C is asserted here exactly, at every PUMA (the two tables are tabulated from the same
records, so the sums must agree to the person). LANP -> B is from the 2016 user note's examples
plus the code list's members, and is checked against the published PUMA table: for every B
group, PUMS's weighted national total against B16001's (printed; a wrong assignment of a code
shows as one group short by that code's total and another over by the same).

CHECKS (all asserted unless said):
  1. tracts sum to the nation per C16001 group, exactly
  2. tracts sum to their PUMA per C16001 group, exactly (also proves the tract -> PUMA key)
  3. B16001 groups sum to their C16001 group at every PUMA, exactly
  4. PUMS against B16001, nationally per B group (printed, bar 5% on groups over 100,000)
"""
import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "us"
NORM = HERE / "data" / "normalized"
RD_GEO = HERE.parent / "religiondots" / "data" / "geo"
SF = "https://www2.census.gov/programs-surveys/acs/summary_file/2024/table-based-SF/data/5YRData/"
FILES = {
    "acsdt5y2024-c16001.dat": SF + "acsdt5y2024-c16001.dat",
    "acsdt5y2024-b16001.dat": SF + "acsdt5y2024-b16001.dat",
    "csv_pus_2024_5y.zip": "https://www2.census.gov/programs-surveys/acs/data/pums/2024/5-Year/csv_pus.zip",
    "ACSPUMS2020_2024CodeLists.xlsx": "https://www2.census.gov/programs-surveys/acs/tech_docs/pums/"
                                      "code_lists/ACSPUMS2020_2024CodeLists.xlsx",
}
PUMS_AGG = RAW / "pums_lanp_puma.csv"
EXCLUDE_STATES = {"72"}   # Puerto Rico is its own entry (queue `pr`)

ENGLISH = "Speak only English"

# Tracts in cb_2024 and in the 2020 tract-to-PUMA file with no row in the 2024 C16001 tract file
# (Suffolk County, NY; the county and PUMA rows count their people). Tract -> state+PUMA.
MISSING_TRACTS = {
    "36103122406": "3603313", "36103122501": "3603313",
    **{f"36103{t}": "3603310" for t in ("145601", "145602", "145603", "145604", "145605", "145702",
                                       "146001", "146105", "146106", "146201", "146204", "201200")},
}

# C16001 groups, by the column of their total (step 3: total, very well, less than very well)
C_GROUPS = {
    3: "Spanish",
    6: "French, Haitian, or Cajun",
    9: "German or other West Germanic languages",
    12: "Russian, Polish, or other Slavic languages",
    15: "Other Indo-European languages",
    18: "Korean",
    21: "Chinese (incl. Mandarin, Cantonese)",
    24: "Vietnamese",
    27: "Tagalog (incl. Filipino)",
    30: "Other Asian and Pacific Island languages",
    33: "Arabic",
    36: "Other and unspecified languages",
}

# B16001 groups: column -> (label, C16001 group)
B_GROUPS = {
    3: ("Spanish", "Spanish"),
    6: ("French (incl. Cajun)", "French, Haitian, or Cajun"),
    9: ("Haitian", "French, Haitian, or Cajun"),
    12: ("Italian", "Other Indo-European languages"),
    15: ("Portuguese", "Other Indo-European languages"),
    18: ("German", "German or other West Germanic languages"),
    21: ("Yiddish, Pennsylvania Dutch or other West Germanic languages",
         "German or other West Germanic languages"),
    24: ("Greek", "Other Indo-European languages"),
    27: ("Russian", "Russian, Polish, or other Slavic languages"),
    30: ("Polish", "Russian, Polish, or other Slavic languages"),
    33: ("Serbo-Croatian", "Russian, Polish, or other Slavic languages"),
    36: ("Ukrainian or other Slavic languages", "Russian, Polish, or other Slavic languages"),
    39: ("Armenian", "Other Indo-European languages"),
    42: ("Persian (incl. Farsi, Dari)", "Other Indo-European languages"),
    45: ("Gujarati", "Other Indo-European languages"),
    48: ("Hindi", "Other Indo-European languages"),
    51: ("Urdu", "Other Indo-European languages"),
    54: ("Punjabi", "Other Indo-European languages"),
    57: ("Bengali", "Other Indo-European languages"),
    60: ("Nepali, Marathi, or other Indic languages", "Other Indo-European languages"),
    63: ("Other Indo-European languages", "Other Indo-European languages"),
    66: ("Telugu", "Other Asian and Pacific Island languages"),
    69: ("Tamil", "Other Asian and Pacific Island languages"),
    72: ("Malayalam, Kannada, or other Dravidian languages", "Other Asian and Pacific Island languages"),
    75: ("Chinese (incl. Mandarin, Cantonese)", "Chinese (incl. Mandarin, Cantonese)"),
    78: ("Japanese", "Other Asian and Pacific Island languages"),
    81: ("Korean", "Korean"),
    84: ("Hmong", "Other Asian and Pacific Island languages"),
    87: ("Vietnamese", "Vietnamese"),
    90: ("Khmer", "Other Asian and Pacific Island languages"),
    93: ("Thai, Lao, or other Tai-Kadai languages", "Other Asian and Pacific Island languages"),
    96: ("Other languages of Asia", "Other Asian and Pacific Island languages"),
    99: ("Tagalog (incl. Filipino)", "Tagalog (incl. Filipino)"),
    102: ("Ilocano, Samoan, Hawaiian, or other Austronesian languages",
          "Other Asian and Pacific Island languages"),
    105: ("Arabic", "Arabic"),
    108: ("Hebrew", "Other and unspecified languages"),
    111: ("Amharic, Somali, or other Afro-Asiatic languages", "Other and unspecified languages"),
    114: ("Yoruba, Twi, Igbo, or other languages of Western Africa", "Other and unspecified languages"),
    117: ("Swahili or other languages of Central, Eastern, and Southern Africa",
          "Other and unspecified languages"),
    120: ("Navajo", "Other and unspecified languages"),
    123: ("Other Native languages of North America", "Other and unspecified languages"),
    126: ("Other and unspecified languages", "Other and unspecified languages"),
}
B_LABEL = {col: lab for col, (lab, _) in B_GROUPS.items()}

# PUMS LANP code -> B16001 column. The user note (2016_Language_User_Note.pdf) gives each B group's
# examples; codes it does not name are placed by family and checked by check 4.
LANP_B = {
    1000: 126, 1025: 126,                       # English-based creoles: "Other and unspecified" (note: Jamaican Creole)
    1055: 9, 1069: 15,                          # Haitian; Kabuverdianu with Portuguese (note)
    1110: 18, 1120: 18,                         # German, Swiss German
    1125: 21, 1130: 21, 1132: 21, 1134: 21,     # Pennsylvania German, Yiddish, Dutch, Afrikaans
    1140: 63, 1141: 63, 1142: 63,               # Scandinavian: Other Indo-European (note)
    1155: 12, 1170: 6, 1175: 6, 1200: 3, 1210: 15,
    1220: 63, 1231: 63, 1235: 24, 1242: 63,     # Romanian, Irish, Greek, Albanian
    1250: 27, 1260: 36, 1262: 36, 1263: 36, 1270: 30, 1273: 36, 1274: 36,
    1275: 33, 1276: 33, 1277: 33, 1278: 33,
    1281: 63, 1283: 63, 1288: 39, 1290: 42, 1292: 42, 1315: 63, 1327: 63,
    1340: 60,                                   # India N.E.C.: "other Indic"
    1350: 48, 1360: 51, 1380: 57, 1420: 54, 1435: 60, 1440: 60, 1450: 45, 1500: 60,
    1530: 60, 1540: 60, 1564: 63,               # Sinhala, Other Indo-Iranian: other Indic; Other IE
    1565: 126, 1582: 126,                       # Finnish, Hungarian: "Other and unspecified" (note: Hungarian)
    1675: 96, 1690: 96,                         # Turkish (note), Mongolian: Other languages of Asia
    1730: 66, 1737: 72, 1750: 72, 1765: 69,
    1900: 90, 1960: 87, 1970: 75, 2000: 75, 2030: 75, 2050: 75,
    2100: 96, 2160: 96, 2270: 96, 2350: 96,     # Tibetan, Burmese, Chin, Karen (note: Burmese, Karen)
    2430: 93, 2475: 93, 2525: 96, 2535: 84, 2560: 78, 2575: 81,
    2715: 102, 2770: 102, 2850: 96,
    2910: 99, 2920: 99,
    2950: 102, 3150: 102, 3190: 102, 3220: 102, 3270: 102, 3350: 102, 3420: 102, 3500: 102,
    3570: 102, 3600: 102,
    4500: 105, 4545: 108,
    4560: 111, 4565: 111, 4590: 111, 4640: 111, 4830: 111, 4840: 111, 4880: 111,
    4900: 117,                                  # Nilo-Saharan
    5150: 117, 5345: 117, 5525: 117, 5645: 117,
    5845: 114, 5900: 114, 5940: 114, 5950: 114, 6120: 114, 6205: 114, 6230: 114, 6290: 114,
    6300: 114, 6370: 114, 6500: 114,
    # Other languages of Africa. B16001 groups the 1,333 DETAILED codes, and two PUMS codes hold
    # detailed codes from more than one group: 6795 ("Nigeria N.E.C." and "Mali N.E.C." beside
    # "Kenya N.E.C." and Khoisan) and 1025 (Caribbean creoles beside Krio and Nigerian Pidgin).
    # With both in "Other and unspecified" check 4 read that group +13.2% and the two African
    # groups -2.0% and -4.6%; 6795 here gives +1.9%, +3.4%, -4.6%. Either way it only moves
    # which group's mix a few tens of thousands of people are split by.
    6795: 114,
    6800: 123, 6839: 123, 6930: 123, 6933: 120, 7019: 123, 7060: 123, 7124: 123,
    7300: 126, 9999: 126,
}


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        path = RAW / name
        if path.exists() and path.stat().st_size > 0:
            continue
        print(f"  fetching {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(req, timeout=3600) as r, open(tmp, "wb") as fh:
            while chunk := r.read(1 << 22):
                fh.write(chunk)
        tmp.replace(path)


def lanp_labels():
    import openpyxl
    wb = openpyxl.load_workbook(RAW / "ACSPUMS2020_2024CodeLists.xlsx", read_only=True)
    out = {}
    for i, r in enumerate(wb["Language"].iter_rows(values_only=True)):
        if i >= 3 and isinstance(r[0], int) and r[1]:
            out[r[0]] = str(r[1]).strip()
    return out


def read_sf(name, table, ncols):
    """A table-based summary file: GEO_ID | <table>_E001 | <table>_M001 | ...; estimates only."""
    cols = ["GEO_ID"] + [f"{table}_E{c:03d}" for c in range(1, ncols + 1)]
    df = pd.read_csv(RAW / name, sep="|", usecols=cols, dtype={"GEO_ID": str})
    df.columns = ["GEO_ID"] + list(range(1, ncols + 1))
    return df


def pums_aggregate():
    """Weighted persons aged 5+ by (state, PUMA, LANP) from the 5-year person file; cached."""
    if PUMS_AGG.exists():
        return pd.read_csv(PUMS_AGG, dtype={"st": str, "puma": str})
    import pyarrow as pa
    import pyarrow.csv as pacsv
    pa.set_cpu_count(4)
    parts = []
    with zipfile.ZipFile(RAW / "csv_pus_2024_5y.zip") as z:
        for member in sorted(n for n in z.namelist() if n.endswith(".csv")):
            print(f"  reading {member}…", flush=True)
            with z.open(member) as fh:
                t = pacsv.read_csv(fh, convert_options=pacsv.ConvertOptions(
                    include_columns=["STATE", "PUMA", "PWGTP", "AGEP", "LANX", "LANP"],
                    column_types={"STATE": pa.string(), "PUMA": pa.string(), "LANP": pa.string(),
                                  "LANX": pa.string()}))
            d = t.to_pandas()
            d = d[d["AGEP"] >= 5]
            d["lanp"] = d["LANP"].fillna("")
            d.loc[d["LANX"] == "2", "lanp"] = "EN"
            if (d["lanp"] == "").any():
                raise SystemExit(f"{member}: {int((d['lanp'] == '').sum())} persons 5+ with no LANX/LANP")
            g = d.groupby(["STATE", "PUMA", "lanp"]).agg(w=("PWGTP", "sum"), n=("PWGTP", "size"))
            parts.append(g.reset_index())
    agg = pd.concat(parts).groupby(["STATE", "PUMA", "lanp"], as_index=False)[["w", "n"]].sum()
    agg = agg.rename(columns={"STATE": "st", "PUMA": "puma"})
    agg.to_csv(PUMS_AGG, index=False)
    return pd.read_csv(PUMS_AGG, dtype={"st": str, "puma": str})


def tract_puma_key(tract_ids):
    """Tract GEOID -> state+PUMA (7 digits), from the 2020 tract-to-PUMA relationship file.

    Connecticut's ACS tracts carry the 2022 planning-region county codes (09110-09190) while the
    relationship file has the old counties. The tracts themselves did not change, so each new CT
    tract (cb_2024) is matched to the 2020 tract (cb_2020) holding its representative point, and
    the two tract codes are asserted equal. Check 2 proves the key."""
    import geopandas as gpd
    rel = pd.read_csv(RD_GEO / "tract_to_puma_2020.txt", dtype=str, encoding="utf-8-sig")
    rel["geoid"] = rel["STATEFP"] + rel["COUNTYFP"] + rel["TRACTCE"]
    rel["puma"] = rel["STATEFP"] + rel["PUMA5CE"]
    key = dict(zip(rel["geoid"], rel["puma"]))
    new = gpd.read_file(f"zip://{RD_GEO / 'cb_2024_us_tract_500k.zip'}", where="STATEFP = '09'")
    old = gpd.read_file(f"zip://{RD_GEO / 'cb_2020_us_tract_500k.zip'}", where="STATEFP = '09'")
    pts = gpd.GeoDataFrame({"new": new["GEOID"]}, geometry=new.representative_point(), crs=new.crs)
    j = gpd.sjoin(pts, old[["GEOID", "geometry"]].to_crs(new.crs), predicate="within")
    if j["new"].duplicated().any() or len(j) != len(new) or (j["new"].str[5:] != j["GEOID"].str[5:]).any():
        raise SystemExit("CT: new tracts do not match the 2020 tracts one to one on code and place")
    for n, o in zip(j["new"], j["GEOID"]):
        key[n] = key[o]
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
    nat = c[c["GEO_ID"] == "0100000US"].iloc[0]
    tr = c[c["GEO_ID"].str.startswith("1400000US")].copy()
    tr["geo_id"] = tr["GEO_ID"].str[9:]
    tr["st"] = tr["geo_id"].str[:2]
    pr = tr[tr["st"].isin(EXCLUDE_STATES)]
    print(f"C16001: {len(tr):,} tracts; Puerto Rico's {len(pr):,} set aside "
          f"({pr[1].sum():,.0f} people 5+)")
    tot = sum(nat[col] for col in c_cols)
    assert tot == nat[1], (tot, nat[1])
    tr = tr[~tr["st"].isin(EXCLUDE_STATES) & (tr[1] > 0)].copy()   # empty (water) tracts dropped
    tr["puma"] = tract_puma_key(tr["geo_id"])

    pu_c = c[c["GEO_ID"].str.startswith("795P200US")].copy()
    pu_c["puma"] = pu_c["GEO_ID"].str[9:]
    pu_c = pu_c[~pu_c["puma"].str[:2].isin(EXCLUDE_STATES)].set_index("puma")

    # The file lacks 14 Suffolk County (NY) tracts that the county and PUMA rows include. Their
    # people are each PUMA's remainder, drawn as one unit per PUMA over those tracts' polygons.
    have = set(tr["geo_id"])
    lost = sorted(t for t, p in MISSING_TRACTS.items() if t not in have)
    if lost != sorted(MISSING_TRACTS):
        raise SystemExit(f"MISSING_TRACTS changed: pinned {len(MISSING_TRACTS)}, absent {len(lost)}")
    sums = tr.groupby("puma")[list(c_cols)].sum()
    resid = pu_c.loc[sums.index, list(c_cols)] - sums
    resid = resid[(resid != 0).any(axis=1)]
    # Elsewhere the relationship file puts a few tracts in the neighbouring PUMA from the one the
    # 2024 tables count them in: pairs of PUMAs off by equal and opposite amounts. They only
    # change which PUMA's mix a tract's groups are split by; bounded and printed, not fixed.
    moved = resid[~resid.index.isin(set(MISSING_TRACTS.values()))]
    if moved.sum().abs().max() != 0 or moved.abs().to_numpy().sum() > 2 * 5000:
        raise SystemExit(f"tract -> PUMA key: remainders that are not pairwise moves:\n{moved}")
    print(f"  tract -> PUMA key: {len(moved)} PUMAs differ from their tracts by moves between "
          f"neighbours, {int(moved.abs().to_numpy().sum() / 2):,} people 5+ in all (bounded at 5,000)")
    resid = resid[resid.index.isin(set(MISSING_TRACTS.values()))]
    if set(resid.index) != set(MISSING_TRACTS.values()) or (resid < 0).any().any():
        raise SystemExit(f"Suffolk remainders not as pinned:\n{resid}")
    extra = resid.reset_index().rename(columns={"index": "puma"})
    extra["geo_id"] = "36103-rest-" + extra["puma"]
    extra["st"] = "36"
    print(f"  {len(MISSING_TRACTS)} Suffolk County tracts absent from the tract rows; their PUMAs' "
          f"remainders ({int(resid.to_numpy().sum()):,} people 5+) become {len(extra)} units")
    tr = pd.concat([tr, extra[tr.columns.intersection(extra.columns)]], ignore_index=True)
    for col in c_cols:                      # check 1 (nation excludes PR: the PRCS is separate)
        got = tr[col].sum()
        if got != nat[col]:
            raise SystemExit(f"check 1: tracts sum {got:,} != nation {nat[col]:,} in {c_cols[col]}")
    print(f"  check 1 ok: tracts (and the two Suffolk remainders) sum to the nation in all 13 rows "
          f"({nat[1]:,} people 5+)")
    sums = tr.groupby("puma")[list(c_cols)].sum()
    if set(sums.index) != set(pu_c.index):
        raise SystemExit(f"check 2: PUMA sets differ: {sorted(set(sums.index) ^ set(pu_c.index))[:5]}")
    ok = ~sums.index.isin(moved.index)
    diff = (sums[ok] - pu_c.loc[sums.index[ok], list(c_cols)]).abs().to_numpy().sum()
    if diff:
        raise SystemExit(f"check 2: tracts do not sum to their PUMA (total abs diff {diff:,})")
    print(f"  check 2 ok: tracts sum exactly to {int(ok.sum()):,} of {len(sums):,} PUMAs in every "
          f"row; the other {int((~ok).sum())} are the moves above")

    long = tr.melt(id_vars=["geo_id", "puma"], value_vars=list(c_cols), var_name="col",
                   value_name="count")
    long["source_category"] = long["col"].map(c_cols)
    long["geo_level"] = "tract"
    long = long[long["count"] > 0]
    long[["geo_id", "geo_level", "puma", "source_category", "count"]].to_csv(NORM / "us.csv", index=False)
    print(f"  wrote us.csv: {len(long):,} rows")

    # ---- B16001 at PUMA; check 3 ----
    b = read_sf("acsdt5y2024-b16001.dat", "B16001", 128)
    bnat = b[b["GEO_ID"] == "0100000US"].iloc[0]
    pb = b[b["GEO_ID"].str.startswith("795P200US")].copy()
    pb["puma"] = pb["GEO_ID"].str[9:]
    pb = pb[~pb["puma"].str[:2].isin(EXCLUDE_STATES)].set_index("puma")
    for ccol, cname in C_GROUPS.items():
        bcols = [bc for bc, (_, cg) in B_GROUPS.items() if cg == cname]
        d = (pb[bcols].sum(axis=1) - pu_c.loc[pb.index, ccol]).abs().sum()
        if d:
            raise SystemExit(f"check 3: B16001 {bcols} do not sum to C16001 {cname} (abs diff {d:,})")
    print(f"  check 3 ok: the 42 B16001 groups sum to the 12 C16001 groups at all {len(pb):,} PUMAs")

    # ---- PUMS; check 4 ----
    labels = lanp_labels()
    pums = pums_aggregate()
    pums = pums[~pums["st"].isin(EXCLUDE_STATES)]
    pums["puma"] = pums["st"] + pums["puma"].str.zfill(5)
    nonen = pums[pums["lanp"] != "EN"].copy()
    nonen["code"] = nonen["lanp"].astype(int)
    unknown = sorted(set(nonen["code"]) - set(LANP_B))
    if unknown:
        raise SystemExit(f"LANP codes with no B16001 group: {unknown}")
    missing_label = sorted(set(LANP_B) - set(labels))
    if missing_label:
        raise SystemExit(f"LANP codes not in the code list: {missing_label}")
    nonen["bcol"] = nonen["code"].map(LANP_B)
    en_w = pums.loc[pums["lanp"] == "EN", "w"].sum()
    print(f"  PUMS: {pums['n'].sum():,} persons 5+ in {pums['puma'].nunique():,} PUMAs; English only "
          f"{en_w:,.0f} against C16001's {nat[2]:,} ({en_w / nat[2] - 1:+.2%})")
    print("  check 4, PUMS against B16001 nationally, per B group:")
    bad = []
    pw = nonen.groupby("bcol")["w"].sum()
    for bcol, lab in B_LABEL.items():
        pub, got = bnat[bcol], pw.get(bcol, 0.0)
        r = got / pub - 1 if pub else 0.0
        flag = "  !!" if pub > 100_000 and abs(r) > 0.05 else ""
        print(f"    {lab[:60]:60s} {pub:>12,} {got:>12,.0f} {r:+7.2%}{flag}")
        if flag:
            bad.append(lab)
    if bad:
        raise SystemExit(f"check 4: {bad}")

    # ---- the split: per PUMA and C group, each LANP code's share ----
    bl = pb[list(B_GROUPS)].reset_index().melt(id_vars="puma", var_name="bcol", value_name="b")
    bl["cgroup"] = bl["bcol"].map(lambda x: B_GROUPS[x][1])
    bl["c"] = bl.groupby(["puma", "cgroup"])["b"].transform("sum")
    bl = bl[bl["b"] > 0]
    bl["b_share"] = bl["b"] / bl["c"]

    mix_p = nonen.groupby(["puma", "bcol", "code"], as_index=False)["w"].sum()
    nonen["st2"] = nonen["puma"].str[:2]
    mix_s = nonen.groupby(["st2", "bcol", "code"], as_index=False)["w"].sum()
    mix_n = nonen.groupby(["bcol", "code"], as_index=False)["w"].sum()
    have_p = set(map(tuple, mix_p[["puma", "bcol"]].drop_duplicates().to_numpy()))
    have_s = set(map(tuple, mix_s[["st2", "bcol"]].drop_duplicates().to_numpy()))

    rows, fb = [], {"puma": 0.0, "state": 0.0, "nation": 0.0}
    gp = {k: g for k, g in mix_p.groupby(["puma", "bcol"])}
    gs = {k: g for k, g in mix_s.groupby(["st2", "bcol"])}
    gn = {k: g for k, g in mix_n.groupby("bcol")}
    for r in bl.itertuples(index=False):
        if (r.puma, r.bcol) in have_p:
            g, how = gp[(r.puma, r.bcol)], "puma"
        elif (r.puma[:2], r.bcol) in have_s:
            g, how = gs[(r.puma[:2], r.bcol)], "state"
        else:
            g, how = gn[r.bcol], "nation"
        fb[how] += r.b
        w = g["w"].to_numpy(dtype=float)
        for code, s in zip(g["code"].to_numpy(), w / w.sum()):
            rows.append((r.puma, r.cgroup, B_LABEL[r.bcol], int(code), labels[int(code)],
                         r.b_share * s, how))
    sp = pd.DataFrame(rows, columns=["puma", "c_group", "b_group", "lanp", "source_category",
                                     "share", "fallback"])
    chk = sp.groupby(["puma", "c_group"])["share"].sum()
    assert (chk - 1).abs().max() < 1e-9, chk[(chk - 1).abs() >= 1e-9].head()
    sp.to_csv(NORM / "us_split.csv", index=False)
    tot_b = sum(fb.values())
    print(f"  wrote us_split.csv: {len(sp):,} rows; people in B16001 groups whose mix came from the "
          + ", ".join(f"{k} {v:,.0f} ({v / tot_b:.2%})" for k, v in fb.items()))


if __name__ == "__main__":
    main()
