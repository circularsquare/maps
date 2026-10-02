"""Mother tongue for every Indian admin unit, from Census 2011 table C-16.

Writes `countries/india/composition.json`, which the viewer draws as a pie over each state,
district and sub-district. Nothing newer exists: the 2021 census never happened, so this is
2011 everywhere, and the pies are sized by helper1m's projected population, not by it.

THE SOURCE. C-16 gives Total/Rural/Urban persons by mother tongue at state, district and
sub-district level, 35 state workbooks (fetch_c16.py). Each unit's block has two tiers, a
census LANGUAGE row (`006000 6 HINDI`) followed by the MOTHER TONGUES grouped under it
(`006102 Bhojpuri`, `006240 Hindi`, `006999 6 Others`). Only the mother-tongue rows are read
here: they sum to the language rows, and they are the finer of the two. The sub-district
codes are the census's own, the same ones SHRUG's polygons carry, so the join to helper1m's
level 3 is exact; districts and states are summed up from it through parent_code.

GROUPS. India reports ~350 mother tongues. Hindi alone is 528M people and folds in Bhojpuri,
Rajasthani, Chhattisgarhi, Magahi, Haryanvi and fifty more, so a pie of census languages
would paint the whole north one colour. So a mother tongue gets its own colour when it has a
million speakers nationally or is SHARE_MIN of some sub-district with ABS_MIN speakers there;
the remaining mother tongues of a language fold into an "other" group for that language if
the remainder itself passes that test, and otherwise into the language's biggest group. A
language with no mother tongue passing is one group if it passes as a whole, and otherwise
goes to "Other languages" with the census's own 124 OTHERS.

Note that the split between "Hindi" and a variety under it is how people answered, not where
the speech changes: Bihar reports Bhojpuri and Magahi far more readily than eastern Uttar
Pradesh does, so the state line shows as a jump.

TWO REPAIRS.
  1. Shajapur district (Madhya Pradesh): five tehsils (Agar, Shajapur, Gulana, Moman
     Badodiya, Shujalpur) carry a single `Hindi` row equal to their whole population, and the
     sixth, Kalapipal, carries language rows for all six. Their real mix is the district's
     mother tongues minus every other tehsil's, spread over the five by population. Any unit
     shaped like that is repaired the same way, and reported.
  2. The 23 `99999` units, "Area not under any Sub-district": 17.4M people in West Bengal,
     Tripura and Karnataka who live in municipal towns outside any block, with no polygon of
     their own. Spreading them over their district would put the Kolkata fringe's languages
     in the Sundarbans. Their towns add up to them exactly (religiondots' in_towns.csv, from
     C-01), and every town has a SHRUG polygon filed under a real sub-district, so each
     unit's mix goes to the sub-districts its towns stand in, in proportion to town
     population. This differs from helper1m's own level-3 population, which spreads those
     people over the district's blocks pro rata.

Reads:
    data/india/c16/DDW-C16-STMT-MDDS-<ss>00.XLSX         fetch_c16.py
    ../religiondots/data/geo/in/in_towns.csv             C-01 town rows (religiondots in_place.py)
    ../religiondots/data/geo/in/shrug-village-pc11.parquet   SHRUG town polygons
    countries/india/adm3.geojson                         helper1m's sub-districts
    scripts/india/language_colors.csv                    the palette, hand-editable

Writes:
    countries/india/composition.json
    scripts/india/language_colors.csv   only rows for groups it lacks; existing rows are kept

Usage:
    python scripts/india/language.py            # from helper1m/
    python scripts/india/language.py --reread   # re-parse the workbooks, ignoring the cache
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "6")

import argparse
import colorsys
import csv
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HELPER = Path(__file__).resolve().parents[2]
MAPS = HELPER.parent
RAW = HELPER / "data" / "india" / "c16"
CACHE = RAW / "c16_subdistrict.pkl"
RD_GEO = MAPS / "religiondots" / "data" / "geo" / "in"
TOWNS = RD_GEO / "in_towns.csv"
VILLAGES = RD_GEO / "shrug-village-pc11.parquet"
COUNTRY = HELPER / "countries" / "india"
ADM3 = COUNTRY / "adm3.geojson"
OUT = COUNTRY / "composition.json"
COLORS = Path(__file__).with_name("language_colors.csv")

NAT_MIN = 1_000_000     # speakers nationally for a mother tongue to get its own colour
SHARE_MIN = 0.30        # ... or this share of some sub-district
ABS_MIN = 10_000        # ... with at least this many speakers there
RESIDUAL_SD = "99999"
OTHER_KEY = "other"

# Census language code -> family, for colouring groups the palette file has no row for.
DRAVIDIAN = {"007", "011", "020", "021", "037", "041", "044", "049", "058", "060", "063",
             "065", "069", "070", "072", "083", "097", "117"}
AUSTROASIATIC = {"018", "032", "048", "050", "054", "055", "062", "067", "068", "091",
                 "092", "093", "106"}
SINO_TIBETAN = {"003", "012", "023", "025", "026", "027", "029", "031", "034", "035", "036",
                "038", "039", "042", "043", "046", "047", "051", "052", "056", "057", "059",
                "061", "064", "066", "071", "073", "074", "076", "077", "078", "079", "080",
                "081", "082", "084", "085", "086", "087", "088", "089", "090", "094", "095",
                "096", "098", "100", "101", "102", "103", "104", "105", "107", "108", "111",
                "112", "113", "114", "115", "116", "118", "119", "120", "121", "122", "123"}
# Hue bands (degrees) and the language codes in each; Indo-Aryan is everything else.
FAMILY_HUES = {"hindi": (0, 50), "indo_aryan": (55, 200), "dravidian": (150, 200),
               "austroasiatic": (20, 45), "sino_tibetan": (230, 320), "other": (0, 0)}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def family(lang):
    if lang == "006":
        return "hindi"
    if lang in DRAVIDIAN:
        return "dravidian"
    if lang in AUSTROASIATIC:
        return "austroasiatic"
    if lang in SINO_TIBETAN:
        return "sino_tibetan"
    if lang in ("024", "028", "040", "124"):
        return "other"
    return "indo_aryan"


# ---------------------------------------------------------------------------- reading

def read_c16(reread=False):
    """Every row of the 35 state workbooks below state level: s, d, sd, mt, name, p.

    `p` is Total persons. Cached, since parsing the workbooks takes about a minute.
    """
    if CACHE.exists() and not reread:
        return pd.read_pickle(CACHE)
    files = [p for p in sorted(glob.glob(str(RAW / "DDW-C16-STMT-MDDS-*.XLSX")))
             if not p.endswith("0000.XLSX")]
    if len(files) != 35:
        sys.exit(f"expected 35 state workbooks in {RAW}, found {len(files)}: "
                 f"run scripts/india/fetch_c16.py")
    frames = []
    for p in files:
        head = pd.read_excel(p, header=None, dtype=str, nrows=6)
        # Assert the layout before trusting any column by position.
        if not (str(head.iloc[2, 5]).startswith("Mother") and head.iloc[2, 7] == "Total"
                and head.iloc[3, 7] == "P"):
            sys.exit(f"{p}: header not where expected:\n{head}")
        df = pd.read_excel(p, header=None, dtype=str, skiprows=6).iloc[:, :8]
        df.columns = ["tab", "s", "d", "sd", "area", "mt", "name", "p"]
        df = df.dropna(subset=["mt"])
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    for c, w in (("s", 2), ("d", 3), ("sd", 5), ("mt", 6)):
        df[c] = df[c].str.strip().str.zfill(w)
    df["p"] = df["p"].astype(np.int64)
    df["name"] = df["name"].str.strip()
    df = df[df["d"] != "000"]
    df.to_pickle(CACHE)
    log(f"parsed {len(files)} workbooks, {len(df):,} rows below state level")
    return df


def unit_matrix(df):
    """Sub-district x mother-tongue counts, with the Shajapur-shaped repair applied.

    Returns (units, mts, M) where units is a list of (d, sd), mts the mother-tongue codes and
    M the count matrix. District rows are kept only to check and repair against.
    """
    is_lang = df["mt"].str.endswith("000")
    # A language with no mother tongues under it (only 124 OTHERS) is read as its own
    # mother tongue, or its people vanish.
    childless = set(df.loc[is_lang, "mt"].str[:3]) - set(df.loc[~is_lang, "mt"].str[:3])
    log(f"  languages with no mother-tongue rows, read whole: {sorted(childless)}")
    df = df.copy()
    df["is_lang"] = is_lang & ~df["mt"].str[:3].isin(childless)
    is_lang = df["is_lang"]
    sub = df[df["sd"] != "00000"]
    dist = df[df["sd"] == "00000"]

    # Every language row should equal the sum of its mother tongues. Report where not.
    lr = sub[is_lang.loc[sub.index]].set_index(["d", "sd", sub.loc[is_lang.loc[sub.index], "mt"].str[:3]])["p"]
    mr = sub[~is_lang.loc[sub.index]].groupby(["d", "sd", sub.loc[~is_lang.loc[sub.index], "mt"].str[:3]])["p"].sum()
    lr.index.names = mr.index.names = ["d", "sd", "l"]
    cmp = pd.concat([lr.rename("lang"), mr.rename("mt")], axis=1).fillna(0)
    cmp = cmp[~cmp.index.get_level_values("l").isin(childless)]
    off = cmp[cmp["lang"] != cmp["mt"]].reset_index()
    if len(off):
        log(f"  language rows that are not the sum of their mother tongues, by unit: "
            f"{sorted(set(off['d'] + off['sd']))}")

    mt_rows = sub[~is_lang.loc[sub.index]]
    mts = sorted(df.loc[~is_lang, "mt"].unique())
    mt_at = {m: i for i, m in enumerate(mts)}
    units = sorted(set(zip(sub["d"], sub["sd"])))
    unit_at = {u: i for i, u in enumerate(units)}
    M = np.zeros((len(units), len(mts)), dtype=np.int64)
    np.add.at(M, ([unit_at[u] for u in zip(mt_rows["d"], mt_rows["sd"])],
                  [mt_at[m] for m in mt_rows["mt"]]), mt_rows["p"].to_numpy())

    dist_mt = dist[~is_lang.loc[dist.index]]
    D = {}
    for d, g in dist_mt.groupby("d"):
        v = np.zeros(len(mts), dtype=np.int64)
        np.add.at(v, [mt_at[m] for m in g["mt"]], g["p"].to_numpy())
        D[d] = v

    # A unit with mother-tongue rows but no language rows is the Shajapur shape: its one row
    # is its whole population, and its real mix sits in a sibling's language rows.
    has_lang = set(zip(sub.loc[is_lang.loc[sub.index], "d"],
                       sub.loc[is_lang.loc[sub.index], "sd"]))
    bare = [u for u in units if u not in has_lang]
    for d in sorted({u[0] for u in bare}):
        rows = [unit_at[u] for u in units if u[0] == d]
        bare_rows = [unit_at[u] for u in bare if u[0] == d]
        good = [r for r in rows if r not in bare_rows]
        resid = D[d] - M[good].sum(axis=0)
        bare_tot = M[bare_rows].sum(axis=1)
        if resid.min() < 0 or resid.sum() != bare_tot.sum():
            sys.exit(f"district {d}: residual does not account for its bare units "
                     f"({resid.sum():,} vs {bare_tot.sum():,})")
        for r, t in zip(bare_rows, bare_tot):
            M[r] = 0
        # Largest-remainder split, so integers still add up to the residual.
        share = resid[None, :] * (bare_tot / bare_tot.sum())[:, None]
        base = np.floor(share).astype(np.int64)
        for j in np.flatnonzero(resid):
            left = resid[j] - base[:, j].sum()
            order = np.argsort(-(share[:, j] - base[:, j]))
            base[order[:left], j] += 1
        M[bare_rows] = base
        names = sorted(set(sub.loc[(sub["d"] == d) & sub["sd"].isin(
            [units[r][1] for r in bare_rows]), "area"]))
        log(f"  repaired district {d}: {len(bare_rows)} tehsils with one bare row "
            f"({', '.join(names)}), {resid.sum():,} people re-split from the district")

    # Every district must now be exactly the sum of its sub-districts, mother tongue by
    # mother tongue.
    for d, v in D.items():
        got = M[[unit_at[u] for u in units if u[0] == d]].sum(axis=0)
        if not np.array_equal(got, v):
            sys.exit(f"district {d}: sub-districts do not sum to the district")
    names = df.drop_duplicates("mt").set_index("mt")["name"]
    log(f"{len(units):,} sub-districts x {len(mts)} mother tongues, "
        f"{M.sum():,} people")
    return units, mts, M, names


# ---------------------------------------------------------------------------- groups

def lang_names(df):
    lr = df[df["mt"].str.endswith("000")].drop_duplicates("mt")
    # "6 HINDI" -> "Hindi"; "30 BHILI/BHILODI" -> "Bhili/Bhilodi"
    return {m[:3]: " ".join(n.split()[1:]).title() for m, n in zip(lr["mt"], lr["name"])}


def passes(col, tot):
    """The test for a colour of its own, on one column of unit counts."""
    if col.sum() >= NAT_MIN:
        return True
    share = col / np.maximum(tot, 1)
    return bool(((share >= SHARE_MIN) & (col >= ABS_MIN)).any())


def make_groups(mts, M, mt_names, lnames):
    """Map every mother tongue to a group. Returns (groups, mt_group index array)."""
    tot = M.sum(axis=1)
    by_lang = {}
    for j, m in enumerate(mts):
        by_lang.setdefault(m[:3], []).append(j)

    groups, assign = [], np.full(len(mts), -1)

    def add(key, en, lang, title):
        groups.append({"key": key, "en": en, "lang": lang, "title": title})
        return len(groups) - 1

    other = None
    for lang in sorted(by_lang):
        cols = by_lang[lang]
        lname = lnames.get(lang, lang)
        if lang == "124":
            continue
        own = [j for j in cols if not mts[j].endswith("999") and passes(M[:, j], tot)]
        rest = [j for j in cols if j not in own]
        if not own:
            if passes(M[:, cols].sum(axis=1), tot):
                g = add(f"l{lang}", lname, lang, f"Census language {lname}, all mother tongues")
                assign[cols] = g
            continue
        for j in sorted(own, key=lambda j: -M[:, j].sum()):
            name = mt_names[mts[j]]
            title = name if name.lower() == lname.lower() else \
                f"{name}, a mother tongue grouped under {lname} in the census"
            assign[j] = add(mts[j], name, lang, title)
        if rest:
            if len(own) > 1 and passes(M[:, rest].sum(axis=1), tot):
                assign[rest] = add(f"l{lang}x", f"Other {lname}", lang,
                                   f"Smaller mother tongues grouped under {lname}, and its "
                                   f"unclassified 'Others'")
            else:
                assign[rest] = assign[max(own, key=lambda j: M[:, j].sum())]
    other = add(OTHER_KEY, "Other languages", "124",
                "Languages too small or scattered for a colour of their own, and the "
                "census's 'Others'")
    assign[assign < 0] = other
    return groups, assign


# ---------------------------------------------------------------------------- colours

HAND = {
    # The big ones, picked so that neighbours on the map differ.
    "006240": "#f2c45a",  # Hindi
    "l006x": "#dcc08f",   # Other Hindi
    "006102": "#d8443c",  # Bhojpuri
    "006489": "#ec8a35",  # Rajasthani
    "006142": "#b8a038",  # Chhattisgarhi
    "006376": "#e377c2",  # Magahi
    "006235": "#9c5a2c",  # Haryanvi
    "006320": "#f2a29a",  # Khortha
    "006400": "#c4651a",  # Marwari
    "006125": "#8a6a32",  # Bundeli
    "006391": "#d9a77c",  # Malvi
    "006503": "#b44f86",  # Sadri
    "006408": "#f5b06b",  # Mewari
    "006030": "#e8d48f",  # Awadhi
    "010008": "#7d2e5c",  # Maithili
    "002007": "#2e8b57",  # Bengali
    "015043": "#9acd5a",  # Odia
    "013071": "#8a6bbf",  # Marathi
    "021046": "#3d7cc9",  # Telugu
    "020027": "#d1495b",  # Tamil
    "007016": "#6f8f1e",  # Kannada; a yellow vanishes into the density fill
    "011016": "#2a9d8f",  # Malayalam
    "005018": "#4fb3bf",  # Gujarati
    "022015": "#1b3f6e",  # Urdu
    "016038": "#4169e1",  # Punjabi
    "001002": "#c9a0dc",  # Assamese
    "008005": "#5b8db8",  # Kashmiri
    "004001": "#a3c4e0",  # Dogri
    "014011": "#5f9ea0",  # Nepali
    "018040": "#6b4226",  # Santali
    "009011": "#b07aa1",  # Konkani
    "117009": "#9c755f",  # Tulu
    "OTHER": "#9a9a9a",
}


def auto_color(fam, i):
    """A colour for the i-th group of a family, stepping through its hue band."""
    if fam == "other":
        return "#9a9a9a"
    lo, hi = FAMILY_HUES[fam]
    h = (lo + ((i * 0.618034) % 1) * (hi - lo)) / 360
    light = (0.42, 0.58, 0.70)[i % 3]
    sat = (0.55, 0.45, 0.60)[(i // 3) % 3]
    r, g, b = colorsys.hls_to_rgb(h, light, sat)
    return "#{:02x}{:02x}{:02x}".format(round(r * 255), round(g * 255), round(b * 255))


def palette(groups):
    """Colours from language_colors.csv; groups it lacks get one and are appended to it.

    The file is hers to hand-edit, so an existing row is never rewritten.
    """
    have = {}
    if COLORS.exists():
        with COLORS.open(encoding="utf-8", newline="") as fh:
            have = {r["key"]: r["color"] for r in csv.DictReader(fh)}
    new, count = [], {}
    for g in groups:
        if g["key"] in have:
            continue
        fam = family(g["lang"])
        c = HAND.get(g["key"]) or (HAND["OTHER"] if g["key"] == OTHER_KEY
                                   else auto_color(fam, count.get(fam, 0)))
        count[fam] = count.get(fam, 0) + 1
        have[g["key"]] = c
        new.append({"key": g["key"], "en": g["en"], "family": fam, "color": c})
    if new:
        exists = COLORS.exists()
        with COLORS.open("a", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["key", "en", "family", "color"])
            if not exists:
                w.writeheader()
            w.writerows(new)
        log(f"added {len(new)} rows to {COLORS.name}")
    return have


# ---------------------------------------------------------------------------- placement

def place_residuals(units, G, adm3_codes, adm3_geom):
    """Move each `99999` unit's counts onto the sub-districts its towns stand in.

    Returns (G with those rows emptied, extra: list of (adm3 code, counts, lon, lat)).
    """
    import pyarrow.parquet as pq
    import shapely

    res_rows = [i for i, (d, sd) in enumerate(units) if sd == RESIDUAL_SD]
    if not res_rows:
        return G, []
    towns = pd.read_csv(TOWNS, dtype={"unit": str, "town": str})
    towns = towns[towns["unit"].str.endswith(RESIDUAL_SD)]
    tbl = pq.read_table(VILLAGES, columns=["pc11_d_id", "pc11_sd_id", "pc11_tv_id", "geometry"],
                        filters=[("pc11_tv_id", "in", sorted(set(towns["town"])))])
    poly = tbl.to_pandas()
    poly["geometry"] = shapely.from_wkb(poly["geometry"])

    extra, spread = [], []
    for i in res_rows:
        d, sd = units[i]
        tw = towns[towns["unit"].str[2:] == d + sd]
        if tw.empty or tw["pop"].sum() != G[i].sum():
            sys.exit(f"{d}{sd}: towns sum to {tw['pop'].sum():,}, unit is {G[i].sum():,}")
        for t in tw.itertuples():
            cand = poly[(poly["pc11_tv_id"] == t.town) & (poly["pc11_d_id"] == d)]
            if len(cand) != 1:
                sys.exit(f"town {t.town} ({t.name}) in district {d}: {len(cand)} polygons")
            pt = cand.geometry.iloc[0].representative_point()
            code = cand["pc11_d_id"].iloc[0] + cand["pc11_sd_id"].iloc[0]
            if code not in adm3_codes:
                # Filed under no polygon helper1m has: take the sub-district it stands in.
                hits = [c for c, g in adm3_geom.items() if c[:3] == d and g.covers(pt)]
                if len(hits) != 1:
                    sys.exit(f"town {t.town} ({t.name}): in {len(hits)} sub-districts")
                code = hits[0]
                spread.append(t.name)
            extra.append((code, G[i] * (t.pop / G[i].sum()), pt.x, pt.y))
        G[i] = 0
    log(f"  {sum(len(towns[towns['unit'].str[2:] == units[i][0] + RESIDUAL_SD]) for i in res_rows)} "
        f"towns of {len(res_rows)} `99999` units placed in their sub-districts"
        + (f"; {len(spread)} by position, not by SHRUG's code" if spread else ""))
    return G, extra


# ---------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reread", action="store_true", help="re-parse the C-16 workbooks")
    args = ap.parse_args()

    df = read_c16(args.reread)
    units, mts, M, mt_names = unit_matrix(df)
    groups, assign = make_groups(mts, M, mt_names, lang_names(df))
    n_g = len(groups)
    G = np.zeros((len(units), n_g))
    for j in range(len(mts)):
        G[:, assign[j]] += M[:, j]
    log(f"{n_g} groups ({sum(1 for g in groups if not g['key'].startswith('l'))} single "
        f"mother tongues)")

    import shapely.geometry
    with ADM3.open(encoding="utf-8") as fh:
        adm3 = json.load(fh)
    geom = {f["properties"]["code"]: shapely.geometry.shape(f["geometry"])
            for f in adm3["features"]}
    parent = {f["properties"]["code"]: f["properties"]["parent_code"]
              for f in adm3["features"]}

    G, extra = place_residuals(units, G, set(geom), geom)

    # Level 3: each census sub-district onto the helper1m polygon with its code, weighted
    # at the polygon's representative point for the pie position.
    acc = {}   # code -> [counts, sum w*x, sum w*y, sum w]

    def put(code, v, x, y):
        a = acc.setdefault(code, [np.zeros(n_g), 0.0, 0.0, 0.0])
        w = v.sum()
        a[0] += v
        a[1] += w * x
        a[2] += w * y
        a[3] += w

    lost = 0
    for i, (d, sd) in enumerate(units):
        if not G[i].sum():
            continue
        code = d + sd
        if code not in geom:
            lost += G[i].sum()
            log(f"  !! no helper1m polygon for {code}: {G[i].sum():,.0f} people dropped")
            continue
        pt = geom[code].representative_point()
        put(code, G[i], pt.x, pt.y)
    for code, v, x, y in extra:
        put(code, v, x, y)
    nopie = sorted(set(geom) - set(acc))
    log(f"level 3: {len(acc):,} of {len(geom):,} polygons; no census unit for {nopie}")

    # Levels 2 and 1, summed up through helper1m's own parents.
    levels = {3: acc}
    up2 = {}
    for code, a in acc.items():
        b = up2.setdefault(parent[code], [np.zeros(n_g), 0.0, 0.0, 0.0])
        for k in range(4):
            b[k] = b[k] + a[k]
    levels[2] = up2
    state_of = {}
    adm2_path = COUNTRY / "adm2.geojson"
    with adm2_path.open(encoding="utf-8") as fh:
        for f in json.load(fh)["features"]:
            state_of[f["properties"]["code"]] = f["properties"]["parent_code"]
    up1 = {}
    for code, a in up2.items():
        b = up1.setdefault(state_of[code], [np.zeros(n_g), 0.0, 0.0, 0.0])
        for k in range(4):
            b[k] = b[k] + a[k]
    levels[1] = up1

    out = {}
    for lvl in (1, 2, 3):
        units_out = {}
        for code, (v, sx, sy, sw) in levels[lvl].items():
            v = np.rint(v).astype(np.int64)
            g = np.flatnonzero(v > 0)
            g = g[np.argsort(-v[g], kind="stable")]
            units_out[code] = {"t": int(v.sum()), "x": round(sx / sw, 4),
                               "y": round(sy / sw, 4),
                               "g": [int(k) for k in g], "k": [int(v[k]) for k in g]}
        out[str(lvl)] = units_out
        log(f"  level {lvl}: {len(units_out):,} units, "
            f"{sum(u['t'] for u in units_out.values()):,} people")

    colors = palette(groups)
    doc = {
        "label": "Mother tongue",
        "year": 2011,
        "levels": out,
        "groups": [{"key": g["key"], "en": g["en"], "title": g["title"],
                    "color": colors[g["key"]]} for g in groups],
        "source": "Census of India 2011, table C-16, population by mother tongue, at "
                  "sub-district level; districts and states summed from it. Mother tongues "
                  f"with {NAT_MIN:,} speakers, or {SHARE_MIN:.0%} of a sub-district, have "
                  "their own colour; the rest fold into their census language. Towns outside "
                  "any sub-district are placed in the sub-district they stand in.",
    }
    with OUT.open("w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, separators=(",", ":"))
    log(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")

    nat = np.zeros(n_g)
    for u in out["1"].values():
        nat[u["g"]] += u["k"]
    for k in np.argsort(-nat)[:25]:
        print(f"    {groups[k]['en']:<28} {nat[k]:>13,.0f}  {nat[k] / nat.sum():6.2%}")


if __name__ == "__main__":
    main()
