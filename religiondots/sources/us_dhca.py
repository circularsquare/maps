"""2020 Census Detailed DHC-A, table T01001: the Sikh and Yazidi race write-ins, by county and tract.

WHY THIS EXISTS. The United States asks no religion question, and spec §3.5a draws the country from
Pew's survey totals over ASARB's congregation rolls. Neither counts Sikhs: ASARB lists 307 gurdwaras
with no adherents, and Pew folds Sikhs into one `other-world-religions` line with Bahá'ís, Daoists,
Jains, Zoroastrians and Shinto. The 2020 census does count them, by accident of coding: a person who
writes "Sikh" in the race question is tabulated as the detailed Asian group Sikh (codes 4305-4309),
and "Yazidi" as a detailed Middle Eastern and North African group. The Detailed DHC-A prints both
for the nation, states, counties, places and tracts.

WHAT THE COUNT IS. People who wrote the word, alone or with any other race or group. It is not a
religion count: a Sikh who ticked "Asian Indian" and wrote nothing is not in it, and the Sikh
Coalition's own reading of the release says so (*Updated Census Figures Severely Undercount U.S.
Sikhs*, 2023, which quotes both national figures pinned below). So it is drawn as counted and
nothing is scaled up; the rest of America's Sikhs stay inside `other.us` (sources/us_dhca.md §3).

HOW THE FILE PUBLISHES. A detailed group is printed for a county, place or tract only where its
noise-infused count is 22 or more (summary file technical document, Table 1), and noise is added to
each geography separately, so tracts do not sum to their county and counties do not sum to the
nation. A withheld cell reads `-888888888` with `ANN` = `X`. `classify()` accepts exactly those two
shapes and raises on anything else.

ITERATION CODES ARE CHECKED BY NAME. T01001 carries only `ITERID`; the label comes from the
iterations list workbook. Every code used here is looked up there and must carry the expected label,
so a renumbered release fails instead of drawing Samoans as Sikhs.

Run: python sources/us_dhca.py --fetch    downloads anything missing, then reads and checks
     python sources/us_dhca.py            reads the cached files and checks
  -> data/normalized/us_dhca.csv: level, geoid, iterid, label, node, count, suppressed
"""
import argparse
import csv
import io
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "sources"))
from fetch_checks import check_body  # noqa: E402

RAW = HERE / "data" / "raw" / "us_dhca"
ZIP = RAW / "2020-ddhc-a.zip"
ITER = RAW / "iterations.xlsx"
OUT = HERE / "data" / "normalized" / "us_dhca.csv"
TABLE = "ddhca_t01001.csv"

URLS = {
    ZIP: "https://www2.census.gov/programs-surveys/decennial/2020/data/detailed-dhc-a/"
         "2020-ddhc-a.zip",
    ITER: "https://www2.census.gov/programs-surveys/decennial/2020/technical-documentation/"
          "complete-tech-docs/detailed-demographic-and-housing-characteristics-file-a/"
          "2020-census-hispanic-origin-and-race-iterations-list.xlsx",
}

# ITERID -> (node drawn, or None for a witness row; the iterations list's label; the national count)
# The Sikh national figures are the Sikh Coalition's quotation of the release, an independent
# reading. The Yazidi ones have no outside witness and are pinned so a re-issue is noticed.
GROUPS = {
    "3845": ("sikhism", "Sikh alone or in any combination", 70697),
    "3788": (None, "Sikh alone", 48321),
    "1207": ("yazidism", "Yazidi alone or in any combination", 630),
    "1096": (None, "Yazidi alone", 444),
}
# A national witness only, read for scale in sources/us_dhca.md §3 (Pew's 8% of Indian American
# adults who are Sikh, applied to it). Its code is looked up by label; the count is the scout's.
ASIAN_INDIAN = ("Asian Indian alone or in any combination", 4_768_846)

# Content-Length on 2026-09-15 (Last-Modified 2024-01-10 for the zip, 2023-09-20 for the list).
PIN_SIZE = {ZIP: 41_121_420, ITER: 113_686}
LEVELS = ("USA", "STATE", "COUNTY", "TRACT")
SUPPRESSED = "-888888888"
THRESHOLD = 22          # smallest count printed below the state (technical document, Table 1)
NOISE = 11              # the margin on a total-count-only cell (technical document, Table 3)

FAILED = []


def say(ok, msg):
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok:
        FAILED.append(msg)


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for path, url in URLS.items():
        if path.exists():
            print(f"  cached {path.name} ({path.stat().st_size:,} bytes)")
            continue
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as r:
            body = r.read()
        print("  " + str(check_body(body, "zip", where=path.name, pin_size=PIN_SIZE.get(path))))
        part = path.with_suffix(path.suffix + ".part")
        part.write_bytes(body)
        os.replace(part, path)


def labels():
    """ITERID -> label, from the iterations list. Code in one cell, label in the next."""
    it = pd.read_excel(ITER, header=None, dtype=str)
    out = {}
    for row in it.itertuples(index=False):
        vals = [v.strip() for v in row if isinstance(v, str) and v.strip()]
        if len(vals) >= 2 and vals[0].isdigit():
            out.setdefault(vals[0], vals[1])
    return out


def classify(count, ann):
    """(count, suppressed) for one cell; raises on any shape the technical document does not name."""
    s = (count or "").strip()
    if s == SUPPRESSED and (ann or "").strip() == "X":
        return None, True
    if s.isdigit():
        return int(s), False
    raise ValueError(f"unrecognised cell COUNT={count!r} ANN={ann!r}")


def read():
    rows = []
    with zipfile.ZipFile(ZIP) as z, z.open(TABLE) as f:
        for r in csv.DictReader(io.TextIOWrapper(f, encoding="latin-1")):
            if r["ITERID"] not in GROUPS or r["REGION_TYPE"] not in LEVELS:
                continue
            if GROUPS[r["ITERID"]][1] == ASIAN_INDIAN[0] and r["REGION_TYPE"] != "USA":
                continue
            n, sup = classify(r["COUNT"], r["ANN"])
            node, label, _ = GROUPS[r["ITERID"]]
            rows.append(dict(level=r["REGION_TYPE"], geoid=r["REGION_ID"].strip(),
                             iterid=r["ITERID"], label=label, node=node or "",
                             count=n, suppressed=sup))
    return pd.DataFrame(rows)


def check(d):
    print("checks")
    lab = labels()
    for code, (_, want, _) in GROUPS.items():
        say(lab.get(code) == want, f"ITERID {code} is {want!r} in the iterations list "
                                   f"(list says {lab.get(code)!r})")

    usa = d[d["level"] == "USA"].set_index("iterid")["count"]
    for code, (_, label, n) in GROUPS.items():
        say(usa.get(code) == n, f"{label}: national {usa.get(code)} = {n:,}")

    pub = d[~d["suppressed"]]
    for code, (node, label, _) in GROUPS.items():
        g = pub[pub["iterid"] == code]
        by = {lv: g[g["level"] == lv] for lv in LEVELS}
        print(f"  {label}: " + ", ".join(
            f"{lv.lower()} {len(by[lv])} rows {by[lv]['count'].sum():,.0f}" for lv in LEVELS[1:]))
        # published sub-state cells are at the threshold or above, except the District of
        # Columbia's county row, which is the state row again (states have no threshold)
        small = g[g["level"].isin(["COUNTY", "TRACT"]) & (g["count"] < THRESHOLD)]
        dc = small[(small["level"] == "COUNTY") & (small["geoid"] == "11001")]
        dc_state = by["STATE"].loc[by["STATE"]["geoid"] == "11", "count"]
        if len(dc):
            say(len(dc_state) == 1 and dc["count"].iloc[0] == dc_state.iloc[0],
                f"{label}: DC's county cell {dc['count'].iloc[0]:.0f} under {THRESHOLD} equals its "
                f"state cell {dc_state.tolist()}")
        small = small.drop(dc.index)
        say(small.empty, f"{label}: no other county or tract cell below {THRESHOLD} "
                         f"({small[['level', 'geoid', 'count']].values.tolist()})")

    # geography shapes: states two digits and no Puerto Rico, counties five, tracts eleven
    for lv, width in (("STATE", 2), ("COUNTY", 5), ("TRACT", 11)):
        g = d[d["level"] == lv]
        bad = g[~g["geoid"].str.fullmatch(rf"\d{{{width}}}")]
        say(bad.empty, f"{lv.lower()} ids are {width} digits ({len(bad)} are not)")
    pr = d[d["geoid"].str.startswith("72") & d["level"].isin(["STATE", "COUNTY", "TRACT"])]
    say(pr.empty, f"no Puerto Rico rows ({len(pr)}); pr is its own entry on the map")

    # joins to the 2020 boundary files the US build draws on
    import geopandas as gpd
    cty = set(gpd.read_file(HERE / "data" / "geo" / "counties2020" / "cb_2020_us_county_500k.shp",
                            ignore_geometry=True)["GEOID"])
    tra = set(gpd.read_file(HERE / "data" / "geo" / "tracts2020" / "cb_2020_us_tract_500k.shp",
                            ignore_geometry=True)["GEOID"])
    for lv, ids, name in (("COUNTY", cty, "counties"), ("TRACT", tra, "tracts")):
        g = pub[(pub["level"] == lv) & (pub["node"] != "")]
        miss = sorted(set(g["geoid"]) - ids)
        say(not miss, f"every drawn {lv.lower()} joins cb_2020_us_{name[:-1]}_500k "
                      f"({len(miss)} do not: {miss[:5]})")

    # alone against alone-or-in-combination, where both are printed: noise is independent per
    # cell, so a small inversion is expected and only one past both margins is a fault
    for aoic, alone in (("3845", "3788"), ("1207", "1096")):
        a = pub[pub["iterid"] == aoic].set_index(["level", "geoid"])["count"]
        b = pub[pub["iterid"] == alone].set_index(["level", "geoid"])["count"]
        j = pd.concat([a.rename("aoic"), b.rename("alone")], axis=1).dropna()
        inv = j[j["alone"] > j["aoic"]]
        far = j[j["alone"] > j["aoic"] + 2 * NOISE]
        say(far.empty, f"{GROUPS[aoic][1]}: alone never exceeds it by more than two noise "
                       f"margins over {len(j)} cells ({len(inv)} small inversions, {len(far)} large)")

    # tracts against their counties
    for code in ("3845", "1207"):
        g = pub[pub["iterid"] == code]
        c = g[g["level"] == "COUNTY"].set_index("geoid")["count"]
        t = g[g["level"] == "TRACT"].copy()
        t["county"] = t["geoid"].str[:5]
        orphan = t[~t["county"].isin(c.index)]
        tc = t[t["county"].isin(c.index)].groupby("county")["count"].agg(["sum", "size"])
        over = tc[tc["sum"] > c.reindex(tc.index) + NOISE * tc["size"]]
        print(f"  {GROUPS[code][1]}: {len(t)} tracts hold {t['count'].sum():,.0f} of "
              f"{c.sum():,.0f} in published counties; {len(orphan)} tracts "
              f"({orphan['count'].sum():,.0f} people) sit in a county with no published count "
              f"and are not drawn")
        say(over.empty, f"{GROUPS[code][1]}: no county's tracts exceed it by more than "
                        f"{NOISE} per tract ({list(over.index[:5])})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="download anything not cached")
    args = ap.parse_args()
    if args.fetch:
        fetch()
    for p in (ZIP, ITER):
        if not p.exists():
            raise SystemExit(f"{p} missing; run with --fetch")
        if p.stat().st_size != PIN_SIZE[p]:
            raise SystemExit(f"{p.name} is {p.stat().st_size:,} bytes, not the pinned "
                             f"{PIN_SIZE[p]:,}: a truncated copy or a re-issue")
    ai = [code for code, lab in labels().items() if lab == ASIAN_INDIAN[0]]
    if len(ai) != 1:
        raise SystemExit(f"{ASIAN_INDIAN[0]!r} names {len(ai)} codes in the iterations list: {ai}")
    GROUPS[ai[0]] = (None, ASIAN_INDIAN[0], ASIAN_INDIAN[1])
    d = read()
    check(d)
    if FAILED:
        raise SystemExit(f"{len(FAILED)} check(s) failed; nothing written")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    part = OUT.with_suffix(".csv.part")
    d.to_csv(part, index=False)
    os.replace(part, OUT)
    print(f"wrote {OUT.relative_to(HERE)}: {len(d):,} rows")


if __name__ == "__main__":
    main()
