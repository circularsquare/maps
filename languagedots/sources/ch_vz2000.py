"""Switzerland: Volkszählung 2000, main language by commune (BFS), on religiondots' commune hexes.

Reads (or fetches) data/raw/ch/ and writes data/normalized/ch.csv. The record is sources/ch.md.

## The table

`px-x-4003000000_123`, *Wohnbevölkerung 2000 nach Wohnsitztyp, Kanton / Bezirk / Gemeinde,
Staatsangehörigkeit (Kategorie) und Hauptsprache*: 3,107 geographic rows (country, 26 cantons,
184 districts, 2,896 communes) x 3 nationality cells x 37 language cells (36 + total). The whole
.px file is one GET from BFS PxWeb, no key (the json-stat API refuses this cube; see fetch()). Found through ckan.opendata.swiss, which mirrors BFS's
catalogue (religiondots' route: BFS rate-limits a crawl of its own PxWeb).

The question was "Welches ist die Sprache, in der Sie denken und die Sie am besten
beherrschen?", one answer: the main language. It is the last Swiss count that asked everyone.
Since 2010 the Strukturerhebung asks a sample, allows several main languages, and publishes six
categories (German, French, Italian, Romansh, English, other) at district level at best; see
sources/ch.md for why the census is drawn instead.

Civil-law residence (`Zivilrechtlicher Wohnsitz`) is used, as religiondots does for the same
census: the published national total refers to it.

## The commune vintage

The census counts 2,896 communes; the placement units (sources/ch_geo.py: religiondots' 2,197
commune polygons, GISCO LAU's 1 January 2020 state, plus Verzasca) number 2,198. BFS's own
correspondence API maps each 2000 commune to its successor (religiondots/sources/ch.py found
the three traps: start on the census date, exclude territory exchanges, and try both the 2020
and 2021 states because GISCO's "2021" is really 1.1.2020). Two more found here: Verzasca
(merged October 2020) was missing from religiondots' layer, and four Muggio-valley communes
have two successors each (SPLIT below). Targets are resolved against the unit list itself, so
every row this writes is drawable by construction.

## Checks

- the 36 categories partition the national total, and Swiss + foreign = total in every commune;
- every 2000 commune lands on a unit, and every category's commune sum equals its national
  figure after the merge;
- per commune, the total equals the same census's religion table (px_122) as religiondots
  fetched it: a second table of the same census, agreeing to the person.

Usage:
    python sources/ch_vz2000.py --fetch     three GETs, ~2 MB
    python sources/ch_vz2000.py             normalise from data/raw/ch/
"""
import csv
import json
import os
import re
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ch")
OUT = os.path.join(ROOT, "data", "normalized", "ch.csv")
RD = os.path.join(os.path.dirname(ROOT), "religiondots")
UNITS = os.path.join(ROOT, "data", "geo", "ch", "ch_units.gpkg")      # sources/ch_geo.py
RD_RELIGION = os.path.join(RD, "data", "raw", "ch", "px-x-4003000000_122.json")

SOURCE_ID = "ch_volkszaehlung_2000_px123"
YEAR = 2000

PX_DOWNLOAD = "https://www.pxweb.bfs.admin.ch/DownloadFile.aspx?file=px-x-4003000000_123"
PXFILE = os.path.join(RAW, "px-x-4003000000_123.px")
WOHNSITZ = "Zivilrechtlicher Wohnsitz"
AGV = "https://www.agvchapp.bfs.admin.ch/api/communes/"
CENSUS_DATE = "06-12-2000"
CORR_DATES = ("01-01-2020", "01-01-2021")
CORR = {d: os.path.join(RAW, f"correspondances_{d}.csv") for d in CORR_DATES}

GEO_DIM = "Kanton (-) / Bezirk (>>) / Gemeinde (......)"
NAT_DIM = "Staatsangehörigkeit (Kategorie)"
LANG_DIM = "Hauptsprache"
TOTAL = "Hauptsprachen - Total"
NAT_TOTAL, SWISS, FOREIGN = "Staatsangehörigkeit - Total", "Schweizer", "Ausländer"
EXPECTED_COMMUNES_2000 = 2_896
EXPECTED_UNITS = 2_198
COMMUNE_RE = re.compile(r"^\.{6}(\d{4})\s+(.*)$")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count", "tier", "year",
           "source_id"]

# SWISS GERMAN AND STANDARD GERMAN (Anita, 2026-10-06: draw Swiss German as its own language).
# The main-language question has one "Deutsch". The same census's family-language question
# ("Welche Sprache(n) sprechen Sie zu Hause, mit den Angehörigen?") offered Schweizerdeutsch and
# Hochdeutsch as separate boxes, but BFS publishes it by language region x nationality only:
# Lüdi & Werlen, *Sprachenlandschaft in der Schweiz*, BFS 2005 (data/raw/ch/luedi_werlen_2005.pdf),
# Tabelle 18 (German region, % of residents) and Tabellen 20-22 (French, Italian, Romansh
# regions, absolute). Per region and nationality: (Standard German only, Swiss German only,
# both). The share drawn as Standard German is (only + both/2) / (all three); "both" is split
# evenly because nothing says which of the two is the person's own.
# Applied to each 2000 commune's German main-language speakers, Swiss and foreign apart (px_123
# carries the nationality split), so the commune's foreign share does the placing that a
# per-canton rate would only do coarsely. The rate is of German-at-home speakers, applied to
# German main-language speakers: that substitution is the model.
HOME_GERMAN = {   # region: {nationality: (Hochdeutsch only, Schweizerdeutsch only, both)}
    "Deutsch":       {SWISS: (1.3, 90.8, 5.4),        FOREIGN: (13.8, 29.1, 6.8)},    # Tab. 18, %
    "Französisch":   {SWISS: (31_467, 82_837, 19_438), FOREIGN: (13_961, 2_527, 1_699)},  # Tab. 20
    "Italienisch":   {SWISS: (6_722, 24_493, 4_054),  FOREIGN: (3_167, 1_143, 307)},      # Tab. 21
    "Rätoromanisch": {SWISS: (272, 9_188, 545),       FOREIGN: (334, 219, 74)},           # Tab. 22
}
HD_SHARE = {(reg, nat): (hd + both / 2) / (hd + sg + both)
            for reg, d in HOME_GERMAN.items() for nat, (hd, sg, both) in d.items()}
DE = "Deutsch"
SG_CAT = "Deutsch: Schweizerdeutsch (modelliert)"
HD_CAT = "Deutsch: Hochdeutsch (modelliert)"
# BFS's language regions for 2000 (Lüdi & Werlen 2005; the BFS 2003 report): 1,669 German,
# 892 French, 269 Italian, 66 Romansh communes, by majority main language among the four.
REGION_COUNTS = {"Deutsch": 1_669, "Französisch": 892, "Italienisch": 269, "Rätoromanisch": 66}


# Bosco/Gurin, the Walser village in Ticino, is Italian-majority in 2000 (40 Italian, 31
# German), but its German speakers are Walser (Gurinerdeutsch, Alemannic), not Standard German
# speakers in Italian Switzerland: German-region rates.
REGION_OVERRIDE = {"5304": "Deutsch"}


def _region(counts, bfs=None):
    """A 2000 commune's language region: the national language most of its people named."""
    if bfs in REGION_OVERRIDE:
        return REGION_OVERRIDE[bfs]
    return max(REGION_COUNTS, key=lambda c: counts[(NAT_TOTAL, c)])


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(PXFILE) and os.path.getsize(PXFILE) > 1_000_000:
        print("  already have", os.path.basename(PXFILE))
    else:
        # THE API REFUSES THIS CUBE WITH A 403. A json-stat POST for every commune x
        # nationality x language is 345,000 cells, over PxWeb's cell limit (religiondots'
        # religion cube, 59,000 cells, passes). The whole .px file is one GET instead.
        print("  GET", PX_DOWNLOAD)
        r = requests.get(PX_DOWNLOAD, timeout=900, headers=UA)
        r.raise_for_status()
        if not r.content.startswith(b"CHARSET") and b"MATRIX=" not in r.content[:2000]:
            raise SystemExit("the download is not a .px file")
        tmp = PXFILE + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(r.content)
        os.replace(tmp, PXFILE)
        print(f"       {os.path.getsize(PXFILE):,} bytes")
    for date, dest in CORR.items():
        if os.path.exists(dest) and os.path.getsize(dest) > 1_000:
            print("  already have", os.path.basename(dest))
            continue
        url = (AGV + "correspondances?includeUnmodified=true&includeTerritoryExchange=false"
               f"&startPeriod={CENSUS_DATE}&endPeriod={date}")
        print("  GET commune correspondence", CENSUS_DATE, "->", date)
        r = requests.get(url, timeout=300, headers=UA)
        r.raise_for_status()
        with open(dest, "wb") as fh:
            fh.write(r.content)


def _px_strings(s):
    return re.findall(r'"((?:[^"]|"")*)"', s)


def _cube():
    """The .px file as a json-stat2-shaped dict, civil-law residence only.

    PX: `KEY="..";` header lines (German, then `KEY[fr]=` French copies), then `DATA=` with
    the cells row-major over STUB then HEADING. CHARSET="ANSI" is cp1252.
    """
    if not os.path.exists(PXFILE):
        raise SystemExit(f"missing {PXFILE}: run with --fetch")
    with open(PXFILE, encoding="cp1252") as fh:
        text = fh.read()
    head, data = text.split("\nDATA=", 1)
    keys = {}
    for m in re.finditer(r'(?ms)^([A-Z][A-Z-]*(?:\("[^"]*"\))?)=(.*?);\s*$', head):
        keys.setdefault(m.group(1), m.group(2))
    stub = _px_strings(keys["STUB"])
    heading = _px_strings(keys["HEADING"])
    ids = stub + heading
    vals = {d: _px_strings(keys[f'VALUES("{d}")']) for d in ids}
    size = [len(vals[d]) for d in ids]
    toks = data.replace(";", " ").split()
    n = 1
    for s in size:
        n *= s
    if len(toks) != n:
        raise SystemExit(f".px has {len(toks):,} cells, expected {n:,}")

    def num(t):
        t = t.strip('"')
        return None if t in (".", "..", "...", "-", "") else int(float(t))

    if ids[0] != "Wohnsitztyp" or vals["Wohnsitztyp"][0] != WOHNSITZ:
        raise SystemExit(f"unexpected first dimension {ids[0]} {vals.get(ids[0])}")
    block = n // size[0]                       # civil-law residence is the first block
    value = [num(t) for t in toks[:block]]
    vals["Wohnsitztyp"] = vals["Wohnsitztyp"][:1]
    size[0] = 1
    return {"id": ids, "size": size, "value": value,
            "dimension": {d: {"category": {"index": {str(i): i for i in range(len(vals[d]))},
                                           "label": {str(i): v for i, v in enumerate(vals[d])}}}
                          for d in ids}}


def _rows(doc, lang_dim=LANG_DIM, nat_dim=NAT_DIM):
    """[(level, bfs, name, canton, {(nat, lang): count})] in the cube's hierarchical order."""
    ids, size = doc["id"], doc["size"]
    labels = {d: [doc["dimension"][d]["category"]["label"][k]
                  for k in sorted(doc["dimension"][d]["category"]["index"],
                                  key=doc["dimension"][d]["category"]["index"].get)]
              for d in ids}
    val = doc["value"]
    strides = [1] * len(ids)
    for i in range(len(ids) - 2, -1, -1):
        strides[i] = strides[i + 1] * size[i + 1]

    def cell(pos):
        k = sum(p * s for p, s in zip(pos, strides))
        v = val[k] if isinstance(val, list) else val.get(str(k))
        return 0 if v is None else int(v)

    gi = ids.index(GEO_DIM)
    others = [d for d in ids if d != GEO_DIM and size[ids.index(d)] > 1]
    out, canton = [], None
    for g, text in enumerate(labels[GEO_DIM]):
        counts = {}
        # walk every combination of the non-singleton dimensions
        combos = [{}]
        for d in others:
            combos = [dict(c, **{d: j}) for c in combos for j in range(size[ids.index(d)])]
        for c in combos:
            pos = [0] * len(ids)
            pos[gi] = g
            for d, j in c.items():
                pos[ids.index(d)] = j
            key = tuple(labels[d][c[d]] for d in others)
            counts[key] = cell(pos)
        m = COMMUNE_RE.match(text)
        if m:
            out.append(("commune", m.group(1), m.group(2).strip(), canton, counts))
        elif text.startswith(">>"):
            continue
        elif text.startswith("- "):
            canton = text[2:].strip()
            out.append(("canton", None, canton, canton, counts))
        else:
            out.append(("country", None, text.strip(), None, counts))
    return out, others, labels


def _units():
    import geopandas as gpd

    if not os.path.exists(UNITS):
        raise SystemExit(f"missing {UNITS}: run sources/ch_geo.py first")
    g = gpd.read_file(UNITS, columns=["unit"], ignore_geometry=True)
    units = set(g["unit"].astype(str))
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{len(units)} units in {UNITS}, expected {EXPECTED_UNITS}")
    return units


# A 2000 commune the API maps to MORE THAN ONE successor even with territory exchanges
# excluded, all four in the Muggio valley. History: on 4 April 2004 Castel San Pietro absorbed
# Casima and Monte, and Campora, a part of Caneggio; on 25 October 2009 Caneggio and five others
# became Breggia. The API chains these and also links each of the four to the other successor.
# Each goes to the commune that holds its people today. Taking the last row, as a plain dict
# does, sent Castel San Pietro itself to Breggia and left its polygon empty (religiondots'
# ch.py does exactly that; noted in sources/ch.md).
SPLIT = {5249: "5249",   # Castel San Pietro: itself
         5248: "5249",   # Casima: merged into Castel San Pietro, 2004
         5256: "5249",   # Monte: merged into Castel San Pietro, 2004
         5246: "5269"}   # Caneggio: merged into Breggia, 2009 (only Campora went to Castel)


def _correspondence(units):
    import pandas as pd

    maps = {}
    for date, path in CORR.items():
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}: run with --fetch")
        df = pd.read_csv(path)
        multi = df.groupby("InitialCode")["TerminalCode"].nunique()
        multi = set(multi[multi > 1].index.astype(int))
        if multi - set(SPLIT):
            raise SystemExit(f"{date}: communes with several successors and no ruling in "
                             f"SPLIT: {sorted(multi - set(SPLIT))}")
        m = dict(zip(df["InitialCode"].astype(int), df["TerminalCode"].astype(int)))
        for code, t in SPLIT.items():
            if code in m:
                m[code] = int(t)
        maps[date] = m
    resolved, by = {}, {d: 0 for d in CORR_DATES}
    for code in set().union(*(m.keys() for m in maps.values())):
        for date in CORR_DATES:
            t = maps[date].get(code)
            if t is not None and f"{int(t):04d}" in units:
                resolved[code] = f"{int(t):04d}"
                by[date] += 1
                break
    return resolved, by


def main():
    if "--fetch" in sys.argv:
        fetch()
    doc = _cube()
    rows, others, labels = _rows(doc)
    if others != [NAT_DIM, LANG_DIM]:
        raise SystemExit(f"unexpected dimension order {others}")
    langs = labels[LANG_DIM]
    cats = [c for c in langs if c != TOTAL]
    ok = True

    def say(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    communes = [r for r in rows if r[0] == "commune"]
    country = [r for r in rows if r[0] == "country"][0][4]
    cantons = [r for r in rows if r[0] == "canton"]
    national = country[(NAT_TOTAL, TOTAL)]
    print(f"  {len(cats)} language categories; national total {national:,}")
    say(len(communes) == EXPECTED_COMMUNES_2000, f"{len(communes):,} communes of 2000")
    say(len(cantons) == 26, f"{len(cantons)} cantons")
    say(sum(country[(NAT_TOTAL, c)] for c in cats) == national,
        "the 36 categories partition the national total")
    bad = [r[1] for r in communes
           if any(r[4][(SWISS, c)] + r[4][(FOREIGN, c)] != r[4][(NAT_TOTAL, c)] for c in langs)
           or sum(r[4][(NAT_TOTAL, c)] for c in cats) != r[4][(NAT_TOTAL, TOTAL)]]
    say(not bad, f"Swiss + foreign = total and categories sum to the total in every commune "
                 f"({len(bad)} failing)")
    say(sum(r[4][(NAT_TOTAL, TOTAL)] for r in communes) == national,
        "communes sum to the national total")

    # a second table of the same census: religion by commune, as religiondots fetched it
    if os.path.exists(RD_RELIGION):
        with open(RD_RELIGION, encoding="utf-8") as fh:
            rel = json.load(fh)
        rrows, rothers, rlab = _rows(rel, lang_dim="Religion", nat_dim=None)
        rtot = {r[1]: r[4][("Religionen - Total",)] for r in rrows if r[0] == "commune"}
        diff = [(r[1], r[2], r[4][(NAT_TOTAL, TOTAL)], rtot.get(r[1])) for r in communes
                if rtot.get(r[1]) != r[4][(NAT_TOTAL, TOTAL)]]
        say(not diff, f"every commune's total equals the census religion table's (px_122), "
                      f"{len(rtot):,} communes compared; {len(diff)} differ")
        for d in diff[:5]:
            print("        ", d)
    else:
        print("  (religiondots' px_122 cube not on disk; second-table check skipped)")

    units = _units()
    resolved, by = _correspondence(units)
    merged, unmatched, regions = {}, [], {}
    for _, bfs, name, canton, counts in communes:
        t = resolved.get(int(bfs))
        if t is None:
            unmatched.append((bfs, name, counts[(NAT_TOTAL, TOTAL)]))
            continue
        slot = merged.setdefault(t, {"names": [], "c": {c: 0 for c in cats}, "hd": 0.0})
        slot["names"].append(name)
        for c in cats:
            slot["c"][c] += counts[(NAT_TOTAL, c)]
        reg = _region(counts, bfs)
        regions[reg] = regions.get(reg, 0) + 1
        slot["hd"] += sum(counts[(nat, DE)] * HD_SHARE[(reg, nat)] for nat in (SWISS, FOREIGN))
    print(f"  commune join: {by} resolved; {len(merged):,} units receive people")
    say(not unmatched, f"every 2000 commune lands on a placement unit ({len(unmatched)} not)")
    for u in unmatched[:10]:
        print("        ", u)
    empty = sorted(units - set(merged))
    print(f"  {len(empty)} placement units receive nobody: {empty[:10]}")
    off = [(c, sum(m["c"][c] for m in merged.values()), country[(NAT_TOTAL, c)]) for c in cats]
    off = [x for x in off if x[1] != x[2]]
    say(not off, "every category's commune sum equals its national figure after the merge")
    for c, s, n in off:
        print(f"         {c}: communes {s:,} vs national {n:,}")

    # BFS's assignment differs in 3 communes (2 Romansh and 1 Italian commune by BFS come out
    # German here, 2026-10-06); the regions' rates for Swiss nationals differ by about a point,
    # so this is a sanity check on the rule, not an exact one.
    gap = sum(abs(regions.get(r, 0) - n) for r, n in REGION_COUNTS.items())
    say(gap <= 10, f"language regions by majority main language {regions} near BFS's "
                   f"{REGION_COUNTS} (off by {gap})")

    if not ok:
        raise SystemExit("reconciliation FAILED")

    out, hd_total, de_total = [], 0, 0
    for t, slot in sorted(merged.items()):
        for c in cats:
            if not slot["c"][c]:
                continue
            row = {"geo_id": t, "geo_level": "commune", "geo_name": sorted(slot["names"])[0],
                   "year": YEAR, "source_id": SOURCE_ID}
            if c != DE:
                out.append(dict(row, source_category=c, count=slot["c"][c], tier="measured"))
                continue
            # German split into Swiss German and Standard German (HOME_GERMAN above)
            hd = min(slot["c"][c], round(slot["hd"]))
            hd_total += hd
            de_total += slot["c"][c]
            for cat, n in ((SG_CAT, slot["c"][c] - hd), (HD_CAT, hd)):
                if n:
                    out.append(dict(row, source_category=cat, count=n, tier="modelled"))
    print(f"  German main language {de_total:,}: drawn as Standard German {hd_total:,} "
          f"({100 * hd_total / de_total:.1f}%), Swiss German {de_total - hd_total:,}")
    for k, v in sorted(HD_SHARE.items()):
        print(f"      Standard German share, {k[0]} region, {k[1]}: {100 * v:.1f}%")
    for level, _, name, _, counts in [r for r in rows if r[0] != "commune"]:
        for c in cats:
            out.append({"geo_id": name, "geo_level": level, "geo_name": name,
                        "source_category": c, "count": counts[(NAT_TOTAL, c)],
                        "tier": "measured", "year": YEAR, "source_id": SOURCE_ID})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(out)
    os.replace(tmp, OUT)
    print(f"\nwrote {OUT} ({len(out):,} rows)")
    print("\n  national, by category:")
    for c in sorted(cats, key=lambda c: -country[(NAT_TOTAL, c)]):
        print(f"    {c:<40} {country[(NAT_TOTAL, c)]:>10,} {100 * country[(NAT_TOTAL, c)] / national:6.2f}%"
              f"   foreign {country[(FOREIGN, c)]:>8,}")


if __name__ == "__main__":
    main()
