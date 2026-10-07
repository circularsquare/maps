"""Hong Kong: 2021 Population Census, usual spoken language, at two tiers of the same census.

    python sources/hk_census.py --fetch    download into data/raw/hk/, then normalise
    python sources/hk_census.py            normalise what is on disk

Writes data/normalized/hk.csv with two kinds of row:

  geo_level=lsg   1,746 Large Subunit Groups (C&SD's LSUG_21C.csv, the census's own small-area
                  release), FIVE groups: Cantonese, Putonghua, Other Chinese dialects, English,
                  Other languages. geo_id is the LSUG code (`11101L`), the key of the boundary file.
  geo_level=dc    18 District Council districts, FIFTEEN groups, from the census's Interactive
                  Data Dissemination Service (IDDS) query API: the five above with "Other Chinese
                  dialects" opened into Hakka, Chiu Chau, Fukien, Sze Yap, Shanghainese and "Other
                  Chinese dialects", and "Other languages" into Filipino (Tagalog), Indonesian
                  (Bahasa Indonesia), Japanese, Thai and "Others". geo_id is the DC letter (A-T).

The question is "usual spoken language": the language a person uses in daily communication at
home, asked of everyone aged 5 and over; mute persons and children under 5 are outside it.

WHY TWO TIERS. The 15-group table exists only by district: IDDS releases Cantonese, Putonghua,
English and "under 5 or mute" for constituency areas, TPU groups and subunit groups and nothing
else, while the CSV releases give the five groups complete down to subunit groups (a "-" there is
nil; the release's "**", not released, never occurs in the language columns, asserted). So the
small-area table carries where, the district table carries which dialect; countries/hk.py shares
each subunit group's two remainders out by its district's mix (the US entry's arrangement).

CHECKS, none of them a tolerance:
  * every LSUG's five groups plus its under-5-or-mute (t_pop minus the five) is >= 0, and the
    1,746 LSUGs sum to the release's own "Land total" row, column by column;
  * the LSUGs, grouped by the TPU at the head of their code into the 159 Large TPU Groups, equal
    LTPUG_21C.CSV column by column: a second release of the same census agreeing per unit;
  * the 15 groups collapse onto DC_21C.CSV's five groups exactly, for all 18 districts;
  * the 18 districts' five groups equal the LSUG land total exactly.
The LSUG -> district assignment is spatial and is checked in sources/hk_geo.py.
"""
import csv
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "hk"
OUT = ROOT / "data" / "normalized" / "hk.csv"

DOC = "https://www.census2021.gov.hk/doc/"
FILES = ["LSUG_21C.zip", "LTPUG_21C.zip", "DC_21C.zip"]
IDDS = "https://idds.census2021.gov.hk/api/query"
IDDS_RAW = RAW / "idds_dc_lang15.json"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

FIVE = {"ul_can": "Cantonese", "ul_put": "Putonghua", "ul_othchi": "Other Chinese dialects",
        "ul_eng": "English", "ul_oth": "Other languages"}

# IDDS class codes (LANG1, grouping LANG1_5_15G) -> the label written to the normalised file,
# and which of the five groups each one belongs to.
LANG15 = {
    "01": ("Cantonese", "Cantonese"),
    "11": ("Putonghua", "Putonghua"),
    "12": ("Hakka", "Other Chinese dialects"),
    "13": ("Chiu Chau", "Other Chinese dialects"),
    "14": ("Fukien", "Other Chinese dialects"),
    "15": ("Sze Yap", "Other Chinese dialects"),
    "16": ("Shanghainese", "Other Chinese dialects"),
    "19": ("Other Chinese dialects", "Other Chinese dialects"),
    "31": ("English", "English"),
    "42": ("Filipino (Tagalog)", "Other languages"),
    "44": ("Indonesian (Bahasa Indonesia)", "Other languages"),
    "45": ("Japanese", "Other languages"),
    "50": ("Thai", "Other languages"),
    "41;43;46;47;48;49;51;52;53;54;59;60;61;62;63;64;65;66;67;69;92": ("Others", "Other languages"),
    "98;99": (None, None),          # aged under 5, or mute: outside the question
}
# IDDS AREA codes for the 18 districts -> the census's DC letters (DC_21C.CSV `dc`/`dc_class`).
DC_CODE = {"11": "A", "12": "B", "13": "C", "14": "D", "27": "E", "23": "F", "24": "G", "25": "H",
           "26": "J", "31": "S", "32": "K", "33": "L", "34": "M", "35": "N", "36": "P", "37": "R",
           "38": "Q", "39": "T"}


def fetch():
    import requests
    RAW.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        dest = RAW / f
        if dest.exists() and dest.stat().st_size > 10_000:
            print("already have", f)
            continue
        r = requests.get(DOC + f, headers=UA, timeout=300)
        r.raise_for_status()
        dest.with_suffix(".part").write_bytes(r.content)
        os.replace(dest.with_suffix(".part"), dest)
        print(f"got {f}: {len(r.content):,} bytes")
    if not IDDS_RAW.exists():
        url = (IDDS + "?cv.LANG1=" + ",".join(LANG15) + "&cv.AREA=" + ",".join(DC_CODE)
               + "&sv.RP=RP_NPER&period=2021&lang=en")
        r = requests.get(url, headers=UA, timeout=300)
        r.raise_for_status()
        d = r.json()
        if d["header"]["status"]["code"] != 0:
            raise SystemExit(f"IDDS refused the query: {d['header']['status']}")
        d["_url"] = url
        IDDS_RAW.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        print(f"got IDDS district x 15-language table: {len(d['dataSet'])} cells")


def read_release(name):
    """A C&SD small-area CSV: header rows 0-4 (row 4 holds the column codes), data until the
    first row whose code cell is not a unit code."""
    with zipfile.ZipFile(RAW / name) as z:
        member = z.namelist()[0]
        txt = z.read(member).decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(txt)))
    codes = rows[4]
    data = []
    for r in rows[5:]:
        if not r or not r[0].strip() or r[0].startswith(("註", "Notes")):
            break
        data.append(dict(zip(codes, r)))
    return codes, data


def num(v, where):
    v = v.strip()
    if v == "-":
        return 0
    if v == "**":
        raise SystemExit(f"!! {where}: '**' (not released) in a language column")
    return int(float(v))


def main():
    if "--fetch" in sys.argv:
        fetch()

    # ---- the 1,746 Large Subunit Groups --------------------------------------------------
    codes, rows = read_release("LSUG_21C.zip")
    for c in list(FIVE) + ["lsbg", "t_pop"]:
        if c not in codes:
            raise SystemExit(f"!! LSUG_21C has no `{c}` column")
    land = [r for r in rows if r["lsbg"] == "999999"]
    units = [r for r in rows if r["lsbg"] != "999999"]
    if len(land) != 1 or len(units) != 1746:
        raise SystemExit(f"!! expected 1,746 LSUGs and one land total, got {len(units)} and {len(land)}")
    seen = set()
    lsg = []
    for r in units:
        u = r["lsbg"]
        if u in seen:
            raise SystemExit(f"!! LSUG {u} twice")
        seen.add(u)
        five = {k: num(r[k], u) for k in FIVE}
        pop = num(r["t_pop"], u)
        if sum(five.values()) > pop:
            raise SystemExit(f"!! LSUG {u}: five groups {sum(five.values())} exceed population {pop}")
        lsg.append((u, r["lsbg_eng"], pop, five))
    tot = {k: sum(f[k] for *_, f in lsg) for k in FIVE}
    for k in FIVE:
        if tot[k] != num(land[0][k], "land"):
            raise SystemExit(f"!! LSUGs sum to {tot[k]:,} {FIVE[k]}, the land total says {land[0][k]}")
    tpop = sum(p for _, _, p, _ in lsg)
    if tpop != num(land[0]["t_pop"], "land"):
        raise SystemExit("!! LSUG populations do not sum to the land total")
    print(f"LSUG_21C: 1,746 Large Subunit Groups, {tpop:,} people on land; the five groups and "
          "the population each sum to the release's land total exactly")
    print("  " + ", ".join(f"{FIVE[k]} {tot[k]:,}" for k in FIVE)
          + f"; under 5 or mute {tpop - sum(tot.values()):,}")

    # ---- second release of the same census: the 159 Large TPU Groups ---------------------
    lcodes, lrows = read_release("LTPUG_21C.zip")
    key = "ltpug" if "ltpug" in lcodes else lcodes[0]
    groups = {}
    for r in lrows:
        g = r[key]
        name = r[lcodes[1]]
        if name.strip() == "Land total":
            continue
        tpus = set()
        # "112 and 115", "113 - 114", "121 - 124 and 133 - 135", "976/01-02 ..." (sub-TPU groups)
        for part in re.split(r"\s+and\s+|,\s*", name.strip()):
            m = re.fullmatch(r"(\d{3})(?:/[\d\-]+)?\s*(?:-\s*(\d{3}))?", part.strip())
            if not m:
                raise SystemExit(f"!! cannot read LTPU group name {name!r}")
            a = int(m.group(1))
            b = int(m.group(2)) if m.group(2) else a
            tpus.update(range(a, b + 1))
        groups[g] = (name, tpus, {k: num(r[k], g) for k in FIVE})
    tpu_to_group = {}
    split_tpu = set()
    for g, (name, tpus, _) in groups.items():
        for t in tpus:
            if t in tpu_to_group and tpu_to_group[t] != g:
                split_tpu.add(t)
            tpu_to_group[t] = g
    # An LSUG can straddle two LTPU groups (its subunits sit in TPUs of both), and a TPU can be
    # split between LTPU groups by subunit. Groups joined that way are compared as one block.
    parent = {g: g for g in groups}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    def join(gs):
        gs = list(gs)
        for g in gs[1:]:
            parent[find(g)] = find(gs[0])

    for t in split_tpu:
        join([g for g, (_, tpus, _) in groups.items() if t in tpus])
    lsg_group = {}
    for u, name, _, five in lsg:
        tpus = {int(x) for x in re.findall(r"\b(\d{3})/", name)}
        gs = {tpu_to_group.get(t) for t in tpus}
        if not tpus or None in gs:
            raise SystemExit(f"!! LSUG {u} {name!r}: its TPUs are in no LTPU group")
        join(gs)
        lsg_group[u] = next(iter(gs))
    agg, want = {}, {}
    for u, _, _, five in lsg:
        b = find(lsg_group[u])
        a = agg.setdefault(b, {k: 0 for k in FIVE})
        for k in FIVE:
            a[k] += five[k]
    for g, (_, _, five) in groups.items():
        wv = want.setdefault(find(g), {k: 0 for k in FIVE})
        for k in FIVE:
            wv[k] += five[k]
    blocks = {}
    for g in groups:
        blocks.setdefault(find(g), []).append(g)
    bad = [b for b in want if agg.get(b) != want[b]]
    single = sum(1 for b in blocks.values() if len(b) == 1)
    print(f"LTPUG_21C: {len(groups)} Large TPU Groups rebuilt from the LSUGs as {len(blocks)} "
          f"blocks ({single} single groups; the rest joined by an LSUG or a TPU they share)")
    if bad:
        for b in bad[:5]:
            print(f"  !! {blocks[b]}: LSUGs {agg.get(b)} vs LTPUG {want[b]}")
        raise SystemExit(f"!! {len(bad)} blocks disagree with LTPUG_21C")
    print(f"  all {len(blocks)} agree with LTPUG_21C in all five groups, exactly")

    # ---- the 18 districts, 15 groups -------------------------------------------------------
    d = json.loads(IDDS_RAW.read_text(encoding="utf-8"))
    dc15 = {}
    for x in d["dataSet"]:
        cc = {c["cvCode"]: c["ccCode"] for c in x["ccList"]}
        lab, grp = LANG15[cc["LANG1"]]
        if lab is None:
            continue
        dc = DC_CODE[cc["AREA"]]
        if float(x["figure"]) != int(float(x["figure"])):
            raise SystemExit(f"!! IDDS figure {x['figure']} is not a whole number")
        dc15[(dc, lab)] = (grp, int(float(x["figure"])))
    if len(dc15) != 18 * 14:
        raise SystemExit(f"!! expected 18 x 14 district cells, got {len(dc15)}")
    dcodes, drows = read_release("DC_21C.zip")
    dc5 = {r["dc_class"]: r for r in drows if r["dc_class"] in DC_CODE.values()}
    if len(dc5) != 18:
        raise SystemExit("!! DC_21C does not hold the 18 districts")
    for dc, r in dc5.items():
        for k, g in FIVE.items():
            s = sum(n for (dd, _), (gg, n) in dc15.items() if dd == dc and gg == g)
            if s != num(r[k], dc):
                raise SystemExit(f"!! district {dc}: IDDS {g} parts sum to {s:,}, DC_21C says {r[k]}")
    for k, g in FIVE.items():
        s = sum(n for (_, _), (gg, n) in dc15.items() if gg == g)
        if s != tot[k]:
            raise SystemExit(f"!! districts' {g} {s:,} != LSUG land total {tot[k]:,}")
    print("IDDS: 18 districts x 14 language groups; each district's groups collapse onto "
          "DC_21C's five exactly, and the districts sum to the LSUG land total")
    nat = {}
    for (_, lab), (_, n) in dc15.items():
        nat[lab] = nat.get(lab, 0) + n
    print("  " + ", ".join(f"{k} {v:,}" for k, v in sorted(nat.items(), key=lambda kv: -kv[1])))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".part")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["geo_level", "geo_id", "geo_name", "group", "source_category", "count"])
        for u, name, _, five in lsg:
            for k, g in FIVE.items():
                if five[k]:
                    w.writerow(["lsg", u, name, g, g, five[k]])
        for (dc, lab), (grp, n) in sorted(dc15.items()):
            w.writerow(["dc", dc, dc5[dc]["dc_eng"], grp, lab, n])
    os.replace(tmp, OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
