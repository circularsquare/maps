"""Mozambique, IV RGPH 2017, mother tongue by province -> data/normalized/mz.csv.

    python sources/mz_rgph.py [--fetch]

INE's Quadro 22 (mother tongue, aged 5+) for each of the eleven provinces and the nation, by
urban and rural residence, from the Wayback Machine's 2019-11-14 captures of INE's retired Plone
site; Quadro 23 (language spoken most at home) as a witness. Writes one row per province x area
(urban, rural; Maputo Cidade urban only) x printed category, canonical spelling.

Inhambane's and Maputo Cidade's plain captures print the numbers against labels shifted one row;
their `-1` re-uploads are read (Q22_FILE). The national table is a different edit in three cells
(NATIONAL_DIFF). sources/mz.md is the record, with every check's numbers.
"""
import csv
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "mz"
OUT = ROOT / "data" / "normalized" / "mz.csv"

WAYBACK = "http://web.archive.org/web/{ts}id_/http://www.ine.gov.mz/iv-rgph-2017/{path}"
Q22 = "quadro-22-populacao-de-5-anos-e-mais-por-idade-segundo-area-de-residencia-sexo-e-lingua-materna"
Q23 = ("quadro-23-populacao-de-5-anos-e-mais-por-idade-segundo-area-de-residencia-sexo-e-lingua-"
       "que-fala-com-mais-frequencia")

# (local name, capture timestamp, path under /iv-rgph-2017/). From the Wayback CDX query in
# sources/mz.md §1. Quadro 22 is mother tongue, Quadro 23 the language spoken most often at home;
# both cover people aged 5 and over. Zambézia's Quadro 23 was never captured.
FILES = [
    ("q22_00", "20191114165121", f"mocambique/08-lingua/{Q22}-mocambique-2017.xlsx"),
    ("q22_01", "20191114002853", f"niassa/{Q22}-provincia-do-niassa-2017.xlsx"),
    ("q22_02", "20191114005018", f"cabo-delgado/{Q22}-provincia-de-cabo-delgado-2017.xlsx"),
    ("q22_03", "20191114013404", f"nampula/{Q22}-provincia-de-nampula-2017.xlsx"),
    ("q22_04", "20191114013816", f"zambezia/{Q22}-provincia-de-zambezia-2017.xlsx"),
    ("q22_05", "20191114015921", f"tete/{Q22}-tete-2017.xlsx"),
    ("q22_06", "20191114024417", f"manica/{Q22}-provincia-de-manica-2017.xlsx"),
    ("q22_07", "20191114030145", f"sofala/{Q22}.xlsx"),
    ("q22_08", "20191114032438", f"inhambane/{Q22}-provincia-de-inhambane-2017.xlsx"),
    ("q22_08b", "20191114034451", f"inhambane/{Q22}-provincia-de-inhambane-2017-1.xlsx"),
    ("q22_09", "20191114042343", f"gaza/{Q22}-provincia-gaza-2017.xlsx"),
    ("q22_10", "20191114045042", f"maputo-provincia/{Q22}-maputo-provincia-2017.xlsx"),
    ("q22_11", "20191114050625",
     "maputo-cidade/quadro-22-populacao-de-5-anos-e-mais-por-idade-segundo-sexo-e-lingua-materna-"
     "maputo-cidade-2017.xlsx"),
    ("q22_11b", "20191114051350",
     "maputo-cidade/quadro-22-populacao-de-5-anos-e-mais-por-idade-segundo-sexo-e-lingua-materna-"
     "maputo-cidade-2017-1.xlsx"),
    ("q23_00", "20191114165109", f"mocambique/08-lingua/{Q23}-em-casa-mocambique-2017.xlsx"),
    ("q23_01", "20191114002906", f"niassa/{Q23}-em-casa-provincia-do-niassa-2017.xlsx"),
    ("q23_02", "20191114005031", f"cabo-delgado/{Q23}-em-casa-provincia-de-cabo-delgado-2017.xlsx"),
    ("q23_03", "20191114011222", f"nampula/{Q23}-em-casa-provincia-de-nampula-2017.xlsx"),
    ("q23_05", "20191114015944", f"tete/{Q23}-em-casa-provincia-de-tete-2017.xlsx"),
    ("q23_06", "20191114024424", f"manica/{Q23}-em-casa-provincia-de-manica-2017.xlsx"),
    ("q23_07", "20191114030150", f"sofala/{Q23}-em-casa.xlsx"),
    ("q23_08", "20191114032452", f"inhambane/{Q23}-em-casa-provincia-de-inhambane-2017.xlsx"),
    ("q23_08b", "20191114034458", f"inhambane/{Q23}-em-casa-provincia-de-inhambane-2017-1.xlsx"),
    ("q23_08c", "20191114032444", f"inhambane/{Q23}-em-casa-provincia-de-inhambane-2017-2.xlsx"),
    ("q23_09", "20191114042353", f"gaza/{Q23}-provincia-gaza-2017.xlsx"),
    ("q23_10", "20191114045047", f"maputo-provincia/{Q23}-em-casa-maputo-provincia-2017.xlsx"),
    ("q23_11", "20191114050632",
     "maputo-cidade/quadro-23-populacao-de-5-anos-e-mais-por-idade-segundo-sexo-e-lingua-que-fala-"
     "com-mais-frequencia-em-casa-maputo-cidade-2017.xlsx"),
    ("q23_11b", "20191114051354",
     "maputo-cidade/quadro-23-populacao-de-5-anos-e-mais-por-idade-segundo-sexo-e-lingua-que-fala-"
     "com-mais-frequencia-em-casa-maputo-cidade-2017-1.xlsx"),
]


def fetch():
    """urllib, not requests: on 2026-10-05 the Wayback Machine answered `requests` 429 for an
    hour while urllib and curl from the same machine got 200."""
    import urllib.error
    import urllib.request

    RAW.mkdir(parents=True, exist_ok=True)
    for name, ts, path in FILES:
        dest = RAW / f"{name}.xlsx"
        if dest.exists() and dest.read_bytes()[:4] == b"PK\x03\x04":
            continue
        url = WAYBACK.format(ts=ts, path=path)
        # The Wayback Machine answers 429/503/504 in bursts; retry with a pause.
        for attempt in range(12):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=180) as r:
                    body = r.read()
                break
            except (urllib.error.URLError, TimeoutError) as e:
                print(f"  {name}: {e}, retrying")
                time.sleep(min(120, 20 * (attempt + 1)))
        else:
            raise SystemExit(f"{name}: the Wayback Machine did not return {url}")
        if body[:4] != b"PK\x03\x04":
            raise SystemExit(f"{name}: not an xlsx -- starts {body[:16]!r}")
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(body)
        os.replace(tmp, dest)
        print(f"got  {name}: {len(body):,} bytes")
        time.sleep(5)


# ---------------------------------------------------------------- read

PROVINCES = {"01": "Niassa", "02": "Cabo Delgado", "03": "Nampula", "04": "Zambézia",
             "05": "Tete", "06": "Manica", "07": "Sofala", "08": "Inhambane", "09": "Gaza",
             "10": "Maputo Província", "11": "Maputo Cidade"}
# which capture of each table is read, where INE's folder holds two (see the docstring)
Q22_FILE = {c: f"q22_{c}" for c in PROVINCES}
Q23_FILE = {c: f"q23_{c}" for c in PROVINCES if c != "04"}
# Inhambane: the plain capture prints the right numbers against labels shifted by one row
# (Portuguese 26,566 is the "other Mozambican" figure, Bitonga 703,856 is Xitswa's); the `-1`
# re-upload relabels it. Chosen because only it makes the provinces' Portuguese, foreign,
# mute and unknown rows add up to the national table (check 1); docstring has the numbers.
Q22_FILE["08"] = "q22_08b"
Q22_FILE["11"] = "q22_11b"      # Maputo Cidade: the same slip and the same `-1` fix
Q23_FILE["08"] = "q23_08b"
Q23_FILE["11"] = "q23_11b"

# Folded printed label -> one canonical label. INE spells the same answer differently from one
# province's file to the next (CINHANJA in Niassa, CINYANJA elsewhere; ELOMWE and ELOMWUE;
# OURAS for OUTRAS in the national file; Outas in Zambézia's); those are merged here and only
# here. A label not in this table stops the run.
SPELLINGS = {
    "portugues": "Português",
    "emakhuwa": "Emakhuwa",
    "xichangana": "Xichangana",
    "elomwue": "Elomwe", "elomwe": "Elomwe",
    "cinyanja": "Cinyanja", "cinhanja": "Cinyanja",
    "cisena": "Cisena",
    "echuwabo": "Echuwabo",
    "cindau": "Cindau",
    "xitswa": "Xitswa",
    "ciyao": "Ciyao",
    "shimakonde": "Shimakonde",
    "kimwani": "Kimwani",
    "kiswalhili": "Kiswahili", "kiswahili": "Kiswahili",
    "coti": "Coti",
    "lolomalolo": "Lolo/Malolo",
    "cinyungwe": "Cinyungwe",
    "cishona": "Cishona",
    "chitewe": "Chitewe",
    "cimanika": "Cimanika",
    "chibalke": "Chibalke",
    "xitshwa": "Xitswa",
    "xirhonga": "Xironga",
    "chichopi": "Cicopi", "cicopichichopi": "Cicopi",
    "bitonga": "Bitonga",
    "mudo": "Mudo", "mudos": "Mudo",
    "ouraslinguasmocambicanas": "Outras línguas moçambicanas",
    "outraslinguasmocambicanas": "Outras línguas moçambicanas",
    "ouraslinguasestrangeiras": "Outras línguas estrangeiras",
    "outraslinguasestrangeiras": "Outras línguas estrangeiras",
    "outaslinguasestrangeiras": "Outras línguas estrangeiras",
    "desconhecida": "Desconhecida",
}
NOT_LANGUAGES = ("Mudo", "Desconhecida")
# THE NATIONAL QUADRO 22 IS A DIFFERENT EDIT FROM THE PROVINCIAL ONES (recorded 2026-10-05).
# Totals, Portuguese and Mudo agree to the person; three cells do not, national minus the sum
# of the provinces. The foreign difference is exactly Cabo Delgado's KISWALHILI (26,261): the
# national table counts Swahili as foreign. The unknowns: the national table has 265,049 fewer,
# and those people sit in its Mozambican languages, i.e. most of Cabo Delgado's 290,465
# unknowns were assigned a language nationally and not provincially. The provinces are drawn.       # in the table, never a language
OTHER_MOZ = "Outras línguas moçambicanas"
NATIONAL_DIFF = {"Outras línguas estrangeiras": 26_261, "Desconhecida": -265_049,
                 "Mozambican languages": 238_788}
FOREIGN = "Outras línguas estrangeiras"


def fold(s):
    s = unicodedata.normalize("NFKD", "" if s is None else str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def read_table(name, title_key, strict=True):
    """One Quadro 22 or 23 -> {area: {category: count}} for both sexes, all ages.

    The sheet is blocks: an area row (TOTAL, URBANA/Urbano, RURAL) with its universe, the
    categories, then HOMENS and MULHERES each with the same categories. Pages repeat the title.
    """
    import openpyxl

    path = RAW / f"{name}.xlsx"
    if not path.exists():
        raise SystemExit(f"missing {path} -- run with --fetch first")
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    rows = [tuple(r) for r in wb.worksheets[0].iter_rows(values_only=True)]
    wb.close()
    title = fold(" ".join(str(c) for r in rows[:2] for c in r if c is not None))
    if not title.startswith(title_key):
        raise SystemExit(f"{name}: title is not {title_key}: {title[:90]!r}")

    out, area, sex = {}, None, None
    univ = {}
    for r in rows:
        lab, val = r[0], r[1] if len(r) > 1 else None
        if lab is None or not isinstance(val, (int, float)):
            continue
        f = fold(lab)
        if f == "total" or f.startswith("urban") or f == "rural":
            area = "total" if f == "total" else ("urban" if f.startswith("urban") else "rural")
            if area in univ:
                raise SystemExit(f"{name}: two {area} blocks")
            univ[area] = {"all": int(val)}
            out[area] = {"all": {}, "homens": {}, "mulheres": {}}
            sex = "all"
            continue
        if f in ("homens", "mulheres"):
            sex = f
            univ[area][sex] = int(val)
            continue
        if f not in SPELLINGS:
            raise SystemExit(f"{name}: label {lab!r} is not in SPELLINGS")
        cat = SPELLINGS[f]
        if cat in out[area][sex]:
            raise SystemExit(f"{name}: {cat} twice in the {area}/{sex} block")
        out[area][sex][cat] = int(val)

    # each block sums to its universe; men + women = all, category by category
    for a, blocks in out.items():
        for s, cats in blocks.items():
            if sum(cats.values()) != univ[a][s]:
                raise SystemExit(f"{name} {a}/{s}: categories sum to {sum(cats.values()):,}, "
                                 f"the row prints {univ[a][s]:,}")
        if strict and set(blocks["homens"]) != set(blocks["all"]) or strict and any(
                blocks["homens"][c] + blocks["mulheres"][c] != v for c, v in blocks["all"].items()):
            raise SystemExit(f"{name} {a}: Homens + Mulheres is not the total row")
    if "urban" in out:
        t, u, ru = out["total"]["all"], out["urban"]["all"], out["rural"]["all"]
        if set(t) != set(u) or any(u[c] + ru[c] != v for c, v in t.items()):
            raise SystemExit(f"{name}: urban + rural is not the total")
    return {a: b["all"] for a, b in out.items()}


def read():
    q22 = {c: read_table(Q22_FILE[c], "quadro22") for c in PROVINCES}
    nat = read_table("q22_00", "quadro22")
    # Quadro 23 is a witness only; Sofala's urban block has a men + women slip, so its sex
    # check is off (its totals are still checked against Quadro 22 in check())
    q23 = {c: read_table(Q23_FILE[c], "quadro23", strict=False) for c in Q23_FILE}
    return q22, nat, q23


# ---------------------------------------------------------------- check


def check(q22, nat, q23):
    tot = lambda d: sum(d.values())  # noqa: E731
    natt = nat["total"]
    print(f"{'province':18} {'aged 5+':>11} {'urban':>6}  languages named")
    for c, t in q22.items():
        u = tot(t["urban"]) / tot(t["total"]) if "urban" in t else 1.0
        named = [k for k in t["total"] if k not in NOT_LANGUAGES + (OTHER_MOZ, FOREIGN)]
        print(f"{PROVINCES[c]:18} {tot(t['total']):11,} {u:6.3f}  {', '.join(named)}")

    # 1. the provinces sum to the national table: in total, and in every category every
    # province prints (Portuguese, foreign, mute, unknown)
    s = {}
    for t in q22.values():
        for k, v in t["total"].items():
            s[k] = s.get(k, 0) + v
    bad = []
    if tot(s) != tot(natt):
        bad.append(f"provinces sum to {tot(s):,}, the national table prints {tot(natt):,}")
    for k in ("Português", "Mudo"):
        if s[k] != natt[k]:
            bad.append(f"{k}: provinces {s[k]:,}, national {natt[k]:,}")
    # the national table is a different edit in three cells: pinned, so a re-issued file fails
    got = {FOREIGN: natt[FOREIGN] - s[FOREIGN], "Desconhecida": natt["Desconhecida"] - s["Desconhecida"]}
    # 2. a language the national table names is named by some provinces and folded into the
    # others' "other Mozambican languages", so the named provincial counts are at most the
    # national one, and Mozambican languages as a whole reconcile exactly
    moz = lambda d: sum(v for k, v in d.items()  # noqa: E731
                        if k not in NOT_LANGUAGES + ("Português", FOREIGN))
    for k, v in natt.items():
        if k in NOT_LANGUAGES + ("Português", FOREIGN, OTHER_MOZ):
            continue
        print(f"  {k:28} national {v:10,}  named by provinces {s.get(k, 0):10,}  "
              f"in other provinces' remainder {v - s.get(k, 0):9,}")
        if s.get(k, 0) > v:
            bad.append(f"{k}: provinces name {s[k]:,}, more than the national {v:,}")
    got["Mozambican languages"] = moz(natt) - sum(moz(t["total"]) for t in q22.values())
    if got != NATIONAL_DIFF:
        bad.append(f"national minus provinces is {got}, pinned {NATIONAL_DIFF}")
    if q22["02"]["total"]["Kiswahili"] != NATIONAL_DIFF[FOREIGN]:
        bad.append("Cabo Delgado's Kiswahili is no longer the foreign-language difference")
    print("  national minus provinces, pinned: " + ", ".join(f"{k} {v:+,}" for k, v in got.items()))
    # 3. the home-language table (Quadro 23) covers the same people, province by province
    for c, t in q23.items():
        if tot(t["total"]) != tot(q22[c]["total"]):
            bad.append(f"{PROVINCES[c]}: Quadro 23 total {tot(t['total']):,} is not Quadro 22's "
                       f"{tot(q22[c]['total']):,}")
    if bad:
        raise SystemExit("does not reconcile:\n  " + "\n  ".join(bad))
    print(f"\nthe eleven provinces reconcile with the national Quadro 22: {tot(natt):,} people "
          "aged 5+, Portuguese and mute to the person; foreign, unknown and Mozambican "
          "languages as a whole differ by the pinned amounts; Quadro 23 has the same totals in the ten provinces captured")

    # mother tongue against home language, the largest shifts (printed, not asserted)
    print("\nmother tongue (Q22) against language spoken most at home (Q23), share of aged 5+:")
    for c, t in q23.items():
        a, b = q22[c]["total"], t["total"]
        n = tot(a)
        keys = sorted(set(a) | set(b), key=lambda k: -a.get(k, 0))[:5]
        print(f"  {PROVINCES[c]:18} " + "  ".join(
            f"{k[:10]} {a.get(k, 0) / n:.3f}->{b.get(k, 0) / n:.3f}" for k in keys))


# ---------------------------------------------------------------- write


def write(q22):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for c, t in q22.items():
        # Maputo Cidade's table has no residence split; the census counts the city as urban
        areas = ("urban", "rural") if "urban" in t else ("total",)
        for a in areas:
            area = "urban" if c == "11" else a
            for k, v in t[a].items():
                rows.append([f"MZ{c}", "province", PROVINCES[c], area, k, v])
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["geo_id", "geo_level", "geo_name", "area", "source_category", "count"])
        w.writerows(rows)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(rows)} rows")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    q22, nat, q23 = read()
    check(q22, nat, q23)
    write(q22)
