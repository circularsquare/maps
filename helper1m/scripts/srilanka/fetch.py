"""Sri Lanka populations for helper1m: province, district, DS division.

Writes helper1m/data/srilanka/population.csv  (columns: code, level, year, pop)

  level 3 = DS division (340), code = census DS uid, e.g. "2103" (district 21, DS 03)
  level 2 = district (25),     code = census district code, e.g. "21"
  level 1 = province (9),      code = census province code, e.g. "2"

Years:
  2024  Census of Population and Housing 2024 (DCS), GN-level population workbook
        (GN_population_excel, 14,008 GN divisions, 21,781,800), summed to DS divisions.
  2012  Census of Population and Housing 2012, district reports Table A1 (population by
        DS division; 331 DS divisions, 20,359,439), carried onto the 2024 DS divisions:
        renames are 1:1, and the nine DS divisions created since 2012 get their parent's
        2012 count split in proportion to the children's 2024 counts (SPLITS below).

Run download.py first. Usage:  C:\\Python39\\python.exe helper1m\\scripts\\srilanka\\fetch.py
"""
from __future__ import annotations

import csv
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
import a1_2012  # noqa: E402
import download  # noqa: E402

HELPER = Path(__file__).resolve().parents[2]
RAW = HELPER / "data" / "srilanka" / "raw"
OUT = HELPER / "data" / "srilanka" / "population.csv"

NATIONAL = {2012: 20_359_439, 2024: 21_781_800}

# 2012 district report folder -> census district code
DISTRICT_CODE = {
    "Colombo": "11", "Gampaha": "12", "Kalutara": "13", "Kandy": "21", "Matale": "22",
    "NuwaraEliya": "23", "Galle": "31", "Matara": "32", "Hambantota": "33", "Jaffna": "41",
    "Mannar": "42", "Vavuniya": "43", "Mullaitivu": "44", "Kilinochchi": "45",
    "Batticaloa": "51", "Ampara": "52", "Trincomalee": "53", "Kurunegala": "61",
    "Puttalam": "62", "Anuradhapura": "71", "Polonnaruwa": "72", "Badulla": "81",
    "Moneragala": "82", "Ratnapura": "91", "Kegalle": "92"}

# 2012 name -> 2024 name, where the fold below does not already pair them.
# Renames (same unit, new name) and spelling differences between the two releases.
ALIAS = {
    ("11", "Hanwella"): "Seethawaka",                # renamed after Seethawaka town
    ("11", "Dehiwala-Mount Lavinia"): "Dehiwala",
    ("11", "Kasbewa"): "Kesbewa",
    ("21", "Gagawata Korale"): "Kandy Four Gravets & Gangawata Korale",
    ("21", "Poojapitiya"): "Pujapitiya",
    ("21", "Delthota"): "Deltota",
    ("31", "Benthota"): "Bentota",
    ("31", "Welivitiya-Ddivithura"): "Welivitiya-Divithura",
    ("41", "Island North(Kayst)"): "Island North (Kayts)",
    ("41", "Valikamam North"): "Valikamam North (Tellipallai)",
    ("41", "Vadamaradchi North (Point Perdro)"): "Vadamaradchi North (Point Pedro)",
    ("42", "Nanaddan"): "Nanattan",
    ("42", "Musalai"): "Musali",
    ("44", "Welioya(*)"): "Welioya",
    ("44", "Puthukudiyruppu"): "Puthukkudiyiruppu",
    ("45", "Pachchilapalli"): "Pachchilaipalli",
    ("51", "Koralai Pattu North"): "Koralai Pattu North (Vaharai)",
    ("51", "Portivu Pattu"): "Porativu Pattu",
    ("51", "Manmunai South & Eravil Pattu"): "Manmunai South & Eruvil pattu",
    ("52", "Kalmunai Tamil Division"): "Kalmunai North Sub",
    ("52", "Sainthamararathu"): "Sainthamaruthu",
    ("52", "Karativu"): "Karaitheevu",
    ("52", "Eragama"): "Irakkamam",                  # Sinhala and Tamil names of one place
    ("52", "Alaydiwembu"): "Alayadiwembu",
    ("53", "Verugal/Echchilampattai"): "Verugal (Eachchilampattu)",
    ("53", "Kantalai"): "Kanthale",
    ("71", "Kahatahagasdigiliya"): "Kahatagasdigiliya",
    ("71", "Nachchadoowa"): "Nachchaduwa",
    ("81", "Uva-Paranagama"): "Uva Paranagama",
    ("82", "Siyambalaanduwa"): "Siyambalanduwa",
    ("91", "Opanayaka"): "Opanayake",
}

# DS divisions split since 2012: 2012 parent(s) -> 2024 children. Parent and child are
# identified by their GN-division numbers (each child's GN numbers sit inside its
# parent's run) and checked by population: the children's 2024 sum is 0.99-1.08x the
# parent's 2012 count, in line with each district's growth.
SPLITS = {
    "23": [(["Kothmale"], ["Kothmale West", "Kothmale East"]),
           (["Hanguranketha"], ["Hanguranketha", "Mathurata"]),
           (["Walapane"], ["Walapane", "Nildandahinna"]),
           (["Nuwara Eliya"], ["Nuwara Eliya", "Thalawakelle"]),
           (["Ambagamuwa"], ["Ambagamuwa Koralaya", "Norwood"])],
    "31": [(["Hikkaduwa"], ["Hikkaduwa", "Rathgama", "Madampagama"]),
           (["Baddegama"], ["Baddegama", "Wanduramba"])],
    "91": [(["Balangoda"], ["Balangoda", "Kaltota"])],
    # Katupotha was a sub-office of Panduwasnuwara in 2012; in 2024 the area is two full
    # DS divisions, West and East, whose GN numbers interleave. Which 2012 unit became
    # which is not clear, so the 2012 pair is split over the 2024 pair by 2024 shares.
    "61": [(["Panduwasnuwara", "Katupotha (Sub Office)"],
            ["Panduwasnuwara West", "Panduwasnuwara East"])],
}


def fold(s):
    s = re.sub(r"\(.*?\)|\*", " ", s.lower())
    s = s.replace("th", "t").replace("dh", "d").replace("w", "v").replace("ee", "i")
    s = s.replace("oo", "u").replace("ck", "k")
    s = re.sub(r"[^a-z]", "", s)
    return re.sub(r"(.)\1+", r"\1", s)


def largest_remainder(weights, total):
    base = sum(weights)
    shares = [total * w / base for w in weights]
    add = [int(s) for s in shares]
    order = sorted(range(len(weights)), key=lambda i: shares[i] - add[i], reverse=True)
    for i in order[:total - sum(add)]:
        add[i] += 1
    return add


def read_2024():
    """{ds_code: (district_code, ds_name, pop)} from the GN workbook."""
    import openpyxl
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wb = openpyxl.load_workbook(RAW / "GN_population.xlsx", read_only=True,
                                    data_only=True)
    rows = list(wb.worksheets[0].iter_rows(values_only=True))
    wb.close()
    head = [str(c).replace("\n", " ") if c else "" for c in rows[3]]
    if head[:9] != ["Province Code", "Province Name", "District Code", "District Name",
                    "DS_Division Code", "DS_Division Name", "GN_Division Code",
                    "GN_Division Name", "GN_Division Number"] or rows[4][9] != "Total":
        raise SystemExit(f"GN workbook header moved: {head}")
    ds = {}
    n = 0
    for r in rows[5:]:
        if r[2] is None:
            continue
        tot, m, f = r[9], r[10], r[11]
        if not all(isinstance(v, int) for v in (tot, m, f)) or tot != m + f:
            raise SystemExit(f"GN row {r[:9]}: total {tot} != male {m} + female {f}")
        if sum(r[13:17]) != tot:
            raise SystemExit(f"GN row {r[:9]}: age groups do not sum to the total")
        code = f"{int(r[2]):02d}{int(r[4]):02d}"
        if str(r[0]) != code[0]:
            raise SystemExit(f"GN row {r[:9]}: province {r[0]} vs district {r[2]}")
        d, nm, p = ds.get(code, (code[:2], str(r[5]).strip(), 0))
        ds[code] = (d, nm, p + tot)
        n += 1
    total = sum(p for _, _, p in ds.values())
    print(f"2024: {n:,} GN divisions in {len(ds)} DS divisions, total {total:,}")
    if n != 14_008 or len(ds) != 340 or total != NATIONAL[2024]:
        raise SystemExit("2024 workbook: expected 14,008 GN, 340 DS, 21,781,800")
    return ds


def carry_2012(ds24):
    """2012 counts on the 2024 DS divisions: {ds_code: pop}."""
    by_name = defaultdict(dict)                    # district -> {2024 name: code}
    for code, (d, nm, _) in ds24.items():
        by_name[d][nm] = code
    out = {}
    grand = 0
    for folder in download.DISTRICTS_2012:
        d = DISTRICT_CODE[folder]
        dtot, rows12 = a1_2012.read(folder)
        grand += dtot
        names24 = by_name[d]
        fold24 = {}
        for nm, code in names24.items():
            if fold(nm) in fold24:
                raise SystemExit(f"fold collision in district {d}: {nm}")
            fold24[fold(nm)] = code
        pop12 = {}
        for nm, v in rows12:
            pop12[nm] = v
        used12, used24 = set(), set()
        for parents, children in SPLITS.get(d, []):
            tot12 = sum(pop12[p] for p in parents)
            kids = [names24[c] for c in children]
            w = [ds24[k][2] for k in kids]
            for k, v in zip(kids, largest_remainder(w, tot12)):
                out[k] = v
            used12.update(parents)
            used24.update(kids)
            print(f"  split {d} {' + '.join(parents)} ({tot12:,}) -> "
                  + ", ".join(f"{c} {out[k]:,}" for c, k in zip(children, kids))
                  + f"; children's 2024 sum / parent 2012 = {sum(w) / tot12:.3f}")
        for nm, v in rows12:
            if nm in used12:
                continue
            target = ALIAS.get((d, nm))
            code = names24.get(target) if target else fold24.get(fold(nm))
            if code is None:
                raise SystemExit(f"2012 DS {nm!r} (district {d}) has no 2024 match")
            if code in used24:
                raise SystemExit(f"2012 DS {nm!r} pairs with an already-used 2024 unit")
            used24.add(code)
            out[code] = v
        left = set(names24.values()) - used24
        if left:
            raise SystemExit(f"district {d}: 2024 DS without a 2012 figure: "
                             + ", ".join(ds24[c][1] for c in left))
        s = sum(out[c] for c in names24.values())
        if s != dtot:
            raise SystemExit(f"district {d}: carried 2012 sums to {s:,}, A1 says {dtot:,}")
    if grand != NATIONAL[2012]:
        raise SystemExit(f"2012 national {grand:,} != {NATIONAL[2012]:,}")
    print(f"2012: 331 DS divisions, {grand:,}, carried onto {len(out)} 2024 DS divisions; "
          "every district equals its A1 row")
    return out


def main():
    ds24 = read_2024()
    ds12 = carry_2012(ds24)
    rows = []
    agg = defaultdict(int)
    for code, (d, nm, p24) in ds24.items():
        for y, v in ((2012, ds12[code]), (2024, p24)):
            rows.append((code, 3, y, v))
            agg[(d, 2, y)] += v
            agg[(code[0], 1, y)] += v
    rows += [(c, lv, y, v) for (c, lv, y), v in agg.items()]
    rows.sort(key=lambda r: (r[1], r[0], r[2]))
    for y, want in NATIONAL.items():
        got = sum(v for (c, lv, yy), v in agg.items() if lv == 1 and yy == y)
        assert got == want, (y, got, want)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "level", "year", "pop"])
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
