"""Austria: the 2001 census's Muslims divided by branch, BORROWING Germany's MLD 2020 shares.

Writes data/normalized/at_mld.csv: per Bundesland, the share of its census Muslims drawn as Sunni,
Shi'a, Alevi and Ahmadi. countries/at.py applies them to every Gemeinde's `Islamisch` row as
`derived` rows rolling back to `islam` (the census counted these Muslims; only the branch is
borrowed). Approved by Anita 2026-10-16 (unspecified_splits/islam.md §5): "split their Muslims by
borrowing the German MLD 2020 origin-region sect shares applied to each country's Muslim origin
mix as best the country publishes it".

WHAT AUSTRIA PUBLISHES. Volkszählung 2001, Tabelle 15, *Bevölkerung nach Religion, Alter und
Staatsangehörigkeit*, in each Land volume and the national one (data/raw/at/*.pdf, the same files
sources/at.py reads): Muslims by citizenship, per Bundesland. That is the finest origin mix
Austria publishes for its Muslims.

THE BORROWED SHARES. BAMF, *Muslimisches Leben in Deutschland 2020*, Table 32: branch by origin
region, measured among Muslims IN GERMANY (sources/de_mld.py holds and checks the table). This is a
German survey's figure applied to Austria; the note says so.

    Türkei                                             -> MLD's Türkei column
    Bosnien-Herzegowina, Jugoslawien (Serbien, Mont.),
    Mazedonien, Kroatien, Slowenien                    -> MLD's Südosteuropa column
    Österreich, every other citizenship, unbekannt     -> NOT SPLIT, stays on `islam`

Austrian citizens (96,052 Muslims nationally, 28%) are not split because the census does not say
where a citizen's family came from, and the other citizenships are a residual whose origins are
not printed. So about a third of Austria's Muslims stay on plain `islam`, plus MLD's own unnamed
share of each split group.

Usage:
    python sources/at_mld.py             write data/normalized/at_mld.csv
    python sources/at_mld.py --dry-run   report, write nothing
"""

import os
import re
import sys

import fitz
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "sources"))
RAW = os.path.join(HERE, "data", "raw", "at")
OUT = os.path.join(HERE, "data", "normalized", "at_mld.csv")

# Land volume -> the first digit of the Gemeinde code (at.csv's geo_id is AT + code).
VOLUMES = {"bgld": ("1", "Burgenland"), "ktn": ("2", "Kärnten"), "noe": ("3", "Niederösterreich"),
           "ooe": ("4", "Oberösterreich"), "sbg": ("5", "Salzburg"), "stmk": ("6", "Steiermark"),
           "tirol": ("7", "Tirol"), "vbg": ("8", "Vorarlberg"), "wien": ("9", "Wien")}
NATIONAL = "oesterreich"
ISLAM_COL = 7          # 0-based: Insgesamt, r.-k., gr.-k., Orthodox, Ev., andere chr., Israelitisch, ISLAMISCH
NCOL = 11

# Citizenship row (regex, tolerant of the volumes' lost umlauts) -> MLD region, or None.
ROWS = [
    (r"^.{0,2}sterreich$", None),
    (r"^Deutschland$", None),
    (r"^Italien$", None),
    (r"^andere EU-Staaten$", None),
    (r"^Bosnien(-| und )Herzegowina$", "Südosteuropa"),
    (r"^Jugoslawien", "Südosteuropa"),
    (r"^Kroatien$", "Südosteuropa"),
    (r"^Mazedonien$", "Südosteuropa"),
    (r"^Slowenien$", "Südosteuropa"),
    (r"^Polen$", None),
    (r"^Rum.{0,2}nien$", None),
    (r"^Schweiz$", None),
    (r"^Slowakei$", None),
    (r"^Tschechische Republik$", None),
    (r"^T.{0,2}rkei$", "Türkei"),
    (r"^Ungarn$", None),
    (r"^Vereinigte Staaten$", None),
    (r"^anderer Staat", None),
]


def _int(s):
    s = s.strip()
    return 0 if s == "-" else int(s.replace(".", ""))


def tabelle15(stem):
    """{citizenship label: Muslims} and the Islam total, from one volume's Tabelle 15."""
    doc = fitz.open(os.path.join(RAW, f"{stem}.pdf"))
    for page in doc:
        t = page.get_text()
        if "Tabelle 15" not in t or "Staatsangeh" not in t:
            continue
        i = t.find("Nach Staatsangeh")
        if i < 0:
            continue
        lines = [x.strip() for x in t[i:].splitlines()]
        lines = [x for x in lines if x]
        nums = lambda k: [_int(v) for v in lines[k:k + NCOL]]
        if "--debug" in sys.argv:
            print(stem, lines[:40])
        # The national volume prints the total row right after the heading; the Länder volumes
        # do not, and their total is the table's own first row (`Insgesamt`).
        if re.match(r"^[\d\-]", lines[1]):
            total, k = nums(1), 1 + NCOL
        else:
            head = [x.strip() for x in t.splitlines() if x.strip()]
            j = head.index("Insgesamt", head.index("11") if "11" in head else 0)
            total, k = [_int(v) for v in head[j + 1:j + 1 + NCOL]], 1
        out = {}
        while k < len(lines):
            label = lines[k]
            if not re.match(r"^[^\d\-]", label) or label.startswith("Altersgruppen"):
                break
            try:
                row = nums(k + 1)
            except ValueError:
                break
            if len(row) < NCOL:
                break
            out[label] = row[ISLAM_COL]
            k += 1 + NCOL
        if "--debug" in sys.argv:
            print(stem, "total", total, "rows", out)
        if sum(out.values()) != total[ISLAM_COL]:
            raise SystemExit(f"{stem}: Tabelle 15 citizenship rows sum to {sum(out.values()):,} "
                             f"Muslims against its total {total[ISLAM_COL]:,}")
        return out, total[ISLAM_COL]
    raise SystemExit(f"{stem}: Tabelle 15 not found")


def classify(label):
    for rx, region in ROWS:
        if re.search(rx, label):
            return region, True
    return None, False


def build():
    import de_mld
    t32 = de_mld.T32
    col = {r: de_mld.REGIONS.index(r) for r in ("Türkei", "Südosteuropa")}
    named = {"Sunnitisch": "MLD 2020 (borrowed): Sunnitisch",
             "Schiitisch": "MLD 2020 (borrowed): Schiitisch",
             "Alevitisch": "MLD 2020 (borrowed): Alevitisch",
             "Ahmadi": "MLD 2020 (borrowed): Ahmadi"}

    nat, nat_total = tabelle15(NATIONAL)
    rows, land_sum = [], 0
    print(f"  {'Land':<18}{'Muslims':>9}{'Türkei':>9}{'SE Eur.':>9}{'unsplit':>9}"
          + "".join(f"{k[:8]:>10}" for k in named))
    for stem, (digit, name) in VOLUMES.items():
        mus, total = tabelle15(stem)
        land_sum += total
        by = {"Türkei": 0, "Südosteuropa": 0, None: 0}
        for label, n in mus.items():
            region, ok = classify(label)
            if not ok:
                raise SystemExit(f"{stem}: citizenship row {label!r} matches nothing in ROWS")
            by[region] += n
        shares = {k: sum(by[r] * t32[k][col[r]] / 100 for r in col) / total for k in named}
        print(f"  {name:<18}{total:>9,}{by['Türkei']:>9,}{by['Südosteuropa']:>9,}{by[None]:>9,}"
              + "".join(f"{100 * shares[k]:>9.1f}%" for k in named))
        for k, cat in named.items():
            rows.append(dict(geo_id=digit, geo_level="bundesland", geo_name=name,
                             source_category=cat, share=round(shares[k], 6),
                             muslims_2001=total, turkish=by["Türkei"],
                             south_east_europe=by["Südosteuropa"], unsplit=by[None],
                             source_id="at_vz_2001_t15 x bamf_mld_2020_t32",
                             note="VZ 2001 Tabelle 15 Muslims by citizenship x MLD 2020 Table 32"))
    if land_sum != nat_total:
        raise SystemExit(f"the nine Länder's Tabelle 15 Muslims sum to {land_sum:,}, the "
                         f"national volume's to {nat_total:,}")
    print(f"  OK nine Länder sum to the national Tabelle 15's {nat_total:,} Muslims")
    out = pd.DataFrame(rows)
    for k, cat in named.items():
        s = out[out["source_category"] == cat]
        print(f"      Austria {k:<12} {100 * (s['share'] * s['muslims_2001']).sum() / nat_total:5.1f}%")
    return out


def main():
    out = build()
    if "--dry-run" in sys.argv:
        print("  --dry-run: nothing written")
        return
    tmp = OUT + ".part"
    out.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, OUT)
    print(f"  wrote {OUT} ({len(out)} rows)")


if __name__ == "__main__":
    main()
