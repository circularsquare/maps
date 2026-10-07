"""Ecuador: INEC, VIII Censo de Poblacion y VII de Vivienda 2022, the languages a person speaks or
communicates in ("Idiomas o lenguas que habla o se comunica"), asked of everyone aged 1 and over.
Several answers allowed: an indigenous language (one, named), Castellano, a foreign language,
Ecuadorian Sign Language.

    python sources/ec_censo.py --fetch     one 6 MB workbook, from the Wayback Machine
    python sources/ec_censo.py             normalise from data/raw/ec/

Writes data/normalized/ec.csv: one row per (canton, label), 221 cantons, every person shared
across the languages they named (spec §3.6), with the mentions beside the shares.

THE SOURCE is INEC's tabulado "2022_CPV_Autoidentificacion_Cultura.xlsx" from the census site's
WordPress uploads. www.censoecuador.gob.ec answers 403 to this machine on every path (the
homepage too, with a browser User-Agent; Apache, not a challenge page), and so do its REDATAM
server and the ANDA microdata catalogue (2026-10-05). --fetch therefore takes, in order:
religiondots' copy of the same file (fetched live from that URL on 2026-09-08, read-only), the
live URL, then the Wayback Machine's 2025-01-14 capture (stored gzip-encoded, unpacked here).
The Wayback capture and religiondots' copy are byte-identical (checked 2026-10-05). Its national
figures are asserted below against INEC's own published ones.

THREE TABLES of the workbook, all per canton, are used:
  5.1  people aged 1+ by WHICH COMBINATION of the four classes they named (indigenous, Castellano,
       foreign, sign): four singles, six pairs, "three or more", "does not speak or communicate".
       This is the cross-table spec §3.6 asks for, so the split is exact, not scaled.
  7.1  mentions of each class (a person counted once in every class they named).
  10.1 speakers of an indigenous language by WHICH indigenous language: 14 named languages and
       "Otras Lenguas Indigenas". It sums exactly to 7.1's indigenous mentions, so each person
       names one indigenous language.
and 1.1 for each canton's whole population (to count the under-1s, who are not asked).

THE SPLIT, per canton:
  * a person who named one class counts 1 to it; a pair counts 1/2 to each of its two.
  * "Tres o mas idiomas" (7,594 people nationally) does not say which three or four. 7.1 minus the
    singles and pairs gives each class's mentions among them, r_l. With n3 such people,
    n4 = sum(r) - 3 n3 named all four, and n3 - r_l named the three without l. So their share is
    exact too: class l gets n4/4 + (r_l - n4)/3. Asserted 0 <= r_l <= n3 and 0 <= n4 <= min(r).
  * a canton's indigenous share is divided among its indigenous languages in 10.1's proportions.
    This assumes that, within one canton, a Shuar speaker is as likely to also speak Castellano
    as a Kichwa speaker is. Nothing published crosses the two.

CHECKS (all asserted)
  * 221 cantons in 24 provinces; every table's cantons are the same set.
  * 5.1: the twelve combination columns sum to the row's own 1+ total on every canton.
  * 7.1 against 5.1: each class's mentions are reproduced by the combinations (via the r_l test
    above), and "No habla" is the same number in both.
  * 10.1: the fifteen languages sum to the row's own total, and that equals 7.1's indigenous
    mentions, on every canton.
  * cantons rebuild their province rows and provinces the national row, in 5.1, 7.1 and 10.1.
  * 1.1's canton population is at least 5.1's 1+ population, and the national figure is
    INEC's 16,938,986.
  * national figures pinned: population 16,938,986 (INEC's census count); 1+ 16,697,236;
    indigenous 659,361; Castellano 16,469,637; foreign 472,520; sign 19,992; Kichwa 538,449;
    Shuar Chicham 58,770 (the workbook's own national rows, pinned so a different file fails).
    Indigenous is 3.95% of people aged 1+, INEC's own headline for the census.
"""

import gzip
import io
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ec")
OUT = os.path.join(ROOT, "data", "normalized", "ec.csv")

NAME = "2022_CPV_Autoidentificacion_Cultura.xlsx"
LIVE = "https://www.censoecuador.gob.ec/wp-content/uploads/2024/02/" + NAME
WAYBACK = "https://web.archive.org/web/20250114111544id_/" + LIVE
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

CLASSES = ["ind", "sp", "ext", "sign"]
LABEL = {"ind": None,                      # split into 10.1's languages
         "sp": "Castellano o Español",
         "ext": "Idioma extranjero",
         "sign": "Lengua de señas ecuatoriana"}
# 5.1's twelve columns, in the sheet's order, as the set of classes each names
COMBOS = [("ind",), ("sp",), ("ext",), ("sign",),
          ("ind", "sp"), ("ind", "ext"), ("ind", "sign"),
          ("sp", "ext"), ("sp", "sign"), ("ext", "sign"),
          "3+", "none"]
COMBO_HEAD = ["Solo idioma o lengua indígena", "Solo castellano o español", "Solo idioma extranjero",
              "Solo Lengua de señas ecuatoriana", "Idioma o lengua indígena y castellano o español",
              "Idioma o lengua indígena e idioma extranjero",
              "Idioma o lengua indígena y lengua de señas ecuatoriana",
              "Castellano o español e idioma extranjero",
              "Castellano o español y lengua de señas ecuatoriana",
              "Idioma extranjero y lengua de señas ecuatoriana", "Tres o más idiomas",
              "No habla/No se comunica"]
MENTION_HEAD = ["Idioma o lengua indígena", "Castellano o Español", "Idioma extranjero",
                "Lengua de señas ecuatoriana", "No habla/No se comunica"]
INDIGENOUS = ["A´Ingae", "Achuar Chicham", "Andwa Pukwano", "Awapit", "Bai Coca", "Chaa´palaa",
              "Kichwa", "Otras Lenguas Indigenas", "Paaikoka", "Sapara", "Shiwiar Chicham",
              "Shuar Chicham", "Siapedee", "Tsa´Fiki", "Wao Tededo"]

PUBLISHED = {"pop": 16_938_986, "pop1": 16_697_236, "ind": 659_361, "sp": 16_469_637,
             "ext": 472_520, "sign": 19_992, "Kichwa": 538_449, "Shuar Chicham": 58_770}
EXPECTED_CANTONS = 221
EXPECTED_PROVINCES = 24


def fetch():
    import requests
    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print("already have", dest)
        return
    body = None
    rd_copy = os.path.join(ROOT, "..", "religiondots", "data", "raw", "ec", NAME)
    if os.path.exists(rd_copy):           # read-only: religiondots fetched it live, 2026-09-08
        with open(rd_copy, "rb") as fh:
            body = fh.read()
        print(f"  copied religiondots' {rd_copy}")
    for url in (() if body else (LIVE, WAYBACK)):
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=600)
        except requests.RequestException as e:
            print(f"  {url}: {e}")
            continue
        print(f"  {r.status_code} {url}")
        if r.status_code == 200:
            body = r.content
            break
    if body is None:
        raise SystemExit("no copy reachable; hand Anita the URL: " + LIVE)
    if body[:2] == b"\x1f\x8b":          # the Wayback capture is stored gzip-encoded
        body = gzip.decompress(body)
    if body[:2] != b"PK":
        raise SystemExit("download is not an xlsx (zip) file")
    with open(dest + ".part", "wb") as fh:
        fh.write(body)
    os.replace(dest + ".part", dest)
    print(f"  wrote {dest} ({len(body):,} bytes)")


def _rows(wb, sheet):
    return list(wb[sheet].iter_rows(values_only=True))


def _header_check(rows, heads, first_col):
    hdr = [str(v).strip() for v in rows[9][first_col:first_col + len(heads)]]
    if hdr != heads:
        raise SystemExit(f"header changed: {hdr} != {heads}")


def _num(v):
    return 0 if v is None else int(v)


def read_levels(rows, first_col, ncols, canton_tot=lambda r: r[3], second=lambda r: r[4]):
    """{('nat',): vals, ('prov', p): vals, ('canton', p, c): vals} from the total rows only.

    A total row has its two innermost label columns both 'Total <name>' (5.1, 7.1, 1.1, 10.1
    alike); a province's has col 2 = 'Total <province>'.
    """
    out = {}
    for r in rows[10:]:
        p, c = r[1], r[2]
        if p is None or c is None or str(p).startswith("Nota"):
            continue
        p, c = str(p).strip(), str(c).strip()
        a, b = str(r[3]).strip(), str(r[4]).strip()
        vals = [_num(v) for v in r[first_col:first_col + ncols]]
        if p == "Total Nacional":
            if a == "Nacional" and b == "Nacional":
                key = ("nat",)
            else:
                continue
        elif c == f"Total {p}":
            if a == c and b == c:
                key = ("prov", p)
            else:
                continue
        else:
            if a == f"Total {c}" and b == f"Total {c}":
                key = ("canton", p, c)
            else:
                continue
        if key in out:
            raise SystemExit(f"duplicate total row {key}")
        out[key] = vals
    return out


def split_canton(combo, ment):
    """Person shares per class for one canton. combo: 12 values (5.1), ment: 4 class mentions."""
    share = dict.fromkeys(CLASSES, 0.0)
    used = dict.fromkeys(CLASSES, 0)
    for spec, n in zip(COMBOS, combo):
        if spec in ("3+", "none"):
            continue
        for cl in spec:
            share[cl] += n / len(spec)
            used[cl] += n
    n3 = combo[COMBOS.index("3+")]
    r = {cl: m - used[cl] for cl, m in zip(CLASSES, ment)}
    if any(v < 0 or v > n3 for v in r.values()):
        raise SystemExit(f"mentions not reproduced by combinations: r={r}, n3={n3}")
    n4 = sum(r.values()) - 3 * n3
    if not (0 <= n4 <= min(r.values())):
        raise SystemExit(f"impossible three-or-more split: r={r}, n3={n3}, n4={n4}")
    for cl in CLASSES:
        share[cl] += n4 / 4 + (r[cl] - n4) / 3
    return share, n3, n4


def main():
    import openpyxl
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()
    path = os.path.join(RAW, NAME)
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}: run with --fetch")
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)

    r51, r71, r101, r11 = (_rows(wb, s) for s in ("5.1", "7.1", "10.1", "1.1"))
    _header_check(r51, COMBO_HEAD, 6)
    _header_check(r71, MENTION_HEAD, 5)
    _header_check(r101, INDIGENOUS, 6)
    t51 = read_levels(r51, 5, 13)       # 1+ total, then the 12 combinations
    t71 = read_levels(r71, 5, 5)
    t101 = read_levels(r101, 5, 16)     # total, then the 15 languages
    t11 = read_levels(r11, 5, 1)        # whole population

    cantons = sorted(k for k in t51 if k[0] == "canton")
    provs = sorted(k for k in t51 if k[0] == "prov")
    assert len(cantons) == EXPECTED_CANTONS, len(cantons)
    assert len(provs) == EXPECTED_PROVINCES, len(provs)
    for name, t in (("7.1", t71), ("1.1", t11)):
        ks = sorted(k for k in t if k[0] == "canton")
        assert ks == cantons, f"{name} cantons differ from 5.1's"
    k101 = set(k for k in t101 if k[0] == "canton")
    assert k101 <= set(cantons), f"10.1 has cantons 5.1 lacks: {k101 - set(cantons)}"
    print(f"  {len(cantons)} cantons, {len(provs)} provinces in 5.1, 7.1, 1.1; "
          f"{len(k101)} cantons with an indigenous speaker in 10.1")

    # internal arithmetic
    zero101 = [0] * 16
    for k in [("nat",)] + provs + cantons:
        c = t51[k]
        assert sum(c[1:]) == c[0], f"5.1 {k}: combinations {sum(c[1:])} != total {c[0]}"
        assert t71[k][4] == c[12], f"{k}: no-habla differs between 7.1 and 5.1"
        lang = t101.get(k, zero101)
        assert sum(lang[1:]) == lang[0], f"10.1 {k}: languages do not sum to the total"
        assert lang[0] == t71[k][0], f"{k}: 10.1 total {lang[0]} != 7.1 indigenous {t71[k][0]}"
        assert t11[k][0] >= c[0], f"{k}: population below the 1+ population"
    print("  5.1 rows sum to their totals; 10.1 = 7.1 indigenous; no-habla agrees; "
          "pop >= pop 1+  (all units)")

    # cantons rebuild provinces, provinces rebuild the nation
    for name, t, n in (("5.1", t51, 13), ("7.1", t71, 5), ("10.1", t101, 16), ("1.1", t11, 1)):
        for pk in provs:
            s = [sum(t.get(ck, [0] * n)[i] for ck in cantons if ck[1] == pk[1]) for i in range(n)]
            assert s == t.get(pk, [0] * n), f"{name}: cantons do not rebuild {pk}"
        s = [sum(t.get(pk, [0] * n)[i] for pk in provs) for i in range(n)]
        assert s == t[("nat",)], f"{name}: provinces do not rebuild the national row"
    print("  cantons rebuild provinces and provinces the nation, in 5.1, 7.1, 10.1 and 1.1")

    nat = {"pop": t11[("nat",)][0], "pop1": t51[("nat",)][0], "ind": t71[("nat",)][0],
           "sp": t71[("nat",)][1], "ext": t71[("nat",)][2], "sign": t71[("nat",)][3],
           "Kichwa": t101[("nat",)][1 + INDIGENOUS.index("Kichwa")],
           "Shuar Chicham": t101[("nat",)][1 + INDIGENOUS.index("Shuar Chicham")]}
    for k, v in PUBLISHED.items():
        assert nat[k] == v, f"national {k} {nat[k]:,} != published {v:,}"
    print("  pinned national figures: all 8 exact "
          f"(indigenous {nat['ind'] / nat['pop1']:.2%} of people aged 1+)")

    rows = []
    tot3 = tot4 = 0
    for k in cantons:
        _, p, c = k
        combo = t51[k][1:]
        share, n3, n4 = split_canton(combo, t71[k][:4])
        tot3 += n3
        tot4 += n4
        base = dict(province=p, canton=c, geo_level="canton", tier="derived")
        for cl, m in zip(CLASSES, t71[k][:4]):
            if cl == "ind":
                lang = t101.get(k, zero101)
                for name, n in zip(INDIGENOUS, lang[1:]):
                    if n:
                        rows.append(dict(base, source_category=name,
                                         count=share["ind"] * n / lang[0], mentions=n))
            elif m:
                rows.append(dict(base, source_category=LABEL[cl], count=share[cl], mentions=m))
        if combo[-1]:
            rows.append(dict(base, source_category="No habla/No se comunica",
                             count=float(combo[-1]), mentions=combo[-1]))
        under1 = t11[k][0] - t51[k][0]
        if under1:
            rows.append(dict(base, source_category="Menores de 1 año (no se pregunta)",
                             count=float(under1), mentions=under1))
        # each canton's shares + no-habla + under-1s rebuild its whole population
        drawn = sum(x["count"] for x in rows if x["canton"] == c and x["province"] == p)
        assert abs(drawn - t11[k][0]) < 1e-6, f"{k}: rows {drawn} != population {t11[k][0]}"
    print(f"  three-or-more: {tot3:,} people, of whom {tot4:,} named all four classes")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT + ".part", index=False, encoding="utf-8")
    os.replace(OUT + ".part", OUT)
    g = df.groupby("source_category")[["count", "mentions"]].sum().sort_values("count",
                                                                               ascending=False)
    print(g.round(0).astype(int).to_string())
    print(f"  wrote {OUT}: {len(df):,} rows, {df['count'].sum():,.0f} people")


if __name__ == "__main__":
    main()
