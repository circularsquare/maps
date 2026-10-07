"""Montenegro: MONSTAT, Popis 2023, mother tongue (maternji jezik) by municipality.

    python sources/me_census.py --fetch    five xlsx GETs, seconds
    python sources/me_census.py            normalise from data/raw/me/

-> data/normalized/me.csv (levels `country` and `municipality`; only `municipality` is drawn)

Tables (no key, no wall):

  t3_maternji_opstine.xlsx   THE DRAWN TABLE. "Tabela 3. Stanovnistvo Crne Gore prema maternjem
      jeziku po opstinama", on the national open-data portal (data.gov.me, dataset
      524a7296-0cc0-4990-bda3-49f69a83695c, resource c9b2405a-...). 26 answers (25 named or
      residual labels and "Ne zeli da se izjasni") by the country and its 25 municipalities,
      each as a count and a percentage column. The 2023 census's 25 municipalities include Tuzi
      (2018) and Zeta (2022), split out of Podgorica.
  t4_govori_opstine.xlsx     Tabela 4, language usually spoken, same layout. A second table of the
      same census: its totals must equal t3's in every municipality. Not drawn.
  t1_nacionalnost_opstine.xlsx  Tabela 1, ethnicity, same layout. The same totals check. Not drawn.
  naselja_maternji_jezik_2023.xlsx  MONSTAT (monstat.org/uploads/files/popis 2021/podaci/; the
      folder keeps the census's first scheduled year, see religiondots/sources/me.md), mother
      tongue by 1,462 settlements. Used only for placement inside a municipality
      (sources/me_geo.py, countries/me.py); here only checked against t3.
  naselja_popis_2023.xlsx    population by settlement and sex; read by sources/me_geo.py.

SUPPRESSION. MONSTAT blanks any cell under 10 as `z` ("zasticen podatak"), and further cells to stop
differencing (the workbooks' own legend says both). The national column has no `z`; the municipal
columns do. Per municipality the hidden people are the total less the published cells, and are
written as one `z (suppressed)` row so every municipality still partitions; resolve() drops it, and
the entry's gap says how many. Per category, the national row less the published municipal cells
gives what each language loses (printed; the record tables it). It is 736 people, 0.12%.

CHECKS: national total 623,633 (MONSTAT's published 2023 population); the national row partitions
exactly; per municipality the published cells never exceed the total; per category the published
municipal cells never exceed the national row; t4's and t1's totals equal t3's in all 25
municipalities; the settlement workbook's 25 municipality names are t3's, and its settlement
cells never exceed the municipal cell of the same answer.
"""

import csv
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "me")
OUT = os.path.join(ROOT, "data", "normalized", "me.csv")

SOURCE_ID = "me_popis_2023_mt"
YEAR = 2023
COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count", "tier", "year",
           "source_id", "note"]

PORTAL = "https://data.gov.me/dataset/524a7296-0cc0-4990-bda3-49f69a83695c/resource/"
MONSTAT = "https://www.monstat.org/uploads/files/popis%202021/podaci/"
FILES = {
    "t1_nacionalnost_opstine.xlsx": PORTAL + "275657ad-fd68-4df0-b9d3-68641e6e09d0/download/"
        "tabela-1.-stanovnitvo-crne-gore-prema-nacionalnoj-odnosno-etnikoj-pripadnosti-po-optinama.xlsx",
    "t3_maternji_opstine.xlsx": PORTAL + "c9b2405a-6bd5-4756-b232-a5446092c86d/download/"
        "tabela-3.-stanovnitvo-crne-gore-prema-maternjem-jeziku-po-optinama.xlsx",
    "t4_govori_opstine.xlsx": PORTAL + "cee5ce3c-9cf4-413f-8637-2efd3e546977/download/"
        "tabela-4.-stanovnitvo-crne-gore-prema-jeziku-kojim-se-uobiajeno-govori-po-optinama.xlsx",
    "naselja_maternji_jezik_2023.xlsx": MONSTAT + "naselja%20maternji%20jezik%20popis%202023.(1).xlsx",
    "naselja_popis_2023.xlsx": MONSTAT + "naselja%20popis%202023.xlsx",
}

NATIONAL = 623_633
NATION = "Crna Gora"
TOTAL = "Ukupno"
SUPPRESSED = "z (suppressed)"
N_MUNI = 25


def fetch():
    import requests
    h = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120 Safari/537.36"}
    os.makedirs(RAW, exist_ok=True)
    for name, url in FILES.items():
        dest = os.path.join(RAW, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
            print("already have", dest)
            continue
        print("GET", url)
        r = requests.get(url, timeout=300, headers=h)
        r.raise_for_status()
        if r.content[:2] != b"PK":
            raise SystemExit(f"{name}: not an xlsx (first bytes {r.content[:60]!r})")
        with open(dest, "wb") as fh:
            fh.write(r.content)
        print(f"  {len(r.content):,} bytes")


def _cell(v):
    """A count, or None for MONSTAT's `z` (suppressed). `-` is a true zero."""
    if isinstance(v, str):
        v = v.replace("\xa0", "").strip()
        if v == "z":
            return None
        if v == "-":
            return 0
        return int(float(v))
    if v != v:                                   # NaN
        raise ValueError("empty cell inside the table")
    return int(v)


def read_wide(name):
    """MONSTAT's municipal layout: row 0 = label, then <unit>, 'u %' pairs. -> {unit: {label: n}}."""
    import pandas as pd
    d = pd.read_excel(os.path.join(RAW, name), header=None)
    hdr = [str(x).strip() for x in d.iloc[0]]
    units = {i: hdr[i] for i in range(1, d.shape[1], 2)}
    if any(hdr[i + 1] != "u %" for i in units):
        raise SystemExit(f"{name}: the count / percent column pairs have moved")
    out = {u: {} for u in units.values()}
    for _, row in d.iloc[1:].iterrows():
        label = row.iloc[0]
        if not isinstance(label, str) or not label.strip():
            continue
        label = label.strip()
        for i, u in units.items():
            out[u][label] = _cell(row.iloc[i])
    return out


def read_settlements():
    """naselja_maternji_jezik_2023.xlsx -> DataFrame: opstina, naselje, total, one column per answer
    (None where suppressed)."""
    import pandas as pd
    d = pd.read_excel(os.path.join(RAW, "naselja_maternji_jezik_2023.xlsx"), header=None)
    legend = " ".join(str(x) for x in d.iloc[-6:, 0])
    if '"z" zaštićen podatak' not in legend:
        raise SystemExit("settlement workbook: the `z` legend is gone; re-read what `z` means")
    hdr = [str(x).strip() for x in d.iloc[1]]
    if hdr[:3] != ["Opština", "Naselje", "Ukupno"]:
        raise SystemExit(f"settlement workbook: header {hdr[:3]}")
    body = d.iloc[2:]
    body = body[body.iloc[:, 1].notna() & body.iloc[:, 0].notna()]
    rows = []
    for _, r in body.iterrows():
        rec = {"opstina": str(r.iloc[0]).strip(), "naselje": str(r.iloc[1]).strip()}
        for j in range(2, len(hdr)):
            rec[hdr[j]] = _cell(r.iloc[j])
        rows.append(rec)
    return pd.DataFrame(rows), hdr[3:]


def check(t3, t4, t1):
    ok = True
    nat = t3[NATION]
    munis = [u for u in t3 if u != NATION]
    good = len(munis) == N_MUNI
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {len(munis)} municipalities (expected {N_MUNI})")

    good = nat[TOTAL] == NATIONAL and all(v is not None for v in nat.values())
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} national total {nat[TOTAL]:,} (published {NATIONAL:,}), "
          "no suppressed national cell")
    s = sum(v for k, v in nat.items() if k != TOTAL)
    good = s == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the {len(nat) - 1} answers partition the national row ({s:,})")

    hidden = {}
    for u in munis:
        pub = sum(v or 0 for k, v in t3[u].items() if k != TOTAL)
        hidden[u] = t3[u][TOTAL] - pub
    neg = [u for u, h in hidden.items() if h < 0]
    ok &= not neg
    print(f"  {'OK ' if not neg else 'BAD'} published cells never exceed a municipality's total "
          f"(hidden {sum(hidden.values()):,} people, {100 * sum(hidden.values()) / NATIONAL:.2f}%)")
    s = sum(t3[u][TOTAL] for u in munis)
    good = s == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the 25 municipal totals sum to the national total ({s:,})")

    print("\n  per answer: national, published in municipalities, hidden by suppression")
    for k in nat:
        if k == TOTAL:
            continue
        pub = sum(t3[u][k] or 0 for u in munis)
        nz = sum(t3[u][k] is None for u in munis)
        if pub > nat[k]:
            ok = False
            print(f"  BAD {k}: municipalities {pub:,} > national {nat[k]:,}")
        if nz:
            print(f"    {k:<38} {nat[k]:>8,} {pub:>8,} {nat[k] - pub:>5,} "
                  f"({100 * (nat[k] - pub) / nat[k]:4.1f}%, {nz} cells)")

    for name, other in (("t4 (language usually spoken)", t4), ("t1 (ethnicity)", t1)):
        off = [u for u in t3 if other.get(u, {}).get(TOTAL) != t3[u][TOTAL]]
        good = not off and set(other) == set(t3)
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {name}: the same 26 columns and the same total in every "
              f"one" + (f" (off: {off})" if off else ""))
    return ok, hidden


def check_settlements(t3):
    st, cats = read_settlements()
    ok = True
    munis = sorted(u for u in t3 if u != NATION)
    names = sorted(st["opstina"].unique())
    good = names == munis
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} settlement workbook: {len(st):,} settlements in the same "
          f"{len(names)} municipalities")
    good = set(cats) == set(k for k in t3[NATION] if k != TOTAL)
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} ... with the same {len(cats)} answers as the municipal table")
    bad = 0
    for u, g in st.groupby("opstina"):
        for c in [TOTAL] + cats:
            pub = g[c].fillna(0).sum()
            ref = t3[u][c]
            if ref is not None and pub > ref:
                bad += 1
    ok &= bad == 0
    print(f"  {'OK ' if not bad else 'BAD'} no municipality's published settlement cells exceed "
          "its municipal cell")
    zt = st[TOTAL].isna()
    print(f"  {int(zt.sum())} settlements have their total suppressed; the rest hold "
          f"{int(st[TOTAL].sum()):,} of {NATIONAL:,} people")
    return ok


def main():
    if "--fetch" in sys.argv:
        fetch()
    t3 = read_wide("t3_maternji_opstine.xlsx")
    t4 = read_wide("t4_govori_opstine.xlsx")
    t1 = read_wide("t1_nacionalnost_opstine.xlsx")
    ok, hidden = check(t3, t4, t1)
    ok &= check_settlements(t3)

    rows = []
    for u, cells in t3.items():
        level = "country" if u == NATION else "municipality"
        for label, n in cells.items():
            if n is None:
                continue
            note = f"level={level}"
            if label == TOTAL:
                note += "; universe total, not a language"
            rows.append({"geo_id": u, "geo_level": level, "geo_name": u, "source_category": label,
                         "count": n, "tier": "measured", "year": YEAR, "source_id": SOURCE_ID,
                         "note": note})
        if level == "municipality" and hidden[u]:
            nz = sum(v is None for v in cells.values())
            rows.append({"geo_id": u, "geo_level": level, "geo_name": u,
                         "source_category": SUPPRESSED, "count": hidden[u], "tier": "measured",
                         "year": YEAR, "source_id": SOURCE_ID,
                         "note": f"level={level}; the {nz} cells MONSTAT printed as z, together"})

    print("\n  Answers, national:")
    for label, n in sorted(t3[NATION].items(), key=lambda kv: -kv[1]):
        print(f"    {n:>10,}  {100.0 * n / NATIONAL:5.2f}%  {label}")
    if not ok:
        raise SystemExit("reconciliation FAILED")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
