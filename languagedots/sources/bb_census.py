"""Barbados: first language from the 2010 census's birthplace and ethnic-origin tables per parish
-> data/normalized/bb.csv.

    python sources/bb_census.py

NO CENSUS LANGUAGE QUESTION (2010, 2021). Built under the 2026-10-05 ruling for countries with no
language question (AGENT_BRIEF §2), as the Bahamas (sources/bs_census.py): the national language
for the native-born, immigrant languages proxied by birthplace. Every row `derived`.

THE TABLES (religiondots' copy of the Barbados Statistical Service's 2010 census workbook,
`../religiondots/data/raw/bb/bb_census_tables_2010.xlsx`, read only), on the tabulable population
(226,193; the census's own undercount is 18%, sources/bb.md):
  * 01.01 total population by parish;
  * 02.04 population by parish and ethnic origin (Black, White, ..., Mixed);
  * 04.02 Barbadian-born population by parish of usual residence (its Total row);
  * 04.01 population by country of birth, national only.

THE MODEL, per parish:
  * foreign-born = parish total - Barbadian-born residents; split by the NATIONAL country-of-birth
    mix of the foreign-born whose country is known ("Countries Unknown", 12,164 of 32,825, is
    left out of the mix), each country on its language (taxonomy/bb2010.py);
  * Barbadian-born -> Bajan, except white Barbadians -> English. White Barbadian-born = the
    parish's White count x its Barbadian-born share (whites assumed as often native-born as
    everyone in the parish). Subtracting the UK/US/Canada-born instead leaves almost none, since
    many of those are the Black children of returning Barbadians (sources/bb.md).
The 2010 tables are used rather than 2021's: the 2021 tabulation covers 136,415 people, about
half the resident population.

CHECKS: 11 parishes; 01.01, 02.04 and 04.02 agree on the national totals (226,193; Barbadian-born
193,368); 02.04's parish totals equal 01.01's; the known countries plus Unknown sum to the
foreign-born total; every parish's drawn rows sum to its population.
"""
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(HERE), str(HERE / "taxonomy")]
from rdlink import RD  # noqa: E402

XLSX = RD / "data" / "raw" / "bb" / "bb_census_tables_2010.xlsx"
OUT = HERE / "data" / "normalized" / "bb.csv"
TOTAL, BORN_BB, FOREIGN, UNKNOWN = 226_193, 193_368, 32_825, 12_164
# 04.01's subtotal and pooled rows that are not a country of birth
NOT_COUNTRY = {"Total", "Total Barbados", "Total Foreign Countries", "Not Stated",
               "Countries Unknown"}


def _rows(wb, sheet):
    return [list(r) for r in wb[sheet].iter_rows(values_only=True)]


def _parish_blocks(rows, header_row):
    """{parish: Total row} for the `   <Parish>` / Total / Male / Female layout of 02.04."""
    hdr = rows[header_row]
    out, cur = {}, None
    for r in rows[header_row + 1:]:
        lab = r[0]
        if isinstance(lab, str) and lab.startswith("   ") and r[1] is None:
            cur = lab.strip()
        elif lab == "Total" and cur:
            out[cur] = {h: int(v or 0) for h, v in zip(hdr[1:], r[1:]) if h}
            cur = None
    return out


def parse():
    import openpyxl
    wb = openpyxl.load_workbook(XLSX, read_only=True)
    # 01.01
    tot = {}
    for r in _rows(wb, "01.01"):
        if (isinstance(r[0], str) and (r[0].startswith("St.") or r[0] == "Christ Church")
                and isinstance(r[1], int)):
            tot[r[0]] = r[1]
    assert len(tot) == 11 and sum(tot.values()) == TOTAL, tot
    # 02.04
    eth = _parish_blocks(_rows(wb, "02.04"), 2)
    assert eth.pop("Total")["Total"] == TOTAL
    assert {k: v["Total"] for k, v in eth.items()} == tot, eth
    for k, v in eth.items():
        assert sum(x for h, x in v.items() if h != "Total") == v["Total"], k
    # 04.02: the first Total row is Barbadian-born by parish of usual residence
    r42 = _rows(wb, "04.02")
    hdr = r42[2]
    row = next(r for r in r42[3:] if r[0] == "Total")
    born = {h: int(v) for h, v in zip(hdr[1:], row[1:]) if h}
    assert born.pop("Total") == BORN_BB and sum(born.values()) == BORN_BB, born
    assert set(born) == set(tot)
    # 04.01: country of birth, national
    cob, lab = {}, None
    for r in _rows(wb, "04.01"):
        v = [c for c in r if c is not None]
        if len(v) == 1 and isinstance(v[0], str):
            lab = v[0].strip()
        elif v and v[0] == "Total" and lab and isinstance(v[1], int):
            if lab == "Countries Unknown":
                assert v[1] == UNKNOWN
            elif lab == "Total Foreign Countries":
                assert v[1] == FOREIGN
            if lab not in NOT_COUNTRY:
                cob[lab] = int(v[1])
            lab = None
    assert sum(cob.values()) + UNKNOWN == FOREIGN, sum(cob.values())
    return tot, eth, born, cob


def build():
    import bb2010
    tot, eth, born, cob = parse()
    rd = pd.read_csv(RD / "data" / "normalized" / "bb.csv", dtype={"geo_id": str},
                     keep_default_na=False, na_values=[])
    ids = dict(rd.loc[rd["geo_level"] == "parish", ["geo_name", "geo_id"]].drop_duplicates()
               .values)
    assert set(ids) == set(tot), (set(ids) ^ set(tot))
    known = sum(cob.values())
    out = []
    for p, n in tot.items():
        fb = n - born[p]
        assert fb >= 0
        white_bb = round(eth[p]["White"] * born[p] / n)
        parts = {"Barbados (white)": white_bb, "Barbados (not white)": born[p] - white_bb}
        raw = {k: fb * v / known for k, v in cob.items()}
        cnt = {k: int(v) for k, v in raw.items()}
        for k in sorted(raw, key=lambda k: raw[k] - cnt[k], reverse=True)[:fb - sum(cnt.values())]:
            cnt[k] += 1
        parts.update(cnt)
        assert sum(parts.values()) == n, p
        for k, v in parts.items():
            if v:
                bb2010.resolve(k)
                out.append(dict(geo_id=ids[p], geo_level="parish", geo_name=p,
                                source_category=k, count=v, tier="derived", year=2010))
        print(f"{ids[p]} {p:<14} {n:>7,}  foreign-born {fb:>6,}  white {eth[p]['White']:>5,}  "
              f"white Barbadian-born {white_bb:>5,}")
    df = pd.DataFrame(out)
    assert df["count"].sum() == TOTAL
    df.to_csv(OUT, index=False, encoding="utf-8")
    nat = df.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print(f"wrote {OUT}: {len(df)} rows, {df['geo_id'].nunique()} parishes, {TOTAL:,} people")
    for k, v in nat.head(12).items():
        print(f"  {k:<32} {v:>8,}  {v / TOTAL:6.2%}")


if __name__ == "__main__":
    build()
