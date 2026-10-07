"""Costa Rica, Censo 2011: country of birth and ethnic self-identification per distrito, for the
languages countries/cr.py had drawn as Spanish. Session edd42a8c-latn, 2026-10-05; the record is
sources/cr.md, "Immigrant and Creole languages"; the shared rule is sources/latam_immig.py.

    python sources/cr_imm.py [--fetch]

Writes data/normalized/cr_imm.csv: geo_id (distrito), origin, count. origin is an ISO alpha-2
country of birth (non-Spanish-speaking countries only: the Spanish-speaking ones stay on
Spanish and are not needed), or "LIMON_AFRO": people in Limon province who identify as
"Negro(a) o afrodescendiente", drawn on Limon Creole English (sources/cr.md has the reasons).

Same server and base as sources/cr_censo.py (INEC REDATAM, CP2011). Variables: P05C LUGPA,
country of birth, ISO 3166 numeric (INEC's own 822-824 for Scotland, Wales, England), asked of
the foreign-born only; P10 ETNIA, ethnic-racial self-identification, asked of everyone not
self-identified as indigenous (P07). An AREALIST of all ~160 countries returns HTTP 500, so a
derived PAISX keeps each non-Spanish-speaking country's own code and puts everyone else on 0,
run in chunks of 35 countries.
CHECKS: 472 distritos in each AREALIST; per country, distrito sums equal the national
frequency of the raw code; ETNIA's distritos sum to its national frequency (4,197,569 asked,
104,143 No Aplica = the self-identified indigenous).
"""
import re
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cr_censo as cc  # noqa: E402
import iso_numeric  # noqa: E402
import latam_immig  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

INEC_GB = {822: "GB", 823: "GB", 824: "GB"}
_H = "RUNDEF Job\n SELECTION ALL\n"
BASE_PROGRAMS = {
    "nat_paiscode": _H + "DEFINE POBLACIO.PC\n AS POBLACIO.LUGPA\n TYPE INTEGER\n RANGE 1-999\n"
                         "TABLE T\n AS FREQUENCY\n OF POBLACIO.PC\n",
    "dist_etnia": _H + "TABLE T\n AS AREALIST\n OF DISTRITO, POBLACIO.ETNIA\n",
    "nat_etnia": _H + "TABLE T\n AS FREQUENCY\n OF POBLACIO.ETNIA\n",
    "nat_pais": _H + "TABLE T\n AS FREQUENCY\n OF POBLACIO.LUGPA\n",
}
AFRO = "Negro(a) o afrodescendiente"


def alpha2(code):
    return INEC_GB.get(code) or iso_numeric.alpha2(code)


def nat_codes():
    out = {}
    for r in cc._rows("nat_paiscode"):
        if re.fullmatch(r"\d+", r[0]) and len(r) >= 2:
            out[int(r[0])] = cc._int(r[1])
    return out


def paisx_program(codes):
    lines = ["DEFINE POBLACIO.PAISX\n AS SWITCH\n"]
    for i, c in enumerate(codes, 1):     # compact 1..k: AREALIST makes a column per value
        lines.append(f"  INCASE POBLACIO.LUGPA = {c}\n   ASSIGN {i}\n")
    lines.append(f"  DEFAULT 0\n TYPE INTEGER\n RANGE 0-{len(codes)}\n")
    return _H + "".join(lines) + "TABLE T\n AS AREALIST\n OF DISTRITO, POBLACIO.PAISX\n"


def _run(name, program):
    dest = cc.RAW / f"cr_{name}.htm"
    if dest.exists() and dest.stat().st_size > 1_000:
        return
    print("RUN", name)
    body = cc._post(program)
    if "<table" not in body.lower():
        raise SystemExit(f"{name}: REDATAM returned no table\n{body[:600]}")
    dest.write_text(body, encoding="utf-8")
    (cc.RAW / f"cr_{name}.txt").write_text(program, encoding="utf-8")
    print(f"  {dest.stat().st_size:,} bytes")
    time.sleep(1)


def non_hispanic_codes():
    nat = nat_codes()
    miss = [c for c in nat if alpha2(c) is None]
    if miss:
        raise SystemExit(f"country codes with no ISO alpha-2: {miss}")
    return sorted(c for c in nat if alpha2(c) not in latam_immig.HISPANIC)


CHUNK = 35     # columns per AREALIST: ~136 at once answers HTTP 500


def chunks():
    c = non_hispanic_codes()
    return [c[i:i + CHUNK] for i in range(0, len(c), CHUNK)]


def fetch():
    cc.RAW.mkdir(parents=True, exist_ok=True)
    for name, p in BASE_PROGRAMS.items():
        _run(name, p)
    for i, chunk in enumerate(chunks()):
        _run(f"dist_paisx{i}", paisx_program(chunk))


def main():
    if "--fetch" in sys.argv:
        fetch()
    nat = nat_codes()
    codes = non_hispanic_codes()
    px = {}
    for i, chunk in enumerate(chunks()):
        part, cols = cc._areal(f"dist_paisx{i}", cc.N_DIST)
        got = {chunk[int(c) - 1] if int(c) else 0: sum(v.get(c, 0) for v in part.values())
               for c in cols}
        bad = [c for c in chunk if got.get(c, 0) != nat[c]]
        assert not bad, f"PAISX against the national frequency: {bad[:5]}"
        assert set(got) - {0} == set(chunk), sorted(set(got) ^ set(chunk))
        assert got.get(0, 0) + sum(nat[c] for c in chunk) == cc.CENSUS_POPULATION, i
        for d, v in part.items():
            px.setdefault(d, {}).update({str(chunk[int(c) - 1]): n for c, n in v.items()
                                         if c != "0"})
    print(f"  OK {len(codes)} non-Spanish-speaking birth countries, {sum(nat[c] for c in codes):,} "
          f"people; distrito sums equal the national frequency for every one")
    et, ecols = cc._areal("dist_etnia", cc.N_DIST)
    natet = {r[0]: cc._int(r[1]) for r in cc._rows("nat_etnia") if len(r) == 4 and r[0] != "Total"
             and re.fullmatch(r"[\d ]+", r[1])}
    for c in ecols:
        s = sum(v.get(c, 0) for v in et.values())
        assert s == natet[c], (c, s, natet[c])
    print(f"  OK ETNIA: distritos sum to the national frequency on all {len(ecols)} answers")
    rows = []
    for d, v in px.items():
        for c, n in v.items():
            if int(c) and n:
                rows.append((d, alpha2(int(c)), n))
    for d, v in et.items():
        if d.startswith("7") and v.get(AFRO, 0):
            rows.append((d, "LIMON_AFRO", v[AFRO]))
    df = pd.DataFrame(rows, columns=["geo_id", "origin", "count"])
    df = df.groupby(["geo_id", "origin"], as_index=False)["count"].sum()
    df.to_csv(cc.NORM / "cr_imm.csv", index=False, encoding="utf-8")
    nat_o = df.groupby("origin")["count"].sum().sort_values(ascending=False)
    print(f"  wrote cr_imm.csv ({len(df):,} rows): " + ", ".join(
        f"{k} {v:,}" for k, v in nat_o.head(15).items()))


if __name__ == "__main__":
    main()
