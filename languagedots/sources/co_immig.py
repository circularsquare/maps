"""Colombia, CNPV 2018 (DANE): people born abroad by country of birth, per municipio ->
data/normalized/co_immig.csv (unit = MUPIO code, iso, count). Read by countries/co.py, which
turns them into languages with sources/latam_immig.py (record: sources/co.md, "Immigrant
languages").

    python sources/co_immig.py [--fetch]

THE SOURCE is DANE's REDATAM server (base CNPVBASE4V2, the full person file; route in
sources/co_cnpv.py), PERSONA.PA3_PAIS_NAC "Pais de nacimiento", asked when PA_LUG_NAC = 3 (born
in another country, 963,492 people). Its labels carry the ISO codes ("862_Venezuela_VE_VEN"), so
no name table is needed. A defined variable may hold about 250 categories, so the recode keeps
every country with 30 or more people (its own value) and folds the rest into one category,
split back at their national shares; people with no country (PA_LUG_NAC = 3 but no valid code)
are spread over the municipio's known countries.

CHECKS: municipios' categories sum to their Total; municipios sum, per category, to the national
FREQUENCY; 1,122 municipios, the same as co.csv's; the foreign-born total equals PA_LUG_NAC's
"En otro pais" (963,492).
"""
import html
import re
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "sources"))
import co_cnpv as cc  # noqa: E402

NORM = HERE / "data" / "normalized"
FOREIGN = 963_492
KEEP_MIN = 30
_FREQ = "RUNDEF Job\n SELECTION ALL\nTABLE T1\n AS FREQUENCY\n OF PERSONA.PA3_PAIS_NAC\n"


def countries():
    """[(code, iso, national count)] from the labelled national frequency."""
    t = html.unescape((cc.RAW / "co_pais.htm").read_text(encoding="utf-8"))
    cells = [re.sub(r"<[^>]+>", "", c).replace("\xa0", " ").strip()
             for c in re.findall(r"<td[^>]*>(.*?)</td>", t, re.S)]
    out = []
    for i, c in enumerate(cells):
        m = re.fullmatch(r"(\d+)_(.+)_([A-Z]{2})_([A-Z]{3})", c)
        if m:
            out.append((int(m.group(1)), m.group(3), cc._num(cells[i + 1])))
    return out


def _program(cs):
    keep = [c for c, _, n in cs if n >= KEEP_MIN]
    lines = ["RUNDEF Job", " SELECTION ALL", "DEFINE PERSONA.X", " AS SWITCH"]
    for i, code in enumerate(keep, start=1):
        lines += [f" INCASE PERSONA.PA_LUG_NAC = 3 AND PERSONA.PA3_PAIS_NAC = {code}",
                  f"  ASSIGN {i}"]
    small, nocode, native = len(keep) + 1, len(keep) + 2, len(keep) + 3
    for code in [c for c, _, n in cs if n < KEEP_MIN]:
        lines += [f" INCASE PERSONA.PA_LUG_NAC = 3 AND PERSONA.PA3_PAIS_NAC = {code}",
                  f"  ASSIGN {small}"]
    lines += [" INCASE PERSONA.PA_LUG_NAC = 3", f"  ASSIGN {nocode}",
              f" DEFAULT {native}", " TYPE INTEGER", f" RANGE 1-{native}",
              "TABLE T1", " AS AREALIST", " OF MUPIO, PERSONA.X", ""]
    return "\n".join(lines), keep


def fetch():
    cc.PROGRAMS = {"co_pais": _FREQ}
    cc.fetch()
    cc.PROGRAMS = {"co_mupio_pais": _program(countries())[0]}
    cc.fetch()


def main():
    if "--fetch" in sys.argv:
        fetch()
    ok = True

    def check(c, m):
        nonlocal ok
        print(("ok    " if c else "FAIL  ") + m)
        ok &= bool(c)
    cs = countries()
    _, keep = _program(cs)
    small, nocode, native = len(keep) + 1, len(keep) + 2, len(keep) + 3
    data, totals = cc.read_arealist("co_mupio_pais")
    data = {"CO" + m: r for m, r in data.items()}
    totals = {"CO" + m: v for m, v in totals.items()}
    check(all(sum(r.values()) == totals[m] for m, r in data.items()),
          f"{len(data)} municipios; categories sum to their Total")
    check(len(data) == cc.N_MUNICIPIOS and sum(totals.values()) == cc.POPULATION,
          f"1,122 municipios summing to {sum(totals.values()):,}")
    nat = {c: n for c, _, n in cs}
    sums = {v: sum(r.get(v, 0) for r in data.values()) for v in range(1, native + 1)}
    mism = [(c, sums[i], nat[c]) for i, c in enumerate(keep, start=1) if sums[i] != nat[c]]
    check(not mism, f"per kept country, municipios sum to the national FREQUENCY ({mism[:3]})")
    small_nat = sum(n for c, _, n in cs if n < KEEP_MIN)
    check(sums[small] == small_nat, f"folded small countries {sums[small]:,} == {small_nat:,}")
    fb = sum(v for k, v in sums.items() if k != native)
    check(fb == FOREIGN, f"foreign-born {fb:,} (no valid country {sums[nocode]:,})")
    base = pd.read_csv(NORM / "co.csv", dtype={"geo_id": str})
    check(set(data) == set(base["geo_id"]), "the same municipios as co.csv")
    if not ok:
        raise SystemExit("checks failed")
    iso = {c: i for c, i, _ in cs}
    small_mix = {iso[c]: n / small_nat for c, _, n in cs if n < KEEP_MIN}
    rows = []
    for m, r in data.items():
        k = sum(r.get(v, 0) for v in range(1, small + 1))
        f = 1 + r.get(nocode, 0) / k if k else 1
        for i, code in enumerate(keep, start=1):
            if r.get(i):
                rows.append((m, iso[code], r[i] * f))
        if r.get(small):
            rows += [(m, j, r[small] * f * s) for j, s in small_mix.items()]
    df = pd.DataFrame(rows, columns=["unit", "iso", "count"])
    df = df.groupby(["unit", "iso"], as_index=False)["count"].sum()
    df.to_csv(NORM / "co_immig.csv", index=False)
    print(f"wrote co_immig.csv ({len(df):,} rows, {df['count'].sum():,.0f} people)")


if __name__ == "__main__":
    main()
