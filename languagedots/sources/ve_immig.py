"""Venezuela, Censo 2011 (INE): people born abroad by country of birth, per parroquia ->
data/normalized/ve_immig.csv (geo_id, iso, count). Read by countries/ve.py, which turns them
into languages with sources/latam_immig.py (record: sources/ve.md, "Immigrant languages").

    python sources/ve_immig.py [--fetch]

THE SOURCE is INE's REDATAM server (base CPV2011, sources/ve_censo.py's route).
PERSONA.ENCUALPAIS "Pais de Nacimiento (Principales Paises)" names 18 countries and "Otro pais"
(27,192); AREALIST of it by PARROQUI. "Otro pais" is split at the national composition of its
countries from PERSONA.CODOTROPAI ("Entidad Federal y Pais de Nacimiento", 148 countries among
those people; `ve_otropais` counts them by code, `ve_codotropai` / `_codes` pair code and label
by two national frequencies of identical counts, asserted). People born abroad with no
ENCUALPAIS answer stay on Spanish (in ve.csv's 1002 row, not here).

CHECKS: parroquias' categories sum to their Total; per country they sum to the national
FREQUENCY; the parroquias are ve.csv's; foreign-born never exceed ve.csv's born-abroad row.
"""
import sys
import unicodedata
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "sources"))
sys.path.insert(0, str(HERE / "taxonomy"))
import ve_censo as vc  # noqa: E402

NORM = HERE / "data" / "normalized"
MAIN = {"Argentina": "AR", "Bolivia": "BO", "Brasil": "BR", "Chile": "CL",
        "China Continental": "CN", "Colombia": "CO", "Cuba": "CU", "Ecuador": "EC",
        "España": "ES", "Estados Unidos": "US", "Guyana": "GY", "Haití": "HT", "Italia": "IT",
        "Líbano": "LB", "Perú": "PE", "Portugal": "PT", "República  Dominicana": "DO",
        "Siria": "SY"}
OTHER = "Otro país"
EXTRA = {"Curazao y Bonaire": "CW", "Islas Caiman": "KY", "Martinica": "MQ",
         "San Vicente y Granadinas": "VC", "Reino Unido": "GB", "Yugoslavia": "YU",
         "Monte Negro (Antiguo Territorio Yugoslavia)": "ME",
         "Serbia(Antiguo Territorio Yugoslavia)": "RS", "Corea Del Norte": "KP",
         "Timor Oriental": "TL", "Vietnam": "VN", "Emiratos Arab Unidos": "AE",
         "Sahara Occidental": "EH"}
UNNAMED = {"Otros Países de América", "Otros Países de Europa", "Otros Países de Asia",
           "Otros Países de Africa"}      # 338 people: dropped from the split

PROGRAMS = {
    "ve_encualpais": "RUNDEF Job\n SELECTION ALL\nTABLE T1\n AS FREQUENCY\n OF PERSONA.ENCUALPAIS\n",
    "ve_parroquia_pais": "RUNDEF Job\n SELECTION ALL\nTABLE T1\n AS AREALIST\n"
                         " OF PARROQUI, PERSONA.ENCUALPAIS\n",
    "ve_otropais": "RUNDEF Job\n SELECTION ALL\nDEFINE PERSONA.X\n AS SWITCH\n"
                   " INCASE PERSONA.ENCUALPAIS = 19\n  ASSIGN PERSONA.CODOTROPAI\n DEFAULT 0\n"
                   " TYPE INTEGER\n RANGE 0-999\nTABLE T1\n AS FREQUENCY\n OF PERSONA.X\n",
    "ve_codotropai": "RUNDEF Job\n SELECTION ALL\nTABLE T1\n AS FREQUENCY\n OF PERSONA.CODOTROPAI\n",
    "ve_codotropai_codes": "RUNDEF Job\n SELECTION ALL\nDEFINE PERSONA.X\n AS PERSONA.CODOTROPAI\n"
                           " TYPE INTEGER\nTABLE T1\n AS FREQUENCY\n OF PERSONA.X\n",
}


def _n(s):
    return "".join(c for c in unicodedata.normalize("NFD", s.lower())
                   if unicodedata.category(c) != "Mn").strip()


def other_mix():
    """{iso: share} of the 'Otro pais' people, from CODOTROPAI."""
    import ar_immig
    import uy2011
    look = {_n(k): i for k, i in {**uy2011.COUNTRY, **ar_immig.ISO, **EXTRA}.items()}
    a, b = vc.read_frequency("ve_codotropai"), vc.read_frequency("ve_codotropai_codes")
    if [x[1] for x in a] != [x[1] for x in b]:
        raise SystemExit("ve_immig: coded and labelled CODOTROPAI frequencies differ")
    label = {c: lab for (lab, _), (c, _) in zip(a, b)}
    o = dict(vc.read_frequency("ve_otropais"))
    o.pop("0", None)
    out = {}
    for c, k in o.items():
        lab = label[c]
        if lab in UNNAMED:
            continue
        iso = look.get(_n(lab))
        if iso is None:
            raise SystemExit(f"ve_immig: no ISO code for {lab!r}")
        out[iso] = out.get(iso, 0) + k
    t = sum(out.values())
    return {k: v / t for k, v in out.items()}, sum(o.values())


def main():
    if "--fetch" in sys.argv:
        vc.PROGRAMS = PROGRAMS
        vc.fetch()
    ok = True

    def check(c, m):
        nonlocal ok
        print(("ok    " if c else "FAIL  ") + m)
        ok &= bool(c)
    nat = dict(vc.read_frequency("ve_encualpais"))
    rows = vc._rows("ve_parroquia_pais")
    header = next(r for r in rows if r[0] == "Código")
    cols = header[1:]
    data, totals = {}, {}
    for r in rows:
        if len(r) == len(cols) + 1 and r[0].isdigit() and len(r[0]) == 6:
            v = [vc._num(x) for x in r[1:]]
            data[r[0]] = dict(zip(cols[:-1], v[:-1]))
            totals[r[0]] = v[-1]
    check(cols[-1] == "Total" and set(cols[:-1]) == set(MAIN) | {OTHER},
          f"{len(data)} parroquias; columns are the 18 countries and Otro pais")
    check(all(sum(d.values()) == totals[p] for p, d in data.items()), "rows sum to their Total")
    mism = [c for c in cols[:-1] if sum(d[c] for d in data.values()) != nat[c]]
    check(not mism, f"per country, parroquias sum to the national FREQUENCY ({mism})")
    base = pd.read_csv(NORM / "ve.csv", dtype={"geo_id": str})
    ab = base[base["code"] == 1002].set_index("geo_id")["count"]
    ids = {"VE" + p for p in data}
    check(ids <= set(base["geo_id"]), f"parroquias are ve.csv's ({len(ids - set(base['geo_id']))} not)")
    over = [p for p in data if totals[p] > ab.get("VE" + p, 0)]
    check(not over, f"foreign-born never exceed ve.csv's born-abroad row ({over[:3]})")
    om, n_other = other_mix()
    check(n_other == nat[OTHER], f"Otro pais split over {len(om)} countries ({n_other:,})")
    if not ok:
        raise SystemExit("checks failed")
    out = []
    for p, d in data.items():
        for c, k in d.items():
            if not k:
                continue
            if c == OTHER:
                out += [("VE" + p, iso, k * s) for iso, s in om.items()]
            else:
                out.append(("VE" + p, MAIN[c], k))
    df = pd.DataFrame(out, columns=["geo_id", "iso", "count"])
    df = df.groupby(["geo_id", "iso"], as_index=False)["count"].sum()
    df.to_csv(NORM / "ve_immig.csv", index=False)
    print(f"wrote ve_immig.csv ({len(df):,} rows, {df['count'].sum():,.0f} people; born-abroad "
          f"row in ve.csv {ab.sum():,})")


if __name__ == "__main__":
    main()
