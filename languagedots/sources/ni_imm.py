"""Nicaragua, Censo 2005: country of birth per municipio, for the immigrant languages
countries/ni.py had drawn as Spanish. Session edd42a8c-latn, 2026-10-05; the record is
sources/ni.md, "Immigrant languages"; the shared rule is sources/latam_immig.py.

    python sources/ni_imm.py [--fetch]

Writes data/normalized/ni_imm.csv: geo_id (INIDE municipio code), origin (ISO alpha-2, or
"US_U18" for the US-born under 18), count, for countries where Spanish is not the main language.

THE VARIABLES (VIVPOB05 on INIDE's REDATAM, the base sources/ni_censo.py reads): P09A, where the
mother lived at the birth (3 = another country, 34,693 people) and P09B, the code of that
municipio or country. P09B prints names; a copy of it as a plain integer prints the codes, and
the two frequencies (universe P09A = 3) pair row by row with equal counts. A derived PAISX
numbers the non-Spanish-speaking countries 1..k, the US-born split by P03 age (under 18, 18+).
CHECKS: 153 municipios; per country, municipio sums equal the national frequency; the zero
column plus the countries equals 5,142,098.
"""
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import latam_immig  # noqa: E402
import ni_censo as nc  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_H = "RUNDEF Job\n SELECTION ALL\n"
_U = " UNIVERSE PERS05.P09A = 3\n"
N_MUN = 153
POP = 5_142_098
LABEL_ISO = {
    "Antigua y Barbuda": "AG", "Bahamas": "BS", "Barbados": "BB", "Brasil": "BR", "Belice": "BZ",
    "Canadá": "CA", "Dominica": "DM", "Estados Unidos": "US", "Granada": "GD", "Guyana": "GY",
    "Haití": "HT", "Jamaica": "JM", "Sn. Cristóbal y Navis": "KN",
    "Sn. Vicente y Granadinas": "VC", "Suriname": "SR", "Trinidad y Tobago": "TT",
    "Afganistán": "AF", "Arabia Saudi": "SA", "Azerbaiyan": "AZ", "Bangladesh": "BD",
    "Camboya": "KH", "Corea Del Sur": "KR", "Corea Del Norte": "KP", "China": "CN",
    "Emiratos Árabes Unidos": "AE", "Filipinas": "PH", "Georgia": "GE", "India": "IN",
    "Indonesia": "ID", "Iraq": "IQ", "Irán": "IR", "Israel": "IL", "Japón": "JP",
    "Jordania": "JO", "Kazakistan": "KZ", "Kuwait": "KW", "Laos": "LA", "Líbano": "LB",
    "Malasia Federación De": "MY", "Maldivas": "MV", "Nepal": "NP", "Pakistán": "PK",
    "Singapur": "SG", "Siria": "SY", "Thailandia": "TH", "Taiwan": "TW", "Turquia": "TR",
    "Uzbekistan": "UZ", "Vietnan": "VN", "Albania": "AL", "Alemania": "DE", "Austria": "AT",
    "Bielorrusia": "BY", "Bélgica": "BE", "Bulgaria": "BG", "Croacia": "HR", "Chipre": "CY",
    "Dinamarca": "DK", "Eslovaquia": "SK", "Finlandia": "FI", "Francia": "FR", "Grecia": "GR",
    "Hungría": "HU", "Irlanda": "IE", "Islandia": "IS", "Italia": "IT", "Luxemburgo": "LU",
    "Moldova": "MD", "Noruega": "NO", "Holanda": "NL", "Polonia": "PL", "Portugal": "PT",
    "Reino Unido": "GB", "Republica Checa": "CZ", "Rusia": "RU", "San Marino": "SM",
    "Suecia": "SE", "Suiza": "CH", "Ucrania": "UA", "Australia": "AU", "Fiji": "FJ",
    "Nueva Zelanda": "NZ", "Angola": "AO", "Botswana": "BW", "Camerún": "CM", "Egipto": "EG",
    "Libia": "LY", "Madagascar": "MG", "Mozambique": "MZ", "Senegal": "SN", "Seychelles": "SC",
    "Sudáfrica": "ZA", "Túnez": "TN", "Zimbabwe": "ZW",
}
# Spanish-speaking origins, regional remainders ("Otros paises de ...") and "Ignorado" stay
# Spanish; any other label stops the script
SPANISH_LABELS = {
    "Argentina", "Bolivia", "Chile", "Colombia", "Costa Rica", "Cuba", "Ecuador", "El Salvador",
    "Guatemala", "Honduras", "México", "Panamá", "Paraguay", "Perú", "Puerto Rico",
    "Republica Dominicana", "Uruguay", "Venezuela", "España", "Otros Países de América",
    "Otros Países de Asia", "Otros Países de Europa", "Otros Países de África", "Ignorado",
}
BASE_PROGRAMS = {
    "nat_paislab": _H + _U + "TABLE T\n AS FREQUENCY\n OF PERS05.P09B\n",
    "nat_paiscod": _H + _U + "DEFINE PERS05.PC\n AS PERS05.P09B\n TYPE INTEGER\n"
                                " RANGE 0-9999\nTABLE T\n AS FREQUENCY\n OF PERS05.PC\n",
}


def _freq(name):
    out = []
    for r in nc._rows(name):
        if len(r) == 4 and r[1] != "Casos" and r[0] != "Total" and "%" in r[2]:
            out.append((r[0], int(r[1].replace(" ", "").replace(",", ""))))
    return out


def countries():
    """[(code, label, iso or None, count)], paired by order and count."""
    a, b = _freq("nat_paiscod"), _freq("nat_paislab")
    assert len(a) == len(b), (len(a), len(b))
    out = []
    for (c, n), (lab, m) in zip(a, b):
        assert n == m, (c, lab, n, m)
        if lab in LABEL_ISO:
            iso = LABEL_ISO[lab]
            assert iso not in latam_immig.HISPANIC, lab
        elif lab in SPANISH_LABELS:
            iso = None
        else:
            raise SystemExit(f"ni_imm: label {lab!r} not planned")
        out.append((int(c), lab, iso, n))
    return out


def cols():
    out = []
    for c, lab, iso, n in countries():
        if iso == "US":
            out += [(c, "US", "adult"), (c, "US_U18", "u18")]
        elif iso:
            out.append((c, iso, None))
    return out


def program():
    lines = ["DEFINE PERS05.PAISX\n AS SWITCH\n"]
    for i, (c, _, age) in enumerate(cols(), 1):
        cond = f"PERS05.P09A = 3 AND PERS05.P09B = {c}"
        if age == "adult":
            cond += " AND PERS05.P03 >= 18"
        elif age == "u18":
            cond += " AND PERS05.P03 < 18"
        lines.append(f"  INCASE {cond}\n   ASSIGN {i}\n")
    lines.append(f"  DEFAULT 0\n TYPE INTEGER\n RANGE 0-{len(cols())}\n")
    return _H + "".join(lines) + "TABLE T\n AS AREALIST\n OF MUN05, PERS05.PAISX\n"


def _run(name, prog):
    dest = nc.RAW / f"ni_{name}.htm"
    if dest.exists() and dest.stat().st_size > 1_000:
        return
    print("RUN", name)
    body = nc._post(prog)
    if "<table" not in body.lower():
        raise SystemExit(f"{name}: no table\n{body[:600]}")
    dest.write_text(body, encoding="utf-8")
    dest.with_suffix(".program.txt").write_text(prog, encoding="utf-8")
    print(f"  {dest.stat().st_size:,} bytes")


def fetch():
    nc.RAW.mkdir(parents=True, exist_ok=True)
    for name, p in BASE_PROGRAMS.items():
        _run(name, p)
    _run("mun_paisx", program())


def main():
    if "--fetch" in sys.argv:
        fetch()
    cl = cols()
    nat = {c: n for c, _, _, n in countries()}
    data, _ = nc._areal("mun_paisx", N_MUN)
    tot = {}
    for v in data.values():
        for k, n in v.items():
            tot[int(k)] = tot.get(int(k), 0) + n
    assert sum(tot.values()) == POP, sum(tot.values())
    for c in {c for c, _, _ in cl}:
        got = sum(tot.get(i, 0) for i, (cc, _, _) in enumerate(cl, 1) if cc == c)
        assert got == nat[c], (c, got, nat[c])
    rows = []
    for geo, v in data.items():
        for k, n in v.items():
            if int(k) and n:
                rows.append((geo, cl[int(k) - 1][1], n))
    df = pd.DataFrame(rows, columns=["geo_id", "origin", "count"])
    df.to_csv(nc.NORM / "ni_imm.csv", index=False, encoding="utf-8")
    s = df.groupby("origin")["count"].sum().sort_values(ascending=False)
    print(f"  checks pass: {N_MUN} municipios; {len(nat)} birthplaces paired; every drawn "
          f"country's municipio sums equal the national frequency; all {POP:,} once")
    print("  " + ", ".join(f"{k} {v:,}" for k, v in s.head(15).items()))


if __name__ == "__main__":
    main()
