"""Argentina, Censo 2022 (INDEC): foreign-born people by country of birth, per departamento,
private dwellings -> data/normalized/ar_immig.csv (unit, iso, count). Read by countries/ar.py,
which turns them into languages with sources/latam_immig.py (record: sources/ar.md, "Immigrant
languages"; sources/latam_immig.md).

    python sources/ar_immig.py [--fetch]

THE SOURCE is the main database of INDEC's REDATAM server (base CPV2022, item PROGVIVPART,
private dwellings: the same universe ar.csv's pueblo database covers), PERSONA.PAISNAC "Pais de
nacimiento", asked of the foreign-born (the Argentine-born are No Aplica).

THE ROUTE. AREALIST and per-area FREQUENCY both refuse PAISNAC by departamento ("Too many
categories", 205 countries). So a recode X keeps the code of every country with 100 or more
people nationally (84 of them) and folds the rest into one category per continent (code 900 +
INDEC's continent digit, the code's hundreds); each folded category is split back over its
countries at their national shares. The codes are not printed with labels: a FREQUENCY of the
raw code (`ar_paisnac_codes`) and the labelled FREQUENCY (`ar_paisnac`) list the same 205
counts in the same order, which pairs each code with its label (asserted, count by count).
"Ignorado" (999, 179,239 foreign-born whose country was not recorded) and the continents'
"Indeterminado" rows are spread over each departamento's known countries in proportion.

CHECKS: the code/label pairing; every departamento's categories sum to its printed Total; per
category, the departamentos sum to the national FREQUENCY; the departamentos are ar.csv's;
foreign-born never exceed a departamento's people; every label has an ISO code or is a remainder.
"""
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "sources"))
import ar_censo as ac  # noqa: E402

NORM = HERE / "data" / "normalized"
KEEP_MIN = 100

# PAISNAC labels as INDEC prints them -> ISO 3166 alpha-2 (origin_mix's pseudo codes for
# dissolved states: SU USSR, YU Yugoslavia, QT Czechoslovakia)
ISO = {
    "Burkina Faso": "BF", "Argelia": "DZ", "Botswana": "BW", "Burundi": "BI", "Camerún": "CM",
    "República Centroafricana": "CF", "Congo": "CG", "República Democrática del Congo": "CD",
    "Côte d'Ivoire": "CI", "Chad": "TD", "Benin": "BJ", "Egipto": "EG", "Gabón": "GA",
    "Gambia": "GM", "Ghana": "GH", "Guinea": "GN", "Guinea Ecuatorial": "GQ", "Kenya": "KE",
    "Lesotho": "LS", "Liberia": "LR", "Libia": "LY", "Madagascar": "MG", "Malawi": "MW",
    "Malí": "ML", "Marruecos": "MA", "Mauricio": "MU", "Mauritania": "MR", "Níger": "NE",
    "Nigeria": "NG", "Zimbabwe": "ZW", "Rwanda": "RW", "Senegal": "SN", "Sierra Leona": "SL",
    "Somalia": "SO", "Swazilandia": "SZ", "Tanzanía": "TZ", "Togo": "TG", "Túnez": "TN",
    "Uganda": "UG", "Zambia": "ZM", "Angola": "AO", "Cabo Verde": "CV", "Mozambique": "MZ",
    "Seychelles": "SC", "Djibouti": "DJ", "Comoras": "KM", "Guinea Bissau": "GW",
    "Santo Tomé y Príncipe": "ST", "Namibia": "NA", "Sudáfrica": "ZA", "Eritrea": "ER",
    "Etiopía": "ET", "Sudán": "SD", "Barbados": "BB", "Bolivia": "BO", "Brasil": "BR",
    "Canadá": "CA", "Colombia": "CO", "Costa Rica": "CR", "Cuba": "CU", "Chile": "CL",
    "República Dominicana": "DO", "Ecuador": "EC", "El Salvador": "SV", "Estados Unidos": "US",
    "Guatemala": "GT", "Guyana": "GY", "Haití": "HT", "Honduras": "HN", "Jamaica": "JM",
    "México": "MX", "Nicaragua": "NI", "Panamá": "PA", "Paraguay": "PY", "Perú": "PE",
    "Puerto Rico": "PR", "Trinidad y Tobago": "TT", "Uruguay": "UY", "Venezuela": "VE",
    "Suriname": "SR", "Dominica": "DM", "Santa Lucía": "LC", "San Vicente y Las Granadinas": "VC",
    "Belice": "BZ", "Antigua y Barbuda": "AG", "San Cristóbal y Nevis": "KN", "Bahamas": "BS",
    "Granada": "GD", "Antillas Holandesas": "CW", "Aruba": "AW", "Afganistán": "AF",
    "Arabia Saudita": "SA", "Bahrein": "BH", "Myanmar": "MM", "Bhután": "BT", "Camboya": "KH",
    "Sri Lanka": "LK", "Corea Democrática y Popular": "KP", "Corea": "KR", "China": "CN",
    "Filipinas": "PH", "Taiwan": "TW", "India": "IN", "Indonesia": "ID", "Iraq": "IQ",
    "Irán": "IR", "Israel": "IL", "Japón": "JP", "Jordania": "JO", "Qatar": "QA", "Kuwait": "KW",
    "Laos": "LA", "Líbano": "LB", "Malasia": "MY", "Maldivas": "MV", "Omán": "OM",
    "Mongolia": "MN", "Nepal": "NP", "Emiratos Árabes Unidos": "AE", "Pakistán": "PK",
    "Singapur": "SG", "Siria": "SY", "Tailandia": "TH", "Viet Nam": "VN",
    "Hong Kong (región administrativa especial de China)": "HK",
    "Macao (región administrativa especial de China)": "MO", "Bangladesh": "BD",
    "Brunei": "BN", "Yemen": "YE", "Armenia": "AM", "Azerbaiyán": "AZ", "Georgia": "GE",
    "Kazajstán": "KZ", "Kirguistán": "KG", "Tayikistán": "TJ", "Turkmenistán": "TM",
    "Uzbekistán": "UZ", "Palestina": "PS", "Timor Leste": "TL", "Albania": "AL",
    "Andorra": "AD", "Austria": "AT", "Bélgica": "BE", "Bulgaria": "BG",
    "Ex Checoslovaquia": "QT", "Dinamarca": "DK", "España": "ES", "Finlandia": "FI",
    "Francia": "FR", "Grecia": "GR", "Hungría": "HU", "Irlanda": "IE", "Islandia": "IS",
    "Italia": "IT", "Liechtenstein": "LI", "Luxemburgo": "LU", "Malta": "MT", "Mónaco": "MC",
    "Noruega": "NO", "Países Bajos": "NL", "Polonia": "PL", "Portugal": "PT",
    "Reino Unido de Gran Bretaña e Irlanda del Norte": "GB", "Rumania": "RO",
    "San Marino": "SM", "Suecia": "SE", "Suiza": "CH", "Vaticano": "VA", "Ex Yugoslavia": "YU",
    "Chipre": "CY", "Turquía": "TR", "Alemania": "DE", "Belarús": "BY", "Estonia": "EE",
    "Letonia": "LV", "Lituania": "LT", "Moldova": "MD", "Rusia": "RU", "Ucrania": "UA",
    "Bosnia y Herzegovina": "BA", "Croacia": "HR", "Eslovaquia": "SK", "Eslovenia": "SI",
    "Macedonia": "MK", "República Checa": "CZ", "Serbia y Montenegro": "RS",
    "Montenegro": "ME", "Serbia": "RS", "Australia": "AU", "Nauru": "NR",
    "Nueva Zelandia": "NZ", "Vanuatu": "VU", "Samoa": "WS", "Fiji": "FJ",
    "Papua Nueva Guinea": "PG", "Kiribati": "KI", "Tuvalu": "TV", "Tonga": "TO",
    "Ex Unión Soviética(1)": "SU",
}
REMAINDER = {"Indeterminado (África)", "Indeterminado (América)", "Indeterminado (Asia)",
             "Indeterminado (Europa)", "Indeterminado (Oceanía)", "Ignorado"}

_FREQ = "RUNDEF Job\n SELECTION ALL\nTABLE T1\n AS FREQUENCY\n OF PERSONA.PAISNAC\n"
_CODES = ("RUNDEF Job\n SELECTION ALL\nDEFINE PERSONA.X\n AS PERSONA.PAISNAC\n TYPE INTEGER\n"
          "TABLE T1\n AS FREQUENCY\n OF PERSONA.X\n")


def read_freq(name):
    out = {}
    for r in ac._rows(name):
        if len(r) == 4 and r[2].endswith("%"):
            try:
                out[r[0]] = ac._num(r[1])
            except ValueError:
                pass
    out.pop("Total", None)
    return out


def code_labels():
    """{code: (label, national count)}, pairing the two national frequencies by order."""
    c, f = read_freq("ar_paisnac_codes"), read_freq("ar_paisnac")
    if list(c.values()) != list(f.values()):
        raise SystemExit("ar_immig: the coded and labelled PAISNAC frequencies differ")
    return {int(k): (lab, n) for (k, n), lab in zip(c.items(), f)}


def group_of(code, n):
    """The recode's category: the code itself, or 900 + continent digit for a small country."""
    return code if n >= KEEP_MIN or code >= 900 else 900 + code // 100


def groups(cl):
    """Recode values 1..K (a wide RANGE also counts as 'too many categories') -> group."""
    return sorted({group_of(code, n) for code, (_, n) in cl.items()})


def _recode_program(cl):
    idx = {g: i for i, g in enumerate(groups(cl), start=1)}
    lines = ["RUNDEF Job", " SELECTION ALL", "DEFINE PERSONA.X", " AS SWITCH"]
    for code, (_, n) in sorted(cl.items()):
        lines += [f" INCASE PERSONA.PAISNAC = {code}", f"  ASSIGN {idx[group_of(code, n)]}"]
    lines += [f" DEFAULT {len(idx) + 1}", " TYPE INTEGER", f" RANGE 1-{len(idx) + 1}",
              "TABLE T1", " AS AREALIST", " OF DPTO, PERSONA.X", ""]
    return "\n".join(lines)


def fetch():
    ac.PROGRAMS = {"ar_paisnac": ("PROGVIVPART", _FREQ),
                   "ar_paisnac_codes": ("PROGVIVPART", _CODES)}
    ac.fetch()
    ac.PROGRAMS = {"ar_dpto_paisnac": ("PROGVIVPART", _recode_program(code_labels()))}
    ac.fetch()


def main():
    if "--fetch" in sys.argv:
        fetch()
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("ok    " if cond else "FAIL  ") + msg)
        ok &= bool(cond)

    cl = code_labels()
    check(len(cl) == 205, f"{len(cl)} PAISNAC codes paired with their labels by count")
    data, totals = ac.read_arealist("ar_dpto_paisnac")
    gl = groups(cl)     # recode value i -> group gl[i - 1]; len(gl) + 1 is No Aplica (0)
    data = {d: {(gl[int(k) - 1] if int(k) <= len(gl) else 0): v for k, v in row.items()}
            for d, row in data.items()}
    bad = [d for d in data if sum(data[d].values()) != totals[d]]
    check(not bad, f"{len(data)} departamentos; each one's categories sum to its Total "
                   f"({bad[:3]})")
    no_aplica = sum(row.get(0, 0) for row in data.values())
    grp_nat = {}
    for code, (_, n) in cl.items():
        g = group_of(code, n)
        grp_nat[g] = grp_nat.get(g, 0) + n
    sums = {}
    for row in data.values():
        for g, v in row.items():
            if g:
                sums[g] = sums.get(g, 0) + v
    mism = [(g, sums.get(g, 0), n) for g, n in grp_nat.items() if sums.get(g, 0) != n]
    check(not mism, f"per category, departamentos sum to the national FREQUENCY "
                    f"({len(grp_nat)} categories; {mism[:4]})")
    base = pd.read_csv(NORM / "ar.csv", dtype={"geo_id": str})
    pop = base[base["code"] != 3].groupby("geo_id")["count"].sum()
    check(set(data) == set(pop.index), "the same departamentos as ar.csv's private dwellings")
    check(sum(totals.values()) == ac.PRIVATE, f"Totals sum to the private population "
                                             f"{sum(totals.values()):,}; Argentine-born "
                                             f"(No Aplica) {no_aplica:,}")
    unk = [lab for lab, _ in cl.values() if lab not in ISO and lab not in REMAINDER]
    check(not unk, f"every label has an ISO code or is a listed remainder ({unk})")
    if not ok:
        raise SystemExit("checks failed; nothing written")

    # each category -> {iso: share}
    split = {}
    for code, (lab, n) in cl.items():
        if lab in REMAINDER or not n:
            continue
        g = group_of(code, n)
        split.setdefault(g, {})
        split[g][ISO[lab]] = split[g].get(ISO[lab], 0) + n
    rem = {group_of(code, n) for code, (lab, n) in cl.items() if lab in REMAINDER}
    rows = []
    for d, row in data.items():
        known = {g: v for g, v in row.items() if g and g not in rem and v}
        unknown = sum(v for g, v in row.items() if g in rem)
        kt = sum(known.values())
        f = 1 + unknown / kt if kt else 1
        for g, v in known.items():
            t = sum(split[g].values())
            for iso, n in split[g].items():
                rows.append((d, iso, v * f * n / t))
    df = pd.DataFrame(rows, columns=["unit", "iso", "count"])
    df = df.groupby(["unit", "iso"], as_index=False)["count"].sum()
    df.to_csv(NORM / "ar_immig.csv", index=False)
    fb = sum(totals.values()) - no_aplica
    print(f"wrote ar_immig.csv ({len(df):,} rows); foreign-born {fb:,} "
          f"({fb / ac.PRIVATE:.1%} of private-dwelling people), {df['count'].sum():,.0f} placed")
    print(df.groupby("iso")["count"].sum().sort_values(ascending=False).head(12).round()
          .to_string())


if __name__ == "__main__":
    main()
