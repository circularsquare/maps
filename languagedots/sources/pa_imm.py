"""Panama, Censo 2023: country of birth per corregimiento, for the immigrant languages
countries/pa.py had folded into Spanish and MICS's unnamed "other". Session edd42a8c-latn,
2026-10-05; the record is sources/pa.md, "Immigrant languages"; the shared rule is
sources/latam_immig.py.

    python sources/pa_imm.py [--fetch]

Writes data/normalized/pa_imm.csv: geo_id (corregimiento), origin (ISO alpha-2), count, for
countries where Spanish is not the main language.

THE VARIABLES (LP2023 on INEC's REDATAM, the base sources/pa_censo.py reads): P_NACIO "where
the mother lived when (NAME) was born" (3 = another country, 249,476 people) and P_NACI_COD, its
code, which prints no labels. RP05_NACI ("lugar de nacimiento recodificado") prints the country
names, in the same code order: nat_code (FREQUENCY of P_NACI_COD, universe P_NACIO = 3) and
nat_label (FREQUENCY of RP05_NACI, same universe) are paired row by row, and every pair's count
must agree (189 of 189). A derived PAISX numbers the non-Spanish-speaking countries 1..k
(AREALIST makes a column per value of a variable's range), run in chunks of 35.
CHECKS: per country, corregimiento sums equal the national frequency; each chunk's zero
column plus its countries equals 4,064,780; 699 corregimientos.
"""
import re
import sys
import time
import urllib.parse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import latam_immig  # noqa: E402
import pa_censo as pc  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_H = "RUNDEF Job\n SELECTION ALL\n"
_U = " UNIVERSE PERSONA.P_NACIO = 3\n"
BASE_PROGRAMS = {
    "pa_nat_nacicod": _H + _U + "TABLE T\n AS FREQUENCY\n OF PERSONA.P_NACI_COD\n",
    "pa_nat_nacilab": _H + _U + "TABLE T\n AS FREQUENCY\n OF PERSONA.RP05_NACI\n",
}
CHUNK = 35
N_CORREG = 699

# RP05_NACI labels -> ISO alpha-2 (None: no country, drawn as Spanish with the Panamanian-born)
LABEL_ISO = {
    "GROENLANDIA": "GL", "CANADA": "CA", "ESTADOS UNIDOS DE AMERICA": "US", "BERMUDAS": "BM",
    "MEXICO": "MX", "GUATEMALA": "GT", "BELICE": "BZ", "EL SALVADOR": "SV", "HONDURAS": "HN",
    "NICARAGUA": "NI", "COSTA RICA": "CR", "CUBA": "CU", "REPUBLICA DOMINICANA": "DO",
    "HAITI": "HT", "ISLA PROVIDENCIA (COLOMBIA)": "CO", "ISLA SAN ANDRES (COLOMBIA)": "CO",
    "GRANADA": "GD", "BARBADOS": "BB", "BAHAMAS": "BS", "JAMAICA": "JM", "ANTIGUA Y BARBUDA": "AG",
    "TRINIDAD Y TOBAGO": "TT", "GUADALUPE Y DEPENDENCIAS": "GP", "MARTINICA": "MQ",
    "DOMINICA": "DM", "SANTA LUCIA": "LC", "SAN VICENTE Y LAS GRANADINAS": "VC",
    "SAN CRISTOBAL Y NIEVES": "KN", "ARUBA": "AW", "CURAZAO": "CW", "SAN MARTIN (PARTE SUR)": "SX",
    "ISLAS VIRGENES (NORTEAMERICANAS)": "VI", "PUERTO RICO": "PR",
    "OTRAS DE LAS INDIAS OCCIDENTALES BRITANICAS": "AG",
    "OTRAS DE LAS INDIAS OCCIDENTALES FRANCESAS": "GP",
    "OTRAS DE LAS INDIAS OCCIDENTALES HOLANDESAS": "CW",
    "COLOMBIA": "CO", "ISLAS GALAPAGOS": "EC", "ECUADOR": "EC", "VENEZUELA": "VE", "BRASIL": "BR",
    "URUGUAY": "UY", "ARGENTINA": "AR", "BOLIVIA": "BO", "PARAGUAY": "PY", "PERU": "PE",
    "CHILE": "CL", "GUYANA": "GY", "GUAYANA FRANCESA": "GF", "SURINAME": "SR",
    "ALBANIA": "AL", "ALEMANIA": "DE", "ANDORRA": "AD", "AUSTRIA": "AT", "ESLOVAQUIA": "SK",
    "REPUBLICA CHECA": "CZ", "BULGARIA": "BG", "DINAMARCA": "DK", "BELGICA": "BE",
    "LUXEMBURGO": "LU", "ESPAÑA": "ES", "ISLAS FEROE": "FO", "FINLANDIA": "FI", "FRANCIA": "FR",
    "GIBRALTAR": "GI", "GRECIA": "GR", "HUNGRIA": "HU", "IRLANDA (EIRE)": "IE", "ISLANDIA": "IS",
    "ITALIA": "IT", "PAISES BAJOS": "NL", "MONACO": "MC", "LIECHTENSTEIN": "LI", "NORUEGA": "NO",
    "POLONIA": "PL", "PORTUGAL": "PT", "REINO UNIDO": "GB", "RUMANIA": "RO", "SUECIA": "SE",
    "SUIZA": "CH", "BOSNIA Y HERZEGOVINA": "BA", "CROACIA": "HR", "ESLOVENIA": "SI",
    "REPUBLICA DE MACEDONIA": "MK", "REPUBLICA DE BELARUS": "BY", "ESTONIA": "EE",
    "LETONIA": "LV", "LITUANIA": "LT", "REPUBLICA DE MOLDOVA": "MD", "UCRANIA": "UA",
    "SERBIA": "RS", "MONTENEGRO": "ME", "KOSOVO": "XK", "OTROS DE EUROPA": None,
    "BRUNEI": "BN", "AFGANISTAN": "AF", "BIRMANIA/MYANMAR": "MM", "BAHREIN": "BH",
    "BANGLADESH": "BD", "CAMBOYA": "KH", "SRI LANKA": "LK", "COREA DEL NORTE": "KP",
    "COREA DEL SUR": "KR", "CHINA (CONTINENTAL)": "CN", "CHINA-TAIWAN (FORMOSA)": "TW",
    "CHIPRE": "CY", "MALASIA": "MY", "MONGOLIA": "MN", "FILIPINAS": "PH", "HONG KONG": "HK",
    "INDIA": "IN", "INDONESIA": "ID", "IRAK": "IQ", "IRAN": "IR", "ISRAEL": "IL", "JAPON": "JP",
    "JORDANIA": "JO", "KUWAIT": "KW", "EMIRATOS ARABES UNIDOS": "AE",
    "REPUBLICA DEMOCRATICA POPULAR DE LAO": "LA", "LIBANO": "LB", "MACAO": "MO", "NEPAL": "NP",
    "OMAN": "OM", "PALESTINA": "PS", "QATAR": "QA", "NUEVA GUINEA OCCIDENTAL": "ID",
    "PAKISTAN": "PK", "ARABIA SAUDITA": "SA", "SINGAPUR": "SG", "SIRIA": "SY", "TAILANDIA": "TH",
    "TURQUIA": "TR", "VIETNAM": "VN", "ARMENIA": "AM", "AZERBAIYAN": "AZ", "GEORGIA": "GE",
    "KAZAJISTAN (KAZAJSTAN)": "KZ", "KIRGUISTAN": "KG", "OTROS DE ASIA": None, "RUSIA": "RU",
    "GUINEA": "GN", "GUINEA ECUATORIAL": "GQ", "ANGOLA": "AO", "ARGELIA": "DZ", "BENIN": "BJ",
    "BOTSUANA": "BW", "CAMERUN": "CM", "COMORAS": "KM", "REPUBLICA DEL CONGO": "CG",
    "RUANDA": "RW", "CABO VERDE": "CV", "EGIPTO, REPUBLICA ARABE DE": "EG", "ETIOPIA": "ET",
    "GABON": "GA", "GHANA": "GH", "CHAD": "TD", "KENIA": "KE", "MADAGASCAR": "MG",
    "LIBERIA": "LR", "LIBIA": "LY", "MARRUECOS": "MA", "MAURICIO": "MU", "MOZAMBIQUE": "MZ",
    "SOMALIA": "SO", "NIGERIA": "NG", "UGANDA": "UG", "NAMIBIA": "NA", "ZIMBABWE": "ZW",
    "MAURITANIA": "MR", "SENEGAL": "SN", "SIERRA LEONA": "SL", "TOGO": "TG",
    "REPUBLICA UNIDA DE TANZANIA": "TZ", "TUNEZ": "TN", "SUDAFRICA, REPUBLICA DE": "ZA",
    "ZAMBIA": "ZM", "COSTA DE MARFIL": "CI", "REPUBLICA ARABE SAHARAUI DEMOCRATICA": "EH",
    "SUDAN DEL SUR": "SS", "OTROS DE AFRICA": None, "AUSTRALIA": "AU", "FIJI": "FJ", "GUAM": "GU",
    "NUEVA ZELANDA": "NZ", "SAMOA AMERICANA": "AS", "ISLAS SALOMON": "SB",
    "OTROS DE OCEANIA": None, "PAIS NO DECLARADO": None,
}


def _post(name, program):
    import requests
    dest = pc.RAW / f"{name}.htm"
    if dest.exists() and dest.stat().st_size > 2_000:
        return
    print("RUN", name)
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0"
    r = s.post(f"{pc.HOST}/RpWebStats.exe/CmdSet?", data={
        "MAIN": "WebServerMain.inl", "BASE": pc.BASE, "LANG": "esp", "CODIGO": "XXUSUARIOXX",
        "ITEM": "PROGRED", "MODE": "RUN", "CMDSET": program, "Submit": "Ejecutar"}, timeout=1800)
    r.raise_for_status()
    m = re.search(r"(RpBases[^\"'&<>]*?\.htm)", r.text)
    if not m:
        raise SystemExit(f"{name}: REDATAM returned no output file.\n{r.text[:800]}")
    t = s.get(f"{pc.HOST}/RpWebUtilities.exe/Text?LFN=" + urllib.parse.quote(m.group(1))
              + "&TYPE=TMP", timeout=1800)
    t.raise_for_status()
    body = t.content.decode("utf-8", errors="replace")
    if "<table" not in body.lower():
        raise SystemExit(f"{name}: REDATAM returned no table.\n{body[:600]}")
    (pc.RAW / f"{name}.program.txt").write_text(program, encoding="utf-8")
    dest.write_text(body, encoding="utf-8")
    print(f"  {dest.stat().st_size:,} bytes")
    time.sleep(1)


def _freq(name):
    out = []
    for r in pc._rows(name):
        if len(r) == 4 and r[1] != "Casos" and r[0] != "Total" and "%" in r[2]:
            out.append((r[0], pc._num(r[1])))
    return out


def countries():
    """{code: (label, iso, count)}, codes and labels paired by order and count."""
    a, b = _freq("pa_nat_nacicod"), _freq("pa_nat_nacilab")
    assert len(a) == len(b) == 189, (len(a), len(b))
    out = {}
    for (c, n), (lab, m) in zip(a, b):
        assert n == m, (c, lab, n, m)
        if lab not in LABEL_ISO:
            raise SystemExit(f"pa_imm: no ISO for {lab!r}")
        out[int(c)] = (lab, LABEL_ISO[lab], n)
    return out


def drawn_codes():
    return sorted(c for c, (_, iso, _n) in countries().items()
                  if iso is not None and iso not in latam_immig.HISPANIC)


def chunks():
    c = drawn_codes()
    return [c[i:i + CHUNK] for i in range(0, len(c), CHUNK)]


def paisx(codes):
    lines = ["DEFINE PERSONA.PAISX\n AS SWITCH\n"]
    for i, c in enumerate(codes, 1):
        lines.append(f"  INCASE PERSONA.P_NACIO = 3 AND PERSONA.P_NACI_COD = {c}\n   ASSIGN {i}\n")
    lines.append(f"  DEFAULT 0\n TYPE INTEGER\n RANGE 0-{len(codes)}\n")
    return _H + "".join(lines) + "TABLE T\n AS AREALIST\n OF CORREG, PERSONA.PAISX\n"


def fetch():
    for name, p in BASE_PROGRAMS.items():
        _post(name, p)
    for i, ch in enumerate(chunks()):
        _post(f"pa_corr_paisx{i}", paisx(ch))


def main():
    if "--fetch" in sys.argv:
        fetch()
    cs = countries()
    rows = []
    for i, ch in enumerate(chunks()):
        part = pc.read_arealist(f"pa_corr_paisx{i}")
        assert len(part) == N_CORREG, len(part)
        for k, c in enumerate(ch, 1):
            s = sum(v.get(k, 0) for v in part.values())
            assert s == cs[c][2], (cs[c][0], s, cs[c][2])
        tot = sum(sum(v.values()) for v in part.values())
        assert tot == pc.CENSUS_POPULATION, tot
        for geo, v in part.items():
            for k, n in v.items():
                if k and n:
                    rows.append((geo.zfill(6), cs[ch[k - 1]][1], n))
    df = pd.DataFrame(rows, columns=["geo_id", "origin", "count"])
    df = df.groupby(["geo_id", "origin"], as_index=False)["count"].sum()
    df.to_csv(pc.NORM / "pa_imm.csv", index=False, encoding="utf-8")
    nat = df.groupby("origin")["count"].sum().sort_values(ascending=False)
    print(f"  checks pass: {len(drawn_codes())} non-Spanish-speaking birth countries, "
          f"{nat.sum():,} people, corregimiento sums equal the national frequency for each")
    print("  " + ", ".join(f"{k} {v:,}" for k, v in nat.head(20).items()))


if __name__ == "__main__":
    main()
