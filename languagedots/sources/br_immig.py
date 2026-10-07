"""Brazil: people born abroad per município, by nationality -> data/normalized/br_immig.csv
(unit = município code, iso, count). Read by countries/br.py, which turns them into languages
with sources/latam_immig.py (record: sources/br.md, "Immigrant languages").

    python sources/br_immig.py [--fetch]

TWO SOURCES, because neither has both halves:
- Censo 2022, SIDRA table 10157 at N6: residents by nationality (brasileiros natos,
  naturalizados, estrangeiros) per município. Naturalizados + estrangeiros is the COUNT drawn.
  The census published no country of birth or nationality per município (as of 2026-10).
- SISMIGRA-ATIVOS (Polícia Federal migration register, OBMigra/UnB release; Anita downloaded it
  2026-10-06 from the UnB SharePoint, `data/raw/br/sismigra_ativos.zip`, unmodified): one row
  per registration with UF of residence and nationality (`pais`). Only `situacao == "Ativo"`
  is used (2,080,214 of 3,879,289 rows); expired, cancelled and excluded registrations are
  mostly people who left, naturalised or died. It gives the MIX: each UF's registrations by
  nationality, applied to every município of that UF. The register has no município column.
  Registrations with no UF (14,190 active) or no nationality ("", "NÃO ESPECIFICADO", "BRASIL")
  are left out of the shares.

So naturalised Brazilians take the mix of active foreign registrations in their UF too; they are
older and more Portuguese, Japanese, Italian and Lebanese than that mix (sources/br.md says so).

CHECKS: 5,570 municípios, the same set as br.csv's; the per-município sum of the two classes
equals the N1 national figures; every `pais` value maps to an ISO code or is a listed blank;
each UF's shares sum to 1; br_immig.csv sums to the census's foreign total.
"""
import json
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "br"
NORM = ROOT / "data" / "normalized"
SISMIGRA = RAW / "sismigra_ativos.zip"
SIDRA = RAW / "sidra_10157_n6.json"
SIDRA_N1 = RAW / "sidra_10157_n1.json"
URL = ("https://servicodados.ibge.gov.br/api/v3/agregados/10157/periodos/2022/variaveis/93"
       "?localidades={lvl}[all]&classificacao=302[6832,6833]|2[6794]|58[95253]")
N_MUN = 5570
NOT_A_COUNTRY = {"", "NÃO ESPECIFICADO", "BRASIL"}

ISO = {
    "AFEGANISTÃO": "AF", "ALBÂNIA": "AL", "ALEMANHA": "DE", "ANDORRA": "AD", "ANGOLA": "AO",
    "ANTÍGUA E BARBUDA": "AG", "ARGENTINA": "AR", "ARGÉLIA": "DZ", "ARMÊNIA": "AM",
    "ARUBA": "AW", "ARÁBIA SAUDITA": "SA", "AUSTRÁLIA": "AU", "AZERBAIDJÃO": "AZ",
    "BAHAMAS": "BS", "BANGLADESH": "BD", "BARBADOS": "BB", "BAREIN": "BH", "BELIZE": "BZ",
    "BENIN": "BJ", "BIELORRÚSSIA": "BY", "BOLÍVIA": "BO", "BOTSWANA": "BW", "BRUNEI": "BN",
    "BULGÁRIA": "BG", "BURKINA FASO": "BF", "BURUNDI": "BI", "BUTÃO": "BT", "BÉLGICA": "BE",
    "CABO VERDE": "CV", "CAMARÕES": "CM", "CAMBOJA": "KH", "CANADÁ": "CA", "CATAR": "QA",
    "CAZAQUISTÃO": "KZ", "CHADE": "TD", "CHILE": "CL", "CHINA": "CN", "CHIPRE": "CY",
    "CINGAPURA-SINGAPURA": "SG", "COLÔMBIA": "CO", "COMORES, ILHAS": "KM",
    # the register lists the Democratic Republic separately, so CONGO is Brazzaville
    "CONGO": "CG", "CORÉIA DO NORTE": "KP", "CORÉIA DO SUL": "KR", "COSTA DO MARFIM": "CI",
    "COSTA RICA": "CR", "CROÁCIA": "HR", "CUBA": "CU", "DINAMARCA": "DK", "DJIBUTI": "DJ",
    "DOMINICA": "DM", "EGITO": "EG", "EL SALVADOR": "SV", "EMIRADOS ÁRABES UNIDOS": "AE",
    "EQUADOR": "EC", "ERITRÉIA": "ER", "ESLOVÁQUIA": "SK", "ESLOVÊNIA": "SI", "ESPANHA": "ES",
    "ESTADO DA PALESTINA": "PS", "ESTADOS FEDERADOS DA MICRONÉSIA": "FM",
    "ESTADOS UNIDOS": "US", "ESTÔNIA": "EE", "ETIÓPIA": "ET", "FIJI": "FJ", "FILIPINAS": "PH",
    "FINLÂNDIA": "FI", "FRANÇA": "FR", "GABÃO": "GA", "GANA": "GH", "GEÓRGIA": "GE",
    "GRÉCIA": "GR", "GUATEMALA": "GT", "GUIANA": "GY", "GUINÉ": "GN", "GUINÉ BISSAU": "GW",
    "GUINÉ EQUATORIAL": "GQ", "GÂMBIA": "GM", "HAITI": "HT", "HOLANDA": "NL", "HONDURAS": "HN",
    "HUNGRIA": "HU", "ILHAS COOK": "CK", "ILHAS MARSHALL": "MH", "ILHAS SALOMÃO": "SB",
    "INDONÉSIA": "ID", "IRAQUE": "IQ", "IRLANDA": "IE", "IRÃ": "IR", "ISLÂNDIA": "IS",
    "ISRAEL": "IL", "ITÁLIA": "IT", "IUGOSLÁVIA": "YU", "IÊMEN": "YE", "JAMAICA": "JM",
    "JAPÃO": "JP", "JORDÂNIA": "JO", "KIRIBATI": "KI", "KOSOVO": "XK", "KUWAIT": "KW",
    "LAOS": "LA", "LESOTO": "LS", "LETÔNIA": "LV", "LIBÉRIA": "LR", "LIECHTENSTEIN": "LI",
    "LITUÂNIA": "LT", "LUXEMBURGO": "LU", "LÍBANO": "LB", "LÍBIA": "LY", "MACEDÔNIA": "MK",
    "MADAGASCAR": "MG", "MALDIVAS, ILHAS": "MV", "MALI": "ML", "MALTA": "MT", "MALÁSIA": "MY",
    "MARROCOS": "MA", "MAURITÂNIA": "MR", "MAURÍCIO, ILHAS": "MU", "MOLDÁVIA": "MD",
    "MONGÓLIA": "MN", "MONTENEGRO": "ME", "MOÇAMBIQUE": "MZ", "MYANMAR": "MM", "MÉXICO": "MX",
    "MÔNACO": "MC", "NAMÍBIA": "NA", "NAURU": "NR", "NEPAL": "NP", "NICARÁGUA": "NI",
    "NIGÉRIA": "NG", "NORUEGA": "NO", "NOVA ZELÂNDIA": "NZ", "NÍGER": "NE", "OMÃ": "OM",
    "PALAU": "PW", "PANAMÁ": "PA", "PAPUA-NOVA GUINÉ": "PG", "PAQUISTÃO": "PK",
    "PARAGUAI": "PY", "PERU": "PE", "POLÔNIA": "PL", "PORTUGAL": "PT", "QUÊNIA": "KE",
    "REINO UNIDO": "GB", "REPÚBLICA CENTRO AFRICANA": "CF",
    "REPÚBLICA DEMOCRÁTICA DO CONGO": "CD", "REPÚBLICA DOMINICANA": "DO",
    "REPÚBLICA TCHECA": "CZ", "ROMÊNIA": "RO", "RUANDA": "RW", "RÚSSIA": "RU", "SAMOA": "WS",
    "SAN MARINO": "SM", "SANTA LÚCIA": "LC", "SENEGAL": "SN", "SERRA LEOA": "SL",
    "SEYCHELLES": "SC", "SOMÁLIA": "SO", "SRI LANKA": "LK", "SUAZILÂNDIA": "SZ", "SUDÃO": "SD",
    "SURINAME": "SR", "SUÉCIA": "SE", "SUÍÇA": "CH", "SÃO TOMÉ E PRÍNCIPE": "ST",
    "SÃO VICENTE E GRANADINAS": "VC", "SÉRVIA": "RS", "SÉRVIA E MONTENEGRO": "RS",
    "SÍRIA": "SY", "TADJIQUISTÃO": "TJ", "TAILÂNDIA": "TH", "TANZÂNIA": "TZ",
    "TIMOR LESTE": "TL", "TOGO": "TG", "TONGA": "TO", "TRINIDAD E TOBAGO": "TT",
    "TUNÍSIA": "TN", "TURQUIA": "TR", "UCRÂNIA": "UA", "UGANDA": "UG", "UNIÃO SOVIÉTICA": "SU",
    "URUGUAI": "UY", "UZBEQUISTÃO": "UZ", "VANUATU": "VU", "VATICANO": "VA",
    "VENEZUELA": "VE", "VIETNÃ": "VN", "ZIMBABWE": "ZW", "ZÂMBIA": "ZM", "ÁFRICA DO SUL": "ZA",
    "ÁUSTRIA": "AT", "ÍNDIA": "IN",
}


def fetch():
    import urllib.request
    for lvl, path in (("N6", SIDRA), ("N1", SIDRA_N1)):
        req = urllib.request.Request(URL.format(lvl=lvl), headers={"User-Agent": "Mozilla/5.0"})
        b = urllib.request.urlopen(req, timeout=300).read()
        if b[:2] == b"\x1f\x8b":          # IBGE's API answers gzip unasked
            import gzip
            b = gzip.decompress(b)
        path.write_bytes(b)
        print("fetched", path.name)


def census(path):
    """{município code: naturalizados + estrangeiros}, and the two classes' totals."""
    out, tot = {}, {}
    for res in json.loads(path.read_text(encoding="utf-8"))[0]["resultados"]:
        cls = list(res["classificacoes"][0]["categoria"].values())[0]
        for s in res["series"]:
            v = s["serie"]["2022"]
            n = int(v) if v not in ("-", "X", "...", "..") else 0
            out[s["localidade"]["id"]] = out.get(s["localidade"]["id"], 0) + n
            tot[cls] = tot.get(cls, 0) + n
    return out, tot


def register():
    d = pd.read_csv(SISMIGRA, sep=";", dtype=str, keep_default_na=False, encoding="utf-8",
                    usecols=["unidade_da_federacao", "pais", "situacao"])
    print(f"SISMIGRA rows {len(d):,}; active {int((d['situacao'] == 'Ativo').sum()):,}")
    return d[d["situacao"] == "Ativo"]


def main():
    if "--fetch" in sys.argv or not SIDRA.exists() or not SIDRA_N1.exists():
        fetch()
    ok = True

    def check(c, m):
        nonlocal ok
        print(("ok    " if c else "FAIL  ") + m)
        ok &= bool(c)

    mun, tot = census(SIDRA)
    _, nat = census(SIDRA_N1)
    check(len(mun) == N_MUN, f"{len(mun):,} municípios in SIDRA 10157")
    # IBGE perturbs small cells: the municípios come out 7 and 4 people under N1
    check(all(abs(tot[k] - nat[k]) <= 0.0001 * nat[k] for k in nat),
          f"municípios {tot} sum to the national figures {nat}, within 0.01%")
    base = pd.read_csv(NORM / "br_status.csv", dtype={"geo_id": str})
    check(set(mun) == set(base["geo_id"]), "the same municípios as br_status.csv")

    a = register()
    unknown = sorted(set(a["pais"]) - set(ISO) - NOT_A_COUNTRY)
    check(not unknown, f"every nationality maps to ISO ({unknown[:5]})")
    a = a[~a["pais"].isin(NOT_A_COUNTRY) & (a["unidade_da_federacao"] != "")]
    a = a.assign(iso=a["pais"].map(ISO))
    by = a.groupby(["unidade_da_federacao", "iso"]).size().rename("n").reset_index()
    by["share"] = by["n"] / by.groupby("unidade_da_federacao")["n"].transform("sum")
    print(f"  shares from {int(by['n'].sum()):,} active registrations with UF and nationality, "
          f"{by['unidade_da_federacao'].nunique()} UFs")
    check(by["unidade_da_federacao"].nunique() == 27, "27 UFs in the register")

    ibge = json.loads((RAW / "ibge_municipios.json").read_text(encoding="utf-8"))
    sigla = {str(m["regiao-imediata"]["regiao-intermediaria"]["UF"]["id"]):
             m["regiao-imediata"]["regiao-intermediaria"]["UF"]["sigla"] for m in ibge}
    uf_of = {str(m["id"]): sigla[str(m["id"])[:2]] for m in ibge}
    check(set(mun) <= set(uf_of), "every município has a UF in the IBGE list")
    shares = {u: g[["iso", "share"]].to_numpy() for u, g in by.groupby("unidade_da_federacao")}
    rows = []
    for m, n in mun.items():
        if n:
            rows += [(m, iso, n * s) for iso, s in shares[uf_of[m]]]
    df = pd.DataFrame(rows, columns=["unit", "iso", "count"])
    check(abs(df["count"].sum() - sum(tot.values())) < 1,
          f"br_immig sums to the municípios' {sum(tot.values()):,} naturalised + foreign")
    if not ok:
        raise SystemExit("checks failed")
    df.to_csv(NORM / "br_immig.csv", index=False)
    top = df.groupby("iso")["count"].sum().sort_values(ascending=False).head(10)
    print(f"wrote br_immig.csv ({len(df):,} rows); top: "
          + ", ".join(f"{k} {v:,.0f}" for k, v in top.items()))


if __name__ == "__main__":
    main()
