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
- Censo 2022, SIDRA table 10161 at N6 (added 2026-10-19): people aged 5+ who lived abroad five
  years before the census, by country, per município (457,003; Venezuela 199,067). It does not
  say who was born abroad (Brazilians coming home are in it), so it is used only to PLACE each
  UF's nationalities among its municípios, never for the counts: see `rake`. Origins in
  RETURNEE_ORIGINS (rich countries with many Brazilian emigrants) are not placed by it.

The mix per município is therefore the UF's register shares raked (IPF) to the município's own
foreign total, seeded by recent arrivals; before 2026-10-19 every município of a UF got the
same mix.

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
ARRIVALS = RAW / "sidra_10161_n6.json"      # condensed: {município: {country name: people}}
UF_CODES = [11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 35, 41,
            42, 43, 50, 51, 52, 53]
ARRIVALS_N1_TOTAL = 457_003                 # SIDRA 10161 at N1, Total
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


# SIDRA 10161 names countries in IBGE's own Portuguese; these are the ones ISO (the register's
# spellings) does not already cover once accents, hyphens and commas are ignored.
ISO_10161 = {
    "AZERBAIJAO": "AZ", "BOSNIA E HERZEGOVINA": "BA", "CINGAPURA": "SG", "COMORES": "KM",
    "FEDERACAO DOS ESTADOS DA MICRONESIA": "FM", "GRANADA": "GD", "GUIANA FRANCESA": "GF",
    "MALAWI": "MW", "MALDIVAS": "MV", "MAURICIO": "MU", "MIANMAR": "MM", "QUIRGUISTAO": "KG",
    "REINO UNIDO (ESCOCIA INGLATERRA IRLANDA DO NORTE E PAIS DE GALES)": "GB",
    "SAO CRISTOVAO E NEVIS": "KN", "TURCOMENISTAO": "TM", "TUVALU": "TV", "ZIMBABWE": "ZW",
}
NOT_A_COUNTRY_10161 = {"NAO SABE PAIS ESTRANGEIRO"}
# Origins whose "lived abroad five years ago" counts are mostly Brazilians coming home (the
# Brazilian diaspora: United States, Japan, Portugal, Europe, Canada, Oceania, Korea, Israel,
# the Gulf). For these the arrivals say nothing about where their citizens live in Brazil, so
# the register's stock is spread by foreign-born population, as before.
RETURNEE_ORIGINS = {"US", "CA", "JP", "KR", "IL", "AE", "QA", "KW", "SA", "AU", "NZ", "PT", "ES",
                    "IT", "FR", "DE", "GB", "IE", "CH", "AT", "NL", "BE", "LU", "SE", "NO", "DK",
                    "FI", "IS", "GR", "MT", "MC", "AD", "LI", "SM", "VA"}


def _norm(s):
    import unicodedata
    s = unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode().upper()
    return " ".join(s.replace("-", " ").replace(",", " ").split())


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
    fetch_arrivals()


def _get(u):
    import gzip
    import time
    import urllib.request
    for _ in range(4):
        try:
            b = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}),
                                       timeout=300).read()
            if b[:2] == b"\x1f\x8b":
                b = gzip.decompress(b)
            return json.loads(b.decode("utf-8"))
        except Exception as e:                      # IBGE's API answers 500 to oversized queries
            print("retry", e)
            time.sleep(3)
    raise SystemExit("failed " + u)


def fetch_arrivals():
    """SIDRA 10161 at N6: {município code: {country name: people}}, 5+ years old and living
    abroad five years before the census, by country. One UF and 25 countries per request
    (Minas Gerais in one is a 500). Municípios with none are absent from the API's answer."""
    base = "https://servicodados.ibge.gov.br/api/v3/agregados/10161"
    meta = _get(base + "/metadados")
    cats = [c for c in meta["classificacoes"] if str(c["id"]) == "2084"][0]["categorias"]
    ids = [str(c["id"]) for c in cats if c["nome"] != "Total"]
    out = {}
    for uf in UF_CODES:
        for i in range(0, len(ids), 25):
            d = _get(f"{base}/periodos/2022/variaveis/13340?localidades=N6[N3[{uf}]]"
                     f"&classificacao=2[6794]|2084[{','.join(ids[i:i + 25])}]")
            for res in d[0]["resultados"]:
                (_, name), = res["classificacoes"][1]["categoria"].items()
                for s in res["series"]:
                    out.setdefault(s["localidade"]["id"], {})[name] = s["serie"]["2022"]
        print("  10161 UF", uf, flush=True)
    ARRIVALS.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print("fetched", ARRIVALS.name)


def arrivals():
    """-> {município: {iso: people}} from SIDRA 10161 ('-' is none, 'X' suppressed, both 0),
    plus the national table's total for the check, and the names it could not map."""
    iso = {_norm(k): v for k, v in ISO.items()}
    iso.update(ISO_10161)
    out, unknown = {}, set()
    for m, row in json.loads(ARRIVALS.read_text(encoding="utf-8")).items():
        for name, v in row.items():
            if not v.isdigit() or int(v) == 0:
                continue
            k = _norm(name)
            if k in NOT_A_COUNTRY_10161:
                continue
            if k not in iso:
                unknown.add(name)
                continue
            d = out.setdefault(m, {})
            d[iso[k]] = d.get(iso[k], 0) + int(v)
    return out, unknown


def rake(f, shares, arr, iters=300):
    """One UF. f {município: foreign + naturalised people}, shares {iso: register share},
    arr {município: {iso: arrivals}}. Returns {(município, iso): people}.

    Columns: the UF's people (sum of f) split by the register's shares, as before. Rows: each
    município's own f. Seed: for an origin that is not in RETURNEE_ORIGINS, its recent arrivals
    per município (SIDRA 10161) plus whatever the column holds beyond them (the older stock),
    spread over the foreign people each município has beyond its recent arrivals; the returnee
    origins are all older stock in that sense. Iterative proportional fitting then makes rows
    and columns add up."""
    import numpy as np
    muns = [m for m, n in f.items() if n > 0]
    isos = sorted(shares)
    fv = np.array([f[m] for m in muns], dtype=float)
    T = fv.sum()
    col = np.array([T * shares[i] for i in isos])
    A = np.zeros((len(muns), len(isos)))               # recent arrivals, origins they can speak for
    for j, i in enumerate(isos):
        if i not in RETURNEE_ORIGINS:
            A[:, j] = [arr.get(m, {}).get(i, 0) for m in muns]
    # the older stock lives in the foreign population the recent arrivals do not account for
    resid = np.maximum(fv - A.sum(axis=1), 0.0) + 0.05 * fv
    g = resid / resid.sum()
    seed = A + np.outer(g, np.maximum(col - A.sum(axis=0), 0.0))
    seed += 1e-6 * np.outer(fv / T, col)                # no empty row or column
    x = seed.copy()
    for _ in range(iters):
        x *= (fv / x.sum(axis=1))[:, None]
        x *= (col / x.sum(axis=0))[None, :]
    x *= (fv / x.sum(axis=1))[:, None]                 # rows are exact, columns within 1e-6
    err = np.abs(x.sum(axis=0) - col).max() / T
    x = np.where((x > 0.01) | (x == x.max(axis=1, keepdims=True)), x, 0.0)                   # drop specks, then restore the row sums
    x *= (fv / x.sum(axis=1))[:, None]
    return {(m, i): x[r, j] for r, m in enumerate(muns) for j, i in enumerate(isos)
            if x[r, j] > 0}, err


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
    elif not ARRIVALS.exists():
        fetch_arrivals()
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
    shares = {u: dict(zip(g["iso"], g["share"])) for u, g in by.groupby("unidade_da_federacao")}
    arr, unknown10161 = arrivals()
    check(not unknown10161, f"every SIDRA 10161 country maps to ISO ({sorted(unknown10161)[:5]})")
    arr_tot = sum(sum(r.values()) for r in arr.values())
    check(abs(arr_tot - ARRIVALS_N1_TOTAL) < 0.001 * ARRIVALS_N1_TOTAL,
          f"10161 municípios sum to {arr_tot:,} against the national {ARRIVALS_N1_TOTAL:,} "
          f"(less 'country not known')")
    check(set(arr) <= set(mun), "every município in 10161 is in 10157")
    by_uf = {}
    for m, n in mun.items():
        if n:
            by_uf.setdefault(uf_of[m], {})[m] = n
    rows, worst = [], 0.0
    for u, f in sorted(by_uf.items()):
        got, err = rake(f, shares[u], arr)
        worst = max(worst, err)
        rows += [(m, iso, n) for (m, iso), n in got.items()]
    check(worst < 1e-3, f"raked columns within {worst:.1e} of the UF totals")
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
