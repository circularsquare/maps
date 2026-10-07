"""Mexico, Censo 2020: the people countries/mx.py had drawn as Spanish who speak something else.
Session edd42a8c-latn, 2026-10-05; the record is sources/mx.md, "Immigrant and settler languages".

    python sources/mx_origin.py [--fetch]

Writes
  data/normalized/mx_origin.csv   geo_id, origin, count: people aged 3+ who said they speak no
                                  indigenous language, by country of birth, per municipio.
                                  origin is ISO 3166 alpha-2; "US_U18" the US-born aged 3-17
                                  (the rest of the US-born are "US"); "XX" born abroad, country
                                  not stated. Mexican-born are not listed.
  data/normalized/mx_settlers.csv geo_id, loc, name, lon, lat, node, count: the long-settled
                                  communities, locality by locality (Plautdietsch in the
                                  Mennonite colonies, Venetian in Chipilo), aged 3+.

1. COUNTRY OF BIRTH. INEGI's "Poblacion 3 anos y mas" cube (the one sources/mx_censo.py reads),
   dimension "Lugar de nacimiento" (entidad, or country under its continent, INEGI's own
   three-digit country codes), sliced to "No habla lengua indigena": the people countries/mx.py
   drew as Spanish. Foreign-born who speak an indigenous language (Guatemalans speaking Mam, say)
   are already on the map on that language. A second query splits the US-born by age.
   CHECKS: 2,469 municipios; per municipio the named countries plus "No especificado" of each
   continent sum to "En otro pais"; the US-born by age sum to the US column; the municipios sum
   to the national row of the same query.

2. MENNONITE COLONIES (Plautdietsch). Nothing in the census names them: its religion question
   finds 6,109 "Anabautista/Menonita" aged 3+ nationally (Cuauhtemoc 420), against a colony
   census of 74,122 (Die Mennonitische Post, Oct 2022, "Mexico colony census brings surprises":
   Manitoba Colony 17,212, Swift Current 3,480). The colony villages are named for their number
   ("Campo 6-B", "Nuevo Progreso Campo Siete", "Campo Menonita Numero Quince"), so they are
   ITER 2020 localities with "Campo" or "Menonit" in the name, in the municipios that hold the
   colonies (COLONY_MUN), with no indigenous-language speakers to speak of (under 10%).
   Witness: Cuauhtemoc's matched villages hold 16,978 people, the Post's Manitoba Colony 17,212.
   Campeche and Durango colony villages mostly have other names ("Hamburgo", "Patio de Flores"),
   so there the matched villages only place the people and the count is the cited state
   figure (STATE_FIGURE), scaled to aged 3+ by the matched villages' own ratio.
   Everyone counted is drawn on Plautdietsch: the colonies are Old Colony Mennonites, whose
   everyday language it is; a few Mexican workers living in the villages are inside the count.

3. CHIPILO (Venetian). Ethnologue, 18th ed. (2015), Venetian (Mexico): 2,500 speakers (2011),
   placed at Chipilo de Francisco Javier Mina, San Gregorio Atzompa, Puebla (ITER 4,059 people),
   scaled to aged 3+ by the locality's own ratio.
"""
import csv
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "mx"
NORM = ROOT / "data" / "normalized"
ITER = ROOT.parent / "religiondots" / "data" / "raw" / "mx" / "iter_00_cpv2020_csv.zip"
EXPORT_URL = "https://www.inegi.org.mx/sistemas/olap/exporta/exporta.aspx"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
PLAUTDIETSCH = "indoeuropean.germanic.continental.lowgerman.plautdietsch"
VENETIAN = "indoeuropean.romance.venetian"

_L = "[Lugar de nacimiento].[Lugar de nacimiento]"
_E = "[Edad Población de 3 años y más].[Edad Población de 3 años y más]"
_ROWS = ("{[Entidad y municipio].[Entidad y municipio].[Total], "
         "Descendants([Entidad y municipio].[Entidad y municipio].[Total], 2)}")
_NOHLI = "[Habla indígena y lengua INALI].[Habla indígena y lengua INALI].[No habla lengua indígena]"
_MEAS = "[Measures].[Población  de 3 años y más]"
QUERIES = {
    "cpv2020_olap_p3mas_municipio_nacimiento_nohli.csv": (
        "select {" + _L + ".[En otro país], " + _L + ".[En los Estados Unidos de América], "
        "Descendants(" + _L + ".[En otro país], 1), Descendants(" + _L + ".[En otro país], 2)} "
        "on columns, " + _ROWS + " on rows from [Poblacion 3 años y mas] where (" + _MEAS + ", "
        + _NOHLI + ")"),
    "cpv2020_olap_p3mas_municipio_eeuu_edad_nohli.csv": (
        "select {" + _E + ".[Total], " + _E + ".[De 3 a 4 años], " + _E + ".[De 5 a 9 años], "
        + _E + ".[De 10 a 14 años], " + _E + ".[15 Años], " + _E + ".[16 Años], " + _E
        + ".[17 Años]} on columns, " + _ROWS + " on rows from [Poblacion 3 años y mas] where ("
        + _MEAS + ", " + _NOHLI + ", " + _L + ".[En los Estados Unidos de América])"),
}

# INEGI country code -> ISO 3166 alpha-2 (the cube prints "<code> <official name>")
INEGI_ISO = {
    101: "AO", 102: "DZ", 104: "BJ", 105: "BW", 106: "BF", 107: "BI", 108: "CV", 109: "CM",
    112: "RW", 113: "CG", 114: "CI", 115: "TD", 116: "DJ", 117: "EG", 118: "ER", 119: "ET",
    120: "GA", 121: "GM", 122: "GH", 123: "GN", 124: "GW", 125: "GQ", 126: "KE", 127: "LS",
    128: "LR", 129: "LY", 130: "MG", 133: "MW", 134: "ML", 135: "MA", 136: "MU", 137: "MR",
    139: "MZ", 140: "NA", 141: "NE", 142: "NG", 143: "CF", 147: "SH", 148: "ST", 149: "SN",
    150: "SC", 151: "SL", 152: "KM", 153: "SO", 154: "ZA", 160: "SZ", 161: "TZ", 162: "TG",
    164: "TN", 165: "UG", 167: "ZM", 168: "ZW", 169: "CD", 170: "EH", 171: "SD", 172: "SS",
    201: "AI", 202: "AG", 204: "AR", 205: "AW", 206: "BS", 207: "BB", 208: "BZ", 209: "BM",
    210: "BO", 211: "BR", 212: "KY", 213: "CA", 214: "CO", 215: "CR", 216: "CU", 217: "CL",
    218: "DM", 219: "EC", 220: "SV", 222: "GD", 223: "GL", 225: "GT", 226: "GY", 228: "HT",
    229: "HN", 230: "JM", 231: "FK", 233: "MS", 234: "NI", 235: "PA", 236: "PY", 237: "PE",
    238: "PR", 239: "DO", 240: "KN", 241: "IM", 242: "VC", 243: "LC", 244: "SR", 245: "TT",
    246: "TC", 247: "UY", 248: "VI", 249: "VG", 250: "VE", 251: "CW", 252: "SX",
    301: "AF", 302: "SA", 303: "AM", 304: "AZ", 305: "BH", 306: "BD", 307: "BT", 308: "BN",
    309: "KH", 312: "KP", 313: "KR", 315: "CN", 316: "TW", 318: "CY", 321: "AE", 322: "PH",
    323: "GE", 325: "IN", 326: "ID", 327: "IR", 328: "IQ", 329: "IL", 330: "JP", 331: "JO",
    332: "KZ", 333: "KG", 334: "KW", 335: "LB", 337: "MY", 338: "MV", 339: "MN", 340: "MM",
    341: "NP", 342: "OM", 343: "PK", 344: "QA", 345: "LA", 346: "SG", 347: "SY", 348: "LK",
    349: "TH", 350: "TJ", 351: "TM", 352: "TR", 353: "UZ", 354: "VN", 355: "YE", 356: "PW",
    357: "TL", 358: "PS",
    401: "AL", 402: "DE", 403: "AD", 405: "AT", 407: "BY", 408: "BE", 409: "BA", 410: "BG",
    411: "HR", 412: "DK", 413: "SK", 414: "SI", 415: "ES", 416: "EE", 417: "FO", 418: "FI",
    419: "FR", 420: "GI", 421: "GR", 422: "HU", 423: "IE", 424: "IS", 425: "IT", 426: "LV",
    427: "LI", 428: "LT", 429: "LU", 430: "MK", 431: "MT", 433: "MD", 434: "MC", 435: "NO",
    436: "NL", 437: "PL", 438: "PT", 439: "GB", 440: "CZ", 441: "RO", 442: "RU", 443: "SM",
    444: "VA", 445: "SE", 446: "CH", 447: "UA", 452: "RS", 453: "ME",
    501: "AU", 503: "CK", 505: "GU", 507: "KI", 512: "MP", 513: "MH", 514: "FM", 516: "NR",
    517: "NU", 520: "NZ", 522: "UM", 523: "PG", 524: "PN", 526: "SB", 527: "WS", 528: "AS",
    530: "TK", 531: "TO", 532: "TV", 533: "VU", 535: "FJ",
}
CONTINENTS = {"África", "América", "Asia", "Europa", "Oceanía", "País insuficientemente especificado"}

# Mennonite colony municipios (sources/mx.md has the reasons): Chihuahua's Manitoba and Swift
# Current colonies and their daughters (Cuauhtemoc, Riva Palacio, Cusihuiriachi, Namiquipa),
# El Sabinal (Ascension), Buenos Aires / El Cuervo (Janos), Ahumada, Nuevo Casas Grandes;
# Durango's Nuevo Ideal colony; Zacatecas's La Honda (Miguel Auza) and La Batea (Sombrerete);
# Campeche's Hopelchen, Hecelchakan, Tenabo and Candelaria colonies; Tamaulipas's Las Adjuntas
# (Casas).
COLONY_MUN = {"08017", "08054", "08018", "08048", "08005", "08035", "08001", "08050", "10039",
              "32029", "32042", "04006", "04005", "04008", "04011", "28008"}
# cited state totals where the villages' names do not find them all (all ages)
STATE_FIGURE = {
    "04": (15_000, "Reuters, 2022 investigation of Mennonite colonies in Campeche, via "
                   "Wikipedia 'Mennonites in Mexico': approx. 15,000"),
    "10": (6_500, "El Siglo de Durango (Laura Ramirez), 2012, via Wikipedia 'Mennonites in "
                  "Mexico': approx. 6,500 in Durango"),
}
CHIPILO = ("21125", 2_500)    # Ethnologue 18th ed. (2015), Venetian (Mexico): 2,500 (2011)


def fetch():
    import requests
    RAW.mkdir(parents=True, exist_ok=True)
    for name, q in QUERIES.items():
        dest = RAW / name
        if dest.exists() and dest.stat().st_size > 50_000:
            continue
        fields = {
            "nomdimfila": "Entidad y municipio", "to_display": "",
            "cube": "Poblacion 3 años y mas", "cubeName": "Poblacion 3 años y mas",
            "nomdimColumna": "Lugar de nacimiento", "Lc_tituloFiltro": "Consulta",
            "Lc_unidadmedida": "Personas", "Lc_sql": q,
            "Lc_conexion": "provider=MSOLAP.8;MDX Compatibility=2;data source=W-OLAPCLPRO22;"
                           "Connect timeout=120;Initial catalog=PV2020_AMD_Poblacion",
            "Lc_titulo": "Población|", "Lc_piepagina": "FUENTE:", "Lc_salida": "0",
            "Lc_StrConexion": "1", "Lc_ValidaDimGeo": "0",
            "Lc_formato": "Texto separado por comas(.csv)",
            "Lc_encabeza": "-", "Cant_Col": "80", "Cant_Fil": "2470", "completo": "completo",
        }
        body = "&".join(requests.utils.quote(k, safe="", encoding="latin-1") + "="
                        + requests.utils.quote(v, safe="", encoding="latin-1")
                        for k, v in fields.items())
        r = requests.post(EXPORT_URL, data=body.encode("ascii"), timeout=900,
                          headers={**UA, "Content-Type": "application/x-www-form-urlencoded"})
        r.raise_for_status()
        if len(r.content) < 50_000 or b"<html" in r.content[:2000].lower():
            raise SystemExit(f"{name}: {len(r.content)} bytes, not the table")
        dest.write_bytes(r.content)
        (RAW / (name[:-4] + ".mdx.txt")).write_text(q, encoding="utf-8")
        print(f"  {name}: {len(r.content):,} B")


def _num(s):
    s = s.strip().replace(",", "")
    return int(s) if s else 0


def read(name):
    """-> header labels, {geo_id or "00": [values]}"""
    text = (RAW / name).read_bytes().decode("cp1252")
    rows = list(csv.reader(text.splitlines(), skipinitialspace=True))
    head = next(r for r in rows if len(r) > 3 and r[0].strip() == "")
    cols = [c.strip() for c in head[2:]]          # code, name, then the columns
    out = {}
    for r in rows:
        k = r[0].strip() if r else ""
        if k == "" and len(r) > 2 and r[1].strip() == "Total":
            out["00"] = [_num(v) for v in r[2:2 + len(cols)]]
        elif re.fullmatch(r"\d\d \d\d\d", k):
            out[k.replace(" ", "")] = [_num(v) for v in r[2:2 + len(cols)]]
    return cols, out


def read_iter():
    z = zipfile.ZipFile(ITER)
    name = next(n for n in z.namelist() if n.endswith("conjunto_de_datos_iter_00CSV20.csv"))
    df = pd.read_csv(z.open(name), dtype=str, encoding="utf-8",
                     usecols=["ENTIDAD", "MUN", "LOC", "NOM_LOC", "LONGITUD", "LATITUD",
                              "POBTOT", "P_3YMAS", "P3YM_HLI"])
    df = df[~df["LOC"].isin(["0000", "9998", "9999"])].copy()
    for c in ("POBTOT", "P_3YMAS", "P3YM_HLI"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["geo_id"] = df["ENTIDAD"] + df["MUN"]
    return df


def _dms(s):
    m = re.match(r"\s*(\d+)\D+(\d+)\D+([\d.]+)\D*([NSEW])", str(s))
    if not m:
        return float("nan")
    v = int(m.group(1)) + int(m.group(2)) / 60 + float(m.group(3)) / 3600
    return -v if m.group(4) in "SW" else v


def settlers():
    it = read_iter()
    pat = re.compile(r"\bcampo\b|menonit", re.I)
    c = it[it["geo_id"].isin(COLONY_MUN) & it["NOM_LOC"].str.contains(pat, na=False)]
    c = c[c["P3YM_HLI"] < 0.1 * c["P_3YMAS"].clip(lower=1)].copy()
    c["count"] = c["P_3YMAS"].astype(float)
    print("  Mennonite colony villages (ITER 2020, aged 3+ / all ages):")
    for st, g in c.groupby(c["geo_id"].str[:2]):
        line = f"    {st}: {len(g)} villages, {g['P_3YMAS'].sum():,} / {g['POBTOT'].sum():,}"
        if st in STATE_FIGURE:
            fig = STATE_FIGURE[st][0]
            f = fig * g["P_3YMAS"].sum() / g["POBTOT"].sum() / g["P_3YMAS"].sum()
            c.loc[g.index, "count"] = g["P_3YMAS"] * f
            line += f"; scaled to the cited {fig:,} (all ages) -> {c.loc[g.index, 'count'].sum():,.0f}"
        print(line)
    cu = c[c["geo_id"] == "08017"]["POBTOT"].sum()
    assert 15_000 < cu < 19_000, cu    # witness: Manitoba Colony 17,212 (Post 2022)
    c["node"] = PLAUTDIETSCH
    ch = it[(it["geo_id"] == CHIPILO[0]) & it["NOM_LOC"].str.contains("Chipilo")].copy()
    assert len(ch) == 1, ch
    ch["count"] = CHIPILO[1] * ch["P_3YMAS"] / ch["POBTOT"]
    ch["node"] = VENETIAN
    out = pd.concat([c, ch])
    out["lon"] = out["LONGITUD"].map(_dms)
    out["lat"] = out["LATITUD"].map(_dms)
    assert out[["lon", "lat"]].notna().all().all()
    return out.rename(columns={"LOC": "loc", "NOM_LOC": "name"})[
        ["geo_id", "loc", "name", "lon", "lat", "node", "count"]]


def main():
    if "--fetch" in sys.argv or not all((RAW / n).exists() for n in QUERIES):
        fetch()
    names = list(QUERIES)
    cols, b = read(names[0])
    acols, a = read(names[1])
    munis = [k for k in b if k != "00"]
    assert len(munis) == 2469, len(munis)
    i_abroad, i_us = cols.index("En otro país"), cols.index("En los Estados Unidos de América")
    cidx = {}
    for j, c in enumerate(cols):
        m = re.match(r"(\d{3}) ", c)
        if m:
            cidx[j] = INEGI_ISO[int(m.group(1))]
    ne = [j for j, c in enumerate(cols) if c == "No especificado"]
    cont = [j for j, c in enumerate(cols) if c in CONTINENTS]
    assert len(cont) == 6 and len(ne) == 5, (cont, ne)
    rows = []
    for k in ["00"] + munis:
        v = b[k]
        named = sum(v[j] for j in cidx)
        assert sum(v[j] for j in cont) == v[i_abroad], k
        assert named + sum(v[j] for j in ne) <= v[i_abroad], k
        if k == "00":
            continue
        us, ua = v[i_us], a[k]
        assert ua[0] == us, (k, ua[0], us)
        u18 = sum(ua[1:])
        for iso, n in (("US", us - u18), ("US_U18", u18), ("XX", v[i_abroad] - named)):
            if n:
                rows.append((k, iso, n))
        for j, iso in cidx.items():
            if v[j]:
                rows.append((k, iso, v[j]))
    df = pd.DataFrame(rows, columns=["geo_id", "origin", "count"])
    df = df.groupby(["geo_id", "origin"], as_index=False)["count"].sum()
    nat = df.groupby("origin")["count"].sum()
    assert nat["US"] + nat["US_U18"] == b["00"][i_us], "US-born: municipios against the Total row"
    assert nat.drop(["US", "US_U18"]).sum() == b["00"][i_abroad], "abroad: municipios against Total"
    print(f"  checks pass: 2,469 municipios; continents sum to 'En otro pais'; US-born by age sum "
          f"to the US column; municipios sum to the national row")
    print(f"  aged 3+, no indigenous language: born in the US {b['00'][i_us]:,} "
          f"({nat['US_U18']:,} aged 3-17), elsewhere abroad {b['00'][i_abroad]:,} "
          f"(country not stated {nat['XX']:,})")
    for k, v in nat.sort_values(ascending=False).head(25).items():
        print(f"    {v:>9,}  {k}")
    df.to_csv(NORM / "mx_origin.csv", index=False, encoding="utf-8")
    s = settlers()
    s.to_csv(NORM / "mx_settlers.csv", index=False, encoding="utf-8")
    print(f"  wrote mx_origin.csv ({len(df):,} rows), mx_settlers.csv: "
          + ", ".join(f"{n.split('.')[-1]} {v:,.0f}" for n, v in s.groupby("node")["count"].sum().items()))


if __name__ == "__main__":
    main()
