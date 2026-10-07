"""Spain: INE's ECEPOV 2021, first language ("lengua inicial") by province.

    python sources/es_ecepov.py --fetch    download into data/raw/es/ whatever is missing
    python sources/es_ecepov.py            normalise from data/raw/es/

-> data/normalized/es.csv, two levels (alternatives, never summed):
   `province`            what is drawn: 52 provinces x source category, with tier
   `province_published`  ECEPOV's own cells, split by nationality (Total / Espanola /
                         Extranjera), kept for the checks and the record

THE QUESTION. Spain's census (2021, register-based) asks nothing. Its companion sample survey,
the Encuesta de Caracteristicas Esenciales de la Poblacion y las Viviendas (ECEPOV, INE, 1 July
2021, about 309,000 dwellings), asked every resident aged 2 and over to list the languages they
know and to mark "the language you spoke first" (lengua inicial). INE publishes it per province
as "Personas segun la lengua inicial mas frecuente por sexo, grupo de edad y nacionalidad"
(tables tpx 55725-55776, one per province in INE code order 01-52), naming the languages frequent
in that province plus combinations and "Otra". Universe: people in family dwellings.

WHY THIS AND NOT THE REGIONAL SURVEYS. The regional sociolinguistic surveys (Idescat's EULP,
the Valencian, Galician, Basque and Navarrese surveys) each ask a first-language question too,
but on different universes (15+ or 16+), years and wordings, and none covers the other 80% of
Spain. ECEPOV asks one question of everyone, everywhere, in the same year, at a sample about
fifty times the size of any of them, and it names the immigrant languages as well. The regional
sources are used for what ECEPOV lacks, which is where inside a province the regional language is
spoken (sources/es_place.py), and as checks.

DERIVED SPLITS (es2021.py's docstring has the labels):
  * combinations are shared equally across the languages named (spec 3.6);
  * foreign nationals' "Otra" is shared out by the province's foreign residents by nationality
    (Padron 1 Jan 2022, INE table 03005, 137 nationalities), each on its main first language,
    counting only languages the province's own table does not name, capped at the cell;
  * Spanish nationals' "Otra" stays on `other`, except its excess over the rest of Spain in
    Illes Balears, Asturias and Melilla, which goes to Catalan, Asturian and Tarifit.

CHECKS: every province's cells sum to its total; nationality parts sum to the total cell; the 52
provinces sum to the 19 autonomous-community tables (tpx 55547-55565) language by language;
province totals against the Padron; ECEPOV's Catalan share against Idescat's EULP 2023 by
province-area, and its Basque against Eustat's 2021 census-based first language by territory.
"""

import io
import json
import os
import re
import ssl
import sys
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "taxonomy"))
RAW = os.path.join(ROOT, "data", "raw", "es")
OUT = os.path.join(ROOT, "data", "normalized", "es.csv")

YEAR = 2021
SOURCE_ID = "es_ecepov2021_lengua_inicial"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")

def _as_table_label(node):
    """An origin language as ECEPOV's tables would record it: every Arabic variety under
    "Árabe", Moldovan under "Rumano" (the table names the language, not the variety)."""
    import es2021
    last = node.split(".")[-1]
    if node.startswith("afroasiatic.") and (node.startswith("afroasiatic.arabic")
                                            or last.endswith("_arabic")
                                            or last in ("darija", "hassaniya")):
        return es2021.ARABIC
    if node == "indoeuropean.romance.moldovan":
        return "indoeuropean.romance.romanian"
    return node


def es_mix(nat):
    """[(node, share)] for a 03005 nationality: the shared origin table (sources/origin_mix.py),
    varieties folded onto the language ECEPOV prints."""
    import es2021
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from origin_mix import mix
    out = {}
    for n, s in mix(es2021.ORIGIN[nat], "es").items():
        n = _as_table_label(n)
        out[n] = out.get(n, 0.0) + s
    return sorted(out.items())


PROV_TPX = {f"{i + 1:02d}": 55725 + i for i in range(52)}   # INE province code -> table
CCAA_TPX = list(range(55547, 55566))                          # 19 communities, check only
ECEPOV_URL = "https://www.ine.es/jaxi/files/tpx/es/csv_bdsc/{tpx}.csv"
PADRON_NAT_URL = "https://www.ine.es/jaxi/files/_px/es/px/t20/e245/p08/l0/03005.px"
EUSTAT_URL = "https://www.eustat.eus/bankupx/api/v1/es/DB/PX_010123_cepv3_lm01.px"
EULP_URL = "https://api.idescat.cat/taules/v2/eulp/{table}/{geo}/data?lang=en"
EULP = {  # Idescat EULP 2023, population 15+ by first language (thousands)
    "eulp2023_at.json": "3170/23128/at",        # 8 areas of the territorial plan
    "eulp2023_mun.json": "3170/23128/mun",      # Barcelona city
    "eulp2023_com.json": "3170/23128/com",      # Aran
    "eulp2023_cat.json": "3170/23128/cat",      # Catalonia
    "eulp2023_freq_cat.json": "3163/23127/cat",  # Catalonia, most frequent languages
}

PROV_NAME = {
    "01": "Araba/Álava", "02": "Albacete", "03": "Alicante/Alacant", "04": "Almería",
    "05": "Ávila", "06": "Badajoz", "07": "Balears, Illes", "08": "Barcelona", "09": "Burgos",
    "10": "Cáceres", "11": "Cádiz", "12": "Castellón/Castelló", "13": "Ciudad Real",
    "14": "Córdoba", "15": "Coruña, A", "16": "Cuenca", "17": "Girona", "18": "Granada",
    "19": "Guadalajara", "20": "Gipuzkoa", "21": "Huelva", "22": "Huesca", "23": "Jaén",
    "24": "León", "25": "Lleida", "26": "Rioja, La", "27": "Lugo", "28": "Madrid",
    "29": "Málaga", "30": "Murcia", "31": "Navarra", "32": "Ourense", "33": "Asturias",
    "34": "Palencia", "35": "Palmas, Las", "36": "Pontevedra", "37": "Salamanca",
    "38": "Santa Cruz de Tenerife", "39": "Cantabria", "40": "Segovia", "41": "Sevilla",
    "42": "Soria", "43": "Tarragona", "44": "Teruel", "45": "Toledo", "46": "Valencia/València",
    "47": "Valladolid", "48": "Bizkaia", "49": "Zamora", "50": "Zaragoza", "51": "Ceuta",
    "52": "Melilla",
}
# Provinces of each community, in the order of CCAA_TPX (INE's community order 01-19)
CCAA = [["04", "11", "14", "18", "21", "23", "29", "41"], ["22", "44", "50"], ["33"], ["07"],
        ["35", "38"], ["39"], ["05", "09", "24", "34", "37", "40", "42", "47", "49"],
        ["02", "13", "16", "19", "45"], ["08", "17", "25", "43"], ["03", "12", "46"],
        ["06", "10"], ["15", "27", "32", "36"], ["28"], ["30"], ["31"], ["01", "20", "48"],
        ["26"], ["51"], ["52"]]


def _get(url, dest, data=None):
    if os.path.exists(dest):
        return
    import time
    req = urllib.request.Request(url, data=data, headers={"User-Agent": UA,
                                                          "Content-Type": "application/json"})
    for attempt in range(4):  # INE's file server answers an occasional 500
        try:
            # Eustat's chain carries a certificate Python's store does not know (curl accepts
            # it); the payload is public statistics, so that one host is read unverified.
            ctx = ssl._create_unverified_context() if "eustat.eus" in url else None
            raw = urllib.request.urlopen(req, timeout=300, context=ctx).read()
            break
        except urllib.error.HTTPError:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(raw)
    print(f"  fetched {os.path.relpath(dest, ROOT)} ({len(raw):,} bytes)")


def fetch():
    for tpx in list(PROV_TPX.values()) + CCAA_TPX:
        _get(ECEPOV_URL.format(tpx=tpx), os.path.join(RAW, "ecepov", f"{tpx}.csv"))
    _get(PADRON_NAT_URL, os.path.join(RAW, "ine_03005.px"))
    q = {"query": [{"code": "sexo", "selection": {"filter": "item", "values": ["10"]}},
                   {"code": "periodo", "selection": {"filter": "item", "values": ["2021"]}}],
         "response": {"format": "json-stat2"}}
    _get(EUSTAT_URL, os.path.join(RAW, "eustat_lm01_2021.json"), data=json.dumps(q).encode())
    for fn, path in EULP.items():
        table, rest = path.split("/", 1)
        tid, geo = rest.split("/")
        _get(EULP_URL.format(table=f"{table}/{tid}", geo=geo), os.path.join(RAW, fn))


# ---------------------------------------------------------------- readers

def num(s):
    s = str(s).strip()
    if s in ("", ".", "nan", "..", "-"):
        return np.nan  # "." is INE's suppression mark (too few sample cases)
    return float(s.replace(".", "").replace(",", "."))


def read_ecepov(tpx):
    raw = open(os.path.join(RAW, "ecepov", f"{tpx}.csv"), "rb").read()
    try:
        txt = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        txt = raw.decode("latin-1")
    d = pd.read_csv(io.StringIO(txt), sep=";", dtype=str)
    d = d[(d["Edad"] == "Total") & (d["Sexo"] == "Ambos Sexos")]
    d["count"] = d["Total"].map(num)
    # Eight provinces suppress one small cell of the Total column ("."; too few sample cases).
    # It is the only gap in its column, so the column's own total gives it back exactly.
    t = d["Nacionalidad"] == "Total"
    gap = t & d["count"].isna()
    if gap.any():
        assert gap.sum() == 1, f"tpx {tpx}: {gap.sum()} suppressed cells in the Total column"
        whole = d.loc[t & (d["Lengua"] == "Total"), "count"].iloc[0]
        rest = d.loc[t & (d["Lengua"] != "Total"), "count"].sum()
        d.loc[gap, "count"] = max(0.0, whole - rest)
        SUPPRESSED.append((tpx, d.loc[gap, "Lengua"].iloc[0], max(0.0, whole - rest)))
    return d[["Nacionalidad", "Lengua", "count"]]


SUPPRESSED = []


def read_px(path):
    """INE PC-Axis, enough of it for 03005: VALUES per dimension and the DATA block."""
    t = open(path, encoding="latin-1").read()
    dims = re.findall(r'(STUB|HEADING)="?(.*?)"?;', t, re.S)
    order = []
    for _, v in dims:
        order += re.findall(r'"?([^",]+)"?', v.replace("\n", ""))
    vals = {m.group(1): re.findall(r'"(.*?)"', m.group(2))
            for m in re.finditer(r'VALUES\("(.*?)"\)=(.*?);', t, re.S)}
    data = re.search(r"DATA=(.*?);", t, re.S).group(1).split()
    data = [np.nan if x.strip('"') in ("..", ".", "-") else float(x.strip('"')) for x in data]
    idx = pd.MultiIndex.from_product([vals[k] for k in order], names=order)
    assert len(idx) == len(data), (len(idx), len(data))
    return pd.Series(data, index=idx)


def padron_nationality(year="2022"):
    """Province (2-digit) x nationality (03005's country rows), foreign residents, 1 Jan."""
    s = read_px(os.path.join(RAW, "ine_03005.px"))
    s = s.xs("Ambos sexos", level="Sexo").xs(year, level="Periodo")
    df = s.reset_index(name="n")
    df = df[df["Provincias"] != "TOTAL ESPAÑA"]
    df["prov"] = df["Provincias"].str[:2]
    return df.rename(columns={"Nacionalidad": "nat"})[["prov", "nat", "n"]]


# ---------------------------------------------------------------- build

def build():
    import es2021

    pub, drawn = [], []
    tables = {}
    for prov, tpx in PROV_TPX.items():
        d = read_ecepov(tpx)
        tables[prov] = d
        for r in d.itertuples():
            pub.append(dict(geo_id=prov, geo_level="province_published",
                            geo_name=PROV_NAME[prov],
                            source_category=f"{r.Lengua} [{r.Nacionalidad}]", count=r.count,
                            tier="modelled"))

    # ---- check: cells sum to totals, nationality parts sum to the whole
    worst_cells = worst_nat = 0.0
    for prov, d in tables.items():
        tot = d[d["Nacionalidad"] == "Total"].set_index("Lengua")["count"]
        assert tot.notna().all(), f"{prov}: suppressed cell in the Total column"
        worst_cells = max(worst_cells, abs(tot.drop("Total").sum() / tot["Total"] - 1))
        for nat in ("Española", "Extranjera"):
            part = d[d["Nacionalidad"] == nat].set_index("Lengua")["count"]
            assert part.drop("Total").fillna(0).sum() <= part["Total"] + 1
        both = d[d["Lengua"] == "Total"].set_index("Nacionalidad")["count"]
        worst_nat = max(worst_nat, abs(both["Española"] + both["Extranjera"] - both["Total"])
                        / both["Total"])
    print(f"check: language cells sum to each province's total, worst {100 * worst_cells:.3f}%")
    print(f"check: Spanish + foreign = total, worst {100 * worst_nat:.3f}%")
    assert worst_cells < 0.005 and worst_nat < 0.002  # Ourense: cells 0.34% short of its total

    # ---- check: provinces sum to the community tables
    worst = 0.0
    for tpx, provs in zip(CCAA_TPX, CCAA):
        c = read_ecepov(tpx)
        c = c[c["Nacionalidad"] == "Total"].set_index("Lengua")["count"]
        s = pd.concat([tables[p][tables[p]["Nacionalidad"] == "Total"].set_index("Lengua")
                       ["count"] for p in provs], axis=1).sum(axis=1)
        for lab in ("Total", "Castellano"):
            worst = max(worst, abs(s[lab] / c[lab] - 1))
    print(f"check: provinces sum to their community's table (total, Spanish), "
          f"worst {100 * worst:.2f}%")
    assert worst < 0.01
    nat_total = sum(t.loc[(t["Nacionalidad"] == "Total") & (t["Lengua"] == "Total"),
                         "count"].iloc[0] for t in tables.values())
    print(f"ECEPOV universe, 52 provinces: {nat_total:,.0f}")

    # ---- Spanish nationals' "Otra": baseline from the provinces without an unlisted language
    rows = []
    for prov, d in tables.items():
        g = d.set_index(["Nacionalidad", "Lengua"])["count"]
        rows.append(dict(prov=prov, sp_otra=g[("Española", "Otra")],
                         sp=g[("Española", "Total")], fo=g[("Extranjera", "Total")],
                         tot=g[("Total", "Total")]))
    base = pd.DataFrame(rows).set_index("prov")
    base["share"] = base["sp_otra"] / base["sp"]
    base["foreign"] = base["fo"] / base["tot"]
    fit = base.drop(index=list(es2021.REGIONAL_OTRA))
    b, a = np.polyfit(fit["foreign"], fit["share"], 1)
    resid = fit["share"] - (a + b * fit["foreign"])
    print(f"Spanish nationals' 'Otra' share = {a:.4f} + {b:.4f} x foreign share "
          f"(49 provinces, residual sd {resid.std() * 100:.2f} points)")
    regional = {}
    for prov, node in es2021.REGIONAL_OTRA.items():
        r = base.loc[prov]
        expect = (a + b * r["foreign"]) * r["sp"]
        z = (r["share"] - (a + b * r["foreign"])) / resid.std()
        regional[prov] = max(0.0, r["sp_otra"] - expect)
        print(f"  {PROV_NAME[prov]}: Spanish 'Otra' {r['sp_otra']:,.0f} ({100 * r['share']:.1f}%), "
              f"expected {expect:,.0f}, {z:+.1f} sd -> {regional[prov]:,.0f} to {node}")
        assert z > 2, f"{prov}: the excess is not distinguishable from noise"

    # ---- Aranese (Occitan of the Val d'Aran): not in ECEPOV's Lleida list, so inside its
    #      "Otra" or its "Catalan". EULP 2023 measures it for Aran alone; drawn as that share of
    #      Aran's Padron population, taken out of Lleida's Spanish nationals' "Otra".
    ar = json.load(open(os.path.join(RAW, "eulp2023_com.json"), encoding="utf-8"))
    al = list(ar["dimension"]["LAN_ISO"]["category"]["index"])
    av = dict(zip(al, [0 if x is None else x for x in ar["value"]]))
    aran_share = (av["OC_ARANESE"] + av["AR_OTHER_LANG"] / 2) / av["TOTAL"]
    emex = json.load(open(os.path.join(RAW, "idescat_emex_com_mun.json"), encoding="utf-8"))
    aran_munis = [m["id"][:5] for c in emex["fitxes"]["v"] if c["id"] == "39" for m in c["v"]]
    mn = pd.read_csv(os.path.join(RAW, "padron2022_muni_nationality.csv"), dtype={"muni": str})
    aran_pop = mn[(mn["nationality"] == "Total") & mn["muni"].isin(aran_munis)]["count"].sum()
    assert len(aran_munis) == 9 and 9000 < aran_pop < 12000, (aran_munis, aran_pop)
    aranese = min(aran_share * aran_pop, base.loc["25", "sp_otra"])
    regional_extra = {("25", es2021.OCCITAN): aranese}
    print(f"Aranese: EULP 2023 {100 * aran_share:.1f}% of Aran's 15+ x Padron 2022 "
          f"{aran_pop:,.0f} = {aranese:,.0f}, out of Lleida's Spanish 'Otra' "
          f"{base.loc['25', 'sp_otra']:,.0f}")

    # ---- foreign nationals' "Otra": shared by nationality
    pn = padron_nationality()
    unknown = set(pn["nat"]) - set(es2021.ORIGIN) - {
        n for n in pn["nat"] if n.isupper() or n.startswith("Resto") or n.startswith("UE(")}
    assert not unknown, f"03005 nationalities without an ORIGIN entry: {sorted(unknown)}"
    pn = pn[pn["nat"].isin(es2021.ORIGIN)]
    split_tot = split_drawn = 0.0
    for prov, d in tables.items():
        tot = d[d["Nacionalidad"] == "Total"].set_index("Lengua")["count"]
        named = {es2021.NAMES[l] for l in tot.index if l in es2021.NAMES}
        named |= {es2021.NAMES[m] for l in tot.index if l in es2021.COMBOS
                  for m in es2021.COMBOS[l]}
        g = d.set_index(["Nacionalidad", "Lengua"])["count"]
        fo_otra = g[("Extranjera", "Otra")]
        if np.isnan(fo_otra):
            fo_otra = 0.0
        sp_otra = g[("Total", "Otra")] - fo_otra
        expect = {}
        for r in pn[pn["prov"] == prov].itertuples():
            for node, share in es_mix(r.nat):
                if node in named or node == es2021.SPANISH or not r.n > 0:
                    continue
                expect[node] = expect.get(node, 0.0) + r.n * share
        e_sum = sum(expect.values())
        k = min(1.0, fo_otra / e_sum) if e_sum > 0 else 0.0
        for node, e in sorted(expect.items()):
            if e * k > 0:
                drawn.append((prov, f"Otra | nationality: {node}", e * k, "derived"))
        reg = regional.get(prov, 0.0)
        if reg:
            drawn.append((prov, f"Otra | regional: {es2021.REGIONAL_OTRA[prov]}", reg, "derived"))
        for (p2, node), c in regional_extra.items():
            if p2 == prov:
                drawn.append((prov, f"Otra | regional: {node}", c, "derived"))
                reg += c
        rest = fo_otra - e_sum * k + sp_otra - reg
        assert rest > -1, (prov, rest)
        drawn.append((prov, "Otra", max(rest, 0.0), "modelled"))
        split_tot += fo_otra
        split_drawn += e_sum * k
        # named languages and combinations
        for lab, c in tot.items():
            if lab in ("Total", "Otra"):
                continue
            if lab in es2021.COMBOS:
                ms = es2021.COMBOS[lab]
                for m in ms:
                    drawn.append((prov, f"{lab} | {m}", c / len(ms), "derived"))
            else:
                drawn.append((prov, lab, c, "modelled"))
    print(f"foreign nationals' 'Otra': {split_tot:,.0f}, of which {split_drawn:,.0f} "
          f"({100 * split_drawn / split_tot:.1f}%) shared to languages by nationality")

    out = pd.DataFrame(drawn, columns=["geo_id", "source_category", "count", "tier"])
    out = out.groupby(["geo_id", "source_category", "tier"], as_index=False)["count"].sum()
    out["geo_level"] = "province"
    out["geo_name"] = out["geo_id"].map(PROV_NAME)
    # ---- check: drawn rows sum to each province's total
    for prov, d in tables.items():
        t = d.loc[(d["Nacionalidad"] == "Total") & (d["Lengua"] == "Total"), "count"].iloc[0]
        s = out.loc[out["geo_id"] == prov, "count"].sum()
        assert abs(s / t - 1) < 0.005, (prov, s, t)
    print("check: drawn rows sum to each province's ECEPOV total (within 0.5%)")
    for lab in out["source_category"].unique():
        es2021.resolve(lab)

    df = pd.concat([out, pd.DataFrame(pub)], ignore_index=True)
    df["year"] = YEAR
    df["source_id"] = SOURCE_ID
    df = df[["geo_id", "geo_level", "geo_name", "source_category", "count", "tier", "year",
             "source_id"]]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {os.path.relpath(OUT, ROOT)}: {len(out)} drawn rows, "
          f"{out['count'].sum():,.0f} people")
    checks(out, tables)


def checks(out, tables):
    """Second sources: Idescat EULP 2023 (Catalonia) and Eustat 2021 (Basque Country)."""
    import es2021
    out = out.copy()
    out["node"] = out["source_category"].map(es2021.resolve)

    def share(provs, node):
        o = out[out["geo_id"].isin(provs)]
        return o.loc[o["node"] == node, "count"].sum() / o["count"].sum()

    e = json.load(open(os.path.join(RAW, "eulp2023_cat.json"), encoding="utf-8"))
    lab = list(e["dimension"]["LAN_ISO"]["category"]["index"])
    v = dict(zip(lab, e["value"]))
    eulp_ca = (v["CA"] + v["CA_ES"] / 2) / v["TOTAL"]
    print(f"check: Catalan first language in Catalonia, ECEPOV 2021 (all ages, both shared) "
          f"{100 * share(['08', '17', '25', '43'], es2021.CATALAN):.1f}% vs EULP 2023 (15+) "
          f"{100 * eulp_ca:.1f}%")
    ez = json.load(open(os.path.join(RAW, "eustat_lm01_2021.json"), encoding="utf-8"))
    dims = ez["id"]
    geo = [d for d in dims if "mbito" in d][0]
    lang = [d for d in dims if d.startswith("lengua")][0]
    gi = list(ez["dimension"][geo]["category"]["index"])
    li = list(ez["dimension"][lang]["category"]["index"])
    vals = np.array(ez["value"], dtype=float).reshape(ez["size"])
    vals = vals.squeeze()
    if vals.shape != (len(gi), len(li)):
        vals = vals.reshape(len(gi), len(li))
    for code, prov in (("01", "01"), ("48", "48"), ("20", "20")):
        r = vals[gi.index(code)]
        eu = (r[li.index("20")] + r[li.index("40")] / 2) / r[li.index("10")]
        print(f"check: Basque first language in {PROV_NAME[prov]}, ECEPOV "
              f"{100 * share([prov], es2021.BASQUE):.1f}% vs Eustat 2021 {100 * eu:.1f}%")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    build()
