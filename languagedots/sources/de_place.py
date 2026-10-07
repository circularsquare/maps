"""Germany: where inside a Land each language's dots go, by citizenship (Zensus 2022).

    python sources/de_place.py --fetch    download the grids and the Gemeinde table if missing
    python sources/de_place.py            build data/geo/de/de_grid_1km.gpkg

The COUNTS stay sources/de_mz.py's (Mikrozensus 2023, per Land). This file only decides where
inside a Land a language's dots land (AGENT_BRIEF §4 item 4, Anita 2026-10-05: "Germany
immigrant languages by foreign citizen is good. Ideally if we have the specific foreign origin,
that'd be best").

THE LAYER is religiondots' 1km Zensus 2022 grid (read-only: geometry, `ars`, `pop`; each cell
clipped to the Gemeinde holding its centre), re-keyed to its INSPIRE cell id and given one weight
column per Mikrozensus label, `w_<slug>`.

THE WEIGHT for language L in cell c of Gemeinde g, two steps, both Zensus 2022 (15 May 2022):
  1. how many people in g hold each of L's citizenships k: G_k(g), Zensus database table
     1000A-1023 "Personen: Staatsangehoerigkeit (Laender)", 203 citizenships x 10,786 Gemeinden;
  2. where inside g they live: p_k(c) / p_k(g), p_k a 1km grid column: k's own column where the
     12-country grid has one (Turkey, Poland, Russia, Kazakhstan, Ukraine, Romania, Italy,
     Greece, Croatia, Bosnia, Netherlands, Austria), else k's citizenship group on the groups
     grid less the countries the 12-country grid carries (EU27 rest, other-Europe rest), else
     the rest of the world (`Sonstige_Welt`).
  w_L(c) = sum over k of factor_k * G_k(g) * p_k(c) / p_k(g). Where p_k(g) is 0 (the grid's
  Cell-Key perturbation zeroed a small group in a small Gemeinde) all foreign citizens stand in
  for p_k, then population.
German is weighted by German citizens (grid column `Deutschland`, table LAND000).

CITIZENSHIP IS NOT LANGUAGE, and is used only as a locator: most of Germany's Russian speakers
are German-citizen Aussiedler, many Turkish speakers are naturalised. The map's counts never
move; a weight only says which part of the Land a language's people are likelier to live in.

CHECKS (printed; the build fails on the first four):
  * every Mikrozensus label has a mapping, every mapped citizenship is a label in the table;
  * grid cell ids recovered from the geometry are unique and all found in both grids, and every
    populated grid cell is in the layer or among the cells religiondots dropped (centre in no
    Gemeinde);
  * the Gemeinde table's codes join the layer's `ars` both ways (reported, and bounded);
  * the table's national citizenship totals against the 12-country grid's;
  * how much of each language's weight sits in the 78 Gemeinden of 100,000 or more, against
    their 31.7% of the population.
"""
import io
import json
import os
import sys
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
RAW = os.path.join(ROOT, "data", "raw", "de")
OUT_DIR = os.path.join(ROOT, "data", "geo", "de")
OUT = os.path.join(OUT_DIR, "de_grid_1km.gpkg")
NORM = os.path.join(ROOT, "data", "normalized", "de.csv")

from rdlink import RD_GEO  # noqa: E402

RD_GRID = RD_GEO / "de" / "de_grid_1km.gpkg"       # read-only

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
GRID_BASE = "https://www.destatis.de/static/DE/zensus/gitterdaten/"
GRIDS = {   # zip name -> the 1km CSV inside it
    "Staatsangehoerigkeit_nach_ausgewaehlten_Laendern.zip":
        "Zensus2022_Staatsangehoerigkeit_nach_Laendern_1km-Gitter.csv",
    "Zensus2022_Staatsangehoerigkeit_Gruppen_in_Gitterzellen.zip":
        "Zensus2022_Staatsangehoerigkeit_Gruppen_1km-Gitter.csv",
}
# Zensus database (ergebnisse.zensus2022.de, open, no login): the SPA's REST API. GET the
# table's structure, put Gemeinden in the columns, POST it back for a flat CSV.
ZDB = "https://ergebnisse.zensus2022.de/proxy/api/rest"
ZDB_TABLE = "1000A-1023"
ZDB_FILE = "zensus2022_1000A-1023_gemeinden.zip"

DASH = "–"   # grid: "exactly zero or changed to zero"

# The 12-country grid's columns, by the table's labels.
GRID_COUNTRY = {
    "Bosnien und Herzegowina": "Bosn_u_Herzegowina", "Griechenland": "Griechenland",
    "Italien": "Italien", "Kasachstan": "Kasachstan", "Kroatien": "Kroatien",
    "Niederlande": "Niederlande", "Österreich": "Oesterreich", "Polen": "Polen",
    "Rumänien": "Rumaenien", "Russische Föderation": "Russ_Foederation", "Türkei": "Tuerkei",
    "Ukraine": "Ukraine",
}
EU27 = {"Belgien", "Bulgarien", "Dänemark", "Estland", "Finnland", "Frankreich", "Kroatien",
        "Slowenien", "Griechenland", "Irland", "Italien", "Lettland", "Litauen", "Luxemburg",
        "Malta", "Niederlande", "Österreich", "Polen", "Portugal", "Rumänien", "Slowakei",
        "Schweden", "Spanien", "Tschechische Republik", "Ungarn", "Zypern"}
# (26 here + Germany = EU27.)

ARAB = ["Syrien", "Irak", "Libanon", "Marokko", "Algerien", "Tunesien", "Ägypten", "Jordanien",
        "Palästinensische Gebiete", "Libyen", "Jemen", "Sudan", "Saudi-Arabien",
        "Vereinigte Arabische Emirate", "Kuwait", "Katar", "Bahrain", "Oman", "Mauretanien"]
SPANISH_AMERICAS = ["Mexiko", "Kolumbien", "Venezuela", "Argentinien", "Chile", "Peru",
                    "Ecuador", "Kuba", "Bolivien", "Dominikanische Republik", "Guatemala",
                    "Honduras", "El Salvador", "Nicaragua", "Costa Rica", "Panama", "Paraguay",
                    "Uruguay"]

# Mikrozensus label -> (slug, citizenships). A citizenship may serve several languages (a weight
# is a shape, not a count). "@europe", "@africa", "@asia", "@foreign" are every citizenship of
# that part of the world (the table's own code ranges LAND1xx/2xx/4xx), or every non-German one.
LANGS = {
    "Deutsch": ("deutsch", ["Deutschland"]),
    "Türkisch": ("tuerkisch", ["Türkei"]),
    "Polnisch": ("polnisch", ["Polen"]),
    # Ukraine at half: Zensus day (15 May 2022) already counts 646k Ukrainians, most of them
    # refugees of that spring, against 241k Russian and 42k Kazakh citizens; Russian is the home
    # language of a part of them only.
    "Russisch": ("russisch", ["Russische Föderation", "Kasachstan", ("Ukraine", 0.5)]),
    "Arabisch": ("arabisch", ARAB),
    "Englisch": ("englisch", ["Vereinigtes Königreich",
                              "Vereinigtes Königreich/ Britische Überseegebiete", "Irland",
                              "Vereinigte Staaten", "Kanada", "Australien", "Neuseeland"]),
    "Französisch": ("franzoesisch", ["Frankreich", "Belgien", "Luxemburg", "Kamerun"]),
    "Italienisch": ("italienisch", ["Italien"]),
    "Spanisch": ("spanisch", ["Spanien"] + SPANISH_AMERICAS),
    "Niederländisch": ("niederlaendisch", ["Niederlande"]),
    "Portugiesisch": ("portugiesisch", ["Portugal", "Brasilien", "Angola", "Mosambik",
                                        "Cabo Verde", "Guinea-Bissau"]),
    "Rumänisch": ("rumaenisch", ["Rumänien", "Moldau, Republik"]),
    "Albanisch": ("albanisch", ["Albanien", "Kosovo", "Nordmazedonien (bis 2019: Mazedonien)"]),
    "Mazedonisch": ("mazedonisch", ["Nordmazedonien (bis 2019: Mazedonien)"]),
    "Kroatisch": ("kroatisch", ["Kroatien"]),
    "Serbisch": ("serbisch", ["Serbien", "Montenegro"]),
    "Bosnisch": ("bosnisch", ["Bosnien und Herzegowina"]),
    "Griechisch": ("griechisch", ["Griechenland", "Zypern"]),
    "Bulgarisch": ("bulgarisch", ["Bulgarien"]),
    "Ungarisch": ("ungarisch", ["Ungarn"]),
    "Ukrainisch": ("ukrainisch", ["Ukraine"]),
    "Dänisch": ("daenisch", ["Dänemark"]),
    "Kurdisch": ("kurdisch", ["Türkei", "Syrien", "Irak", "Iran"]),
    "Persisch": ("persisch", ["Iran", "Afghanistan"]),
    "Paschtu": ("paschtu", ["Afghanistan", "Pakistan"]),
    "Urdu": ("urdu", ["Pakistan"]),
    "Hindi": ("hindi", ["Indien"]),
    "Chinesisch": ("chinesisch", ["China", "China (Hongkong)", "China (Macau)", "Taiwan"]),
    "Vietnamesisch": ("vietnamesisch", ["Vietnam"]),
    "Eine andere in Europa gesprochene Sprache": ("andere_europa", ["@europe"]),
    "Eine andere in Asien gesprochene Sprache": ("andere_asien", ["@asia"]),
    "Eine andere in Afrika gesprochene Sprache": ("andere_afrika", ["@africa"]),
    "Eine sonstige Sprache": ("sonstige", ["@foreign"]),
}


# ---------------------------------------------------------------------------------- fetch
def fetch():
    import requests
    os.makedirs(RAW, exist_ok=True)
    for name in GRIDS:
        path = os.path.join(RAW, name)
        if os.path.exists(path):
            continue
        r = requests.get(GRID_BASE + name, headers={"User-Agent": UA}, timeout=600)
        r.raise_for_status()
        with open(path + ".part", "wb") as f:
            f.write(r.content)
        os.replace(path + ".part", path)
        print(f"  fetched {name} ({len(r.content):,} bytes)")
    path = os.path.join(RAW, ZDB_FILE)
    if not os.path.exists(path):
        h = {"User-Agent": UA, "Accept": "*/*",
             "Referer": "https://ergebnisse.zensus2022.de/datenbank/online/"}
        st = requests.get(f"{ZDB}/tables/{ZDB_TABLE}/structure", headers=h, timeout=300)
        st.raise_for_status()
        state = st.json()["initialState"]
        # default layout: Germany | Laender across, citizenship down. Swap the Laender block for
        # the Gemeinden block (GEOGM4); everything else as the site sends it.
        gm = [k for k, v in state["variableBlocks"].items() if v["mainVariable"] == "GEOGM4"]
        assert len(gm) == 1, gm
        col = state["tableStructure"]["colTitle"]
        assert state["variableBlocks"][col[1]["blockCode"]]["mainVariable"] == "GEOBL1", col
        col[1]["blockCode"] = gm[0]
        h["Content-Type"] = "application/json"
        r = requests.post(f"{ZDB}/tables/{ZDB_TABLE}/download/ffcsv/de", headers=h,
                          data=json.dumps(state), timeout=900)
        r.raise_for_status()
        with open(path + ".part", "wb") as f:
            f.write(r.content)
        os.replace(path + ".part", path)
        print(f"  fetched {ZDB_FILE} ({len(r.content):,} bytes)")


# ----------------------------------------------------------------------------------- read
def read_grid(zip_name):
    with zipfile.ZipFile(os.path.join(RAW, zip_name)) as z:
        inner = [n for n in z.namelist() if n.endswith(GRIDS[zip_name])]
        assert len(inner) == 1, (zip_name, z.namelist())
        with z.open(inner[0]) as fh:
            df = pd.read_csv(fh, sep=";", encoding="utf-8", dtype=str)
    df = df.set_index("GITTER_ID_1km").drop(columns=["x_mp_1km", "y_mp_1km"])
    bad = set()
    for c in df.columns:
        v = df[c].str.strip()
        ok = v.str.fullmatch(r"\d+") | (v == DASH)
        bad |= set(v[~ok].unique())
        df[c] = pd.to_numeric(v.where(v != DASH, "0"), errors="coerce").fillna(0)
    assert not bad, f"{zip_name}: unrecognised cells {sorted(bad)[:5]}"
    return df.astype(np.int64)


def read_table():
    """Gemeinde x citizenship counts -> DataFrame (index ars, columns = citizenship labels)."""
    with zipfile.ZipFile(os.path.join(RAW, ZDB_FILE)) as z:
        name = [n for n in z.namelist() if n.endswith(".csv")][0]
        with z.open(name) as fh:
            df = pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8-sig"), sep=";", dtype=str,
                             usecols=["1_variable_code", "1_variable_attribute_code",
                                      "2_variable_attribute_code", "2_variable_attribute_label",
                                      "value", "value_unit"])
    df = df[(df["1_variable_code"] == "GEOGM4") & (df["value_unit"] == "Anzahl")]
    v = df["value"].str.strip()
    nonnum = ~v.str.fullmatch(r"\d+")
    print(f"  table {ZDB_TABLE}: {len(df):,} Gemeinde x citizenship cells, "
          f"{int(nonnum.sum()):,} not a number ({sorted(v[nonnum].unique())[:6]}, read as 0)")
    df["n"] = pd.to_numeric(v.where(~nonnum, "0"))
    df = df[df["2_variable_attribute_code"].str.startswith("LAND", na=False)]
    codes = df.drop_duplicates("2_variable_attribute_code").set_index(
        "2_variable_attribute_code")["2_variable_attribute_label"]
    wide = df.pivot_table(index="1_variable_attribute_code", columns="2_variable_attribute_code",
                          values="n", aggfunc="sum", fill_value=0)
    return wide, codes


def _items(ctz):
    """LANGS entries are a label or (label, factor)."""
    return [(c, 1.0) if isinstance(c, str) else c for c in ctz]


# Citizenships a remainder group leaves out: every one a named language lists, and the
# German-speaking neighbours (their people are inside German, not "another European language").
NAMED = ({n for _, ctz in LANGS.values() for n, _ in _items(ctz) if not n.startswith("@")}
         | {"Österreich", "Schweiz", "Liechtenstein"})


def expand(ctz, codes):
    """A LANGS citizenship list -> {table code: factor}. "@europe", "@africa" and "@asia" are
    that part of the world's citizenships LESS those a named language already lists (the
    Mikrozensus' "another language spoken in Europe" is not Polish or Turkish); "@foreign" is
    every foreign citizenship."""
    lab2code = {lab: c for c, lab in codes.items()}
    land = [c for c in codes.index if c != "LAND000"]
    out = {}
    for name, f in _items(ctz):
        if name.startswith("@"):
            digit = {"@foreign": None, "@europe": "1", "@africa": "2", "@asia": "4"}[name]
            for c in land:
                if digit is None or (c[4] == digit and codes[c] not in NAMED):
                    out[c] = f
        else:
            if name not in lab2code:
                raise SystemExit(f"citizenship {name!r} is not a label in {ZDB_TABLE}")
            out[lab2code[name]] = f
    return out


def proxy_of(code, codes):
    """The grid column that places one citizenship inside a Gemeinde: its own column on the
    12-country grid, else its group's column less the countries the 12-country grid already
    carries (`EU27_rest`, `Europa_rest`), else `Sonstige_Welt`."""
    lab = codes[code]
    if code == "LAND000":
        return "Deutschland"
    if lab in GRID_COUNTRY:
        return GRID_COUNTRY[lab]
    if lab in EU27:
        return "EU27_rest"
    if code.startswith("LAND1"):
        return "Europa_rest"
    return "Sonstige_Welt"


def cell_ids(place):
    """INSPIRE 1km cell id of each layer row, from its (clipped) geometry; None for the
    Gemeinde polygons religiondots added where no cell centre fell."""
    p = place.to_crs(3035)
    b = p.geometry.bounds
    c = p.geometry.centroid
    x0 = np.floor(c.x.to_numpy() / 1000) * 1000
    y0 = np.floor(c.y.to_numpy() / 1000) * 1000
    inside = ((b.minx.to_numpy() >= x0 - 1) & (b.maxx.to_numpy() <= x0 + 1001)
              & (b.miny.to_numpy() >= y0 - 1) & (b.maxy.to_numpy() <= y0 + 1001))
    big = ((b.maxx - b.minx) > 1001) | ((b.maxy - b.miny) > 1001)
    ids = pd.Series("CRS3035RES1000mN" + y0.astype(np.int64).astype(str) + "E"
                    + x0.astype(np.int64).astype(str), index=place.index)
    ids[big.to_numpy()] = None
    odd = (~inside) & (~big.to_numpy())
    return ids, int(big.sum()), int(odd.sum())


# ----------------------------------------------------------------------------------- main
def main():
    if "--fetch" in sys.argv:
        fetch()
    import geopandas as gpd

    print("reading religiondots' 1km layer (read-only)…")
    place = gpd.read_file(RD_GRID)
    place = place[["ars", "pop", "geometry"]].copy()
    place["ars"] = place["ars"].astype(str)
    ids, n_big, n_odd = cell_ids(place)
    print(f"  {len(place):,} rows; {n_big} whole-Gemeinde polygons (no 1km centre fell in them); "
          f"{n_odd} cell(s) whose geometry overhangs its recovered square")
    dup = ids.dropna().duplicated(keep=False)
    if dup.any():
        # A cell split between two rows would hand both its whole count. Keep it on the larger
        # piece; the other piece keeps population weight only.
        d = ids.dropna()[dup]
        area = place.loc[d.index].to_crs(3035).area
        for cid, grp in area.groupby(d):
            for i in grp.index.drop(grp.idxmax()):
                ids[i] = None
        print(f"  {int(dup.sum())} rows share {d.nunique()} cell id(s): kept on the larger piece")
    assert not ids.dropna().duplicated().any()

    ctry = read_grid(list(GRIDS)[0])
    grp = read_grid(list(GRIDS)[1])
    print(f"  grids: {len(ctry):,} cells (12 countries), {len(grp):,} cells (groups)")
    have = ids.dropna()
    for name, g in (("12-country", ctry), ("groups", grp)):
        miss = ~have.isin(g.index)
        mpop = place.loc[miss[miss].index, "pop"].sum()
        print(f"  layer -> {name} grid: {int(miss.sum()):,} layer cells missing "
              f"({int(mpop):,} people; they take zero there)")
        # the 12-country file leaves out a few hundred tiny cells; the groups file has them all
        assert mpop < 0.0005 * place["pop"].sum(), f"{name}: too many layer cells missing"
        out = g.index.difference(have.to_numpy())
        print(f"  {name} grid -> layer: {len(out):,} grid cells not in the layer "
              f"({int(g.loc[out, 'Insgesamt_Bevoelkerung'].sum()):,} people: centres in no "
              f"Gemeinde, dropped by religiondots)")
        # religiondots' record: 1,436 cells, 159,392 people (sources/de_grid.md)
        assert len(out) < 1500
        assert g.loc[out, "Insgesamt_Bevoelkerung"].sum() < 0.003 * g["Insgesamt_Bevoelkerung"].sum()
    cols = pd.concat([ctry.drop(columns=["Insgesamt_Bevoelkerung", "Deutschland"]), grp], axis=1)
    cells = cols.reindex(ids.fillna("-").to_numpy()).fillna(0).to_numpy(dtype=float)
    cells = pd.DataFrame(cells, columns=cols.columns, index=place.index)

    wide, codes = read_table()
    print(f"  table: {wide.shape[0]:,} Gemeinden x {wide.shape[1]} citizenships "
          f"({int(wide.drop(columns=['LAND000'], errors='ignore').sum().sum()):,} foreign "
          f"citizens, {int(wide['LAND000'].sum()):,} German)")
    t_ars, p_ars = set(wide.index), set(place["ars"])
    only_t, only_p = sorted(t_ars - p_ars), sorted(p_ars - t_ars)
    tot = wide.sum(axis=1)
    print(f"  join, table -> layer: {len(t_ars & p_ars):,} matched, {len(only_t)} table-only "
          f"({int(tot.reindex(only_t).sum()):,} people)")
    print(f"  join, layer -> table: {len(only_p)} layer-only Gemeinden "
          f"({int(place.loc[place['ars'].isin(only_p), 'pop'].sum()):,} people)")
    assert not only_t and not only_p, "the Gemeinde table and the layer do not join both ways"

    # national totals: table vs 12-country grid
    lab2code = {lab: c for c, lab in codes.items()}
    print("  national citizens, table vs 12-country grid:")
    for lab, gc in GRID_COUNTRY.items():
        a, b = wide[lab2code[lab]].sum(), ctry[gc].sum()
        print(f"    {lab:26s} {a:>10,}  {b:>10,}  {100 * (b / a - 1):+5.1f}%")
        assert abs(b / a - 1) < 0.05, lab

    # ---- weights
    labels = pd.read_csv(NORM, dtype={"geo_id": str})
    labels = set(labels.loc[labels["geo_level"] == "land", "source_category"])
    missing = labels - set(LANGS)
    assert not missing, f"Mikrozensus labels with no citizenship mapping: {sorted(missing)}"
    ars = place["ars"].to_numpy()
    pop = place["pop"].to_numpy(dtype=float)
    foreign = cells["Ausland_Sonstige"].to_numpy()
    in_table = np.isin(ars, list(t_ars))

    # group columns less the countries the 12-country grid carries on their own
    eu_own = [GRID_COUNTRY[l] for l in GRID_COUNTRY if l in EU27]
    eur_own = [GRID_COUNTRY[l] for l in GRID_COUNTRY if l not in EU27]
    cells["EU27_rest"] = (cells["EU27_Land"] - cells[eu_own].sum(axis=1)).clip(lower=0)
    cells["Europa_rest"] = (cells["Sonstiges_Europa"] - cells[eur_own].sum(axis=1)).clip(lower=0)
    # the grid's groups against the table's code ranges, nationally
    europe = [c for c in codes.index if c.startswith("LAND1")]
    t_eu = wide[[c for c in europe if codes[c] in EU27]].sum().sum()
    t_eur = wide[[c for c in europe if codes[c] not in EU27]].sum().sum()
    t_w = wide[[c for c in codes.index if c != "LAND000" and not c.startswith("LAND1")]].sum().sum()
    print("  citizenship groups, table vs groups grid: "
          f"EU27 {t_eu:,} vs {int(cells['EU27_Land'].sum()):,}; other Europe {t_eur:,} vs "
          f"{int(cells['Sonstiges_Europa'].sum()):,}; rest of the world {t_w:,} vs "
          f"{int(cells['Sonstige_Welt'].sum()):,} (grid `Sonstige`, stateless and unknown: "
          f"{int(cells['Sonstige'].sum()):,})")
    for a, b in ((t_eu, cells["EU27_Land"].sum()), (t_eur, cells["Sonstiges_Europa"].sum()),
                 (t_w, cells["Sonstige_Welt"].sum())):
        assert abs(b / a - 1) < 0.05, "the grid's groups are not the table's code ranges"

    def within(sig):
        """Each cell's share of its Gemeinde, by `sig`, then all foreign, then population."""
        s = pd.DataFrame({"sig": sig, "for": foreign, "pop": pop}).groupby(ars).transform("sum")
        s_sig, s_for, s_pop = (s[c].to_numpy() for c in ("sig", "for", "pop"))
        by_sig = s_sig > 0
        by_for = ~by_sig & (s_for > 0)
        by_pop = ~by_sig & ~by_for & (s_pop > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(by_sig, sig / s_sig,
                           np.where(by_for, foreign / s_for, np.where(by_pop, pop / s_pop, 0.0)))
        used = {"signal": len(set(ars[by_sig])), "foreign": len(set(ars[by_for])),
                "pop": len(set(ars[by_pop]))}
        return np.nan_to_num(out), used

    assert in_table.all(), "layer Gemeinden missing from the table"
    shares, used_by = {}, {}
    for col in ["Deutschland", "EU27_rest", "Europa_rest", "Sonstige_Welt",
                *GRID_COUNTRY.values()]:
        shares[col], used_by[col] = within(cells[col].to_numpy())

    report = []
    for lab, (slug, ctz) in LANGS.items():
        tcodes = expand(ctz, codes)
        by_proxy = {}
        for c, f in tcodes.items():
            by_proxy.setdefault(proxy_of(c, codes), []).append((c, f))
        w = np.zeros(len(place))
        parts = []
        for proxy, items in by_proxy.items():
            G = sum(wide[c] * f for c, f in items)          # people per Gemeinde
            w += pd.Series(ars).map(G).fillna(0).to_numpy(dtype=float) * shares[proxy]
            parts.append(f"{proxy} {int(G.sum()):,}")
        place[f"w_{slug}"] = w
        report.append((lab, len(tcodes), "; ".join(parts)))

    print("\n  each language: how many citizenships, and their people by the grid column that "
          "places them inside a Gemeinde:")
    for lab, n, parts in report:
        print(f"    {lab:42s} {n:>3}  {parts}")
    print("  Gemeinden where a column sums to zero (Cell-Key) and all foreign, then population, "
          "stand in: " + ", ".join(f"{c} {u['foreign']}+{u['pop']}" for c, u in used_by.items()))

    # ---- the city check: share of weight in Gemeinden of 100,000+ people, per language
    gem_pop = pd.Series(pop).groupby(ars).sum()
    big = set(gem_pop.index[gem_pop >= 100_000])
    in_big = np.isin(ars, list(big))
    print(f"\n  weight in the {len(big)} Gemeinden of 100,000+ (they hold "
          f"{100 * pop[in_big].sum() / pop.sum():.1f}% of the population):")
    for lab, (slug, _) in LANGS.items():
        w = place[f"w_{slug}"].to_numpy()
        print(f"    {lab:42s} {100 * w[in_big].sum() / w.sum():5.1f}%")

    os.makedirs(OUT_DIR, exist_ok=True)
    place["cell"] = ids
    tmp = OUT + ".tmp.gpkg"
    if os.path.exists(tmp):
        os.remove(tmp)
    place.to_file(tmp, layer="grid1km", driver="GPKG")
    os.replace(tmp, OUT)
    print(f"\nwrote {OUT} ({len(place):,} rows, {len(LANGS)} weight columns)")


if __name__ == "__main__":
    main()
