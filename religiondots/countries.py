"""
Per-country wiring for the scatter step: where the counts are, where the polygons are, and
how a source category becomes a religiondots node.

Everything downstream of this file is country-agnostic. Everything above it is per-country by
necessity — the sources do not agree about anything, and spec §3.9, §8.1 and §2.3 are three
different ways of saying so.

Each entry supplies:
  counts()      -> DataFrame [unit, node, count]   unit = the geography the counts are ON
                optionally `tier` (spec §7): `measured` / `derived` / `modelled`, per row.
                Missing means `measured`, which is right for a census read at its own
                geography and wrong for anything that was spread, so an adapter that spreads
                must say so. `derived` and `modelled` draw DESATURATED, and the weakest tier
                on a (unit, node) pair wins — a pair that is part measurement and part
                estimate is not a measurement.
  units         path to the polygons for `unit`, and the column holding its id
  place         path to a finer polygon layer used to place dots inside a unit, and the
                column linking it back to `unit`. spec §8.2: these units are designed to a
                population target, so an equal share of dots per unit is already a population
                weighting and no population data is read.

and, for the viewer, which draws one country at a time and needs to say whose data it is:
  name          what the country is called in the picker, which is also what it sorts by
  name_in       optional, the same name in the sentence "Religion in ___" — the article is
                part of the name in English for some countries and not others, and the
                picker wants "United States" where the title wants "the United States"
  source        the agency and instrument, one line, under the title
  basis         which quantity the numbers are (spec §3.1) — never mixed, and now never
                silently mixed either, since two countries are no longer on screen at once
  note_public   the country's own caveat, shown in the about panel when it is selected. The
                per-country note that used to be a cross-border paragraph belongs here: with
                a single country on screen the interesting comparison is inside it.
  view          optional [w, s, e, n] to fly to, where the data bbox is the wrong picture —
                the US spans Hawaii to Maine and fitting that shows mostly ocean. Defaults to
                the bbox of the country's own dots, computed in tiles.py.
"""
from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "taxonomy"))


def _us_counts():
    """ASARB 2020: county x 372 bodies, already mapped to the taxonomy by hand."""
    paths = pd.read_csv(HERE / "taxonomy" / "usrc_groups.csv", dtype={"Group Code": str})
    path_of = dict(zip(paths["Group Code"], paths["path"]))
    df = pd.read_excel(HERE / "data" / "raw" / "2020_USRC_Group_Detail.xlsx",
                       sheet_name="2020 Group by County",
                       dtype={"FIPS": str, "Group Code": str})
    df["unit"] = df["FIPS"].str.strip().str.zfill(5)
    df["node"] = df["Group Code"].str.strip().str.zfill(3).map(path_of)
    df = df[df["node"].notna() & (df["node"] != "UNMAPPED")]
    return df.rename(columns={"Adherents": "count", "Congregations": "congregations"})[
        ["unit", "node", "count", "congregations"]]


def _us_counts_rebased():
    """spec §3.5a: ASARB's rolls, plus the self-identification residual on top.

    ASARB's numbers are untouched — every one of its 372 bodies keeps the county figure it
    always had, tagged `measured`. What is added is one row per (county, root) for the people
    the survey finds and no roll holds, tagged `derived` — recorded per §7, drawn identically
    to everything else since the desaturation was removed on 2026-09-04.

    `_us_counts` stays ASARB-only ON PURPOSE and is not merely an implementation detail:
    us_weights.py fits its §8.4 demographic model against it, and fitting a model of where
    ASARB's adherents live against rows that are a survey residual would be training on the
    output. The two functions must not be merged.
    """
    from us_rebase import residual_counts

    roll = _us_counts()
    res = residual_counts(roll)      # measured against the roll as drawn, not ASARB's state sheet
    roll["may_ring"] = True
    roll["tier"] = "measured"
    return pd.concat([roll, res], ignore_index=True)


def _us_place_weight(place):
    """spec §8.4 — imported lazily so a country that does not use it never pays for it."""
    from us_weights import load_weighter

    return load_weighter(place)


def _ca_counts():
    """StatCan 2021 at CSD, allocated to 147 categories, mapped at branch level (§2.4)."""
    from ca2021 import resolve

    src = pd.read_csv(HERE / "data" / "normalized" / "ca.csv",
                      dtype={"geo_id": str}, low_memory=False)
    src["parent"] = src["note"].str.extract(r"parent=([^;]*)")
    prov = src[src.geo_level == "province"]
    parent_of = (prov.dropna(subset=["parent"]).drop_duplicates("source_category")
                 .set_index("source_category")["parent"].to_dict())
    parent_of = {k: (v if isinstance(v, str) and v else None) for k, v in parent_of.items()}

    df = pd.read_csv(HERE / "data" / "normalized" / "ca_csd_allocated.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df["node"] = df["source_category"].map(lambda c: resolve(c, parent_of))
    df = df[df["node"].notna()]
    df["congregations"] = 0
    # spec §3.10: an allocated count may never become a ring, because a ring asserts presence
    # and allocation only spreads a total. `tier` is `measured` where a fine column had a
    # single child (nothing was allocated) and `derived` otherwise.
    df["may_ring"] = df["tier"] == "measured"
    return df.rename(columns={"geo_id": "unit"})[
        ["unit", "node", "count", "congregations", "may_ring", "tier"]]


def _cz_counts():
    """ČSÚ 2021 at the finest available unit: 78 categories, mapped at branch level.

    The only country so far that needs no allocate.py step — it publishes its finest
    categories at its finest geography, so nothing here is derived and every row may ring
    (spec §3.9/§3.10, sources.md §9b).
    """
    from cz2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "cz.csv",
                     dtype={"geo_id": str}, low_memory=False)

    # ONE level per place, and the two levels are alternatives. The file carries the whole
    # territorial hierarchy — obec, city district, ORP, okres, kraj, NUTS2, country — so
    # reading it as delivered counts the country eight times over.
    #
    # City districts subdivide 8 statutory cities and REPLACE them: Prague is 1,301,432
    # people in one obec, 12.4% of the country in a single polygon, and 57 city districts
    # instead. sources/cz_geo.py derives which 8 obce are replaced (spatially, from the
    # polygons) and writes the list; the matching polygon layer is cz_finest.gpkg. Both
    # levels are published by ČSÚ, so this is measured data and not an allocation.
    replaced = set(pd.read_csv(HERE / "data" / "geo" / "cz" / "cz_replaced.csv",
                               dtype=str)["kod"])
    df = df[((df["geo_level"] == "municipality") & ~df["geo_id"].isin(replaced))
            | (df["geo_level"] == "city_district")]

    # DROP THE EXPLICIT ZEROS, and they are most of the file. Because ČSÚ publishes a
    # complete partition it emits a row for every category in every municipality whether
    # anyone is there or not: 417,083 of 494,066 municipal rows are zeros, 84% of the file.
    # ASARB and the other sources list only what they found, so nothing before Czechia had
    # to think about this.
    #
    # Keeping them costs nothing in dots — zero people is zero dots — but a ring asserts
    # PRESENCE (spec §4.3), and every zero would become a ring claiming a body is in a
    # village it is not in. Left in, Czechia drew 277,987 rings against 7,346 dots, which
    # is a near-solid mask of false claims.
    df = df[df["count"] > 0]

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna()]
    df["congregations"] = 0
    return df.rename(columns={"geo_id": "unit"})[
        ["unit", "node", "count", "congregations"]]


def _br_counts():
    """IBGE Censo 2010 at municipio: 56 categories, mapped at branch level.

    2010 rather than 2022 because 2022 publishes NINE categories and lumps 47.4M
    evangelicals into one (sources/br.md §1). spec §3.4's rescale — 2022 municipal totals
    split by 2010 municipal shares — is the intended end state and is not done yet, so
    Brazil is drawn at its own year and note_public says so.
    """
    from br2010 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "br.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[(df["year"] == 2010) & (df["geo_level"] == "municipio")]

    # TAKE ONLY THE LEAVES OF IBGE'S OWN TREE. Classification 133 is nested three deep and
    # every level is published at municipio, so counting the file as delivered counts most
    # Brazilians three times — once as "Evangélicas", once as "Evangélicas de origem
    # pentecostal", once as "Igreja Assembléia de Deus". br.py writes each row's code and
    # parent into `note`; a code that is nobody's parent is a leaf. Derived rather than
    # hand-listed so a category IBGE adds later cannot silently be double counted.
    code = df["note"].str.extract(r"code=(\d+)")[0]
    parent = df["note"].str.extract(r"parent=(\d*)")[0]
    internal = set(parent.dropna()) - {""}
    df = df[code.notna() & ~code.isin(internal)]

    # As Czechia: IBGE emits an explicit zero where nobody reported a category, and a zero
    # is an absence. Left in, they would become presence rings (spec §4.3).
    df = df[df["count"] > 0]

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna()]
    df["congregations"] = 0
    return df.rename(columns={"geo_id": "unit"})[
        ["unit", "node", "count", "congregations"]]


def _allocated_counts(cc, fine, module):
    """The shared shape for a source that arrived through allocate.py.

    au / ie / mx all split fine categories from fine geography (spec §3.9), so the file
    countries.py reads is `<cc>_<fine>_allocated.csv` rather than `<cc>.csv`: it carries
    every category at the fine geography, tagged `measured` where a fine column had a
    single child and `derived` where a coarse total was spread out.

    §3.10: a derived count may never become a ring. Allocation spreads a total and cannot
    establish that anyone is present, so `may_ring` is exactly `tier == "measured"`.
    """
    df = pd.read_csv(HERE / "data" / "normalized" / f"{cc}_{fine}_allocated.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df["node"] = df["source_category"].map(module.resolve)
    df = df[df["node"].notna()]
    # An allocated zero is an absence, and an absence must not become a presence ring
    # (§4.3) — the lesson Czechia's 417,083 explicit zeros taught.
    df = df[df["count"] > 0]
    df["congregations"] = 0
    df["may_ring"] = df["tier"] == "measured"
    return df.rename(columns={"geo_id": "unit"})[
        ["unit", "node", "count", "congregations", "may_ring", "tier"]]


def _au_counts():
    """ABS 2021 at SA2, allocated to 148 categories (§2.4, branch level)."""
    import au2021
    return _allocated_counts("au", "sa2", au2021)


def _ie_counts():
    """CSO 2022 at Small Area, allocated to 24 categories."""
    import ie2022
    return _allocated_counts("ie", "small_area", ie2022)


def _mx_counts():
    """INEGI 2020 at municipio, allocated to 23 categories."""
    import mx2020
    return _allocated_counts("mx", "municipio", mx2020)


def _nz_counts():
    """Stats NZ at SA2, allocated to 159 categories.

    NOTE: these are RESPONSES, not people — the census allows up to four religions per
    person, so the categories sum to 5,003,112 against an SA2 population of 4,993,920.
    A New Zealand dot is a response; every other country's dot is a person (nz2023.py).
    """
    import nz2023
    return _allocated_counts("nz", "sa2", nz2023)


def _uk_counts():
    """The UK's three drawn censuses, unioned onto one unit namespace.

    England and Wales (ONS, 2021) and Northern Ireland (NISRA, 2021) arrive allocated;
    Scotland (NRS, 2022) publishes its 13 categories at Output Area already and needs no
    allocation, so it is read straight from uk.csv and every Scottish row is `measured`.

    The three code namespaces are disjoint — E00/W00, S00, N20 — so they share one `unit`
    column without a prefix (sources/uk_geo.py checks this rather than assuming it).

    NISRA's second question, religion brought up in, is a different variable and is not
    read here at all (sources/uk.md §1).
    """
    import uk2021

    frames = []
    for stem in ("uk_ew", "uk_ni"):
        d = pd.read_csv(HERE / "data" / "normalized" / f"{stem}_allocated.csv",
                        dtype={"geo_id": str}, low_memory=False)
        d["may_ring"] = d["tier"] == "measured"
        frames.append(d[["geo_id", "source_category", "count", "may_ring", "tier"]])

    sc = pd.read_csv(HERE / "data" / "normalized" / "uk.csv",
                     usecols=["geo_id", "geo_level", "source_category", "count",
                              "source_id"],
                     dtype={"geo_id": str}, low_memory=False)
    sc = sc[(sc["source_id"] == "uk_sc_census_2022")
            & (sc["geo_level"] == "output_area")].copy()
    sc["may_ring"] = True                      # nothing was allocated; all measured
    sc["tier"] = "measured"
    frames.append(sc[["geo_id", "source_category", "count", "may_ring", "tier"]])

    df = pd.concat(frames, ignore_index=True)
    df["node"] = df["source_category"].map(uk2021.resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    return df.rename(columns={"geo_id": "unit"})[
        ["unit", "node", "count", "congregations", "may_ring", "tier"]]


def _pl_counts():
    """GUS NSP 2021 at gmina: 139 named churches on 2,477 units, mapped at branch level.

    The second country after Czechia that needs no allocate.py step — GUS publishes named
    denominations at its finest geography, so nothing here is derived and every row may
    ring (spec §3.9/§3.10, sources.md §9e).

    ONE level, and the file carries four. pl.csv holds gmina, powiat, voivodeship and
    country, which are the same 38 million people counted four times; reading it as
    delivered would quadruple the country.

    The join key is SIX digits, not the seven GUS prints. TERYT's seventh digit is the
    gmina TYPE (1 urban / 2 rural / 3 mixed), and the GISCO LAU boundaries do not carry
    it, so `pl_gminy.gpkg` is keyed on the first six — which are already unique per gmina.
    sources/pl_geo.py derives that key and checks the join both ways.

    Unlike Czechia there are no explicit zeros to strip: GUS lists only the denominations
    it found in a gmina, so the 21,926 gmina rows are all positive and every one may ring.
    """
    from pl2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "pl.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"] == "gmina"].copy()

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["unit"] = df["geo_id"].str[:6]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


def _in_counts():
    """Census of India 2011 at sub-district: 91 categories on 5,988 units.

    ONE level. in.csv carries nation, state, district and subdistrict — the same 1.21
    billion people counted four times — and the allocated file adds a fifth reading of the
    finest one. Only `in_subdistrict_allocated.csv` is read here; it already contains the
    six census religions untouched plus the `Other religions and persuasions` bucket split
    into its 83 named religions, so reading in.csv as well would double the country.

    WHAT IS DERIVED AND WHAT IS NOT, because India's ratio is unusually good. The six
    religions — 99.34% of the population — are published at the sub-district and are
    `measured`. Only the 7,937,734 people in `Other religions and persuasions` are split
    from state-level structure, so India is 0.66% estimated against Australia's much larger
    share. And within that 0.66%, 245 of the (state, column) pairs have a single named
    religion and are therefore exact rather than allocated.

    THE ALLOCATION IS WITHIN EACH STATE, which is what makes it defensible at all. India is
    the first source where `allocate.py --within` was needed and the reason is visible in
    one line: Sanamahi is 100% Manipur, Niam Khasi 100% Meghalaya, Donyi-Polo 98% Arunachal
    Pradesh. Pooling the states into one national composition — which is what every earlier
    country does — would have put Manipuri and Arunachali religions into every sub-district
    in India in proportion to its `Other` count. Allocated within states, each religion
    reproduces its published state distribution exactly.

    The Annexure's 47 write-in sects are in in.csv and resolve to None here on purpose;
    see taxonomy/in2011.py for why a table that names 573 Shia Muslims is not a sect
    breakdown.
    """
    from in2011 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "in_subdistrict_allocated.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["unit"] = df["geo_id"]
    df["congregations"] = 0
    # spec §3.10: an allocated count may never ring, because a ring asserts presence and
    # allocation only spreads a total. India currently draws no rings at all — every node
    # reaches a dot somewhere — so this changes nothing today, and it is set anyway
    # because scatter.py defaults a missing `may_ring` to True. Without it, a future dot
    # value or a shrunken category would let an allocated Adivasi religion claim presence
    # in a sub-district that may have none of it.
    df["may_ring"] = df["tier"] == "measured"
    return df[["unit", "node", "count", "congregations", "may_ring", "tier"]]


def _ro_counts():
    """INS RPL 2021 at UAT: 23 recognised cults on 3,181 units, mapped at branch level.

    ONE level; ro.csv also carries judet and country, which are the same 19 million people
    counted again.

    THE KEY IS INDIRECT. The census publishes no SIRUTA code — rows are named only — so
    `geo_id` here is the string "COUNTY|NAME", and sources/ro_geo.py resolves it to a
    SIRUTA code through the Eurostat LAU-NUTS correspondence table and writes the result
    to ro_uat_lookup.csv. That resolution is where the work is (name folding, ş/ș, and
    four places settled by elimination inside their county), and it is done once there
    rather than every time this runs.

    INS SUPPRESSES. `*` marks a confidential cell and sources/ro.py drops those rows
    rather than guessing, so 16,493 people — 0.087% of the country — are in a category
    somewhere and not in any row here. Nothing else is lost: the totals reconcile exactly.
    """
    from ro2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "ro.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"] == "uat"].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "ro" / "ro_uat_lookup.csv",
                      dtype={"geo_id": str, "kod": str})
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["kod"])))
    missing = df["unit"].isna().sum()
    if missing:
        raise SystemExit(f"{missing} ro.csv rows have no SIRUTA code -- re-run "
                         "sources/ro_geo.py, the lookup is stale")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


def _ee_counts():
    """Statistics Estonia 2021 at the finest unit it publishes: 21 categories, branch level.

    TWO LEVELS, AND THEY ARE ALTERNATIVES. The 8 Tallinn linnaosad REPLACE Tallinn; they do
    not nest under it for drawing. Tallinn as one municipality is 33.07% of the 15+
    population in a single 159 km² polygon, which would have been the worst capital case on
    the map by a factor of three. Statistics Estonia publishes RL21452 for the districts as
    well, so this is measured, not allocated — the same situation as Czechia.

    THE KEY IS A SLICE of the 14-character PxWeb place code, which concatenates EHAK codes:
    a municipality is `code[4:8]` and a city district is `code[8:12]`. sources/ee_geo.py
    checks the two namespaces do not collide, and re-keys four polygons whose EHAK code
    changed between the census and the 2024 boundary release.

    EVERYTHING IS ROUNDED TO BASE 10 (spec §3.8), so nothing reconciles exactly and is not
    meant to. The universe is persons aged 15 and over — no Estonian child is drawn.
    """
    from ee2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "ee.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"].isin(("municipality", "city_district"))].copy()
    df["unit"] = [c[4:8] if lv == "municipality" else c[8:12]
                  for c, lv in zip(df["geo_id"], df["geo_level"])]

    replaced = set(pd.read_csv(HERE / "data" / "geo" / "ee" / "ee_replaced.csv",
                               dtype=str)["kod"])
    df = df[~((df["geo_level"] == "municipality") & (df["unit"].isin(replaced)))]

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


def _hr_counts():
    """DZS Popis 2021 at town/municipality: 12 categories, mapped at branch level.

    ZAGREB'S 17 DISTRICTS ARE SUMMED BACK INTO ONE UNIT, which is the reverse of what
    Czechia and Estonia do and is not a choice. DZS publishes religion for the 17 gradske
    četvrti and not for Grad Zagreb as a whole — the census has 555 municipalities where
    Croatia has 556 — but no boundary source for the districts was found (GISCO stops at
    the municipality, and OSM has nothing at admin_level 9 or 10 inside Zagreb). So the
    data supports the split and the geometry does not, and 18.4% of Croatia is one polygon.
    sources/hr_geo.py writes the lookup that routes all 17 districts to LAU 01333.

    The census carries NO geographic codes — rows are (županija, name) — so the resolution
    to LAU codes is done once in sources/hr_geo.py and read from disk here, as Romania does.
    """
    from hr2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "hr.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"].isin(("municipality", "city_district"))].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "hr" / "hr_lookup.csv",
                      dtype={"geo_id": str, "kod": str})
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["kod"])))
    missing = df["unit"].isna().sum()
    if missing:
        raise SystemExit(f"{missing} hr.csv rows have no LAU code -- re-run "
                         "sources/hr_geo.py, the lookup is stale")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    # the 17 Zagreb districts collapse onto one unit, so re-aggregate
    return (df.groupby(["unit", "node"], as_index=False)
              .agg(count=("count", "sum"), congregations=("congregations", "max")))


def _nz_place_unit(g):
    """SA1 -> its SA2, via the concordance nz_geo.md §5 derived spatially.

    Stats NZ publishes no SA1->SA2 lookup reachable without a datafinder key, and neither
    the SA1 boundary service nor the meshblock service carries an SA2 column. SA1s nest
    inside SA2s by construction, so the spatial join that produced the CSV is exact.

    LANDWATER 21 is Inland Water — 71 SA1s holding six people between them. Mapped to NaN
    so `groupby` drops them and no dot is ever placed in a lake.
    """
    lut = pd.read_csv(HERE / "data" / "geo" / "nz" / "sa1_2023_to_sa2_2023.csv", dtype=str)
    sa2 = dict(zip(lut["SA12023_V1_00"], lut["SA22023_V1_00"]))
    unit = g["SA12023_V1_00"].astype(str).map(sa2)
    return unit.where(g["LANDWATER"].astype(str) != "21")


class _DeGridWeighter:
    """Place a German dot on where that religion actually is, not on where people are.

    spec §8.2 splits placement from magnitude, and everywhere else on this map the
    placement weight is a PROXY — an equal share per unit that was engineered to a
    population target, or, for the US, a demographic model fitted to guess a
    denomination's position inside a county (§8.4). Germany needs neither: destatis
    publishes the same three categories on the 1km INSPIRE grid, so the weight for
    `christianity.catholic` inside Munich is Munich's own per-cell Catholic count.

    That makes this the only country here whose within-unit placement is MEASURED. It is
    not a model and carries no fitted parameter, so §7's confidence machinery has nothing
    to mark: a Catholic dot in Neukölln is there because the register put Catholics in
    that square kilometre.

    Falls back to cell population, and then to equal shares, wherever the node has no
    column or sums to zero in that unit — the latter happens where the Cell-Key
    perturbation zeroed a small category across every cell of a small Gemeinde while the
    Gemeinde table still shows a few people (sources/de.md §3).
    """

    COLUMN = {
        "christianity.catholic": "kath",
        "christianity.protestant": "ev",
        "unrecorded": "son",
    }

    def __init__(self, place):
        self.pop = place["pop"].to_numpy(dtype=float)
        self.col = {k: place[v].to_numpy(dtype=float) for k, v in
                    ((n, c) for n, c in self.COLUMN.items() if c in place.columns)}
        self.n_measured = 0
        self.n_pop = 0
        self.n_uniform = 0

    def weights(self, node, idx, count, plain=False):
        col = self.col.get(node)
        if col is not None:
            w = col[idx]
            if w.sum() > 0:
                self.n_measured += 1
                return w
        pop = self.pop[idx]
        if pop.sum() > 0:
            self.n_pop += 1
            return pop
        self.n_uniform += 1
        return None

    def summary(self):
        return (f"{self.n_measured:,} (unit, node) rows placed on that religion's OWN 1km "
                f"grid counts, {self.n_pop:,} on cell population where the category is "
                f"zero across the unit's cells, {self.n_uniform:,} on equal shares "
                f"(sources/de_grid.py)")


def _de_place_weight(place):
    """countries.py hook. `place` is the 1km grid GeoDataFrame scatter.py has read."""
    if "kath" not in place.columns:
        print("  !! de_grid_1km.gpkg has no religion columns — run sources/de_grid.py; "
              "placing on equal shares (§8.2)")
        return None
    return _DeGridWeighter(place)


def _de_counts():
    """Zensus 2022 at Gemeinde: three categories on 10,786 units.

    The shallowest country on the map and the only one where that is a property of the
    instrument. Zensus 2022 asks nothing about religion; the figures are read off the
    Melderegister, which records membership of the two churches that levy church tax, so
    `basis` is `roll` (spec §3.1) and 51.8% of the country lands on one node.

    No allocation step, and none is possible: destatis publishes at no coarser AND no
    finer CATEGORY than this. What it does publish finer is geography — the same three
    numbers on a 100m grid — which is a placement upgrade rather than a detail upgrade.

    ONE level. de.csv carries `country` alongside `gemeinde`, which is the same 82.7
    million people counted twice.

    Every category is positive somewhere and the three partition each Gemeinde, so unlike
    Czechia there are explicit zeros to strip: 178 cells across the file are the true-zero
    dash, mostly Catholics in East German villages.
    """
    from de2022 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "de.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"] == "gemeinde"].copy()

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["unit"] = df["geo_id"]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


def _hu_counts():
    """Népszámlálás 2022 at settlement: 28 categories on 3,177 units.

    ONE level, and only the allocated file is read. hu.csv carries settlement, county and
    country — the same 9.6 million people counted three times — and
    `hu_settlement_allocated.csv` already holds every settlement column untouched plus the
    three that WBS008 refines, so reading hu.csv as well would double the country.

    98.1% of it is MEASURED. The allocation only touches three of the eleven settlement
    columns — Orthodox Christian, Other Christian denomination, and the non-Christian
    bucket, 184,147 people between them — and 160 of the (vármegye, column) pairs have a
    single category and so come out exact rather than derived. The other eight columns,
    including all 2.6M Roman Catholics and all 944k Calvinists, are published at the
    settlement itself.

    ALLOCATED WITHIN EACH VÁRMEGYE, not pooled. Hungary's minority churches are as
    regional as India's: the Romanian Orthodox are along the Romanian border, the Serbian
    Orthodox around Szentendre and Lórév, the Greek Catholics overwhelmingly in
    Szabolcs-Szatmár-Bereg. A pooled national composition would smear each of them evenly
    across the country, which is the failure --within exists to prevent.

    BUDAPEST IS 23 UNITS, NOT ONE. The capital is 17.9% of Hungary and GISCO stops at the
    city boundary; sources/hu_geo.py takes the 23 kerület from geoBoundaries ADM2 and clips
    them to GISCO's Budapest. This is the fix Croatia could not make for Zagreb.
    """
    from hu2022 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "hu_settlement_allocated.csv",
                     dtype={"geo_id": str}, low_memory=False)

    lut = pd.read_csv(HERE / "data" / "geo" / "hu" / "hu_lookup.csv",
                      dtype={"geo_id": str, "kod": str})
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["kod"])))
    missing = df["unit"].isna().sum()
    if missing:
        raise SystemExit(f"{missing} hu rows have no settlement code -- re-run "
                         "sources/hu_geo.py, the lookup is stale")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    # spec §3.10: an allocated count may never ring, because a ring asserts presence and
    # allocation only spreads a total. The Anglicans of Hungary are 372 people in a
    # country of 3,177 settlements, and a ring in every one of them would be a claim the
    # source does not make.
    df["may_ring"] = df["tier"] == "measured"
    return (df.groupby(["unit", "node"], as_index=False)
              .agg(count=("count", "sum"), congregations=("congregations", "max"),
                   may_ring=("may_ring", "max"), tier=("tier", "min")))


def _mk_counts():
    """SSO Popis 2021 at municipality: 13 categories on 80 units.

    ONE level. mk.csv carries `country` as well, which is the same 1.84M people again.

    NO ALLOCATION, and none is possible: SSO publishes these categories at this geography
    and nothing finer or coarser, so every row is `measured` and may ring. Czechia's shape,
    for a much shallower table.

    THE DRAWN POPULATION IS 92.5% OF THE COUNTRY. Four categories resolve to nothing —
    the universe total, the 1,964 who declined, the 894 unknown, and the 132,260 people
    whose data came from administrative registers and who were never asked. That last one
    is 7.2% and is a coverage residual rather than a refusal; taxonomy/mk2021.py says why
    it is not irreligion.
    """
    from mk2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "mk.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"] == "municipality"].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "mk" / "mk_lookup.csv",
                      dtype={"geo_id": str, "kod": str})
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["kod"])))
    missing = df["unit"].isna().sum()
    if missing:
        raise SystemExit(f"{missing} mk.csv rows have no LAU code -- re-run "
                         "sources/mk_geo.py, the lookup is stale")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


COUNTRIES = {
    "us": dict(
        name="United States",
        name_in="the United States",
        source="U.S. Religion Census 2020 (ASARB) and Pew Religious Landscape Study 2023-24",
        basis="self-identification, with membership rolls inside it",
        # §3.5a's "the declaration stays quiet": one sentence here, the numbers in the build
        # log and in counts.json for anyone who looks, and nothing on the map itself. It
        # assumed §7's desaturation would carry "this is modelled" on screen; that was
        # removed 2026-09-04, so THIS NOTE IS NOW THE ONLY PLACE A READER LEARNS IT, and the
        # sentence about the paler dots below has to keep doing that work alone.
        note_public=(
            "The United States asks no religion question, so the totals here are Pew's "
            "survey and the detail inside them is the U.S. Religion Census — 372 bodies "
            "reporting who is on their books, which is 48.4% of the country. A little over "
            "half of the dots here are the difference between the two: people the survey "
            "finds and no membership roll holds, placed among each county's residents who "
            "are on nobody's roll. Those are an estimate and are drawn the same as the "
            "counted ones — nothing on the map marks which is which. It is also the only "
            "reason an American non-religious population can be drawn at all: a roll's "
            "residual means “on no roll”, which is not “no religion”. "
            "Two things it rests on: the survey counts adults and this applies their answers "
            "to children too, and 1.4% of people answered nothing and are not drawn at all. "
            "Adherents are attributed to "
            "the congregation's county rather than the member's home. Counties are the "
            "finest thing anyone counts — the study does not record congregation addresses "
            "— so where a dot sits INSIDE a county is an estimate from the neighbourhood's "
            "ancestry and birthplace, not a measurement, and only for bodies where that "
            "could be checked against the county figures. The rest are spread across the "
            "county's population. Judaism is the known bad case: nothing in the census marks "
            "it, so Jewish neighbourhoods are not drawn as Jewish and the bodies that can be "
            "placed take the space instead."),
        view=[-125.0, 24.0, -66.5, 49.8],
        counts=_us_counts_rebased,
        units=None,              # counts are on counties; tracts carry both ids
        unit_key=None,
        place=HERE / "data" / "geo" / "tracts2020" / "cb_2020_us_tract_500k.shp",
        place_unit=lambda g: g["STATEFP"] + g["COUNTYFP"],
        # spec §8.4. Real ACS tract populations for every node instead of §8.2's equal-share
        # approximation, plus a demographic redistribution inside the county for the 26 nodes
        # whose held-out-metro correlation earned one. Falls back to §8.2 if unbuilt.
        place_weight=_us_place_weight,
        note="re-based on self-identification (spec §3.5a): Pew supplies the root totals, "
             "ASARB's rolls are the structure inside them, and the residual is drawn "
             "`derived`. Adherents are attributed to the congregation's county, not the "
             "member's (§3.6). Within a county, placement is a demographic estimate for "
             "some bodies and population-weighted for the rest (§8.4).",
    ),
    "ca": dict(
        name="Canada",
        source="Census of Population 2021 (Statistics Canada)",
        basis="self-identification, 25% long-form sample",
        view=[-128.0, 42.0, -55.0, 58.0],
        note_public=(
            "The census asks the person, so this is what people say they are rather than who "
            "is on a roll — and self-description is always the larger number. Categories "
            "below the province level are derived: StatCan publishes 168 religions by "
            "province and 25 by subdivision, never both, so the fine ones are split out "
            "proportionally and can show composition but not presence. 241 subdivisions "
            "publish religion built on ≥50% long-form non-response."),
        counts=_ca_counts,
        # StatCan's DA boundary file carries only DAUID / PRUID — no CSD link, and a DAUID
        # (province + census division + DA) does not contain one. Rather than fetch the
        # Geographic Attribute File for the lookup, derive it spatially: dissemination areas
        # nest exactly inside census subdivisions by construction, so a representative-point
        # join is not an approximation. `sjoin` also generalises to any country whose fine
        # layer omits the id of the unit the counts are on.
        units=HERE / "data" / "geo" / "ca" / "csd" / "lcsd000b21a_e.shp",
        unit_key="DGUID",
        place=HERE / "data" / "geo" / "ca" / "da" / "lda_000b21a_e.shp",
        place_unit="sjoin",
        note="StatCan is self_id from a 25% long-form sample; not comparable with the US "
             "roll across the border (spec §3.1).",
    ),
    "cz": dict(
        name="Czechia",
        source="Sčítání 2021 (Czech Statistical Office)",
        basis="self-identification, voluntary question",
        note_public=(
            "The religion question was voluntary and 30% of the country did not answer. "
            "Those people are not drawn at all, so this map shows 7.4 million of 10.5 "
            "million — and the share who answered runs from 11% to 81% between "
            "municipalities, so it is not an even haircut. What is drawn is unusually "
            "good: 78 categories published at municipality level with no rounding and no "
            "suppression, down to bodies with a single adherent. Jedi is the thirteenth "
            "largest answer, ahead of Jehovah's Witnesses, and is drawn as what it is."),
        counts=_cz_counts,
        # Czechia is Ireland's case: the counts are already ON the finest unit, so there is
        # no separate placement layer and no allocation inside a unit. Czech obce have a
        # median population of 435 — finer than a US census tract (3,424) and about the
        # size of an Australian SA1 — so an equal share per polygon is a good population
        # weighting almost everywhere (spec §8.2), and in the 8 statutory cities the city
        # districts carry it the rest of the way.
        #
        # cz_finest.gpkg is built by sources/cz_geo.py: 6,250 obce + 142 city districts,
        # which is the finest complete cover of the country ČSÚ publishes religion for.
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "cz" / "cz_finest.gpkg",
        place_unit=lambda g: g["kod"].astype(str),
        note="ČSÚ is self_id on a voluntary question; the 30% who did not answer are "
             "excluded rather than drawn (spec §3.5).",
    ),
    "au": dict(
        name="Australia",
        source="Census of Population and Housing 2021 (ABS)",
        basis="self-identification, voluntary question",
        view=[112.0, -44.0, 154.5, -9.5],
        note_public=(
            "The religion question is the only voluntary one on the Australian census, and "
            "about 7% left it blank; those people are not drawn. What is drawn is the "
            "deepest list on this map outside the United States — 148 groups, including "
            "three separate Orthodox communions that most sources collapse into one, and "
            "the Mandaeans, of whom Australia now holds more than Iraq does. Groups below "
            "the state level are derived: the ABS publishes 150 religions nationally and "
            "34 by SA2, so the fine ones are split out proportionally and can show "
            "composition but not presence."),
        counts=_au_counts,
        # Counts are on SA2; SA1s carry their parent's code, so no spatial join is needed.
        # SA1s are built to about 406 people, which is the cleanest §8.2 case in the
        # project — finer than a US tract and 25x finer than the SA2 the counts are on.
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "au" / "SA1_2021_AUST_GDA2020" /
              "SA1_2021_AUST_GDA2020.shp",
        place_unit=lambda g: g["SA2_CODE21"].astype(str),
        note="ABS is self_id on the census's only voluntary question; categories below "
             "state level are allocated (spec §3.9).",
    ),
    "ie": dict(
        name="Ireland",
        source="Census 2022 (CSO)",
        basis="self-identification",
        note_public=(
            "The finest geography on this map: 18,919 Small Areas, about 90 households "
            "each, so the dots sit where the people actually are rather than being spread "
            "across a county. The categories are the other way round — CSO publishes five "
            "at Small Area and 24 by county, so everything below Catholic, no religion and "
            "not stated is derived. One row reads 'Orthodox (Greek, Coptic, Russian)', "
            "which welds two churches that separated in 451 into a single number."),
        counts=_ie_counts,
        # The counts are already ON the finest unit, as in Czechia — no placement layer.
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ie" / "smallareas2022" / "SMALL_AREA_2022.shp",
        place_unit=lambda g: g["SA_GUID__1"].astype(str),
        note="CSO is self_id; categories below county level are allocated (spec §3.9).",
    ),
    "mx": dict(
        name="Mexico",
        source="Censo de Poblacion y Vivienda 2020 (INEGI)",
        basis="self-identification",
        note_public=(
            "INEGI separates people with no religion from believers with no affiliation, "
            "which most censuses do not: 9.5 million against 3.1 million, and folding the "
            "second into the first would overstate Mexican irreligion by a third. The "
            "denominations are thin by comparison — 23 categories, and everything except "
            "Catholic is derived from state-level shares. 'Other religions' is a single "
            "248,000-person bucket holding Buddhists, Hindus and Orthodox Christians "
            "together."),
        counts=_mx_counts,
        # Counts are on municipio; AGEBs carry their municipio's code in the first five
        # characters of CVEGEO, so no spatial join. 81,451 AGEBs against 2,469 municipios,
        # and INEGI builds them to a population target — urban ones to about 2,500 people
        # — which is what §8.2 asks for. Both urban and rural AGEBs are present and every
        # municipio has at least one, so nothing falls through.
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "mx" / "mg2020" / "conjunto_de_datos" / "00a.shp",
        place_unit=lambda g: g["CVE_ENT"].astype(str) + g["CVE_MUN"].astype(str),
        note="INEGI is self_id; every category except Catolica is allocated from entidad "
             "level (spec §3.9).",
    ),
    "nz": dict(
        name="New Zealand",
        source="Census 2023 (Stats NZ), 2018 structure",
        basis="self-identification, up to 4 responses per person",
        note_public=(
            "Two things here are unlike the rest of the map. The census lets a person give "
            "up to four religions, so a dot is a response rather than a person — about "
            "9,000 people are drawn twice. And the denominations are five years older than "
            "the totals: Stats NZ published 166 categories in 2018 and only 13 by area in "
            "2023, so the fine ones are 2018 shares applied to 2023 counts. Ratana and "
            "Ringatu, the churches founded by Maori prophets, are counted separately here "
            "and almost nowhere else. So is Jedi, at 22,605 — more than Baha'i, Jain, "
            "Taoist and Zoroastrian combined."),
        counts=_nz_counts,
        # SA1 2023, clipped to the coastline. Median 150 people, IQR 120-183 — the tightest
        # placement layer on the map, fourteen times finer than the SA2 the counts are on
        # and much tighter than a US census tract (spec §8.2).
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "nz" / "sa1_2023_clipped.geojson",
        place_unit=_nz_place_unit,
        note="Stats NZ is self_id with multiple response; categories below the national "
             "level are allocated from the 2018 table (spec §3.9 and §3.4 at once).",
    ),
    "uk": dict(
        name="United Kingdom",
        name_in="the United Kingdom",
        source="Censuses of 2021 and 2022 (ONS, NRS, NISRA)",
        basis="self-identification, voluntary question in England and Wales",
        view=[-8.7, 49.8, 2.0, 61.0],
        note_public=(
            "Three censuses, three agencies, three category lists — and England and Wales "
            "publish no Christian denomination at all, at any geography, for 27.5 million "
            "people. The write-in detail there is everything OUTSIDE Christianity: Pagan, "
            "Alevi, Jain, Ravidassia, Yazidi, Vodun. Scotland names the Church of Scotland "
            "and the Roman Catholics and stops. Northern Ireland, where the denomination "
            "is the political fact, names twenty-two Christian bodies including four kinds "
            "of Presbyterian — and is the only agency here that counts people as Mixed "
            "Catholic / Protestant."),
        counts=_uk_counts,
        # The counts are already on the finest units published — Output Areas in England,
        # Wales and Scotland, Data Zones in Northern Ireland — so there is no placement
        # layer, as in Czechia and Ireland. These are the finest units on the map: an E&W
        # Output Area is about 130 households.
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "uk" / "uk_units.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        note="Three censuses kept apart by source_id and never summed into a UK total "
             "(sources/uk.md); England and Wales and Northern Ireland are allocated "
             "(spec §3.9), Scotland is not.",
    ),
    "br": dict(
        name="Brazil",
        source="Censo Demográfico 2010 (IBGE)",
        basis="self-identification, census sample",
        # The dot bbox runs to 28.8°W — Martim Vaz, in the Atlantic, which belongs to the
        # município of Vitória, as Fernando de Noronha belongs to Pernambuco. Real data, but
        # fitting it puts 6° of empty ocean beside the country. Mainland only.
        view=[-74.2, -34.0, -34.2, 5.5],
        note_public=(
            "This is 2010, not the latest census. The 2022 census counted religion again "
            "and Brazil changed — evangelicals went from 22% to 27% and Umbanda and "
            "Candomblé tripled — but IBGE published only nine categories for 2022 and has "
            "withheld the evangelical breakdown over data quality, saying it may never "
            "appear. So the denominations here are the last ones measured: Assembleia de "
            "Deus and Congregação Cristã apart from each other, Umbanda apart from "
            "Candomblé, at the level of the município. Nine million people answered only "
            "“evangelical” and are drawn as that."),
        counts=_br_counts,
        # 2010 boundaries for 2010 data (spec §8.1). Brazil created five municipios between
        # the censuses, all by splitting existing ones, so every 2010 code still exists in a
        # current mesh — a join to one would succeed and quietly shrink five parents by the
        # territory that became a child. sources/br_geo.py fetches the 2010 mesh per state.
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "br" / "br_municipios_2010.gpkg",
        place_unit=lambda g: g["kod"].astype(str),
        note="IBGE 2010 is self_id from the census sample; municipal figures do not sum to "
             "IBGE's national figures, by construction (sources/br.md §4).",
    ),
    "pl": dict(
        name="Poland",
        source="Narodowy Spis Powszechny 2021 (Statistics Poland)",
        basis="self-identification, voluntary question",
        view=[14.0, 48.9, 24.2, 55.0],
        note_public=(
            "The religion question was voluntary and 20.5% of the country refused it. "
            "Those people are not drawn, so this map shows 30.2 million of 38.0 million. "
            "What is drawn is unusually detailed: 139 churches named at the level of the "
            "gmina, with no rounding and no suppression, and the tail is individual "
            "congregations rather than denominations — the Betel congregation in Warsaw "
            "is two people and is on the map as itself. Poland is 98% Latin Catholic "
            "among those who named a church, so the interest is entirely in the other "
            "2%: Orthodoxy along the Belarusian border, Lutherans in Cieszyn Silesia, "
            "the Mariavites — a Polish movement of 1906 and the only Old Catholic church "
            "anywhere with a Polish origin — and Old Believers in Masuria."),
        counts=_pl_counts,
        # Like Czechia and Ireland: the counts are already ON the finest unit GUS
        # publishes, so there is no separate placement layer and no allocation inside a
        # unit. Median gmina population is about 7,500, twice a US census tract, so an
        # equal share per polygon is a reasonable weighting nearly everywhere (spec §8.2).
        #
        # WHERE IT IS NOT: Warszawa is one gmina holding 1.79M people, 4.7% of the country
        # in a single 517 km² polygon, and Kraków, Łódź, Wrocław and Poznań are each one
        # too. Czechia had a fix for exactly this — ČSÚ publishes 142 city districts — and
        # GUS does not, so it stands. sources/pl_geo.md records it.
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "pl" / "pl_gminy.gpkg",
        place_unit=lambda g: g["kod"].astype(str),
        note="GUS is self_id on a voluntary question; the 20.5% who refused are excluded "
             "rather than drawn (spec §3.5).",
    ),
    "ro": dict(
        name="Romania",
        source="Recensământul Populaţiei şi Locuinţelor 2021 (INS)",
        basis="self-identification, partly from administrative registers",
        view=[20.2, 43.5, 30.0, 48.4],
        note_public=(
            "Religion could not be established for 14% of Romania. The 2021 census was "
            "built largely from administrative registers, which do not record religion, "
            "so this is an absent variable rather than a refusal — and those 2.7 million "
            "people are not drawn, leaving 16.4 million of 19.1 million. The 23 "
            "categories are Romania's list of state-recognised cults, so the detail is "
            "set by statute rather than by the question: no denomination outside the "
            "list is named at all. What the list does carry is unusual — the Lipovan Old "
            "Believers of the Danube delta, the largest such population any census "
            "publishes; the Hungarian Unitarians of Transylvania, a church continuous "
            "since 1568; and the Saxon and Hungarian Lutheran churches counted apart."),
        counts=_ro_counts,
        # UATs are the count layer and the placement layer: INS publishes religion at no
        # finer unit. Median UAT is about 3,000 people, the finest count geography on the
        # map after Ireland's Small Areas and the UK's Output Areas.
        #
        # Bucharest is the exception and it is a bad one — one UAT holding 9.8% of the
        # country in 240 km², worse than Warsaw's 4.7% and close to Prague's 12.4%. The
        # six sectors exist as administrative units but INS publishes no religion for
        # them, so subdividing would invent structure the source does not have.
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ro" / "ro_uat.gpkg",
        place_unit=lambda g: g["kod"].astype(str),
        note="INS suppresses small cells with '*'; those rows are dropped rather than "
             "estimated, costing 0.087% of the country (sources/ro.md §3).",
    ),
    "ee": dict(
        name="Estonia",
        source="Rahvaloendus 2021 (Statistics Estonia)",
        basis="self-identification, voluntary question, persons aged 15+",
        view=[21.5, 57.4, 28.3, 59.8],
        note_public=(
            "The least religious country on this map: 58% of Estonians aged 15 and over "
            "say they feel no affiliation to any religion, and a further 11% declined the "
            "question. Only children are missing for a different reason — the question is "
            "asked from age 15, so no Estonian child is drawn at all. Among those who do "
            "report a religion, Orthodoxy is larger than Lutheranism, which is the "
            "opposite of the country's history and follows the Russian-speaking "
            "population of Ida-Viru and Tallinn. Two things here are enumerated nowhere "
            "else on earth: Maausk and Taarausk, the Estonian native faith, counted as "
            "themselves; and the Old Believers of Lake Peipus."),
        counts=_ee_counts,
        # Tallinn is replaced by its 8 linnaosad, which Statistics Estonia publishes
        # religion for. Without that one polygon would be a third of the country.
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ee" / "ee_finest.gpkg",
        place_unit=lambda g: g["kod"].astype(str),
        note="Counts are rounded to base 10 (spec §3.8) and the universe is persons aged "
             "15+, not the whole population (sources/ee.md §2).",
    ),
    "hr": dict(
        name="Croatia",
        source="Popis stanovništva 2021 (Croatian Bureau of Statistics)",
        basis="self-identification",
        view=[13.3, 42.3, 19.5, 46.6],
        note_public=(
            "Croatia is 79% Catholic, so what this map shows is the other fifth: the "
            "Serbian Orthodox belt along the Bosnian and Serbian borders, the Muslim "
            "populations of the cities, and Istria — which is by a distance the least "
            "religious part of the country. The categories are shallow here by choice "
            "rather than by necessity: the census also names 54 individual churches at "
            "this same geography, including four Orthodox jurisdictions counted "
            "separately and eleven Jewish communities, and that table is not yet drawn."),
        counts=_hr_counts,
        # Zagreb is one polygon holding 18.4% of the country. The census would allow 17,
        # but the district boundaries were not found — see _hr_counts().
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "hr" / "hr_opcine.gpkg",
        place_unit=lambda g: g["kod"].astype(str),
        note="DZS neither rounds nor suppresses; the categories partition every unit "
             "exactly. Zagreb's 17 districts are summed into one (sources/hr_geo.md).",
    ),
    "in": dict(
        name="India",
        source="Census of India 2011, table C-01 and its Appendix (ORGI)",
        basis="self-identification, reported by the head of household",
        view=[67.5, 6.5, 97.8, 36.0],
        note_public=(
            "One in six people on earth, and the oldest source on this map by a decade: "
            "the 2021 census has never been held, so 2011 is not the best Indian figure "
            "but the only one. The census offers six boxes — Hindu, Muslim, Christian, "
            "Sikh, Buddhist, Jain — and clubs every sect into them, so nothing finer is "
            "knowable here: India's Shia and Sunni, its Syro-Malabar Catholics and its "
            "Ismailis are all inside a single category and no table anywhere separates "
            "them. What the census does do, and almost uniquely, is write down what the "
            "7.9 million people who refused all six boxes actually said. Those answers are "
            "83 named religions, nearly all Adivasi: Sarna of the Chotanagpur sacred "
            "groves, five million strong and still campaigning for a box of its own; the "
            "Gondi religion of the central highlands; Donyi-Polo, the Sun-and-Moon faith "
            "the Tani peoples of Arunachal organised in the 1970s against the missions; "
            "Sanamahi, revived in Manipur against an 18th-century conversion; Niam Khasi "
            "and Niamtre in a Meghalaya that is three-quarters Christian. Those 83 are "
            "published only by state, so their placement within a state is derived. The "
            "six large religions are not — they are counted on all 5,988 sub-districts. "
            "India has no 'no religion' box at all, so the blank space on this map where "
            "irreligion would be is a property of the question and not of the country."),
        counts=_in_counts,
        # Sub-districts are the count layer AND the placement layer, as in Poland, Romania
        # and Brazil. India breaks §8.2's usual trick rather than satisfying it: the finer
        # geography that exists — 645,828 villages and 4,135 towns — is natural settlements
        # ranging from ten people to two million, not units engineered to a population
        # target, so an equal share per polygon would weight a hamlet like a city.
        #
        # The cost, stated plainly: the median sub-district holds about 204,000 people,
        # the coarsest count unit on the map. Its median AREA, 551 km², is finer than a
        # Brazilian município's 1,527 km², so at national and state zoom the grain is
        # comparable to a country already drawn; at city zoom India is blockier than
        # anywhere else.
        #
        # No `place_weight` here, and the reason is that there is nothing to weight: the
        # placement layer IS the count layer, one polygon per unit. India is the obvious
        # second customer for that hook after the US — SHRUG publishes 645,828 village
        # points with 2011 population — and sources/in_geo.md §4 has the two pieces of
        # wiring it needs.
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "in" / "in_subdistricts.gpkg",
        place_unit=lambda g: g["kod"].astype(str),
        note="ORGI is self_id but answered by the head of household, not per person, "
             "which is why `religion not stated` is only 0.24% (spec §3.1). The 0.66% in "
             "`Other religions and persuasions` is allocated within each state, not "
             "pooled nationally (allocate.py --within; spec §3.10).",
    ),
    "de": dict(
        name="Germany",
        source="Zensus 2022, Sonderauswertung Religionszugehörigkeit (Destatis)",
        basis="administrative register (church-tax records), not a question",
        view=[5.4, 47.0, 15.6, 55.3],
        note_public=(
            "Nobody was asked. The 2022 census carries no religion question at all, and "
            "these figures are read off the population register, which records church "
            "membership because it determines church tax. So the data can see the two "
            "churches that levy it and nothing else: 51.8% of Germany is one grey "
            "category holding everyone the register has no religious body for. Germany's "
            "roughly four million Muslims are in there, with its Orthodox Christians, its "
            "Jewish communities, its free churches, the Old Catholics, and everyone who "
            "belongs to nothing — indistinguishable from each other, because the register "
            "never knew. That is not a judgement about who counts; it is the shape of the "
            "instrument, and no German source anywhere is deeper. What survives is the "
            "confessional map itself, at 10,786 municipalities: Catholic Bavaria, the "
            "Rhineland and the Saarland against a Protestant north, a boundary largely "
            "settled in the sixteenth century and still legible village by village. The "
            "sharpest line is not that one. In the former East, 81% belong to no church, "
            "against 45% in the West, and the Eichsfeld — a Catholic enclave that stayed "
            "Catholic through forty years of the GDR — still reads at 80% against a "
            "Thuringia of 74% none."),
        counts=_de_counts,
        # Gemeinden are the COUNT layer; the 1km INSPIRE grid is the PLACEMENT layer, and
        # Germany is the one country on this map where §8.2's approximation is dropped
        # outright rather than bounded (sources/de_grid.py).
        #
        # It had to be. Gemeinden are historical units, not units engineered to a
        # population target, so §8.2's usual trick does not apply at all: they run from
        # Dierfeld's 9 people to Berlin's 3,596,999 in ONE polygon, and 78 of them hold
        # 31.6% of the country. The median is 1,797, finer than a Polish gmina, so two
        # thirds of Germany was already drawn well — and the other third was drawn as
        # city-sized blobs, with Neukölln and Zehlendorf identical. Unlike Czechia's
        # Prague and Estonia's Tallinn there is no district-level religion table to swap
        # in.
        #
        # What replaced it is not a finer proxy, it is the same measurement at 1km:
        # destatis publishes THE SAME THREE CATEGORIES per grid cell, so `place_weight`
        # weights each religion by its OWN count in each square kilometre. Berlin goes
        # from 1 polygon to 799 cells, Hamburg to 655, Munich to 305. The US needs a
        # fitted demographic model to do this (§8.4); Germany just reads it.
        #
        # 34 Gemeinden holding 8,199 people (0.0099%) are too small for any 1km centre to
        # land inside them and carry their own polygon as a single cell, so the layer
        # covers all 10,786 units and nothing is unplaceable.
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "de" / "de_grid_1km.gpkg",
        place_unit=lambda g: g["ars"].astype(str),
        place_weight=_de_place_weight,
        note="Basis is `roll`, not `self_id`: this is a register of church-tax "
             "liability, so it is comparable with the United States and NOT with any "
             "census that asks the person (spec §3.1). Counts carry Cell-Key "
             "perturbation, so categories do not sum exactly to the published "
             "population — by 174 people in 82.7 million (sources/de.md §3).",
    ),
    "hu": dict(
        name="Hungary",
        source="Népszámlálás 2022, tables WBS003 and WBS008 (KSH)",
        basis="self-identification",
        view=[16.0, 45.6, 23.0, 48.7],
        note_public=(
            "Two out of every five Hungarians did not answer the religion question in "
            "2022 — 3.85 million people, the largest non-response on this map by a wide "
            "margin, and up from 27% in 2011. Answering was voluntary and the share who "
            "declined has risen at every census since the question came back in 2001, so "
            "the blank is a fact about the question rather than about belief: nothing "
            "here says what those people are, and this map does not guess. What is left "
            "is 60% of the country, and within it the historic pattern is still sharp. "
            "Catholic Hungary is the west and the north — 55% of the answers west of the "
            "Danube. East of the Tisza it is 20%, and a third of the answers there are "
            "Calvinist instead: the Reformation took hold on the plain in the 16th "
            "century and the Counter-Reformation never fully undid it. The Greek "
            "Catholics, 165,000 of them, are almost all in the north-east, and their "
            "historic seat at Hajdúdorog is still four-fifths Greek Catholic. Budapest "
            "is drawn as its 23 districts rather than as one shape, and they are not "
            "alike: 27% report no religion in the Castle district and 40% in Csepel."),
        counts=_hu_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "hu" / "hu_settlements.gpkg",
        place_unit=lambda g: g["kod"].astype(str),
        note="Settlement counts are measured for 98.1% of the population; the Orthodox, "
             "other-Christian and non-Christian columns are split from vármegye-level "
             "structure WITHIN each vármegye (allocate.py --within, spec §3.10). "
             "`Catholic, rite not stated` is derived as Catholic minus its two named "
             "rites — KSH publishes the parent and the children but never the remainder "
             "(sources/hu.md §4).",
    ),
    "mk": dict(
        name="North Macedonia",
        source="Попис 2021 (State Statistical Office)",
        basis="self-identification",
        view=[20.4, 40.8, 23.1, 42.4],
        note_public=(
            "Two communities and a long thin tail. Orthodox Christians and people who "
            "answered simply 'Christian' are together 59% of the country and Muslims are "
            "32%, and both follow the ethnic map almost exactly — Orthodox where the "
            "population is Macedonian, Serb or Vlach, Muslim where it is Albanian, "
            "Turkish, Roma, Bosniak or Torbeš. **Read 'Orthodox' and 'Christian' "
            "together.** The census offered both and the choice between them turns out to "
            "be regional rather than doctrinal: in the eastern municipalities half the "
            "population wrote 'Christian' — 76% of Rosoman, 70% of Makedonska Kamenica — "
            "where in the west and in Skopje almost everyone wrote 'Orthodox'. Taken "
            "apart they draw a divide in eastern Macedonia that is about how people "
            "answered, not what they believe. That correlation is the thing to hold in "
            "mind while reading this one: at 80 municipalities it is close to being an "
            "ethnic map with religious labels, and the census asks for a religion rather "
            "than a church, so 847,000 Orthodox arrive with no jurisdiction attached and "
            "the Sunni and Bektashi of the west are not told apart. One category in nine "
            "is not a religion at all: 132,260 people, 7.2%, were taken from "
            "administrative registers rather than enumerated in person and carry no "
            "answer, so this map draws 92.5% of the country. Irreligion is 0.5%, among "
            "the lowest anywhere here."),
        counts=_mk_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "mk" / "mk_opstini.gpkg",
        place_unit=lambda g: g["kod"].astype(str),
        note="80 municipalities is the source's ceiling for religion, not a choice: the "
             "same census publishes ethnicity by settlement and religion only by "
             "municipality. Refining religion inside a municipality from that ethnicity "
             "table is what spec §14.4 forbids, so the coarse grain stands "
             "(sources/mk.md §2).",
    ),
}
