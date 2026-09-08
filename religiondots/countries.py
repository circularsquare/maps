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
                optionally `roll` (spec §7a-i-1): for a `derived` row, the node its SOURCE
                COLUMN names — the coarser category the source counted at this same unit,
                which the roll-up falls back to when a reader asks what was measured. It is
                NOT an ancestor lookup: Hungary's Baptists came out of a column called
                `Other Christian denomination`, which is nowhere above them in the tree.
                Missing means "walk the tree", which is what every country did before.
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
  how           spec §7c: WHAT KIND OF THING these numbers are, in the same words for every
                country, so the reader can rank them against each other without parsing
                sixty-eight agencies in five languages. `source` already carries the
                citation and cannot do this: "Sčítání 2021 (Czech Statistical Office)" and
                "Sreda «Arena» Atlas 2012" look alike and are a census and a survey.
                A phrase, not a sentence, and it never repeats what `grain` says.
  grain         how fine the counts are, in a labelled row under the title. The unit and its
                average population, and nothing else: the row's label says what the number
                is, so the string must not say it again.

  fill          spec §7d: what a `derived` country's filled-in rows were filled in FROM, as a
                phrase completing "41% filled in ___" — "from the 2010 census", "from the
                same census at province level". Only for a country that HAS derived rows; the
                viewer falls back to "from broader counts" without it, which is the wording
                Anita rejected for describing fifteen real published tables as vaguely as
                possible. `allocate.py`'s invocation in COMMANDS.txt names the coarse level.
  gap           who or what the source leaves out, a few words, shown as the `not drawn` row.
                Absent for most countries and that is the healthy state: a row every country
                carries is a row nobody reads. Not a summary of `note_public` — if it needs a
                second line it belongs there instead.

                THESE FOUR ARE PLAIN TEXT — no markup and NO EM DASHES, Anita 2026-09-07.
                They are escaped by the viewer, so a `<b>` or a backtick would show the reader
                its own characters, and the dash is most of what made the block read as
                machine-written. Punctuate with a comma, a semicolon or a bracket. Asserted at
                the foot of this file, so a slip fails the import rather than the review.
  note_public   the country's own caveat, shown in the about panel when it is selected. The
                per-country note that used to be a cross-border paragraph belongs here: with
                a single country on screen the interesting comparison is inside it.

                HOW TO WRITE ONE — Anita, 2026-09-07: *"trying to keep it not sounding very
                ai, so people dont get ick."* The about panel renders `**bold**`, `*italic*`
                and `` `code` `` (spec §7d), and `tools/check_md.py` fails on a marker it
                cannot convert. Beyond that:

                  * A `**bold sentence.**` that STARTS a sentence is read as a topic sentence
                    and becomes a PARAGRAPH BREAK, losing its bold. That is the intended way
                    to structure a long note. Bold inside a sentence survives and is for a
                    figure: `Catholicism is **38.0%** and falling`.
                  * So do not reach for bold to make a point loud. The paragraph break is the
                    structure; bolding it too is the listicle voice, which is the specific
                    thing that reads as machine-written.
                  * Prefer a comma, a semicolon or a bracket to an em dash. The notes written
                    before this rule are full of them and are Anita's to clean up; do not
                    convert an existing one as a side effect of editing near it, and do not
                    add new ones.
                  * Say the specific thing. "from the 2010 census" beats "from an earlier
                    source"; a phrasing general enough to fit every country describes none of
                    them and reads as evasion.
                  * [[feedback_label_voice]]'s rules still apply to any short text: no
                    flourish, no loaded verbs, ask whose point of view a verb encodes.
  view          optional [w, s, e, n] to fly to, where the data bbox is the wrong picture —
                the US spans Hawaii to Maine and fitting that shows mostly ocean. Defaults to
                the bbox of the country's own dots, computed in tiles.py.
"""
import json
from pathlib import Path
import re
import sys

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "taxonomy"))


def _column_of(note, key="parent_column"):
    """The source COLUMN a derived row was split out of, off the normalised file's note.

    spec §7a-i-1. `allocate.py` writes `parent_column=<the fine table's own column>` as the
    LAST field of the note and `br_rescale.py` writes `group22=<the 2022 category>`; both
    name a category the source measured AT THE DRAWN UNIT, which is what the roll-up needs
    and what an ancestor walk up the religion tree cannot give it.

    THE `home` STRIP IS A SCAR, not a feature. Between 2026-09-03 and 2026-09-07 allocate.py
    emitted a pandas Series repr here instead of the column name, so `in` and `hu` carry
    `parent_column=home    Other religions and persuasions\\nhome    ...\\nName: 6, dtype:
    object`. The bug is fixed; the two files still have to be regenerated, and until they
    are this reads them correctly rather than silently dropping two countries.
    """
    if not isinstance(note, str):
        return None
    i = note.find(key + "=")
    if i < 0:
        return None
    # Stop at a `;` as well as a newline: allocate.py puts `parent_column` LAST so nothing
    # follows it, but br_rescale.py writes `group22=...; structure_share=...` and taking the
    # rest of the line there silently matched no category at all.
    v = note[i + len(key) + 1:].split("\n")[0].split(";")[0].strip()
    v = re.sub(r"^home\s+", "", v)
    return v or None


def _add_roll(df, columns, key="parent_column"):
    """Attach the `roll` column: the node of each derived row's own source column."""
    col = df["note"].map(lambda n: _column_of(n, key))
    df["roll"] = col.map(lambda c: columns.get(c) if c is not None else None)
    # A measured row is already at a level somebody counted and must never be moved.
    df.loc[df["tier"] != "derived", "roll"] = None
    return df


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
    import ca2021
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
    _add_roll(df, ca2021.COLUMNS)
    return df.rename(columns={"geo_id": "unit"})[
        ["unit", "node", "count", "congregations", "may_ring", "tier", "roll"]]


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


class _BrSetorWeighter:
    """Split a município's dots across its census setores by setor POPULATION.

    Mechanically the same as `_PhBarangayWeighter` above; kept separate because the reason
    it is needed differs, and because a weighter is the place a country's placement
    argument lives. If a third country needs one, these two should become one class.

    §8.2 assumes a placement unit is small enough that spreading dots evenly inside it is
    harmless. **São Paulo is one polygon holding 11.5M people**, so for Brazil that
    assumption fails at exactly the places a reader looks first — the dots would be as
    dense in the Serra da Cantareira as on Avenida Paulista. Setores fix it: ~452,000 of
    them, nesting inside the município by code.

    AND EQUAL SHARES PER SETOR IS NOT ENOUGH, which is where this differs from the US.
    American tracts are built to ~4,000 people each, so an equal split is already a
    population weighting and `place_weight` is unnecessary. Brazilian setores are built to
    roughly 300 households in cities and fewer in the country, and rural ones cover
    enormous areas — an equal split would systematically pull Brazil's dots into the
    countryside. The weight is the setor's own 2022 resident population (`v0001`).

    It is a POPULATION weight, not a religion one. Nothing measures where a given church's
    members live inside a município, so a Catholic dot and an Assembleia de Deus dot are
    spread identically, and every município's total is exactly IBGE's either way. §14.4
    permits precisely this and no more: refine placement, never invent magnitude. Read a
    cluster as "this município, drawn where its people are", never as a neighbourhood.
    """

    def __init__(self, place):
        self.pop = place["pop"].to_numpy(dtype=float)
        self.n_pop = 0
        self.n_uniform = 0

    def weights(self, node, idx, count, plain=False):
        pop = self.pop[idx]
        if pop.sum() > 0:
            self.n_pop += 1
            return pop
        self.n_uniform += 1
        return None

    def summary(self):
        return (f"{self.n_pop:,} (unit, node) rows placed on setor population, "
                f"{self.n_uniform:,} on equal shares where a município's setores sum to "
                f"zero (sources/br_setores.py)")


def _br_place_weight(place):
    """countries.py hook. `place` is the setor layer scatter.py has read."""
    if "pop" not in place.columns:
        print("  !! br_setores_2022.gpkg has no `pop` column — run sources/br_setores.py; "
              "placing on equal shares (§8.2)")
        return None
    return _BrSetorWeighter(place)


def _br_counts():
    """IBGE Censo 2022 totals on Censo 2010 structure, at município — spec §3.4, BUILT
    2026-09-04.

    `br_rescale.py` writes the file this reads. Until then Brazil was drawn at 2010 as it
    stood, because 2022 publishes NINE categories and lumps 47.4M evangelicals into one
    (sources/br.md §1); §3.4 has described the fix since the project began and it is now
    done. Every município's rescaled rows sum to that município's own 2022 total exactly.

    TWO THINGS CHANGED FOR THE READER, and note_public carries both.

    The totals are 2022, so Brazil is no longer fifteen years stale: Catholics 64.6% ->
    56.7%, evangelicals 22.2% -> 26.9%, Umbanda e Candomblé 588,810 -> 1,849,835.

    The universe is now **people aged 10 or over**, because that is who the 2022 religion
    question was put to. The drawn population falls from 190.8M to 176.3M. It is NOT scaled
    back up to the whole population, for Chile's reason (§14.4): the source publishes an
    exact partition of its own universe, and inventing the rest would be a larger claim
    than anything else on this map makes.

    TIER IS NOT UNIFORM HERE, which is unusual. The three 2022 categories that map to a
    single 2010 leaf — Católica Apostólica Romana, Espírita, Tradições indígenas — pass
    through untouched and are `measured`, 58.7% of the drawn people. The rest is `derived`:
    a 2022 magnitude wearing a 2010 shape, and §3.10 forbids it from ringing.
    """
    import br2010
    from br2010 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "br_municipio_rescaled.csv",
                     dtype={"geo_id": str}, low_memory=False)

    # The rescaled file is already leaves-only — br_rescale.py derives them the same way,
    # from IBGE's own parent chain — so there is no nesting left to collapse here.
    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    df["may_ring"] = df["tier"] == "measured"
    # §7a-i-1. Brazil's "column" is the 2022 GROUP the município total was published for —
    # `group22=` in the note rather than `parent_column=`, but the same claim: IBGE counted
    # that many evangelicals in that município, and only their denomination is 2010's.
    _add_roll(df, br2010.COLUMNS, key="group22")
    return df.rename(columns={"geo_id": "unit"})[
        ["unit", "node", "count", "congregations", "may_ring", "tier", "roll"]]


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
    # §7a-i-1: where the module names its fine columns, a derived row remembers which one it
    # came out of. Without it the roll-up has only the tree to walk, and the tree does not
    # know that Ireland's Anglicans were counted in a cell called `Other religion`.
    _add_roll(df, getattr(module, "COLUMNS", {}))
    return df.rename(columns={"geo_id": "unit"})[
        ["unit", "node", "count", "congregations", "may_ring", "tier", "roll"]]


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
        frames.append(d[["geo_id", "source_category", "count", "may_ring", "tier", "note"]])

    sc = pd.read_csv(HERE / "data" / "normalized" / "uk.csv",
                     usecols=["geo_id", "geo_level", "source_category", "count",
                              "source_id"],
                     dtype={"geo_id": str}, low_memory=False)
    sc = sc[(sc["source_id"] == "uk_sc_census_2022")
            & (sc["geo_level"] == "output_area")].copy()
    sc["may_ring"] = True                      # nothing was allocated; all measured
    sc["tier"] = "measured"
    sc["note"] = None                          # nothing to roll up TO, and nothing to roll
    frames.append(sc[["geo_id", "source_category", "count", "may_ring", "tier", "note"]])

    df = pd.concat(frames, ignore_index=True)
    df["node"] = df["source_category"].map(uk2021.resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    _add_roll(df, uk2021.COLUMNS)
    return df.rename(columns={"geo_id": "unit"})[
        ["unit", "node", "count", "congregations", "may_ring", "tier", "roll"]]


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
    import in2011
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
    _add_roll(df, in2011.COLUMNS)
    return df[["unit", "node", "count", "congregations", "may_ring", "tier", "roll"]]


class _InSettlementWeighter:
    """India's dots land on villages and towns, weighted by their own 2011 population.

    spec §8.2a is the section that says India cannot do what every other country does:
    there is no statistical layer between the sub-district and the settlement, so an equal
    share per polygon would weight a hamlet like a city. sources/in_place.py builds the
    layer anyway and does the weighting there, where the census totals are in hand — a
    village's `t_pop2011` from SHRUG, a town's population from C-01 itself, and an
    area-proportional share of whatever a unit's total does not account for.

    So there is nothing per-node here and there cannot be: **India publishes no religion at
    any geography finer than the sub-district.** This is a population weight, exactly as
    Serbia's and Kenya's are. A Muslim dot and a Hindu dot in Malappuram are spread the same
    way, and a cluster means "this sub-district, drawn where its people actually live" and
    never a neighbourhood reading. Germany (§8.2b) is the only country on the map that can
    say more, because destatis publishes the religion on the grid.
    """

    def __init__(self, place):
        self.pop = place["pop"].to_numpy(dtype=float)
        self.n_pop = 0
        self.n_uniform = 0

    def weights(self, node, idx, count, plain=False):
        pop = self.pop[idx]
        if pop.sum() > 0:
            self.n_pop += 1
            return pop
        self.n_uniform += 1
        return None

    def summary(self):
        return (f"{self.n_pop:,} (unit, node) rows placed on village and town population, "
                f"{self.n_uniform:,} on equal shares where a unit's settlements sum to zero "
                f"(sources/in_place.py)")


def _in_place_weight(place):
    """countries.py hook. `place` is the settlement layer scatter.py has read."""
    if "pop" not in place.columns:
        print("  !! in_places.gpkg has no `pop` column — run sources/in_place.py; "
              "placing on equal shares (§8.2)")
        return None
    return _InSettlementWeighter(place)


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
        # EVERYTHING BELOW IS THE `Sonstige` COLUMN AND NOT THE RELIGION'S OWN, and the
        # paragraphs above do not describe any of it. destatis publishes a grid for its
        # three categories and for nothing else; these nodes are carved out of the third
        # one — Jews by ZWST's count (de_zwst.py), the rest by the ESS model (de_ess.py) —
        # so `son` is the column they were counted in rather than a column that counts
        # them.
        #
        # It is used because inside a city it is a far better locator than total
        # population: `son` is where the register has no church for people, which is
        # where the Muslim, Orthodox and migrant-descended populations actually are. In
        # Neukölln it is most of the neighbourhood and in a Bavarian village it is a
        # tenth of it. But it is a PROXY — the "placement is measured" claim above is
        # true of the three Zensus categories and of nothing else here.
        "judaism": "son",
        "islam": "son",
        "christianity.orthodox": "son",
        "christianity.evangelical": "son",
        "christianity": "son",
        "other.de": "son",
    }
    # christianity.protestant is NOT in PROXY even though the ESS split adds a modelled
    # slice to it: the node's own `ev` column is the right weight for the measured 19.1M
    # that dominates it, and `weights()` is asked per (unit, node) and cannot see tier.
    PROXY = {"judaism", "islam", "christianity.orthodox",
             "christianity.evangelical", "christianity", "other.de"}

    # --- Route B: citizenship sharpens two of those six ---------------------------------
    # `son` is half of Germany and contains every secular German, so on its own it spreads
    # Muslim dots evenly across a city. destatis publishes citizenship on the SAME 1km
    # cells (sources/de_grid.py), and Turkish and Bosnian passports are a far sharper
    # locator: 1.5 million people whose position inside Berlin is Neukölln and Kreuzberg
    # rather than Zehlendorf.
    #
    # IT CANNOT BE USED ALONE, and the reason is the whole design. Citizenship is not
    # origin — most of Germany's Turkish-descended population holds German passports and
    # is invisible to this column — so placing every Muslim dot on it would assert that
    # naturalised families live exactly where non-naturalised ones do, and would put
    # nobody at all in the many cells where the community is entirely German-citizen.
    #
    # SO THE BLEND COMES FROM THE UNIT'S OWN ARITHMETIC, which is `_ItWeighter`'s move
    # (Italy, §8.4a) with the ratio computed rather than chosen. In each Gemeinde the
    # signal accounts for C people out of the node's N: the citizenship vector gets C/N
    # of the weight and `son` keeps the rest. Where the signal is absent it is pure `son`
    # and nothing changes; where it accounts for the whole node it is pure citizenship.
    # No constant is picked anywhere, so there is nothing here to tune.
    CITIZENSHIP = {"islam": "isl_ctz", "christianity.orthodox": "orth_ctz"}

    def __init__(self, place):
        self.pop = place["pop"].to_numpy(dtype=float)
        self.col = {k: place[v].to_numpy(dtype=float) for k, v in
                    ((n, c) for n, c in self.COLUMN.items() if c in place.columns)}
        self.ctz = {k: place[v].to_numpy(dtype=float) for k, v in
                    ((n, c) for n, c in self.CITIZENSHIP.items() if c in place.columns)}
        self.n_measured = 0
        self.n_proxy = 0
        self.n_pop = 0
        self.n_uniform = 0
        # per node: [rows sharpened, sum of the blend fraction, rows where it capped at 1]
        self.ctz_stat = {n: [0, 0.0, 0] for n in self.CITIZENSHIP}

    def weights(self, node, idx, count, plain=False):
        col = self.col.get(node)
        if col is not None:
            w = col[idx]
            if w.sum() > 0:
                if node in self.PROXY:
                    w = self._sharpen(node, idx, count, w)
                    self.n_proxy += 1
                else:
                    self.n_measured += 1
                return w
        pop = self.pop[idx]
        if pop.sum() > 0:
            self.n_pop += 1
            return pop
        self.n_uniform += 1
        return None

    def _sharpen(self, node, idx, count, base):
        """Blend the `son` weight with the citizenship signal, in the unit's own ratio."""
        sig = self.ctz.get(node)
        if sig is None or not count:
            return base
        c = sig[idx]
        total = c.sum()
        if total <= 0:
            return base
        raw = float(total) / float(count)
        f = min(1.0, raw)                  # how much of the node the signal accounts for
        s = self.ctz_stat[node]
        s[0] += 1
        s[1] += f
        s[2] += raw >= 1.0
        return (c / total) * f + (base / base.sum()) * (1.0 - f)

    def summary(self):
        proxy = (f", {self.n_proxy:,} on the `Sonstige` column they were carved out of "
                 f"(a proxy and not that religion's own count)" if self.n_proxy else "")
        bits = []
        for node, (n, tot, cap) in self.ctz_stat.items():
            if not n:
                continue
            # A node capped almost everywhere is NOT a healthy blend: it means the
            # measured citizenship count exceeds the modelled magnitude across the
            # country, so the weight is citizenship alone. Said out loud rather than
            # buried, because it is also evidence about the MODEL — see sources/de.md §10.
            bits.append(f"{node} {n:,} rows at {100 * tot / n:.0f}% citizenship"
                        + (f", CAPPED in {100 * cap / n:.0f}% of them" if cap else ""))
        ctz = (" — sharpened by the citizenship grid: " + "; ".join(bits)) if bits else ""
        return (f"{self.n_measured:,} (unit, node) rows placed on that religion's OWN 1km "
                f"grid counts{proxy}{ctz}, {self.n_pop:,} on cell population where the "
                f"category is zero across the unit's cells, {self.n_uniform:,} on equal "
                f"shares (sources/de_grid.py)")


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

    df = _de_add_zwst(df)
    df = _de_split_ess(df)

    df["congregations"] = 0
    if "tier" not in df.columns:
        return df[["unit", "node", "count", "congregations"]]
    return df[["unit", "node", "count", "congregations", "tier"]]


def _de_add_zwst(df):
    """Carve the Jewish communities out of `unrecorded` — sources/de_zwst.py, de.md §7.

    THE ONLY PART OF GERMANY'S 42.8 MILLION GREY CELL THAT A SECOND SOURCE CAN COUNT
    RATHER THAN ESTIMATE. ZWST publishes membership per community and de_zwst.py seats
    each one in a Gemeinde, so this is a `roll` meeting a `roll` (spec §3.1) and Germany
    stays `measured` throughout — unlike the ESS split scouted in de.md §6, which would
    spend that.

    SUBTRACT BEFORE ADDING. These people are inside `Sonstige, keine, ohne Angabe` today,
    because Jewish communities are public-law religious societies in most Laender and the
    Melderegister lumps every body that is not the two big churches into that one cell.
    Adding a judaism row without taking it back out of `unrecorded` would give Germany
    87,934 people it does not have and break the unit totals.

    The subtraction cannot go negative: de_zwst.py asserts every seat has room, and the
    tightest is Straubing at 4.5% of its `unrecorded`. It is asserted again here anyway,
    because that check lives in a file this one does not run.
    """
    path = HERE / "data" / "normalized" / "de_zwst.csv"
    if not path.exists():
        print("  !! de_zwst.csv missing — run sources/de_zwst.py; Germany's Jewish "
              "communities stay inside `unrecorded` (sources/de.md §7)")
        return df

    z = pd.read_csv(path, dtype={"geo_id": str})
    members = z.groupby("geo_id")["count"].sum()

    seats = df["unit"].isin(members.index)
    grey = df["node"] == "unrecorded"
    take = df.loc[seats & grey, "unit"].map(members)

    if len(take) != len(members):
        missing = sorted(set(members.index) - set(df.loc[grey, "unit"]))
        raise SystemExit(f"de_zwst: {len(missing)} seats have no `unrecorded` row to "
                         f"subtract from, e.g. {missing[:5]} — did de.csv change vintage?")
    short = df.loc[take.index, "count"] - take
    if (short < 0).any():
        bad = df.loc[short[short < 0].index, "unit"].tolist()
        raise SystemExit(f"de_zwst: members exceed `unrecorded` in {bad} — the seat join "
                         "is wrong, or the two sources are different vintages")
    df.loc[take.index, "count"] = short

    add = pd.DataFrame({"unit": members.index, "node": "judaism",
                        "count": members.to_numpy()})
    print(f"  {members.sum():,} ZWST members moved from `unrecorded` to `judaism` "
          f"in {len(members)} Gemeinden (sources/de_zwst.py)")
    return pd.concat([df, add], ignore_index=True)


def _de_split_ess(df):
    """Split what is left of `unrecorded` by each Bundesland's ESS composition.

    spec §3.4's SPLIT and §14.10's fractional-share model, and everything about it is in
    sources/de_ess.py. In one line: the register counted 42.8 million people into a cell
    it cannot see inside, ESS asked 13,643 Germans which denomination they belong to, and
    the register's own Catholic and Protestant totals are what makes the two comparable
    — ESS reproduces them to within 0.4 and 0.8 points without being told either.

    THE SPLIT IS PARTIAL ON PURPOSE, and this is the design decision. Only the ~19% of the
    residual that ESS assigns to a NAMED RELIGION is drawn. The 41% who told ESS they
    belong to no religion stay in `unrecorded`, where the register put them, because
    (a) `unrecorded` is a measured cell and replacing it with a modelled `unaffiliated`
    trades a count for an estimate, and (b) §14.12's rule is that ancestry-shaped cells
    model well and attitude-shaped ones badly — Islam and Orthodoxy in Germany are close
    to functions of descent, and non-belief is not. So Germany stays about 90% measured
    and `inferred dots: not shown` returns it to nearly the map it is today.

    Everything here is `modelled` and not `derived`, on §7b's test: the tiers are about
    whether anybody was counted, and nobody counted Germany's Muslims at any level. Note
    what that means for §7's control — the viewer REMOVES modelled dots rather than
    rolling them up, so these do not fall back to `unrecorded`, they go.
    """
    path = HERE / "data" / "normalized" / "de_ess.csv"
    if not path.exists():
        print("  !! de_ess.csv missing — run sources/de_ess.py; Germany's Muslims, "
              "Orthodox and free churches stay inside `unrecorded` (sources/de.md §6)")
        return df

    from de2022 import resolve

    e = pd.read_csv(path, dtype={"geo_id": str})
    e["node"] = e["source_category"].map(resolve)
    if e["node"].isna().any():
        bad = sorted(e.loc[e["node"].isna(), "source_category"].unique())
        raise SystemExit(f"de_ess: unmapped ESS categories {bad} — add them to "
                         "taxonomy/de2022.py")
    # one share per (Land, node): two ESS answers can share a node (other.de)
    share = e.groupby(["geo_id", "node"])["count"].sum()

    # the residual stops being whole people the moment it is split, and leaving `count`
    # as int64 makes the subtraction below a silent pandas dtype error
    df["count"] = df["count"].astype(float)

    grey = df["node"] == "unrecorded"
    land = df.loc[grey, "unit"].str[:2]
    bucket = df.loc[grey, "count"].to_numpy(dtype=float)

    frames, taken = [], pd.Series(0.0, index=df.index[grey])
    for (gid, node), s in share.items():
        m = (land == gid).to_numpy()
        if not m.any():
            raise SystemExit(f"de_ess: no Gemeinde has Land prefix {gid!r}")
        n = bucket * m * float(s)
        taken += pd.Series(n, index=taken.index)
        keep = n > 0
        frames.append(pd.DataFrame({
            "unit": df.loc[grey, "unit"].to_numpy()[keep],
            "node": node, "count": n[keep], "tier": "modelled"}))

    if (taken > df.loc[grey, "count"]).any():
        raise SystemExit("de_ess: a Land's shares take more than its residual")
    df.loc[grey, "count"] = df.loc[grey, "count"] - taken

    add = pd.concat(frames, ignore_index=True)
    add = add.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()
    df["tier"] = "measured"
    out = pd.concat([df, add], ignore_index=True)

    n = add["count"].sum()
    print(f"  {n:,.0f} people ({100 * n / df['count'].sum():.1f}% of Germany) split out "
          f"of `unrecorded` into {add['node'].nunique()} religions by Bundesland, "
          f"MODELLED (sources/de_ess.py)")
    return out


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
    import hu2022
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
    _add_roll(df, hu2022.COLUMNS)
    # Several geo_ids can share one settlement code, so the rows are summed. `tier` and
    # `roll` KEY the group rather than being aggregated over it — spec §7's "a qualifier on
    # a row must not be aggregated over the rows it qualifies", which taking `min(tier)`
    # here did: a settlement with one large measured row and one small allocated one had
    # the whole of it relabelled `derived` by the small one.
    return (df.groupby(["unit", "node", "tier", "roll"], as_index=False, dropna=False)
              .agg(count=("count", "sum"), congregations=("congregations", "max"),
                   may_ring=("may_ring", "max")))


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


def _lk_counts():
    """DCS CPH 2024 at GN division: 6 categories on 14,003 units.

    ONE level, no allocation, nothing modelled — the census publishes these categories at
    this geography and the map draws exactly that. 14,003 units for 21.8M people is about
    1,555 people each, the finest grain in the project outside the US tracts and the German
    grid, and it arrives without any of the machinery those two needed.

    THE WHOLE POPULATION IS DRAWN, which is true of no other country here and is not the
    compliment it sounds like. DCS's six categories partition all 21,781,800 people with no
    'not stated', no 'no religion' and no refusal line at all, so everyone is assigned a
    religion whether or not they profess one. §3.5's "undercounting is marked, not filled"
    has nothing to mark; the thing to say instead is that Sri Lankan irreligion is not
    absent from this map, it is invisible on it, distributed among the six.

    PLACEMENT IS COARSER FOR 53 UNITS. The only boundary set that reaches GN divisions is
    OCHA's COD, valid 2022, and 53 of the 2024 census's divisions have no 2022 polygon.
    Their dots are placed in the unmatched remainder of their DS division instead — 95,641
    people, 0.44%. The counts are still measured; it is where inside the map they sit that
    is looser, so `tier` stays `measured` and sources/lk_geo.py carries the detail.
    """
    from lk2024 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "lk.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"] == "gnd"].copy()

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


def _cl_counts():
    """INE Censo 2024 at comuna: 13 categories on 346 comunas, people aged 15 or over.

    ONE level, no allocation, nothing modelled. INE publishes these categories at this
    geography and the map draws exactly that, so every row is `measured` and may ring.

    THE UNIVERSE IS 15+ AND IT IS NOT SCALED UP TO THE WHOLE POPULATION — the call is
    argued at length in sources/cl.py. In one line: Chile's own data shows the share
    professing a religion running 96.0% at 65+ down to 63.9% at 15-29, so handing under-15s
    the 75.1% adult average would overstate religion among children by six to eleven points,
    and §14.4 forbids inventing a magnitude when the source publishes an exact one. §3.5a
    scales Pew onto American children only because the alternative there is drawing half the
    country as nothing at all; Chile has no such hole. Drawn: 81.8% of the population.

    ANTÁRTICA IS DROPPED. Comuna 12202, 60 people aged 15+, has no polygon in COD's admin3
    and would in any case put a dot near the South Pole. It draws no dot at 1:1,000 either
    way. sources/cl_geo.py names it; this is where it leaves the data.
    """
    from cl2024 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "cl.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"] == "comuna"].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "cl" / "cl_lookup.csv",
                      dtype={"geo_id": str})
    drawn = set(lut[lut["drawn"] == 1]["geo_id"])
    missing = sorted(set(df["geo_id"]) - drawn)
    if missing != ["12202"]:
        raise SystemExit(f"expected only Antártica to lack a polygon, got {missing} -- "
                         "re-run sources/cl_geo.py, the lookup is stale")
    df = df[df["geo_id"].isin(drawn)]

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


class _KeHexWeighter:
    """Split a county's dots across Kontur 400 m hexagons by hex POPULATION.

    Kenya is the case that makes §8.2 load-bearing rather than tidy. The counts are 47
    counties for 47.2M people — the coarsest counting geography on this map — and the
    counties are wildly uneven in habitability: Turkana is 68,680 km² with 926,976 people,
    Marsabit 70,961 km² with 459,785. An equal share per polygon would wash the northern
    half of the country in evenly spaced dots over empty desert, and since Wajir, Mandera
    and Garissa are each 97-99% Muslim that wash would be one colour and the loudest thing
    on the map.

    So the weight is the hexagon's own modelled population (sources/ke_grid.py). It is a
    POPULATION weight, not a religion one: nothing measures where Kenya's Catholics sit
    inside a county, so a Catholic dot and a Muslim dot are spread identically. Read the
    map as "religion by county, drawn where Kenyans live", never as county-internal detail.
    """

    def __init__(self, place):
        self.pop = place["pop"].to_numpy(dtype=float)
        self.n_pop = 0
        self.n_uniform = 0

    def weights(self, node, idx, count, plain=False):
        pop = self.pop[idx]
        if pop.sum() > 0:
            self.n_pop += 1
            return pop
        self.n_uniform += 1
        return None

    def summary(self):
        return (f"{self.n_pop:,} (unit, node) rows placed on Kontur hex population, "
                f"{self.n_uniform:,} on equal shares where a county's hexes sum to zero "
                f"(sources/ke_grid.py)")


def _ke_place_weight(place):
    """countries.py hook. `place` is the hex layer scatter.py has read."""
    if "pop" not in place.columns:
        print("  !! ke_hexes.gpkg has no `pop` column — run sources/ke_grid.py; "
              "placing on equal shares (§8.2)")
        return None
    return _KeHexWeighter(place)


class _MwHexWeighter(_KeHexWeighter):
    """Split a district's dots across Kontur 400 m hexagons by hex POPULATION.

    Kenya's class unchanged; Malawi's reason for needing it is different and worth keeping
    beside it. Kenya's problem is empty desert inside huge counties. **Malawi's is water.**
    Lake Malawi is 29,600 km² and it is not a hole in the country — the district boundaries
    run out into the middle of it, so Karonga, Rumphi, Nkhata Bay, Likoma, Salima,
    Nkhotakota and Mangochi each own a slab of open lake. An equal share per polygon would
    put a fifth of Malawi's dots on water, in a band down the whole eastern side, and
    Likoma — an island of 14,527 people whose polygon is almost all lake — would be a wash
    over nothing. A population grid has no hexes on the lake, so the problem does not arise
    rather than being patched (§8.2c, and Ethiopia's finding at §9u).

    Same caveat as Kenya's: it is a POPULATION weight and not a religion one. Nothing
    measures where Malawi's Anglicans sit inside a district, so an Anglican dot and a Muslim
    dot are spread identically. Read it as "religion by district, drawn where Malawians
    live".
    """

    def summary(self):
        return (f"{self.n_pop:,} (unit, node) rows placed on Kontur hex population, "
                f"{self.n_uniform:,} on equal shares where a district's hexes sum to zero "
                f"(sources/mw_grid.py)")


def _mw_place_weight(place):
    """countries.py hook. `place` is the hex layer scatter.py has read."""
    if "pop" not in place.columns:
        print("  !! mw_hexes.gpkg has no `pop` column — run sources/mw_grid.py; "
              "placing on equal shares (§8.2)")
        return None
    return _MwHexWeighter(place)


class _MuHexWeighter(_KeHexWeighter):
    """Split a unit's dots across Kontur 400 m hexagons by hex POPULATION.

    Mauritius needs this least of the four countries that use it — 182 units averaging
    11 km² and 6,800 people, so an equal share per polygon is already close. Two things
    still go wrong without it: the coastal VCAs own their LAGOON, because the boundaries run
    out to the reef, and the few big rural units (Grande Rivière Noire 43.5 km², Tamarin
    48.0 km²) are mostly gorge and cane with their people along one road.

    Kontur is coarse relative to this country — 2,072 hexes for 182 units — and three
    cross-district slivers of 0.09 to 3.6 km² contain no hex centroid at all. They hold 890
    people between them, 0.07%, and fall back to an equal share inside their own polygon,
    which at that size is not an approximation of anything.
    """

    def summary(self):
        return (f"{self.n_pop:,} (unit, node) rows placed on Kontur hex population, "
                f"{self.n_uniform:,} on equal shares in the three sliver units that "
                f"contain no hex (sources/mu_grid.py)")


def _mu_place_weight(place):
    """countries.py hook. `place` is the hex layer scatter.py has read."""
    if "pop" not in place.columns:
        print("  !! mu_hexes.gpkg has no `pop` column — run sources/mu_grid.py; "
              "placing on equal shares (§8.2)")
        return None
    return _MuHexWeighter(place)


def _mu_counts():
    """Statistics Mauritius 2022 HPC Table D6: 13 drawn categories on 182 units.

    ONE level, no allocation, nothing modelled — every row is `measured` and may ring.

    182 UNITS FOR 1,233,097 PEOPLE IS ~6,800 EACH, the finest counting geography on this map
    after Sri Lanka's GN divisions, the German grid and the UK's output areas, and finer than
    all of them relative to the size of the country. The units are Municipal Council Wards
    and Village Council Areas, which is the tier Statistics Mauritius calls `R`.

    183 CENSUS ROWS BECOME 182 DRAWN UNITS. OpenStreetMap has no polygon for Vacoas-Phoenix
    Ward 5 or Ward 6-West, so those two rows share one unit built from the remainder of the
    town — 35,664 people, 2.89%, and the cost is one internal boundary inside one town.
    `mu_lookup.csv` maps both geo_ids to it and the rows are summed here.

    ONE CATEGORY RESOLVES TO NOTHING and it is the universe row, so the drawn population is
    the whole table: **1,233,097, the entire resident population the census enumerated.**
    `Other & Not stated` is drawn rather than dropped — Mauritius is the only source here
    that pools a non-answer into a residual with no split at any geography, so §3.5's usual
    move is unavailable and the cell is marked instead. See taxonomy/mu2022.py.
    """
    from mu2022 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "mu.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "unit"].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "mu" / "mu_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"mu.csv units with no polygon: {missing} -- re-run "
                         "sources/mu_geo.py, the lookup is stale")
    if df["unit"].nunique() != 182:
        raise SystemExit(f"{df['unit'].nunique()} units, expected 182")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    # Two census rows share the Vacoas remainder unit, so (unit, node) is not unique until
    # they are summed. Every other country's rows are already one per pair.
    df = df.groupby(["unit", "node"], as_index=False)["count"].sum()
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


def _mw_counts():
    """NSO 2018 PHC Table E5 at district: 10 drawn categories on 32 districts.

    ONE level, no allocation, nothing modelled — NSO publishes these categories at this
    geography and the map draws exactly that, so every row is `measured` and may ring.

    32 UNITS FOR 17.56M PEOPLE IS ~549,000 EACH, which is finer per head than Kenya's 47
    counties (1.0M) and than Ghana's regions, and it is NSO's ceiling: Table E5 is the only
    religion table below the national one anywhere in the 311-page report, and the district
    reports carry no religion at all. sources/mw.md §2.

    THE FOUR CITIES ARE SEPARATE UNITS AND NOT PARTS OF THEIR DISTRICTS. Mzuzu, Lilongwe,
    Zomba and Blantyre Cities are printed as peers of Mzimba, Lilongwe, Zomba and Blantyre —
    2,115,867 people, 12.0% of the country — and `sources/mw.py` proves it rather than
    assuming it, by checking that each region's districts sum to the region row on all
    eleven columns. Reading them as nested would double-count them; reading them as absent
    would lose an eighth of Malawi into the wrong polygons.

    ONE CATEGORY RESOLVES TO NOTHING and it is the universe row, so the drawn population is
    the whole table: **17,563,749, which is the entire census count.** NSO publishes no
    `not stated` cell for religion and no residual — the ten denominations sum to the total
    exactly on all 36 printed rows — so Malawi is one of the very few countries here with
    no §3.5 gap of any kind.
    """
    from mw2018 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "mw.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "district"].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "mw" / "mw_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"mw.csv districts with no polygon: {missing} -- re-run "
                         "sources/mw_geo.py, the lookup is stale")
    if df["unit"].nunique() != 32:
        raise SystemExit(f"{df['unit'].nunique()} districts, expected 32")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


def _bj_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    Benin wants the grid for the ordinary reason and a coastal one. Karimama and Malanville
    are large northern communes holding most of the W National Park and almost nobody;
    Cotonou is 679,012 people in 80 km². And the lagoons — Lac Nokoué and the Porto-Novo
    lagoon — are INSIDE the communes rather than cut out of them, so an equal share would
    scatter Sô-Ava's dots over open water. A population grid has no hexes on empty water and
    does have them over Ganvié, the stilt town of ~30,000 built on that lake, which is spec
    §8.2c-i's point standing in one place (sources/bj_grid.py).
    """
    return _kontur_place_weight(place, "bj_hexes.gpkg", "sources/bj_grid.py")


def _bj_counts():
    """INStaD RGPH-4 2013, Tableau 8 at commune: 10 drawn categories on 77 communes.

    ONE level, no allocation, nothing modelled — INStaD publishes these categories at this
    geography and the map draws exactly that, so every row is `measured` and may ring.

    THE COUNTS ARE ARITHMETIC ON TWO PUBLISHED FIGURES, WHICH IS NOT THE SAME AS AN
    ESTIMATE. Tableau 8 prints shares to one decimal and Tableau 2 of the same booklet
    prints the commune's population, so `count = pct/100 x total` and nothing is carried,
    fitted or inferred — every person here was counted by INStaD in the commune they are
    drawn in. The cost is precision, not confidence: +/-0.05% of a unit, which is +/-34
    people in a 68,000-person commune. §7a's tiers are about whether anybody was counted,
    so these are `measured`.

    77 UNITS FOR 10.0M PEOPLE IS ~130,000 EACH — finer per head than Malawi's districts and
    eight times finer than Kenya's counties.

    TWO CATEGORIES DO NOT REACH THE MAP AND ONLY ONE IS A LOSS. `Non déclaré (calculé)` is
    the computed complement of the ten published shares, 120,826 people at 1.21%, and §3.5
    marks non-response rather than filling it. There is no other gap: the drawn population
    is 9,887,923, which is 98.79% of the census.

    COTONOU IS ONE POLYGON AND THE CENSUS OFFERED THIRTEEN. bj.csv carries the thirteen
    arrondissement rows and they are not drawn — no boundary layer for them could be
    verified. sources/bj.md §5 has the measurement; this is §3.10's rule taken the
    conservative way for once.
    """
    from bj2013 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "bj.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "commune"].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "bj" / "bj_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"bj.csv communes with no polygon: {missing} -- re-run "
                         "sources/bj_geo.py, the lookup is stale")
    if df["unit"].nunique() != 77:
        raise SystemExit(f"{df['unit'].nunique()} communes, expected 77")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


def _zw_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    Zimbabwe is Kenya's placement problem in its purest form: 10 provinces for 15.2M people
    is the coarsest counting geography on this map, and the provinces are wildly uneven.
    Matabeleland North and South are 129,197 km² between them — a third of the country,
    much of it Hwange, the Zambezi escarpment and dry ranching land — while Harare and
    Bulawayo are 872 km² and 479 km² holding 3.1M people. An equal share would wash the
    empty west and squash a fifth of the country into two specks. It also handles Lake
    Kariba, 5,580 km² of which sits inside the provinces (sources/zw_grid.py).
    """
    return _kontur_place_weight(place, "zw_hexes.gpkg", "sources/zw_grid.py")


def _zw_counts():
    """ZIMSTAT 2022 PHC Table 2.14(c) at province: 11 drawn categories on 10 provinces.

    ONE level, no allocation, nothing modelled — every row is `measured` and may ring.

    10 UNITS FOR 15.2M PEOPLE IS ~1.5M EACH, THE COARSEST COUNTING GEOGRAPHY ON THIS MAP,
    and it is ZIMSTAT's ceiling rather than a choice: Table 2.14 is the only religion table
    in the 259-page report, and religion got none of the five 2022 PHC thematic reports.
    Spec §3.9b is why it is drawn anyway — there is no minimum unit count; take the finest
    geography published, and say what the country therefore cannot show.

    ONE CATEGORY RESOLVES TO NOTHING and it is the universe row, so the drawn population is
    the whole table: **15,178,957, the entire census count.** The eleven categories sum to
    the province total on all ten rows exactly — no `not stated`, no residual, no §3.5 gap.

    THE NO-RELIGION CATEGORY IS THE LITERAL STRING `None` AND PANDAS WILL DELETE IT. §12's
    Philippine trap, third sighting after `ph` and `gy`: default `read_csv` parsing turns
    that cell into NaN, it then resolves to nothing in the taxonomy, and 1,255,578 people —
    8.3% of Zimbabwe, the category a religion map most needs to be honest about — vanish
    with no error and no count anywhere. `keep_default_na=False, na_values=[""]` is
    load-bearing on this country, not boilerplate.
    """
    from zw2022 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "zw.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "province"].copy()
    if "None" not in set(df["source_category"]):
        raise SystemExit("zw.csv has no `None` category -- it has been read as NaN, and "
                         "1.26M people are about to disappear (§12, the Philippine trap)")

    lut = pd.read_csv(HERE / "data" / "geo" / "zw" / "zw_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"zw.csv provinces with no polygon: {missing} -- re-run "
                         "sources/zw_geo.py, the lookup is stale")
    if df["unit"].nunique() != 10:
        raise SystemExit(f"{df['unit'].nunique()} provinces, expected 10")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


def _ni_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    Nicaragua needs this for both of §8.2's reasons at once. EMPTINESS: the two Caribbean
    autonomous regions are 46% of the land and 12% of the people, and Waspám alone is
    9,341 km² — larger than eleven whole departments. WATER: Lake Cocibolca is 8,264 km²
    and Xolotlán 1,042, and the municipal boundaries run out into both.

    And the two compound, because the emptiest units are the ones carrying the category
    Nicaragua is drawn for: Prinzapolka, Puerto Cabezas and Waspám are 43-53% Moravian and
    are among the largest municipios in the country. An equal share per polygon would paint
    the Moravian coast across uninhabited rainforest (sources/ni_grid.py).
    """
    return _kontur_place_weight(place, "ni_hexes.gpkg", "sources/ni_grid.py")


def _ni_counts():
    """INIDE 2005 census variable P13 at municipio: 8 categories on 153 municipios.

    ONE level, no allocation, nothing modelled — every row is `measured` and may ring.

    THE COUNTS COME OUT OF INIDE'S OWN REDATAM SERVER, NOT OUT OF A PUBLICATION. INIDE
    prints religion by DEPARTMENT (17 units, Volume I CUADRO 12) and serves it by
    municipality and by comarca; Volume IV's 546 pages of municipal tables carry no religion
    table at all. sources/ni.py runs the query and checks its nine national figures against
    the 2006 printed volume, which is a genuinely independent witness (sources.md §11x).

    THE COMARCA TABLE EXISTS AND IS NOT DRAWN. `OF COMR05, PERS05.P13` returns 2,579 units
    at 1,759 people each and reconciles to the same total. There are no comarca boundaries
    published anywhere — OCHA's COD-AB stops at municipio and geoBoundaries 404s on NIC
    ADM3 — so the geography is what limits this country, not the counts.

    THE UNIVERSE IS AGE 5 AND OVER. 4,537,200 of a 5,142,098 census population; the 604,898
    under-fives were never asked and are in `gap=` rather than drawn as a §3.5 undercount.

    THE JOIN IS ON NAME AND MUST STAY THAT WAY. COD's pcode is `NI` + a code in INIDE's own
    format and 145 of 153 agree, which makes a code join look right and be wrong: INIDE's
    9105 is Waspám and COD's NI9105 is Mulukukú. See sources/ni_geo.py, which refuses to run
    if that stops being true.
    """
    from ni2005 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "ni.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "municipio"].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "ni" / "ni_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"ni.csv municipios with no polygon: {missing} -- re-run "
                         "sources/ni_geo.py, the lookup is stale")
    if df["unit"].nunique() != 153:
        raise SystemExit(f"{df['unit'].nunique()} municipios, expected 153")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


def _pe_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    Peru needs this for both of §8.2's reasons at once, and harder than most. EMPTINESS: the
    Amazonian districts run to thousands of km2 at well under one person per km2, and Loreto
    alone is 29% of the land and 3% of the people. THE DESERT is the same problem inverted --
    the coastal districts are rainless waste with everybody in an irrigated valley a few km
    wide. And the two compound on the altiplano, where the most Adventist districts are large,
    high and mostly empty (sources/pe_grid.py).
    """
    return _kontur_place_weight(place, "pe_hexes.gpkg", "sources/pe_grid.py")


def _pe_counts():
    """INEI 2017 census variable C5P26 at district: 8 categories on 1,873 drawn units.

    ONE level, no allocation, nothing modelled -- every row is `measured` and may ring.

    THE COUNTS COME OUT OF INEI'S OWN REDATAM SERVER, NOT OUT OF A PUBLICATION, AND THAT IS
    WHAT MAKES THE COUNTRY DEEP. What INEI PUBLISHES is four categories; what it SERVES is
    eight, at 1,874 districts, unauthenticated. The UNSD oracle reports four because four is
    what INEI forwarded to it -- and the published `otra religión` of 1,115,872 is EXACTLY
    the five extra columns added together, which sources/pe.py asserts. sources.md §11y.

    THE UNIVERSE IS AGE 12 AND OVER. 23,196,391 of a 29,381,884 census population; the
    6,185,493 under-twelves were never asked and are in `gap=` rather than drawn as a §3.5
    undercount. Within the universe the eight categories are an exact partition on all 1,874
    districts -- there is no `no especificado` cell and 100% of the table is drawn.

    THE JOIN IS ON CODE, WHICH REVERSES NICARAGUA DELIBERATELY. COD's adm3_pcode is `PE` +
    the six-digit ubigeo the census tabulates on; 1,872 of 1,874 codes are present and 1,870
    of those agree on the district name outright. A NAME join would be the risky one here,
    because Peru has many districts sharing a name across provinces. sources/pe_geo.py has
    the argument and three witnesses.

    AND TWO DISTRICTS SHARE ONE POLYGON. COD carries a single `Mazamari - Pangoa` polygon
    where the census has two districts, so pe_lookup.csv sends both to PE120699 and they are
    summed here. 62,229 people, 0.27% of the universe, drawn at half Peru's usual resolution
    and not separable on the map.
    """
    from pe2017 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "pe.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "distrito"].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "pe" / "pe_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"pe.csv districts with no polygon: {missing} -- re-run "
                         "sources/pe_geo.py, the lookup is stale")
    if df["unit"].nunique() != 1873:
        raise SystemExit(f"{df['unit'].nunique()} units, expected 1873")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    # Mazamari and Pangoa are two census districts on one polygon; sum them there.
    df = df.groupby(["unit", "node"], as_index=False)["count"].sum()
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


def _kz_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    17 regions over 2.7 MILLION km2 -- the largest units on this map by area by a long way.
    Karaganda region alone is 428,000 km2, bigger than Germany and Poland together, at a
    national density of 7 people per km2, and the population sits in a ring round the edge
    plus Astana and Almaty. An equal share would scatter Karaganda's dots evenly across the
    Betpak-Dala desert (sources/kz_grid.py).
    """
    return _kontur_place_weight(place, "kz_hexes.gpkg", "sources/kz_grid.py")


def _kz_counts():
    """MODELLED (spec §14.10): 8 drawn nodes on 17 regions, and not one is a count.

    KAZAKHSTAN PUBLISHES RELIGION NATIONALLY ONLY -- established four ways in sources.md
    §11u -- so the choice was a modelled country or no country. The model is the census's
    own ethnicity-by-region crossed with the census's own religion-by-ethnicity:

        count(region, religion) = SUM_ethnicity pop(region, eth) x share(religion | eth)

    NOTHING IS INVENTED. §14.4 rule 1 -- never estimate a magnitude a source does not
    publish -- holds by identity: every person placed is a person BNS counted in that
    region, and the model only decides the column. Both margins come back EXACT, which is
    arithmetic and not luck, and sources/kz.py asserts them.

    EVERY ROW IS `modelled` IN §7. There is no measured tier here at all; this is the first
    country on the map of which that is true (China derives from ethnicity too, but §14.5's
    religio-ethnic categories are a different and stronger claim than a fractional share).

    AND IT HAS THE HELD-OUT TEST §14.10's fifth condition asks for, which most modelled
    countries do not. The volume publishes religion x nationality separately for URBAN and
    RURAL Kazakhstan -- the model's own assumption, stated as a testable claim about a
    partition the model never sees. Predicting it from the national coefficients puts
    313,745 people, 1.64% of the country, on the wrong side of the town/country line. Islam
    and Orthodoxy come back to within 3%; `Неверующие` misses by -13% urban and +42% rural,
    because non-belief is an urban behaviour inside every ethnic group and ancestry cannot
    see it. The urban/rural coefficients are deliberately NOT used -- spending the check to
    improve the fit would leave the country with no independent evidence at all.

    THE 11.01% WHO REFUSED TO STATE *ARE* DRAWN, on `unknown`, which reverses this file's
    first version (Anita, 2026-09-07). The census form decides it: Question 11 offers seven
    options and the sixth is `Отказываюсь указать` -- "I decline to state", printed, numbered,
    chosen. That is an answer, not the derived residual §3.5 and tt2011.py are written about,
    and drawing it redistributes nobody. Kazakhstan is 100.00% drawn. taxonomy/kz2021.py has
    the argument; it is the model's SECOND-worst cell and note_public says so.
    """
    from kz2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "kz.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])

    lut = pd.read_csv(HERE / "data" / "geo" / "kz" / "kz_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"kz.csv regions with no polygon: {missing} -- re-run "
                         "sources/kz_geo.py, the lookup is stale")
    if df["unit"].nunique() != 17:
        raise SystemExit(f"{df['unit'].nunique()} regions, expected 17")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    # EVERY row, without exception — there is no measured tier in this country (§7).
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


def _np_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    753 local levels, fine in PEOPLE (38,400 each) and wild in AREA, because Nepal's federal
    map was drawn to equalise population across the Terai, the middle hills and the
    Himalaya at once: Chandragiri is ~90,000 people in ~50 km2 and Namkha in Humla is ~2,500
    in 2,290 km2. §8.2's trick — fine units make a population layer unnecessary — is about
    units that are fine in AREA, and these are not. An equal share would spread the northern
    units' dots evenly over glaciers and ridge lines, and the north is exactly where the Bon
    and the highest Buddhist shares are (sources/np_grid.py).
    """
    return _kontur_place_weight(place, "np_hexes.gpkg", "sources/np_grid.py")


def _np_counts():
    """NSO NPHC 2021 religion Table 1 at local level: 10 categories on 753 local levels.

    ONE level, no allocation, nothing modelled — every row is `measured` and may ring.

    AN EXACT PARTITION. The ten categories sum to the row total on all 918 rows of the
    source and there is no `Other`, no `Not stated` and no residual anywhere in the table,
    so every person in a drawn unit is in a named category. The only thing dropped here is
    the universe row.

    THE 0.82% THAT IS NOT DRAWN IS THE INSTITUTIONAL POPULATION AND IT IS A §3.7 CASE.
    NSO tabulates 239,098 people — barracks, prisons, hospitals, hostels, and Nepal's
    monasteries and gompas — at DISTRICT level only, as a row beside the district's local
    levels rather than inside them. There is no finer geography for it in the source, and
    spreading it across a district's local levels would invent one: the institutional
    population is concentrated in specific places by its nature, so a population-weighted
    spread would be actively wrong rather than merely uncertain. Dropped and stated on the
    map instead, per §3.5 — the `gap=` line below is where the reader learns it, and §3.7's
    point that this is exactly the population R3 most wants to see stands.
    """
    from np2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "np.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "local"].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "np" / "np_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"np.csv local levels with no polygon: {missing[:8]} -- re-run "
                         "sources/np_geo.py, the lookup is stale")
    if df["unit"].nunique() != 753:
        raise SystemExit(f"{df['unit'].nunique()} local levels, expected 753")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


def _th_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    76 provinces for 66.0M people, and wildly uneven: Bangkok is 8.3M in 1,571 km2 while
    Mae Hong Son is 209,200 in 12,681 km2 of forested mountain. **It matters most in the
    deep south, and for the opposite reason to Cambodia's** — Pattani, Yala and Narathiwat
    are dense rather than empty, with their people along the coast and the Pattani river
    and their interiors in the Sankalakhiri range, so an equal share per polygon would put
    this map's sharpest religious boundary in the wrong place inside each province.
    Kontur's 419,176 hexes reproduce the census at 1.085x nationally with a per-province
    median of 0.90 and **not one of the 76 outside a factor of two**, against 38 of 76 for
    a shuffled null. sources/th_grid.py has the evidence.
    """
    return _kontur_place_weight(place, "th_hexes.gpkg", "sources/th_grid.py")


def _th_counts():
    """NSO 2010 census at province: 8 nodes on 76 changwat, via spec §3.10.

    **THE CATEGORIES AND THE GEOGRAPHY COME OUT OF DIFFERENT DOCUMENTS AND ARE REUNITED BY
    `allocate.py`.** No published Thai census table crosses religion with changwat, in
    either census — 2010's Table 4 and 2000's Table 5 both cut religion by
    municipal/non-municipal only, and the per-province files that once existed died with
    `statbbi.nso.go.th`. What survives is the nine categories at five regions, and
    Buddhist and Muslim percentages at all 76 provinces on a two-page provincial indicator
    sheet. So each province's residual — its population minus its Buddhists and Muslims —
    is split by its OWN REGION's composition of the other seven (`--within 1`).

    **98.5% OF THE COUNTRY IS THEREFORE `measured` AND 1.5% IS `derived`**, which is a much
    better split than an allocation usually buys, and the reason is that the two categories
    published per province are the two that hold 98.5% of Thailand. Canada's allocation is
    71.3% derived; this is 1.5%.

    THE RESIDUAL IS SPLIT PER REGION AND NOT NATIONALLY, and that is the whole reason for
    `--within`: Christianity is 3.05% of the North and 0.35% of the Northeast, so a pooled
    national share would take the hill churches of Chiang Mai and Mae Hong Son and scatter
    them evenly across Isan. §3.10c found this with India and Thailand is the second
    customer.

    ONE PROVINCE IN 76 HAS NO SHEET. Kanchanaburi's `<Province>_T.pdf` was never archived
    anywhere in the Wayback Machine, so its Buddhist and Muslim shares are its region's and
    its rows say so in `note`. 848,000 people, 1.3% of the country, and it is the only
    province here whose headline figures are not its own.
    """
    import th2010
    return _allocated_counts("th", "province", th2010)


def _kh_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    25 provinces for 15.55M people, and wildly uneven: Phnom Penh is 2.28M in 679 km2 while
    Mondul Kiri is 92,213 in 14,288 km2 of the Eastern Highlands. An equal share would wash
    the empty north-east and squash a seventh of the country into one speck — and the
    north-east is exactly where the only substantial non-Buddhist colour on Cambodia's map
    is, so the wash would put it across 25,000 km2 of forest. It also handles the Tonle
    Sap, which runs from ~2,700 to ~16,000 km2 and sits inside five provinces
    (sources/kh_grid.py).
    """
    return _kontur_place_weight(place, "kh_hexes.gpkg", "sources/kh_grid.py")


def _kh_counts():
    """NIS 2019 GPCC Table 2.5.1 at province: 4 drawn categories on 25 provinces.

    ONE level, no allocation, nothing modelled — every row is `measured` and may ring.

    THE COUNTS ARE DERIVED FROM TWO PUBLISHED TABLES AND NEITHER OF THEM HOLDS A COUNT.
    NIS publishes religion as PERCENTAGES to one decimal place (Table 2.5.1) and province
    populations as counts (Table 2.1.1); `sources/kh.py` multiplies them and apportions by
    largest remainder so a province's four categories sum to its published population
    exactly. That is arithmetic on two published figures rather than an estimate, so the
    rows stay `measured` — but every cell carries a rounding band of +/-0.0005 x the
    province population, which is +/-1,141 people in Phnom Penh. See sources/kh.md §3.

    FIFTEEN OF THE HUNDRED CELLS ARE ZERO, and all fifteen are `Other` in provinces where
    NIS prints `0.0`. They are dropped by the `count > 0` filter below and draw nothing,
    which is right: zero is the published figure. It is not evidence that nobody is there
    — a `0.0` is anything under 0.05% — and §3.5 says to mark that rather than fill it.
    """
    from kh2019 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "kh.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "province"].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "kh" / "kh_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"kh.csv provinces with no polygon: {missing} -- re-run "
                         "sources/kh_geo.py, the lookup is stale")
    if df["unit"].nunique() != 25:
        raise SystemExit(f"{df['unit'].nunique()} provinces, expected 25")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


def _mm_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    15 states for 51.5M is ~3.4M each, the second coarsest counting geography on this map,
    and they are wildly uneven: Yangon is 7.36M in 9,867 km2 against Kachin's 1.64M in
    88,978 and Chin's 479k in 36,018 of mountain. An equal share would wash the empty north
    and squash a seventh of the country into one speck.

    AND THE GRID IS 2023 AGAINST A 2014 CENSUS, WHICH MATTERS IN ONE STATE. Rakhine reads
    0.58x against the population it is drawn on and 0.89x against the enumerated count
    alone; the difference is the ~750,000 Rohingya who left in 2017 and are not in a 2023
    surface. So Rakhine's `unenumerated` dots are weighted towards where people live now,
    which is south of where those people were. Nothing published can fix it — see
    sources/mm_grid.py.
    """
    return _kontur_place_weight(place, "mm_hexes.gpkg", "sources/mm_grid.py")


def _mm_counts():
    """DOP 2014 Census Vol 2-C Table 1 at State/Region: 8 drawn categories on 15 units.

    ONE level and no allocation — but NOT all `measured`, which is the point of the country.

    THE EIGHTH CATEGORY IS NOT A RELIGION. `Estimated Non-enumerated population` is DOP's
    own estimate of the people the census did not reach: 1,206,353 nationally, of which
    Rakhine is 1,090,000 and Kayin and Kachin the rest. It maps to `unenumerated` and every
    row of it is `modelled` (§7), because the tiers are about whether anybody was counted
    and here nobody was — 1,090,000 is a round number in the source because it is an
    estimate. `inferred dots: hidden` therefore empties exactly that node and shows the
    census as the state published it.

    THE `Total` ROW IS THE ENUMERATED TOTAL AND IS NOT THIS COUNTRY'S UNIVERSE. Enumerated
    50,279,900 plus non-enumerated 1,206,353 = 51,486,253, which is the report's own overall
    figure and what Myanmar draws.
    """
    from mm2014 import MODELLED, resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "mm.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "state_region"].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "mm" / "mm_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"mm.csv states with no polygon: {missing} -- re-run "
                         "sources/mm_geo.py, the lookup is stale")
    if df["unit"].nunique() != 15:
        raise SystemExit(f"{df['unit'].nunique()} states, expected 15")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    if "unenumerated" not in set(df["node"]):
        raise SystemExit("mm.csv resolved no `unenumerated` rows -- the non-enumerated "
                         "column has been lost, and Rakhine is about to draw 96% Buddhist "
                         "(spec §14.2)")
    df["tier"] = df["node"].map(lambda n: "modelled" if n in MODELLED else "measured")
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


def _ke_counts():
    """KNBS 2019 KPHC Volume IV Table 2.30 at county: 11 drawn categories on 47 counties.

    ONE level, no allocation, nothing modelled — every row is `measured` and may ring.

    47 UNITS FOR 47.2M PEOPLE IS THE COARSEST COUNTING GEOGRAPHY ON THIS MAP, coarser than
    the Philippines' 117, and it is KNBS's ceiling rather than a choice made here: every
    other table in Volume IV is published "by County and Sub-County" and religion is the
    one that stops at county. sources/ke.md §2. The dots are then spread across 230,139
    Kontur hexagons by population, so they land where Kenyans live — but nothing measures
    which part of a county a given church's members are in.

    THREE CATEGORIES RESOLVE TO NOTHING: the universe total, `Don't Know` (73,253) and
    `Not Stated` (6,909). The last two are non-answers per §3.5, so the drawn population is
    47,133,120 — 99.83% of the table's universe, itself 99.26% of the census count. The
    people the census never asked are in hotels, hospitals, prisons, children's homes,
    travelling or sleeping outdoors, and the table says so in its own footnote.
    """
    from ke2019 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "ke.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "county"].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "ke" / "ke_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"ke.csv counties with no polygon: {missing} -- re-run "
                         "sources/ke_geo.py, the lookup is stale")
    if df["unit"].nunique() != 47:
        raise SystemExit(f"{df['unit'].nunique()} counties, expected 47")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


def _gh_counts():
    """GSS 2021 PHC at district: 7 drawn categories on 272 units.

    ONE level in effect, no allocation, nothing modelled — GSS publishes these categories
    at this geography and the map draws exactly that, so every row is `measured` and may
    ring.

    THE DRAWN TIER IS TWO geo_levels AND THAT IS THE ONLY FIDDLY THING HERE. The cube holds
    Ghana, 16 regions, 261 MMDAs, and — for six metropolitan districts — a further split
    into 17 sub-metros. The sub-metros REPLACE their parents (§12's Czechia/Estonia rule)
    because GSS ships boundaries for both tiers, so the drawn set is
    `district` + `submetro` = 255 + 17 = 272. Taking `district` alone is the silent failure
    to avoid: it looks right, reconciles against nothing, and loses 1.7M people in Accra,
    Kumasi, Tema, Tamale, Sekondi-Takoradi and Cape Coast.

    TWO CATEGORIES RESOLVE TO NOTHING and neither is a person lost. `Total` is the universe
    row, and `Christian` is a parent published beside all four of its children, which sum
    to it exactly — see taxonomy/gh2021.py. So the drawn population is the whole universe,
    30,753,327, which is 99.74% of the census count; the other 0.26% never answered the
    question and GSS publishes no cell for them.
    """
    from gh2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "gh.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"].isin(["district", "submetro"])].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "gh" / "gh_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"{len(missing)} gh.csv units have no polygon ({missing[:3]}) -- "
                         "re-run sources/gh_geo.py, the lookup is stale")
    if df["unit"].nunique() != 272:
        raise SystemExit(f"{df['unit'].nunique()} drawn units, expected 272")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


class _IdHexWeighter:
    """Split a drawn unit's dots across Kontur 400 m hexagons by hex POPULATION.

    Indonesia's placement problem is not Kenya's and shows up somewhere else. Most of the
    country is drawn at kecamatan, small enough that an equal share is honest. What is not
    honest is (a) the 89 regencies drawn whole, several of them enormous and nearly empty
    Papuan and Kalimantan units, which is Kenya's failure exactly; and (b) the DENSE URBAN
    kecamatan, which is the one you notice — Cengkareng is 513,920 people and Cakung
    503,846, and an even wash across each polygon makes a city read as flat-shaded tiles
    with administrative edges instead of a built-up area with a shape.

    THE WEIGHT IS POPULATION, NEVER RELIGION, and the distinction is the whole of §14.4
    here. BPS publishes religion at kecamatan and nothing below it, so a Muslim dot and a
    Buddhist dot inside one kecamatan are spread identically. The map gets better at saying
    WHERE THE PEOPLE ARE and no better at all at saying who they are: the street-level
    sorting of, say, Kelapa Gading stays invisible, and weighting religions differently
    inside a unit would be inventing a magnitude the source does not publish.
    """

    def __init__(self, place):
        self.pop = place["pop"].to_numpy(dtype=float)
        self.n_pop = 0
        self.n_uniform = 0

    def weights(self, node, idx, count, plain=False):
        pop = self.pop[idx]
        if pop.sum() > 0:
            self.n_pop += 1
            return pop
        self.n_uniform += 1
        return None

    def summary(self):
        return (f"{self.n_pop:,} (unit, node) rows placed on Kontur hex population, "
                f"{self.n_uniform:,} on equal shares where a unit's cells sum to zero "
                f"(sources/id_grid.py)")


def _id_place_weight(place):
    """countries.py hook. `place` is the hex layer scatter.py has read."""
    if "pop" not in place.columns:
        print("  !! id_hexes.gpkg has no `pop` column — run sources/id_grid.py; "
              "placing on equal shares (§8.2)")
        return None
    return _IdHexWeighter(place)


def _id_counts():
    """BPS Sensus Penduduk 2010: 7 drawn categories on 5,211 units, 237.1M people.

    ONE level in effect and nothing modelled, but **the drawn tier is decided PER UNIT
    rather than by rule**, which is Ghana's shape (§9n) taken one step further. A regency's
    kecamatan REPLACE it where they sum to it exactly in every category — 403 of 492 — and
    the regency is drawn where they do not. So the drawn set is
    `kecamatan` + `regency` + one `province_residual` = 5,122 + 89 + 1 = 5,212, it is
    disjoint, it covers the country once, and **every row is `measured`** and may ring.

    THE ONE `province_residual` IS KALIMANTAN UTARA. Its five regencies became a province in
    2012 and BPS serves them under neither — wid=25 is empty and their regency slots are
    holes — so they were briefly written off as an unrecoverable 0.22% and left a visible
    hole in northern Borneo. They are recoverable without any other source: Kalimantan
    Timur's row is the 2010 province and its listing has only the nine that remain, so the
    per-category residual IS those five. 524,656 people, summing to the published total
    exactly and non-negative in every category. They were carved wholly out of one province
    and are contiguous, so the residual has a shape and is drawn as one unit over it.

    THE TWO LEVELS THAT MUST NOT BE DRAWN ARE IN THE SAME FILE. `regency_covered` is a
    regency its kecamatan already cover, and `kecamatan_partial` are the kecamatan of a
    regency whose listing is short. Either one added to the drawn tier double counts, and
    `kecamatan_partial` on its own would put a whole unit's population into part of it —
    88 regencies have an incomplete sub-district listing and the worst, Kolaka, carries
    82,726 of its true 255,712 people. sources/id.md §7.

    AND THE COMPLETENESS TEST IS PER CATEGORY, NOT ON THE TOTAL. Nduga (9429) publishes
    eight kecamatan carrying a `Total` row and no religion categories at all: the totals
    reconcile exactly, so a total-only test would promote it and draw 79,053 Kristen as
    79,053 people with no religion. It is drawn as a regency instead.

    THREE CATEGORIES RESOLVE TO NOTHING: the universe total, `Tidak Terjawab` (not
    answered, 139,128) and `Tidak Ditanyakan` (not asked, 754,485) — two non-response
    categories that mean opposite things and are kept apart (§3.5, §9p). So the drawn
    population is 236,223,057 of the tier's 237,116,670, itself 99.78% of SP2010's
    237,641,326; the missing 0.22% is the five regencies that became Kalimantan Utara in
    2012, which BPS serves under neither province.
    """
    from id2010 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "id.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    # `province_residual` is the third drawn level and there is exactly one of it:
    # Kalimantan Utara, recovered as Kalimantan Timur's per-category residual because BPS
    # serves its five regencies under neither province. Leaving it out is a 524,656-person
    # hole in the north of Borneo that nothing errors about.
    df = df[df["geo_level"].isin(["kecamatan", "regency", "province_residual"])].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "id" / "id_drawn_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"{len(missing)} id.csv units have no polygon ({missing[:3]}) -- "
                         "re-run sources/id_geo.py, the lookup is stale")
    if df["unit"].nunique() != 5212:
        raise SystemExit(f"{df['unit'].nunique()} drawn units, expected 5,212")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


class _PhBarangayWeighter:
    """Split a unit's dots across its barangays by barangay POPULATION.

    Every other placement layer on this map is one a statistical agency designed to a
    population target, so an equal share per polygon is already a population weighting
    (spec §8.2). Philippine barangays are not that: they are the country's political base
    unit, they range from a few hundred people to well over a hundred thousand, and
    Quezon City's 142 hold as many people as several whole provinces. An equal share would
    put as many dots in an empty upland barangay as in a Metro Manila one.

    So the weight is the barangay's own 2020 population. It is a POPULATION weight and not
    a religion one — unlike Germany, nothing here measures where a given church's members
    live inside a province, so a Baptist dot and a Catholic dot are spread the same way.
    That is §8.2's proxy doing its ordinary job, and it is the reason the map should be
    read as "religion by province, drawn where the people are" rather than as a
    measurement at barangay grain.
    """

    def __init__(self, place):
        self.pop = place["pop"].to_numpy(dtype=float)
        self.n_pop = 0
        self.n_uniform = 0

    def weights(self, node, idx, count, plain=False):
        pop = self.pop[idx]
        if pop.sum() > 0:
            self.n_pop += 1
            return pop
        self.n_uniform += 1
        return None

    def summary(self):
        return (f"{self.n_pop:,} (unit, node) rows placed on barangay population, "
                f"{self.n_uniform:,} on equal shares where a unit's barangays sum to zero "
                f"(sources/ph_geo.py)")


def _ph_place_weight(place):
    """countries.py hook. `place` is the barangay layer scatter.py has read."""
    if "pop" not in place.columns:
        print("  !! ph_barangays.gpkg has no `pop` column — run sources/ph_geo.py; "
              "placing on equal shares (§8.2)")
        return None
    return _PhBarangayWeighter(place)


def _ph_counts():
    """PSA 2020 CPH: 129 categories on 117 provinces, HUCs and the BARMM interim province.

    ONE level, no allocation, nothing modelled. PSA publishes the whole 129 x 117 matrix
    and it is a true partition — the categories sum to each unit's household population
    exactly, in all 117, with no residual to compute (§3.2 has nothing to do here) — so
    every row is `measured` and may ring.

    THE FINE TIER IS province + city + municipality AND IT PARTITIONS THE COUNTRY. The
    `province` rows already EXCLUDE any highly urbanised city inside them and the `city`
    rows are those 33 HUCs plus the City of Isabela; `municipality` is Pateros, the only
    one in NCR. The `region` and `country` rows are aggregates of these and are dropped —
    adding them would double the country.

    THE UNIVERSE IS THE HOUSEHOLD POPULATION, 108,667,043 of 109,035,343 (spec §3.7). The
    368,300 not in it are the institutional population, and unlike Chile's 15+ gap this
    one is NOT scaled up: at 0.34% it changes no share, and it is the one gap this project
    would most like to see, since seminaries, convents and monasteries are exactly what a
    household table cannot reach. Drawn: 99.66% of the country.
    """
    from ph2020 import resolve

    # keep_default_na=False: PSA's category for no religion is the string "None", which
    # pandas turns into NaN under default parsing. It would then fail to resolve and be
    # dropped four lines down, silently removing 43,931 people who answered the question.
    df = pd.read_csv(HERE / "data" / "normalized" / "ph.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"].isin(["province", "city", "municipality"])].copy()

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


class _RuHexWeighter:
    """Split a federal subject's dots across its 3km hexes by hex POPULATION.

    The most consequential placement weight on this map, because Russia has the worst
    within-unit problem on it: the counts are at federal subject, a mean of 1.8 million
    people over a mean of 200,000 km², and Sakha alone is 3.08 million km² with a million
    people living along four rivers. spec §8.2's usual trick — an equal share over a layer
    an agency built to a population target — has nothing to work with here, so the weight
    is a measured population surface (Kontur H3 r6, sources/ru_geo.py).

    It is a POPULATION weight and not a religion one. Nothing in Russia measures where the
    Old Believers or the Sunni live inside a subject, so every node in a subject is spread
    identically and a Buddhist dot in Buryatia sits where Buryatia's people are, not where
    its Buddhists are. Germany is the only country here that does better, because destatis
    publishes religion on its grid; Russia publishes religion for 79 polygons and nothing
    else at all.
    """

    def __init__(self, place):
        self.pop = place["pop"].to_numpy(dtype=float)
        self.n_pop = 0
        self.n_uniform = 0

    def weights(self, node, idx, count, plain=False):
        pop = self.pop[idx]
        if pop.sum() > 0:
            self.n_pop += 1
            return pop
        self.n_uniform += 1
        return None

    def summary(self):
        return (f"{self.n_pop:,} (unit, node) rows placed on 3km hex population, "
                f"{self.n_uniform:,} on equal shares where a subject's hexes sum to zero "
                f"(sources/ru_geo.py)")


def _ru_place_weight(place):
    """countries.py hook. `place` is the 3km hex layer scatter.py has read."""
    if "pop" not in place.columns:
        print("  !! ru_grid_3km.gpkg has no `pop` column — run sources/ru_geo.py; "
              "placing on equal shares, which for Russia is very wrong (§8.2)")
        return None
    return _RuHexWeighter(place)


def _ru_counts():
    """Sreda Arena 2012 at federal subject: 18 answers on 79 units.

    ONE level, no allocation, and none is possible — Arena published one cross-tabulation
    and there is nothing finer or coarser to reconcile it against. ru.csv also carries a
    `country` level, which is Arena's own national column and the same people again.

    THE COARSEST GEOGRAPHY ON THIS MAP BY A WIDE MARGIN. 79 units for 142.6 million people
    is 1.8 million each; North Macedonia, the previous worst, is 23,000. Every dot inside a
    subject is drawn from the same distribution because that is the only thing measured,
    and sources/ru_geo.py places them on a 3km population surface so that at least they sit
    where Russians actually live.

    AND THE ONLY COUNTRY DRAWN FROM A SURVEY AS ITS PRIMARY SOURCE. Russia's census has not
    asked about religion since 1937 and the 2021 census does not either, so there is no
    census route. Arena is 56,900 respondents, about 720 per subject: the large categories
    are solid and the small ones are sampling noise wearing a map. `tier` is `modelled` on
    every row for that reason, which is what spec §7 is for — nothing here is a count of
    anybody, and the Old Believers at 0.32% are the clearest case (see ru2012.py).

    ARENA PUBLISHES SHARES, NOT PEOPLE. sources/ru.py multiplies them by the 2021 census
    population per subject — §3.4's "structure from the detailed source, totals from the
    recent one", the same rule Brazil follows. Applying 2012 shares to 2021 populations is
    deliberate rather than lazy: the Muslim republics grew and the Russian oblasts shrank
    over those nine years, and this carries that shift instead of freezing it.

    ARENA COVERS 79 OF 83, AND THE OTHER FOUR ARE FILLED FROM CENSUS ETHNICITY. It has no
    Chechnya, no Ingushetia, no Nenets and no Chukotka, and the first two are the most
    Muslim republics in the country — so the hole sat exactly where Islam is densest and
    made Russia read as less Muslim than it is. `ru_fill.py` estimates those four from the
    2021 census ethnic composition through a relationship fitted on Arena's own 79 measured
    subjects (islam = 0.724 x muslim ethnic share, R² = 0.964), and this adapter reads the
    filled file. Their rows carry `source_id = ru_ethnic_fill_2021` and are separable at
    any point; every Russian row is `modelled` either way. See ru_fill.py for why Chechnya
    comes out at 84.8% rather than the 98.5% a naive ethnic fill would give.
    """
    from ru2012 import resolve

    path = HERE / "data" / "normalized" / "ru_filled.csv"
    if not path.exists():                       # the fill is a separate step, like br_rescale
        print("  !! ru_filled.csv missing — run `python ru_fill.py`; drawing Arena's 79 "
              "subjects only, so Chechnya and Ingushetia will be blank")
        path = HERE / "data" / "normalized" / "ru.csv"
    df = pd.read_csv(path, dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"] == "subject"].copy()

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["tier"] = "modelled"
    return df[["unit", "node", "count", "congregations", "tier"]]


class _KonturHexWeighter:
    """Split a unit's dots across its Kontur hexes by hex POPULATION.

    Serbia and Lithuania are Kenya's problem at a tenth and a twentieth of the scale, and
    the same answer applies to all three: the counting units are historical districts —
    Serbian opštine average 461 km², Lithuanian savivaldybės 1,088 km² — rather than units
    an agency engineered to a population target, so §8.2's equal share over a fine layer
    has nothing here to be equal over. The weight is a measured population surface
    (Kontur H3 r8; sources/rs_geo.py, sources/lt_geo.py).

    It is a POPULATION weight and not a religion one: neither country measures where a
    given church's members live inside a municipality, so a Catholic dot and an Orthodox
    dot in Subotica are spread the same way. Read a cluster as "this municipality, drawn
    where people actually live", never as a neighbourhood reading.

    One class rather than one per country, because there is nothing per-country in it. The
    Russian and Kenyan weighters above predate it and are the same code; folding them in is
    a tidy-up nobody has needed yet.
    """

    def __init__(self, place, built_by):
        self.pop = place["pop"].to_numpy(dtype=float)
        self.built_by = built_by
        self.n_pop = 0
        self.n_uniform = 0

    def weights(self, node, idx, count, plain=False):
        pop = self.pop[idx]
        if pop.sum() > 0:
            self.n_pop += 1
            return pop
        self.n_uniform += 1
        return None

    def summary(self):
        return (f"{self.n_pop:,} (unit, node) rows placed on Kontur hex population, "
                f"{self.n_uniform:,} on equal shares where a unit's hexes sum to zero "
                f"({self.built_by})")


def _kontur_place_weight(place, grid, builder):
    if "pop" not in place.columns:
        print(f"  !! {grid} has no `pop` column — run {builder}; "
              "placing on equal shares (§8.2)")
        return None
    return _KonturHexWeighter(place, builder)


def _rs_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "rs_grid_400m.gpkg", "sources/rs_geo.py")


def _lt_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "lt_grid_400m.gpkg", "sources/lt_geo.py")


def _cn_place_weight(place):
    """countries.py hook. `place` is the 3km hex layer scatter.py has read."""
    return _kontur_place_weight(place, "cn_grid_3km.gpkg", "sources/cn_geo.py")


def _cn_counts():
    """China 2000 census ETHNICITY at county, rescaled onto 2010 provincial totals.

    THE ONLY COUNTRY HERE WHOSE RELIGION IS NOT IN ITS SOURCE AT ALL. China has never
    asked, so what is counted is nationality and what is drawn is spec §14.5's permitted
    derivation: an ethnic category may imply a religion where the category was itself
    constituted religiously, at no finer geography than the ethnicity is published at.
    `taxonomy/cn2000.py` argues each of the 56 nationalities. Ten Muslim, three Tibetan
    Buddhist, one Theravada, six with a fractional Protestant share, and the other
    thirty-six claim nothing. Nothing here is a count of a religious person, and §7a's
    control removes all the colour at once.

    AND SINCE 2026-09-08 THAT IS NO LONGER THE WHOLE COUNTRY. A fourth layer, below, carves
    Han Buddhism and Protestantism out of the grey at province grain from the pooled Chinese
    General Social Survey — `self_id`, the first thing in China drawn from what people said
    about themselves. It is much the largest colour here: 54.6M Mahayana Buddhists and 20.2M
    Protestants, against the 30.6M the ethnic derivation reaches. China goes from 2.5%
    coloured to 8.4%. See sources/cn_cgss.py, whose docstring carries the whole argument.

    91.6% OF THE COUNTRY IS ONE GREY NODE, AND THAT IS THE POINT RATHER THAN A DEFECT.
    Until 2026-09-07 only 2.2% of China was drawn at all and the eastern half of the map
    was blank; spec §6.12's coverage wash was the only thing standing between that blank
    and a reader concluding nobody in eastern China believes anything. §14.7 decided to
    fill it and §14.13 unblocked it. The 1.14 billion Han are now on `unknown` — counted,
    located, and nothing claimed — because Han religion is §14.5's religiously-mixed row
    and the folk-religion / irreligious boundary is mostly an artefact of how the question
    is asked. **Refusing to draw that boundary is the decision; showing the people is not
    the same as guessing about them.**

    THE MONGOLS ARE NOT DRAWN, AND IT IS THE LARGEST SINGLE CALL IN THE COUNTRY. spec §12
    and §14.5 both send Mongol to Tibetan Buddhism; at 5.81M in 2000 that is more people
    than Tibetans, so Inner Mongolia rather than Tibet would have been the largest block
    of Vajrayana dots in China. §14.5 requires the coefficient to be "documented rather
    than fitted" and for Mongols there is nothing to document — Anita's call, 2026-09-05.
    Tu (241k) goes with them for the same reason. See taxonomy/cn2000.py.

    THE GEOGRAPHY IS 2000 AND THE MAGNITUDES ARE 2010, per spec §3.4. No county-level
    ethnic table newer than 2000 is in the open; the NBS publishes the 2020 provincial
    one as a JPEG scan and the 2010 one as HTML. So each group's county figures are
    scaled to its 2010 provincial total, which moves magnitude and not shape.

    ---- REWRITTEN 2026-09-07, spec §14.13, and the country went from 2.2% to 100% ----

    EVERYBODY IS DRAWN NOW, AT THREE STRENGTHS OF CLAIM, AND ONE SOURCE ROW CAN BECOME
    TWO. `cn2000.shares()` returns a list of (node, share, tier) per nationality instead
    of a single node, so this is a fan-out rather than a `.map()`:

      * 15 nationalities -> `islam` / `buddhism.vajrayana` / `buddhism.theravada` at 1.0,
        tier `derived`. Unchanged, and still spec §14.5's derivation.
      * 6 nationalities -> `christianity.protestant` at a fractional share with the
        remainder on `unknown`, BOTH tier `modelled`. The southwestern mission peoples,
        permitted by §14.9 and applied by us under §14.10; the coefficients and the three
        conditions they had to meet are argued in cn2000.py.
      * everyone else, 1.21 billion people -> `unknown` at 1.0, tier `derived`. Nothing
        is claimed about them; the row says only that the census counted them and where.

    WHY `unknown` IS `derived` AND NOT `measured`, WHICH IS THE ONE SUBTLE CALL HERE.
    Nobody's religion was measured, but that is not what the tier is about — the row makes
    no religious claim at all, so on the religion question it could arguably be anything.
    What settles it is §3.4: the county figure is a 2000 count carried onto a 2010
    provincial total, and §7's table puts "somebody counted this and the number was
    carried to a finer place" squarely in `derived`. Brazil is 41.2% derived for exactly
    this reason. So China stays 100% not-measured and `inferred dots: hidden` still empties
    it — which §14.6 called the honest test of the country and it survives the rewrite.
    """
    from cn2000 import shares

    df = pd.read_csv(HERE / "data" / "normalized" / "cn.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[(df["geo_level"] == "county") & (df["count"] > 0)]

    parts = []
    for cat, sub in df.groupby("source_category", sort=False):
        for node, share, tier in shares(cat):
            if share <= 0:
                continue
            parts.append(pd.DataFrame({
                "unit": sub["geo_id"].to_numpy(),
                "node": node,
                "count": sub["count"].to_numpy(dtype=float) * share,
                "tier": tier,
            }))
    out = pd.concat(parts, ignore_index=True)

    # ---- the CGSS self-id layer, sources/cn_cgss.py -------------------------------
    #
    # THE ONLY THING IN CHINA DRAWN FROM WHAT SOMEBODY SAID ABOUT THEMSELVES. Everything
    # above is spec §14.5's derivation from the nationality column — a claim the map makes
    # about people. This is the pooled Chinese General Social Survey (2012 + 2017 + 2021,
    # n = 32,495, 29 of 31 provinces, 99.2% of the population), which asks 您的宗教信仰是什么
    # and is therefore §3.1's `self_id`, the same basis as Vietnam's census and Russia's Arena.
    #
    # ONLY THE `unknown` RESIDUAL IS CARVED, AND THAT IS THE WHOLE OF THE ARITHMETIC. If the
    # provincial share were applied to a province's entire population, Qinghai and Gansu would
    # count their Tibetans twice — once as `buddhism.vajrayana` from ethnicity and again as
    # generic 佛教 here. So the share eats the grey and nothing else. The cost runs the other
    # way and is named in note_public: in the four provinces where the derived population is
    # large, some of CGSS's Buddhist respondents WERE those people, so re-spreading over the
    # residual alone runs a little high there. Everywhere else the derived share is under 2%.
    #
    # TWO CATEGORIES OF SIX SURVIVE §14.10. Buddhism passes cleanly (χ² p = 4e-184; Zhejiang
    # 15.7% CI 14.0–17.5 against Anhui 0.9% CI 0.4–1.4) and Protestantism is drawn on Anita's
    # call with its weakness disclosed — its spatial variation is highly significant but its
    # 2012↔2021 rank stability is only +0.17. Islam is NOT taken from CGSS: its provincial
    # subsamples find 0.07x Qinghai's Muslims and 2.7x Ningxia's, which is a lottery over
    # sampling units rather than a bias a weight could fix, and §14.5's county derivation is
    # better. Nationally the two agree (CGSS 1.87–2.56%, derivation 1.83%) and that is the
    # first external check §14.5 has ever had. folk, Daoism and Catholicism are too thin.
    #
    # THE LEVEL IS THE POOLED ONE BECAUSE OF THE DENOMINATOR, NOT BECAUSE OF THE SAMPLE SIZE.
    # cn.csv is 2000 structure on 2010 provincial totals, so the people being coloured are the
    # 2010 census's people; pooled and n-weighted, CGSS centres on about 2015, where the 2021
    # wave alone is eleven years downstream of its own denominator. Reported religiosity fell
    # steadily across the waves (any religion 14.5% → 10.6% → 7.5%), so this layer is about
    # half again larger than 2021 alone would draw and note_public says which way it leans.
    cgss = pd.read_csv(HERE / "data" / "normalized" / "cn_cgss.csv")
    prov = df.assign(_p=df["note"].str.extract(r"province=([^;]+)")[0]) \
             .drop_duplicates("geo_id").set_index("geo_id")["_p"]
    piv = cgss.pivot(index="province", columns="node", values="share")

    unk = out["node"] == "unknown"
    province = out["unit"].map(prov)
    carve = {}
    total_share = pd.Series(0.0, index=out.index)
    for node in piv.columns:
        s = province.map(piv[node]).fillna(0.0).where(unk, 0.0)
        carve[node] = s
        total_share = total_share + s
    if (total_share > 1).any():
        raise ValueError("cn_cgss shares sum past 1 in some province")

    carved = [out.assign(count=out["count"] * (1.0 - total_share))]
    for node, s in carve.items():
        part = out.loc[s > 0].copy()
        part["count"] = part["count"] * s[s > 0]
        part["node"] = node
        part["tier"] = "modelled"      # the coefficient is CGSS's, not the census's
        carved.append(part)
    out = pd.concat(carved, ignore_index=True)

    # Several nationalities land on `unknown` in the same county — Han, Miao, Manchu and
    # the non-Christian remainder of the Lisu are four rows saying the same thing. scatter.py
    # would group them anyway; doing it here takes ~60,000 rows to ~10,000 and keeps the
    # `modelled` remainder separate from the `derived` one, which is §7's "the tier keys
    # the row, it does not aggregate over it".
    out = out.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()
    out = out[out["count"] > 0]
    out["congregations"] = 0
    return out[["unit", "node", "count", "congregations", "tier"]]


def _rs_counts():
    """RZS Popis 2022 at municipality: 11 drawn categories on 168 units.

    ONE level, no allocation, nothing modelled. RZS publishes this table at municipality
    and nothing finer, and the categories are a true partition of each unit's population
    with no suppression and no rounding anywhere in it — so every row is `measured` and
    may ring. Croatia's shape (§9e), one country over and one census later.

    THE DRAWN TIER IS 168 UNITS AND THE SHEET CONTAINS SIX LEVELS. rs.csv also carries the
    republic, two halves, four regions, 25 oblasti and — the one that is easy to miss —
    four `city` rows, which are `Grad Niš`, `Grad Požarevac`, `Grad Užice` and
    `Grad Vranje` sitting in the municipality tier as parents of their own city
    municipalities. Filtering to `geo_level == "municipality"` drops all six aggregates,
    which is why sources/rs.py re-levels those four rather than leaving them looking like
    ordinary municipalities.

    BELGRADE AND NIŠ ARRIVE PRE-SPLIT, which removes §12's capital-in-one-polygon problem
    for free: Belgrade is 25.3% of Serbia and is drawn as its 17 city municipalities,
    Niš as its 5. Hungary got the same gift from KSH.

    THE DRAWN POPULATION IS 92.07% OF THE COUNTRY. Two categories resolve to nothing —
    169,486 who declined a question the constitution makes voluntary, and 355,484 whose
    religion RZS records as unknown. taxonomy/rs2022.py argues why neither is irreligion,
    and why excluding the second is not neutral: it is concentrated in the least religious
    municipalities in the country.
    """
    from rs2022 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "rs.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"] == "municipality"].copy()

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


def _lt_counts():
    """Statistics Lithuania, 2021 census, at municipality: 15 nodes on 60 units.

    ONE level, no allocation, nothing modelled. The cube publishes 16 religions at
    municipality and there is nothing finer, so every drawn row is `measured`. lt.csv also
    carries the country, two NUTS2 regions and ten counties, which are the same people
    three more times.

    THE TABLE IS DEEPER THAN ITS SIZE SUGGESTS — Lithuania separates Roman from Greek
    Catholics, Orthodox from Old Believers, and names the Karaims, which no other census on
    this map does. 2.8 million people carry fifteen distinct nodes; Serbia's 6.6 million
    carry eight.

    AND IT IS THE PROJECT'S SHARPEST CASE OF §3.8. 298 of the 1,020 municipality cells are
    withheld as confidential — disclosure control on small religions — so **60.4% of
    Lithuania's Karaims, 29.3% of its Greek Catholics and 28.2% of its Adventists have no
    municipality to be drawn in**, against 0.06% of the population overall. Those people
    are not filled in (§3.5); sources/lt.py reports the shortfall per category. The map
    therefore understates exactly the categories it is most interesting for, and it
    understates them by more the smaller they are.

    THE DRAWN POPULATION IS 86.3% OF THE COUNTRY. `Nenurodyta` — not stated, 384,094
    people, 13.67% — is excluded as a refusal, and taxonomy/lt2021.py notes that it has
    trebled since 2001 while the explicit no-religion answer has not moved.
    """
    from lt2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "lt.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"] == "municipality"].copy()
    df["geo_id"] = df["geo_id"].str.zfill(2)

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


def _kr_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "kr_grid_400m.gpkg", "sources/kr_geo.py")


def _kr_counts():
    """South Korea 2015 census, at si/gun/gu: 10 nodes on 229 units.

    ONE level, no allocation, nothing modelled. kr.csv also carries the country and the 17
    provinces, which are the same people twice more, and a `gu` level of 35 rows that is
    deliberately NOT drawn — KOSIS publishes the general gu of twelve large cities and no
    boundary set carries them, so the drawn tier is the 229 those cities belong to. See
    sources/kr.py.

    THE LAST TIME KOREA WAS ASKED. The religion question was dropped after 2015, so unlike
    every other country here this is not a vintage waiting to be superseded.
    """
    from kr2015 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "kr.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "sigungu"].copy()
    df["count"] = df["count"].astype(int)

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


def _gy_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "gy_grid_400m.gpkg", "sources/gy_geo.py")


def _vn_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "vn_grid_400m.gpkg", "sources/vn_geo.py")


def _vn_counts():
    """Vietnam 2009 census at province: 13 nodes on 63 units.

    ONE level, no allocation, nothing modelled — every row is `measured`.

    **100% OF THE CENSUS POPULATION IS DRAWN, AND FOUR FIFTHS OF IT LANDS ON ONE NODE.**
    Biểu 7 counts only people who reported belonging to a state-recognised religious
    organisation — 15.65M — and has no row at any geography for the other 70.2M. sources/vn.py
    computes that row as each province's Biểu 1 population minus its Biểu 7 total, which is
    the complement of a published partition and reconciles to the person, and it resolves to
    **`unknown`**: a node whose entire content is that these people were counted and the
    source does not say what they practise. Anita's call 2026-09-06, and spec §14.7's decision
    for China taken first for Vietnam. Drawing them as `unaffiliated` would be the largest
    false claim available on this map; leaving them out left the country reading as empty,
    which §6.12 could label and not fix.

    vn.csv also carries the country rows for 2009 AND the 2019 national table, which has no
    geography at all; only `province` is read, so neither is drawn.
    """
    from vn2009 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "vn.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "province"].copy()
    df["count"] = df["count"].astype(int)

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


def _gy_counts():
    """Guyana 2012 census, at administrative region: 13 nodes on 10 units.

    ONE level, no allocation, nothing modelled, and **100% of the census drawn** — the
    thirteen categories partition the population exactly and there is no non-response
    column to leave out, because the Bureau of Statistics prorated non-response into the
    categories before publishing (see note_public). gy.csv also carries the country row,
    which is the same people a second time.

    TEN UNITS IS THE WHOLE COUNTRY AND IT IS FINER THAN IT SOUNDS. 74,700 people per unit
    is between Lithuania's 40,000 and Kenya's 1,012,000 — Guyana is simply small. The
    reason the map still reads is that Guyana's religious geography is almost entirely a
    coast/interior split, and the regions are cut across exactly that grain.
    """
    from gy2012 import resolve

    # keep_default_na=False: the Bureau's category for no religion is the string "None",
    # so default parsing turns 23,419 people into NaN, they fail to resolve, and the rows
    # are dropped with no error anywhere — §12's Philippines trap, in the second country
    # to hit it. Every check upstream of this line still passes.
    df = pd.read_csv(HERE / "data" / "normalized" / "gy.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "region"].copy()
    df["count"] = df["count"].astype(int)

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


class _EtHexWeighter(_KeHexWeighter):
    """Kenya's hex weighter, on Ethiopia's woredas. Same logic, and it is needed for the
    same reason more sharply: the 50 largest woredas are 38.4% of the land, 5.3% of the
    people and 75% Muslim on average, so an equal share per polygon would wash two fifths
    of the country in one colour over the Ogaden. sources/et_geo.py has the numbers.

    A POPULATION weight, not a religion one — nothing measures where a woreda's Orthodox
    sit inside it, so every node's dots are spread identically.
    """

    def summary(self):
        return (f"{self.n_pop:,} (unit, node) rows placed on Kontur hex population, "
                f"{self.n_uniform:,} on equal shares where a woreda's hexes sum to zero "
                f"(sources/et_geo.py)")


def _et_place_weight(place):
    """countries.py hook. `place` is the hex layer scatter.py has read."""
    if "pop" not in place.columns:
        print("  !! et_hexes.gpkg has no `pop` column — run sources/et_geo.py; "
              "placing on equal shares (§8.2)")
        return None
    return _EtHexWeighter(place)


def _et_counts():
    """Ethiopia 2007 census at woreda: 6 nodes on 738 units.

    ONE level, no allocation, nothing modelled — every row is `measured` and may ring.

    **100% OF THE PUBLISHED TABULATION IS DRAWN, and that is not a rounding claim.** The
    six categories partition the census population exactly: the 738 woredas with data sum
    to 73,750,932 category by category, which is the national figure. There is no
    non-response cell to leave out — see note_public, because that is a fact about the
    tabulation rather than about Ethiopia.

    et.csv also carries the country, region and zone rows, which are the same people two
    and three more times; only `woreda` is read.
    """
    from et2007 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "et.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "woreda"].copy()
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != 738:
        raise SystemExit(f"{df['geo_id'].nunique()} woredas, expected 738 -- re-run "
                         "sources/et.py")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


class _CiHexWeighter(_KeHexWeighter):
    """Kenya's hex weighter, on Côte d'Ivoire's 33 régions. The 10 largest are 47.6% of the
    country's land and 28.9% of its people; Bounkani alone is 21,800 km² holding 427,037,
    against Abidjan's 2,153 km² holding 6.32 million. An equal share per polygon would
    smear the north's dots across the empty Comoé park — and Bounkani is exactly where
    `Animiste` is 24.7%, so a uniform fill would blur the one category this country shows
    most sharply. sources/ci_geo.py has the numbers.

    A POPULATION weight, not a religion one — nothing measures where a région's Harrists
    sit inside it, so every node's dots are spread identically.
    """

    def summary(self):
        return (f"{self.n_pop:,} (unit, node) rows placed on Kontur hex population, "
                f"{self.n_uniform:,} on equal shares where a région's hexes sum to zero "
                f"(sources/ci_geo.py)")


def _ci_place_weight(place):
    """countries.py hook. `place` is the hex layer scatter.py has read."""
    if "pop" not in place.columns:
        print("  !! ci_hexes.gpkg has no `pop` column — run sources/ci_geo.py; "
              "placing on equal shares (§8.2)")
        return None
    return _CiHexWeighter(place)


# sources/ci.py §7a emits these two in place of `Autres religions chrétiennes`. Their
# national magnitudes are ANStat's; their per-région split is a uniform national ratio.
_CI_DERIVED = {"Évangélique", "Autres chrétiens, hors évangéliques"}


def _ci_counts():
    """Côte d'Ivoire RGPH 2021 at région: 8 nodes on 33 units.

    ONE level, no allocation, nothing modelled — every row is `measured` and may ring.

    **97.79% of the ordinary-household population is drawn.** The missing 2.21% is `ND`
    (`Non déclaré`), a non-answer taken off the tree per §3.5 as Kenya's `Not Stated` is
    (taxonomy/ci2021.py). It is NOT `Sans religion`, which is a published cell 5.7x larger
    and is drawn.

    The counts in ci.csv are already the product of Tableau 4.6's percentages, the annex's
    régional populations and Tableau 4.1's national magnitudes — see sources/ci.py, which
    rescales each category so the 33 units sum to its published national count.
    """
    from ci2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "ci.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != 33:
        raise SystemExit(f"{df['geo_id'].nunique()} régions, expected 33 -- re-run "
                         "sources/ci.py")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0

    # §7a's évangélique split is DERIVED: the national magnitude is ANStat's, the per-région
    # distribution is a uniform national ratio and is not established by any source. So the
    # two halves are `tier="derived"` and may never ring (§3.10) — a ring asserts presence
    # in a unit, and nothing here establishes presence unit by unit. Everything else in the
    # file is a published régional percentage and stays `measured`.
    derived = df["source_category"].isin(_CI_DERIVED)
    df["tier"] = derived.map({True: "derived", False: "measured"})
    df["may_ring"] = ~derived
    n_der = int(derived.sum())
    if not n_der:
        raise SystemExit("no derived rows in ci.csv -- sources/ci.py's évangélique "
                         "split did not run; re-run it")
    return df[["unit", "node", "count", "congregations", "may_ring", "tier"]]


class _CfHexWeighter(_KeHexWeighter):
    """Kenya's hex weighter, on the Central African Republic's communes, and CAR is a
    stronger case for it than either Kenya or Ethiopia: the 50 largest communes are 73.0%
    of the country's land and 28.9% of its people. Yalinga is 42,260 km² with 4,768 people
    and Djémah 37,065 km² with 1,845 — 0.11 and 0.05 people per km². An equal share per
    polygon would fill the whole east with an even wash over country that is very nearly
    empty. sources/cf_geo.py has the numbers.

    A POPULATION weight, not a religion one — nothing measures where a commune's Muslims
    sit inside it, so every node's dots are spread identically.
    """

    def summary(self):
        return (f"{self.n_pop:,} (unit, node) rows placed on Kontur hex population, "
                f"{self.n_uniform:,} on equal shares where a commune's hexes sum to zero "
                f"(sources/cf_geo.py)")


def _cf_place_weight(place):
    """countries.py hook. `place` is the hex layer scatter.py has read."""
    if "pop" not in place.columns:
        print("  !! cf_hexes.gpkg has no `pop` column — run sources/cf_geo.py; "
              "placing on equal shares (§8.2)")
        return None
    return _CfHexWeighter(place)


def _cf_counts():
    """CAR RGPH03 2003 at commune: 5 nodes on 177 units.

    ONE level, no allocation, nothing modelled — every row is `measured` and may ring.

    **98.50% of the census is drawn, and the missing 1.50% is not a category.** The five
    cells partition the religion universe of 3,836,736 to within ±2 per row (independent
    rounding, §9at's shape), but that universe is itself 58,403 short of the RGPH03
    population of 3,895,139, which the Ethnicity sheet of the same workbook carries. Those
    people were counted and not asked, or asked and not tabulated; there is no cell for
    them. See note_public — that is a fact about the tabulation rather than about CAR.

    cf.csv also carries the country, prefecture and sous-préfecture rows, which are the
    same people three more times; only `commune` is read.
    """
    from cf2003 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "cf.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "commune"].copy()
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != 177:
        raise SystemExit(f"{df['geo_id'].nunique()} communes, expected 177 -- re-run "
                         "sources/cf.py")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


class _PkHexWeighter(_KeHexWeighter):
    """Kenya's hex weighter, on Pakistan's districts. Balochistan is why: 44% of the
    country's land, 6% of its people, 99.3% Muslim, and Chagai district alone is 44,748 km²
    holding 226,508 people. An equal share per polygon would wash the Makran and the Kharan
    desert — nearly half the map — in evenly spaced dots of one colour.

    A POPULATION weight, not a religion one. Nothing measures where Tharparkar's Hindus sit
    inside Tharparkar, so every node's dots are spread identically.
    """

    def summary(self):
        return (f"{self.n_pop:,} (unit, node) rows placed on Kontur hex population, "
                f"{self.n_uniform:,} on equal shares where a district's hexes sum to zero "
                f"(sources/pk_geo.py)")


def _pk_place_weight(place):
    """countries.py hook. `place` is the hex layer scatter.py has read."""
    if "pop" not in place.columns:
        print("  !! pk_hexes.gpkg has no `pop` column — run sources/pk_geo.py; "
              "placing on equal shares (§8.2)")
        return None
    return _PkHexWeighter(place)


def _pk_counts():
    """Pakistan 2017 census at district: 5 drawn nodes on 135 districts.

    ONE level, no allocation, nothing modelled — every row is `measured` and may ring.

    **SIX SOURCE CATEGORIES BECOME FIVE NODES**, because taxonomy/pk2017.py sends both
    `Hinduism` and `Scheduled Castes` to `hinduism`: the second is a caste category, not a
    religion, and PBS itself says the 2017 split between the two was poorly differentiated
    and was fixed in 2023. Together they are Pakistan's 4,444,870 Hindus.

    **THE TIER IS DISTRICT AND NOT TEHSIL, AND THAT IS §14.4.** pk.csv also carries the 585
    tehsil-level rows and they are deliberately not read. PBS publishes religion by
    district; its tehsil release has no religion table at all. sources/pk.md §3.

    TWENTY DISTRICTS HAVE NO DATA — Azad Kashmir's ten and Gilgit-Baltistan's ten, which
    PBS did not publish. They draw blank, and note_public says so.
    """
    from pk2017 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "pk.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "district"].copy()
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != 135:
        raise SystemExit(f"{df['geo_id'].nunique()} districts, expected 135 -- re-run "
                         "sources/pk.py")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    # Hinduism and Scheduled Castes both land on `hinduism`, so the two rows for a district
    # must be added rather than left as duplicates -- scatter.py allocates per (unit, node).
    df = df.groupby(["unit", "node"], as_index=False).agg(
        {"count": "sum", "congregations": "sum"})
    return df[["unit", "node", "count", "congregations"]]


class _BdHexWeighter(_KeHexWeighter):
    """Kenya's hex weighter, on Bangladesh's upazilas — and here for the OPPOSITE reason.

    Ethiopia and Pakistan need it because a handful of enormous desert units would wash half
    the map in one colour. Bangladesh is the most uniform counting geography on this map —
    544 units averaging 258 km² over a country at 1,027 people/km² — and it needs the weight
    anyway, because the few units that are NOT uniform are precisely the ones the country is
    worth drawing for:

      * the **Chittagong Hill Tracts** — Thanchi at 22 people/km², Belai Chhari 27,
        Baghaichhari 60, against a national 1,027 — which hold every Buddhist and
        tribal-Christian dot in Bangladesh (Juraichhari is 94.6% Buddhist, Ruma 38.2%
        Christian). Placed uniformly, the most distinctive geography on the map smears
        evenly across empty forested ridge.
      * the **Sundarbans** — Shyamnagar, Koyra, Mongla and Dacope, the largest units outside
        the Hill Tracts and mostly uninhabited mangrove. Dacope is also the most Hindu
        upazila in the country at 56.5%.

    A POPULATION weight, not a religion one. Nothing measures where Dacope's Hindus sit
    inside Dacope, so every node's dots are spread identically. sources/bd_geo.py has the
    numbers, and the limit at the other end — central Dhaka thanas smaller than a few hexes.
    """

    def summary(self):
        return (f"{self.n_pop:,} (unit, node) rows placed on Kontur hex population, "
                f"{self.n_uniform:,} on equal shares where an upazila's hexes sum to zero "
                f"(sources/bd_geo.py)")


def _bd_place_weight(place):
    """countries.py hook. `place` is the hex layer scatter.py has read."""
    if "pop" not in place.columns:
        print("  !! bd_hexes.gpkg has no `pop` column — run sources/bd_geo.py; "
              "placing on equal shares (§8.2)")
        return None
    return _BdHexWeighter(place)


def _bd_counts():
    """Bangladesh 2011 census at upazila: 5 nodes on 544 units.

    ONE level, no allocation, nothing modelled — every row is `measured` and may ring.

    **100% OF THE PUBLISHED TABULATION IS DRAWN, and the source proves it twice over.** The
    five categories sum to each unit's own published `RLG_TPOP` on all 617 rows of the file,
    and the 544 upazilas sum to 144,043,696 — BBS's census population — category by
    category. There is no non-response cell to leave out; see note_public, because that is a
    fact about the tabulation rather than about Bangladesh.

    bd.csv also carries the country, division and zila rows, which are the same people three
    more times; only `upazila` is read.
    """
    from bd2011 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "bd.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "upazila"].copy()
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != 544:
        raise SystemExit(f"{df['geo_id'].nunique()} upazilas, expected 544 -- re-run "
                         "sources/bd.py")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


class _MyHexWeighter(_KeHexWeighter):
    """Kenya's hex weighter, on Malaysia's administrative districts.

    Malaysia needs it for the Ethiopia reason rather than the Bangladesh one, and the
    units that need it most are the ones the country is worth drawing for:

      * **interior Sarawak.** Belaga is 16,196 km² with 22,502 people — 1.4 per km²,
        against Malaysia's national 98 — and Bukit Mabong, Kapit and Song are the same
        shape. They are also 86-90% Christian, with everyone living along the Rajang
        and its tributaries and nobody at all on the ridges between. Spread uniformly,
        the most distinctive religious geography in the country washes evenly across
        empty rainforest.
      * **interior Sabah and the peninsular highlands**, which hold the Orang Asli
        districts where `other.my` and the misleading `unaffiliated` cell concentrate
        (my2020.py). Cameron Highlands, Gua Musang, Lipis and Hulu Perak are large,
        mountainous and mostly empty, and what is in them sits in a few valleys.

    A POPULATION weight, not a religion one. Nothing measures where Belaga's Christians
    sit inside Belaga, so every node's dots are spread identically. sources/my_geo.py has
    the numbers, including the eleven districts whose Kontur/census ratio falls outside
    0.6-1.6 and why none of them is a bad polygon.
    """

    def summary(self):
        return (f"{self.n_pop:,} (unit, node) rows placed on Kontur hex population, "
                f"{self.n_uniform:,} on equal shares where a district's hexes sum to zero "
                f"(sources/my_geo.py)")


def _my_place_weight(place):
    """countries.py hook. `place` is the hex layer scatter.py has read."""
    if "pop" not in place.columns:
        print("  !! my_hexes.gpkg has no `pop` column — run sources/my_geo.py; "
              "placing on equal shares (§8.2)")
        return None
    return _MyHexWeighter(place)


def _my_counts():
    """Malaysia 2020 census at administrative district: 7 nodes on 160 units.

    ONE level, no allocation, nothing modelled — every row is `measured` and may ring.

    **100% OF THE PUBLISHED TABULATION IS DRAWN, and it reconciles three ways.** Within
    each of the sixteen state volumes the seven categories sum to the state total and the
    districts sum to the state total; across volumes the sixteen states sum, category by
    category, to the separately published national Table 6 — a different publication, so
    the check is external rather than the file agreeing with itself. The grand total is
    32,447,385, DOSM's census population.

    my.csv also carries the country and state rows, which are the same people twice more;
    only `district` is read.
    """
    from my2020 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "my.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "district"].copy()
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != 160:
        raise SystemExit(f"{df['geo_id'].nunique()} districts, expected 160 -- re-run "
                         "sources/my.py")

    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if unmapped:
        raise SystemExit(f"unmapped Malaysian categories: {unmapped}")
    df = df[df["count"] > 0]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


def _ge_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "ge_grid_400m.gpkg", "sources/ge_geo.py")


def _ge_counts():
    """Geostat 2014 census at region: 10 nodes on 11 units.

    ONE level, no allocation, nothing modelled. ge.csv also carries the GEORGIA row, which
    is the same people again.

    **`None` IS A CATEGORY NAME HERE AND PANDAS READS IT AS NaN.** §9m's trap, third
    country. Georgia's irreligious answer is the literal string `None`, so a default
    `read_csv` silently deletes 19,080 people and the map draws a country with no
    unaffiliated population at all. `keep_default_na=False` is not defensive tidiness in
    this file; it is the difference between drawing a category and not.

    **THE COARSEST GEOGRAPHY DRAWN, AND IT EARNS ITS PLACE ON PEOPLE PER UNIT.** 11 regions
    for 3.7 million is about 334,000 each — finer per person than Russia's 79 federal
    subjects at 1.8 million, which is the right comparison and the one that settles it.
    Nothing finer exists: Geostat's census database publishes municipalities for marital
    status and not for religion (sources/ge.md).
    """
    from ge2014 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "ge.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "region"].copy()
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != 11:
        raise SystemExit(f"{df['geo_id'].nunique()} regions, expected 11 -- re-run "
                         "sources/ge.py")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


def _il_counts():
    """CBS 2022 census at statistical area: nodes on 2,968 units.

    **TWO geo_levels, and both are drawn.** CBS splits 142 localities into statistical areas
    and publishes the other 1,043 whole, so the drawn tier is `statarea` PLUS the localities
    that have none. A locality that HAS statistical areas is not in the file at locality
    level at all (sources/il.py drops it), so there is nothing here to double-count -- but
    filtering to one level, which is what every other country in this file does, would
    silently delete either every city or every village.

    **THE UNIT COUNT IS SMALLER THAN THE FILE'S, ON PURPOSE.** il.csv carries every unit CBS
    publishes, including the Judea and Samaria Area and East Jerusalem, because a normalised
    file reproduces its source. The units LAYER is cut on the Green Line, so 267 units have
    counts and no polygon and drop out at the join. That is the design and not a leak;
    `sources/il_geo.py` holds the reasoning and `data/geo/il/dropped_units.json` the list.

    **THE BASIS IS `roll`.** Israel's religion comes from the population register, not from
    a question, so these figures are not comparable with a census that asks (spec §3.1). The
    register has no irreligion box, which is why Israel draws as ~100% religious and why the
    observance split is worth its complications.
    """
    from il2022 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "il.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"].isin(("statarea", "locality"))].copy()
    df["count"] = df["count"].astype(float)

    df["node"] = df["source_category"].map(resolve)
    # `Other religions` is the lump CBS publishes instead of a breakdown, and sources/il.py
    # is supposed to have resolved it against sub-district totals before writing this file.
    # If any survives, it is people about to be dropped without a word -- say so instead.
    lump = df[df["node"].isna() & df["source_category"].str.contains("Other religions")]
    if len(lump):
        raise SystemExit(
            f"il.csv still carries {len(lump):,} unresolved 'Other religions' rows "
            f"({lump['count'].sum():,.0f} people) -- sources/il.py did not run its "
            "allocation. Mapping them to one religion would erase Israel's Christians.")

    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})

    # THE TERRITORIAL CUT HAPPENS HERE, NOT ONLY AT THE JOIN, AND THE DIFFERENCE IS THE
    # LEGEND. il.csv carries every unit CBS publishes; sources/il_geo.py builds polygons only
    # for what is drawn. Leaving the rest in would let scatter.py drop them at the join with
    # a warning — but `counts.json` is built from THIS function, so the legend would go on
    # totalling ~870,000 people who are nowhere on the map. Excluding them here makes the cut
    # one decision in one place, and the count is asserted so a boundary change cannot
    # silently move it.
    dropped_path = HERE / "data" / "geo" / "il" / "dropped_units.json"
    if dropped_path.exists():
        with open(dropped_path, encoding="utf-8") as fh:
            dropped = set(json.load(fh))
        gone = df[df["unit"].isin(dropped)]
        df = df[~df["unit"].isin(dropped)]
        print(f"  il: {len(dropped):,} units beyond the Green Line excluded "
              f"({gone['count'].sum():,.0f} people) — West Bank, Gaza and East Jerusalem; "
              "the Golan is kept (sources/il_geo.py)")
    else:
        raise SystemExit(
            "il: data/geo/il/dropped_units.json is missing — run sources/il_geo.py. "
            "Without it the West Bank and East Jerusalem would be counted in the legend.")

    df["congregations"] = 0
    # §3.10: an allocated row spreads a total and cannot establish presence, so it may not
    # ring. The observance rows are `derived` too and cannot either.
    df["may_ring"] = df["tier"] == "measured"
    return df[["unit", "node", "count", "congregations", "may_ring", "tier"]]


def _xk_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "xk_grid_400m.gpkg", "sources/xk_geo.py")


def _xk_counts():
    """ASK Census 2024 at municipality: 5 nodes on 38 units.

    ONE level, no allocation, nothing modelled. xk.csv also carries the KOSOVA row, which
    is the same people again, and the 2011 census, which sources/xk.py drops.

    **A PERFECT PARTITION, AND THE BLANKS ARE PROVEN ZEROS RATHER THAN ASSUMED ONES.** The
    six categories sum to each unit's total and the 38 municipalities sum to the national
    row category by category, both exactly; 21 blank cells read as zero is the only reading
    that leaves those sums intact (sources/xk.py). Lithuania's §3.8 trap, with the opposite
    answer and a proof rather than a guess.

    **THE FOUR NORTHERN MUNICIPALITIES ARE CORRECTED FROM ASK'S OWN ESTIMATE — Anita,
    2026-09-06.** They were drawn as published for one build and that was wrong in a way the
    note could not fix: their Serb population refused enumeration, and what WAS counted there
    is not a thin sample but a different population, so three of the four came out MAJORITY
    MUSLIM in a table whose own author knows better.

    ASK publishes the correction itself. `census2024_63.px` is the ethnicity table *with
    estimation* and it is identical to the enumerated one everywhere except these four, where
    it restores 16,949 people — **16,369 of them Serbs, 96.6%**. There is no
    religion-with-estimation table, so the step is Serb -> Orthodox: spec §14.5, argued in
    `taxonomy/xk2024.py`, and Kosovo passes its three tests more cleanly than China does.

    THREE THINGS THIS DELIBERATELY DOES NOT DO. It does not touch the enumerated rows — the
    derivation ADDS the missing Orthodox rather than restating what was counted. It does not
    place the 580 non-Serb people in the estimate, because nothing says what they are and the
    enumerated composition of these four is the thing that is not representative (§3.5). And
    it does not launder itself: every added row is `tier="derived"`, so §7a's control removes
    all of it at once and the raw table is one click away.
    """
    from xk2024 import ETHNIC_DERIVATION, resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "xk.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"] == "municipality"].copy()
    if df["geo_id"].nunique() != 38:
        raise SystemExit(f"{df['geo_id'].nunique()} municipalities, expected 38 -- re-run "
                         "sources/xk.py")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["tier"] = "measured"
    out = df[["unit", "node", "count", "congregations", "tier"]]

    # ---- the northern derivation (§14.5) --------------------------------------------
    north_path = HERE / "data" / "normalized" / "xk_north.csv"
    if not north_path.exists():
        raise SystemExit(f"missing {north_path} -- re-run sources/xk.py, which writes it "
                         "alongside xk.csv. Without it the four northern municipalities "
                         "draw as published, which reads them as majority Muslim.")
    nn = pd.read_csv(north_path, dtype={"geo_id": str}, low_memory=False)
    nn = nn[nn["ethnicity"].isin(ETHNIC_DERIVATION) & (nn["added"] > 0)].copy()
    if nn.empty:
        raise SystemExit("xk_north.csv carries no positive derived rows -- the estimate is "
                         "the whole point of reading it")

    # xk_north.csv keys on the municipality NAME; xk.csv keys on ASK's numeric code. Join
    # them through the name rather than assuming the codes line up across two tables.
    key = (pd.read_csv(HERE / "data" / "normalized" / "xk.csv",
                       dtype={"geo_id": str}, low_memory=False)
           .query("geo_level == 'municipality'")[["geo_id", "geo_name"]]
           .drop_duplicates("geo_id"))
    key = dict(zip(key["geo_name"], key["geo_id"]))
    nn["unit"] = nn["geo_name"].map(key)
    if nn["unit"].isna().any():
        raise SystemExit(f"northern municipalities not found in xk.csv: "
                         f"{sorted(nn.loc[nn['unit'].isna(), 'geo_name'])}")

    nn["node"] = nn["ethnicity"].map(ETHNIC_DERIVATION)
    nn = nn.rename(columns={"added": "count"})
    nn["congregations"] = 0
    nn["tier"] = "derived"
    print(f"  xk: +{nn['count'].sum():,} derived Orthodox across "
          f"{nn['unit'].nunique()} northern municipalities (spec §14.5, ASK's own estimate)")

    return pd.concat([out, nn[["unit", "node", "count", "congregations", "tier"]]],
                     ignore_index=True)


def _pt_counts():
    """INE Censos 2021 at freguesia: 11 nodes on 3,092 units.

    ONE level, no allocation, nothing modelled — every drawn row is `measured` and may ring.
    pt.csv also carries the country, three NUTS1, nine NUTS2, 26 NUTS3 and 308 municípios,
    which are the same people five more times; only `freguesia` is read.

    **A PERFECT PARTITION AT EVERY LEVEL, WITH NO SUPPRESSION ANYWHERE.** The 11 categories
    sum to each unit's own published total on all 3,439 units, and the 3,092 freguesias sum
    to the national figure category by category with a largest discrepancy of zero. There is
    no rounding, no withheld cell and no `not stated` column — Judaism is published down to
    single people, which is how Belmonte's 49 survive to be drawn.

    **THE UNIVERSE IS PEOPLE AGED 15 AND OVER WHO ANSWERED, AND IT IS NOT SCALED UP.**
    8,781,900 of a 10,343,066 population: 1,331,188 children are outside the question and
    229,978 more declined and were removed by INE from the denominator rather than published
    (sources/pt.py). Chile is the other 15+ source here and cl2024.py takes the same line —
    drawing 85% of a country is honest, and inflating it to 100% on the assumption that
    children and refusers look like their neighbours is not.
    """
    from pt2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "pt.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"] == "freguesia"].copy()
    if df["geo_id"].nunique() != 3_092:
        raise SystemExit(f"{df['geo_id'].nunique()} freguesias, expected 3,092 -- re-run "
                         "sources/pt.py")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]



class _EsMuniWeighter:
    """Split a province's dots across its municipios by municipal POPULATION.

    Spain's counting geography is the province — 52 units, 940,000 people each — and nothing
    in the country measures religion below it, so this is a population weight and not a
    religion one, exactly as sources/ru.py's is. A Muslim dot in Almería sits where Almería's
    people are, not where its Muslims are, and the same is true of every other node.

    **What that costs is visible and worth stating.** Almería's Muslims are really the
    greenhouse belt — El Ejido, Níjar, La Mojonera, Roquetas — and this spreads them evenly
    over a province that also contains the Sierra de los Filabres. The fix exists and is not
    a weight: the Observatorio del Pluralismo Religioso publishes 7,756 geocoded non-Catholic
    places of worship over 1,378 municipios, which is a §4.4 location layer. Using it here
    would make the map's dots follow buildings rather than people, which is a different claim
    from the one the counts support.
    """

    def __init__(self, place):
        self.pop = place["pop"].to_numpy(dtype=float)
        self.n_pop = 0
        self.n_uniform = 0

    def weights(self, node, idx, count, plain=False):
        pop = self.pop[idx]
        if pop.sum() > 0:
            self.n_pop += 1
            return pop
        self.n_uniform += 1
        return None

    def summary(self):
        return (f"{self.n_pop:,} (unit, node) rows placed on municipal population, "
                f"{self.n_uniform:,} on equal shares")


def _es_place_weight(place):
    """Spain's 8,131 municipios, weighted by WHICH population the dots are of.

    §9as's finding, and Spain is the country it matters most in: **CIS does not sample
    foreigners at all**, so the foreign half is not a correction to the survey but the other
    15% of the country — 7.4M people, a larger share than Italy's 8.5% — and every one of
    them was being scattered by where Spaniards live. INE's table 33571 gives Spanish and
    foreign nationals per municipio at the same five-digit code this layer already uses.
    See `_ItWeighter`; placement only, never a magnitude.
    """
    if not all(c in place.columns for c in ("pop", "spanish", "foreign", "unit")):
        print("  !! es_municipios.gpkg lacks the nationality columns — run "
              "sources/es_geo.py --fetch then sources/es_geo.py")
        return _EsMuniWeighter(place) if "pop" in place.columns else None
    from es2026 import resolve
    return _ItWeighter(place, _foreign_share("es", resolve, "province"),
                       citizen_col="spanish", foreign_col="foreign",
                       citizen_label="Spanish", place_label="municipio")


def _es_counts():
    """Spain at province: 41 nodes on 52 units, from two sources that partition the country.

    ONE level and no allocation, but TWO POPULATIONS, and the reason is the whole country:

      * **Spanish citizens, 42.4M.** CIS's monthly barómetro, 101 studies pooled over three
        years, 464,524 respondents, all 52 provinces — about 8,900 each, which is by a wide
        margin the largest survey sample behind any country on this map. Six answers.
      * **Foreign nationals, 7.4M.** INE's count by province x nationality crossed with Pew's
        composition for each origin country (taxonomy/es_origin.py). **CIS does not sample
        them at all** — its `NACIONALIDAD` variable has two values, both Spanish — so this is
        not a correction to the survey, it is the other 15% of the country.

    Every row is tier `modelled`: a survey is not a count of anybody, and neither is a
    nationality model. §7 draws both desaturated and the about panel says which is which.

    **98.36% of Spain is drawn.** The foreign half is drawn whole; 1.9% of citizens refuse
    CIS's question and spec §3.5 marks that rather than filling it.

    **The one place the two halves have to be reconciled is Islam**, because Spanish-citizen
    Muslims are inside CIS's universe and inside its single unnamed "other religion" cell.
    sources/es.py splits them out using UCIDE's province table; in four provinces — Almería,
    Teruel, Ceuta and Melilla — UCIDE's figure exceeds that whole cell and is capped to it,
    so those four are drawn LESS Muslim than UCIDE would have them, not more.
    """
    from es2026 import resolve

    esp = pd.read_csv(HERE / "data" / "normalized" / "es.csv", dtype={"geo_id": str},
                      keep_default_na=False, na_values=[""])
    esp = esp[esp["geo_level"] == "province"].copy()
    esp["node"] = esp["source_category"].map(resolve)
    unmapped = sorted(set(esp.loc[esp["node"].isna(), "source_category"]))
    if unmapped:
        raise SystemExit(f"es.csv has unmapped source categories: {unmapped}")

    ext = pd.read_csv(HERE / "data" / "normalized" / "es_foreign.csv",
                      dtype={"geo_id": str})
    ext = ext[ext["geo_level"] == "province"]

    df = pd.concat([esp[["geo_id", "node", "count"]], ext[["geo_id", "node", "count"]]],
                   ignore_index=True)
    df["congregations"] = 0
    # Both halves are estimates — a survey is not a count of anybody and neither is a
    # nationality model — so nothing here is `measured` and §7 draws all of it desaturated.
    df["tier"] = "modelled"
    return df.rename(columns={"geo_id": "unit"})[
        ["unit", "node", "count", "congregations", "tier"]]



class _GrLauWeighter:
    """Split a NUTS 2 region's dots across its LAUs, by LAU population.

    Greece's counting geography is NUTS 2 — thirteen regions and Mount Athos — and the
    placement layer is 6,137 LAUs, so this is doing most of the work of making the map look
    like a country rather than like thirteen blobs. It is a population weight and not a
    religion one: nothing in Greece measures religion below the region, so a Muslim dot in
    Attiki sits where Attiki's people are rather than where its mosques are.

    **The one place that costs something visible is Thrace.** The minority is concentrated in
    Rodopi and Xanthi, which are two of Anatoliki Makedonia-Thraki's five regional units, and
    this spreads it evenly across all five — so Kavala and Drama draw Muslim dots they should
    not, and Rodopi draws fewer than it should. The foreign half is available at NUTS 3 and
    could be placed that way; the minority is not, and mixing the two would put the sharper
    geography on the half that has the weaker claim to it.
    """

    def __init__(self, place):
        self.pop = place["pop"].to_numpy(dtype=float)
        self.n_pop = 0
        self.n_uniform = 0

    def weights(self, node, idx, count, plain=False):
        pop = self.pop[idx]
        if pop.sum() > 0:
            self.n_pop += 1
            return pop
        self.n_uniform += 1
        return None

    def summary(self):
        return (f"{self.n_pop:,} (unit, node) rows placed on LAU population, "
                f"{self.n_uniform:,} on equal shares")


def _gr_place_weight(place):
    if "pop" not in place.columns:
        print("  !! gr_lau.gpkg has no `pop` column — run sources/gr_geo.py")
        return None
    return _GrLauWeighter(place)


def _gr_counts():
    """Greece at NUTS 2: 48 nodes on 14 units, from two halves of one census.

    Greece has not asked about religion since 1951. The two populations are Spain's (§9y):

      * **Greek citizens, 9.72M.** ESS rounds 5, 10 and 11 pooled and restricted to
        `ctzcntr = Yes` — 7,885 respondents over 13 regions, about 600 each, which is the
        same order as Russia's federal subjects.
      * **Foreign residents, 759k.** Eurostat's 2021 census table `cens_21ctz_r3`, 200 named
        citizenships at NUTS 3, crossed with Pew's composition for each origin country.

    **Both come out of the same census table**, which publishes `NAT` and `FOR` next to the
    named citizenships, so the halves partition the country by construction rather than by
    reconciliation. 99.8% of Greece is drawn.

    **Two cells are authored and both are in taxonomy/gr2024.py.** The Muslim minority of
    Western Thrace is split out of its region's citizen population, because ESS reaches
    essentially none of it; and Mount Athos, an extra-regio unit of 1,744 monks that no sample
    will ever contain, is drawn Orthodox.
    """
    from gr2024 import resolve

    cit = pd.read_csv(HERE / "data" / "normalized" / "gr.csv", dtype={"geo_id": str},
                      keep_default_na=False, na_values=[""])
    cit = cit[cit["geo_level"] == "nuts2"].copy()
    cit["node"] = cit["source_category"].map(resolve)
    unmapped = sorted(set(cit.loc[cit["node"].isna(), "source_category"]))
    if unmapped:
        raise SystemExit(f"gr.csv has unmapped source categories: {unmapped}")

    ext = pd.read_csv(HERE / "data" / "normalized" / "gr_foreign.csv",
                      dtype={"geo_id": str})
    ext = ext[ext["geo_level"] == "nuts2"]

    df = pd.concat([cit[["geo_id", "node", "count"]], ext[["geo_id", "node", "count"]]],
                   ignore_index=True)
    df["congregations"] = 0
    # A survey is not a count of anybody, a nationality model is not either, and the two
    # authored cells are assertions — so nothing here is `measured` and §7 desaturates it all.
    df["tier"] = "modelled"
    return df.rename(columns={"geo_id": "unit"})[
        ["unit", "node", "count", "congregations", "tier"]]


def _fr_place_weight(place):
    """France's 34,476 communes, weighted by commune population.

    The same weighter Greece uses, and it earns its keep harder here than anywhere else on
    the map. France's counting geography is 21 anciennes régions at 3.10M people each — the
    coarsest here — so 1,642 communes per counted unit is all that stands between this and
    twenty-one flat blobs.

    **And it is a population weight, not a religion one, which in France costs something
    nameable.** Île-de-France is drawn as one composition over 12.2M people, so its 11.4%
    Muslim share spreads across the whole region in proportion to where anyone lives. The
    real geography — Seine-Saint-Denis against Yvelines — is invisible, and no source on this
    map can supply it. sources/fr.md says so in the terms §8.2 asks for.
    """
    if "pop" not in place.columns:
        print("  !! fr_lau.gpkg has no `pop` column — run sources/fr_geo.py")
        return None
    if not all(c in place.columns for c in ("french", "foreign", "unit")):
        print("  !! fr_lau.gpkg lacks the nationality columns — run "
              "sources/fr_geo.py --fetch then sources/fr_geo.py")
        return _GrLauWeighter(place)
    # §9as, and France is where it does the most work: 26 régions of 2.6M people is the
    # coarsest counting geography on this map, so the placement weight is most of what makes
    # the country look like a country. INSEE's RP 2021 TD_NAT1 gives French and foreign
    # nationals per commune at the same code GISCO carries. The five overseas régions are
    # drawn from Pew as whole units and are largely outside TD_NAT1; their communes keep the
    # population weight, which changes nothing because DOM residents are overwhelmingly
    # French nationals. See `_ItWeighter`.
    from fr2024 import resolve
    return _ItWeighter(place, _foreign_share("fr", resolve, "nuts3"),
                       citizen_col="french", foreign_col="foreign",
                       citizen_label="French", place_label="commune")


def _fr_counts():
    """France at NUTS 2: 21 anciennes régions, from two halves of one census table.

    France has never asked about religion in a census and is barred by law from doing so.
    The two populations are Greece's (§9z), both larger:

      * **French citizens, 60.3M.** ESS rounds 5-11 pooled and restricted to
        `ctzcntr = Yes` — 12,678 respondents over 21 régions, about 600 each, the same
        order as Greece's regions and Russia's federal subjects.
      * **Foreign residents, 4.74M.** Eurostat's 2021 census table `cens_21ctz_r3`, 200
        named citizenships at NUTS 3 covering **100.00%** of the foreign population,
        crossed with Pew's composition for each origin country.

    **Both come out of the same census table**, which publishes `NAT` and `FOR` next to the
    named citizenships, so the halves partition by construction rather than by
    reconciliation.

      * **The five overseas régions, 2.22M.** Pew's own 2020 country estimates, one per
        territory, at the one geography where a Pew country row is also a NUTS 2 unit — so
        nothing is downscaled and §14.3's resolution rule holds by identity. Basis
        `estimate`, which is §3.1's own word for a Pew figure. Added 2026-09-07.

    **NO CELL IS AUTHORED, which is what separates this from Greece.** Greece needed two —
    the Thracian minority ESS is blind to, and Mount Athos. France needs none: the survey
    finds Alsace's Protestants, Île-de-France's Muslims and Jews and the Mediterranean's
    Muslims unaided, and where it is weak (the banlieues) no published régional figure
    exists to substitute. Inventing one is spec §14.4's first prohibition.

    **99.30% of France is drawn on 26 units, and the missing 0.51% is Corsica.** ESS's frame
    is metropolitan and excludes Corsica and the overseas régions; Pew covers the five
    overseas territories and not Corsica, which is part of metropolitan France and has no
    ISO code of its own. Nothing is borrowed for it — a metropolitan mixture would draw the
    national average and say nothing true — and §6.12's coverage wash says which.
    """
    from fr2024 import resolve

    cit = pd.read_csv(HERE / "data" / "normalized" / "fr.csv", dtype={"geo_id": str},
                      keep_default_na=False, na_values=[""])
    cit = cit[cit["geo_level"] == "nuts3"].copy()
    cit["node"] = cit["source_category"].map(resolve)
    unmapped = sorted(set(cit.loc[cit["node"].isna(), "source_category"]))
    if unmapped:
        raise SystemExit(f"fr.csv has unmapped source categories: {unmapped}")

    ext = pd.read_csv(HERE / "data" / "normalized" / "fr_foreign.csv",
                      dtype={"geo_id": str})
    ext = ext[ext["geo_level"] == "nuts3"]

    df = pd.concat([cit[["geo_id", "node", "count"]], ext[["geo_id", "node", "count"]]],
                   ignore_index=True)
    df["congregations"] = 0
    # A survey is not a count of anybody and a nationality model is not either, so nothing
    # here is `measured` and §7's inferred-dots mode empties the country completely.
    df["tier"] = "modelled"
    return df.rename(columns={"geo_id": "unit"})[
        ["unit", "node", "count", "congregations", "tier"]]


class _ItWeighter:
    """Place each half of Italy by ITS OWN population, not by the province's total.

    Anita, on the finished map: *"it does look a bit weird to see hinduism spread like
    population-proportionally throughout the italian countryside when in reality i imagine
    it's much more urbanized."* Correct, and the cause is not a missing urban/rural religion
    statistic — Italy's Hindus are 166,000 people and **essentially all of them are in the
    foreign half**, which was being scattered across each province in proportion to where
    *Italians* live. ESS's `domicil` was checked as the obvious fix and is useless for it:
    among citizens it gives Islam no urban gradient at all (1.10% village against 1.10% big
    city) and "Eastern religions" a 4.4x one on about twenty respondents.

    So `sources/it_geo.py` puts ISTAT's comune-level citizenship counts on the placement
    layer and this splits the two:

        citizen-half people  ->  `ital`     53.6M Italian citizens per comune
        foreign-half people  ->  `foreign`   5.4M foreign citizens per comune

    **A node is usually both**, which is why this cannot be a column swap. `islam.sunni`
    draws 491,000 citizens and 1.65M foreign residents, so its weight in a province is the
    two vectors blended in that province's own proportion — computed per (unit, node) from
    the same two CSVs `_it_counts` reads, so the blend cannot drift from the counts.

    **Nothing about a magnitude changes.** Every province's totals are still Eurostat's;
    only the within-province scatter moves, which is what §8.2 asks for.

    Its limit, stated because the map still invites the opposite reading: this places
    *foreigners* well, not *Hindus* specifically. RCS is comune x individual citizenship and
    would put Indians in the Agro Pontino and Chinese in Prato by name, but ISTAT's own
    country-code list is a dead link (see `it_geo.py`), and at 107 province the composition
    already carries most of that — Prato and Latina are each their own provincia.
    """

    def __init__(self, place, fshare, citizen_col="ital", foreign_col="foreign",
                 citizen_label="Italian", place_label="comune"):
        self.pop = place["pop"].to_numpy(dtype=float)
        self.ital = place[citizen_col].to_numpy(dtype=float)
        self.foreign = place[foreign_col].to_numpy(dtype=float)
        self.unit = place["unit"].to_numpy()
        self.fshare = fshare
        self.citizen_label = citizen_label
        self.place_label = place_label
        self.n_split = 0
        self.n_one = 0
        self.n_pop = 0
        self.n_uniform = 0

    def weights(self, node, idx, count, plain=False):
        import numpy as np

        fs = self.fshare.get((self.unit[idx[0]], node))
        if fs is None:
            # A node with no row in either CSV for this unit should not reach here, but a
            # future edition could add one; fall back to the old behaviour rather than
            # dropping it.
            pop = self.pop[idx]
            if pop.sum() > 0:
                self.n_pop += 1
                return pop
            self.n_uniform += 1
            return None

        i, f, p = self.ital[idx], self.foreign[idx], self.pop[idx]
        w = np.zeros(len(idx), dtype=float)
        for share, vec in ((1.0 - fs, i), (fs, f)):
            if share <= 0:
                continue
            s = vec.sum()
            # 16 comuni have no RCS row (they merged after the 2021 boundary vintage) and a
            # handful of provinces could in principle have no foreign residents at all;
            # either way that half falls back to total population rather than vanishing.
            base = vec / s if s > 0 else (p / p.sum() if p.sum() > 0 else None)
            if base is None:
                continue
            w += share * base
        if w.sum() <= 0:
            if p.sum() > 0:
                self.n_pop += 1
                return p
            self.n_uniform += 1
            return None
        if 0.0 < fs < 1.0:
            self.n_split += 1
        else:
            self.n_one += 1
        return w

    def summary(self):
        return (f"{self.n_split:,} (unit, node) rows placed on a blend of "
                f"{self.citizen_label} and foreign {self.place_label} population, "
                f"{self.n_one:,} on one of the two alone, {self.n_pop:,} on total "
                f"population, {self.n_uniform:,} on equal shares")


def _it_place_weight(place):
    """Italy's 7,903 comuni, weighted by WHICH population the dots are of. See `_ItWeighter`."""
    missing = [c for c in ("pop", "ital", "foreign", "unit") if c not in place.columns]
    if missing:
        print(f"  !! it_lau.gpkg has no {missing} — run sources/it_geo.py --fetch "
              f"then sources/it_geo.py")
        return None
    return _ItWeighter(place, _it_foreign_share())


def _foreign_share(cc, resolve, level):
    """{(unit, node): foreign people / all people} — the blend `_ItWeighter` applies.

    Read from the same two files the country's `_counts` reads and summed the same way, so a
    change to either half moves the counts and the placement together. Italy, Spain and
    France all have exactly this shape: a citizen half whose categories need resolving
    through a taxonomy module, and a foreign half already carrying `node`.
    """
    cit = pd.read_csv(HERE / "data" / "normalized" / f"{cc}.csv", dtype={"geo_id": str},
                      keep_default_na=False, na_values=[""])
    cit = cit[cit["geo_level"] == level].copy()
    cit["node"] = cit["source_category"].map(resolve)
    ext = pd.read_csv(HERE / "data" / "normalized" / f"{cc}_foreign.csv",
                      dtype={"geo_id": str})
    ext = ext[ext["geo_level"] == level]

    c = cit.groupby(["geo_id", "node"])["count"].sum()
    f = ext.groupby(["geo_id", "node"])["count"].sum()
    both = pd.concat([c.rename("cit"), f.rename("for")], axis=1).fillna(0.0)
    tot = both["cit"] + both["for"]
    share = (both["for"] / tot.where(tot > 0)).fillna(0.0)
    return {k: float(v) for k, v in share.items()}


def _it_foreign_share():
    from it2024 import resolve
    return _foreign_share("it", resolve, "nuts3")


def _it_counts():
    """Italy at NUTS 3: 107 province, from two halves and THREE resolutions.

    Italy has never asked about religion and **ISTAT does not collect it at all** — it is
    treated as sensitive data and is absent from the census, the permanent census and every
    multiscopo. So the halves are Greece's (§9z) and France's (§9ab):

      * **Italian citizens, 54.0M.** ESS pooled and restricted to `ctzcntr = Yes`.
      * **Foreign residents, 5.03M.** Eurostat's 2021 census table `cens_21ctz_r3`, 200
        named citizenships at NUTS 3 covering **100.00%** of the foreign population,
        crossed with Pew's composition for each origin country.

    **THIS IS THE FIRST COUNTRY DRAWN AT MORE THAN ONE RESOLUTION AT ONCE, AND IT IS
    ANITA'S CALL (§14).** ESS gave Italy NUTS 2 in rounds 6 and 8 and then took it back:
    rounds 9, 10 and 11 hold two thirds of the sample and all of the recent vintage, and
    carry only the five ripartizioni. Greece and France both met an asymmetry like this and
    resolved it by throwing the finer level away. Italy does the opposite:

      | foreign residents        | NUTS 3, 107 province   | measured citizenship counts |
      | Catholic / unaffiliated  | NUTS 2, 20 regioni     | rounds 6+8, 3,368 respondents |
      | every other category     | NUTS 1, 5 ripartizioni | rounds 9+10+11, 7,663 respondents |

    **The reason the mix is worth its explanation is where Italy's minorities are.** The
    citizen minorities are 1.38M people, 2.55% of citizens; the foreign residents are 5.03M
    and are drawn at 107 units. Four fifths of everyone this map exists to show is in the
    half with the finest geography in Europe, and the coarse level lands on the fifth that
    no Italian instrument can locate anyway. Drawing the whole country at NUTS 1 to keep one
    number per unit would have made Italy the coarsest thing on the map to protect a
    precision the foreign half already has.

    **Ten of twenty-one regioni take their Catholic/unaffiliated ratio from the
    ripartizione instead, and that is 14.3% of the population.** Molise is in no ESS round
    at all; the other nine are under `sources/it.py`'s `N_FLOOR` of 100 pooled respondents.
    The floor was added after disbelieving the output — at 23 respondents Trento drove South
    Tyrol to 48% unaffiliated and made it Italy's least Catholic region, which is the
    opposite of everything else known about it.

    **HOW THREE LEVELS BECOME ONE COLUMN.** scatter.py takes a single `unit`, so counting
    happens at NUTS 3 and the coarse rows are spread down proportional to each province's
    own **citizen** population. Nothing is invented: the dots would be placed by population
    inside the coarse unit regardless (§8.2), so this changes nothing spatially — only which
    column the pipeline reads. It does buy one real thing, which is that the citizen half is
    spread by citizen rather than total population, and Italy's foreign share runs from 1.6%
    in Carbonia to 20.6% in Prato. Every citizen row's `note` names the level its
    composition came from.
    """
    from it2024 import resolve

    cit = pd.read_csv(HERE / "data" / "normalized" / "it.csv", dtype={"geo_id": str},
                      keep_default_na=False, na_values=[""])
    cit = cit[cit["geo_level"] == "nuts3"].copy()
    cit["node"] = cit["source_category"].map(resolve)
    unmapped = sorted(set(cit.loc[cit["node"].isna(), "source_category"]))
    if unmapped:
        raise SystemExit(f"it.csv has unmapped source categories: {unmapped}")

    ext = pd.read_csv(HERE / "data" / "normalized" / "it_foreign.csv",
                      dtype={"geo_id": str})
    ext = ext[ext["geo_level"] == "nuts3"]

    df = pd.concat([cit[["geo_id", "node", "count"]], ext[["geo_id", "node", "count"]]],
                   ignore_index=True)
    df["congregations"] = 0
    # A survey is not a count of anybody and a nationality model is not either, so nothing
    # here is `measured` and §7's inferred-dots mode empties the country completely.
    df["tier"] = "modelled"
    return df.rename(columns={"geo_id": "unit"})[
        ["unit", "node", "count", "congregations", "tier"]]


def _ba_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "ba_grid_400m.gpkg", "sources/ba_geo.py")


def _ba_counts():
    """BHAS Popis 2013 at municipality: 5 nodes on 142 units.

    ONE level, no allocation, nothing modelled — every drawn row is `measured` and may ring.

    **THE PARTITION IS EXACT IN BOTH DIRECTIONS**, which is the whole reconciliation and is
    better than most sources here manage: the eight categories sum to each municipality's own
    total, and the 142 municipalities sum to BHAS's published national row category by
    category, both with a discrepancy of zero. Nothing is suppressed, rounded or prorated.
    sources/ba.py asserts it rather than reporting it.

    **98.9% of the country is drawn** — 3,491,871 of 3,531,159. What is not is the two
    non-response cells —
    `Nisu se izjasnili` (32,700 active refusals) and `Bez odgovora` (6,588 with no answer
    recorded) — which BHAS publishes apart and taxonomy/ba2013.py keeps apart, per §3.5.
    """
    from ba2013 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "ba.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"] == "municipality"].copy()
    if df["geo_id"].nunique() != 142:
        raise SystemExit(f"{df['geo_id'].nunique()} municipalities, expected 142 -- re-run "
                         "sources/ba.py")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "tier"]]


def _jm_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "jm_grid_400m.gpkg", "sources/jm_geo.py")


def _jm_counts():
    """STATIN 2011 census at parish: 15 nodes on 14 units.

    ONE level, no allocation, nothing modelled — every drawn row is `measured` and may ring.
    jm.csv also carries the JAMAICA row, which is the same people again, and is dropped.

    **THE COUNTRY IS DRAWN FOR ITS CATEGORIES AND THE GEOGRAPHY IS THE PRICE.** 14 parishes
    is the coarsest counting tier on this map after Guyana's 10 and Georgia's 11, and 19
    religion categories is the best denominational detail in the Americas outside ASARB.
    §11j put the trade as *"Kenya was drawn on 47 units for its categories, and Jamaica's are
    better while its units are three times fewer"*; spec §3.9b removed the unit-count floor
    that had been holding it back.

    **4,124 PEOPLE ARE ABSENT FROM THE SOURCE AND CANNOT BE DRAWN.** STATIN excluded Bahá'í,
    Hinduism, Islam and Judaism from the parish tables — 269, 1,836, 1,513 and 506 people —
    and they are absent rather than pooled into `Other religion`. sources/jm.py asserts the
    gap is exactly 4,124 so a re-release cannot change that silently, and `gap` below states
    it on the map, because a blank cannot distinguish "no Muslims here" from "Muslims were
    not tabulated here" (§6.12).
    """
    from jm2011 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "jm.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"] == "parish"].copy()
    if df["geo_id"].nunique() != 14:
        raise SystemExit(f"{df['geo_id'].nunique()} parishes, expected 14 -- re-run "
                         "sources/jm.py")

    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"])
                      - {"No Data"})
    if unmapped:
        raise SystemExit(f"jm.csv has unmapped source categories: {unmapped}")
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "tier"]]


def _li_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "li_grid_400m.gpkg", "sources/li_geo.py")


def _li_counts():
    """Amt für Statistik Volkszählung 2015 at commune: 10 nodes on 11 units.

    ONE level, no allocation, nothing modelled — every drawn row is `measured` and may ring.
    li.csv also carries the Liechtenstein national row, which is the same people again.

    **AN EXACT PARTITION IN BOTH DIRECTIONS**, which a country of 37,622 people can afford:
    the 11 categories sum to each commune's own total, and the 11 communes sum to the
    national row category by category, both with a gap of zero. Nothing is suppressed,
    rounded or prorated — the census publishes single people, and Planken's one member of
    `Other Christian communities` is on the map.

    **96.7% of the country is drawn** — 36,393 of 37,622. What is not is `Not stated`, 1,229
    people, which taxonomy/li2015.py excludes per §3.5 and which is NOT `No religious
    affiliation`, a separate answer taken by 2,623.
    """
    from li2015 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "li.csv", low_memory=False)
    df = df[df["geo_level"] == "municipality"].copy()
    if df["geo_id"].nunique() != 11:
        raise SystemExit(f"{df['geo_id'].nunique()} communes, expected 11 -- re-run "
                         "sources/li.py")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "tier"]]


def _vc_counts():
    """SVG Statistical Office 2012 census at enumeration district: 16 nodes on 221 units.

    ONE level, no allocation, nothing modelled — every drawn row is `measured` and may ring.
    vc.csv also carries the country and 13 census divisions, which are the same people twice
    more; only `ed` is read.

    **THE FINEST GEOGRAPHY IN THE PROJECT PER HEAD** — 221 units over 109,188 people, a
    median of 415 people and 0.66 km2 each — and the smallest country on the map.

    **THE RECONCILIATION IS CROSS-TABLE AND IS STRONGER THAN A WITHIN-TABLE ONE.** This file
    publishes no religion total; the 18 religion cells are checked against `ETH_TPOP`, the
    separately tabulated ethnicity universe, and they agree TO THE PERSON on all 235 rows at
    every level (sources/vc.py). Two independent questions, one answer.

    **RINGS DO MORE WORK HERE THAN ANYWHERE ELSE.** Six categories are under 400 people —
    Presbyterian 294, Salvation Army 287, Mormon 207, Muslim 111, Hindu 89, Traditional 74 —
    and at 1:1,000 none of them draws a dot. Section 4.3's presence marks are what put them on
    the map at all, and this country is the argument for having built them.

    95.3% of the census is drawn; the 4.67% not stated is spec 3.5.
    """
    from vc2012 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "vc.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"] == "ed"].copy()
    if df["geo_id"].nunique() != 221:
        raise SystemExit(f"{df['geo_id'].nunique()} enumeration districts, expected 221 -- "
                         "re-run sources/vc.py")

    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"]) - {"Not stated"})
    if unmapped:
        raise SystemExit(f"vc.csv has unmapped source categories: {unmapped}")
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["tier"] = "measured"
    # Two categories share `other.vc`, so collapse before returning.
    df = (df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum())
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


def _ch_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "ch_grid_400m.gpkg", "sources/ch_geo.py")


def _ch_counts():
    """Switzerland at commune: 17 nodes on 2,196 units, and every row is `derived`.

    **THE COUNTS COME FROM `ch_rescale.py`, NOT FROM `ch.csv`**, and the difference is the
    whole country. `sources/ch.py` normalises the 2000 census — the last time Switzerland
    asked everybody — onto 2021 commune boundaries. `ch_rescale.py` then fits current
    Strukturerhebung canton totals onto that commune-by-category structure (spec §3.4), which
    is Brazil's move (§9-br) with one extra step.

    **NOTHING HERE IS `measured` AND THAT IS NOT AN OVERSIGHT.** Brazil's rescale changes the
    categories and keeps the geography, so a 2022 município total is a measured number for
    that município and 103.6M Brazilians pass through untouched. Switzerland's changes the
    geography too: the measured quantity is a *canton* total being spread over that canton's
    communes. Every drawn row is therefore `derived`, and §3.10 forbids all of it from
    becoming a presence ring.

    **TWO MARGINS ARE PRESERVED, NOT ONE.** A plain Brazil-style rescale would hold each
    commune's share of its canton's Reformed population fixed since 2000, which puts dots in
    alpine communes that have emptied and starves the suburbs people moved to — a real
    distortion on a map whose subject is where people are. So the fit is an IPF on current
    commune populations (GISCO's `POP_2021`) *and* the survey's canton category totals, with
    the 2000 census supplying only the association between commune and religion. Both margins
    come out exact.

    **99.1% of the survey's universe is drawn** — 7,438,908 of 7,506,664. The remainder is
    `Ohne Angabe`, which taxonomy/ch2000.py excludes per §3.5 and which must not be read as
    `Keine Zugehörigkeit`, a separate answer taken by 36.8%.
    """
    from ch2000 import resolve

    path = HERE / "data" / "normalized" / "ch_commune_rescaled.csv"
    if not path.exists():
        raise SystemExit(f"missing {path} -- run `python ch_rescale.py`. countries.py "
                         "deliberately does NOT read data/normalized/ch.csv: that file is "
                         "the 2000 census as counted, and drawing it would put a "
                         "twenty-six-year-old Switzerland on the map.")
    df = pd.read_csv(path, dtype={"geo_id": str}, low_memory=False)
    if df["geo_id"].nunique() != 2196:
        raise SystemExit(f"{df['geo_id'].nunique()} communes, expected 2,196 -- re-run "
                         "ch_rescale.py")

    df["node"] = df["node_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["tier"] = "derived"
    return df[["unit", "node", "count", "congregations", "tier"]]


class _SkGridWeighter(_KeHexWeighter):
    """Split an obec's dots across the census's OWN 1 km cells by cell population.

    Slovakia is the second country here after Germany whose placement layer is measured
    rather than modelled: `obyv_grid_1km` is SODB 2021 redistributed to 49,969 cells and
    sums to 5,449,270, the census total to the person. So this is a Kontur-shaped weighter
    with none of Kontur's modelling caveat.

    It is still a POPULATION weight and not a religion one — nothing measures where an
    obec's Lutherans sit inside it — so a Catholic dot and a Lutheran dot spread
    identically. Read the map as religion by obec, drawn where Slovaks live.

    2,927 obce over 49,000 km² is 17 km² each, so this matters less than it does in Kenya
    or Kazakhstan; it earns its place on the big rural obce of the north, where the built-up
    part is one valley floor inside a polygon that runs up into the Tatras.
    """

    def summary(self):
        return (f"{self.n_pop:,} (unit, node) rows placed on the census's own 1 km cell "
                f"population, {self.n_uniform:,} on equal shares where an obec's cells sum "
                f"to zero (sources/sk_geo.py)")


def _sk_place_weight(place):
    """countries.py hook. `place` is the 1 km grid GeoDataFrame scatter.py has read."""
    if "pop" not in place.columns:
        print("  !! sk_grid_1km.gpkg has no `pop` column — run sources/sk_geo.py; "
              "placing on equal shares (§8.2)")
        return None
    return _SkGridWeighter(place)


def _sk_counts():
    """SODB 2021 at obec: 10 nodes on 2,927 units.

    ONE level, no allocation, nothing modelled — every drawn row is `measured` and may ring.
    sk.csv also carries the national row, which is the same people again, and the `ostatné`
    residual this drops.

    THERE IS NO JOIN. The religion counts and the municipal polygons are fields and geometry
    on the SAME ArcGIS feature layer (`gis.scitanie.sk`, layer 4 of
    `obyv_ekchar_nabo_vekskup`), so `geo_id` is the polygon's own `uzemie` and §12's first
    two shapes of failure — a silent drop and a confident wrong pairing — cannot arise here.
    Four sweeps had recorded the country as walled while this sat open; see sources.md §11aa.

    `ostatné` IS 7.83% AND IS DRAWN ON `other.sk` — Anita's call, 2026-09-08, reversing a
    first build that excluded it on §3.5. UNSD table 28 publishes the same census with 21
    categories, agrees with this service to the person on the total and on all nine named
    churches, and shows the residual contains `nezistené` — 353,797 people, 83% of the cell.
    Excluding it left 7.83% of the country as a hole, and §6.12 says a hole reads as an
    absence of PEOPLE; drawing it keeps them, at the cost of a node that is not comparable
    with any other country's `other`. The node is labelled `Other or not stated (Slovakia)`
    for that reason, and its density is a map of the census's reach rather than of religion:
    0.0-58.8% between obce, peaking in Roma settlements and city centres. Every drawn row is
    still `measured`. See taxonomy/sk2021.py and sources/sk.md.
    """
    from sk2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "sk.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "obec"].copy()
    df["unit"] = df["geo_id"].astype(str)
    if df["unit"].nunique() != 2927:
        raise SystemExit(f"{df['unit'].nunique()} obce, expected 2,927")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


def _me_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "me_grid_400m.gpkg", "sources/me_geo.py")


def _me_counts():
    """MONSTAT Popis 2023 at municipality: 10 nodes on 23 units.

    ONE level, no allocation, nothing modelled — every drawn row is `measured` and may ring.
    me.csv also carries the national row, which is the same people again, and the two
    residual categories this drops.

    **93.4% OF THE COUNTED COUNTRY IS DRAWN, AND THE TWO THINGS MISSING ARE DIFFERENT.**
    581,388 of 622,537. `Ne želi da se izjasni` is 11,924 people who declined the question
    and are not drawn per §3.5. `zaštićen podatak` is 29,225 people MONSTAT withheld as `z` —
    disclosure control, not an answer — and §3.8 removes those rather than placing them;
    §6.3a's `unknown` names suppression as the one thing it will not accept.

    **THE SUPPRESSION IS THE WHOLE CAVEAT AND IT IS NOT EVENLY SPREAD.** It protects small
    counts, a small count is a LOCAL minority, and so it lands hardest where the national
    majority is scarce: Petnjica loses 29.4% of its people to it, Šavnik 25.4%, Rožaje 21.3%,
    against Tivat 0.8% and Podgorica 0.9%. Per category — which is the only honest way to
    report it (§3.8, Lithuania) — **Islam loses 10.6% of itself against Orthodoxy's 2.7%**,
    measured against MONSTAT's own published national figures. So every municipality on this
    map under-reports whichever of the two is its own minority, and the Bosniak and Albanian
    municipalities under-report most.

    **AND A FURTHER 219 SETTLEMENTS ARE NOT DRAWN AT ALL**, because MONSTAT withheld their
    populations too — between 219 and 1,971 people at the disclosure threshold of ten. They
    are in `gap=` rather than estimated.
    """
    from me2023 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "me.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"] == "municipality"].copy()
    if df["geo_id"].nunique() != 23:
        raise SystemExit(f"{df['geo_id'].nunique()} municipalities, expected 23 -- re-run "
                         "sources/me.py")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "tier"]]


def _sr_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "sr_hexes.gpkg", "sources/sr_grid.py")


def _sr_counts():
    """ABS Census 7 (2004) at ressort: 5 nodes on 62 units.

    ONE level, no allocation, nothing modelled — every drawn row is `measured` and may ring.

    **THE FINEST GEOGRAPHY IN THE AMERICAS HERE, AND THE SHALLOWEST QUESTION IN THE
    CARIBBEAN.** 62 ressorten at ~7,900 people each, and five religion categories. Both
    facts are the country: Suriname is 13.45% Muslim and 19.93% Hindu — the highest Muslim
    share this map draws in the Americas — and the grain is fine enough to show that neither
    is spread evenly (Hinduism 65% in Jarikaba, Islam 48% in Nieuw Amsterdam).

    **84.33% of Suriname is drawn.** What is not is `Don't know/No answer`, 77,204 people,
    **15.67% — the largest non-answer on this map**, which taxonomy/sr2004.py excludes per
    §3.5. Every share drawn here is a share of everybody, not of the people who answered.
    """
    from sr2004 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "sr.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"] == "ressort"].copy()
    if df["geo_id"].nunique() != 62:
        raise SystemExit(f"{df['geo_id'].nunique()} ressorten, expected 62 -- re-run "
                         "sources/sr.py")

    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"])
                      - {"Don't know/No answer", "Total"})
    if unmapped:
        raise SystemExit(f"sr.csv has unmapped source categories: {unmapped}")
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "tier"]]


def _tt_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "tt_hexes.gpkg", "sources/tt_grid.py")


def _tt_counts():
    """CSO 2011 census at municipality: 16 nodes on 15 units.

    ONE level, no allocation, nothing modelled — every drawn row is `measured` and may ring.
    tt.csv carries only the 15 drawn units; the table's `TRINIDAD AND TOBAGO` and `TRINIDAD`
    rows are nested universes used as checks in sources/tt.py and are not written.

    **`keep_default_na=False`, for the same reason as Belize.** `None` is a category name
    here too — 28,842 people — and a bare `pd.read_csv` silently turns it into NaN.

    **88.90% of the country is drawn, and the 11.10% that is not is the largest non-answer
    on this map outside the United States.** `Not Stated` is 146,798 people; §3.5 marks it
    rather than filling it, and `note_public` says so, because every share drawn here is a
    share of the whole non-institutional population rather than of the people who answered.

    **The universe is the NON-INSTITUTIONAL population**, 1,322,546 of the census's
    1,328,019. The missing 5,473 are not scaled in (§14.4).
    """
    from tt2011 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "tt.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[])
    df = df[df["geo_level"] == "municipality"].copy()
    if df["geo_id"].nunique() != 15:
        raise SystemExit(f"{df['geo_id'].nunique()} municipalities, expected 15 -- re-run "
                         "sources/tt.py")
    if "None" not in set(df["source_category"]):
        raise SystemExit("tt.csv has no `None` category -- it has been read as NaN. "
                         "pd.read_csv needs keep_default_na=False here.")

    df["count"] = pd.to_numeric(df["count"])
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"])
                      - {"Not Stated", "Total"})
    if unmapped:
        raise SystemExit(f"tt.csv has unmapped source categories: {unmapped}")
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "tier"]]


def _bb_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "bb_hexes.gpkg", "sources/bb_grid.py")


def _bb_counts():
    """BSS 2010 census at parish: 21 nodes on 11 units.

    ONE level, no allocation, nothing modelled — every drawn row is `measured` and may ring.

    **98.77% of the TABULABLE population is drawn** — 223,419 of 226,193 — the missing part
    being `Not Stated`, 2,774 people, excluded by taxonomy/bb2010.py per §3.5.

    **BUT THE TABULABLE POPULATION IS NOT BARBADOS.** It is 226,193 of an estimated resident
    277,821: the 2010 census has an **18% undercount**, and BSS says so in its own Table A.
    So this country is drawn on 80.4% of its people, and — the part that matters for a map —
    **coverage is not uniform across parishes**, running from 74.6% of St. James to 96.1%
    of St. John. An under-covered parish therefore draws proportionally fewer dots than its
    true population warrants. Nothing here scales it (§14.4), because scaling would assume
    the people the census missed have the same religion mix as the people it found, and
    nothing establishes that.

    **`No Religious Affiliation` is 20.59%** and is spelled out rather than being the literal
    string `None`, so unlike Belize, Trinidad, the Bahamas and Cayman this file's read is not
    load-bearing on `keep_default_na=False`. It is passed anyway, for consistency and
    because `Not Stated` is the next thing that would go wrong.
    """
    from bb2010 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "bb.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[])
    df = df[df["geo_level"] == "parish"].copy()
    if df["geo_id"].nunique() != 11:
        raise SystemExit(f"{df['geo_id'].nunique()} parishes, expected 11 -- re-run "
                         "sources/bb.py")
    if "No Religious Affiliation" not in set(df["source_category"]):
        raise SystemExit("bb.csv has no `No Religious Affiliation` category -- that is "
                         "20.6% of Barbados and the column whose header the source leaves "
                         "BLANK (sources/bb.py). Re-run it.")

    df["count"] = pd.to_numeric(df["count"])
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"])
                      - {"Not Stated", "Total"})
    if unmapped:
        raise SystemExit(f"bb.csv has unmapped source categories: {unmapped}")
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "tier"]]


def _lc_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "lc_hexes.gpkg", "sources/lc_grid.py")


def _lc_counts():
    """CSO 2022 census at district: 21 nodes on 10 units.

    ONE level, no allocation, nothing modelled — every drawn row is `measured` and may ring.

    **95.89% of the household population is drawn** — 164,764 of 171,834 — the missing part
    being `Not reported`, 7,064 people, excluded by taxonomy/lc2022.py per §3.5.

    **WHAT IS DRAWN IS THE CENSUS'S OWN ESTIMATE AND NOT ITS RAW COUNT.** CSO measured a
    **23.3% undercount** and weighted every district back up before publishing — factors
    1.107 in Anse La Raye to 1.507 in Laborie (`sources/lc.py`). That is the exact inverse
    of Barbados (`_bb_counts`), where BSS publishes the uncorrected count, warns that its
    parish tables are understated, and this project declines to scale them (§14.4). Nothing
    here scales anything either; the difference is entirely on the publisher's side.

    **`Mennonite` IS DRAWN AS `christianity.evangelical`, AND THAT IS THE ONE READING
    DECISION ON THIS COUNTRY.** 3,760 people, 2.19%. The census's own questionnaire calls
    that option `Evangelical`, the 2010 census has `Evangelical` at the same 2.2% and no
    Mennonite row, and there is no Mennonite community of that size in Saint Lucia.
    `sources/lc.py` machine-checks the questionnaire against the table on every run and
    refuses to build if the discrepancy changes shape; `taxonomy/lc2022.py` has the full
    chain. **`lc.csv` still carries the source's own label** (§12).

    **THE TABLE MISSES ITS OWN MARGINS BY UP TO 3 PEOPLE** — the cells are independently
    rounded weighted estimates — so the drawn total is 171,829 against a published 171,834.
    Five people in 0.003%; `sources/lc.py` prints the whole spread.
    """
    from lc2022 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "lc.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[])
    df = df[df["geo_level"] == "district"].copy()
    if df["geo_id"].nunique() != 10:
        raise SystemExit(f"{df['geo_id'].nunique()} districts, expected 10 -- re-run "
                         "sources/lc.py")
    if "Mennonite" not in set(df["source_category"]):
        raise SystemExit("lc.csv has no `Mennonite` category -- that is the row the "
                         "census questionnaire calls `Evangelical`, 2.2% of Saint Lucia "
                         "(taxonomy/lc2022.py). Re-run sources/lc.py.")

    df["count"] = pd.to_numeric(df["count"])
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"])
                      - {"Not reported", "Total"})
    if unmapped:
        raise SystemExit(f"lc.csv has unmapped source categories: {unmapped}")
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "tier"]]


def _gd_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "gd_hexes.gpkg", "sources/gd_grid.py")


def _gd_counts():
    """CSO 2021 census at parish: 25 nodes on 7 units.

    ONE level, no allocation, nothing modelled — every drawn row is `measured` and may ring.

    **`geo_level == "parish"` IS LOAD-BEARING, because gd.csv carries two tiers.** The
    census publishes 8 units — it reports the **Town of St. George** apart from the rest of
    the parish — and no boundary set anywhere publishes the town, so `sources/gd.py` writes
    the census's own 8 at `census_unit` and the 7 drawable ones at `parish`. Reading the
    wrong one would double-count St. George. Guarded below.

    **92.89% of the drawn universe is on the map** — 100,581 of 108,279 — the missing part
    being `NOT STATED`, 7,698 people, excluded by taxonomy/gd2021.py per §3.5. That is one
    of the largest non-answers here, and **its geography is the thing the fold hides**: the
    Town of St. George refused at 15.8% against 9.9% for the rest of its parish and 0.91%
    in St. Mark.

    **AND THE UNIVERSE IS ALMOST THE WHOLE COUNTRY**, which is unusual for this region:
    108,279 of a census 109,021, so 99.3% of Grenada is inside the table before the refusals
    come out. Barbados draws on 81.4% of its own estimate and Cayman on 96.3%.
    """
    from gd2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "gd.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[])
    df = df[df["geo_level"] == "parish"].copy()
    if df["geo_id"].nunique() != 7:
        raise SystemExit(f"{df['geo_id'].nunique()} parishes, expected 7 -- gd.csv also "
                         "holds the census's own 8-unit tier at geo_level=census_unit, "
                         "which must NOT be drawn; re-run sources/gd.py")

    df["count"] = pd.to_numeric(df["count"])
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"])
                      - {"NOT STATED", "TOTAL"})
    if unmapped:
        raise SystemExit(f"gd.csv has unmapped source categories: {unmapped}")
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "tier"]]


def _ky_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "ky_hexes.gpkg", "sources/ky_grid.py")


def _ky_counts():
    """ESO 2021 census at district: 15 nodes on 6 units.

    ONE level, no allocation, nothing modelled — every drawn row is `measured` and may ring.

    **`keep_default_na=False`**, as for Belize, Trinidad and the Bahamas. `None` is a
    category name here too and it is the SECOND largest answer in the country — 11,502
    people, 16.72% — so a bare `pd.read_csv` would delete a sixth of the Cayman Islands
    while every check in `sources/ky.py` still passed. Guarded below.

    **THE UNIVERSE IS SMALLER THAN THE CENSUS AND IT IS NOT ONLY REFUSALS.** ESO's tables
    all run on the *census survey tabular population count*, 68,811. The census counted
    71,432; the gap is 327 people in institutions plus a **2,294-person weighted
    non-response estimate** published only nationally. Neither is scaled in (§14.4).

    **98.58% of that universe is drawn** — 67,836 of 68,811. What is not: `DK/NS`, 967
    people, excluded by taxonomy/ky2021.py per §3.5; and eight people in cells ESO printed
    as a dash, dropped by the `count > 0` filter along with every other empty cell.
    """
    from ky2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "ky.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[])
    df = df[df["geo_level"] == "district"].copy()
    if df["geo_id"].nunique() != 6:
        raise SystemExit(f"{df['geo_id'].nunique()} districts, expected 6 -- re-run "
                         "sources/ky.py")
    if "None" not in set(df["source_category"]):
        raise SystemExit("ky.csv has no `None` category -- it has been read as NaN. "
                         "pd.read_csv needs keep_default_na=False here; without it a sixth "
                         "of the Cayman Islands disappears silently.")

    df["count"] = pd.to_numeric(df["count"])
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"])
                      - {"DK/NS", "Total"})
    if unmapped:
        raise SystemExit(f"ky.csv has unmapped source categories: {unmapped}")
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "tier"]]


def _bs_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "bs_hexes.gpkg", "sources/bs_grid.py")


def _bs_counts():
    """BNSI 2022 census at island: 22 nodes on 18 units.

    ONE level, no allocation, nothing modelled — every drawn row is `measured` and may ring.

    **`keep_default_na=False` IS LOAD-BEARING**, for the third time in the Caribbean after
    Belize and Trinidad. `NONE` is a category name here too — 24,668 people, 6.20% of the
    country, the sixth largest answer — and a bare `pd.read_csv` turns those sixteen rows
    into NaN before `resolve()` ever sees them. Guarded below by asserting the category
    survives the parse.

    **95.21% of the Bahamas is drawn.** 379,091 of 398,165; what is not is `NOT STATED`,
    19,074 people, which taxonomy/bs2022.py excludes per §3.5.

    **ONE UNIT HOLDS 74.5% OF THE COUNTRY.** New Providence is 296,732 people, so most of
    what a reader sees on this map is one polygon's composition spread over Nassau by
    Kontur's grid. The seventeen Family Islands are the part with real geography in it, and
    they are also where the census's own suppression bites — see `bs2022.py` on `OTHER
    RELIGION`.
    """
    from bs2022 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "bs.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[])
    df = df[df["geo_level"] == "island"].copy()
    if df["geo_id"].nunique() != 18:
        raise SystemExit(f"{df['geo_id'].nunique()} islands, expected 18 -- re-run "
                         "sources/bs.py")
    if "NONE" not in set(df["source_category"]):
        raise SystemExit("bs.csv has no `NONE` category -- it has been read as NaN. "
                         "pd.read_csv needs keep_default_na=False here; without it 6.2% of "
                         "the Bahamas disappears and every check in sources/bs.py still "
                         "passes, because that file checks the PDF and not this read.")

    df["count"] = pd.to_numeric(df["count"])
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"])
                      - {"NOT STATED", "TOTAL"})
    if unmapped:
        raise SystemExit(f"bs.csv has unmapped source categories: {unmapped}")
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "tier"]]


def _bz_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "bz_hexes.gpkg", "sources/bz_grid.py")


def _bz_counts():
    """SIB 2022 census at district: 11 nodes on 6 units.

    ONE level, no allocation, nothing modelled — every drawn row is `measured` and may ring.

    **`keep_default_na=False` IS LOAD-BEARING AND IS NOT A STYLE CHOICE.** Belize's second
    largest religion answer is the literal string `None`, 123,373 people and 31.04% of the
    country. A bare `pd.read_csv` turns those six rows into NaN, `resolve()` never sees them,
    and the country silently loses **the largest non-Catholic category on its map** — with
    every reconciliation in `sources/bz.py` still passing, because that file checks the
    workbook and not this read. Measured: the default read drops exactly 6 rows and
    123,372.67 people. Guarded below by asserting the category survives the parse.

    **THE COUNTS ARE FLOATS.** SIB publishes undercount-adjusted census figures, so the
    national total is 397,483.456 rather than an integer (`sources/bz.py`). Nothing rounds
    them here; the dot allocator takes fractional counts already.

    **98.96% of Belize is drawn** — 393,348.7 of 397,483.5. What is not is `Don't Know/Not
    Stated`, 4,135 people, which taxonomy/bz2022.py excludes per §3.5.
    """
    from bz2022 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "bz.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[])
    df = df[df["geo_level"] == "district"].copy()
    if df["geo_id"].nunique() != 6:
        raise SystemExit(f"{df['geo_id'].nunique()} districts, expected 6 -- re-run "
                         "sources/bz.py")
    if "None" not in set(df["source_category"]):
        raise SystemExit("bz.csv has no `None` category -- it has been read as NaN. "
                         "pd.read_csv needs keep_default_na=False here; without it 31% of "
                         "Belize disappears and every other check still passes.")

    df["count"] = pd.to_numeric(df["count"])
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"])
                      - {"Don't Know/Not Stated", "Total"})
    if unmapped:
        raise SystemExit(f"bz.csv has unmapped source categories: {unmapped}")
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "tier"]]


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
        how="church membership rolls, topped up by a national survey",
        grain="counties, 104,000 people on average",
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
             "`modelled` — half the American dots, and nobody counted them at any level "
             "(us_rebase.py, 2026-09-05). Adherents are attributed to the congregation's county, not the "
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
        how="census, 2021, 25% sample",
        fill="from the same census at province level",
        grain="census subdivisions, 7,000 people on average",
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
        how="census, 2021, voluntary question",
        grain="municipalities, 1,150 people on average",
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
        how="census, 2021, voluntary question (7% left it blank)",
        fill="from the same census's national table",
        grain="statistical areas, 9,700 people on average",
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
        how="census, 2022",
        fill="from the same census at county level",
        grain="Small Areas, 250 people on average",
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
        how="census, 2020",
        fill="from the same census at state level",
        grain="municipios, 51,000 people on average",
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
        how="census, 2023 totals with 2018 denominations",
        fill="from the 2018 census",
        grain="statistical areas, 2,000 people on average",
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
        how="censuses, 2021 and 2022, voluntary in England and Wales",
        fill="from the same censuses at a coarser geography",
        grain="output areas, 260 people on average",
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
        source="Censo Demográfico 2022 totals on 2010 denominations (IBGE)",
        basis="self-identification, census sample, people aged 10 or over",
        # The dot bbox runs to 28.8°W — Martim Vaz, in the Atlantic, which belongs to the
        # município of Vitória, as Fernando de Noronha belongs to Pernambuco. Real data, but
        # fitting it puts 6° of empty ocean beside the country. Mainland only.
        view=[-74.2, -34.0, -34.2, 5.5],
        note_public=(
            "**Two censuses, and each supplies half of this map.** How many people are in "
            "each município and each broad group is the 2022 census; which denomination "
            "they are drawn as is the 2010 one. IBGE published only nine categories for "
            "2022 and withheld the evangelical breakdown over data quality, saying it may "
            "never appear — so 2010 is not merely the older list, it is the only municipal "
            "denominational data Brazil has. Each município's 2022 evangelical total is "
            "split by that same município's 2010 mix, which is why Assembleia de Deus and "
            "Congregação Cristã can be drawn apart at all. "
            "**The change between the two is the point.** Catholics fell from 64.6% to "
            "56.7% and evangelicals rose from 22.2% to 26.9%, and Umbanda and Candomblé "
            "more than tripled, from 589,000 to 1,850,000 — the largest proportional move "
            "between the censuses, and whether that is growth, reduced stigma or a changed "
            "question is not something the numbers can say. "
            "**Two things to hold.** The 2022 question was asked only of people **aged 10 "
            "or over**, so this draws 176.3 million of Brazil's 203 million and nothing "
            "here scales it up to children. And about 41% of these dots carry a 2022 "
            "magnitude on a 2010 shape: where a denomination has grown or shrunk unevenly "
            "inside a município since 2010, this map cannot see it. The Catholic, Spiritist "
            "and indigenous-tradition dots are the exception — those three came through 2022 "
            "untouched, and they are most of the country. "
            "**Where the dots sit inside a município is a population estimate, not a "
            "measurement.** Religion is published per município and São Paulo is a single "
            "one holding 11.5 million people, so the dots are spread across Brazil's "
            "452,000 census setores in proportion to how many people live in each. That "
            "puts them on the streets rather than in the forest, but nothing measures which "
            "setor a given church's members are in — a Catholic dot and an Assembleia de "
            "Deus dot are spread the same way. Read a cluster as this município drawn where "
            "its people are, never as a neighbourhood."),
        how="census, 2022 totals with 2010 denominations",
        fill="from the 2010 census",
        grain="municípios, 32,000 people on average",
        counts=_br_counts,
        # PLACEMENT IS SETORES, COUNTS ARE MUNICÍPIOS — added 2026-09-05. The counts are
        # published per município and nothing finer exists, but a município is far too big
        # to spread dots evenly inside (São Paulo is one polygon, 11.5M people), so the
        # dots are placed across the ~452,000 census setores weighted by setor population.
        # No `units`/`unit_key` spatial join is needed: a setor's code STARTS WITH its
        # município's, so the assignment is a string slice (sources/br_setores.py).
        #
        # 2022 vintage throughout (§8.1), changed 2026-09-04 with the §3.4 rescale.
        # br_municipios_2022.gpkg is still built and is the right layer for anything that
        # wants municipal outlines; it is no longer what places the dots.
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "br" / "br_setores_2022.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_br_place_weight,
        note="PLACEMENT IS SETORES (2026-09-05, spec §8.2d): the counts are municipal and "
             "cannot be finer, but a município is far too large to spread dots evenly "
             "inside — São Paulo alone is one polygon holding 11.5M people — so the dots "
             "sit in 466,996 census setores weighted by setor population "
             "(sources/br_setores.md). Same 176,291 dots as before, across 150,088 polygons "
             "instead of 5,565, with no fallbacks. A population weight, not a religion one. "
             "§3.4 built 2026-09-04: 2022 municipal totals split by 2010 municipal shares "
             "(br_rescale.py). 58.7% of the drawn people are `measured` — the three 2022 "
             "categories that map to one 2010 leaf — and 41.3% `derived`, which may never "
             "ring. Both censuses are sample tabulations and municipal figures do not sum "
             "to IBGE's national ones, by construction (sources/br.md §4).",
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
        how="census, 2021, voluntary (20.5% refused)",
        grain="gminas, 12,000 people on average",
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
        how="census, 2021, built from registers; missing for 14%",
        grain="communes and towns, 5,100 people on average",
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
        how="census, 2021, voluntary, ages 15 and over",
        grain="municipalities, 11,000 people on average",
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
        how="census, 2021",
        grain="municipalities, 6,700 people on average",
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
        how="census, 2011, answered by the head of household",
        fill="from the same census at state level",
        grain="sub-districts, 200,000 people on average",
        counts=_in_counts,
        # THE COUNT LAYER IS STILL THE SUB-DISTRICT AND NOTHING HERE CHANGES THAT. India's
        # religion figures are published on 5,988 sub-districts and are read from exactly
        # those; the median one still holds about 204,000 people and is still the coarsest
        # count unit on the map. What changed is only where inside one a dot may land.
        #
        # §8.2a is the section that said India could not have a placement layer: its finer
        # geography is 645,828 villages and 4,135 towns, natural settlements running from
        # ten people to two million rather than units built to a population target, so an
        # equal share per polygon would weight a hamlet like a city. The answer is not to
        # share equally but to WEIGHT BY THE SETTLEMENT'S OWN POPULATION, which SHRUG and
        # C-01 between them publish for very nearly all of them — and `place_weight`, built
        # for the US in §8.4, is the hook that takes it.
        #
        # sources/in_place.py does the joining and the weighting; its docstring carries the
        # traps, of which the sharp one is that 3,892 six-digit codes name both a village
        # and a town. Every unit's weights sum to its census total, because each unit also
        # carries its own outline holding whatever its settlements do not account for —
        # Assam publishes no village populations at all, so that fallback is the whole of
        # its rural placement and is precisely §8.2a's behaviour. Nowhere is worse than
        # before; most places are a great deal better.
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "in" / "in_places.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_in_place_weight,
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
            "churches that levy it and nothing else: 51.8% of Germany was one grey "
            "category holding everyone the register has no religious body for — Muslims, "
            "Orthodox Christians, the free churches, the Old Catholics and everyone who "
            "belongs to nothing, indistinguishable from each other, because the register "
            "never knew. That is not a judgement about who counts; it is the shape of the "
            "instrument. Two things have been pulled back out of that cell, and they are "
            "not equally solid. The 87,934 members of Germany's 105 Jewish communities "
            "are COUNTED, by their own welfare organisation, and placed at each "
            "community's seat — affiliated membership rather than everyone Jewish, and a "
            "community serves a whole region from one address, so read those dots as "
            "where the communities are. The Muslim, Orthodox, free-church and other dots "
            "are MODELLED: nobody counts them anywhere in Germany, so the European Social "
            "Survey's answers from 13,643 Germans are used to divide the grey cell, one "
            "composition per federal state. Two independent instruments agreeing is what "
            "licenses that — the survey finds 24.7% Catholic and 23.9% Protestant where "
            "the register counts 25.1% and 23.1%, without being told either. The Muslim "
            "dots are the weakest thing on this page and they are too few: the survey "
            "implies about 3.5 million where the federal migration office estimates 5.5, "
            "because a German-language household survey of adults misses recent arrivals "
            "and a young population. The figure is left as the survey reports it rather "
            "than scaled to fit. The 41% who told the survey they belong to no religion "
            "are deliberately NOT drawn as non-religious and stay in the grey: the "
            "register genuinely counted them into it, and where somebody stands on belief "
            "is the last thing a model like this can see. What survives untouched is the "
            "confessional map itself, at 10,786 municipalities: Catholic Bavaria, the "
            "Rhineland and the Saarland against a Protestant north, a boundary largely "
            "settled in the sixteenth century and still legible village by village. The "
            "sharpest line is not that one. In the former East, 81% belong to no church, "
            "against 45% in the West, and the Eichsfeld — a Catholic enclave that stayed "
            "Catholic through forty years of the GDR — still reads at 80% against a "
            "Thuringia of 74% none."),
        how="church-tax registers; nobody was asked",
        grain="municipalities, 7,700 people on average",
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
             "population — by 174 people in 82.7 million (sources/de.md §3). Judaism is "
             "a SECOND SOURCE on the same basis: ZWST's 2025 membership statistic, "
             "87,934 people in 105 communities seated in 96 Gemeinden and subtracted out "
             "of `unrecorded` rather than added to it (sources/de.md §7). A roll counts "
             "the institution and not the member (§3.6), and these are regional "
             "catchments — Düsseldorf's 6,371 covers much of the lower Rhine. Islam, "
             "Orthodoxy, the free churches and other.de are a THIRD source and the only "
             "modelled part of the country: ESS `rlgdnade` pooled over rounds 6-11 "
             "splits the residual by Bundesland (§3.4, §14.10), 9.7% of Germany, "
             "`modelled` in §7 and therefore removed rather than rolled up when inferred "
             "dots are hidden. The non-religious are NOT split out and stay measured in "
             "`unrecorded` — §14.12's ancestry/attitude rule (sources/de_ess.py). Those "
             "dots are PLACED on destatis' citizenship grid, which shares cell ids with "
             "the religion one: Turkish and Bosnian passports for Islam, Greek, Romanian, "
             "Russian and Ukrainian for Orthodoxy, blended with `son` in each Gemeinde's "
             "own ratio (sources/de.md §10). Placement only, never a magnitude (§8.2).",
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
        how="census, 2022",
        fill="from the same census at county level",
        grain="settlements, 1,800 people on average",
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
        how="census, 2021",
        grain="municipalities, 21,000 people on average",
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
    "lk": dict(
        name="Sri Lanka",
        source="Census of Population and Housing 2024 (Department of Census and Statistics)",
        basis="self-identification",
        view=[79.4, 5.7, 82.1, 10.0],
        note_public=(
            "The finest map here, and the shallowest. Sri Lanka's 2024 census publishes "
            "religion for all 14,003 **Grama Niladhari divisions** — about 1,550 people "
            "each, so a dot sits in something the size of a few streets or one village — "
            "and it offers only six answers to put in them: Buddhist, Hindu, Islam, Roman "
            "Catholic, other Christian, other. Nowhere else on this map is the geography "
            "this good and the religion this coarse. Buddhists are 69.8% of the country "
            "and arrive with no school attached, though the island is one of Theravada's "
            "historic homes; Muslims are 10.7% with no branch given. What the resolution "
            "does buy is that the four traditions sit in visibly different places rather "
            "than blended into district averages: the Hindu north around Jaffna and, "
            "separately, the Hindu hill country in the tea districts — Sri Lankan Tamils "
            "and Indian-origin Tamils, two populations the religion question cannot tell "
            "apart but the map can; Muslims along the eastern coast from Trincomalee to "
            "Kalmunai and in pockets inland; and a Roman Catholic coastal strip running "
            "north from Negombo through Chilaw, which is the sharpest religious boundary "
            "in the country and is invisible at any coarser grain. "
            "**Everyone is drawn, and that is a fact about the question rather than the "
            "country.** The six categories account for all 21,781,800 people: there is no "
            "'no religion', no 'not stated' and no refusal line anywhere in this census. "
            "Irreligious Sri Lankans are not missing from this map, they are counted "
            "inside one of the six, and no total here can be read as a measure of belief. "
            "One more thing the census does to itself: where a religion has fewer than ten "
            "people in a GN division its count is moved into 'other', so the small groups "
            "— the Bahá'ís, the Parsis, the Malay Muslims — cannot be recovered at this "
            "grain, and 'other' holds an unknown mixture of them and that suppressed "
            "tail."),
        how="census, 2024",
        grain="village divisions, 1,550 people on average",
        counts=_lk_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "lk" / "lk_gnd.gpkg",
        place_unit=lambda g: g["gnd"].astype(str),
        note="The census codes and the COD boundary codes are the same shape and disagree: "
             "13 DS divisions were renumbered between COD's 2022 vintage and the 2024 "
             "census, so joining on the pcode places 762,824 people in the wrong division "
             "with no symptom. sources/lk_geo.py aligns DS divisions by name first and "
             "only then matches GN codes inside a pair. 53 GN divisions have no 2022 "
             "polygon and are placed in their DS division's unmatched remainder.",
    ),
    "cl": dict(
        name="Chile",
        source="Censo de Población y Vivienda 2024 (INE)",
        basis="self-identification, people aged 15 or over",
        # Chile's own dots run to Easter Island at 109°W, which would fit the country into a
        # sliver of ocean. Continental Chile only; Rapa Nui is drawn, just not framed.
        view=[-76.5, -56.0, -66.0, -17.3],
        note_public=(
            "Chile asked about religion in 2024 for the first time since 2002 — the 2017 "
            "census was abbreviated and left it out — and wrote the question with the "
            "national religious-affairs office rather than inheriting it, so it names more "
            "bodies than most censuses this size: Jehovah's Witnesses, the Latter-day "
            "Saints, the Orthodox and the Bahá'í each have their own answer. "
            "**What the 22-year gap shows is a country changing fast.** Catholics are 53.7% "
            "of adults, against 70.0% in 2002 and 76.9% in 1992; people reporting no "
            "religion are 25.7%, against 8.3% in 2002. That is not evenly spread across "
            "ages — 96% of over-65s profess a religion and 64% of 15-29s do — so much of it "
            "is one generation replacing another. "
            "**And the map of it is as much economic as regional.** Evangelical and "
            "Protestant Chile is 16.2% nationally but is concentrated hard in the coal and "
            "forestry towns of the Biobío coast, where it is the majority faith — Los Álamos "
            "62%, Curanilahue 62%, Lota 61% — and stays high through La Araucanía. The least "
            "religious places are the wealthy eastern comunas of Santiago: Providencia is "
            "43% no-religion and Ñuñoa 40%. The most Catholic are rural Maule and the "
            "islands of Chiloé, around 80%. The small named groups sit in eastern Santiago "
            "almost by definition — a third of Chile's Jews are in Las Condes, Lo Barnechea "
            "and Vitacura — except the Muslims, whose largest single community is in "
            "Iquique. "
            "**Two things to hold while reading it.** The question was put only to people "
            "**aged 15 or over**, and nothing here scales that up to children, so this map "
            "draws 81.8% of Chile and its shares are shares of adults. And 'Evangélica o "
            "protestante' is a single cell covering 2.5 million people: Chilean "
            "Protestantism is overwhelmingly Pentecostal, and the two largest Pentecostal "
            "churches are the biggest religious bodies in the country after the Catholic "
            "Church, but no table separates them, so they are drawn as one colour."),
        how="census, 2024, ages 15 and over",
        grain="comunas, 44,000 people on average",
        counts=_cl_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "cl" / "cl_comunas.gpkg",
        place_unit=lambda g: g["comuna"].astype(str),
        note="346 comunas is INE's ceiling for religion and the table is exact — no "
             "suppression, no blanks, every comuna's categories summing to its own 15+ "
             "total. The 15+ universe is NOT scaled up to the whole population; "
             "sources/cl.py argues why, and the short version is that Chile's religiosity "
             "gradient by age is 32 points wide, so a flat scale-up would be measurably "
             "wrong rather than merely uncertain. Antártica (60 people) has no polygon and "
             "is dropped. COD mislabels Pozo Almonte as 'Tocopilla', so the names here come "
             "from INE (sources/cl_geo.py).",
    ),
    "ph": dict(
        name="Philippines",
        name_in="the Philippines",
        source="2020 Census of Population and Housing (Philippine Statistics Authority)",
        basis="self-identification, household population",
        view=[116.5, 4.3, 127.0, 21.4],
        note_public=(
            "The Philippine census asks for a religion and offers **129 named bodies** to "
            "answer with — the longest list on this map after the US Religion Census, and "
            "the longest anywhere that people answered for themselves. It is also the "
            "most lopsided. Of the 129, one hundred and twenty-six are Christian, and the "
            "entire non-Abrahamic world gets three: Islam, Buddhist, and Tribal religion. "
            "**There is no Hindu box, no Jewish box, no Sikh box and no Chinese folk "
            "religion box anywhere on the form**, so everyone in those traditions is "
            "inside 'other religious affiliations' with no way out. The question is four "
            "levels deep on Philippine evangelicalism and zero levels deep on everything "
            "else, which is a fact about what the country argues about rather than about "
            "who lives in it. "
            "**Catholicism is 78.9% and the interesting thing is the shape of where it "
            "is not.** Three edges do almost all the work. Muslim Mindanao and Sulu are "
            "not a gradient but a wall — Sulu is 95% Muslim and 0.1% Catholic, Tawi-Tawi "
            "97%, Lanao del Sur 95% — and the boundary falls between provinces rather "
            "than running through them. The **Cordillera** is the Protestant region, and "
            "it is the sharpest thing on the northern half of the map: Mountain Province "
            "is 49% Protestant against 42% Catholic, and a quarter of the whole province "
            "is Episcopalian, which is the Anglican mission at Sagada still visible a "
            "century later; Ifugao, Benguet, Kalinga and Apayao run 31-39%. And **Ilocos "
            "Norte is Aglipayan** — 21% of the province belongs to the church Gregorio "
            "Aglipay founded in 1902 and was born a few miles from. "
            "**Iglesia ni Cristo is the country's third largest religious body and has no "
            "home province.** 2.8 million people, founded in Manila in 1914, and its "
            "highest share anywhere is 7.5% in Tarlac. Nearly every other body on this "
            "map has a region; INC has a country, which is unusual enough to be worth "
            "looking for as you pan. "
            "**Two things the census does that the map inherits.** It offered *Aglipay* "
            "and *Iglesia Filipina Independiente* as separate answers, and they are the "
            "same church: 818,916 people chose one name and 640,076 the other, and in "
            "Ilocos Norte both are used side by side. They are added back together here, "
            "which makes the Aglipayan church the fourth largest body in the country at "
            "1.46 million. And **43,931 people, four hundredths of one percent, reported "
            "no religion** — a number that measures the question rather than the country. "
            "One household member answered for everyone, and 'none' is a hard answer to "
            "give on a relative's behalf where belonging is assumed."),
        how="census, 2020",
        grain="provinces and cities, 930,000 people on average",
        counts=_ph_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ph" / "ph_barangays.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ph_place_weight,
        note="THE COUNTS ARE COARSE AND THE PLACEMENT IS FINE, and the two must not be "
             "confused. PSA publishes religion at province + highly urbanised city and "
             "nowhere below it — 117 units for 108.7M people, about 929,000 each, which is "
             "the coarsest counting geography on this map. The dots are then spread across "
             "42,042 barangays weighted by barangay population (sources/ph_geo.py), so "
             "they land where Filipinos live rather than evenly across a province, but "
             "nothing measures which barangay a given church's members are in. Read a "
             "cluster as 'this province, drawn where its people are', never as a "
             "neighbourhood. "
             "The 33 HUCs are cut out of their provinces by the census and by the "
             "polygons alike, and the BARMM Interim Province — 63 barangays with no "
             "polygon in any boundary set — is reconstructed from the US Census Bureau's "
             "own tagging of them. The universe is the household population, 99.66% of the "
             "country; the missing 0.34% is the institutional population, which is where "
             "the seminaries and convents are (spec §3.7).",
    ),
    "gh": dict(
        name="Ghana",
        source="2021 Population and Housing Census (Ghana Statistical Service)",
        basis="self-identification",
        view=[-3.35, 4.5, 1.3, 11.25],
        note_public=(
            "The first African country on this map, and the shape of its question is the "
            "first thing to know about it. Ghana's census offers **four Christian boxes** "
            "— Catholic, Protestant, Pentecostal/Charismatic, Other Christian — and one "
            "box each for Islam, Traditionalist, Other and No Religion. So 71% of the "
            "country is resolved four ways and everyone else is resolved once, which is a "
            "fact about what Ghana counts rather than about who lives there. Nothing here "
            "separates Sunni from the large and old Ahmadi community, and nothing names a "
            "single one of the African Independent Churches, which are most of 'Other "
            "Christian'. "
            "**Pentecostal and Charismatic Christianity is the largest answer in the "
            "country at 31.6%**, bigger than any single Christian category anywhere else "
            "on this map outside the United States, and it is a 20th-century arrival "
            "rather than a mission inheritance: the Church of Pentecost and the Accra "
            "megachurches together outnumber Catholics three to one. It peaks in Greater "
            "Accra at 47.3% and in the Ada districts at nearly 60%. "
            "**The north–south divide is the strongest thing on the map and it is not a "
            "gradient.** Islam is 66.5% of the Northern region and 4.7% of Volta; four "
            "districts around Tamale — Nanton, Kumbungu, Tolon, Savelugu — are between 95% "
            "and 99% Muslim, which is as near-total as any unit on this map outside Sulu. "
            "**And there is a Catholic island inside the Muslim north.** Upper West is "
            "33.8% Catholic against a national 10%, and Nandom is **88.6%** — the highest "
            "single-body share of any district in Ghana. That is one mission field, the "
            "White Fathers at Navrongo and Jirapa from 1906, still legible as a hard edge "
            "a century later; its neighbours Jirapa, Nadowli Kaleo and Lawra all run "
            "48-64%. "
            "**Traditional religion survives in a belt, not a scatter.** 3.25% nationally, "
            "but 43.6% in Tatale Sanguli, 41.0% in Nabdam and 40.1% in Nanumba South, and "
            "close to zero across the whole Akan south — the median district is 0.5%. Read "
            "the number as a floor: the form makes Traditionalist exclusive of the "
            "Christian and Muslim boxes, and in Ghana traditional practice very often "
            "accompanies one of those rather than replacing it, so anyone who would answer "
            "both is counted in the other column."),
        how="census, 2021",
        grain="districts, 113,000 people on average",
        counts=_gh_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "gh" / "gh_districts.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        note="272 units for 30.8M people — the 261 MMDAs with the six metropolitan "
             "districts replaced by their 17 sub-metros, which is GSS's own finest "
             "publication of religion and its own boundary file for it (sources/gh_geo.py; "
             "geoBoundaries has 260 units on a 2019 vintage and no sub-metros). About "
             "113,000 people per unit, so read a cluster as a district and never as a "
             "neighbourhood. Placement is uniform inside each district: nothing on the "
             "Ghanaian side of this map weights dots by where people live, so the empty "
             "half of a large northern district gets as many dots as the town in it. "
             "Ghana is the first country to need INLAND water subtracted — GSS runs its "
             "districts straight across Lake Volta, and 397 of the first build's 30,750 "
             "dots, 1.29%, were on open water. sources/gh_geo.py cuts HydroLAKES out of "
             "the placement polygons and that is now zero (spec §8.2; water.py does the "
             "sea only and names this as its known gap). The universe is the 99.74% who answered the "
             "question; GSS publishes no 'not stated' cell and the missing 78,692 are "
             "spread evenly across every age and education band, so they are missingness "
             "rather than a group, and are not scaled up (sources/gh.md §4).",
    ),
    "ru": dict(
        name="Russia",
        source="Sreda «Arena» Atlas of Religions and Nationalities 2012",
        basis="self-identification, sample survey",
        # REQUIRED here, not cosmetic. Filling Chukotka put dots east of 180°, so the
        # measured bbox is [-175.3 … 179.5] — which does not mean "Russia is wide", it
        # means "Russia wraps", and fitting it frames the entire globe. This stops at the
        # antimeridian and gives up the Chukotkan sliver beyond it: 47,000 people who are
        # still drawn and still reachable by panning. The US `view` exists for the same
        # class of reason (Alaska and Hawaii); this is the antimeridian version.
        view=[19.0, 41.0, 180.0, 78.0],
        note_public=(
            "**The only country here that is not drawn from a census.** Russia has not "
            "asked about religion in a census since 1937 and the 2021 census does not ask "
            "either, so this map is a survey: Sreda's Arena project, 56,900 respondents "
            "across 79 of the then 83 federal subjects, in the summer of 2012. That is "
            "about 720 people per region, so read the large shares and distrust the small "
            "ones — the Old Believers at 0.32% are roughly 180 respondents spread over 79 "
            "regions, and where they appear on this map is close to noise even though the "
            "national figure is real. "
            "**It is also the coarsest map here by a wide margin**, at 79 units for 142.6 "
            "million people, about 1.8 million each. Every dot inside a region is drawn "
            "from the same mixture, because a region is all the survey measures; the dots "
            "are placed on a 3km population grid so they at least sit where Russians "
            "actually live, which in Sakha means four river valleys in an area the size of "
            "India. "
            "**What makes it worth drawing anyway is the question.** Arena offered "
            "seventeen answers and split things almost nothing else does: the Russian "
            "Orthodox Church from Orthodoxy outside it and from the Old Believers, and "
            "Sunni from Shia from Muslims who decline both. It is the only source on this "
            "map that asks a Muslim which branch, and the answer turns out to be regional "
            "rather than doctrinal — **Dagestan says Sunni (48.6%) and Tatarstan and "
            "Bashkortostan say neither** (31.5% and 38.3% answered 'I profess Islam, but "
            "am neither Sunni nor Shia', against 1.4% and 0.4% Sunni). Nationally that "
            "'neither' answer is 4.7% of Russia against Sunni's 1.7%, so what looks like "
            "a map of Islamic branches is substantially a map of how the question was "
            "taken. "
            "**Two answers about belief rather than belonging are 38% of the country "
            "between them.** 24.9% say they believe in God but profess no particular "
            "religion — 44% in Karelia, 41% in Komi and Amur — and 12.9% say they do not "
            "believe in God, which peaks in the Far East and southern Siberia: Primorsky "
            "34.8%, Altai Krai 27.4%, Sakha 25.6%. Arena never offers 'no religion' as an "
            "option at all, so the people who would tick that box elsewhere are split "
            "between those two here. "
            "**The Orthodox core is the Black Earth, not Moscow.** The Russian Orthodox "
            "Church is 41.3% nationally and runs 78.4% in Tambov, 71.3% in Lipetsk and "
            "69.3% in Nizhny Novgorod, against 52.8% in Moscow and 26.6% in Primorsky. "
            "Two regions are outright majority something else — **Tuva 61.8% Buddhist** and "
            "**Dagestan 82.6% Muslim** across its three Islamic answers — and in Kalmykia "
            "the largest single answer is Buddhism at 37.6%, which is true of nowhere else "
            "in Europe. North Ossetia is the strangest column in the table: 49.2% Russian "
            "Orthodox and 29.4% practising the traditional Ossetian religion at once. "
            "**Four regions were never surveyed and are estimated, not measured.** Arena "
            "covers 79 of 83: it has no Chechnya, no Ingushetia, no Nenets and no "
            "Chukotka, and the first two are the most Muslim republics in Russia. Those "
            "four — 2.1 million people, 1.5% of the country — are filled in from the 2021 "
            "census's ethnic composition, using the relationship between ethnicity and "
            "Arena's own answers measured across the other 79 regions. That relationship "
            "is strong but it is not one-to-one: **only about seven in ten ethnically "
            "Muslim Russians actually give an Islam answer**, the rest saying they believe "
            "in God without a religion, or not at all. So Chechnya is drawn at 85% Muslim "
            "rather than the 98% its ethnic make-up alone would suggest. Treat those four "
            "regions as an informed estimate and the other 79 as a survey. A further 5.4% "
            "of the country answered 'difficult to say', which is itself regional — 18% in "
            "Magadan and 16% in Sakhalin against 0.4% in North Ossetia. "
            "**And it is fourteen years old**, in a period when this is one of the things "
            "about Russia most likely to have moved. No successor survey exists."),
        how="survey, 56,900 people, 2012; no census asks",
        grain="federal subjects, 1.7m people on average",
        counts=_ru_counts,
        # Counts are on the federal subject; the Kontur hexes carry no subject code, so
        # `units` + `unit_key` puts scatter.py on the spatial-join path and sources/ru_geo.py
        # has already assigned and clipped every hex. `place_unit` reads the column it wrote.
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ru" / "ru_grid_3km.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ru_place_weight,
        note="Arena publishes SHARES, not people; sources/ru.py multiplies them by the 2021 "
             "census population per subject (spec §3.4, Brazil's rule). Every row is tier "
             "`modelled` — a 720-per-region survey is not a count of anybody. Placement is "
             "Kontur's 3km H3 population grid, 107,240 populated hexes covering 3.9M km² of "
             "a 16.4M km² country, which is the single most useful thing the layer does "
             "here; its own totals reproduce the census to 0.984x nationally and stay "
             "inside a factor of two in all 79 subjects (sources/ru_geo.py). Boundaries are "
             "geoBoundaries ADM1, which carries 83 subjects and no Crimea or Sevastopol — "
             "the same composition Arena surveyed, so the two agree about what Russia is "
             "without any editing.",
    ),
    "id": dict(
        name="Indonesia",
        source="Sensus Penduduk 2010 (Badan Pusat Statistik)",
        basis="self-identification, one of the six recognised religions",
        view=[94.9, -11.1, 141.1, 6.1],
        note_public=(
            "**The largest Muslim population on earth, and the whole of the interesting "
            "map is in the other 12.8%.** Indonesia is 87.2% Muslim and the Muslim regions "
            "are as near-total as anything drawn here — Aceh Timur is 99.91%, Aceh Utara "
            "99.90%, Lombok Timur 99.88% across 1.1 million people. What makes the country "
            "worth drawing is that its minorities are not scattered: almost every one of "
            "them has a homeland, and at sub-district resolution you can see the edges. "
            "**Four of them, each on its own island group.** The Papuan highlands are "
            "Protestant to a degree almost nothing else on this map matches — Nduga is "
            "100.0% and Lanny Jaya 99.84% — and the Christian belt runs east through "
            "Maluku. **Flores and Timor are Catholic**: Manggarai 94.7%, Ngada 91.4%, "
            "Sikka 88.0%, a Portuguese inheritance that stops at the Flores Sea. **Bali is "
            "Hindu** at 98.6% in Bangli and 95.2% in Gianyar — but only 76.4% in Badung, "
            "which is Kuta and the airport, and that gap is the clearest thing on this map "
            "that migration does to a religious geography. **And the Buddhists are Chinese "
            "Indonesian and coastal**: Singkawang in West Kalimantan is 29.7%, Tanjung "
            "Pinang 12.9%, Pontianak 12.0%, Medan 8.8%. "
            "**Confucianism is 116,916 people and it is one archipelago.** Bangka is "
            "5.67% Khong Hu Chu against a national 0.049% — a hundredfold — and the whole "
            "province of Bangka-Belitung is 3.25%. The number is also a political artefact: "
            "recognition was withdrawn under the New Order in 1979 and restored in 2000, so "
            "**2010 is the first Indonesian census that counts Confucians at all**, and the "
            "community is generally reckoned far larger than this. "
            "**The most interesting cell is the one labelled 'other', and it is a floor.** "
            "`Lainnya` is 0.13% nationally and lands exactly where Indonesia's indigenous "
            "religions are: Katingan 19.9%, Gunung Mas 16.8% and Murung Raya 15.2% in "
            "Central Kalimantan, which is **Kaharingan**, the Dayak religion; and Sumba "
            "Barat 19.4%, Sumba Timur 13.1%, Sabu Raijua 13.7%, which is **Marapu**. Both "
            "are far bigger than those numbers. In 2010 the belief systems collectively "
            "called Aliran Kepercayaan had no standing on the census form — registration "
            "came only with a 2017 Constitutional Court ruling — so adherents recorded one "
            "of the six recognised religions instead, and Kaharingan was administratively "
            "counted as Hinduism outright. That is most of why interior Kalimantan draws "
            "Hindu at all. "
            "**And there is no box for having no religion.** The census asks which of six "
            "religions you belong to, so Indonesia draws with an entirely empty irreligious "
            "population — not measured at zero, never offered. Every share on this map "
            "should be read as an answer to that question and not as a statement of "
            "belief."),
        how="census, 2010, six permitted answers",
        grain="sub-districts, 46,000 people each (regencies for 89 of 492)",
        counts=_id_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "id" / "id_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_id_place_weight,
        note="THE DRAWN TIER IS DECIDED PER UNIT, WHICH NO OTHER COUNTRY HERE DOES. BPS "
             "publishes religion at kecamatan (sub-district) as well as at kabupaten/kota, "
             "but 88 of the 492 regencies have an incomplete sub-district listing — no gap, "
             "no marker, no error, and the worst of them carries 82,726 of its true 255,712 "
             "people. So a regency's kecamatan REPLACE it only where they sum to it exactly "
             "in every category (403 of 492), and the regency is drawn where they do not. "
             "The result is 5,122 kecamatan plus 89 regencies = 5,211 units, disjoint, and "
             "every one of them measured rather than allocated. Ghana's two-geo_level shape "
             "(sources.md §9n), decided by arithmetic instead of by rule. "
             "AND THE COMPLETENESS TEST IS PER CATEGORY. Nduga in Papua publishes eight "
             "kecamatan carrying a population total and no religion categories at all, so "
             "the totals reconcile perfectly while the religion does not; a total-only test "
             "would have drawn 79,053 Kristen as 79,053 people with no religion. "
             "COVERAGE IS 99.78% OF SP2010. The missing 0.22% is the five regencies that "
             "became Kalimantan Utara in 2012: BPS serves a 2010 census through a "
             "post-2012 geography, and those units are under neither province. "
             "PLACEMENT IS UNIFORM WITHIN EACH DRAWN UNIT and no finer layer is used, which "
             "is honest at kecamatan scale and coarse for the 89 regencies — several of "
             "them are large and thinly populated Papuan and Kalimantan units where a "
             "population grid would place the dots much better. That is the obvious next "
             "improvement. "
             "ON §14: this is exactly the resolution BPS itself publishes, so the map makes "
             "no claim finer than the state's own. Note that the shallow category list is a "
             "fact about Indonesian law rather than about Indonesian religion — six "
             "recognised religions, one cell for everything else — and sources/id.md §5 "
             "sets out what that hides.",
    ),
    "ke": dict(
        name="Kenya",
        source="2019 Kenya Population and Housing Census, Volume IV (KNBS)",
        basis="self-identification, conventional household population",
        view=[33.8, -4.8, 42.1, 5.6],
        note_public=(
            "**The deepest religion question in Africa, on the coarsest geography this map "
            "draws.** KNBS offers thirteen answers where Ghana offers nine and most of the "
            "continent offers none, and two of them — *Evangelical Churches* and *African "
            "Instituted Churches* — are counted by no other census anywhere on this map. "
            "It also gives Hindus, the Orthodox and traditional religion cells of their own "
            "instead of burying them in a residual. The price is 47 counties for 47.2 "
            "million people, about a million each. "
            "**The Kenyan split of Christianity is not the usual one, and reading it as the "
            "usual one will mislead you.** *Protestant* here means the mainline mission "
            "inheritance — the Anglican Church of Kenya, the Presbyterian Church of East "
            "Africa, the Methodists — while *Evangelical Churches* is a peer category, not "
            "a subset, holding the Africa Inland Church, the Baptists and the Pentecostal "
            "Assemblies of God. An Anglican here is a Protestant; a Baptist here is an "
            "Evangelical. Together they are 53.9% of the country. "
            "**African Instituted Churches are 3.29 million people and they have a "
            "homeland.** These are the churches founded in Africa by Africans outside the "
            "missions — in Kenya the Legio Maria, the Nomiya Luo Church, the African Israel "
            "Nineveh Church, the Akorino — and they are overwhelmingly a Luo and western "
            "Kenyan phenomenon: Siaya is 23.9%, Kisumu 18.2%, Homa Bay 17.8%, against a "
            "national 7.0%. Almost nowhere else on earth counts these churches at all, so "
            "this is one of the few places the map can show them. "
            "**The Muslim north-east is as near-total as anything drawn here.** Mandera is "
            "99.4% Muslim, Wajir 99.0%, Garissa 97.6% — the sharpest such block outside "
            "Sulu — and the coast runs high behind it. Against that, Kenya's Hindus are "
            "60,287 people and effectively two cities: Nairobi holds 38,141 of them. "
            "**Two smaller things worth finding.** Traditional religion survives among the "
            "northern pastoralists and almost nowhere else — Marsabit 15.5%, Samburu 9.9% "
            "against a national 0.68% — and **Kilifi is 10.2% no-religion**, two and a half "
            "times the next county and a genuine outlier on the Mijikenda coast."),
        how="census, 2019",
        grain="counties, 1.0m people on average",
        counts=_ke_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ke" / "ke_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ke_place_weight,
        note="THE COUNTS ARE COARSE AND THE PLACEMENT IS FINE, and the two must not be "
             "confused. KNBS publishes religion at county and nowhere below it — 47 units "
             "for 47.2M people, the coarsest counting geography on this map. That is the "
             "office's ceiling and not a choice made here: every other table in Volume IV "
             "is published 'by County and Sub-County' and religion is the one that stops. "
             "The census collects to the enumeration area, so the finer data exists and is "
             "not released; IPUMS's 2019 sample reaches division and is the upgrade path "
             "(sources.md §10a). "
             "The dots are spread across 230,139 Kontur 400m hexagons weighted by hex "
             "population (sources/ke_grid.py) — the first country on this map to use "
             "Kontur, and Kenya needs it: Turkana is 68,680 km² and Marsabit 70,961 km², "
             "so an equal share per polygon would have washed the empty north in dots of "
             "one colour. Read a cluster as 'this county, drawn where Kenyans live'. "
             "The universe is the conventional household population: 47,213,282 against a "
             "census 47,564,296, the difference being people in hotels, hospitals, prisons "
             "and children's homes, travellers and outdoor sleepers, who were never asked "
             "(the table's own footnote). `Don't Know` (73,253) and `Not Stated` (6,909) "
             "are non-answers and off the tree per §3.5, so 99.83% of the table is drawn.",
    ),
    "mu": dict(
        name="Mauritius",
        source="2022 Housing and Population Census, Volume II Table D6 (Statistics Mauritius)",
        basis="self-identification, religion as reported by the respondent",
        # THE VIEW IS THE MAIN ISLAND ONLY, AND RODRIGUES IS DELIBERATELY OUT OF IT.
        # Rodrigues is 600 km east and the main island is 45 km across, so a box holding both
        # is 13:1 mostly-ocean: it shrinks Mauritius to a speck in the corner and puts
        # Rodrigues behind the legend panel. Rodrigues is drawn and is one pan away, and
        # note_public says what is on it so nobody has to find it by accident.
        view=[57.25, -20.55, 57.85, -19.95],
        note_public=(
            "**Mauritius is the only place on this map where Hinduism is counted as more "
            "than one thing.** Statistics Mauritius asks about religion and takes five "
            "different Hindu answers — Marathi, Tamil, Telugu, Vedic/Arya Samaj, and "
            "everyone else — then publishes them at village level in a country that is "
            "47.9% Hindu. India's census does not do this. Guyana's, which is a quarter "
            "Hindu, does not. Four nodes on the religion tree exist because of this one "
            "table. "
            "**They are not the same kind of category, and the map is worth reading twice "
            "for it.** Marathi, Tamil and Telugu Hindus descend from indentured labourers "
            "out of three different parts of India and have kept separate temples, "
            "priesthoods and festival calendars for a century and a half — these are "
            "communities. **Arya Samaj is a movement**: Dayananda Saraswati's 1875 reform, "
            "Vedas alone and no image worship, which arrived in 1910 and split Mauritian "
            "Hinduism hard enough to shape its politics for decades. Anyone can join it and "
            "7,422 people have. "
            "**Each community has its own map and none of them is the one you would guess.** "
            "Marathi Hindus are the southwest coast and almost nowhere else — La Gaulette "
            "27.7%, Baie du Cap 27.0%, against 1.5% nationally. Tamil Hindus are southern "
            "and central, strongest in Savanne at 8.2%, and **thinner in Port Louis (3.7%) "
            "than in the country as a whole.** The Bhojpuri-descended majority is the cane "
            "belt: Camp Thorel is 95.4%. "
            "**Port Louis Ward 5 is 96.8% Muslim** — 17,058 people, and one of the most "
            "nearly total single-religion units drawn anywhere here. The city as a whole is "
            "40.9% against 18.2% nationally. "
            "**And Rodrigues is a different country — pan 600 km east to see it.** The six "
            "regions of that island run 84.9% to 91.9% Roman Catholic and 0.5% Hindu, "
            "against 24.9% and 38.5% on the main island: a Creole Catholic population inside "
            "a Hindu-majority republic, and the sharpest internal contrast any country on "
            "this map holds. It is outside the opening view because a box holding both "
            "islands is thirteen parts ocean and shows neither."),
        how="census, 2022",
        grain="wards and village councils, 6,800 people on average",
        counts=_mu_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "mu" / "mu_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_mu_place_weight,
        note="THE WHOLE RESIDENT POPULATION IS DRAWN — 1,233,097, every person the 2022 "
             "census enumerated. Only the universe row resolves to nothing. "
             "THE GEOGRAPHY IS THE FINEST PER HEAD ON THIS MAP AFTER SRI LANKA AND THE "
             "GERMAN GRID: 182 units over 1.23M people, about 6,800 each, on Municipal "
             "Council Wards and Village Council Areas. "
             "THE CATEGORY LIST IS THE PRICE. Table D5 of the same report names sixty-odd "
             "individual bodies — La Voix de la Delivrance, Peniel Tabernacle, Full Gospel "
             "Church, Christian Tamil, Church of England, Presbyterian, Methodist — but "
             "only at ISLAND level, three units. D6 has the geography and pools them into "
             "thirteen groups, of which `Other Christian` is 6.2%. §3.9's trade, made by "
             "the office, inside one publication. "
             "NO HUMANITARIAN SOURCE HAS THESE BOUNDARIES AND OPENSTREETMAP DOES. COD-AB "
             "Mauritius stops at 12 districts and geoBoundaries at ADM1; the drawn tier "
             "exists only in OSM, as 164 relations at admin_level=8 and 35 at 9. Every "
             "pairing was then checked SPATIALLY against the district the census printed it "
             "under — 182 of 182 — because D6 has no code column and a name join on 183 "
             "French place names is where a confident wrong pairing would live. It caught "
             "one: OSM and the census both split Rivière du Poste into East and West and "
             "they are not the same split, so that VCA is rebuilt and re-cut on the "
             "district line. "
             "TWO UNITS SHARE ONE POLYGON. OSM has no boundary for Vacoas-Phoenix Ward 5 or "
             "Ward 6-West, so both are drawn on the remainder of the town after its five "
             "mapped wards are removed — 35,664 people, 2.89%, and one internal boundary "
             "lost inside one town. "
             "`Other & Not stated` (6,931, 0.56%) IS DRAWN AND IS NOT A CLEAN CATEGORY. "
             "Mauritius is the only source here that pools a non-answer into a residual and "
             "publishes no split, so §3.5's usual move — take the non-answer off the tree — "
             "is unavailable. Read it as a ceiling on Mauritius's other religions rather "
             "than a count of them. "
             "AND `Buddhist/Chinese` (5,053) IS ONE CELL FOR TWO THINGS the tree keeps "
             "apart. D5 splits it nationally into Buddhist 2,178, Chinese 2,434 and Other "
             "Chinese 441; D6 does not, so it is drawn on `chinesefolk` as a syncretic "
             "whole per §3.3, and taxonomy/mu2022.py records what that costs.",
    ),
    "mw": dict(
        name="Malawi",
        source="2018 Malawi Population and Housing Census, Table E5 (NSO)",
        basis="self-identification, denomination question, whole census population",
        view=[32.5, -17.3, 36.2, -9.2],
        note_public=(
            "**Malawi asks which denomination you belong to, not which religion**, and it "
            "is the only source on this map that names a single Presbyterian body. Eight of "
            "the ten answers are Christian groupings; the other two are Islam and no "
            "religion. That buys detail no other African census here offers — Catholic, "
            "CCAP and Anglican each counted apart — and it costs the rest, because Buddhism, "
            "Hinduism, Judaism and the Bahá'ís have no cell at district and sit in one "
            "residual. "
            "**The mission map of the 1880s is still legible.** The Church of Central "
            "Africa Presbyterian is the 1924 union of three Scottish and Dutch Reformed "
            "missions and you can still see all three: Livingstonia in the north (Mzuzu "
            "City 28.0%, Mzimba 23.9%, Rumphi 22.7%) and Nkhoma in the centre (Lilongwe "
            "City 23.2%, Dowa 22.4%), against 8.8% across the whole south, where Blantyre "
            "synod is smallest and shares the ground. "
            "**Likoma is 74.6% Anglican** — the sharpest single-denomination figure of any "
            "district in Malawi, on an island of 18 km² in the middle of the lake where the "
            "Universities' Mission put its cathedral in 1903. Ntchisi (21.5%) and "
            "Nkhotakota (15.3%) are the lakeshore stations behind it, and then it stops: "
            "those three units hold a third of the country's Anglicans on 4% of its people. "
            "**The Muslim south is Yao country and it is one block, not a scatter.** "
            "Mangochi is 72.7% and Machinga 67.0%, with Balaka and Salima behind them, "
            "against 13.8% nationally and 0.08% in Chitipa at the Tanzanian border — a "
            "900-fold range. These are the communities converted along the 19th-century "
            "trade routes from Kilwa, and they sit on the southern lakeshore in one piece. "
            "**Traditional religion is Dedza.** 6.13% there against 1.06% nationally: "
            "one district holds more than a quarter of every traditionalist NSO counted. "
            "Read the national figure as a floor — the box is exclusive of the Christian "
            "and Muslim ones, and Chewa Nyau practice commonly accompanies church "
            "membership rather than replacing it. "
            "**And the largest cell in the country is a residual.** *Other Christian "
            "Denominations* is 26.6%, more than the Catholics, and it runs from 48.3% in "
            "Nkhata Bay to 5.5% on Likoma. Most of it is Malawi's very large independent "
            "and Zion church sector, which the census does not name."),
        how="census, 2018",
        grain="districts, 549,000 people on average",
        counts=_mw_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "mw" / "mw_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_mw_place_weight,
        note="THE WHOLE CENSUS IS DRAWN AND THERE IS NO GAP OF ANY KIND, which is rare "
             "here. NSO publishes no `not stated` cell for religion and no residual row: "
             "the ten denominations sum to 17,563,749 exactly on all 36 printed rows, and "
             "that figure is the full 2018 census count. Every category except the universe "
             "total resolves to a node, so 100% of the table is drawn. "
             "THE COUNTS ARE DISTRICT AND THE PLACEMENT IS FINE, and the two must not be "
             "confused. Table E5 is the only religion table below the national one anywhere "
             "in the 311-page report and the 32 district reports carry none, so district is "
             "NSO's ceiling rather than a choice made here. 32 units for 17.56M people is "
             "~549,000 each — finer per head than Kenya's counties. "
             "THE FOUR CITIES ARE PEERS OF THEIR DISTRICTS, NOT PARTS OF THEM. Mzuzu, "
             "Lilongwe, Zomba and Blantyre Cities are 2,115,867 people, 12.0% of the "
             "country, printed beside Mzimba, Lilongwe, Zomba and Blantyre rather than "
             "inside them; sources/mw.py proves it by checking that each region's districts "
             "sum to the region row on all eleven columns, and COD-AB's ADM2 carries the "
             "same 32 units in the same 7/10/15 split. "
             "The dots are spread across 75,802 Kontur 400m hexagons weighted by hex "
             "population (sources/mw_grid.py), and Malawi's reason for needing that is "
             "WATER rather than emptiness: Lake Malawi is 29,600 km² and the district "
             "boundaries run into the middle of it, so an equal share per polygon would "
             "draw a fifth of the country onto open lake. A population grid has no hexes "
             "there, so §8.2c's problem does not arise instead of being patched. "
             "ONE CATEGORY IS A MERGE OF THREE TRADITIONS. `SDA/Baptist/Apostolic` is "
             "1,644,829 people, 9.4%, and the tree keeps Adventists, Baptists and the "
             "African Apostolic churches in three different places. It goes to an "
             "answer-node that names the merge rather than being split by assumption or "
             "buried in `Other Christian` — see taxonomy/mw2018.py, which flags this as the "
             "one arguable call in the file. "
             "AND `Other Denomination` IS PARTLY KNOWN AND NOT SEPARABLE: Table 3.4 splits "
             "the same national figure into Buddhism 5,506, Hinduism 3,211 and other "
             "non-Christian 983,587, and no table in the report gives any of the three a "
             "geography. Those 8,717 Buddhists and Hindus are drawn inside `other.mw`.",
    ),
    "bj": dict(
        name="Benin",
        source="RGPH-4 2013, the twelve departmental Principaux indicateurs, Tableau 8 "
               "(INStaD)",
        basis="self-identification, whole census population",
        view=[0.6, 6.1, 4.0, 12.6],
        note_public=(
            "**Benin is the only country on this map whose census counts Vodun by name.** "
            "Everywhere else in Africa a form offers one *Traditionalist* box against a "
            "column of named churches; Benin offers *Vodoun* and *other traditional* as "
            "two separate answers, and they turn out to be two different religions in two "
            "different halves of the country. Vodun was banned under the Marxist government "
            "of the 1970s and recognised outright in 1996, and 10 January is a national "
            "holiday for it — which is why the question can be asked here at all. "
            "**And the Vodun heartland is not where you would look for it.** The five "
            "highest communes are Djakotomey at 69%, Toviklin 66%, Lalo 56%, Aplahoué 55% "
            "and Klouékanmè 51% — the whole of the Couffo, which is Adja country in the "
            "south-west. Abomey, capital of the kingdom of Dahomey and the name in every "
            "history of the religion, is 24%; Ouidah, the other famous name, sits in a "
            "department at 12%. "
            "**The north-west is a different traditional religion entirely.** *Autres "
            "traditionnelles* is 54% in Boukoumbé, 42% in Cobly and 37% in Tanguiéta, all "
            "in the Atacora highlands, where Vodun is 6%. These are the traditions of the "
            "Bètammaribè and their neighbours, the people whose fortified *tata* houses "
            "are the region's landmark. "
            "**Read the traditional figures as floors, and one of them for a specific "
            "reason.** The boxes are exclusive of *Catholique* and *Islam*, and in Benin "
            "the same person is very commonly both. Beyond that, *no religion* is 5.8% "
            "nationally and 45% in Toucountouna, 27% in Kérou and 20% in Cobly — the same "
            "Atacora communes that lead on traditional religion, and not Cotonou, where a "
            "secularising population would show. Some of that answer is very probably "
            "practice with no church and no name on the form. It is drawn as the census "
            "published it. "
            "**The Celestial Church of Christ has a cell of its own and is 6.8% of "
            "Benin** — an African church founded in Porto-Novo in 1947, still centred "
            "exactly where it started: Sô-Ava 30%, Akpro-Missérété 26%, Bonou 24%, the "
            "Ouémé valley and the lagoons. No other source on this map counts a single "
            "African Instituted Church at that size. "
            "**The north is Muslim and the boundary is sharp.** Karimama is 95%, Malanville "
            "94%, Ségbana 92% — the Niger valley — against 0.3% in Djakotomey, a "
            "three-hundred-fold range across 77 communes."),
        how="census, 2013",
        grain="communes, 130,000 people on average",
        counts=_bj_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "bj" / "bj_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_bj_place_weight,
        note="THE SOURCE PRINTS SHARES AND THE COUNTS ARE ARITHMETIC, NOT AN ESTIMATE. "
             "Tableau 8 gives ten religion percentages per commune to one decimal and "
             "Tableau 2 of the same booklet gives the commune's population, so every count "
             "here is a published share times a published total — no join to a second "
             "document, nothing carried from a coarser level, nothing fitted. The rows are "
             "`measured` and the caveat is precision: one decimal is ±0.05% of a unit, or "
             "±34 people in a 68,000-person commune. "
             "sources.md §11p recorded Benin as needing commune totals from the Résultats "
             "définitifs; it does not, and each booklet is self-contained. "
             "THE TEN SHARES SUM TO 98.81% AND THE REMAINDER IS NON-RESPONSE. 120,826 "
             "people, 1.21%, computed as the complement and not drawn (§3.5). RGPH-4's "
             "religion tabulations carry exactly ten categories wherever they appear, and "
             "ten values rounded to 0.1pp have a standard error of 0.09pp against an "
             "observed 0.6-2.0pp per commune, so the gap is a category rather than the "
             "rounding. The drawn population is 9,887,923. "
             "COTONOU IS ONE POLYGON AND THE CENSUS OFFERED THIRTEEN. The Littoral booklet "
             "publishes religion for Cotonou's 13 arrondissements — 679,012 people, 6.8% "
             "of Benin — and they are parsed, checked against Cotonou's own row, and NOT "
             "drawn: COD ships no ADM3 for Benin, and geoBoundaries' OpenStreetMap "
             "arrondissements could not be verified against the census's own arrondissement "
             "populations (ratio band 0.64-1.98, r=0.81). §3.10's rule taken the "
             "conservative way; sources/bj.md §5 has the measurement. "
             "The dots are spread across 75,951 Kontur 400m hexagons weighted by hex "
             "population (sources/bj_grid.py). Benin needs that for empty northern "
             "communes AND for water: the lagoons are inside the communes rather than cut "
             "out of them, and Sô-Ava contains Ganvié, a town of ~30,000 built on stilts "
             "over Lac Nokoué. The grid finds Ganvié because it has buildings, and finds "
             "nothing on the open lake — which is also why Sô-Ava and Aguégués are the two "
             "communes Kontur models worst (0.32× and 0.30×). "
             "THE FIRST NAMED AFRICAN TRADITION ON THIS MAP. `indigenous.african.vodun` was "
             "added for Benin; `Autres traditionnelles` stays on the parent node because no "
             "source names the Atacora traditions individually yet.",
    ),
    "ni": dict(
        name="Nicaragua",
        source="VIII Censo de Población y IV de Vivienda 2005, variable P13 (INIDE), "
               "tabulated at municipio from INIDE's own Redatam server",
        basis="self-identification, population aged 5 and over",
        view=[-87.7, 10.7, -82.7, 15.1],
        note_public=(
            "**Nicaragua is the only census on this map that names the Moravian Church**, "
            "and it names it while naming no Anglicans, no Baptists, no Adventists and no "
            "Latter-day Saints — which is a statement about standing rather than size. On "
            "the Caribbean coast the Moravians are the historic church: they arrived in "
            "1849, ran the schools and clinics of the Miskito and Creole coast under the "
            "British protectorate and after it, and their congregations are the institution "
            "the two autonomous regions are organised around. "
            "**So the map has a coastline on it.** 73,902 Moravians are 1.63% of Nicaragua "
            "and **53.3% of Prinzapolka, 50.9% of Puerto Cabezas and 43.6% of Waspám** — "
            "the first place on this map where the Moravians are anybody's plurality. Nine "
            "municipalities are over 10%, they are all on the Caribbean, and between them "
            "they hold 94% of every Moravian in the country on 5% of its people. Forty-one "
            "municipalities have none at all. "
            "**The other coastal category is the one the census did not name.** `Otra` runs "
            "at 1.63% nationally and **44.1% on Corn Island**, 21.4% in Laguna de Perlas "
            "and 17.5% in Bluefields, against 0.02% in the far north-west — a spread of "
            "more than two thousand to one, and every unit above 8% is on the Caribbean. "
            "The Anglican church of the Mosquito Coast and the Jamaican Baptist mission "
            "that worked the same Creole towns from the 1840s are the obvious contents and "
            "neither has a box on this form. The map does not split the cell. "
            "**No religion is 15.7% and it is not an urban figure.** It peaks in the "
            "northern mountains and on the agricultural frontier — Santa María 39.8%, Murra "
            "36.1%, Wiwilí de Jinotega 32.9% — and bottoms out on the Moravian coast, where "
            "Waspám is 0.41% and Puerto Cabezas 1.22%. That is the opposite of the usual "
            "shape, and the Pacific cities sit in the middle of the range rather than at "
            "the top. "
            "**The Catholic heartland is the cattle country of the centre and north.** San "
            "Francisco de Cuapa is 93.5% — the highest in Nicaragua, and Cuapa is the "
            "village of the 1980 Marian apparitions and a national pilgrimage site — with "
            "Camoapa at 91.7% and a block of Nueva Segovia and Madriz municipalities behind "
            "them. The evangelical map is almost its negative: Waslala 37.3%, Murra 36.5%, "
            "Paiwas 33.8%, all of it the interior frontier rather than the cities."),
        how="census, 2005, ages 5 and over",
        grain="municipios, 30,000 people on average",
        gap="under-fives, 11.8% of the country, who were not asked the religion question",
        counts=_ni_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ni" / "ni_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ni_place_weight,
        note="THE SOURCE IS A QUERY, NOT A PUBLICATION, AND THAT IS WHAT MAKES THE COUNTRY "
             "DRAWABLE AT THIS RESOLUTION. INIDE runs an open, unauthenticated Redatam "
             "webserver over the 2005 census microdata (redatam.inide.gob.ni). What INIDE "
             "PRINTS is religion by department — 17 units, Volume I CUADRO 12 — and Volume "
             "IV's 546 pages of municipal tables contain no religion table at all. What it "
             "SERVES is the same variable at 153 municipios and at 2,579 comarcas. That is "
             "a 9x resolution gap between an office's printed and served output on the same "
             "website, and it is why sources.md §11x adds the rule: ask whether the office "
             "runs a Redatam instance before reading its PDFs. "
             "THE COMARCA TABLE IS REAL, FREE, AND UNPLOTTABLE. 2,579 units at 1,759 people "
             "each would be among the finest geographies on this map. No comarca boundaries "
             "are published: OCHA's cod-ab-nic says in its own words that it is "
             "'structured into 2 levels' and geoBoundaries 404s on NIC ADM3. For once the "
             "BOUNDARIES are the ceiling and not the counts. "
             "THE UNIVERSE IS AGE 5 AND OVER — 4,537,200 of a 5,142,098 census population. "
             "The 604,898 under-fives were never asked, which is a different thing from "
             "being missed, so they are in `gap=` and not drawn as a §3.5 undercount. "
             "Within the universe the eight categories are an exact partition: they sum to "
             "the municipio total on all 153 rows, there is no `no especificado` cell, and "
             "100% of the table is drawn. "
             "THE READ IS CHECKED AGAINST THE PRINTED VOLUME, WHICH SHARES NO CODE PATH "
             "WITH IT. All nine of CUADRO 12's national figures, typeset in 2006, are "
             "reproduced exactly by a 2026 query against the microdata. Nothing else here "
             "would catch a column landing in the wrong place, because every internal "
             "identity reconciles whichever order the columns are read in. "
             "THE JOIN IS ON NAME AND THE CODE JOIN IS A TRAP. COD's adm2_pcode is 'NI' + a "
             "code in INIDE's own four-digit format and 145 of 153 match, which is exactly "
             "§12's shape-2 failure: ten municipalities were renumbered between 2005 and "
             "COD's 2023 vintage, and five of them collide rather than going missing. INIDE "
             "9105 is Waspám; COD NI9105 is Mulukukú — so a code join would move a 43.6% "
             "Moravian border municipality inland and every total would still reconcile. "
             "sources/ni_geo.py joins on name, confirms it with the department prefix on "
             "all 153, and then checks that the most Moravian municipios really are the "
             "easternmost using COD's own centroid longitudes — a witness that uses neither "
             "name nor code. It refuses to run if the code join ever stops being wrong. "
             "The dots are spread across 47,270 Kontur 400m hexagons weighted by hex "
             "population (sources/ni_grid.py). AND THE VINTAGE GAP THERE IS EIGHTEEN YEARS, "
             "THE LARGEST ON THIS MAP: the counts are 2005 and the placement grid is 2023. "
             "It moves dots within a unit and never between units, so no count is affected, "
             "but in the eastern frontier municipios the dots land in settlements that had "
             "barely begun when the census was taken — Prinzapolka is both the most Moravian "
             "municipality and the second most re-settled. It also means the Kontur/census "
             "ratio band cannot be the discriminating check here, unlike Zimbabwe's; the "
             "correlation is (r=0.948 on 153 units, best of 0.296 over 2,000 shuffles), and "
             "sources/ni_grid.py says which one is carrying it.",
    ),
    "pe": dict(
        name="Peru",
        source="Censos Nacionales 2017: XII de Población, VII de Vivienda y III de "
               "Comunidades Indígenas, variable C5P26 (INEI), tabulated at district from "
               "INEI's own Redatam server",
        basis="self-identification, population aged 12 and over",
        view=[-81.5, -18.5, -68.5, 0.1],
        note_public=(
            "**Peru's census names eight religions, and its own published tables name "
            "four.** INEI printed Catholic, Evangelical, *other religion* and none — and "
            "that *other religion* of 1,115,872 people turns out to be five separate "
            "census answers added together: Adventists, Jehovah's Witnesses, "
            "Latter-day Saints, plain 'Christian', and a genuine remainder. The eight are "
            "all there in the microdata, at 1,874 districts, and this map draws them. "
            "**The Adventists are the reason to look.** 353,430 people, 1.5% of Peru — and "
            "not spread thinly. They are **two regions**. The first is the Aymara "
            "altiplano around Lake Titicaca, where the Adventist mission at Platería opened "
            "in 1898 and ran the schools: San Antón is **23.4%** Adventist, Crucero 22.1%, "
            "Amantaní 20.9%, Huacullani 17.5%. The second is 800 km north in the Alto Mayo "
            "colonisation frontier — Yantaló 17.8%, Omia 17.1%, San Fernando 16.8%. "
            "Between the two, 399 districts have no Adventist at all. "
            "**And Peru is the first census on this map to print a box for the Latter-day "
            "Saints.** 113,659 people, named by the state rather than counted by the church "
            "or hidden in an 'other' bucket. They are a southern coastal population — "
            "Pacocha 2.2%, Islay 2.1%, Mollendo 1.9% — and absent from 914 districts. "
            "**Evangelicals are 14.1% and they are the periphery, not the cities.** Elías "
            "Soplín Vargas in San Martín is 77.9%; Uchuraccay and Anchihuay in Ayacucho are "
            "66.9% and 65.3%; El Cenepa and Río Santiago — the Awajún and Wampís districts "
            "of Amazonas — are 65.1% and 59.7%. They are above zero in every one of the "
            "1,874 districts, which nothing else here manages. "
            "**No religion is 5.1%, and reading it as secularity would be wrong.** It peaks "
            "in indigenous Amazonia: Puerto Bermúdez 37.7%, Awajún 30.5%, Raymondi 26.9% — "
            "Asháninka and Awajún country. The census offers no box for Amazonian "
            "indigenous religion, and 'none' is where a form with no box for your religion "
            "puts you. "
            "**The smallest category has the sharpest geography.** `Otra` is 0.41% "
            "nationally and 20.7% in Yavarí, 19.2% in Tournavista, 18.9% in San Pablo — "
            "Amazon river and colonisation districts on the Brazilian and Colombian "
            "frontier, a spread of fifty to one. The Israelitas del Nuevo Pacto Universal, "
            "a Peruvian church founded in 1968 whose settlement colonies are in exactly "
            "these places, are the strongest candidate for what is inside it. The map does "
            "not split the cell."),
        how="census, 2017, ages 12 and over",
        grain="districts, 12,378 people on average",
        gap="under-twelves, 21.1% of the country, who were not asked the religion question",
        counts=_pe_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "pe" / "pe_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_pe_place_weight,
        note="THE SOURCE IS A QUERY, NOT A PUBLICATION, AND HERE THAT DOUBLES THE CATEGORY "
             "COUNT RATHER THAN THE RESOLUTION. INEI runs an open, unauthenticated Redatam "
             "webserver over the 2017 census microdata (censos2017.inei.gob.pe/bininei). "
             "sources.md §11t DECLINED Peru on the UNSD oracle's report of four categories; "
             "§11y reopened it because four is what INEI FORWARDED to UNSD, not what the "
             "census holds. The variable is Poblacio.C5P26 and it returns EIGHT named "
             "columns: Católica, Evangélica, Otra, Ninguna, Cristiano, Adventista, Testigo "
             "de Jehová, Mormones. That is the limit of §11w's 'ask the oracle' rule — the "
             "oracle ranks what was reported, and a country can be deeper than its return. "
             "THE PUBLISHED FOUR-CATEGORY TABLE IS WHERE THE OTHER FIVE WENT, AND THAT IS "
             "ALSO THE INDEPENDENT CHECK. INEI's own release prints Católica 17,635,339, "
             "Evangélica 3,264,819, otra religión 1,115,872 and Ninguna 1,180,361; the "
             "query reproduces the first, second and fourth exactly, and its `otra "
             "religión` is EXACTLY Otra + Cristiano + Adventista + Testigo de Jehová + "
             "Mormones. Every internal identity here reconciles whichever order the columns "
             "are read in — the three geographies agree to the person — so the published "
             "figures are the only thing that would catch a column landing in the wrong "
             "place. sources/pe.py asserts all four. "
             "THE UNIVERSE IS AGE 12 AND OVER — 23,196,391 of a 29,381,884 census "
             "population. The 6,185,493 under-twelves were never asked, which is a "
             "different thing from being missed, so they are in `gap=` and not drawn as a "
             "§3.5 undercount. Within the universe the eight categories are an exact "
             "partition on all 1,874 districts, there is no `no especificado` cell, and "
             "100% of the table is drawn. "
             "THE JOIN IS ON CODE, WHICH REVERSES NICARAGUA ON PURPOSE. COD's adm3_pcode is "
             "'PE' + the six-digit ubigeo the census tabulates on; 1,872 of 1,874 codes are "
             "present and 1,870 agree on the name outright, the two exceptions being "
             "spelling (Hualla/Huaya, San Pedro de Laraos/Laraos), each confirmed by "
             "reading the whole province's list out of both sources. A NAME join would be "
             "the risky one here, because Peru has many districts sharing a name across "
             "provinces and it would have to be disambiguated by the code. "
             "AND TWO DISTRICTS SHARE ONE POLYGON, WHICH IS THE WHOLE 1,874-VS-1,873 GAP. "
             "§11y read COD's ADM3 count as a vintage difference; it is not. COD carries a "
             "single 'Mazamari - Pangoa' polygon in Satipo, Junín where the census has two "
             "districts and no separate polygon for either. Both census codes go to "
             "PE120699 and are summed there: 62,229 people, 0.27% of the universe, drawn at "
             "half Peru's usual resolution and not separable on the map. Nothing is "
             "dropped. "
             "THE WITNESS THAT USES NEITHER NAME NOR CODE IS SPATIAL SMOOTHNESS, AND ITS "
             "FIRST VERSION WAS WRONG. It asserted that the most Adventist districts are "
             "the Puno altiplano, on the history of the 1898 Platería mission — and it "
             "fired, because the prior was wrong and not the join: Yantaló, Omia and San "
             "Fernando are the Alto Mayo, Peru's other Adventist region, 800 km north. So "
             "the check is now the property that made the naive version tempting, stated "
             "without naming anywhere: religion shares are spatially smooth, and a permuted "
             "join would destroy that while leaving every name, code and total intact. "
             "Catholic r=0.754, Evangelical r=0.750, Adventist r=0.690 against the eight "
             "nearest neighbours, with a best of 0.12 over 200 random re-pairings of the "
             "same shares. "
             "The dots are spread across 258,279 Kontur 400m hexagons weighted by hex "
             "population (sources/pe_grid.py), and THAT correlation is the strongest check "
             "the join gets, because a modelled 2023 grid shares no lineage with either "
             "INEI's counts or OCHA's boundaries: r=0.959 on 1,871 units against a best of "
             "0.067 over 500 shuffles. The vintage gap is six years, the SMALLEST on this "
             "map. The ratio band is not the check here and the reason is size rather than "
             "vintage: Kontur models population from building footprints, which on a "
             "district of 237 people is noise, so the spread narrows monotonically with "
             "district size (the largest 365 districts sit inside a factor of 5, the "
             "smallest 133 reach 12.6). Two districts in Bongará, Amazonas — Chisquilla and "
             "Recta, 403 people between them — are too small to contain a single hex "
             "CENTROID and fall back to their own polygon as one uniform placement cell, "
             "because scatter.py drops a unit with no placement polygon and would have lost "
             "them silently.",
    ),
    "zw": dict(
        name="Zimbabwe",
        source="2022 Population and Housing Census Report, Table 2.14 (ZIMSTAT)",
        basis="self-identification, whole census population",
        view=[24.8, -22.6, 33.4, -15.4],
        note_public=(
            "**Two of every five Zimbabweans belong to an Apostolic church, and no other "
            "source on this map counts them at all.** The Vapostori are the white-robed "
            "prophetic churches founded in the 1930s by Johane Masowe and Johane Marange — "
            "worshipping in the open air rather than in buildings, with prophecy and faith "
            "healing at the centre, and a deliberate break from both the mission churches "
            "and the ancestral religion. At 40.3% they are the largest single religious "
            "answer in the country, larger than every other Christian category put "
            "together, and they nearly double the size of this map's African Instituted "
            "Churches. ZIMSTAT gives them one cell, so the dozens of separate Apostolic "
            "bodies cannot be told apart here. "
            "**The country splits into two Christianities and the split is town against "
            "country.** The Apostolic churches are the Shona north and east — Mashonaland "
            "Central 53%, Manicaland 50% — and fall to 21% in Bulawayo and 27% in Harare. "
            "The Pentecostals are the exact inverse: 29% in Harare and 25% in Bulawayo "
            "against 11% in Mashonaland Central. Between them they are most of Zimbabwe. "
            "**Matabeleland is the different half of the country in almost every "
            "category.** It has the lowest Apostolic share, the highest *other Christian* "
            "at 13-16% — the Brethren in Christ and Adventist mission field — and the "
            "highest *no religion*, 13.5% in Matabeleland South against 4.5% in "
            "Manicaland. Note that no-religion here is NOT an urban figure: both cities are "
            "below the national rate. "
            "**African traditional religion is 5.0%, and read that as a floor.** The box is "
            "exclusive of the Christian ones, and Shona and Ndebele practice — the "
            "ancestral *midzimu*, the Mwari shrines of the Matobo hills — commonly "
            "accompanies church membership rather than replacing it. Zimbabwe is a sharper "
            "case than most, because the Apostolic churches themselves grew out of that "
            "overlap. "
            "**And the 6,845 people recorded as Jewish are probably not who you would "
            "assume.** Zimbabwe's historic Ashkenazi community has largely emigrated and "
            "numbers in the hundreds; this figure is ten times larger and peaks in rural "
            "Midlands, Masvingo and Manicaland rather than in the two cities. It most "
            "likely counts the Lemba, who claim Judaic descent and keep dietary and "
            "circumcision laws. The census does not say."),
        how="census, 2022",
        grain="provinces, 1.5 million people on average",
        counts=_zw_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "zw" / "zw_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_zw_place_weight,
        note="THE COARSEST COUNTING GEOGRAPHY ON THIS MAP — 10 provinces for 15,178,957 "
             "people, ~1.5M each, against Kenya's 47 counties at 1.0M. It is ZIMSTAT's "
             "ceiling and not a choice made here: Table 2.14 is the only religion table in "
             "the 259-page 2022 PHC report, ZIMSTAT published five thematic reports and "
             "religion got none of them, and the district and ward files carry population "
             "only. Spec §3.9b is why it is drawn anyway — there is no minimum unit count. "
             "READ IT AS COMPOSITION, NEVER AS LOCATION. A province is a sixth of a "
             "country here, so a cluster of dots says 'this province, drawn where "
             "Zimbabweans live' and nothing at all about which town. "
             "THE WHOLE CENSUS IS DRAWN AND THERE IS NO GAP OF ANY KIND. The eleven "
             "categories sum to the province total on all ten rows and to 15,178,957 "
             "nationally — no `not stated`, no residual, no §3.5 gap. The third source "
             "here of which that is true, after Malawi and Guyana. "
             "THE `Male + Female == Total` CHECK IS WHAT VERIFIES THE READ. Tables 2.14(a) "
             "and (b) are the sex breakdowns and only (c) is drawn; the other two are "
             "parsed anyway because every other identity in this table reconciles inside "
             "one table whichever way its columns were taken, and this one does not. All "
             "132 cells agree. "
             "THE NO-RELIGION CELL IS THE LITERAL STRING `None` and pandas deletes it "
             "without the `keep_default_na=False` flags — §12's Philippine trap, third "
             "sighting, and it would silently remove 1,255,578 people. `_zw_counts` asserts "
             "the category is present rather than trusting the flags. "
             "The dots are spread across 245,701 Kontur 400m hexagons weighted by hex "
             "population (sources/zw_grid.py), which matters more here than almost "
             "anywhere: Matabeleland is a third of the country, holds 1.59M people, and has "
             "the least typical religious mix in Zimbabwe, so an equal share per polygon "
             "would paint that mix across empty bush. It also removes Lake Kariba, 5,580 "
             "km² of which is inside the provinces. "
             "AND ZIMBABWE IS THE MIRROR OF BENIN ON HOW THE JOIN IS CHECKED: here the "
             "Kontur/census ratio band is tight and discriminating (0.83-1.11 across ten "
             "very uneven provinces; a shuffle fails a median 4 of 10) while the "
             "correlation is weak, because ten similar log-populations correlate by luck. "
             "Both are measured and sources/zw_grid.py says which one is carrying it.",
    ),
    "kz": dict(
        name="Kazakhstan",
        source="National Population Census 2021 (Bureau of National Statistics) — ethnicity "
               "by region × religion by ethnicity, modelled",
        basis="ethnicity-derived from the census's own religion × nationality table",
        view=[46.0, 40.5, 87.5, 55.5],
        # No `gap` line: every one of the seven answers Kazakhstan's Question 11 offers is on
        # the tree, refusal included, so 100.00% of the census is drawn. It had one until
        # 2026-09-07 (see taxonomy/kz2021.py's note on `Отказались указать`).
        note_public=(
            "**Every dot in Kazakhstan is modelled, and no dot is a count of anybody where "
            "it is drawn.** Kazakhstan asks about religion and publishes the answer for the "
            "country as a whole and **nowhere else** — not by oblast, not in 2021 and not "
            "in 2009. What is drawn here is the census's own ethnic map of the country, "
            "recoloured by the census's own national table of which religions each "
            "nationality reported. **Read it as 'this is what this region's ancestry "
            "implies', never as 'this many Muslims live here'.** "
            "**The one thing the model is certainly right about is the big pattern**, "
            "because in Kazakhstan religion really does track ancestry closely: Kazakhs are "
            "89.2% Muslim and Russians 85.5% Christian, and the north-south gradient the "
            "map shows is the Slavic settlement of the northern steppe under the Empire and "
            "the Virgin Lands campaign. North Kazakhstan is 49% Christian against "
            "Kyzylorda's 2%, and that contrast is real. "
            "**What it cannot see is anything that is about a person rather than their "
            "descent.** There is a check on this, and it is worth stating because it is "
            "unusual: the census publishes its religion-by-nationality table separately for "
            "town and country, which is the model's own assumption written down as "
            "something testable. Tested that way the model puts **1.6% of the country on "
            "the wrong side of the town/country line** — Islam and Orthodoxy come back "
            "within 3%, but **non-believers are out by 42% in rural Kazakhstan**, because "
            "not believing is an urban habit inside every ethnic group at once and ancestry "
            "cannot see it. Wherever this map draws non-belief, treat the location as the "
            "weakest thing on the page. "
            "**A ninth of the country declined to answer, and they are on the map.** 2.1 "
            "million people, 11.0%, chose *I decline to state* — **which is option 6 of the "
            "seven the census form actually offers**, printed and numbered, not a blank "
            "somebody left. So it is an answer, and it is drawn as one, in its own colour: "
            "nobody is quietly shared out among the religions. Where those dots cluster is "
            "worth looking at on its own. "
            "**Some part of that refusal is probably about the law.** Kazakhstan requires "
            "religious groups to register, refuses registration to Jehovah's Witnesses and "
            "to Ahmadi Muslims, and prosecutes unregistered worship; the Protestant house "
            "churches that bear the most of it are inside the 9,419 Protestants drawn here. "
            "Read that number, and the Muslim one, as floors. "
            "**But do not read a pattern into where they are.** Those dots come out almost "
            "evenly spread — every region between 9.4% and 12.2% — and **that flatness is "
            "the model's, not Kazakhstan's.** Refusal is placed here by ancestry, and "
            "Kazakhs (9.3%) and Russians (7.5%) decline at similar rates, so every region "
            "lands near the national 11%. The real variation is by town and country, which "
            "is exactly what ancestry cannot see: tested against the census's own "
            "urban/rural figures the model is out by −8% in towns and +15% in villages. "
            "**The national total is exact; the map of it is the weakest thing here after "
            "non-belief.** "
            "**Do not read the Russian border.** Kazakhstan looks vastly less secular than "
            "Russia 200 km away — 3.6% non-believers in North Kazakhstan against 52% of "
            "Omsk reporting no religious institution — and **almost all of that cliff is "
            "the questionnaire.** Russia's survey offers *believes in God, professes no "
            "religion*, which **24.9% of Russians choose and which Kazakhstan's census does "
            "not offer at all**. A Russian in Petropavl who believes vaguely and attends "
            "nothing has to pick something else, and 85.3% of Kazakhstan's Russians are "
            "recorded Orthodox against 43% of Russia's population. Russia's largest "
            "non-institutional answer has not vanished at the border; it is inside "
            "Kazakhstan's Orthodox count — and the 11% who declined to state, drawn here in "
            "their own colour, are the rest of it. "
            "**Kazakhstan splits Christianity three ways, which most censuses here do not** "
            "— Orthodox, Catholic and Protestant as separate boxes. 99.1% of Kazakhstani "
            "Christians are Orthodox; the Catholics are the descendants of Poles and "
            "Germans deported to the steppe in the 1930s and 40s, which is why Karaganda "
            "has a cathedral; and **82% of the country's Buddhists are Koreans**, deported "
            "from the Soviet Far East in 1937."),
        how="census, 2021, but religion asked only nationally; spread by ethnicity",
        # The `counted` row above it already says none of this was counted, so this one is
        # free to answer the question it is labelled with. It used to carry the modelling
        # warning instead, because it was then the only line under the title that could.
        grain=("17 regions, 1.1m people on average; the religion table behind them is national, "
               "not regional"),
        counts=_kz_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "kz" / "kz_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_kz_place_weight,
        note="THE FIRST COUNTRY HERE WITH NO MEASURED TIER AT ALL. Every row is `modelled` "
             "(§7, §14.10). Kazakhstan publishes religion nationally only — sources.md §11u "
             "establishes that four ways, including that not one of the 542 pages of the "
             "2021 religion volume names an oblast while the LANGUAGE chapter of the same "
             "volume has a regional cut. "
             "THE MODEL IS THE CENSUS CROSSED WITH ITSELF: ethnicity by region (workbook "
             "sheet 2.1) times share(religion | ethnicity) (volume ch.12, national). Both "
             "inputs are the same census, the same office and the same round, which makes "
             "the coefficients better attested than any other modelled country here — "
             "Greece, Spain and France all multiply a state count by a THIRD PARTY's "
             "composition. "
             "NO MAGNITUDE IS ESTIMATED (§14.4 rule 1). Every person placed is a person BNS "
             "counted in that region; the model only decides the column. Both margins are "
             "reproduced EXACTLY — every religion's national total to the person, every "
             "region's population to the person — and that is arithmetic rather than luck, "
             "because the coefficients are conditional on ethnicity and the two "
             "publications' ethnic margins agree to the person on all eighteen groups. "
             "THERE IS A REAL HELD-OUT TEST, which §14.10's fifth condition asks for and "
             "most modelled countries cannot supply. The volume publishes religion × "
             "nationality separately for urban and rural Kazakhstan; predicting that from "
             "the national coefficients misplaces 313,745 people, 1.64%. Islam +1.7%/-2.3%, "
             "Orthodoxy +0.9%/-2.4%, but `Неверующие` -13.0%/+41.8% and the refusal cell "
             "-7.8%/+15.0%. THE URBAN/RURAL COEFFICIENTS ARE DELIBERATELY NOT USED: fitting "
             "to them would improve the map and destroy the only independent evidence the "
             "country has. "
             "11.01% IS NOT DRAWN. `Отказались указать`, 2,112,653 people, is the largest "
             "cell this map declines to draw anywhere — §3.5 and tt2011.py's precedent on "
             "Trinidad's 11.10%. It is also the cell the held-out test predicts worst. "
             "17 REGIONS IS THE 2021 VINTAGE AND THE BOUNDARY FILE IS 2023. COD-AB ships 20 "
             "ADM1 polygons because Kazakhstan created Abay, Jetisu and Ulytau in 2022; "
             "kz_geo.py dissolves the three pairs back. No polygon is cut and the reform "
             "split no rayon, so the ADM2 count (218, which the census also publishes) "
             "checks it, and Kontur's per-region band checks it again. "
             "218 UNITS WERE AVAILABLE AND WERE NOT TAKEN. The census publishes ethnicity "
             "at 218 rayons as an exact partition and COD has exactly 218 ADM2 polygons. "
             "Two reasons: 218 units of inference is a much stronger claim than 17 for a "
             "country where nothing is counted, and the join would be 218 fuzzy "
             "Russian-to-English transliteration matches with no shared code, which is "
             "§12's shape 2 with 218 chances to fire.",
    ),
    "np": dict(
        name="Nepal",
        source="National Population and Housing Census 2021 (NPHC 2078), religion Table 1 "
               "(National Statistics Office)",
        basis="self-identification, whole census population",
        view=[80.0, 26.3, 88.3, 30.5],
        gap="the institutional population, 0.8%, which is published by district only",
        note_public=(
            "**Nepal asks about ten religions and three of them are drawn on no other map "
            "here.** `Kirat`, `Prakriti` and `Bon` are boxes on the census form, not "
            "write-ins recovered from a residual, and between them they are **1.09 million "
            "people**. "
            "**Kirat Mundhum is the sharpest thing on this map's Asian half.** It is the "
            "religion of the Limbu, Rai, Yakkha and Sunuwar of the eastern hills — an oral "
            "scripture, the *Mundhum*, recited by *phedangma* priests — and it is 924,204 "
            "people, 3.17% of Nepal. Koshi province is **16.8%** Kirat and Sudurpashchim, "
            "at the other end of the country, is 0.01%. Panchthar district is **55.7%**, "
            "Taplejung 44.2%, and Mahakulung in Solukhumbu reaches **87.3%**. The edge "
            "against the Hindu middle hills is abrupt rather than gradual, and it is a "
            "border that has been moving: the Kirat count has risen at every census since "
            "1991, as a revival movement asserts a distinct identity against being "
            "recorded as Hindu. "
            "**Bon is not where you would expect it.** 67,223 people, and the obvious "
            "guess is the trans-Himalayan north — Mustang and Dolpa, where the Yungdrung "
            "Bon monasteries are. The census puts most of it in **Gandaki's middle hills**: "
            "Manang 6.1%, Gorkha 5.7%, Lamjung 4.7%, and Dharche in Gorkha at 32.7%. That "
            "is **Gurung (Tamu) country**, and the likeliest reading is that most of this "
            "cell is the Tamu shamanic tradition — the *pye-ta lhu-ta*, with its *pachyu* "
            "and *klepri* priests — which Gurungs describe as Bon and which is related to, "
            "but not the same institution as, the monastic Bon of Dolpa. One box, two "
            "things, and the map cannot separate them. "
            "**`Prakriti` is the Nepali word for nature, offered as a religion box.** "
            "102,048 people, and it is one region rather than a scatter of odd answers: "
            "Rukum East 16.6%, Rolpa 8.2%, Thawang 45.0% — the Kham Magar hills of the "
            "mid-west. NSO publishes no gloss on what it covers, so what the map can say "
            "is that these people answered the question and did not answer it with any of "
            "the world religions on the form. "
            "**The Muslim Terai is continuous with India across the border**, and both "
            "sides are now drawn. Rautahat is 22.6%, Banke 18.7%, Kapilbastu 18.2% — "
            "against 0.00% in Bajhang and under 0.3% across the whole far west. "
            "**Christianity is the fastest-growing answer in Nepal** — under 0.5% in 2001, "
            "1.4% in 2011, 1.76% now — and it is not in the capital or the Terai but the "
            "central hills, among the same Tamang and Magar communities the Kirat and "
            "Prakriti boxes draw from: Dhading 7.6%, Makwanpur 6.1%, Gorkha 6.0%. **Read "
            "that number as a floor.** Nepal's 2017 penal code criminalises conversion and "
            "'hurting religious sentiment', and people have been prosecuted under it; a "
            "census answer given in that setting undercounts rather than over. "
            "**The five world religions arrive undivided.** One Hindu box for 23.7 million "
            "people — the second-largest Hindu population on earth — with no sampradaya, no "
            "caste tradition and no sect; one Buddhist box covering Tibetan Vajrayana, the "
            "Newar Vajrayana of the Kathmandu valley that exists nowhere else, and a "
            "twentieth-century Theravada revival; one Muslim box; one Christian box naming "
            "no church. Nothing here splits them, because the census does not."),
        how="census, 2021",
        grain="753 local levels (palikas), 38,000 people on average",
        counts=_np_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "np" / "np_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_np_place_weight,
        note="THE FINEST COUNTING GEOGRAPHY OF ANY LARGE ASIAN COUNTRY HERE. 753 local "
             "levels at ~38,400 people each, against India's 5,988 sub-districts at "
             "~202,000 and Bangladesh's 544 upazilas at ~306,000 — Nepal counts religion "
             "roughly five times finer than either neighbour, on a table that is one 270 KB "
             "spreadsheet. "
             "AN EXACT PARTITION, WITH NO RESIDUAL AND NO `NOT STATED`. The ten categories "
             "sum to the row total on all 918 rows of the source, and the local levels, the "
             "districts and the provinces each sum to 29,164,578. Nothing is suppressed and "
             "nothing rounded, so 100% of the census population is in a named category — "
             "the fourth source here of which that is true after Zimbabwe, Malawi and "
             "Guyana, and by far the largest. "
             "THE 0.82% NOT DRAWN IS THE INSTITUTIONAL POPULATION, and it is §3.7's case "
             "exactly. NSO gives 239,098 people — barracks, prisons, hospitals, hostels, "
             "and Nepal's monasteries and gompas — one row per district and no finer "
             "geography at all. Spreading them across a district's local levels would "
             "invent a location, and this is a population that is concentrated by its "
             "nature, so the invented spread would be actively wrong rather than merely "
             "uncertain. Dropped, and said on the map in `gap`. "
             "THE JOIN IS ON NAMES AND THE BOUNDARY FILE CARVES OUT 22 NATIONAL PARKS. "
             "COD-AB has 775 ADM3 polygons against the census's 753 local levels, and the "
             "difference is Chitawan, Parsa, Bardiya, Khaptad, Langtang, Shivapuri, "
             "Shuklaphanta, Koshi Tappu and Dhorpatan, several of them split across "
             "districts. The p-code's unit-type digit separates them and reproduces Nepal's "
             "official 6 metropolitan / 11 sub-metropolitan / 276 municipality / 460 rural "
             "municipality composition exactly, which is the evidence that it means what it "
             "looks like — and it has to be done BEFORE the name join, because four parks "
             "share a name with a palika in the same district. Nobody is attributed to a "
             "park and no dot lands in one. "
             "751 OF 753 NAMES MATCH BY A DERIVED FOLD, not by a hard-coded alias list: "
             "strip the unit-type word (`Gaunpalika`, `Nagarpalika`, `Municipality`, and "
             "NSO's own `Metropolitian City`) and a leading district name, then match "
             "inside the district. The last one, `Melanchi` against COD's `Melamchi`, "
             "falls through to a unique edit-distance-1 match among that district's "
             "unclaimed polygons. "
             "PLACEMENT IS KONTUR'S 400 m GRID, 104,129 hexes, and Nepal needs it more than "
             "its unit count suggests: the local levels are fine in people and wild in "
             "area, from ~50 km² in the Kathmandu valley to 2,290 km² in Humla. Ten of the "
             "753 sit outside a factor of three against the census, against a shuffled "
             "median of 224, and log populations correlate at r = 0.9055 against a best "
             "shuffle of 0.1350. The ten are two different things — a town smeared into its "
             "hinterland (Rohini 4.09 beside Siddharthanagar 0.26, and pooling them gives "
             "1.31) and a block of the Parsa Terai that Kontur simply over-models and that "
             "does not pool away. Neither changes a count; the grid is a within-unit weight "
             "(§8.2). "
             "READ A CLUSTER AS COMPOSITION AT PALIKA SCALE. A local level averages 38,400 "
             "people over ~190 km², so a dot says 'this palika, drawn where Nepalis live' "
             "and nothing about which ward or village.",
    ),
    "mm": dict(
        name="Myanmar",
        source="2014 Census Report Volume 2-C: Religion, Table 1 (Department of Population)",
        basis="self-identification, enumerated census population, plus the state's own "
              "estimate of who it did not enumerate",
        view=[92.0, 9.4, 101.4, 28.7],
        note_public=(
            "**The most important number on this map of Myanmar is the one that is not a "
            "religion.** The 2014 census did not enumerate an estimated 1,206,353 people, "
            "and 1,090,000 of them are in Rakhine State — **34% of everybody there.** The "
            "census report says why, in its own words: *\"In Rakhine, an estimated 1.09 "
            "million people were not enumerated in the Census because they were not allowed "
            "to self-identify using a name not recognized by the Government.\"* That is the "
            "Rohingya. The remainder is Kayin (69,753) and Kachin (46,600), areas that were "
            "not under government control when the census was taken. "
            "**They are drawn as *not enumerated* rather than as Muslim, and the report "
            "itself would allow the stronger claim.** It states that *\"it is assumed that "
            "the non-enumerated population in Rakhine is mainly affiliated with the Islamic "
            "faith\"*, and publishes a second national figure on that basis: Islam 4.3% "
            "instead of the 2.3% the enumerated count gives. But it applies that assumption "
            "only to the country as a whole, and this map draws states — so these dots say "
            "that these people were not counted, and nothing about what they believe. "
            "**Without them Rakhine reads 96.2% Buddhist**, which is the census's own "
            "exclusion turned into a finding. "
            "**Christianity is an upland religion here and the boundary is sharp.** Chin is "
            "**85.4% Christian**, Kayah 45.8% and Kachin 33.8% — the American Baptist "
            "mission field from 1813 onward, plus Catholics and Anglicans, none of which the "
            "census separates — against 1.1% in Magway and Nay Pyi Taw. Chin and Kachin are "
            "the only states where something other than Buddhism holds a plurality or comes "
            "close. "
            "**Almost all of the country's traditional religion is in one state.** Shan "
            "holds 383,072 of Myanmar's 408,045 Animists, 94% of the national figure. Read "
            "that as a floor: nat propitiation is close to universal in Myanmar and normally "
            "accompanies Buddhism rather than replacing it, and a census that allows one "
            "religion per person counts those people as Buddhist. "
            "**The census is from 2014 and the country has since had a coup and a civil "
            "war.** Roughly three quarters of a million Rohingya left Rakhine for Bangladesh "
            "in 2017, three years after this count, and millions more people have been "
            "displaced since 2021. This is a picture of 2014 and is not a current one. "
            "**Fifteen units and seven categories is everything the census published.** "
            "There is no district or township religion table anywhere, no Buddhist school, "
            "no branch of Islam, and no Christian body named."),
        how="census, 2014, plus the state's count of who it missed",
        grain="states and regions, 3.4 million people on average",
        counts=_mm_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "mm" / "mm_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_mm_place_weight,
        note="THE ENTIRE PUBLISHED RELIGION OUTPUT OF THE 2014 CENSUS IS TWO TABLES IN A "
             "17-PAGE REPORT — Union plus 15 States/Regions, and a 1973/1983/2014 time "
             "series. There is no district or township religion table anywhere: not "
             "withheld at a finer tier, not hard to find, absent. So 15 units is not §14's "
             "resolution limit being applied, it is all there is, and §3.9b is why it is "
             "drawn. READ IT AS COMPOSITION, NEVER AS LOCATION — a state here averages 3.4 "
             "million people. "
             "THE NON-ENUMERATED ARE AN EIGHTH CATEGORY AND EVERY ROW OF IT IS `modelled`. "
             "They are not a religion and not an answer: they are DOP's own estimate of "
             "people the census did not reach, and 1,090,000 is a round number in the source "
             "because it is an estimate. Mapping them to `islam` — which DOP's own Union-"
             "level assumption would support — would invent a magnitude at a resolution the "
             "source does not publish, which is §14 rule 1. `inferred dots: hidden` empties "
             "this node exactly, which is the honest test of the country. "
             "THE `Total` ROW IS THE ENUMERATED TOTAL AND IS NOT THE UNIVERSE. 50,279,900 "
             "enumerated plus 1,206,353 non-enumerated is 51,486,253, the report's own "
             "overall figure and what is drawn. "
             "THE COUNTING GEOGRAPHY IS NOT THE ADMINISTRATIVE ONE AND THE FEATURE COUNT "
             "SAYS SO: COD's ADM1 has 18 features because the standard p-codes split Bago in "
             "two and Shan in three, against the census's 15 rows. They are dissolved by "
             "rule — strip a parenthesised qualifier and group — and MIMU's own religion "
             "sheet confirms the intent by coding them MMR111 and MMR222. Every "
             "multi-member group is asserted CONTIGUOUS, and the 15 are asserted to tile "
             "with no overlap. "
             "THE PARSE HAS TWO CHECKS A COLUMN PERMUTATION CANNOT SURVIVE. DOP prints a "
             "percentage under every count, so count/total must reproduce it on all 110 "
             "cells; and MIMU published an independent p-coded transcription of the same "
             "table, matched here BY ITS FIGURES rather than by name or position, 1:1 both "
             "ways. One cell of the report is simply wrong — Kachin's Hindu share is 0.349% "
             "and prints as 0.4 — and it is listed so a second one fails the build. "
             "THE PLACEMENT GRID IS 2023 AND THE CENSUS IS 2014, AND IN ONE STATE THAT IS "
             "VISIBLE. Kontur reads 0.58× on Rakhine against the population drawn there and "
             "0.89× against the enumerated count alone, in line with the other fourteen "
             "states; the gap is the ~750,000 Rohingya who left in 2017. So Rakhine's "
             "not-enumerated dots sit south of where those people actually lived, in the "
             "northern townships. No source publishes them below state level, a uniform "
             "spread would put them in the Arakan mountains, and inventing a northern "
             "concentration would be §14.4.",
    ),
    "kh": dict(
        name="Cambodia",
        source="General Population Census 2019, Tables 2.5.1 and 2.1.1 (NIS)",
        basis="self-identification, whole census population",
        view=[102.2, 9.9, 107.8, 14.8],
        note_public=(
            "**Cambodia is 97.1% Buddhist and the entire map is in the remaining 3%**, "
            "which sits in two places that have nothing to do with each other. "
            "**The Cham Muslim belt follows the water.** Tbong Khmum is 11.8% Muslim — "
            "91,667 people, the largest share and the largest number in the country — and "
            "behind it come Kratie at 6.6%, Kampong Chhnang at 5.8%, Stung Treng at 4.7% "
            "and Koh Kong at 4.6%. That is the Mekong upstream of Phnom Penh and the shore "
            "of the Tonle Sap, and it is where the Cham have lived since they arrived from "
            "the Champa kingdom of central Vietnam from the fifteenth century onward. "
            "Against 0.1% in Svay Rieng and Kampong Speu, a hundred kilometres away. "
            "**Read the Muslim figure as a floor, for a specific reason**: the Cham were "
            "singled out for destruction under Democratic Kampuchea and are estimated to "
            "have lost between a third and a half of their people between 1975 and 1979. "
            "**The north-eastern highlands are the other half of the map, and the census "
            "will not name what is there.** Ratanak Kiri is 23.2% *other religion* and "
            "Mondul Kiri 21.2%, while fifteen of the twenty-five provinces are printed as "
            "0.0%. It is the sharpest such geography anywhere on this map. NIS says in the "
            "paragraph above its own table what the cell mostly is — *the local religious "
            "system of the highland tribal groups* — which is the animist tradition of the "
            "Bunong, Tampuan, Jarai, Kreung, Brao and Kavet: spirit forests, buffalo "
            "sacrifice, ancestor practice. It gets no box of its own on the form, so this "
            "map can show that a fifth of two provinces answers none of the three named "
            "religions and cannot show what they answer instead. "
            "**Christianity is 0.32% of Cambodia and its highest shares are in those same "
            "two provinces** — Mondul Kiri 4.0%, Ratanak Kiri 2.1%, twelve and six times "
            "the national rate — which is evangelical mission among the same highland "
            "peoples. Phnom Penh has more Christians in absolute terms and a much lower "
            "share. The two categories are working on the same population and the map "
            "shows both at once. "
            "**Four categories is all the census offers.** There is no cell for the "
            "Buddhist school, none for the branch of Islam, and none naming a single "
            "Christian body, so nothing on this map distinguishes the Mahanikay from the "
            "Thommayut, the mainstream Sunni majority from the Kan Imam San of Udong, or a "
            "Catholic from a Pentecostal."),
        how="census, 2019",
        grain="provinces, 622,000 people on average",
        counts=_kh_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "kh" / "kh_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_kh_place_weight,
        note="THE SOURCE PUBLISHES PERCENTAGES AND NOT COUNTS, WHICH IS WHY TWO TABLES ARE "
             "READ. Table 2.5.1 gives each province's religious composition to one decimal "
             "place and no absolute figure anywhere; Table 2.1.1, nine pages earlier, gives "
             "each province's population. The counts here are the product, apportioned by "
             "largest remainder so a province's four categories sum to its published "
             "population exactly. That is arithmetic on two published figures rather than "
             "an estimate, so the rows are `measured` — but every cell carries a rounding "
             "band of ±0.0005 × the province population, ±1,141 people in Phnom Penh and "
             "±21 in Kep. "
             "FIFTEEN OF THE HUNDRED CELLS ARE PRINTED AS `0.0` AND DRAW NOTHING. All "
             "fifteen are `Other`. Zero is the published figure and is drawn as zero; it is "
             "not evidence that nobody is there, since `0.0` is anything under 0.05% — "
             "§3.5, marked and not filled. "
             "25 PROVINCES FOR 15.55M IS NIS's CEILING RATHER THAN A CHOICE. Table 2.5.1 is "
             "the only religion table in the 304-page report, and the 2008 census's own "
             "priority-table list gives `A4 Population by Religion` an age and a sex "
             "dimension and no geography at all. Spec §3.9b is why it is drawn anyway — "
             "there is no minimum unit count. READ IT AS COMPOSITION, NEVER AS LOCATION: a "
             "cluster of dots says 'this province, drawn where Cambodians live' and nothing "
             "about which village. "
             "THE UNIVERSE EXCLUDES CAMBODIANS WORKING ABROAD, which both tables state in a "
             "footnote. That is a large population — several hundred thousand in Thailand "
             "alone — and it is why the census total is 15,552,211 rather than the ~16.5M a "
             "projection would give. "
             "THE FOUR CATEGORIES SUM TO 100% AND THERE IS NO `not stated` CELL, so the "
             "whole of that universe is drawn. "
             "THE JOIN IS CHECKED ON A KEY THE PAIRING DOES NOT USE. Provinces are paired "
             "to OCHA's polygons by a Khmer-romanisation fold (three of the twenty-five "
             "names differ — Otdar/Oddar Meanchey, Siem Reap/Siemreap, Tbong/Tboung Khmum), "
             "and then the census's PRINT POSITION is checked against COD's own "
             "`ADM1_PCODE`, which agrees on all 25. The two have different origins, so a "
             "transposed row would break it while every total still reconciled. "
             "The dots are spread across 77,453 Kontur 400m hexagons weighted by hex "
             "population (sources/kh_grid.py), which matters because the north-east is both "
             "the emptiest part of the country and the only part with a distinctive "
             "religious mix. It also removes the Tonle Sap, which runs from ~2,700 to "
             "~16,000 km² and lies inside five provinces. "
             "KONTUR MODELS 4.6× TOO MANY PEOPLE IN PAILIN — 374,607 against a census "
             "75,112, where every other province sits between 0.62× and 1.72×. The join was "
             "cleared four ways (the polygons tile with no overlap, COD's own areas match "
             "the geometry, Pailin's hexes are inside Pailin's bounding box, and the "
             "population is spread over 779 hexes rather than spiking), so this is the "
             "model's error and not the map's. It changes no count — the grid is a "
             "within-unit weight — but Pailin's dots sit on the least trustworthy placement "
             "surface in the country.",
    ),
    "rs": dict(
        name="Serbia",
        source="Popis stanovništva 2022 (Republički zavod za statistiku)",
        basis="self-identification, voluntary question",
        view=[18.6, 42.1, 23.2, 46.3],
        note_public=(
            "**Serbia is 81% Orthodox and the whole of the interesting map is in the "
            "remaining fifth**, which sits in two places and nowhere else. "
            "**Vojvodina in the north is where the Habsburg border used to be**, and it "
            "still reads that way. Kanjiža is 85% Catholic, Senta 74%, Ada 72%, Subotica "
            "48% — Hungarian towns along the Tisza — while **Bački Petrovac is 57% "
            "Protestant and Kovačica 41%**, which are the Slovak Lutheran colonies "
            "planted there in the 1740s and still legible as two dark spots in an "
            "otherwise Orthodox province. The census offers one Protestant cell, so "
            "nothing on the map says Lutheran; the geography says it instead. "
            "**The Muslim map is two separate places 300 km apart.** The Sandžak in the "
            "south-west is Bosniak — Tutin 94%, Novi Pazar 83%, Sjenica 78%, Prijepolje "
            "47% — and the Preševo valley on the Macedonian border is Albanian: Preševo "
            "94%, Bujanovac 69%. Tutin and Preševo are the two least Orthodox "
            "municipalities in Serbia, at 2.0% and 4.5%. "
            "**Irreligion is 1.25% and is almost entirely four Belgrade municipalities.** "
            "Stari grad reports 7.7% atheist or agnostic, Vračar 6.2%, Savski venac 5.4%, "
            "against 0.32% in Lazarevac an hour down the road. On this map that is a very "
            "small number: Czechia is 20 points higher and Estonia higher again. "
            "**Two answers are not drawn and together they are 7.9% of the country.** "
            "169,486 people declined the question, which the constitution makes voluntary, "
            "and they are concentrated in Vojvodina's mixed towns — Dimitrovgrad 10.2%, "
            "Subotica 10.1% — where declaring anything carries the most weight. A further "
            "355,484 are recorded as unknown, and those are a different pattern "
            "altogether: central Belgrade, 17.3% in Savski venac and 14.4% in Stari grad "
            "against under 1% in Preševo. That second group tracks the declared-atheist "
            "share, so **leaving it out makes every share on this map slightly more "
            "religious than the country is** — the correction runs one way and this map "
            "does not make it. "
            "**And 602 Jews in the whole country**, of whom 78 are in Stari grad and 66 "
            "in Novi Sad. Before 1941 Belgrade and Novi Sad each held thousands."),
        how="census, 2022, voluntary question",
        grain="municipalities, 36,000 people on average",
        counts=_rs_counts,
        # Counts are on the 168 municipalities; the Kontur hexes carry no municipality
        # code of their own, so sources/rs_geo.py assigns and clips every hex and writes
        # the `unit` column this reads. Russia's and Kenya's wiring exactly.
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "rs" / "rs_grid_400m.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_rs_place_weight,
        note="RZS is self_id on a question the constitution makes voluntary; the 2.55% who "
             "declined and the 5.35% RZS records as unknown are both excluded rather than "
             "drawn (spec §3.5), so this is 92.07% of Serbia. The table is exact — no "
             "rounding, no suppression, every municipality's categories summing to its own "
             "total — and it is published at municipality and nowhere finer, so no "
             "category is allocated and every row is measured. "
             "Boundaries are GISCO LAU 2021, which carries Serbia at no download because "
             "its set is not the EU27; GISCO splits Novi Sad into Novi Sad and "
             "Petrovaradin where the census does not, so Petrovaradin is dissolved back "
             "in (sources/rs_geo.py). Belgrade arrives as its 17 city municipalities and "
             "Niš as its 5, which removes §12's capital-in-one-polygon problem for free — "
             "and creates the only name collision in the file, since both cities have a "
             "Palilula. Placement is Kontur's 400 m H3 population grid, 59,823 hexes, a "
             "median of 312 per municipality; Kontur under-models the Albanian-majority "
             "Preševo valley by a factor of five, so Bujanovac's and Preševo's dots sit "
             "on the weakest surface in the country. Kosovo is in the source as an empty "
             "row and is not drawn from it.",
    ),
    "lt": dict(
        name="Lithuania",
        source="Gyventojų surašymas 2021 (Statistics Lithuania)",
        basis="self-identification, voluntary question",
        view=[20.8, 53.8, 26.9, 56.5],
        note_public=(
            "**The most detailed religion question in Europe on a country this size, and "
            "every one of its answers is a historical boundary somewhere.** Lithuania is "
            "74% Roman Catholic and separates things no other census on this map "
            "separates: Roman from Greek Catholics, Orthodox from Old Believers, and the "
            "Karaims from the Jews. "
            "**Biržai is the Reformed municipality.** 8.9% of it is Evangelical Reformed "
            "against 0.49% in the next-highest place in the country — this is the Radvila "
            "family's Calvinist estate, granted in the 1560s, and four and a half centuries "
            "later it is still a single bright spot with nothing around it. "
            "**The Lutherans are the old Prussian border.** Tauragė 9.2%, Pagėgiai 5.4%, "
            "Šilutė 4.6%, Jurbarkas 3.4% — a band along the Nemunas that was Lithuania "
            "Minor under Prussia, Lutheran since the Reformation and never Catholic. The "
            "line between it and Catholic Samogitia (Šilalė is 91.5% Catholic) is a "
            "sixteenth-century state border still visible in a 2021 census. "
            "**Visaginas is 49% Orthodox** — a town built in the 1970s for the Ignalina "
            "nuclear plant and populated from across the Soviet Union, and the only "
            "municipality in Lithuania that is not majority Catholic. **The Old Believers "
            "are the north-east**: Zarasai 12.1%, Švenčionys 5.0%, descendants of refugees "
            "from the Russian church reforms of the 1650s, and 18,196 of them — the largest "
            "count of Old Believers any source on this map makes directly. "
            "**Two very small communities, and a warning about both.** 2,165 Sunni Muslims "
            "are the Lipka Tatars, settled around Vilnius and Alytus since the fourteenth "
            "century. 255 people answered Karaim — the Turkic-speaking Karaite community "
            "brought from Crimea in 1397, whose historic home is Trakai. **Only 101 of them "
            "are drawn, and not in Trakai.** Statistics Lithuania withholds any cell small "
            "enough to identify people, so 60% of the Karaims, 29% of the Greek Catholics "
            "and 28% of the Adventists have no municipality on this map at all. The "
            "suppression is 0.06% of Lithuania and a majority of its smallest religion, "
            "which is what disclosure control does: it hides the rare things, and this map "
            "does not fill them back in. "
            "**And 13.7% did not answer.** That is a refusal, not irreligion — 6.1% "
            "separately said they belong to no religion, and that share has not moved since "
            "2001 while non-response has trebled from 5.4%. Almost all of the fall in "
            "Catholic identification over twenty years has gone into the blank rather than "
            "into 'none'. The people who did say 'no religion' are not in Vilnius: Joniškis "
            "(12.5%), Akmenė (12.3%) and Klaipėda (11.8%) lead it, which is the north and "
            "the coast rather than the capital."),
        how="census, 2021, voluntary question",
        grain="municipalities, 40,000 people on average",
        counts=_lt_counts,
        # Counts are on the 60 savivaldybės; the Kontur hexes carry no municipality code,
        # so sources/lt_geo.py assigns and clips every hex and writes the `unit` column
        # this reads. Serbia's, Kenya's and Russia's wiring exactly.
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "lt" / "lt_grid_400m.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_lt_place_weight,
        note="Statistics Lithuania is self_id on a voluntary question; the 13.67% not "
             "stated are excluded rather than drawn (spec §3.5), so this is 86.3% of the "
             "country. The table is published at municipality and nowhere finer, so nothing "
             "is allocated and every drawn row is measured. "
             "**298 of the 1,020 municipality cells are withheld as confidential** and are "
             "NOT filled in — 1,683 people, 0.06% of Lithuania, but 60.4% of its Karaims, "
             "29.3% of its Greek Catholics and 28.2% of its Adventists (spec §3.8; "
             "sources/lt.py reports the shortfall per category). Small religions are "
             "understated here in proportion to how small they are. "
             "The cube is SDMX from `osp-rs.stat.gov.lt`, which is open — the Cloudflare "
             "wall recorded in sources.md §11 is on `osp.stat.gov.lt`, the web UI, and the "
             "two are different machines. "
             "Boundaries are GISCO LAU 2021 and the join is by CODE: GISCO's `LAU_ID` is "
             "the savivaldybė code the census keys on, so there is no name matching at all. "
             "Placement is Kontur's 400 m H3 grid, 63,766 hexes, a median of 1,206 per "
             "municipality. Kontur displaces city population outward — Šiauliai city reads "
             "0.48x and the rajono ring around it 1.92x — but every city/ring pair closes "
             "between 0.89x and 1.05x, which is what says the join is right and the surface "
             "is merely blurred (sources/lt_geo.py).",
    ),
    "kr": dict(
        name="South Korea",
        # English rather than 인구총조사, unlike mk's Cyrillic `Попис 2021`: the panel is read
        # text and does not reflow, so a long source line overlaps the granularity line
        # beneath it. The table id is what actually identifies this source anyway.
        source="Population Census 2015 (KOSIS table DT_1PM1502)",
        # Kept short deliberately — the panel does not reflow, and a longer basis line
        # overlaps the granularity line beneath it.
        basis="self-identification, 20% census sample",
        view=[125.8, 33.0, 129.7, 38.7],
        note_public=(
            "**A majority of South Koreans report no religion at all** — 56.1%, the "
            "largest single answer in the country and the highest share of any country "
            "drawn here. What is left divides three ways: Protestant 19.7%, Buddhist "
            "15.5%, Catholic 7.9%. "
            "**And the Buddhist and Protestant halves are geographic opposites.** Buddhism "
            "is the south-east — Ulsan 29.8%, South Gyeongsang 29.4%, Busan 28.5%, Daegu "
            "23.8% — and thins to 8.6% in North Jeolla and 8.8% in Incheon. Protestantism "
            "runs the other way: North Jeolla 26.9%, Seoul 24.2%, South Jeolla 23.2%, "
            "against 10.5% in South Gyeongsang. That is the Yeongnam/Honam line, the "
            "deepest regional division in Korean politics, drawn here in religion. "
            "**Catholicism is Gangnam.** Seoul's Gangnam-gu is 16.3% Catholic and Seocho-gu "
            "16.1% — the wealthy districts south of the river — against a national 7.9%. "
            "Nowhere else in the country reaches those numbers, and the pattern is class "
            "rather than region. "
            "**Won Buddhism has a homeland and the map finds it exactly.** Sotaesan founded "
            "it in 1916 at Yeonggwang and put its headquarters at Iksan; 2015 counts Iksan "
            "at 4.07% and Yeonggwang at 3.91% against a national 0.17%, more than twenty "
            "times over. A movement of 84,141 people is legible as two bright spots on the "
            "county it started in. **Daesun Jinrihoe does the same at Yeoju** (0.86% "
            "against 0.08%), where its temple complex is. "
            "**Confucianism is counted as a religion here and almost nowhere else** — "
            "75,703 people, concentrated in the old lineage country of the south-west: "
            "Haenam 1.53%, Jangheung 1.36%. Read it as the institutional core around the "
            "hyanggyo, not as Confucian practice, which is near-universal and is not what "
            "anyone is reporting. "
            "**This is the last Korean census that asked.** The question was dropped after "
            "2015, so nothing here will be updated."),
        how="census, 2015, 20% sample",
        grain="si/gun/gu, 214,000 people on average",
        counts=_kr_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "kr" / "kr_grid_400m.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_kr_place_weight,
        note="THE FIGURES ARE A 20% SAMPLE GROSSED UP. 2015 was a register-based census "
             "and religion rode on the sample survey rather than the register, so every "
             "cell carries sampling error — which matters most for the small categories "
             "this country is worth drawing for. Daejonggyo is 3,101 people nationally, so "
             "its district cells are a few sampled households each. "
             "The universe is 49,052,389 against a census population of 51,069,375, a gap "
             "of 3.95% the table does not explain; it is reported rather than filled "
             "(spec §3.5, sources/kr.md §5). "
             "**KOSIS bot-blocks its own data endpoints**, so this table cannot be fetched "
             "by script — the metadata endpoint is open and the download endpoints answer "
             "200 with an HTML alert. Anita downloaded it through a browser on 2026-09-05; "
             "sources.md §11g has the click path. "
             "The drawn tier is 229 si/gun/gu. KOSIS publishes 252 at its second level, the "
             "difference being the general gu of twelve large cities, and no boundary set "
             "carries those — they are in kr.csv at a `gu` level, undrawn, against a source "
             "that ever ships them. Seoul's 25 gu are unaffected: those are full local "
             "governments and are drawn. "
             "The join is by NAME inside a province and never across the country, because "
             "Jung-gu, Dong-gu, Nam-gu, Seo-gu and Buk-gu each name five or six different "
             "districts nationally. Provinces bridge on ISO 3166-2:KR; districts are "
             "romanised from Hangul and matched on a deliberately loose fold, 1:1 or "
             "reported. **geoBoundaries omits an entire county** — Yeonggwang-gun, 53,984 "
             "people — and it is rebuilt from the eleven ADM3 eup and myeon that lie "
             "outside every ADM2 polygon, 481 km² against a published 475. That matters "
             "more than a missing rural county usually would: Yeonggwang is where Won "
             "Buddhism was founded and the second most Won Buddhist place on earth. "
             "Placement is Kontur's 400 m grid, 71,478 hexes; every unit's Kontur/census "
             "ratio lands between 0.30 and 3.5 with a national 1.044, which is what "
             "confirms the name join.",
    ),
    "gy": dict(
        name="Guyana",
        source="Population and Housing Census 2012 (Bureau of Statistics)",
        basis="self-identification",
        view=[-61.6, 1.0, -56.4, 8.7],
        note_public=(
            "**Guyana is the indenture map.** A quarter of the country is Hindu and a "
            "fifteenth Muslim — the descendants of labourers brought from India after "
            "emancipation to cut sugar — and they are not spread about: they are on the "
            "coastal strip, in the order the plantations were. **East Berbice-Corentyne is "
            "42.1% Hindu**, Essequibo Islands-West Demerara 37.7%, Mahaica-Berbice 34.1%, "
            "Pomeroon-Supenaam 33.2%. The Muslims sit on the same ground a third as thick — "
            "Essequibo Islands 11.8%, East Berbice 9.5% — because they came on the same "
            "ships to the same estates. "
            "**And the interior is the exact photographic negative of it.** Upper "
            "Takutu-Upper Essequibo, the Rupununi savannah, is **50.1% Roman Catholic and "
            "0.4% Hindu**; Potaro-Siparuni is 39.8% Catholic. That is the Amerindian "
            "interior and the Catholic missions that worked it, and the two halves of "
            "Guyana barely touch: the country's most Hindu region and its most Catholic one "
            "are 200 km apart with almost nothing in between, because almost nobody lives "
            "in between. "
            "**Barima-Waini in the north-west is 39.9% Pentecostal and 33.8% Catholic** — "
            "three quarters of it between two churches — and Pentecostalism is the largest "
            "Christian answer in the country at 22.8%, ahead of the Anglicans and Catholics "
            "put together. "
            "**Linden is the third Guyana.** Upper Demerara-Berbice, the bauxite region, is "
            "36.0% Pentecostal, 14.8% Seventh Day Adventist, and carries both the country's "
            "highest no-religion share (7.2%) and its highest Rastafarian share (1.3%). It "
            "is Afro-Guyanese, industrial, and almost untouched by the Hindu-Muslim coast "
            "20 km away. "
            "**3,496 Rastafarians are counted here by name**, which almost no census "
            "anywhere does — most fold them into a residual — and 421 Bahá'ís. "
            "**What is missing from this map is Amerindian religion**, and its absence is "
            "an artefact of the form rather than a finding: the 2012 census offered no box "
            "for traditional practice, so the nine Amerindian nations of the interior "
            "answered one of the Christian categories instead. Read the Catholic interior "
            "as 'the church people gave as their answer', not as the whole of what is "
            "practised there."),
        how="census, 2012",
        grain="administrative regions, 75,000 people on average",
        counts=_gy_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "gy" / "gy_grid_400m.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_gy_place_weight,
        note="THE WHOLE COUNTRY IS ONE PAGE OF A PDF. The Bureau of Statistics runs no "
             "dissemination platform of any kind; Table 2.19 of the *Final 2012 Census "
             "Compendium 2* is the entire source, and it is a perfect partition — the "
             "thirteen categories sum to each region's total, the ten regions sum to the "
             "separately published national Table 2.17 category by category, and the grand "
             "total is the census population of 746,955. 100% of it is drawn. "
             "**Non-response was prorated into the categories by the office and cannot be "
             "undone.** 363 not stated, 16,331 no-contact and 7,443 institutional — 24,137 "
             "people, 3.2% of Guyana — were distributed across the thirteen categories in "
             "proportion before publication. This is the only source on the map where that "
             "has happened, so unlike everywhere else there is no non-response to leave "
             "undrawn (spec §3.5), and every count here carries its share of those people. "
             "The census is 2012; the 2022 census has published only a preliminary report "
             "with no religion table in it. "
             "Boundaries are geoBoundaries ADM1 and **the join is a published standard**: "
             "the census names no region at all — its columns are `Region 1`..`Region 10` "
             "— and geoBoundaries carries `shapeISO`, which is ISO 3166-2:GY, which is the "
             "ten regions in region-number order. No names were matched, which is as well, "
             "because the boundary file misspells Region 1 as `Barina-Waini`. "
             "Placement is Kontur's 400 m H3 grid, 5,773 hexes, and Guyana is the country "
             "that most needs it: Region 4 holds 41.7% of the population on 1.0% of the "
             "land while Region 8 is 11,077 people over 20,555 km², so an equal share per "
             "polygon would have washed the empty interior in dots of a single colour. "
             "Every region's Kontur/census ratio sits between 0.94x and 1.40x, which is "
             "what confirms the ISO join; the two loosest are Cuyuni-Mazaruni and the "
             "Rupununi, where a building-footprint model over-predicts scattered interior "
             "settlement, so placement WITHIN those two is the weakest on this country.",
    ),
    "vn": dict(
        name="Vietnam",
        source="Population and Housing Census 2009, Biểu 7 (General Statistics Office)",
        basis="self-identification",
        view=[101.5, 8.0, 110.5, 23.6],
        note_public=(
            "**Four fifths of this map is one colour, and that colour means we do not "
            "know.** Vietnam's census asks which of the state-recognised religious "
            "organisations a person belongs to, and **81.8% of the country belongs to "
            "none of them**. That is not irreligion, which is why those people are drawn "
            "as *Religion unknown* rather than as no religion. Ancestor veneration is "
            "close to universal, the village đình and the mother-goddess rites of đạo Mẫu "
            "are everywhere, and the great majority of people who would call themselves "
            "Buddhist in conversation are in that grey rather than in the Buddhist figure "
            "below — the census counts 6.8 million Buddhists in 2009 in a country usually "
            "described as around 45% Buddhist by practice. Read the coloured dots as "
            "**registered religion** and the grey as a question the census did not ask. "
            "**What is drawn is intensely regional, far more so than in any other country "
            "here.** Six provinces are over 45% affiliated and six are under 2%: **An Giang "
            "is 94.5%** and Sơn La is 0.4%. Almost nothing about Vietnamese religion is "
            "evenly spread. "
            "**An Giang is the reason.** The Mekong Delta produced its own Buddhist "
            "movements in the nineteenth and twentieth centuries and the census counts four "
            "of them separately: Bửu Sơn Kỳ Hương (1849), Tứ Ân Hiếu Nghĩa (1867), Hòa Hảo "
            "(1939) and Hiếu Nghĩa Tà Lơn. **An Giang alone is 43.7% Hòa Hảo** — 936,974 "
            "people, 65% of all Hòa Hảo in Vietnam — plus 84% of the country's Tứ Ân Hiếu "
            "Nghĩa and 76% of its Bửu Sơn Kỳ Hương. These are lay movements with no clergy "
            "and no temples, and they exist essentially in one province and its neighbours: "
            "Cần Thơ is 19.1% Hòa Hảo and Đồng Tháp 11.8%. "
            "**Tây Ninh is the other one-province religion.** Caodaism was founded there in "
            "1926, its Holy See is there, and the province is **35.6% Caodaist** — 47% of "
            "all Caodaists in the country. "
            "**Catholicism has two homelands and they are 1,500 km apart.** Đồng Nai is "
            "32.1% Catholic and Nam Định in the Red River Delta holds 369,793 — the "
            "seventeenth-century Jesuit mission field, and the place most of the southern "
            "Catholics came from when nearly a million people moved south in 1954. "
            "**The Central Highlands are the missionary map.** Kon Tum is 31.2% Catholic, "
            "Lâm Đồng 25.6%, Đắk Nông 20.5% — and the Protestant share peaks in the same "
            "provinces (Đắk Nông 10.3%, Gia Lai and Đắk Lắk 8.6%) rather than anywhere Kinh "
            "Vietnamese live. Protestantism here is Montagnard, and in Điện Biên (7.5%) and "
            "Lai Châu (7.1%) it is Hmong. It is also the only category that grew between the "
            "two censuses, by 31%. "
            "**Trà Vinh is 49.7% Buddhist and its Buddhism is not the same religion as Ho "
            "Chi Minh City's.** The delta provinces of Trà Vinh and Sóc Trăng are Khmer "
            "Krom, and Khmer Buddhism is Theravada; the northern and urban Buddhism is "
            "Mahayana. The census offers one box marked Buddhist and the map cannot show "
            "the difference. "
            "**Ninh Thuận is the Cham province.** 7.2% Cham Balamon — the last living Hindu "
            "tradition descended from the Indianised kingdoms of Southeast Asia — and 4.5% "
            "Muslim, and those Muslims are mostly **Bani**, a thousand-year-old Cham "
            "localisation of Islam that the census does not distinguish from the Sunni Cham "
            "of An Giang and Saigon. Two halves of one Cham religious system, and only one "
            "of them has a colour here."),
        how="census, 2009, state-recognised organisations only",
        grain="provinces, 1.4 million people on average",
        counts=_vn_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "vn" / "vn_grid_400m.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_vn_place_weight,
        note="**THE 70.2 MILLION IN GREY ARE COMPUTED, NOT PRINTED.** Biểu 7's universe is "
             "people with a religion, so the census publishes no row anywhere for anyone "
             "else; each province's *Religion unknown* figure is its published population "
             "minus its published religious total, which is the complement of a partition "
             "rather than an estimate — the religions, the not-stated cell and the residual "
             "sum to the population in all 63 provinces to the person. The 2019 census "
             "publishes the same category directly, nationally, at 83,046,105. "
             "**THE 2019 CENSUS IS BETTER AND HAS NO GEOGRAPHY, SO 2009 IS DRAWN.** Both "
             "censuses ask about religion. The 2019 volume publishes the answer on **one "
             "page, nationally**, with no province table anywhere, while giving ethnicity a "
             "full province × 54-group tabulation over 167 pages; the 2009 volume publishes "
             "religion by province over 32 pages. So the newer census is the better "
             "measurement of how many and says nothing about where, and this map is the "
             "older one drawn as it stands, at its own year (spec §3.4, as India 2011 and "
             "Russia 2012 are). It is NOT rescaled to 2019 totals: that would mean one "
             "national factor per religion applied to all 63 provinces, and the factors are "
             "not credible as history — Buddhism −32.3%, Hòa Hảo −31.4% and Cao Đài −31.2% "
             "over the same decade in which the population grew 12%, three unrelated "
             "traditions moving together to within one percentage point. That is the "
             "instrument changing, and smearing it evenly over every province would assert "
             "something nobody measured. "
             "**Three bodies recognised after 2009 therefore have no geography at all** and "
             "are absent from the map though present in the data: the Seventh-day Adventists "
             "(11,830), the Latter-day Saints (4,281) and Hiếu Nghĩa Tà Lơn (401). "
             "**The table reconciles completely.** All 63 provinces sum to their own printed "
             "totals, the provinces sum to the national row in every one of the fourteen "
             "categories, and each of the six socio-economic regions sums to its own row — "
             "which is also what verifies the province-to-region composition, since the "
             "census prints regions and provinces as separate flat blocks with no marker of "
             "which belongs to which. Non-response is 30 people nationally, the smallest on "
             "this map by three orders of magnitude, and is reported rather than drawn. "
             "**Boundaries are geoBoundaries ADM1 pinned to a pre-2025 commit**, because "
             "Vietnam merged its 63 provinces into 34 on 1 July 2025 and a current file "
             "would be a different country from the one the census counted. The join is a "
             "hand-built bridge from GSO's administrative codes to ISO 3166-2:VN — two "
             "numeric-looking code spaces that agree on **no** province, so a naive join "
             "would have produced 63 silent wrong answers — and it is re-derived by name on "
             "every run, with Ho Chi Minh City forced by elimination as the only unmatched "
             "province against the only unclaimed code. The boundary file's 64th feature is "
             "Côn Đảo, an offshore district carrying its parent's ISO code, so dissolving on "
             "that code reassembles the province. "
             "Placement is Kontur's 400 m H3 grid, 231,746 hexes; every province's "
             "Kontur/census ratio lands between 0.73 and 1.88 with a national 1.14, which is "
             "the expected shape for a 2023 surface over a 2009 census in a country that "
             "grew 12% — and it is also what confirms the code bridge, since a scrambled one "
             "would pair Ho Chi Minh City's 7.2 million with Bắc Kạn's 294,000.",
    ),
    "cn": dict(
        name="China",
        source=("2000 census nationality by county (NBS) on 2010 provincial totals; "
                "religion shares from the Chinese General Social Survey, 2012+2017+2021"),
        basis=("ethnicity, derived — no census asked about religion; plus self-identified "
               "religion from a pooled national survey, at province"),
        view=[73.0, 17.5, 135.5, 54.0],
        note_public=(
            "**China has never asked anybody on this map what their religion is.** No "
            "Chinese census has carried a religion question, so unlike every other "
            "country here there is no answer to draw. What the state does count, and "
            "publish by county, is **nationality** — and for a small number of China's 56 "
            "nationalities the ethnic category and a religion are the same historical "
            "object rather than two correlated ones. A Hui person is *defined* as a "
            "Chinese-speaking Muslim, with no separate language or territory; the Turkic "
            "and Iranian peoples of the northwest were Muslim before any of them was a "
            "census category; Tibetan Buddhism is what the word Tibetan carries. Those are "
            "what is drawn here, and nothing else is. "
            "**The second thing drawn here is what people say about themselves when a "
            "survey asks, and it is deliberately a small number.** China's censuses do not "
            "ask, but its main academic social survey does — *which religion do you belong "
            "to* — and pooling three waves of it gives about 32,000 answers across 29 of "
            "the 31 provinces. Roughly **one person in twelve names a religion**. Those are "
            "the Buddhist and Protestant dots across eastern China: Buddhism heaviest in "
            "Zhejiang, Fujian and Jiangxi at ten to fifteen percent, Protestantism heaviest "
            "in Henan, which is the province usually described as China's Christian "
            "heartland, and in the northeast. "
            "**The other eleven in twelve are still grey, and that is a choice rather than "
            "a finding.** They are drawn as *Religion unknown* — counted by the census, "
            "placed where it puts them, nothing claimed about what they believe. **They are "
            "emphatically not drawn as irreligious**, because in China the two halves of "
            "that survey question are not equally trustworthy. Ask people to name a "
            "religion and about 92% name none; ask instead whether they tend graves, visit "
            "temples or believe in deities and most of it comes back. Pew's *Measuring "
            "Religion in China* puts Buddhism alone at 4% by self-identification and 33% by "
            "belief, from the same two surveys in the same year. **So the answer *yes, I am "
            "a Buddhist* is a measurement and the answer *none* is mostly an artefact of "
            "the wording.** This map draws the first and leaves the second grey. Chinese "
            "folk religion, Daoism and most of a Christian population usually estimated in "
            "the tens of millions are still inside that grey. "
            "**And the number that is drawn is falling, which may not be about belief.** "
            "Across the three survey waves used here — 2012, 2017 and 2021 — the share "
            "naming any religion at all fell from 14.5% to 7.5%, in every category at once, "
            "including Islam in a population whose Muslim nationalities were growing. Some "
            "of that is likely a real change in what people are willing to tell an "
            "interviewer. This map pools the waves, so it sits nearer the middle of that "
            "range than the end of it. "
            "**Every coloured dot here is an inference, and the `inferred dots` control "
            "removes all of them.** That is the honest test of this country: turn it on and "
            "China loses its colour entirely, because nothing in it was counted as religion "
            "by anybody. "
            "**The Mongols are deliberately absent too**, and they are the biggest "
            "judgement call in the country. Tibetan Buddhism among Mongols is real history, "
            "but at 5.8 million people they would have outnumbered Tibetans and made Inner "
            "Mongolia the largest Buddhist region in China on the strength of an assumption "
            "nothing measures — decades after the monastic system they would have been "
            "counted through was dismantled. The same reasoning leaves out the Tu, and it "
            "is the reason to trust what remains. "
            "**What the map does show is real and is not obvious.** Islam in China is not "
            "only Xinjiang: the Hui live in every province, so the Muslim layer runs from "
            "Kashgar to Kaifeng and down to a single village cluster in Sanya on Hainan. "
            "Linxia in Gansu and the Ningxia countryside are as densely Muslim as anywhere "
            "in the northwest. Tibetan Buddhism reaches far outside Tibet, across western "
            "Sichuan, Qinghai and southern Gansu. And China's only Theravada population is "
            "the Dai of Xishuangbanna and Dehong, whose monasteries belong to the "
            "Southeast Asian world rather than the Chinese one. "
            "**If you want to know which single layer here to trust least, it is "
            "Protestantism.** Where it is heaviest — Henan, Heilongjiang, Jiangsu, Zhejiang "
            "— is well attested from outside this survey. But the *ordering* of the "
            "provinces is much less steady between the 2012 and 2021 waves than Buddhism's "
            "is, so read the Protestant layer as a reasonable picture of where Chinese "
            "Protestantism has been over the last decade rather than a precise one of where "
            "it is now. Buddhism is on much firmer ground: its provincial pattern holds "
            "across all three waves, and the coastal southeast really is the Buddhist part "
            "of China. "
            "**The Protestant dots in the far southwest are built differently from the rest "
            "and are the least certain thing here.** Six peoples of the Yunnan border — the "
            "Lisu, Lahu, Jingpo, Wa, Nu and "
            "Derung — were reached by Protestant missions between about 1900 and 1935 and "
            "large parts of them have been Christian ever since. Nujiang, the Lisu "
            "prefecture on the Myanmar border, contains what is usually described as the "
            "first majority-Christian county in China. Nobody counts them: the shares used "
            "here are Joshua Project's, a missionary organisation's estimates, applied to "
            "the census's own count of each nationality. **They probably run high** — for "
            "the Lisu, Joshua Project says 80% where the figure usually reported as the "
            "official one works out to about 43% — so read this layer as *where*, "
            "confidently, and *how many*, within a factor of about two. "
            "**And half a million Christians are missing from it, in a place this map "
            "cannot put them.** The A-Hmao and Gha-Mu of northwestern Guizhou, converted by "
            "the Pollard mission in the 1900s, are as Christian as the Lisu — but the "
            "census counts them as *Miao* along with nine million other people spread over "
            "five provinces, and there is no way to pull them out. They are in the grey. "
            "**The vintage is split.** Where people are comes from the 2000 census, the "
            "last one whose county-level ethnic tables are public; how many there are comes "
            "from 2010. Twenty-five years is long enough that the cities have grown and "
            "these dots do not know it, and the migrant communities of the coast — a few "
            "thousand Uyghurs and Dai in Zhejiang and Shanghai — are placed by a pattern "
            "that predates them."),
        how=("no census question; ethnicity for the minorities, a pooled survey by province for "
             "everyone else"),
        fill="from the 2000 census's ethnicity table",
        grain=("counties, 470,000 people on average; the survey layer's shares are provincial, so "
               "Buddhism and Protestantism vary between provinces and not within them"),
        # `gap` (§6.12). It said "Han majority not shown" until §14.14 drew them, then
        # "97% of these dots say only that somebody was counted" until the CGSS layer of
        # 2026-09-08 took the grey from 97.5% to 91.6%. The failure mode it guards against
        # has never moved: this is still the country where a reader is most likely to read
        # the grey as irreligion, which is exactly what it is not.
        gap=("a religion for 92% of these dots; with no census question they say only that "
             "somebody was counted"),
        counts=_cn_counts,
        # Counts are on the GB/T 2260 county adcode; the Kontur hexes carry no adcode, so
        # sources/cn_geo.py assigns and clips every hex and writes the `unit` this reads.
        # Russia's, Kenya's, Serbia's and Lithuania's wiring exactly.
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "cn" / "cn_grid_3km.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_cn_place_weight,
        note="**100% of the census population is drawn and 97.5% of it is `unknown`** — "
             "rewritten 2026-09-07, spec §14.13, after CFPS refused the data access §14.7 "
             "had planned a Han Buddhist share around. Nothing is measured: the derived "
             "rows are §14.5's ethnic derivation and the modelled rows are §14.9's "
             "fractional Christian share over six Yunnan border peoples, whose "
             "coefficients are Joshua Project's and whose three admission conditions are "
             "argued in taxonomy/cn2000.py. **The gate is NOT a threshold over Joshua "
             "Project's own numbers** — that was tried, and it returns 121 million Han "
             "Christians in the Wu- and Min-speaking southeast against CGSS's 1.7% "
             "nationally, because an interested source's bias is not uniform across its "
             "own rows. Selection is made on evidence outside the missionary literature and "
             "only the coefficient comes from JP. "
             "The output check, which is all this country has: the six peoples come to "
             "~876,000 Christians, essentially all in Yunnan, against a province usually "
             "reported to hold over a million Protestants — consistent, and leaving room "
             "for the A-Hmao Miao and Han that this map cannot place. "
             "Structure is the 2000 census at "
             "county (Harvard's `chinacensus` dataverse, CC0, 2,859 counties — the table "
             "reaches township and that is deliberately NOT used, per §14.5's ceiling); "
             "magnitudes are the 2010 census by province (NBS table 1-6), per §3.4. The "
             "2020 edition of that table exists only as a JPEG scan. "
             "**The county join is by name and it is the fragile part**: the census carries "
             "romanised names and no codes, DataV carries Chinese names and the adcode, and "
             "2,691 of 2,859 counties resolve — 2,561 by name, 30 by code order (both lists "
             "are in GB/T 2260 order, which is the only thing separating Yining city from "
             "Yining county), 100 by a hand table of administrative changes. The 168 that "
             "do not resolve are eastern urban districts holding 69,728 drawn people, "
             "0.26%, and they are dropped rather than spread (spec §3.5). "
             "**geoBoundaries CHN ADM2 was tried and rejected** — duplicated polygons, "
             "counties abolished in the 1980s, units in the wrong province, corrupted "
             "romanisation, 59.9% match. sources/cn_geo.py has the evidence. "
             "Placement is Kontur's 3km H3 grid, 167,890 hexes; its totals reproduce the "
             "2010 census at 1.063x nationally with a per-province median of 1.048 and "
             "everything inside a factor of two, which is also what says the DataV polygons "
             "are WGS84 rather than GCJ-02 offset.",
    ),
    "et": dict(
        name="Ethiopia",
        source="2007 Population and Housing Census (CSA), U.S. Census Bureau tabulation",
        basis="self-identification",
        view=[32.9, 3.3, 48.1, 15.1],
        note_public=(
            "**Ethiopia is the sharpest religious boundary on this map, and it is a line "
            "of altitude.** The Orthodox highland and the Muslim lowland meet along the "
            "edge of the escarpment and barely mix across it: **Tigray is 95.6% Orthodox "
            "and Amhara 82.5%**, while **the Somali region is 98.4% Muslim and Āfar "
            "95.3%**. That is not a gradient. **104 of the 738 woredas are over 99% one "
            "religion** — 50 Orthodox, 54 Muslim — and there is no other country here "
            "where so much of the map is effectively single-coloured. "
            "**The Ethiopian Orthodox Tewahedo Church is 32.1 million people and it is not "
            "Eastern Orthodox.** It is *Oriental* Orthodox — out of communion with "
            "Constantinople since the Council of Chalcedon in 451, alongside the Copts, "
            "the Armenians and the Syriacs — and it is by a wide margin the largest body "
            "in that communion anywhere. Most maps colour it the same as Greek and Russian "
            "Orthodoxy; this one does not. "
            "**The Protestant south is the fastest-changing thing in the country.** "
            "*Protestant* in Ethiopia means **P'ent'ay** — the evangelical and Pentecostal "
            "churches together, the Mekane Yesus and Kale Heywet above all — and in the "
            "south it is not a minority: **Sidama is 84.4% Protestant, Gambella 70.1%, and "
            "the old SNNPR 48.4%**, against 18.5% nationally. Bensa woreda alone is 92.8%. "
            "This is a twentieth-century mission geography laid over ground the Orthodox "
            "Church never held. "
            "**Oromia is where all three meet.** The largest region, 27.0 million people, "
            "and the only one with no majority at all: **47.6% Muslim, 30.4% Orthodox, "
            "17.7% Protestant**. The most religiously mixed woredas in Ethiopia are all "
            "here or just south of it — Ale is 37% Muslim, 33% Orthodox, 29% Protestant. "
            "**Traditional religion survives in the south-west and the Borana, and nowhere "
            "else.** 2.65% nationally, but **Surima is 96.3%, Hamer 91.3%, Dasenech 81.9% "
            "and Bena Tsemay 74.5%** — the South Omo peoples — and Dire in the Borana "
            "lowlands is 75.6%. Half of everyone counted as Traditional in Ethiopia lives "
            "in 25 of the 738 woredas. **Treat the number as a floor**: the box is "
            "exclusive of the Christian and Muslim ones, and Ethiopian traditional "
            "practice commonly accompanies one of them rather than replacing it. "
            "**Catholics are 0.72% and they have one homeland and one enclave.** Erob, in "
            "the Tigrayan mountains on the Eritrean border, is **40.6% Catholic** and "
            "nothing else in the country is close; the second cluster is in Wolayta "
            "(Damot Pulasa 17.1%, Damot Gale 11.1%). Most of them belong to the Ethiopian "
            "Catholic Church, which is Eastern Catholic of the Ge'ez rite, not Latin. "
            "**Addis Ababa has an internal gradient worth zooming into.** The city is "
            "74.7% Orthodox, but Addis Ketema is **30.6% Muslim** and Kolfe Keraniyo "
            "27.7%, against Yeka's 6.8% — the old merkato quarters against the newer "
            "eastern ones. "
            "**One caution about the residual.** `Other` is 0.64% nationally but reaches "
            "**21.8% in Bore woreda** and 19.1% in Girja, both in Guji, Oromia — a "
            "concentration a six-cell question cannot explain, and most likely "
            "Waaqeffanna, the Oromo traditional religion, being recorded here rather than "
            "under *Traditional*. The census does not say, so it is drawn as it was "
            "published."),
        how="census, 2007",
        grain="woredas, 100,000 people on average",
        counts=_et_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "et" / "et_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_et_place_weight,
        note="THE COUNTS AND THE BOUNDARIES COME OUT OF THE SAME FILE, which has not "
             "happened before here. The source is not the Ethiopian statistical agency: "
             "the **U.S. Census Bureau** publishes Ethiopia's 2007 census tabulations on "
             "HDX as a geodatabase with the boundaries beside them, both keyed on "
             "`GEO_MATCH`, so the join is an identity and there is nothing to verify about "
             "it. sources.md §11b had ranked Ethiopia 2007 the largest untried African "
             "source and priced it at eleven regional PDF volumes on a moved website; none "
             "of that was needed. sources.md §11h is the general finding. "
             "**A perfect partition.** 738 woredas sum to 73,750,932 — the published 2007 "
             "census population — category by category, all six, so 100% of the tabulation "
             "is drawn. **There is no non-response cell at all**, which is unusual (§3.5) "
             "and does not mean nobody refused: it means the 2007 tabulation distributed or "
             "never published one, and nothing here can undo that. "
             "**The sentinel is `-999` and it parses as a number.** Four woredas carry it "
             "in every category; summed naively they remove 23,976 people, 0.03%, small "
             "enough to look like rounding. Masked before summing, which is how the exact "
             "partition appears. Five woredas have no data at all — three in Āfar marked "
             "'Population data not available' by USCB, which is the tell that the 2007 "
             "census itself did not fully enumerate parts of Āfar and Somali. They are "
             "dropped and cost nothing, because the national total excludes them too. "
             "**The counts are 2007 and the boundaries are 2021.** Sidama is a separate "
             "region here and was inside SNNPR in 2007, so USCB's re-cutting is real and "
             "reaches ADM1; at woreda level 418 units carry a note saying which census-era "
             "unit they came out of, and 70 census-era woredas are split across two to "
             "four modern ones. Every count is an integer and the partition is exact, so "
             "nothing is duplicated — but the per-unit split is USCB's work and is not "
             "independently checked here. The note is carried into every row of et.csv. "
             "**The census is 2007 and there has been no successor.** The 2017 census was "
             "postponed four times and abandoned; Ethiopia has not counted itself in "
             "eighteen years, and the population has since roughly doubled. Read this as "
             "the shape of Ethiopian religion, not its current size. "
             "Placement is Kontur's 400 m H3 grid, 422,726 hexes weighted by hex "
             "population, and Ethiopia needs it more than Kenya did: the 50 largest "
             "woredas are 38.4% of the land, 5.3% of the people and 75% Muslim on average. "
             "Kontur is 2023 against a 2007 census, so its total is 1.71x — expected, and "
             "used only as a within-woreda weight. Inland water needed no clipping: the "
             "eleven great lakes, Mago and Nech Sar and the Gambella reserve are separate "
             "polygons in the geodatabase and are cut out of the woredas already.",
    ),
    "cf": dict(
        name="Central African Republic",
        source="RGPH03 2003 census (ICASEES), U.S. Census Bureau tabulation",
        basis="self-identification",
        view=[14.2, 2.0, 27.6, 11.2],
        note_public=(
            "**The Central African Republic is the most Protestant country on this map.** "
            "52.3% against 29.3% Catholic — a ratio no other country here reaches, and it "
            "is the map of four mission societies that divided the country between them in "
            "the 1920s and never overlapped much afterwards. The census names none of them, "
            "so this is one undivided colour where Malawi next door would show three. "
            "**The Muslim north-east is the sharpest line in the country.** Vakaga, on the "
            "Chad and Sudan borders, is 87.4% Muslim and Bamingui-Bangoran 44.5%, against "
            "2.1% in Nana-Grébizi and 2.3% in Ouham. Ouandja commune is 96.1%. Nationally "
            "Islam is 10.5%, and the top twenty communes out of 177 hold 57% of it. "
            "**This is where the country was in 2003, and that matters more here than "
            "almost anywhere else on this map.** The war that began in 2013 displaced a "
            "large part of the Muslim population of the west and centre — Bangui's PK5 "
            "quarter, Bossangoa, Bouar, Carnot — and much of it never came back. There has "
            "been no census since RGPH03, so nothing newer exists to draw. Read this as the "
            "religious geography of the CAR immediately before that, not as it stands. "
            "**The census offers no traditional religion box at all**, where Ghana, Kenya, "
            "Ethiopia, Malawi and Benin each offer one. The form has five answers: "
            "Catholic, Protestant, Muslim, other religion, no religion. So CAR's "
            "traditional religions are not on this map — not undercounted, absent — and "
            "the two cells that would hold them tell you where they are: **`other "
            "religion` reaches 23.7% in Topia and 21% in Moboma and Baleloko, and `no "
            "religion` peaks in exactly the same communes**, all of them in Lobaye and "
            "Mambéré-Kadéï, the south-western forest, the Aka homeland and Gbaya and "
            "Ngbaka country. Two residuals with one geography, and it is the geography of "
            "the question that was not asked."),
        how="census, 2003",
        grain="communes, 21,700 people on average",
        # `gap` (§6.12): 1.50% of the census is not in the religion table and has no cell.
        gap="1.5% of the census, which was not asked, or not tabulated, and has no category",
        counts=_cf_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "cf" / "cf_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_cf_place_weight,
        note="98.50% OF THE CENSUS IS DRAWN AND THE MISSING 1.50% IS NOT A CATEGORY. The "
             "five cells partition a religion universe of 3,836,736; the RGPH03 population "
             "is 3,895,139, which the Ethnicity sheet of the same USCB workbook carries. "
             "So 58,403 people were counted and are not in this table, with no cell "
             "saying so — the data dictionary's wording, `Total population reporting a "
             "religion or belief system`, is the only place it is stated. Per commune the "
             "coverage runs 92.6%–99.9%, median 98.8%, two communes below 95%, which is "
             "evenly-spread non-response rather than a structural hole. Reported, not "
             "filled (§3.5). "
             "THE `Age-Sex` SHEET IS A 2016 ESTIMATE AND IS NOT A DENOMINATOR. Its "
             "national total is 5,052,901, 31% above the census; using it would turn a "
             "98.5%-covered country into a 76%-covered one. Only the Ethnicity sheet is "
             "the right vintage. "
             "THE COUNTS AND THE BOUNDARIES ARE THE SAME VINTAGE, which Ethiopia's were "
             "not. The religion layer keys to `CF_GEOG1_ADM3_2003`, the 2003 set, so there "
             "is no re-cutting to trust and only four communes carry a USCB note at all. "
             "The workbook also ships a 2021 boundary set with 181 communes for its "
             "Population and Displacement tables; sources/cf_geo.py asserts the layer name "
             "so the wrong one cannot be read as a join failure. "
             "THE JOIN IS AN IDENTITY AND IT IS MEASURED: 177 polygons, 177 count rows, "
             "zero keys on either side alone, zero duplicates — §11h's free join, checked "
             "on the sixth country of the series. "
             "THE PARTITION IS EXACT TO ROUNDING AND TWO-SIDED: categories minus total "
             "runs −2..+2 over all 267 rows, 131 of them exact, and the national row is "
             "−1 on 3.8 million. Nothing is one-sided, which is what independently rounded "
             "published figures look like (§9at) rather than a dropped category. "
             "Placement is Kontur's H3 grid, 34,651 hexes weighted by hex population, and "
             "CAR needs it more than Ethiopia did: the 50 largest communes are 73.0% of "
             "the land and 28.9% of the people. "
             "**AND KONTUR IS NOT INDEPENDENT OF THIS CENSUS HERE, WHICH IS NEW.** 78.5% "
             "of communes sit within ±5% of the median Kontur/census ratio, against 34.0% "
             "of Ethiopia's woredas — too tight to be independent modelling. CAR has had "
             "no census since 2003, so Kontur had nothing newer to build on and its "
             "extract is close to a constant rescale of the table being drawn. The ratio "
             "agreeing therefore proves nothing about either source. The grid is still "
             "used, because the thing it is used FOR — where inside a commune the people "
             "are — comes from settlement footprints and is its own measurement; the level "
             "is never read. sources/cf_geo.py measures this on every run.",
    ),
    "ci": dict(
        name="Côte d'Ivoire",
        source="RGPH 2021, tome 1 (ANStat), Tableaux 4.1 and 4.6",
        basis="self-identification, ordinary households",
        view=[-8.7, 4.1, -2.3, 10.9],
        note_public=(
            "**Côte d'Ivoire is the only country on any map here that counts the Harrist "
            "Church.** 140,482 people follow William Wadé Harris, the Liberian preacher who "
            "walked this coast in 1913-15 in a white robe with a bamboo cross and is "
            "usually credited with more conversions than any missionary in African history. "
            "**His route is still on the map**: the church is 2.4% in La Mé, 1.7% in "
            "Grands-Ponts and 1.6% in Agnéby-Tiassa — the southern lagoons — and a printed "
            "**0.0% in seven northern régions**, which he never reached. A century later "
            "the census can still see where he walked. "
            "**The country divides north and south almost perfectly.** Islam is 42.5% "
            "nationally and runs **95.7% in Folon and 92.3% in Kabadougou** on the Malian "
            "and Guinean border, against 13.9% in N'Zi in the centre-east — a sevenfold "
            "range. Christianity is 40.3% and does the opposite. There is no gradient in "
            "the middle so much as a line. "
            "**Traditional religion survives in one place and it is Bounkani.** 24.7% "
            "there against 2.2% nationally — Lobi and Koulango country in the north-east "
            "corner, on the Burkinabè and Ghanaian border. Every other région is under 8%. "
            "Read the figure as a floor everywhere: the box is exclusive of the Christian "
            "and Muslim ones, and in Côte d'Ivoire the same person is very often both. "
            "**And `no religion` is 12.6% but it is not what it looks like.** It is 29.8% "
            "in Tonkpi and 28.9% in Poro against **3.6% in Abidjan** — the exact inverse of "
            "where a secularising urban population would be. Some unknown part of it is "
            "traditional practice with no church and no box on the form. "
            "**Nearly one in five Ivorians is an evangelical, and that is the fastest "
            "change here.** *Autres chrétiens* went from 3.1% of the country in 1998 to "
            "20.0% in 2021 — twelvefold in a generation — while animists fell from 11.9% "
            "to 2.2% and the Harrist church shrank in absolute numbers, from 197,515 "
            "people to 140,482. The evangelical wave is largely where the animists went. "
            "**The evangelical share is the census's own figure, but its map is not.** "
            "ANStat publishes évangéliques at 18.6% nationally and gives them no geography "
            "at all, so the dots here are that national share applied evenly to every "
            "région's *other Christian* total. The size is measured; **the pattern is "
            "not** — real evangelicals are very likely more southern and more urban than "
            "this shows. "
            "**And what is left of that cell still hides a church Benin counts by name.** "
            "After the evangelicals are taken out, 559,000 people remain in *other "
            "Christian*, and the census collected *Céleste*, *Bouddhiste* and *Témoin de "
            "Jéhovah* as answers it never printed. The Celestial Church of Christ is in "
            "there, mapped commune by commune in Benin next door and invisible here."),
        how="census, 2021",
        fill="from the same census's national total",
        grain="régions, 887,000 people on average",
        # `gap` (§6.12): `ND` is 2.21% and is a non-answer, off the tree per §3.5.
        gap="2.2% who did not state a religion",
        counts=_ci_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ci" / "ci_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ci_place_weight,
        note="THE OFFICE WAS NEVER THE ROUTE. `ins.ci` is a parked cPanel page that 404s "
             "every document it used to serve; the office is now ANStat and `anstat.ci` "
             "returns 403 to every scripted request, static PDFs included. The file is the "
             "WAYBACK MACHINE's copy of ANStat's own URL — §11f's technique applied to a "
             "live host rather than a dead one. "
             "AND THE ORACLE WAS TWO CENSUSES STALE: UNSD table 28 lists Côte d'Ivoire at "
             "2014 with 22.7M people; this is the RGPH 2021 at 29.4M. "
             "THE ARCHIVE TRUNCATES, AND THE OBVIOUS CAPTURE IS THE BROKEN ONE. The "
             "2021-04 captures deliver exactly 1,048,576 bytes (2^20) with no %%EOF and "
             "PyMuPDF opens them anyway, reporting a plausible page count. sources/ci.py "
             "queries the CDX for every capture, takes the largest, and asserts the "
             "trailer. "
             "THREE TABLES, ONE VOLUME — §3.4's move. Tableau 4.6 is religion by "
             "district/région in PERCENTAGES to one decimal; Tableau 4.1 is the same nine "
             "categories nationally in COUNTS; the annex is population by région in counts. "
             "Shares from 4.6, denominators from the annex, magnitudes from 4.1, then each "
             "category rescaled so the 33 units sum to its published national count — "
             "which removes the one-decimal rounding from every national figure and leaves "
             "it only in the within-country distribution. The rescales run 0.980–1.025. "
             "TWO NESTED TIERS IN ONE COLUMN, which is Serbia's §9p in a third country: "
             "Tableau 4.6 interleaves the 14 districts with their régions and marks "
             "neither, so summing the column double-counts the country. The separator is "
             "not the layout — it is the population annex, which lists régions and the two "
             "autonomous districts and no other district. That lands on exactly 33 and is "
             "self-checking. "
             "AND `Lacs` IS PRINTED TWICE with different figures; the second is Lagunes, "
             "identifiable because its children in the table are Agnéby-Tiassa, "
             "Grands-Ponts and La Mé. Both are districts so neither is drawn, but ci.py "
             "asserts the count is still two. "
             "THE SAME DOCUMENT USES TWO THOUSANDS SEPARATORS: Tableau 4.1 groups digits "
             "with U+2009 THIN SPACE and the annex with an ordinary space, so a regex "
             "written against one silently matches NOTHING on the other and the failure "
             "reads as 'that table is not on this page'. Every line is folded before any "
             "pattern is tried. "
             "Boundaries are geoBoundaries CIV ADM2, 33 polygons against 33 units, joined "
             "by name: 30 fold directly and 3 need an alias (District Autonome d'Abidjan, "
             "District Autonome de Yamoussoukro, and `Me` for `La Mé`), each unambiguous. "
             "Placement is Kontur's H3 grid, 142,653 hexes weighted by hex population. "
             "**And unlike the Central African Republic (§9av), Kontur here IS independent "
             "of the census**: 27.3% of régions sit within ±5% of the median ratio against "
             "CAR's 78.5%, which is what a 2021 census against a 2023 grid should look "
             "like. ci_geo.py measures it on every run. "
             "THE ÉVANGÉLIQUE SPLIT IS DERIVED AND IS ANITA'S CALL (2026-09-07). Tome 1 "
             "never divides `Autres religions chrétiennes` (6,004,781, 20.5%); the "
             "RÉSULTATS GLOBAUX DÉFINITIFS does, in one sentence of prose — \"20% d'autres "
             "chrétiens, composés principalement des évangéliques (18,6%)\" — whose "
             "percentages are of the total population and reproduce that publication's own "
             "% column exactly. So 5,445,459 évangéliques is the SOURCE'S magnitude. The "
             "geography is nobody's: no publication gives them by région, so the national "
             "ratio is applied uniformly and both halves inherit the residual's shape. "
             "Those rows are `tier=derived` and may never ring (§3.10). "
             "AND THE TWO PUBLICATIONS DISAGREE about where `autres chrétiens` ends and "
             "`autres religions` begins — by 159,208 people, while agreeing on their sum "
             "to the person (6,057,832 both ways). Subtracting the évangéliques from tome "
             "1's larger cell leaves the disputed people in the REMAINDER, which is where "
             "they belong if the Résultats Globaux is right that they are not Christian, "
             "so the évangélique figure is unaffected either way.",
    ),
    "pk": dict(
        name="Pakistan",
        source="2017 Population and Housing Census (PBS), U.S. Census Bureau tabulation",
        basis="self-identification",
        view=[60.8, 23.6, 77.9, 37.1],
        note_public=(
            "**Pakistan is 96.5% Muslim and 84 of its 135 districts are over 99% Muslim, "
            "so almost the whole of this map is one colour.** That is the honest headline. "
            "What makes it worth drawing is that the remaining 3.5% is not spread thin — it "
            "is concentrated into two of the sharpest minority geographies on this map. "
            "**The Thar desert is the largest Hindu population outside India and Nepal, and "
            "it is a border phenomenon.** Umerkot district is **52.2% Hindu** — the only "
            "district in Pakistan without a Muslim majority — with Tharparkar at 43.4%, "
            "Mirpur Khas 38.7%, Tando Allahyar 34.2%, Badin 23.6% and Sanghar 21.8%. "
            "Sindh as a whole is 8.7% Hindu against a national 2.1%, and the rest of the "
            "country is essentially empty of Hindus: Punjab is 0.19%, Khyber Pakhtunkhwa "
            "0.02%. This is the part of Sindh that did not empty in 1947. "
            "**Two census cells make up that population and the map merges them.** PBS "
            "counts *Hinduism* (3.60m) and *Scheduled Castes* (0.85m) as separate answers. "
            "Scheduled Castes is a **caste** category, not a religion — Pakistan's Dalit "
            "communities, the Meghwar, Bheel and Kolhi of the Thar — and its members are "
            "Hindu. Together they are 4,444,870 people, the figure usually quoted for the "
            "country. PBS's own 2023 report says the only change from 2017 was *improvement "
            "in reporting of scheduled caste by clear differentiation between Hindu and "
            "scheduled caste*, which is a publisher describing its own 2017 split as "
            "unreliable, so the two are drawn as one. "
            "**The Christian belt is central Punjab and it is an urban and industrial "
            "geography.** Lahore is 5.1% Christian, Sheikhupura 3.8%, Gujranwala 3.6%, "
            "Sialkot 3.5%, Kasur 3.5%, Faisalabad 3.4% — and Islamabad 4.3%. Roughly half "
            "are Catholic and half belong to the Church of Pakistan, a 1970 union of "
            "Anglicans, Methodists, Lutherans and Presbyterians; the census separates none "
            "of it. "
            "**Ahmadis are counted, and the count is a floor by a large and unknown "
            "margin.** 191,737 people, 0.09%, and a third of them are in one district — "
            "Chiniot, at 4.4%, which contains Rabwah, the movement's Pakistani "
            "headquarters. Ahmadis identify as Muslim; Pakistan's constitution declares "
            "them non-Muslim and its penal code makes it a criminal offence for them to say "
            "otherwise, which is why the census prints *Qadiani/Ahmadi* as a category "
            "beside *Muslim* rather than inside it. **This map files them under Islam**, "
            "because the tree describes what people are rather than what a state's law "
            "says they may call themselves. Registering as Ahmadi carries real consequences "
            "— a separate electoral roll, and a declaration disavowing the movement's "
            "founder required to obtain a passport — and the community has organised census "
            "boycotts on that ground since 1974. Every independent estimate is several "
            "times this figure. "
            "**The blank in the north is missing data, not empty land.** Azad Kashmir and "
            "Gilgit-Baltistan — twenty districts and about 6 million people — have no "
            "religion figures at all, because PBS did not publish them for the disputed "
            "regions. Gilgit-Baltistan is also where Pakistan's Shia population is most "
            "concentrated, so the one place where the Sunni/Shia division would be most "
            "visible is the one place with no data. "
            "**And Sikhs and Parsis are invisible here, which is an artefact of the 2017 "
            "form.** It offers one `Other` cell holding 43,253 people — 0.02%, the smallest "
            "residual on this map — for the Sikhs of Nankana Sahib, the Parsis of Karachi, "
            "the Bahá'ís, the Kalasha and everyone else. **The 2023 census gives Sikhs and "
            "Parsis cells of their own**, so this is a fact about the question rather than "
            "about the country. "
            "**One place in that residual is findable on the map, and it is not the one you "
            "would guess.** Chitral is 835 per 100,000 `Other` — **forty times the national "
            "rate, and 39.8% of the whole cell for Khyber Pakhtunkhwa** — which is the "
            "**Kalasha**, the roughly four thousand people of the Bumburet, Rumbur and Birir "
            "valleys who practise Pakistan's last indigenous polytheistic religion and live "
            "nowhere else. Karachi South (129 per 100,000) is the Parsis, and Nankana Sahib "
            "(122) the Sikhs of Guru Nanak's birthplace. So these dots are one undivided "
            "colour holding several communities that do not overlap at all — which is also "
            "why the 2023 shares cannot be used to split them."),
        how="census, 2017",
        grain="districts, 1.5m people on average",
        counts=_pk_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "pk" / "pk_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_pk_place_weight,
        note="**THE TIER IS DISTRICT BECAUSE OF spec §14.4, AND THE FILE OFFERED FINER.** "
             "The USCB geodatabase carries religion at 585 fourth-order units — tehsils, "
             "talukas and thanas — and that is not drawn. PBS publishes religion **by "
             "district**; its own tehsil release (`sindh_tehsil.pdf` and its provincial "
             "siblings) carries Table 4 only, area and population and density, with no "
             "religion table at all — checked by downloading and reading it. §14.4 says no "
             "resolution finer than the state's own publication of the variable, and for "
             "this variable the ceiling is the district. It costs less than it sounds: at "
             "tehsil the map would put Lalian at 13.6% Ahmadi, at district it puts Chiniot "
             "at 4.4%, and the Thar and Punjab geographies survive intact either way. "
             "135 districts for 207.7m people is 1.54m each — coarser than everything here "
             "except Kenya's 1.01m. "
             "**A perfect partition.** The six categories sum to 207,684,626 — PBS's own "
             "published census total — and every level of the file reproduces it category "
             "by category. There is **no non-response cell at all**, as in Ethiopia (§9u); "
             "that is a fact about the tabulation and does not mean nobody refused. "
             "**No `-999` here.** Ethiopia's file used that sentinel and this one uses real "
             "nulls, so the convention is per country rather than per publisher; "
             "`sources/pk.py` asserts zero negative cells so a future release that adopts "
             "it fails loudly instead of silently subtracting. "
             "Source is USCB's transcription of PBS *Table 9, Population by sex, religion "
             "and rural/urban*, with the boundaries in the same geodatabase keyed on the "
             "same `GEO_MATCH` — the join is an identity (sources.md §11h). "
             "**The 2023 census is better data and is not obtainable.** It counts 241.5m, "
             "adds Sikh and Parsi cells and fixes the Hindu/Scheduled-Caste split — but "
             "publishes religion only by province in the reports that can be downloaded, "
             "and its district tables live on `census23.pbos.gov.pk`, which refuses "
             "connections and whose only Wayback captures are the bare root page. "
             "Placement is Kontur's 400 m H3 grid, 364,357 hexes weighted by hex "
             "population; Balochistan needs it, being 44% of the land, 6% of the people and "
             "99.3% Muslim. Kontur 2023 against a 2017 census is 1.14x with a per-district "
             "median of 1.12 — a tight band, where Ethiopia's sixteen-year gap needed a "
             "wide one.",
    ),
    "bd": dict(
        name="Bangladesh",
        source="2011 Population and Housing Census (BBS), U.S. Census Bureau tabulation",
        basis="self-identification",
        view=[88.0, 20.6, 92.7, 26.7],
        note_public=(
            "**Bangladesh is 90.4% Muslim, and the reason to look at it is the other "
            "9.6%.** It is the fourth-largest Muslim population in the world — 130.2 "
            "million, more than every Arab country combined — and on 544 upazilas of about "
            "265,000 people each the minorities resolve into three quite separate "
            "geographies rather than a thin national scatter. "
            "**The Hindus are the largest Hindu population anywhere outside India: 12.3 "
            "million, 8.5%.** Larger than Nepal's, and about the size of Ohio. They are "
            "concentrated and the concentration is old — **Dacope upazila is 56.5% Hindu**, "
            "Kotalipara 49.9%, Sulla 47.0% — running in three belts: the southwest of "
            "Khulna division around the Sundarbans, the tea districts of Sylhet, and the "
            "northwest around Dinajpur and Thakurgaon. Read the figure as a moment in a "
            "long decline rather than a steady state: the Hindu share of this territory "
            "was about 22% at partition and roughly 13.5% in 1974. "
            "**The Chittagong Hill Tracts are a Buddhist and tribal-Christian country "
            "inside a Muslim one, and nothing else on this map looks like them.** "
            "Juraichhari is **94.6% Buddhist**, Naniarchar 83.4%, Lakshmichhari 79.4%, and "
            "six Hill Tracts upazilas are majority Buddhist. The Buddhism is Theravada — "
            "the Chakma, Marma and Rakhine, and the Barua of Chattogram plain, who are "
            "among the oldest continuously Buddhist communities in South Asia. Beside it, "
            "in the same hills, is the only place in Bangladesh where Christianity is "
            "visible at all: **Ruma is 38.2% Christian and Thanchi 36.4%**, against 0.31% "
            "nationally — twentieth-century mission ground among the Bawm, Mru and Khumi. "
            "**And the `Other` cell is where the indigenous religions went.** 0.14% "
            "nationally but **15.3% in Ruma** and 7.8% in Thanchi, sitting beside the "
            "Christian and Buddhist peaks rather than instead of them. A five-box question "
            "has nowhere to put Mru, Khyang or Bawm traditional practice, so it lands here. "
            "Treat it as a floor, and as the shape of a category the census does not have. "
            "**Two things the census cannot show.** It offers no cell for Ahmadi Muslims, "
            "who are perhaps 100,000 people and have had mosques sealed and communities "
            "attacked — Pakistan's census counts them separately and this one does not, so "
            "they are inside the 130 million. And it names no Christian denomination, "
            "although roughly two-thirds of Bangladeshi Christians are Catholic. "
            "**The counts are 2011 and there has been a census since.** The 2022 census "
            "counts about 165 million and asked religion again; its upazila tables are not "
            "reachable from here (sources.md §11j). Read the shares as current and the "
            "magnitudes as a decade old."),
        how="census, 2011",
        grain="upazilas, 265,000 people on average",
        counts=_bd_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "bd" / "bd_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_bd_place_weight,
        note="THE COUNTS AND THE BOUNDARIES COME OUT OF THE SAME FILE, the third country "
             "here to do so after Ethiopia (§9u) and Pakistan (§9t). The source is not the "
             "Bangladesh Bureau of Statistics directly: the **U.S. Census Bureau** publishes "
             "BBS's 2011 census tabulations on HDX as a geodatabase with the upazila "
             "boundaries beside them, both keyed on `GEO_MATCH`, so the join is an identity "
             "— **544 polygons, 544 counted units, zero unmatched**. sources.md §11h is the "
             "general finding and §11j is the audit that picked this country out of the "
             "four left in that series. "
             "**AND THE VINTAGES MATCH, WHICH ETHIOPIA'S DID NOT.** Ethiopia is a 2007 "
             "census re-cut onto 2021 woredas, with 418 units carrying a lineage note "
             "saying which census-era unit they came from. Bangladesh's layers are "
             "`BD_GEOG_ADM3_2011` against `..._2011census`, so there is no re-cutting at "
             "all and spec §8.1 is satisfied outright. The tell is that `USCBCMNT` is empty "
             "on every one of the 617 rows — an empty lineage column is what a matched "
             "vintage looks like, and `sources/bd.py` asserts it rather than skipping it "
             "for being blank. "
             "**A perfect partition, checked two ways.** The five categories sum to each "
             "unit's own published `RLG_TPOP` on all 617 rows, and the 544 upazilas sum to "
             "144,043,696 — the published census population — category by category. So "
             "100% of the tabulation is drawn. There is **no non-response cell at all**, as "
             "in Ethiopia and Pakistan; that is a fact about the tabulation and does not "
             "mean nobody refused. "
             "**Neither null convention appears here.** Ethiopia's file uses `-999` as a "
             "sentinel that parses as a number; Pakistan's uses real nulls; this one has "
             "zero of each, so the convention is per FILE rather than per publisher. "
             "`sources/bd.py` asserts both counts at zero, which is a stronger check than "
             "masking defensively would have been. "
             "**Five categories is the shallowest question on this map attached to its "
             "fourth-largest population**, and every one of them maps to a parent or a root: "
             "Muslim to `islam` with no school (Hanafi Sunni, but the census does not say), "
             "Buddhist to `buddhism` and not `.theravada` — the same call lk2024.py makes "
             "for Sri Lanka, and taking it differently here would draw a Theravada boundary "
             "at the Bengal border that is an artefact of two ingest decisions. "
             "Placement is Kontur's 400 m H3 grid, 145,658 hexes weighted by hex "
             "population. Bangladesh is the most uniform counting geography here and needs "
             "the weight anyway, because the units that are not uniform are the Hill Tracts "
             "and the Sundarbans — the two places the minorities live. Kontur 2023 against "
             "a 2011 census is 1.199x, a band derived from twelve years of growth rather "
             "than copied from Ethiopia's or Pakistan's. "
             "**Inland water is NOT free here, unlike Ethiopia.** That file carries lakes "
             "and parks as separate polygons cut out of the woredas; this one has 544 "
             "polygons and 544 counted units, so the delta's rivers are inside them. "
             "`water.py`'s tidal clip reaches far up the Meghna estuary and Kontur is empty "
             "over open channel, which between them handle it. "
             "**And one limit at the small end.** A Kontur r8 hex is about 0.80 km² and "
             "central Dhaka's thanas are 0.8-3 km², so four of them — Adabor, Sutrapur, "
             "Kalabagan, Kotwali, 0.41% of the country — are under 60% covered by hex "
             "centroids and their dots crowd into the covered part. Left alone because the "
             "alternative is an equal share over the same 2 km², which is not better; "
             "reported in the bd_geo.py build log rather than silently accepted.",
    ),
    "my": dict(
        name="Malaysia",
        source="Banci Penduduk dan Perumahan Malaysia 2020, Jadual 7 (DOSM)",
        basis="self-identification",
        view=[99.3, 0.5, 119.5, 7.6],
        note_public=(
            "**Malaysia holds a wider religious range inside one border than any other "
            "country on this map.** Terengganu is 97.3% Muslim and Kelantan 95.5%; "
            "Sarawak, 700 km away across the South China Sea, is **50.1% Christian**. No "
            "other country here runs from one of those to the other. "
            "**Borneo is the reason to look.** Sarawak's interior districts are among the "
            "most Christian places on this map — **Tebedu 93.1%, Kapit 89.6%, Lubok Antu "
            "88.0%, Belaga 86.2%** — and Sabah adds Tambunan at 79.8% and Tenom at 70.4%. "
            "This is mission ground among the Iban, Bidayuh, Kadazan-Dusun and Murut, "
            "worked from the nineteenth century, and the largest Protestant body in the "
            "interior is the Sidang Injil Borneo, which appears on no other map here. The "
            "census names no denomination, so all of it — Catholic, Anglican, SIB, Basel "
            "— is inside one colour. "
            "**The peninsula is a different country religiously.** Islam is the state "
            "religion and constitutionally tied to Malay identity, and the Muslim share "
            "tracks the Malay one closely. Against it sit the Chinese and Indian "
            "communities the colonial economy brought: **Timur Laut (George Town) is 52.5% "
            "Buddhist**, Kampar 45.8%, Kinta 33.5% — the tin valleys and the Straits ports "
            "— while the Hindu share follows the rubber estates and the railway, reaching "
            "**21.5% in Bagan Datuk**, 17.3% in Port Dickson and 16.9% in Klang. "
            "**Two cells mean something other than what they say, and both matter.** "
            "*Others* is 0.9% and holds six named traditions at once — Sikh, Taoist, "
            "Confucian, Bahá'í, Chinese folk and animist — so **Chinese temple practice, "
            "which this map draws separately for China and Vietnam, cannot be separated "
            "here at all**. And *no religion*, 0.8%, is not a secular geography: it peaks "
            "at **35.9% in Kecil Lojing** and runs 13.4% in Rompin, 12.5% in Selangau and "
            "8.8% in Cameron Highlands — every one an Orang Asli or interior indigenous "
            "district, and none of them a city. Read it as indigenous practice with no box "
            "on the form rather than as irreligion. Kuala Lumpur, for comparison, is 0.9%. "
            "**And a third of the grey is not about religion at all.** *Religion unknown* "
            "is 0.9% and **97% male** — 67,664 men to 25 women in Perak alone. Malaysia "
            "counts its roughly 2.7 million non-citizens, overwhelmingly male labour in "
            "plantations, construction and factories, and the near-certain reading is "
            "workers counted for a headcount without the religion question being put. It "
            "is left undrawn as its own category rather than spread, because spreading it "
            "would invent religion for exactly those people in exactly those districts."),
        how="census, 2020",
        grain="administrative districts, 203,000 people on average",
        counts=_my_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "my" / "my_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_my_place_weight,
        note="SIXTEEN PUBLICATIONS, ONE FILE EACH, AND FINDING WHICH FILE IS THE WHOLE "
             "TRICK. DOSM publishes *Penemuan Utama Banci 2020* per state, and each "
             "state's download list holds about twenty-one files of which twenty are "
             "`MYLOCAL STATS` socioeconomic tables with no religion in them. The one that "
             "matters is `<STATE> JADUAL 1 HINGGA 16`, and in Perak's list it was record 21 "
             "of 21, alone on page 3. Sabah and Sarawak title theirs *State Sabah* and "
             "*State Sarawak* where the peninsular ones say *Negeri*, and each sits among "
             "27 and 40 per-district publications. sources.md §11s has the route; it needs "
             "a free eStatistik registration and a browser. "
             "**THE RECONCILIATION IS EXTERNAL, WHICH IS RARE HERE.** Within each state "
             "volume the seven categories sum to the state total and the districts sum to "
             "the state total; then all sixteen states sum, category by category, to the "
             "separately published national volume's Table 6. That last check is a "
             "different publication rather than the file agreeing with itself, and it "
             "passes to the person on all seven categories and all sixteen states. The "
             "standalone Kampar district volume was downloaded first and agrees with its "
             "row in the Perak state volume on all eight cells. "
             "**SARAWAK SHIPS AN EMPTY DECOY OF ITS OWN RELIGION TABLE**, and it is the "
             "trap worth naming: sheet `7` carries the correct title, headers, footnote and "
             "all forty district names in capitals — with every value cell blank. The real "
             "table is `7 (T)`. A reader taking the first sheet whose title matches gets a "
             "Sarawak with nobody in it while every other check still passes. Three more "
             "traps in sources/my.py's docstring, including `-` as an in-band zero (nine "
             "Sabah districts) and DOSM spacing its own `Sex : Total` marker two different "
             "ways between the state and national volumes. "
             "**RELIGION STOPS AT ADMINISTRATIVE DISTRICT.** The mukim workbook covers all "
             "1,756 sub-districts for the whole country and carries population, ethnicity "
             "and age only; the state volumes' mukim table is population and households. "
             "160 districts is the floor and there is no finer religion tabulation at any "
             "price. "
             "Boundaries are geoBoundaries `MYS ADM2`, vintage 2020 — the census year — "
             "joined by name after three renames (Kulaijaya→Kulai, Ledang→Tangkak, "
             "Nabawan/Persiangan→Nabawan) and **verified spatially**, every polygon given a "
             "state by point-in-polygon against ADM1 that must match the census. "
             "**Putrajaya is missing from ADM2 and lies entirely inside Sepang**, so it is "
             "subtracted from Sepang before being added rather than appended — appending "
             "would double-count 48.7 km². Placement is Kontur, 144,439 hexes, national "
             "ratio 1.050; 125 of 160 districts sit between 0.8 and 1.25, and the eleven "
             "outliers were checked against polygon area and are genuine Kontur/census "
             "differences rather than bad boundaries (sources/my_geo.py).",
    ),
    "es": dict(
        name="Spain",
        source="CIS barómetros 2023–26 (citizens) + INE padrón x Pew 2020 (foreign residents)",
        basis="self-identification, sample survey (citizens); nationality-derived (residents)",
        view=[-18.3, 27.5, 4.5, 43.9],
        note_public=(
            "**Spain has never asked about religion in a census, and it is one of the "
            "best-measured countries here anyway.** The CIS barómetro asks every single "
            "month and publishes the microdata free, so pooling three years gives 464,524 "
            "answers with a province code — about 8,900 per province, and by a wide margin "
            "the largest survey behind any country on this map. "
            "**But the barómetro only interviews Spanish citizens.** Its nationality "
            "variable has two values, 'Spanish' and 'Spanish and another', so the 7.4 "
            "million foreign nationals living in Spain — 15% of the country, and most of "
            "its religious variety — are outside the frame rather than under-sampled in "
            "it. They are drawn separately, from INE's count of who lives in each province "
            "and where they are from, crossed with the religious make-up of each origin "
            "country. **So half this map is what people said and half is where they came "
            "from**, and the second half is an upper bound: it cannot see anyone who "
            "stopped practising after arriving. "
            "**Catholicism is 51% and the country is more irreligious than Catholic if you "
            "count practice.** 18% of citizens call themselves practising Catholics and "
            "37% non-practising, against 13% agnostic, 16% atheist and 12% indifferent — "
            "36% who claim no religion at all, a share exceeded in Europe only by Czechia "
            "and Estonia. The Catholic map runs southwest to northeast: **Jaén 69%, "
            "Badajoz 66%, Ciudad Real 66% against Girona 41% and Barcelona 41%**, with the "
            "Basque provinces the most atheist and agnostic in Spain at about 31%. "
            "**Islam is 5.8% and the two enclaves are a different country.** Melilla is "
            "38% Muslim and Ceuta 32% — the only majority-or-near-majority Muslim places "
            "in the European Union — and after them come **Almería at 17%, Lleida 14%, "
            "Girona 14%, Tarragona 12% and Murcia 10%**, which is the intensive "
            "agriculture belt and its Moroccan workforce rather than the cities. Madrid is "
            "below the national average. "
            "**The second-largest immigration is invisible in every other account of "
            "Spain.** 630,000 Romanians and 120,000 Bulgarians make the Romanian Orthodox "
            "Church the country's third-largest religious body, and its geography is not "
            "the cities either: **Castellón is 7.8% Romanian Orthodox, Cuenca 5.8%, Lleida "
            "5.2%, Guadalajara 5.1%.** Protestantism, at 1.6%, is the opposite — Alicante "
            "3.7% and Málaga 3.4%, which is British and northern European retirement plus "
            "Latin American evangelical churches in the same provinces. "
            "**The biggest hole is one cell on the CIS form.** Everything that is not "
            "Catholicism is offered to Spanish citizens as a single box, 'a believer of "
            "another religion', with no follow-up asking which. Spanish Muslims are split "
            "back out of it using UCIDE's province figures; the rest — Spain's own "
            "evangelicals, its naturalised Orthodox, about 110,000 Jehovah's Witnesses and "
            "45,000 Jews — stays in one unnamed 2%. **And in Almería, Teruel, Ceuta and "
            "Melilla that cell is smaller than UCIDE's count of Spanish Muslims alone**, so "
            "those four provinces are drawn less Muslim than UCIDE would have them. "
            "**Within a province the dots are not scattered blindly**: people counted as "
            "foreign nationals are placed where foreign nationals live, municipio by "
            "municipio, and Spaniards where Spaniards live. It changes less here than it "
            "would elsewhere — 55.3% of Spain's foreign residents live in cities against "
            "53.5% of its citizens, because the foreign population is on the coast, in the "
            "Almerian greenhouses and in the islands as much as in Madrid and Barcelona. "
            "**About 2% of Spain is not drawn**: the citizens who declined the question."),
        how="opinion poll, 464,524 answers; foreign residents by nationality",
        grain="provinces, 940,000 people on average",
        counts=_es_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "es" / "es_municipios.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_es_place_weight,
        note="**Two sources that partition the country rather than compete for it, and "
             "finding that out was the whole ingest.** CIS's `NACIONALIDAD` variable takes "
             "exactly two values, both Spanish, so its 3-4% 'other religion' figure is a "
             "share of citizens and not of residents — which is why it looks irreconcilable "
             "with UCIDE's 5%-of-Spain Muslim estimate and is not: 4.6% of 42.4M citizens "
             "is 1.9M, and UCIDE's Spanish-citizen Muslim figure is 1.09M, inside it. "
             "**101 CIS studies pooled, 464,524 respondents, and two rejected.** CIS reuses "
             "the variable name `RELIGION` for a different question in estudios 3462 and "
             "3506, where code 1 means 'no religion' instead of 'practising Catholic'. "
             "Pooling those would have moved several hundred thousand irreligious Spaniards "
             "into the Catholic column with every total still summing; taxonomy/es2026.py "
             "checks the whole value-label signature rather than the codes, which is the "
             "only reason it was caught. "
             "**The microdata is open and the site is not.** cis.es answers a plain fetch of "
             "its catalogue with a BunkerWeb challenge page, while `/documents/d/guest/MD<n>` "
             "and `MD<n>-zip` — two naming conventions, both live — serve the zips with no "
             "key at all. §9s's KOSIS wall inverted. "
             "**Boundaries cost nothing.** GISCO LAU 2021 carries Spain's 8,131 municipios "
             "with INE's own five-digit code as `LAU_ID`, and the first two digits ARE the "
             "province, so the counting geography is derivable from the placement geography "
             "with no join and none of §8.1's failure modes. "
             "**Three checks that passed and one that is a limit.** INE's 121 nationality "
             "leaves partition its published foreign total to +0.000%; UCIDE's 52-province "
             "table sums to the national figure the report states in its own prose "
             "(1,085,593), which a mis-parsed column could not do; and the foreign-half "
             "Muslim total, 1.82M, sits 30% above UCIDE's implied foreign figure, which is "
             "the expected direction for a Pew national composition applied to migrants and "
             "is reported rather than tuned away. The limit is vintage: INE's detailed "
             "nationality series stops in **2022** and the totals are July 2026, so every "
             "province's nationality mix is rescaled uniformly (§3.4) and the post-2022 "
             "Ukrainian arrivals in particular are understated. "
             "**Placement is municipal population (§8.2), which is a population weight and "
             "not a religion one** — Almería's Muslims spread over the whole province rather "
             "than over the El Ejido greenhouses. The Observatorio del Pluralismo Religioso's "
             "7,756 geocoded non-Catholic places of worship would fix it and are a §4.4 "
             "layer, not a weight; that is the biggest upgrade outstanding here.",
    ),
    "gr": dict(
        name="Greece",
        name_in="Greece",
        source="ESS rounds 5/10/11 (citizens) + Eurostat census 2021 x Pew 2020 (residents)",
        basis="self-identification, sample survey (citizens); nationality-derived (residents)",
        view=[19.2, 34.7, 28.4, 41.8],
        note_public=(
            "**Greece has not asked about religion in a census since 1951**, and this is the "
            "first map of it that is not simply a national number painted flat. It is built "
            "the way Spain is, from two populations that between them are the whole country: "
            "**9.7 million Greek citizens**, drawn from three pooled rounds of the European "
            "Social Survey, and **759,000 foreign residents**, drawn from the 2021 census's "
            "own count of who lives in each region and where they are from. "
            "**Greece is 84% Greek Orthodox and 7% of people say they belong to no religion**, "
            "which is the second largest answer and is concentrated in Attiki, Peloponnisos "
            "and Thessalia rather than spread evenly. Dytiki Makedonia is the most Orthodox "
            "region at 95%. "
            "**The Muslim minority of Western Thrace is the one thing a survey cannot see, "
            "and it is put back by hand.** 100,000 to 120,000 people in Rodopi, Xanthi and "
            "Evros descend from the minority recognised by the 1923 Treaty of Lausanne. They "
            "are Greek citizens, they are Turkish- and Pomak-speaking, and a Greek-language "
            "national sample of 2,700 people finds almost none of them — nine in 2010 and "
            "none at all in 2020 or 2023. So the published figure is used instead, and it "
            "makes **Anatoliki Makedonia-Thraki 22% Muslim**, against 2-6% everywhere else. "
            "Without it this map would have said the historically Muslim region of Greece was "
            "its least Muslim one. "
            "**The rest of Greece's Islam is an immigration and it is everywhere.** Albanians "
            "are half of all foreign residents; after them come Pakistanis, Bangladeshis, "
            "Afghans, Egyptians and Syrians, and the result is that **every one of the "
            "thirteen regions is at least 2% Muslim** — highest in the Dodecanese and the "
            "Ionian islands, where foreign residents are more than a tenth of the population, "
            "and in Attiki, which holds Greece's largest Muslim population in absolute terms. "
            "**The Catholics of the Cyclades survive in the data.** Notio Aigaio comes out "
            "10% Catholic, against 1% nationally — Syros, Tinos and Naxos, where Latin-rite "
            "communities have been continuous since the Venetian period. That share rests on "
            "a few dozen respondents and should be read as 'a lot, here' rather than as a "
            "number. "
            "**And Mount Athos is drawn as itself.** The monastic republic is a separate "
            "statistical region of 1,744 people that no survey will ever sample; only "
            "Orthodox monks may live there, and it is drawn accordingly rather than from "
            "mainland Macedonia's mixture. "
            "**What the sources cannot do.** The survey offers eight denominations and no "
            "atheist or agnostic option, so everyone who reports no religion lands in one "
            "category and Greece has nothing on the `secular` node at all. The immigrant half "
            "counts people as their country of origin's religion, which is an upper bound: it "
            "cannot see anyone who stopped practising, or converted, after arriving — and for "
            "Albanians, who are documented to have adopted Orthodox identity in Greece in "
            "large numbers, that limitation is doing real work."),
        how="survey, 7,885 people; foreign residents by nationality",
        grain="regions, 750,000 people on average",
        counts=_gr_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "gr" / "gr_lau.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_gr_place_weight,
        note="**Two halves of one census table, which is why they partition exactly.** "
             "Eurostat's `cens_21ctz_r3` — 2021 census, population by citizenship at NUTS 3, "
             "221 citizenships, keyless JSON — publishes `NAT` and `FOR` alongside its named "
             "countries, so the denominator of the citizen half and the numerator of the "
             "foreign half are rows of the same file. 10,482,482 people against ELSTAT's own "
             "census total of 10,482,487, and the 200 named citizenships cover 99.84% of the "
             "foreign population. **It is EU-wide and is the right starting point for any "
             "future EU country; Spain's foreign half should be moved onto it.** "
             "**The ESS API is open and nobody says so.** `ess.sikt.no` is an SPA returning "
             "the same 1,070-byte shell for every path, and its `/env.js` names the backend "
             "in one line: `api.nsd.no/graphql`, which answers ANONYMOUS queries including "
             "server-side cross-tabulation. A country's religion-by-region table is one "
             "request and no microdata moves — which matters, because the portal's own "
             "download flow does require an account. Three gotchas, all in sources/gr.py: "
             "`breakVariables` takes variable NAMES not the UUIDs the search returns; its "
             "type is `[String!]!` and anything looser is rejected; and `region` across all "
             "30 countries trips `E204TooManyCategoriesInVariable`, which `byVariables: "
             "[\"cntry\"]` avoids where every documented form of `subsetJson` returns a bare "
             "422. "
             "**`ctzcntr` is what stops the halves double-counting.** Unlike Spain's CIS, ESS "
             "does sample non-citizens — and badly, and worse over time: 203 of 2,713 "
             "respondents in round 5, 83 of 2,800 in round 10, 87 of 2,757 in round 11, "
             "against a true 7.2%. Restricting to citizens removes the overlap and the "
             "undercount at once. Greece is in rounds 1, 2, 4, 5, 10 and 11 and the first "
             "three have no `region` variable at all, so three rounds pool to 7,885 citizen "
             "respondents. "
             "**Two traps in reading the sources, both of which fail silently.** ESS labels "
             "the Greek regions in Latin in round 5 and in Greek in rounds 10-11, so pooling "
             "on labels rather than codes splits all thirteen regions in two; and the "
             "Eurostat geo dimension holds every NUTS level in one column, so a prefix filter "
             "counts the same people four times and reported Greece at 41.9 million. "
             "**The cross-check is the reason to believe it.** Muslims come out at **5.08% "
             "of Greece** against Pew's own independent 2020 country estimate of **5.12%** — "
             "from Eurostat counts, Pew origin compositions and one minority figure, none of "
             "which is Pew's Greece row. That is also what settles the Albanian coefficient: "
             "Albania is 49% of Greece's foreign residents, Pew puts it at 59% Muslim, the "
             "literature says Albanians in Greece are more Orthodox than that, and excluding "
             "them gives 2.31% — less than half. The undocumented adjustment would have been "
             "the error. "
             "**Boundaries were free and the join was one zero-pad.** GISCO's LAU bundle "
             "ships both the 6,137 Greek polygons and the workbook mapping each to its NUTS "
             "3; Excel had stripped the leading zero from every LAU code in regions 01-09, "
             "which is 644 of them and reads as 'the workbook is missing rows'.",
    ),
    "pt": dict(
        name="Portugal",
        source="Censos 2021 (INE)",
        basis="self-identification, voluntary question, people aged 15 and over",
        view=[-9.6, 36.9, -6.1, 42.2],
        note_public=(
            "**Portugal is 80.2% Catholic — the highest share of any country on this map "
            "that asks the question directly** — and it is drawn on 3,092 freguesias, "
            "about 2,800 answering people each, which is fine enough that the exceptions "
            "are individual villages rather than regions. "
            "**The country has a gradient and it runs north to south.** The Azores are "
            "91.6% Catholic and the Norte 88.1%; Grande Lisboa is 68.4% and the Península "
            "de Setúbal 65.3%, where a quarter of people report no religion against 8.7% "
            "in the Norte. That is the older split — a rural, clerical north against the "
            "latifundia south, where the Church was weak long before the 20th century — "
            "and it is still the strongest pattern in the data. "
            "**The most striking thing on the map is 30 km of the Alentejo coast.** In São "
            "Teotónio, 17.1% of people are Hindu; in neighbouring Longueira/Almograve, "
            "17.1% are Buddhist, 9.3% Hindu and 6.2% Muslim, and only 43.7% Catholic — the "
            "least Catholic freguesia in Portugal. This is the intensive berry and "
            "greenhouse belt around Odemira and its South and Southeast Asian workforce, "
            "and it appeared within about fifteen years. Nothing else in Western Europe on "
            "this map looks like it. "
            "**Belmonte is the other one, and it is much older.** 49 people in one "
            "freguesia report Judaism — 1.6%, against 0.03% nationally, and the highest "
            "Jewish share in the country by a wide margin. They are the descendants of the "
            "crypto-Jewish community that kept practising in secret for roughly five "
            "centuries after the forced conversion of 1497 and returned openly only in the "
            "1970s. A census that publishes single people at this grain is what makes 49 "
            "of them visible at all. "
            "**Two immigrations show cleanly.** The Orthodox are Ukrainian, Romanian and "
            "Moldovan and their geography is the Algarve rather than Lisbon — 3.2% of "
            "Algarve answers, 8.5% in Almancil — because they came for the tourism labour "
            "market. And Lisbon's Muslims are concentrated: Santa Maria Maior, the old "
            "Mouraria, is 18.2% Muslim against 0.4% nationally. "
            "**What the source cannot show.** One cell holds every Protestant and "
            "Evangelical from Lusitanian Anglicans to Brazilian Pentecostals; one holds all "
            "Muslims, although Portugal's community is substantially Ismaili from "
            "Mozambique and the Imamat's seat is in Lisbon. "
            "**And what is not drawn.** Children were not asked — the question covers "
            "people 15 and over — and 230,000 more, 2.6% of that universe, declined and "
            "were removed by INE rather than published as a category. So every share here "
            "is a share of the adults who answered, and about 15% of Portugal is absent "
            "from this map entirely."),
        how="census, 2021, voluntary, ages 15 and over",
        grain="freguesias, 2,800 people on average",
        counts=_pt_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "pt" / "pt_freguesias.gpkg",
        place_unit=lambda g: g["kod"].astype(str),
        note="**THE CLEANEST INGEST IN THE PROJECT, AND IT NEEDED ONE GET AND NO NEW "
             "DOWNLOAD FOR THE BOUNDARIES.** INE indicator `0012311` is 7.4 MB of JSON with "
             "no key and no wall, and GISCO LAU 2021 — on disk since Poland (§9e) — carries "
             "Portugal's LAU as the freguesia with a six-digit `LAU_ID` that is INE's own "
             "`geocod` character for character. **3,092 counted units, 3,092 polygons, zero "
             "unmatched either way, and the names agree on all 3,092** once accents are "
             "folded. Every other country here has paid for its join; this one did not. "
             "**The find was the catalogue, not the table.** `xml_indic.jsp?opc=3` is the "
             "endpoint that looks like INE's catalogue and it is a trap — 326 recently "
             "updated indicators, zero hits for religion. `opc=2` is the real one: 13,098 "
             "indicators, 21 MB, with `geo_lastlevel` per row, so 'which Portuguese census "
             "tables reach the freguesia' is a string search. An earlier probe of this "
             "office concluded Portugal published nothing on religion and was one digit "
             "away from the answer (sources.md §11k). "
             "**A perfect partition with no suppression at any level**, checked per category "
             "at both drawn tiers: largest discrepancy zero. "
             "**The universe is the caveat and INE does not flag it in the table.** The 11 "
             "categories sum to the published total exactly, which reads as a mandatory "
             "question; it is not one. The 15+ population is 9,011,878 and this table holds "
             "8,781,900, so 229,978 people — 2.55% — declined and were removed from the "
             "denominator rather than given a cell. Guyana's §9r footnote problem without "
             "the footnote, found by differencing against indicator 0011609; sources/pt.py "
             "asserts the gap so a future vintage cannot change it silently. "
             "**Placement is uniform within the freguesia (§8.2), which is the one thing "
             "left undone.** The median freguesia is 16.5 km² and that is fine, but the "
             "Alentejo units run to 863 km² at single-digit people per km², so dots there "
             "spread across empty cork forest. Kontur would fix it as it did for Kenya and "
             "Ethiopia; the tier is already fine enough that this is an improvement rather "
             "than a correction. "
             "**The Azores and Madeira are drawn** and sit outside the default view.",
    ),
    "xk": dict(
        name="Kosovo",
        source="Census 2024 (Kosovo Agency of Statistics)",
        basis="self-identification",
        view=[20.0, 41.85, 21.8, 43.25],
        note_public=(
            "**The four municipalities in the north are estimated, not counted.** Kosovo's "
            "Serbs largely refused the 2024 census, and there the result was not a thin "
            "count but a misleading one: Zveçan returned 434 people and Zubin Potok 763, in "
            "Serb-majority municipalities of several thousand, and because whoever did "
            "answer was disproportionately not Serb, three of the four came out as majority "
            "Muslim. That is not what those places are. So the missing people are put back "
            "from **the statistics agency's own estimate** — its ethnicity table published "
            "*with estimation* restores 16,949 people across the four, 96.6% of them Serbs — "
            "and they are drawn as Orthodox. **Those dots are an inference from ethnicity, "
            "not a count of anybody's answer**, and the confidence control removes them. "
            "The 2011 census did not enumerate these four at all. "
            "**Everywhere else the map is straightforward and the country is 93.5% "
            "Muslim** — the highest Muslim share in Europe, Hanafi Sunni, with a Sufi tekke "
            "tradition in Gjakovë and Prizren that no census category separates. "
            "**The Catholics are the interesting minority, and they are Albanian rather "
            "than foreign.** 1.75% nationally, but **16.8% of Klinë and 14.6% of "
            "Gjakovë** — the Catholic Albanians of the Dukagjin plain in the west, who "
            "never converted under Ottoman rule — against 0.07% in Gjilan in the east. "
            "Mother Teresa came from this community. "
            "**The Orthodox that were counted are almost entirely in the enclaves.** "
            "Partesh is 99.5% Orthodox, Ranillug 94.5%, Shtërpcë 75.1%, Graçanicë 46.5% — "
            "Serb municipalities created after 2008, each a few thousand people, and each "
            "sitting inside an otherwise Muslim country. Those are real measurements, "
            "unlike the north's. "
            "**Only 0.50% report no religion — the lowest share of any country on this "
            "map**, and a third the number who declined to answer at all. Those 23,718 "
            "refusals are not drawn; 40% of them are in Prishtinë."),
        how="census, 2024",
        fill="from the agency's own estimate for the four northern municipalities",
        grain="municipalities, 41,000 people on average",
        counts=_xk_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "xk" / "xk_grid_400m.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_xk_place_weight,
        note="**THE COUNTS ARE CLEAN AND THE COUNTRY IS NOT, AND THOSE ARE SEPARATE "
             "FACTS.** ASK's `census2024_10.px` is an open PxWeb table — 10 KB, no key, no "
             "wall — and it reconciles exactly: six categories partition every unit, and "
             "the 38 municipalities sum to the national row category by category with a "
             "gap of zero. 21 cells come back blank and the exact partition is what proves "
             "they are true zeros rather than Lithuania's disclosure control (§9q); "
             "sources/xk.py asserts that rather than assuming it. "
             "**The catalogue walk nearly missed it for a dull reason.** askdata's PxWeb "
             "root returns `dbid` where every other PxWeb here returns `id`, so a walker "
             "keyed on `id` reads the root as nameless nodes and descends into none. "
             "Moldova's does the same and looked equally empty. "
             "**KOSOVO IS THE HOLE IN THE GISCO LAU FILE.** That file covers the EU27 plus "
             "the candidates — AL, BG, CH, IS, MK, NO, RS are all in it — and Kosovo is the "
             "one Balkan country it does not carry, because the EU has no agreed status "
             "for it. geoBoundaries XKX ADM2 supplies the 38 instead, joined by name: "
             "Albanian nouns have definite and indefinite forms and ASK writes one while "
             "geoBoundaries writes the other, so ten differ by a single final vowel. "
             "Stemming that away plus five explicit aliases gives 38/38. "
             "**The independent check measures the boycott instead of assuming it.** §9p "
             "verifies a name-join by requiring every unit's population ratio to sit in a "
             "tight band. Kosovo cannot pass that and should not: Kontur 2023 knows about "
             "people the 2024 census did not reach, so the 34 enumerated municipalities sit "
             "at a median 1.06x while Leposaviq, Zubin Potok and Zveçan come out at 3.2x, "
             "6.6x and 12.4x. The band is asserted on the 34 and reported on the four — the "
             "boycott confirmed by a second, unrelated source rather than by a news story. "
             "**North Mitrovica is the one it cannot see**, at 1.11x, because "
             "geoBoundaries splits Mitrovica along the Ibar through the middle of one "
             "continuous city and the north bank's hexes fall to the southern municipality. "
             "**THE FOUR NORTHERN MUNICIPALITIES ARE DERIVED FROM ASK'S OWN ESTIMATE — "
             "Anita, 2026-09-06, after one build that drew them as published.** Drawing "
             "them raw was wrong in a way no note could repair: the map itself said Zubin "
             "Potok was 89% Muslim. `census2024_63.px` is ASK's ethnicity table *with "
             "estimation*, identical to the enumerated one everywhere except these four, "
             "where it restores 16,949 people of whom **16,369 — 96.6% — are Serbs**. There "
             "is no religion-with-estimation table, so the step is Serb -> Orthodox: spec "
             "§14.5, and Kosovo passes its three tests more cleanly than China does. The "
             "Serb/Croat/Bosniak distinction *is* a religious boundary drawn over a common "
             "language, which is condition one; Kosovo's Serbs are not a religiously mixed "
             "group, which is condition two, and `sources/xk.py` asserts the addition stays "
             "above 90% one ethnicity so it cannot quietly stop being true; and both tables "
             "are per municipality, so nothing is spread finer than it was published. "
             "**Every derived row is `tier=\"derived\"`** and §7a's control strips them in "
             "one click. The enumerated rows are untouched — the derivation adds the missing "
             "Orthodox rather than restating what was counted — and the 580 non-Serb people "
             "in the estimate are left undrawn, because nothing says what they are and the "
             "enumerated composition of these four is precisely what is not representative. "
             "**No earlier census helps**: 2011 leaves the same four null and counts fewer "
             "Orthodox nationally (25,837), 1991 was boycotted from the Albanian side, and "
             "1981 — the last with full participation — asked nationality rather than "
             "religion, on pre-2008 boundaries. "
             "**Both censuses are in the table**, 2011 and 2024; only 2024 is drawn. "
             "Placement is Kontur's 400 m H3 grid, 9,258 hexes, because 38 municipalities "
             "over 10,900 km² average 287 km² and Kosovo's people are in the "
             "Prishtinë-Ferizaj-Prizren corridor rather than on the Sharr.",
    ),
    "ba": dict(
        name="Bosnia and Herzegovina",
        source="Census 2013 (Agency for Statistics of Bosnia and Herzegovina)",
        basis="self-identification",
        view=[15.6, 42.5, 19.7, 45.35],
        note_public=(
            "**Three religions and almost nothing else — 96.6% of the country is Muslim, "
            "Orthodox or Catholic**, and the census offers no fourth religious box. That "
            "is not a thin form so much as an accurate one: in Bosnia religion and "
            "nationality are near-substitutes, Bosniak with Muslim, Serb with Orthodox, "
            "Croat with Catholic, and a question about either returns much the same "
            "answer. So this map is legible at a glance and tells you almost nothing "
            "about belief or practice. "
            "**At 50.7% Muslim it is the second most Muslim country in Europe** after "
            "Kosovo, and the only one on this map where three large religions meet. "
            "**The striking thing is how completely separated they are.** Most "
            "municipalities are not mixed at all: Bužim is 99.7% Muslim, Posušje 99.8% "
            "Catholic, Ribnik 99.5% Orthodox, and 84 of the 142 units are over 90% one "
            "religion. **That geography is not ancient — it was made between 1992 and "
            "1995**, and this census is the first since. Srebrenica, 73% Muslim in 1991, "
            "returns 55% Muslim and 45% Orthodox here. "
            "**The exceptions are worth finding.** Brčko in the north is the only "
            "genuinely three-way unit — 44% Muslim, 35% Orthodox, 21% Catholic — and it "
            "is the district that was placed under international arbitration precisely "
            "because neither entity could be given it. Mostar is split 50/46 Catholic and "
            "Muslim across the Neretva, and the central Bosnian valley towns — Vitez, "
            "Busovača, Kiseljak, Novi Travnik, Jajce — each sit near half and half. "
            "**Irreligion is a Sarajevo phenomenon and it is small.** 1.1% nationally, "
            "the lowest of any European country here, but 8.6% in Centar Sarajevo and "
            "7.1% in Novo Sarajevo. There is no 'no religion' box at all — the only "
            "irreligious answers are the positions *atheist* and *agnostic* — so that "
            "1.1% is a floor rather than a measurement. "
            "**The 2013 results are disputed and the dispute is about who counts as "
            "resident, not about religion.** Republika Srpska's statistical institute "
            "rejected BHAS's treatment of people living abroad and publishes lower "
            "figures for its own entity. The numbers here are BHAS's, which are the ones "
            "the state, Eurostat and the EU use."),
        how="census, 2013",
        grain="municipalities, 25,000 people on average",
        counts=_ba_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ba" / "ba_grid_400m.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ba_place_weight,
        note="**§11c FILED THIS COUNTRY AS 'REACHABLE, NOT QUICK' AND BOTH HALVES WERE "
             "TRUE WITHOUT EITHER MATTERING.** It recorded `popis.gov.ba` as a React SPA "
             "with no data endpoint and the results as 12-19 MB PDF books, then stopped. "
             "§12's grep-the-bundle rule was run here and returned **zero `/api` routes in "
             "all 125 KB** — the correct negative, not a failed search, because there is "
             "nothing behind the SPA to serve: the books are static files under "
             "`/popis2013/doc/`. The directory listing 403s while the files themselves are "
             "open, so the one thing that looked like a wall was a statement about a "
             "directory. **A 403 on a directory is not a 403 on its contents**, and the "
             "file was one guessed path away for a year. "
             "**The reconciliation is as good as any source here has had.** Table 4.3 is "
             "twelve printed pages; its eight categories partition every municipality "
             "exactly, and the 142 municipalities sum to the published national row "
             "category by category with a gap of **zero** — no suppression, no rounding, "
             "single people published. "
             "**The parse trap that would have shipped quietly is the neighbouring "
             "table.** 4.1 (entity) and 4.2 (canton) sit immediately before 4.3, use "
             "identical headers and an identical three-row-per-unit structure, and a "
             "generous page window swallows all three — summing to **5,583,946 against a "
             "country of 3,531,159** while still passing a per-row partition check, "
             "because each of those rows is internally consistent. Only the national total "
             "catches it. The window is pinned and the caption asserted. "
             "**And geoBoundaries BIH ADM3 is dirty in four ways, with 142 units against "
             "the census's 142** — the coincidence that makes §9s's missing-unit trap look "
             "like a clean join. One polygon is named **`Republika Srpska`** and is "
             "actually Višegrad; **`Novi Grad` appears twice** and the two are 130 km "
             "apart, so a name join is wrong about Novi Grad Sarajevo's 118,553 people; "
             "`Kupres` and `Kupres (BiH)` are the RS and Federation halves with the "
             "labels reading backwards; and `Kupra na Uni` is a typo for Krupa na Uni. "
             "All four are repaired in `sources/ba_geo.py` **before** any matching, each "
             "with a geometric assertion beside it, so the join itself stays a plain name "
             "join that either works completely or fails completely — 142/142, both ways. "
             "**The Kontur check had to have its band measured rather than inherited** "
             "(§9u): the counts are 2013 and the surface 2023, and BiH emigrated heavily "
             "in between, so the ratio sits at a median of **0.94x** rather than near 1. "
             "What the check tests is that the ratio holds *together* — 141 of 142 units "
             "inside median/3 to median×3, with only Usora outside.",
    ),
    "li": dict(
        name="Liechtenstein",
        source="Volkszählung 2015 (Amt für Statistik)",
        basis="self-identification",
        view=[9.41, 47.03, 9.68, 47.29],
        note_public=(
            "**The smallest country on this map and one of the most finely counted** — "
            "37,622 people over 11 communes, about 3,400 each, which is finer per head than "
            "most of Europe here. The census publishes single people: Planken's one member "
            "of an 'other Christian' church is a dot. "
            "**73.4% Roman Catholic, and it is the state church** — Article 37 of the "
            "constitution names it, which makes Liechtenstein one of the last places in "
            "Europe where that is literally so. Disestablishment has been debated since 2012 "
            "and has not happened. "
            "**Only 7.0% report no religion, about a fifth of Switzerland's share twenty "
            "kilometres away** — the lowest in Western Europe on this map. That gap between "
            "two neighbouring Alpine countries is the most striking thing here, and it is "
            "not a measurement artefact: both asked the same kind of question. "
            "**Islam is 5.9% and it is the guest-worker migration**, Turkish and Bosnian, "
            "settled around the industrial communes: **Eschen 11.4% and Gamprin 8.0% "
            "against Planken 0.2% and Schellenberg 1.2%**. The citizenship split is the "
            "sharpest in the table — **13.1% of foreign residents against 2.2% of "
            "Liechtenstein citizens** — and the same split runs the other way for "
            "Catholicism, 84.0% of citizens against 52.6% of foreigners. A third of the "
            "country holds a foreign passport. "
            "**The form separates Reformed from Lutheran**, which almost nothing else on "
            "this map does: two state-recognised Protestant churches, one Swiss-facing and "
            "one Austrian, at 2,365 and 447 people. "
            "**There is no Jewish box on the form at all**, so Liechtenstein's Jews are "
            "inside 'other religious communities' and the country stays unlit when Judaism "
            "is selected — the question was not put. **3.3% did not state a religion** and "
            "are not drawn."),
        how="census, 2015",
        grain="communes, 3,400 people on average",
        counts=_li_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "li" / "li_grid_400m.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_li_place_weight,
        note="**FOUND IN SWITZERLAND'S CATALOGUE, WHICH IS THE REUSABLE PART.** "
             "`ckan.opendata.swiss` carries the Liechtenstein statistics office alongside "
             "BFS, so the single search that solved Switzerland (§9ad) returned this too. "
             "Nobody had looked: §11c and §11k both swept Europe and neither mentions "
             "Liechtenstein. **A national open-data portal may index a neighbour**, and a "
             "microstate is exactly the kind of country a sweep walks past. "
             "**THE BOUNDARIES WERE ALREADY ON DISK AND THE JOIN IS ELEVEN EXACT STRINGS.** "
             "GISCO LAU 2021 — §9e's file, downloaded for Poland — carries all 11 communes "
             "with the names the census uses, so there is no folding, no stemming and no "
             "alias table. Liechtenstein is absent from the EU-27 correspondence *workbook* "
             "and present in the boundary *shapefile*, which is the same distinction "
             "Switzerland turns on. "
             "**ITS json-stat2 EMITTER IS BROKEN AND NOTHING SAYS SO.** Asking for "
             "`json-stat2` returns HTTP 200, a well-formed document declaring "
             "`size: [1,12,1,1,12]` — 144 cells — and a `value` array holding **one "
             "element**. `json-stat` and `csv` both answer correctly, so this is one "
             "serialiser rather than a wall. §5a again: *a 200 with a valid-looking envelope "
             "is not a download; check the payload against the shape the same response "
             "declares.* `filter: \"all\"` with `values: [\"*\"]` is also ignored here and "
             "has to be an explicit item list — the second PxWeb server in two days with "
             "non-portable selection semantics, after BFS's empty query dropping the "
             "geography dimension. "
             "**THE COMMUNES ARE NOT CONTIGUOUS, WHICH IS WHY THE GRID IS NOT OPTIONAL.** "
             "Liechtenstein divides its high alpine pasture among the valley communes as "
             "exclaves: **Vaduz is six separate polygons, Schaan four, Balzers and Planken "
             "three**, and seven of the eleven are fragmented. Nobody lives in the detached "
             "pieces — summer grazing above 1,500 m — so §8.2's equal share would scatter a "
             "third of Vaduz's dots onto an empty mountainside. Kontur's 171 hexes fix it. "
             "**The `-` cells are true zeros and the partition proves it**, which is "
             "Kosovo's argument rather than Lithuania's: there is no disclosure threshold "
             "here at all, so a blank cannot hide a small number. "
             "**Vintage: 2015 is the last one.** The table offers 2010 and 2015 and nothing "
             "since; Liechtenstein's later population statistics are register-based and "
             "carry no religion.",
    ),
    "ch": dict(
        name="Switzerland",
        source="Volkszählung 2000 structure on Strukturerhebung 2024 canton totals (BFS)",
        basis="self-identification; current magnitudes, 2000 composition",
        view=[5.8, 45.75, 10.6, 47.85],
        note_public=(
            "**This map is 2024 in size and 2000 in shape, and the difference matters.** "
            "Switzerland last asked everybody about religion in the 2000 census. Since 2010 "
            "the question lives in a sample survey that publishes eight categories and stops "
            "at the canton, so the choice is between the right detail on the right places a "
            "quarter-century out of date, and the right year on twenty-six units. Both are "
            "used: the survey says how many, the census says where and which church. "
            "**Nothing here is a count.** "
            "**What has actually happened is the fastest religious change on this map.** In "
            "2000 Switzerland was 41.8% Catholic and 33.0% Reformed, with 11.1% reporting no "
            "religion. It is now 30.0%, 18.7% and **36.8%** — no religion is the largest "
            "single answer in the country, and it grew more than threefold inside one "
            "generation. "
            "**The confessional map underneath it is five hundred years old and still "
            "legible.** The Reformation split Switzerland canton by canton and the line has "
            "barely moved: **Uri and Appenzell Innerrhoden are 68% Catholic, Valais 62%**, "
            "against **Bern at 14% and Basel-Stadt at 13%**. At commune level it is sharper "
            "still — Muotathal 84% Catholic, Poschiavo 82%, against the Emmental villages of "
            "Sumiswald and Lützelflüh at 63% Reformed. Appenzell was partitioned into two "
            "half-cantons over religion in 1597 and the two halves are still 68% Catholic and "
            "overwhelmingly Reformed respectively. "
            "**Irreligion is urban, and it is Protestant cantons that went furthest.** "
            "Basel-Stadt is 60%, Neuchâtel 57%, Geneva 50%, against 22% in Uri and 18% in "
            "Appenzell Innerrhoden. "
            "**The 6.0% Muslim population is industrial rather than metropolitan** — "
            "Böttstein 24%, Gerlafingen 23%, St. Margrethen 21%, small towns along the Aare "
            "and the Rhine rather than the big cities — and it is Balkan and Turkish. **The "
            "Hindus are Tamil**, 46,000 of them and more numerous than Switzerland's Jews and "
            "Buddhists combined, from the Sri Lankan asylum migration of the 1980s; they show "
            "up in Solothurn and Emmental factory towns. And **Möhlin and Magden in the Fricktal "
            "are 15% Christ Catholic** — the Old Catholic church that broke with Rome in 1871 "
            "over papal infallibility, which survives as a public-law church in a handful of "
            "Swiss communes and almost nowhere else. "
            "**A Swiss dot is an adult.** The survey asks people aged 15 and over living in "
            "private households, so children, and people in institutions, are outside the map "
            "rather than inside it — as in Brazil, Chile and Portugal."),
        how="census, 2000, resized to 2024 survey totals",
        fill="from the 2000 census",
        grain="communes, 3,400 people on average",
        counts=_ch_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ch" / "ch_grid_400m.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ch_place_weight,
        note="**§11c PRICED THIS COUNTRY AT AN HOUR OF CRAWLING AND THE CRAWL DOES NOT "
             "WORK.** It recorded that BFS PxWeb lists 650 databases with opaque ids and the "
             "human title only inside each one, so finding the religion table means "
             "enumerating all of them. That was tried: **BFS rate-limits it into total "
             "failure, 55 of 55 requests returning nothing at one per second.** "
             "`ckan.opendata.swiss` — the national portal that mirrors BFS's own catalogue — "
             "answers the same question in one call and hands over `px-x-4003000000_122` "
             "directly. **When an office rate-limits its own API, ask the national open-data "
             "portal that mirrors it**; the same call also turned up Liechtenstein's "
             "statistics office, which nobody had looked at. "
             "**THE CATEGORY LIST IS THE REASON TO DRAW A SOURCE THIS OLD.** Nineteen cells, "
             "seven of them Protestant — Reformed, Methodist, neo-pietist evangelical, "
             "Pentecostal, New Apostolic, Jehovah's Witnesses and a named Protestant "
             "residual — plus Christ Catholic separately from Roman Catholic, and Orthodox, "
             "Jewish, Islamic, Buddhist and Hindu. No other European census here can express "
             "half of that; the survey that replaced it has **eight**. "
             "**THE COMMUNE VINTAGE WAS THE REAL WORK.** The census counts 2,896 communes "
             "and GISCO LAU 2021 carries 2,242, because Swiss communes have been merging "
             "continuously; a naive join on the BFS number loses 820 of them and 606,086 "
             "people. BFS publishes the correspondence itself as an open keyless API "
             "(`agvchapp.bfs.admin.ch/api/communes/correspondances`), and three things about "
             "using it are worth keeping: `startPeriod` has to be the census date and not the "
             "following 1 January, or the 2001 Fribourg mergers vanish; territory exchanges "
             "must be excluded or the map stops being a function; and **GISCO's 'LAU 2021' "
             "for Switzerland is actually the 1 January 2020 commune state** — asking for "
             "2021 leaves fifteen communes pointing at codes the shapefile does not have. "
             "Both states are requested and whichever target exists wins, which resolves all "
             "2,896. *A boundary file named for a year is not necessarily that year's state.* "
             "**45 OF THE 2,242 LAU FEATURES ARE NOT COMMUNES** — the lake surfaces, which "
             "BFS numbers in the 9xxx block and apportions to no municipality, and the Ticino "
             "and Graubünden *comunanze*, common land held jointly. They are dropped against "
             "BFS's own register rather than by a code range, so a renumbering fails loudly. "
             "**AND ONE SILENT TRAP IN THE PxWeb CALL ITSELF**: `{\"query\": []}` returns a "
             "1.6 KB cube with the geography dimension **absent** — a 200, valid json-stat2, "
             "and no geography at all. An empty query does not mean everything on this "
             "server; `sources/ch.py` asserts the dimension survived.",
    ),
    "sk": dict(
        name="Slovakia",
        source="SODB 2021 (Štatistický úrad SR)",
        basis="self-identification",
        view=[16.83, 47.73, 22.57, 49.61],
        gap=("none, every counted person is drawn; but 7.8% of them are one other or not "
             "stated cell, and 83% of that is people who did not answer rather than people "
             "of another religion"),
        note_public=(
            "**Two halves of one state, and one of the sharpest religious contrasts on "
            "this map runs along the border between them.** Slovakia is **55.8% Roman "
            "Catholic** and Czechia, which it was part of until 1993, is 7.0%. Nothing "
            "else about the two countries diverges like this, and the line is visible on "
            "the map as an edge rather than a gradient. "
            "**The Lutherans are the national revival and they are in the middle of the "
            "country.** The Evangelical Church of the Augsburg Confession is 5.3% "
            "nationally but the historic church of the Slovak literary language, and its "
            "people are in the central uplands — Turiec, Liptov, Gemer and the Zvolen "
            "basin — rather than in the Catholic west or the Greek Catholic east. "
            "**The Greek Catholics are the east, and this is the largest Byzantine-rite "
            "Catholic population on this map after Romania's.** 218,235 people, 4.0%, "
            "concentrated in Prešov and the Rusyn villages along the Polish and Ukrainian "
            "borders. The church was suppressed outright in 1950 and restored in 1968, "
            "which is a shorter and less complete rupture than Romania's. "
            "**The Reformed are Hungarian.** The Reformed Christian Church's 85,271 people "
            "sit in a narrow strip along the southern border, in the same districts that "
            "report a Hungarian mother tongue — the same church and the same minority as "
            "the Reformed across the frontier in Hungary and in Romania's Székely Land. "
            "**Nearly a quarter of the country reports no religion, and the question it "
            "answers is not Czechia's.** Slovakia's form offers one box, *bez "
            "náboženského vyznania*, and no atheist, agnostic or believing-without-"
            "belonging option; Czechia offers all of those. So 23.8% here and 47.8% there "
            "are answers to differently-shaped questions and the gap between them is "
            "partly the form. "
            "**The grey *other or not stated* dots are mostly people who did not answer, "
            "and where they cluster they are measuring the census rather than religion.** "
            "The published municipal table folds *not ascertained* — 6.5% of the country — "
            "into one cell together with the Baptists, Adventists, Jews, Old Catholics, "
            "Hussites and Bahá'ís, and nothing at this geography separates them. It is "
            "7.8% of Slovakia nationally but runs from nothing to **58.8% in Košice's "
            "Luník IX**, with Pavlovce nad Uhom at 28.7%, Jasov at 26.9% and Bratislava's "
            "old town at 16.4%, against under 3% across the Orava and Kysuce villages. "
            "Those peaks are Roma settlements and city centres — places where a census "
            "form comes back unanswered — so read a dense patch of this colour as the "
            "limit of the count, not as an unusual faith."),
        how="a census question, 2021",
        grain="municipalities, 1,900 people on average",
        counts=_sk_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "sk" / "sk_grid_1km.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_sk_place_weight,
        note="**FOUR SWEEPS RECORDED SLOVAKIA AS WALLED AND THE ROUTE WAS NONE OF THE ONES "
             "THEY LOOKED FOR** (sources.md §11aa). §11, §11c, §11k and §11o all went at "
             "`slovak.statistics.sk`, which is still a 403, and at "
             "`datacube.statistics.sk`, which is wide open and whose **678 cubes contain no "
             "religion at all** — SODB 2021 is simply not in DATAcube. The census results "
             "live on their own GIS portal: `gis.scitanie.sk` is a public ArcGIS Server "
             "with 86 hosted services, no key, and layer 4 of `obyv_ekchar_nabo_vekskup` "
             "carries **religion counts and municipal polygons as fields and geometry on "
             "the same features**. That makes Slovakia the first country here with no "
             "join between counts and boundaries, so §12's two commonest failures cannot "
             "arise. "
             "**THE PARTITION IS EXACT AND WITNESSED THREE TIMES.** The eleven category "
             "columns sum to `spolu` sums to 5,449,270 across all 2,927 obce with a "
             "difference of zero; fetching the 8 kraje from a different layer reproduces "
             "every one of the twelve totals; and UNSD Demographic Yearbook table 28, a "
             "different publisher of the same census, agrees to the person on the total "
             "and on all nine named churches. "
             "**IT IS ALSO THE SECOND COUNTRY AFTER GERMANY WHOSE PLACEMENT IS MEASURED.** "
             "`obyv_grid_1km` is SODB 2021 on 49,969 1 km cells and sums to the census "
             "total exactly, so no Kontur extract is used. **But it is not an independent "
             "check** (§9av): the grid is the same enumeration as the counts, so the "
             "ratio band says nothing about the census and only validates the cell-to-obec "
             "assignment. "
             "**THREE ARCGIS TRAPS, ALL LIVE.** `maxRecordCount` is 2000 against 2,927 "
             "obce, so an unpaged query returns two-thirds of the country with "
             "`exceededTransferLimit` buried in the response and no error — §5a in a new "
             "disguise, and every per-row check would still pass. `supportsPagination` is "
             "not advertised and `resultOffset` works anyway. `supportedQueryFormats` says "
             "JSON only and `f=geojson` works. "
             "**AND THE PLACEMENT RULE ME_GEO.PY USES IS WRONG HERE.** Assigning a 1 km "
             "cell to the obec containing its centre credits a village smaller than the "
             "cell with the whole cell's people: Záborie, 170 people, came out weighted at "
             "1,409. Cells are split by area of intersection instead, and renormalised "
             "against the area inside Slovakia so that a border cell's people are not lost "
             "to the part of its square lying over Austria."),
    "me": dict(
        name="Montenegro",
        source="Popis 2023 (MONSTAT)",
        basis="self-identification",
        view=[18.3, 41.75, 20.5, 43.65],
        gap=("4.7% of the country, withheld by MONSTAT's disclosure control; a further 219 small "
             "settlements have their populations withheld too"),
        note_public=(
            "**The sharpest religious boundary in Europe over the shortest distance.** "
            "Montenegro is 69% Orthodox and 18% Muslim in a country of 620,000 people and "
            "13,800 km², and the two do not mix so much as sit either side of a line. In "
            "the northeast, in the Sandžak, **Gusinje is 84% Muslim, Rožaje 77%, Plav 72% "
            "and Petnjica 70%** — Bosniak municipalities against **Mojkovac at 94% "
            "Orthodox and Nikšić at 92%**, an hour's drive away. "
            "**The Catholics are two separate communities in two corners.** 3.1% "
            "nationally, but **Tivat is 16.3% and Kotor 10.0%** — the Croats of the Bay of "
            "Kotor, whose Catholicism is Venetian and six hundred years old — while "
            "**Ulcinj at 9.2%** on the Albanian border is a different community "
            "altogether, Albanian rather than Croat. The Archdiocese of Bar between them "
            "is one of the oldest sees in the region. "
            "**The Orthodox category is one box and two churches are claiming it.** The "
            "census asks 'Orthodox' and names no jurisdiction. The Serbian Orthodox Church "
            "holds the great majority and the monasteries; the Montenegrin Orthodox "
            "Church, self-declared autocephalous in 1993, is recognised by nobody and "
            "claims a substantial minority. That dispute brought down a government in 2020 "
            "and the census was taken with it still live, so these dots say 'Orthodox' and "
            "deliberately do not take a side. "
            "**There is no 'no religion' box at all.** The irreligious answers are "
            "*atheist* and *agnostic*, which are positions rather than an absence, and "
            "together they are 2.5% — concentrated on the coast and in the old capital, "
            "**Budva 4.8%, Herceg Novi 4.1%, Cetinje 4.0%**. Read that as a floor: "
            "Montenegro was never offered the question Czechia was. "
            "**Nearly 5% of the country is missing from this map and it is missing "
            "unevenly.** MONSTAT withholds any cell it judges too small to publish, which "
            "means it withholds *local minorities* — and so the places that lose most are "
            "the ones where the national majority is scarce. **Petnjica loses 29% of its "
            "people to this, Šavnik 25%, Rožaje 21%, Ulcinj 15%**, against under 1% in "
            "Tivat and Podgorica. Islam loses 10.6% of itself nationally and Orthodoxy "
            "2.7%. So every municipality here under-shows whichever religion is its own "
            "minority, and the Bosniak and Albanian ones under-show most. "
            "**Tuzi and Ulcinj are Montenegro's Albanian municipalities and only one of "
            "them is on this map.** Tuzi was split from Podgorica in 2018 and no boundary "
            "file yet published has it, so its people are drawn inside Podgorica and its "
            "Catholic and Muslim majority is averaged into the capital's."),
        how="census, 2023",
        grain="municipalities, 27,000 people on average",
        counts=_me_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "me" / "me_grid_400m.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_me_place_weight,
        note="**§11c AND §11k BOTH RECORDED THAT MONSTAT PUBLISHES NO XLSX AND BOTH WERE "
             "WRONG, FOR A REASON THAT GENERALISES.** Montenegro's 2023 census lives under "
             "`uploads/files/popis 2021/` — the year it was first scheduled for, before it "
             "was postponed twice — so every path search for `2023` misses it, and the two "
             "earlier sweeps walked the census landing pages without descending into their "
             "fourteen subpages. Enumerating `page.php?id=` across the range found it on "
             "id=2342 in one pass. **When an office's folder names disagree with its "
             "publication titles, enumerate the pages rather than guessing the paths.** "
             "**AND WHAT IT PUBLISHES IS FINER THAN THE MAP CAN USE.** `naselja vjera popis "
             "2023..xlsx` is religion at the **settlement** — 1,462 of them, 425 people "
             "each, which would be the finest counting geography on this map after "
             "Estonia. It is aggregated to 23 municipalities here because **no settlement "
             "geometry exists**: geoBoundaries has only ADM0 and ADM1 for Montenegro, and "
             "the GISCO LAU file that covers Albania's 61 bashki, Serbia's 169 and "
             "Liechtenstein's 11 has **zero Montenegrin features** — checked directly, not "
             "inferred. The counts for a much better map are on disk and waiting for a "
             "boundary file. "
             "**THE `z` SENTINEL IS SUPPRESSION AND THE WORKBOOK SAYS SO IN ITS OWN "
             "LEGEND** — `\"z\" zaštićen podatak`, beside `\"-\" nema pojave` for a true "
             "zero. So this is Lithuania's case (§9q) and not Kosovo's (§9w), the two look "
             "identical from the cell alone, and reading it as zero would delete 29,225 "
             "people. `sources/me.py` asserts the legend is still present, because it is "
             "the only evidence for the reading. "
             "**IT CANNOT BE DIFFERENCED OUT, AND THAT IS WORTH KNOWING BEFORE TRYING.** "
             "The obvious attack is that a settlement with one suppressed cell gives it "
             "away by subtraction from its own total. **Of the 745 settlements that publish "
             "a total and carry a `z`, exactly zero have only one.** MONSTAT's "
             "complementary suppression is properly done and the check asserts it, so a "
             "sloppier future vintage is noticed rather than silently exploited. "
             "**THE DISCLOSURE THRESHOLD IS TEN AND IS READABLE OFF THE DATA** — no value "
             "anywhere in either 2023 settlement workbook is below it — which bounds a "
             "primary suppression at 1-9. It does **not** bound the total: 223 settlements "
             "have a gap larger than 9 x their `z` count, so those cells are complementary "
             "rather than small, and they hold 23,932 of the 29,225. "
             "**THE TOTAL COLUMN IS SUPPRESSED TOO, IN 219 SETTLEMENTS**, which this build "
             "assumed away first and which showed up as the categories out-summing the "
             "country by 31 people. Where no denominator is published there is no residual "
             "to compute, so those settlements are dropped whole rather than half-drawn. "
             "**AND THE BOUNDARY VINTAGE COSTS A REAL UNIT.** Montenegro has been splitting "
             "municipalities for a decade — Petnjica off Berane 2013, Gusinje off Plav "
             "2014, Tuzi off Podgorica 2018, Zeta 2022 — and the geoBoundaries cut has the "
             "first two and not the last two. Tuzi and Zeta are folded back into Podgorica, "
             "23 units against the census's 25. Podgorica's Kontur ratio of 0.95x against a "
             "median of 1.08x is what says the merge is right, and it is the only place a "
             "wrong one would show.",
    ),
    "jm": dict(
        name="Jamaica",
        source="Census 2011 (Statistical Institute of Jamaica)",
        basis="self-identification",
        view=[-78.45, 17.65, -76.15, 18.60],
        gap="Bahá'ís, Hindus, Muslims and Jews, who were left out of the parish tables",
        note_public=(
            "**Jamaica is drawn for its categories, not its geography.** 14 parishes is "
            "coarse — about 190,000 people each — but the census names **19 religions**, "
            "which is the most detailed religion question in the Americas outside the "
            "United States, and three of its answers exist nowhere else on this map. "
            "**Rastafari is counted where it began — 29,026 people.** Read that as a floor "
            "rather than a count: Rastafari is a way of life more than a membership, census "
            "enumeration of it is widely thought to undercount, and 1.1% is far below any "
            "cultural estimate of how many Jamaicans live by it. Nothing here scales it up. "
            "It is highest in Kingston (1.5%). "
            "**Revival Zion and Pukkumina get their own colour**, 36,296 people. They came "
            "out of the Great Revival of 1860-61, when a Christian revival met the surviving "
            "Afro-Jamaican spirit practice, and they are drawn beside Umbanda and Candomblé "
            "rather than as a Christian denomination, because that is what they are. **They "
            "peak in Saint Thomas at 3.6%**, the eastern parish where the Kongo-derived "
            "tradition concentrated. "
            "**The Church of God bodies are the biggest thing in Jamaican religion and "
            "almost nobody counts them apart.** Four of them here — in Jamaica, of Prophecy, "
            "New Testament, and other — **689,868 people between them, 25.7% of the "
            "country**, more than any single denomination. "
            "**And 21.4% report no religion, the highest share in the Americas on this "
            "map**, rising to **34.1% in Kingston** against 11.8% in Manchester. "
            "**What is missing is specific and worth naming.** Jamaica's Bahá'ís, Hindus, "
            "Muslims and Jews — 4,124 people between them — were left out of the parish "
            "tables by the statistical institute, so they are absent from this map "
            "entirely rather than folded into another colour. Jamaica has all four "
            "communities; this source simply does not place them."),
        how="census, 2011",
        grain="parishes, 191,000 people on average",
        counts=_jm_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "jm" / "jm_grid_400m.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_jm_place_weight,
        note="**THE COUNTRY §3.9b UNBLOCKED, AND IT WAS NEVER A TECHNICAL PROBLEM.** §11j "
             "verified this file end to end on 2026-09-06 and still left it unbuilt, as "
             "\"a category source on a geography that fails the floor\" — 14 parishes "
             "against a 12-unit rejection invented in §11d. That floor was already dead "
             "when it was last cited: Guyana (10 regions) and Georgia (11) had both been "
             "drawn. Anita withdrew it and the country took an afternoon. "
             "**THE JOIN IS FREE, WHICH IS THE WHOLE POINT OF THE USCB SERIES.** "
             "`GEO_MATCH` keys the counts to the boundaries by construction — 14 table "
             "keys, 14 geo keys, 14 matched, 0 unmatched — where Bosnia the same week "
             "needed four documented repairs to geoBoundaries before a name join could be "
             "attempted. And the vintage pair is checked rather than assumed: the "
             "geodatabase ships GEOG1 (2011 census) and GEOG2 (2012 survey), and §11j's CAR "
             "finding is that taking the newer layers silently breaks the join. "
             "**THE PARISH TABLES ARE MISSING FOUR RELIGIONS AND ONLY THE METADATA SHEET "
             "SAYS SO.** STATIN's national universe is 2,683,105 and the 19 published "
             "columns sum to 2,678,981; the 4,124-person difference is Bahá'í, Hinduism, "
             "Islam and Judaism, excluded from the parish tables. They are **absent, not "
             "pooled** — every parish's 19 cells sum to its own total exactly and the 14 "
             "parishes sum to the national row exactly on all 19 categories, so a build "
             "from the data sheet alone would assert Jamaica has no Muslims, Hindus, "
             "Bahá'ís or Jews at all **and every reconciliation it ran would pass**. "
             "`sources/jm.py` asserts the gap is exactly 4,124 rather than tolerating it. "
             "This is the country §11h's read-the-metadata-first rule was written for. "
             "**The ADM2 tier exists and the religion table does not reach it.** The gdb "
             "ships STATIN's `Special Areas`, and §9p's lesson is that a level can hide "
             "inside the finest one — checked, and it does not: the religion sheet is 15 "
             "rows, one country and 14 parishes. "
             "**`Other religion` is 6.3% and its geography is sharp** — 2.9% in Kingston "
             "against 14.2% in Westmoreland — which by §9r's rule makes it a missing "
             "category rather than a mixture. What is in it is not published, and it is "
             "left as an open question rather than guessed at. "
             "**Placement is Kontur's 400 m grid**, 13,373 hexes: 14 parishes over 10,991 "
             "km² averages 785 km², and uniform scatter would put dots on the Cockpit "
             "Country and the Blue Mountains. The Kontur/census ratio is the tightest on "
             "this map — **0.99x to 1.07x across all 14** — because the vintages are close "
             "and Jamaica's population barely moved between them.",
    ),
    "vc": dict(
        name="Saint Vincent and the Grenadines",
        source="Census 2012 (Saint Vincent and the Grenadines Statistical Office)",
        basis="self-identification",
        view=[-61.60, 12.50, -61.05, 13.42],
        note_public=(
            "**The most finely counted country on this map.** 221 enumeration districts for "
            "109,188 people — a median of **415 people per unit**, over an area smaller than "
            "the Isle of Wight — with **18 religions named**. Nothing else here counts this "
            "few people at a time, and it is why religions of a hundred people are visible "
            "at all. "
            "**Six categories are under 400 people**: Presbyterian 294, Salvation Army 287, "
            "Mormon 207, Muslim 111, Hindu 89, Traditional 74. At one dot per thousand "
            "people none of them draws a dot, so they appear as presence rings — the mark "
            "that says *this religion is here* without claiming how many. "
            "**Saint Vincent is 27.6% Pentecostal**, the highest share of any country drawn "
            "here, with Anglicans at 13.9% and Adventists at 11.6%. The Anglican and "
            "Methodist inheritance is the British colonial church; the Pentecostal majority "
            "arrived in the twentieth century and overtook it. "
            "**Only 7.5% report no religion — against 21.4% in Jamaica**, 160 km away and "
            "drawn from the same series. That is the sharpest irreligion contrast between "
            "neighbours anywhere on this map. "
            "**Rastafari is 1.08% here and 1.08% in Jamaica** — two independently designed "
            "censuses arriving at the same share — but it is spread quite differently: "
            "present in **186 of the 221 districts**, where the other small religions sit in "
            "twenty or thirty. "
            "**And the census's `Traditional` cell is not what it looks like.** 74 people, "
            "and the obvious reading is the Kalinago — the largest surviving indigenous "
            "community in the eastern Caribbean, at Sandy Bay in the north. It is not them: "
            "every one of the eight most-Kalinago districts returns zero Traditional. What "
            "it is instead, this census cannot say, so those 74 people are drawn in the "
            "residual rather than assigned to a religion nobody can verify."),
        how="census, 2012",
        grain="enumeration districts, 415 people on average",
        counts=_vc_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "vc" / "vc_eds.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        note="**THE COMPANION 11j ASKED FOR, AND THE CROSS-TABLE CHECK IS THE REASON TO "
             "TRUST IT.** 11j said *\"build it as a companion to Jamaica or not at all\"*; "
             "Jamaica landed 2026-09-06 (9ab) and this followed. **This file publishes NO "
             "religion total** — there is no `RLG_RTOTL` the way Jamaica has one — so the 18 "
             "religion cells are checked against `ETH_TPOP`, the separately tabulated "
             "ethnicity universe sitting in the same sheet. They agree **to the person on "
             "all 235 rows at every level**, and the 13 divisions and 221 districts each sum "
             "to the national row on all 18 categories with zero discrepancy. Two "
             "independently tabulated questions agreeing exactly is a better check than a "
             "column summing to its own neighbour. It also settles the universe: religion is "
             "asked of everybody, so no share needs a 15+ caveat. "
             "**KONTUR WAS BUILT, MEASURED AND REMOVED, AND THE REASON GENERALISES.** Every "
             "country since Kenya has been placed on Kontur's 400 m grid, so it was the "
             "default here too. A Kontur r8 hex is ~0.16 km2 and the median enumeration "
             "district is **0.66 km2**, with 43 units smaller than a single hex: the country "
             "holds **509 hexes**, **78 of 219 populated districts get none at all**, and the "
             "per-district Kontur/census ratio runs p10 0.00, median 0.45, p90 2.68. A "
             "weighting absent for a third of units and scattering over an order of "
             "magnitude for the rest is noise, not a weighting. Placement is 8.2's uniform "
             "within-unit instead, which a 0.66 km2 unit does not need improving on. **The "
             "rule: a population grid must be finer than the counting tier to be worth "
             "anything, and Kontur r8 stops paying at roughly 1 km2 per unit** — every "
             "earlier customer was far above that line, so the floor had never been reached. "
             "What it costs is bounded and named: the largest district is 44 km2 of "
             "uninhabited Soufriere massif, and uniform scatter puts one or two dots up a "
             "volcano. "
             "**The boundaries came out of a coastal engineering report.** The statistical "
             "office does not publish them; USCB digitised them from *Figure 3* of the "
             "Georgetown Coastal Defense environmental assessment. 384.6 km2 against the "
             "country's 389, and the exact `GEO_MATCH` join — 221/221, sixth country in the "
             "series, sixth exact join — is what vouches for it. "
             "**`Traditional` is not the indigenous population, and that was tested rather "
             "than assumed** (12's Philippines co-location technique, applied to the "
             "ethnicity columns in the same sheet): r = -0.03 against the Indigenous share, "
             "and zero Traditional in all eight of the most-indigenous districts. Filed in "
             "the residual with `afrodiasporic` recorded as the node that would be wanted if "
             "it were ever resolved.",
    ),
    "ge": dict(
        name="Georgia",
        source="2014 General Population Census (Geostat)",
        basis="self-identification",
        view=[40.0, 41.0, 46.8, 43.6],
        note_public=(
            "**Georgia is 83% Orthodox and the interesting quarter of it is the south.** "
            "The Georgian Orthodox Church is one of the oldest autocephalies anywhere — the "
            "country converted in the 320s — and across the western and central regions it "
            "runs to 99%. The map is drawn on eleven regions, so read it as regional "
            "composition rather than as neighbourhoods. "
            "**There are two entirely separate Muslim populations and the census puts them "
            "in one box.** In **Adjara**, on the Black Sea, 39.8% are Muslim and they are "
            "**Georgian-speaking Sunnis** — converted under three centuries of Ottoman rule "
            "and Georgian in language, name and everything else. In **Kvemo Kartli** on the "
            "Azerbaijani border, 43.0% are Muslim and they are **Azerbaijanis, largely "
            "Shia**; lowland Kakheti has the same population at 12.1%. Nothing in the "
            "source separates them, so both draw the same colour, and that is a limit of "
            "the census rather than of the country. "
            "**Samtskhe-Javakheti in the south is a different country religiously.** 39.9% "
            "Armenian Apostolic — the Armenian-majority districts of Akhalkalaki and "
            "Ninotsminda — and **9.4% Catholic, which is 78% of all the Catholics in "
            "Georgia**, the Armenian Catholics of Akhaltsikhe. Orthodoxy is a minority "
            "there at 45%. "
            "**And Tbilisi holds almost all of Georgia's Yazidis** — 8,124 of 8,591, 95% of "
            "them in one city. Kurmanji-speaking, descended largely from refugees of the "
            "Ottoman persecutions of the 1910s and 1920s, and one of the few Yazidi "
            "communities anywhere with a purpose-built temple outside Iraq. "
            "**Only 0.5% report no religion**, which is remarkable for a country that spent "
            "seventy years in the Soviet Union — and lower than anywhere on this map except "
            "Kosovo. Adjara is the exception at 2.8%. "
            "**Two parts of Georgia are not on this map at all.** The census could not "
            "enumerate Abkhazia or the Tskhinvali region (South Ossetia), so both are blank "
            "here — not empty, uncounted. About 1.2% of the people who were counted "
            "declined the question or left it blank and are not drawn either."),
        how="census, 2014",
        grain="regions, 334,000 people on average",
        # `gap` — one line under the country's name in the viewer (§6.12). See index.html's
        # note beside `gapNote`: a blank on a dot map cannot distinguish "nobody here is
        # religious" from "nobody counted here", and this is the only place that difference
        # is stated where the blank is actually on screen. A few words, never a sentence
        # that wants a second line.
        gap="Abkhazia and South Ossetia",
        counts=_ge_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ge" / "ge_grid_400m.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ge_place_weight,
        note="**THE TABLE IS ON A HOST NOTHING LINKS TO.** Geostat's census pages publish "
             "religion as a single 33 KB `.xls` at region level and that is all the earlier "
             "scouting found (§11k). The same table is also in a live PxWeb: "
             "`census.geostat.ge` is 404 and was never archived, `api.geostat.ge` answers "
             "200 with an empty body, and `pc-axis.geostat.ge` serves the **default IIS "
             "splash page** at its root — but `pc-axis.geostat.ge/PXWeb/api/v1/en/` is a "
             "working catalogue with a whole `Population Census 2014` database in it. **A "
             "default IIS page is not a dead host; it is a host with nothing mounted at "
             "`/`.** "
             "**Three departures from the modern PxWeb contract, on one server.** Its root "
             "returns `dbid` rather than `id` — the third instance after Kosovo and Moldova "
             "(§11k) — so a generic walker reads it as empty; an empty `{\"query\": []}` "
             "POST 404s where every other PxWeb here accepts it; and `json-stat2` 404s "
             "while `json-stat` v1 works. Each alone looks like 'the table is not there'. "
             "**11 REGIONS, AND THE COARSENESS IS THE POINT OF ARGUMENT.** Anita's test, "
             "and it is the right one: 334,000 people per unit is finer per person than "
             "Russia's 79 federal subjects at 1.8 million, and Russia is drawn. Judge a "
             "geography against the rest of the map, not by unit count. Nothing finer "
             "exists — Geostat's own census database publishes municipalities for marital "
             "status and only regions for religion. "
             "**THE SUPPRESSION IS BOUNDED AND THE CODELIST GIVES THE BOUND AWAY.** A "
             "thirteenth `region` value is not a region: it is the string `… is less or "
             "equal to 10`, the legend for the withheld cells, sitting inside the dimension. "
             "So `sources/ge.py` can assert something stronger than usual — every category's "
             "(national − sum of regions) must be at most ten times its number of withheld "
             "cells. It holds on all twelve: **7 withheld cells and 22 people unaccounted "
             "for in 3.7 million.** "
             "**`None` IS A CATEGORY NAME AND PANDAS READS IT AS NaN** — §9m's trap in its "
             "third country. Georgia's irreligious answer is literally `None`, so a default "
             "`read_csv` deletes 19,080 people without a word. "
             "**Abkhazia and the Tskhinvali region were not enumerated, and they are "
             "handled differently because the boundary files handle them differently.** "
             "Abkhazia is its own geoBoundaries ADM1 with no census row, so it is dropped "
             "from the units layer and simply draws nothing. South Ossetia has no ADM1 of "
             "its own — its municipalities sit inside Shida Kartli (Java) and "
             "Mtskheta-Mtianeti (Akhalgori) — so those two ADM2 polygons are subtracted "
             "from the placement grid, and no dot lands on ground the census did not count. "
             "That correction turns out small (6,093 modelled people, because Kontur's "
             "Georgian extract barely covers South Ossetia) and is worth making anyway. "
             "**The city/ring pair recurs, as §9q said it would.** Kontur/census is 0.85x "
             "for Tbilisi and **1.66x for Mtskheta-Mtianeti, the region wrapped around it** "
             "— geoBoundaries' Tbilisi polygon is 249 km² against the city's ~500, so outer "
             "Tbilisi sits in its ring region here. A fourth post-Soviet country with the "
             "same artefact. It is a placement fact and not a count fact: every region's "
             "dot total still comes from the census. Asserted on the other ten, reported on "
             "the ring. "
             "Placement is Kontur's 400 m H3 grid, 25,189 hexes — eleven regions over "
             "61,000 enumerated km² is 5,500 km² each, and Georgia is two mountain ranges "
             "with the people in the valleys.",
    ),
    "il": dict(
        name="Israel",
        source="2022 Census of Population and Housing (CBS)",
        basis="population register",
        view=[34.2, 29.4, 35.95, 33.35],
        note_public=(
            "**Israel's religion is read off the population register, not asked.** Every "
            "other country here counts what people said about themselves; this one records "
            "what the state has on file, assigned at registration from parentage or a "
            "recognised conversion. The two are not the same measurement and the "
            "percentages are not comparable with a census that asks. "
            "**The register has no box for having no religion**, so Israel draws as almost "
            "entirely religious — which is a fact about the form and not about the country. "
            "The observance colours are the corrective: **53% of Israeli Jews describe "
            "their household as secular**, and they are drawn inside Judaism because that "
            "is where the register puts them. "
            "**Haredi Israel is the sharpest pattern on the map.** 6.4% of the country and "
            "hardly spread at all — Bene Beraq is 84% ultra-religious, Modi'in Illit and "
            "Beitar Illit almost entirely so, and in Jerusalem the Haredi quarters sit "
            "against secular ones street by street. The map is drawn on statistical areas "
            "of about 3,000 people, which is fine enough to show that edge. "
            "**Where the observance split is NOT drawn, Judaism is one colour.** The "
            "question is asked of every household, Arab ones included, so applying it where "
            "a unit is religiously mixed would attribute one group's answers to another. It "
            "is used only where a unit is at least 85% Jewish; elsewhere the Jewish dots "
            "carry no observance. "
            "**The Druze are the largest count of them anywhere** — about 153,000, in the "
            "Galilee and Carmel villages and in the four Golan villages, where most "
            "residents have declined Israeli citizenship. "
            "**Christians are one cell.** CBS does not separate Greek Orthodox from Greek "
            "Catholic, Latin, Maronite, Armenian or Syriac, nor Arab Christians from the "
            "large ex-Soviet and migrant Christian population, so some of the oldest "
            "continuously resident churches in the world draw a single colour. "
            "**\"Others\" is not irreligion.** About 442,000 people are on the register with "
            "no religious classification, overwhelmingly immigrants under the Law of Return "
            "who are not Jewish by halakha. Nobody asked them; the register simply has no "
            "entry, which is why they are drawn grey."),
        how="population register, not a census question",
        fill="from each area's own household-lifestyle table",
        grain="statistical areas, about 3,000 people each",
        # §6.12: a blank on a dot map cannot tell "nobody here is religious" from "nobody
        # counted here". This is the only place that difference is stated while the blank is
        # on screen, and Israel's blank is a deliberate territorial decision rather than a
        # hole in the data — see sources/il_geo.py.
        gap="the West Bank, Gaza and East Jerusalem; the Golan is drawn",
        counts=_il_counts,
        units=None,
        unit_key=None,
        # PLACEMENT IS THE UNIT POLYGON AND THERE IS NO GRID — §8.2e, measured rather than
        # skipped. Israel's median statistical area is 0.69 km² against a 1.17 km² Kontur
        # hex, so the grid is COARSER than the tier it would refine: 42% of units get no hex
        # at all and 69% are smaller than one. `sources/il_geo.py` runs the test on every
        # build and prints the verdict, so this cannot quietly become wrong.
        place=HERE / "data" / "geo" / "il" / "il_units.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        note="**THE MAP STOPS AT THE GREEN LINE AND THE GOLAN IS THE ONE EXCEPTION** — "
             "Anita, 2026-09-07. The West Bank and Gaza are not drawn, and neither is East "
             "Jerusalem: cutting only its Palestinian neighbourhoods while keeping Gilo, "
             "Pisgat Ze'ev, Ramot and Neve Ya'akov would draw the settlements and erase the "
             "people they were built among, which is §14.2's second risk exactly. So "
             "Jerusalem draws as a western fragment and 267 of 3,235 units are dropped. The "
             "Golan IS drawn, against the same rule, because Israel counts those 56,600 "
             "people and nobody else does — 24,900 of them Druze. The cut is OCHA's "
             "published oPt polygon rather than a line drawn here, checked against ten "
             "hand-picked points on every build; geoBoundaries' Palestine excludes annexed "
             "Jerusalem and fails five of them, which is the trap this check exists for. "
             "**THE DASHBOARD COLLAPSES MINORITIES INTO 'Other religions' AND IT IS NOT "
             "'Others'.** For units below some size CBS publishes the dominant group and a "
             "lump: Nazareth returns Muslims 73.1% / Other religions 26.9%, and that 26.9% "
             "is essentially all Christian. Read naively it erases Israel's Christians. "
             "sources/il.py resolves the lump against sub-district totals and "
             "`_il_counts` refuses to build if any survives. "
             "**THE DATA IS ONE REQUEST PER AREA AND THE HOST THROTTLES.** The area IDs are "
             "opaque hashes with no relation to CBS codes; they come from the census site's "
             "own htmx autocomplete, `/he/partials/search/area`, which nothing links to. "
             "Fetching ~4,600 of them at 0.15s intervals got this machine IP-blocked from "
             "census.cbs.gov.il for hours. sources/il.py now reuses one connection, backs "
             "off, and caches every 50 units so the run is resumable — and it is deliberately "
             "slow. Do not parallelise it. "
             "**THERE IS NO PLACEMENT GRID, AND THAT WAS MEASURED** (§8.2e). A Kontur grid "
             "was built here first and thrown away: Israel's median statistical area is "
             "0.69 km² against a 1.17 km² hex, so the grid is COARSER than the tier it "
             "would refine — 42% of units get no hex at all, 69% are smaller than one, and "
             "the per-unit Kontur/census ratio runs 0.27 to 3.49. That is noise, not a "
             "weighting. Placement is the unit polygon, i.e. §8.2's uniform share, which "
             "§8.2e argues is the better answer here rather than a fallback; "
             "`sources/il_geo.py` re-runs the test on every build so the decision cannot "
             "rot. What it costs: the few genuinely large units — Negev Bedouin localities "
             "and regional councils, up to 195 km² — get an even wash.",
    ),
    "fr": dict(
        name="France",
        name_in="France",
        source="ESS rounds 5–11 (citizens) + Eurostat census 2021 × Pew 2020 (residents)",
        basis="self-identification, sample survey (citizens); nationality-derived (residents)",
        view=[-5.4, 42.2, 8.4, 51.2],
        note_public=(
            "**France has never asked, and is forbidden by law from asking.** Collecting "
            "religion in official statistics is tightly restricted here, so there is no "
            "census figure to draw and there never will be. What there is instead is a "
            "survey that asks the question directly: the European Social Survey has put it "
            "to French residents in seven rounds since 2010, and pooling them gives 12,678 "
            "citizens with a région attached. The 4.9 million people living in France on "
            "another country's passport are counted separately, from the 2021 census's own "
            "record of who lives where and where they are from. "
            "**Half of France reports no religion — 48.8%, the largest answer on the map "
            "here and larger than Catholicism.** That share has not moved in fourteen "
            "years: it was 52.6% in 2010 and 53.5% in 2023–24, with no trend in between. "
            "What has moved is what the other half is. Catholicism is **38.0%** and falling "
            "about a point every five years; Islam is **8.1%** and rising. "
            "**Alsace is the exception to everything and the survey finds it unaided.** "
            "10.4% Protestant against 2.2% nationally — the Lutheran and Reformed churches "
            "of the one part of France where the 1801 Concordat never lapsed, where the "
            "state still pays clergy and religion is taught in public schools. It is also "
            "10.0% Muslim, so the most Protestant région in France is very nearly its "
            "second most Muslim one. "
            "**Islam's geography is the cities and the industrial north-east.** "
            "Île-de-France **16.4%**, Provence-Alpes-Côte d'Azur **11.6%**, Alsace 10.0%, "
            "Franche-Comté 9.4% and Rhône-Alpes 8.9% — Paris, Marseille, Strasbourg, "
            "Sochaux and Lyon, which is to say where the car plants and the ports were — "
            "against **1.8% in Poitou-Charentes** and 2.4% in Bretagne. "
            "**And the déchristianisé west and centre are still there.** Poitou-Charentes "
            "is **63.8%** no religion, Centre-Val de Loire 57.9% and Picardie 58.0%, "
            "against Alsace at 38.0% and Lorraine at 42.7%. That is roughly the map "
            "Gabriel Le Bras and Fernand Boulard drew from Mass attendance in the 1940s and "
            "1950s, sixty years and one collapse in practice later. "
            "**Jews are 0.53% of France and 1.7% of Île-de-France**, the largest Jewish "
            "population in Europe and the third largest anywhere, and its concentration in "
            "and around Paris is the sharpest of any group here after Alsace's "
            "Protestants. "
            "**What the sources cannot do, and the first one is the big one.** The survey "
            "offers denominations, not positions: there is no atheist or agnostic box, so "
            "everyone who says they belong to nothing lands in a single category and "
            "**France — of all countries — has nothing on the `secular` node at all.** The "
            "immigrant half counts people as their country of origin's religion, an upper "
            "bound that cannot see conversion, lapse or anyone who stopped practising after "
            "arriving. France's Buddhists, the largest community in Europe, are drawn at "
            "0.13% against Pew's 0.71%, because a Buddhist with a French passport has "
            "nowhere to go on the form. "
            "**The two halves are counted at different grains, and it matters most in the "
            "Paris region.** Foreign residents are counted by *département* — 94 of them — "
            "so Île-de-France is drawn as eight units rather than one: **Seine-Saint-Denis "
            "comes out 21.5% Muslim against Seine-et-Marne's 13.9%**, where until recently "
            "the whole region showed a single 16.3%. French citizens are still counted by "
            "*région*, because the survey has nothing finer, so **every département inside "
            "one région shares its citizens' composition** and the real spread is wider "
            "than what is drawn. Below the département nothing is measured at all: within "
            "Paris the twenty arrondissements are one number. "
            "**Within a région, though, the dots are not scattered blindly.** People "
            "counted as foreign nationals are placed where foreign nationals actually "
            "live, commune by commune, and French nationals where French nationals live — "
            "and those are very different maps: **64.5% of France's foreign residents live "
            "in cities against 36.1% of its citizens.** So a Muslim or Buddhist dot sits in "
            "a town rather than in the countryside around it. That is a statement about "
            "where a population lives, not about where a religion is: it cannot tell one "
            "commune from its neighbour, and inside a city it says nothing at all. "
            "**And the five overseas régions are a different country on this map.** They are "
            "outside the survey's frame and are drawn instead from Pew's own estimate for "
            "each territory — which works here because each one *is* a single statistical "
            "region, so nothing is being guessed at a finer grain than it was published. "
            "**Mayotte is 98.8% Muslim**, the only French département that is, and it is "
            "absent from the European census table altogether, so its people are on this map "
            "only because a second source counted them. **La Réunion is 4.5% Hindu and 4.2% "
            "Muslim** — the Malbar and the Zarabe, descended from indentured Tamil labourers "
            "and Gujarati traders — which makes it the most religiously mixed part of "
            "France. **Guyane is 9.2% traditional religion**, the Maroon communities of the "
            "Maroni and the Amerindian peoples of the interior. And where metropolitan France "
            "reports 48.8% no religion, Guadeloupe and Martinique report 2.5% and 2.7%. "
            "**The overseas régions are drawn coarser than the mainland, and that is the "
            "instrument and not the place**: Pew publishes seven broad families per "
            "territory, so their Christianity is one undivided colour where metropolitan "
            "France's is split into Catholic, Protestant and Orthodox. Martinique and "
            "Guadeloupe are overwhelmingly Catholic in every account of them; no source "
            "publishes the split, so this map does not draw it. "
            "**0.51% of France is not drawn**: Corsica, which the survey does not sample and "
            "which Pew does not publish separately because it is not a territory."),
        how="survey, 12,678 people; foreign residents by nationality",
        grain="departments for foreign residents (680,000 people); regions for French citizens",
        counts=_fr_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "fr" / "fr_lau.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_fr_place_weight,
        note="**The country spec §14.3 was written against, drawn — and the paragraph that "
             "excluded it was excluding a different route.** §14.3 uses France by name for "
             "the move this map does not make: *\"estimating religion there would mean "
             "inventing the magnitude as well as the location, most plausibly from "
             "surnames, origin or nationality\"*. That was true of every route known when it "
             "was written, and ESS is none of them: it asks French residents which religion "
             "they belong to and publishes it by région, so 92.7% of the people here come "
             "from a survey that asked them. spec §14.11 is the amendment; §14.10, decided "
             "the same day, is what permits the other 7.3%. "
             "**`sources.md` §11l closed France on a geography ESS does not use.** Its "
             "people-per-unit table judged the country on the 13 post-2016 régions — 5.26M "
             "each, \"fails badly\" — and ESS carries the **21 anciennes régions**: "
             "NUTS-2010 `FR10`/`FR21`…`FR82` in rounds 5–7, NUTS-2016 `FR10`/`FRB0`…`FRL0` "
             "in rounds 8–11, a clean 1:1 recode. 3.10M per unit rather than 5.26M. **A "
             "scouting note that rejects a country on a number should say where the number "
             "came from**, because this one was assumed rather than read off the source, "
             "and it cost the country a day short of a fortnight. "
             "**Seven rounds pool where Greece got three, and the category list does not "
             "collapse.** Every round offers and uses at least ten denominations, so "
             "nothing blinks out the way `Islam` does in Greek round 11. What pooling costs "
             "instead is time: rounds 5–11 span 2010–2024. The drift is smaller than that "
             "sounds — \"no religion\" is flat across the whole window — but Islam among "
             "citizens rises 4.6% → 6.8%, which is naturalisation rather than conversion, "
             "so the pooled figure understates the present by about a point. "
             "**NO CELL IS AUTHORED, and that is the difference from Greece.** Greece "
             "needed the Thracian minority put back by hand because a Greek-language sample "
             "reaches none of it. France's sample is not blind anywhere comparable: it "
             "finds Alsace's Protestants, Île-de-France's Muslims and Jews and the "
             "Mediterranean's Muslims on its own. Where it is weak — the banlieues — there "
             "is no published régional figure to substitute and inventing one is §14.4's "
             "first prohibition. "
             "**The foreign half is better than Greece's and the check is looser.** "
             "`cens_21ctz_r3`'s 200 named citizenships cover **100.00%** of France's "
             "4.74M foreign residents, against Greece's 99.84%, so the unnamed-remainder "
             "rescale is a no-op. Portugal, Algeria and Morocco are a third of it. The "
             "cross-check per §14.10: Muslims come out at **8.04%** of the drawn population "
             "against Pew's own independent 2020 estimate of **9.10%** — from Eurostat "
             "counts, Pew origin compositions and an ESS survey, none of which is Pew's "
             "France row. Greece landed at 5.08 against 5.12; this is looser, the "
             "categories are eight times larger, and the gap runs the direction a "
             "self-identification survey always runs against a composite estimate. Nothing "
             "is tuned to close it. "
             "**The foreign half IS drawn at the 94 départements as of 2026-09-08, and this "
             "paragraph used to say the opposite.** It declined them on the reasoning this "
             "file gives for Greece — mixing would put the sharper geography on the half "
             "with the weaker claim to it, making the most-inferred part of France also the "
             "most precise-looking part of it. **Italy (§9as) showed that rule has an "
             "unstated premise: that the fine half is the small half.** France's foreign "
             "half is 4.74M people and holds most of what a religion map of this country is "
             "for, and the price of the old rule was the thing §8 named as the single "
             "biggest defect here — that the map could say Île-de-France is 16% Muslim and "
             "nothing whatever about Seine-Saint-Denis. It now says **21.5% against "
             "Seine-et-Marne's 13.9%**, and the eight départements of the Paris region "
             "separate. The basis did not change and neither did the model; Eurostat "
             "publishes at NUTS 3, so §14.3's *never model finer than the source publishes* "
             "is satisfied by the source rather than by an argument. **The citizen half "
             "stays at the région** — ESS has nothing finer — so the drawn spread is "
             "narrower than the real one, and every citizen row's `note` names the région "
             "its composition came from. "
             "**The overseas régions are a THIRD instrument, added 2026-09-07, and the "
             "reason it is allowed is geometric rather than statistical.** The first build "
             "left them undrawn because ESS's frame is metropolitan and borrowing a "
             "metropolitan composition for Martinique would be a false statement rather than "
             "an honest silence. Pew turns out to publish **all five as separate countries** "
             "in the same file `origin_religion.py` was already reading — and **each DOM is "
             "exactly one NUTS 2 unit**, so a Pew country row IS a unit row and nothing is "
             "downscaled at all. §14.3's *never model finer than the source publishes* is "
             "satisfied by identity, which is the cleanest case of it on the map. Basis "
             "`estimate`, §3.1's own word for a Pew figure, and the foreign half is "
             "deliberately not run over these units because Pew's estimate already covers "
             "every resident whatever passport they hold. "
             "**The magnitudes were checked before they were used**: Pew's own populations "
             "land at 0.98-1.01× the 2021 census for the four units the census carries, "
             "which is independent confirmation from a source that is not the census, and "
             "`fr.py` asserts the band rather than reporting it. **Mayotte has no census row "
             "at all** — declared in the Eurostat geo dimension, no values — so it is Pew's "
             "on both shares and total, and France's one overwhelmingly Muslim département "
             "is on this map only because a second source counted it. `fr.py` now asserts "
             "the census row stays empty for the opposite reason to before: a row appearing "
             "would let the foreign half reach the unit and double-count against Pew. "
             "**Corsica is the one thing left undrawn, and it was looked for.** ESS carries "
             "709 variables and exactly one geography (`region`, plus `regunit` and "
             "`domicil`), so no round reaches it; Pew publishes the DOM because they have "
             "ISO codes and not Corsica because it is metropolitan France. What remains is a "
             "national survey with ~10 Corsican respondents, or the Annuario Pontificio's "
             "diocese of Ajaccio — and that is a `roll` against a `self_id` map, which §3.1 "
             "forbids and which would draw Corsica far more Catholic than the mainland "
             "purely as an artefact. 0.51%, and §6.12's wash marks it. "
             "**Boundaries were free and the trap that bit Greece did not fire.** GISCO's "
             "LAU bundle ships all 34,966 communes and the workbook mapping each to its "
             "département; 34,966 matched both ways, zero either side. Greece lost 644 "
             "codes to Excel stripping leading zeros and France has the same exposure — "
             "every commune in départements 01–09 — and escapes it because **Corsica's "
             "codes are `2A001` and `2B033`**, and one alphanumeric value forces the whole "
             "column to be read as text. `sources/fr_geo.py` guards it anyway, because a "
             "vintage without Corsica would put it straight back.",
    ),
    "it": dict(
        name="Italy",
        name_in="Italy",
        source="ESS rounds 6–11 (citizens) + Eurostat census 2021 × Pew 2020 (residents)",
        basis="self-identification, sample survey (citizens); nationality-derived (residents)",
        view=[6.2, 35.3, 18.8, 47.3],
        note_public=(
            "**Italy has never asked, and its statistical office does not collect religion "
            "at all** — it is treated as sensitive data and is absent from the census, the "
            "permanent census and every household survey ISTAT runs. So the country is "
            "drawn the way France and Greece are: a survey that does ask, for the 54.0 "
            "million people with Italian passports, and the 2021 census's own record of who "
            "lives where and where they are from, for the 5.0 million without. "
            "**Two thirds of Italy is Catholic and a quarter belongs to nothing** — 66.6% "
            "against 24.5%, with Islam at 3.8% and Orthodoxy at 2.7%. "
            "**The least religious places are Tuscany and Emilia-Romagna, and that is the "
            "oldest political map in the country.** Tuscany reports **43.2% no religion** "
            "and Emilia-Romagna **37.4%**, against **11.1% in Sicily** and 15.7% in Puglia. "
            "Those two regions are the core of the *zone rosse*, the anticlerical and "
            "communist heartland of the post-war republic, and the survey finds them "
            "unaided sixty years later. Sicily is 83.4% Catholic; Tuscany is 47.0%. "
            "**Islam is a map of where the work is.** Emilia-Romagna and Lombardy are both "
            "**6.0%** Muslim and Liguria 5.4%, against **1.3% in Sardinia** and 1.6% in "
            "Puglia — and at province level it is sharper still: Piacenza 7.1%, Imperia "
            "7.1%, Brescia 7.0%, Bergamo 6.7% and Modena 6.7%, the engineering and "
            "food-processing belt along the Via Emilia and the Lombard valleys, against "
            "0.75% in Oristano. This is the one thing on the map that runs north-south the "
            "opposite way to Catholicism. "
            "**Italy's largest single non-Catholic group is Romanian Orthodox** — 1.05 "
            "million people, more than every Protestant, Jewish, Buddhist and Hindu "
            "community on this map of Italy combined. Orthodoxy peaks in Lazio at 4.5% and "
            "in Viterbo province at 5.0%, and it is almost entirely a story of the last "
            "twenty-five years. "
            "**Prato is 20.6% foreign, the highest of any Italian province**, and Parma, "
            "Piacenza and Milan follow at about 15%. "
            "**What this map cannot do, and the first one is the big one.** The survey gave "
            "Italy twenty regions in 2012 and 2016 and then stopped: every round since has "
            "published only five macro-regions. So the Catholic and no-religion shares are "
            "drawn at the region — for 86% of the population; ten smaller regions had too "
            "few respondents and take their ratio from the macro-region instead — while "
            "**every smaller religion among Italian citizens is drawn at five units of 11.8 "
            "million people.** Foreign residents are drawn at 107 provinces throughout. "
            "That mixture is deliberate: four fifths of Italy's religious minorities are in "
            "the foreign half, so the coarse level lands on the part no Italian source can "
            "locate anyway. "
            "**Within a unit the dots are not scattered blindly.** People counted as "
            "foreign nationals are placed where foreign nationals live, comune by comune, "
            "and Italian citizens where Italian citizens live — 42.5% of Italy's foreign "
            "residents are in cities against 34.4% of its citizens. **And the religion that "
            "prompted this barely moved, which is worth knowing**: Italy's Hindus are in "
            "Rome and Milan and then the Po valley dairy belt and the Agro Pontino — "
            "Brescia, Bergamo, Mantova, Cremona, Latina — because they are largely Punjabi "
            "agricultural labour. They really are a substantially rural population. "
            "**The Jewish figure is roughly two and a half times too high** and is drawn "
            "anyway rather than dropped. Nine survey respondents scale to 64,000; the Union "
            "of Italian Jewish Communities has about 24,000 registered members. It is here "
            "so that a real community is visible, with the error stated rather than hidden. "
            "**Protestants are undercounted for a nameable reason.** The survey finds "
            "71,000, and Italy's Pentecostals alone number over 300,000 — but the "
            "\"other Christian\" answer holds 295,000, and the two added together match the "
            "independent count almost exactly. Italian Pentecostals and Jehovah's Witnesses "
            "do not tick *Protestant* on a form. Read the two together. "
            "**There is no atheist or agnostic box**, so everyone who reports belonging to "
            "nothing lands in a single category and Italy has nothing on the `secular` node "
            "— the same gap as France and Greece. The immigrant half counts people as their "
            "origin country's religion, an upper bound blind to conversion and lapse. And "
            "the **Arbëreshë**, the 60,000 Italo-Albanians of Calabria and Sicily who are "
            "Catholic of the Byzantine rite, are folded into Latin Catholicism, because the "
            "survey offers Italians exactly one Catholic box."),
        how="survey, 11,000 people; foreign residents by nationality",
        grain="provinces for foreign residents; regions and macro-regions for Italian citizens",
        counts=_it_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "it" / "it_lau.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_it_place_weight,
        note="**The first country here drawn at three resolutions at once, and the reason "
             "is that ESS gave Italy a geography and then took it back.** Rounds 6 (2012) "
             "and 8 (2016) carry `region` at NUTS 2 — 19 and 20 regioni. Rounds 9, 10 and "
             "11 carry it at NUTS 1, five ripartizioni, and they hold two thirds of the "
             "sample and all of the recent vintage. `regunit` says which level each round "
             "used and `it.py` asserts it rather than trusting it, so a future release that "
             "moves Italy again cannot be pooled as though nothing had changed. "
             "**Greece and France both met this asymmetry and threw the finer level away; "
             "Italy keeps it, on Anita's call.** Their reasoning was that mixing puts the "
             "sharper geography on the half with the weaker claim to it. Italy inverts the "
             "premise: its citizen minorities are 1.38M people and its foreign residents "
             "5.03M, so **four fifths of what this map exists to show is in the half that "
             "has 107 province**, and flattening everything to five units would have cost "
             "the country its best data to protect its worst. "
             "**THE GEO FILTER THAT LOOKED FINE AND WAS NOT.** The first build drew 97 "
             "province and 52.4M people, reported 99.83% coverage, and balanced perfectly — "
             "because the regex for a NUTS 3 code assumed the last character was a digit. "
             "Lombardia has twelve province and NUTS ran out of numerals, so Mantova is "
             "`ITC4A`, Lodi `ITC4B`, **Milano `ITC4C`** and Monza `ITC4D`; Sardegna and "
             "Sicilia do the same. Ten province and 6.6 million people vanished, including "
             "the largest one in the country, and **every internal check still passed** "
             "because the coverage percentage was taken against the truncated total. "
             "`it.py` now asserts the province count and the national population against "
             "the placement layer, which is built from a different file by a different "
             "script (§8.1). A coverage figure computed from the same filter that lost the "
             "rows cannot detect that the rows were lost. "
             "**THE SAMPLE FLOOR, ADDED BY DISBELIEVING THE OUTPUT.** The first honest build "
             "made **South Tyrol the least Catholic region in Italy** — 48% no religion — "
             "on 29 pooled respondents, with Trento's 23 close behind. Every other account "
             "of South Tyrol has it among the most observant places in the country. spec "
             "§3.9b withdrew the floor on how many UNITS a country may have and says "
             "nothing about how few PEOPLE may stand behind one, so `N_FLOOR = 100` was "
             "added: below it a regione's Catholic/unaffiliated ratio comes from its "
             "ripartizione. Ten of twenty-one fall back, which is **14.3% of the "
             "population** rather than half the map, because the ten are the small ones. "
             "The number is a knob and moving it is a §14 decision, not a tuning. "
             "**The cross-checks are against CESNUR, which nothing in the build reads.** "
             "Muslim citizens land at 491,000 against CESNUR's 417,900 (1.18×) and the "
             "Protestant-plus-other-Christian total at 366,000 against 378,000 (0.97×), "
             "which is the better of the two because it also diagnoses why the Protestant "
             "cell alone is wrong. Orthodox citizens come out at 0.53× — ESS finds about "
             "half of them — though the Orthodox total across BOTH halves, 1.6M, matches "
             "CESNUR's ~1.5M residents, so the shortfall is in the citizen/foreigner split "
             "rather than the magnitude. **Jews are 2.66× and that one is simply wrong**, "
             "on nine respondents. "
             "**And the divergence worth flagging is `unaffiliated`: 24.4% here against "
             "Pew's 13.3% for Italy.** That is an instrument difference rather than an "
             "error — ESS asks whether you belong to a particular religion or denomination, "
             "which collects far more \"no\" than a question asking what your religion is — "
             "and it is the same gap that puts France at 48.8%. Nothing is adjusted to "
             "close it; the basis is stated and the reader is told which question was "
             "asked. "
             "**The otto per mille is NOT used, and the reasoning is in `sources.md` "
             "§11l-ii.** Italy publishes a denominational roll of 42 million taxpayers "
             "across fourteen named confessions, by region, and it looked for one day like "
             "the best source in Western Europe. It is a spending vote and not an "
             "affiliation count: it gives the Waldensians 497,013 choices against a "
             "membership near 25,000, and the Orthodox Archdiocese 39,359 against 1.5 "
             "million residents. **The same column overstates one confession twentyfold and "
             "understates another by forty**, which is the proof it is measuring something "
             "else. Comune-level counts exist and are held by the Agenzia delle Entrate; "
             "they would make a broken measure more precise and are not worth requesting.",
    ),
    "bz": dict(
        name="Belize",
        source="Census 2022 (Statistical Institute of Belize)",
        basis="self-identification",
        view=[-89.30, 15.80, -87.35, 18.55],
        note_public=(
            "**The only census on this map that counts Mennonites.** 15,440 people, 3.9% of "
            "Belize, and they are where the history puts them: **9.9% of Orange Walk and "
            "8.9% of Corozal** against 0.5% of Belize District. The Kleine Gemeinde and Old "
            "Colony communities arrived from Mexico and Canada in 1958 on an agreement that "
            "granted exemption from military service and control of their own schools, and "
            "they farm the north. Nothing else here has ever filled that colour outside the "
            "United States. "
            "**And 31.0% of Belize reports no religion — the highest share this map draws "
            "anywhere in the Americas**, above Jamaica's 21.4%. It is very unevenly spread: "
            "**46.6% in Stann Creek** against 21.7% in Toledo. That is a large rise on 2010 "
            "and the census offers no explanation for it; it is drawn as published. "
            "**Belize is Catholic and Pentecostal along a north-south line.** Catholicism is "
            "37.9% in Corozal and 37.3% in Orange Walk — the Spanish-speaking Mestizo north "
            "— and falls to 26.1% in Stann Creek. Pentecostalism runs the other way, 14.9% "
            "in Cayo and 13.3% in Toledo. Baptists are **12.0% of Toledo**, the Maya south, "
            "against 0.9% of Orange Walk. "
            "**Six districts is coarse and the country is small, so read this as six "
            "readings rather than a map of Belize.** The census publishes religion at "
            "district and nowhere else, though it publishes population down to village. "
            "**What the question does not ask is as important as what it does.** SIB names "
            "nine Christian bodies and pools everything else into `Other` (6.3%). There is "
            "no cell for Hinduism, none for Islam, and none for any indigenous or "
            "Afro-Caribbean tradition, in a country that has all of them. Belize's Hindus "
            "and Muslims, its Bahá'ís, its Rastafari, Maya traditional practice in Toledo "
            "and the Garifuna *dugu* of the Stann Creek coast are either inside that grey "
            "residual or invisible inside a Christian colour, and this source cannot say "
            "which."),
        how="census, 2022",
        grain="districts, 66,000 people on average",
        counts=_bz_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "bz" / "bz_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_bz_place_weight,
        note="**THE `None` TRAP, AND IT IS THE MOST DANGEROUS THING IN THIS COUNTRY.** "
             "Belize's second largest answer is the literal string `None` — 123,373 people, "
             "31.04%. `pd.read_csv` converts it to NaN by default, which drops exactly six "
             "rows and 123,372.67 people, and **every other check in the pipeline still "
             "passes** because `sources/bz.py` reconciles the workbook rather than this "
             "read. `_bz_counts` passes `keep_default_na=False` and then asserts the "
             "category survived, because a silent 31% loss is not something a reviewer would "
             "see on the map — Belize would simply look devout. "
             "**DISTRICT ORDER IS THE SECOND TRAP.** SIB prints its six districts north to "
             "south (Corozal first) and COD-AB codes them alphabetically (`BZ01` is Belize "
             "District), so numbering the table by position — the way `sources/mw.py` "
             "legitimately does — would mismatch five of six units while every total still "
             "reconciled. `sources/bz.py` carries the pcode against the name and "
             "`sources/bz_geo.py` asserts the same pairing from the boundary side. The join "
             "is then 6/6 both ways with no name variants at all, which is a first here. "
             "**THE FIGURES ARE FRACTIONAL AND THAT IS THE SOURCE.** SIB publishes "
             "undercount-adjusted counts throughout — the national total is 397,483.456 — "
             "so every identity in `sources/bz.py` is asserted to a 1e-6 relative tolerance "
             "rather than to zero. The 2010 column in the same workbook is fractional too, "
             "so this is SIB's standing practice and not a one-off. "
             "**MALE + FEMALE == TOTAL IS THE CHECK ON THE READ.** The sheet lays every "
             "group out as three columns and only Total is drawn, but all three are read: "
             "the sex columns are the only check that would catch a district's block landing "
             "one column-group left or right, since every other identity reconciles inside "
             "one group whichever columns were taken. 91 cells, zero failures. Zimbabwe's "
             "panel rule (§9aj) applied to a workbook. "
             "**PLACEMENT IS KONTUR'S 400 m GRID**, 5,095 hexes. Six districts over 22,966 "
             "km² averages 3,828 km² — five times Jamaica's — and Cayo holds the Chiquibul "
             "and the Maya Mountains while Toledo is mostly rainforest, so uniform scatter "
             "would draw a large part of Belize into empty bush. The per-district "
             "Kontur/census ratio runs **0.80x to 1.10x**, tight for an 18-month vintage "
             "gap. 6.8% of the grid's people sit outside every district — the Mexican and "
             "Guatemalan border overrun — and are dropped; Ambergris Caye is inside, which "
             "the 1.02x ratio for Belize District confirms.",
    ),
    "tt": dict(
        name="Trinidad and Tobago",
        source="Census 2011 (Central Statistical Office)",
        basis="self-identification",
        view=[-62.10, 9.95, -60.40, 11.40],
        gap="the institutional population: prisons, hospitals, homes",
        note_public=(
            "**The only census on this map that counts Orisha and Spiritual Baptists.** "
            "Between them they are **86,920 people, 6.6% of the country — more than its "
            "Muslims**, and neither tradition has ever been counted under its own name "
            "anywhere else this map draws. "
            "**Spiritual Baptists are 5.67%, more numerous than Trinidad's Anglicans** and "
            "nearly five times its ordinary Baptists, which the census counts as a separate "
            "answer. Baptist Protestantism and West African practice fused in this one "
            "rather than one absorbing the other — scripture and hymnody alongside spirit "
            "possession, the mourning ground, bell-ringing and water rites. The religion "
            "was **banned outright from 1917 to 1951**, and 30 March is a public holiday, "
            "Spiritual Baptist Liberation Day. They are 13.0% of Point Fortin and 10.6% of "
            "Tobago. "
            "**Orisha — historically Shango — is 11,918 people**, Yoruba orisha worship "
            "carried over in the nineteenth century and the direct sibling of Candomblé and "
            "Santería. Read it as a floor: in Trinidad, Orisha and Spiritual Baptist "
            "practice overlap heavily and many people take part in both, while a census "
            "offers one box. "
            "**Trinidad is the second Hindu geography in the Americas.** 240,100 people, "
            "18.2%, and it is concentrated exactly where indenture put it — **43.0% of "
            "Penal/Debe**, 31.3% of Couva/Tabaquite/Talparo, 30.0% of Chaguanas — against "
            "0.7% of Tobago. Muslims, at 5.0%, follow the same belt. "
            "**And the Presbyterians are Indo-Trinidadian, which the map shows without "
            "being told.** They peak in San Fernando (5.5%) and Penal/Debe (5.3%), the same "
            "units that are most Hindu, because Trinidad's Presbyterian church grew out of "
            "the Canadian Mission to the Indians from 1868 and drew its converts from the "
            "indentured population. Tobago is 0.2%. "
            "**Tobago is a different country religiously.** Roman Catholic 6.6% there "
            "against 44.8% in Diego Martin, with the country's highest Anglican (12.8%), "
            "Adventist (16.3%) and Pentecostal (14.7%) shares. Trinidad is Catholic and "
            "Hindu; Tobago is Protestant. "
            "**11.1% did not state a religion — the largest non-answer on this map outside "
            "the United States.** Every share here is a share of everybody, not of the "
            "people who answered, so a religion's share among answerers is about a tenth "
            "higher than what is drawn. Nothing redistributes it."),
        how="census, 2011",
        grain="municipalities, 88,000 people on average",
        counts=_tt_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "tt" / "tt_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_tt_place_weight,
        note="**THE SOURCE PDF THE PUBLISHER LINKS IS TRUNCATED, AND THE FINDING "
             "GENERALISES.** CSO serves the 2011 Demographic Report at two paths. The one "
             "the site links and a media search finds first — `/2019/03/TRINIDAD-AND-"
             "TOBAGO-2011-Demographic-Report.pdf` — is **281,190 bytes, starts `%PDF-1.4`, "
             "ends mid-stream with no `%%EOF`, and the server's `Content-Length` matches "
             "the delivered bytes exactly**. PyMuPDF opens it, sets `is_repaired=True` and "
             "reports `page_count = 0` without raising. The intact 442-page copy is live at "
             "`/2020/01/2011-Demographic-Report.pdf`. **A complete download is not an "
             "intact file — check the trailer, not the byte count** — and `sources/tt.py` "
             "asserts `%%EOF` and the page count in `fetch()`. Wayback also held two good "
             "captures, but the publisher's own second path was found first and a live copy "
             "is the citable one. `sources.md` §11t. "
             "**NO IDENTITY IN TABLE 8 IS EXACT AND THAT IS THE SOURCE, NOT THE PARSE.** "
             "CSO's 2011 figures are weighted — its own per-municipality workbooks publish "
             "fractional people — so every printed integer is independently rounded. Across "
             "the 359 identity checks `sources/tt.py` runs, the spread is **-2:4, -1:55, "
             "0:249, +1:47, +2:4**: symmetric, bounded at two people, 69% exact. The bound "
             "is asserted at 2 and the distribution is printed, because that is what would "
             "reveal a re-typeset page; a parse error is one-sided and large. "
             "**THE PARSE IS DRIVEN BY EXPECTATION AND WORKS ON TOKENS, NOT LINES.** Page "
             "168 packs three figures onto one line and puts the national row's first "
             "figure on its label line, while every other page is one figure per line. And "
             "**`-` is the nil marker**, not a missing value — a parser that skipped "
             "non-numeric tokens would shift every later figure in the row left by one and "
             "still produce nine plausible numbers. "
             "**ONE LABEL CHANGES CASE BETWEEN PANELS**: the island is `Tobago` in BOTH "
             "SEXES and `TOBAGO` in MALE and FEMALE. Matching is case-folded, which is safe "
             "only because the walk is expectation-driven — `TRINIDAD` and `TRINIDAD AND "
             "TOBAGO` are told apart by what comes next rather than by matching. "
             "**TWO NESTED UNIVERSES GIVE A TWO-LEVEL RECONCILIATION**: the 14 Trinidad "
             "municipalities sum to the printed `TRINIDAD` row, and TRINIDAD + Tobago sums "
             "to `TRINIDAD AND TOBAGO`, on all 18 columns. The table then repeats in full "
             "for MALE (p186-203) and FEMALE (p204-221), and Male + Female == Both Sexes on "
             "all 306 cells is the only check that would catch a token landing in the wrong "
             "column, since every other identity reconciles inside one panel. "
             "**THE JOIN IS 15/15 BOTH WAYS.** COD's ADM1 is the census's municipality tier "
             "exactly, independently confirmed by CSO publishing those same 15 as separate "
             "workbooks. Ten names differ — Table 8 writes `City of` and `Borough of` and "
             "uses `/` where COD uses `-` — and `fold()` handles all ten with no alias "
             "table. CSO publishes no code, so `tt.py` carries COD's pcode by name and "
             "`tt_geo.py` asserts the pairing from the other side. "
             "**PLACEMENT IS KONTUR'S 400 m GRID**, 4,610 hexes of 0.670 km². The problem "
             "here is not empty land but the unit size range — **Arima 13.1 km² to Sangre "
             "Grande 931.0 km², 71-fold** — and the small units are the dense ones. The "
             "per-municipality Kontur/census ratio runs **0.95x to 1.31x** around a national "
             "1.15x, which is a level shift across a twelve-year vintage gap rather than a "
             "shape problem, and only the shape is used. Arima and Port of Spain hold just "
             "18 hexes each; that is thin, and it is also where it matters least, which is "
             "the opposite of Saint Vincent's case where the grid was coarser than the "
             "counting units and was removed.",
    ),
    "sr": dict(
        name="Suriname",
        source="Census 7 (2004) — Algemeen Bureau voor de Statistiek",
        basis="self-identification",
        view=[-58.15, 1.80, -53.90, 6.10],
        note_public=(
            "**The most Muslim country in the Americas, and it is not close.** 13.5% of "
            "Suriname is Muslim and 19.9% is Hindu — the legacy of Javanese and Indian "
            "indenture under Dutch rule — and at 7,900 people per unit the map is fine "
            "enough to show that neither is spread evenly. **Hinduism reaches 65% in "
            "Jarikaba** and 60% in the Westelijke Polders; **Islam reaches 48% in Nieuw "
            "Amsterdam** and 47% in Lelydorp. Christianity runs the other way, up to 71% in "
            "Para Zuid and 60% in Brownsweg. With Guyana and Trinidad this is the "
            "Indo-Caribbean world the rest of the hemisphere does not have. "
            "**This is 2004, and that is the newest whole-country religion table Suriname "
            "has.** The 2012 census asked religion in far more detail — it names the "
            "Moravians, the Catholics, the Full Gospel churches, Sunni and Ahmadiyya "
            "Islam, Sanatan and Arya Hinduism separately — but published it **nationally "
            "only**, and its district reports reach three of ten districts. A ninth census "
            "was taken in 2024–25 and has published nothing yet. So the map trades twenty "
            "years of currency for the only geography that exists. "
            "**Christianity is one undivided colour here and it should not be.** Suriname's "
            "Christianity is Moravian, Catholic and Pentecostal in different places — the "
            "Evangelische Broedergemeente has been there since 1735 — and none of that can "
            "be drawn from this table. It is not filled in from the 2012 national figures, "
            "because that would invent where each denomination lives. "
            "**And 15.7% answered 'don't know' or nothing at all** — the largest non-answer "
            "on this map. Every share here is a share of everybody, so a religion's share "
            "among people who answered is about a fifth higher than what is drawn. "
            "**What is missing has a name.** *Winti*, the Afro-Surinamese religion of the "
            "Maroon and Creole populations and the sibling of Vodou and Candomblé, has no "
            "cell of its own: it sits inside a combined `Traditional religion and other` "
            "category with indigenous Amerindian religion, Judaism and the Jehovah's "
            "Witnesses. It was **illegal in Suriname until 1971** and is widely practised "
            "alongside a church, so a one-answer census undercounts it twice over."),
        how="census, 2004; the 2012 and 2024 censuses publish religion nationally only",
        grain="ressorten, 7,900 people on average",
        counts=_sr_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "sr" / "sr_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_sr_place_weight,
        note="**§11t CALLED THE VINTAGE A FORK AND IT IS NOT ONE.** The choice looked like "
             "fine-geography-2004 against deep-categories-2012; it was checked and 2012 has "
             "no usable geography at all. Volume 1 carries the full denominational list "
             "cut by ethnicity and nationality and by **no geography**; the "
             "Districtsresultaten presentations carry religion for **3 of 10 districts** "
             "(Volume III only) — verified against the text layer, since all three volumes "
             "have one, so the absence is the source's and not a scan artefact; and Census "
             "9, fielded to July 2025, has published nothing across 3,662 media items. 2004 "
             "is the only whole-country sub-national religion table Suriname has. "
             "**THE FILE HAS NO DISTRICT COLUMN AND THE NAMES ARE NOT UNIQUE.** The ressort "
             "workbook is 62 unlabelled columns; there is a `Welgelegen` in both Paramaribo "
             "and Coronie and a `Centrum` in both Paramaribo and Brokopondo, so a name-only "
             "join collides on four units. The sibling `district-profiel-census.xls` "
             "publishes **ressorten per district** — 12, 7, 5, 3, 6, 6, 6, 5, 6, 6 — which "
             "sums to 62 and consumes the columns in order, which is what makes the "
             "district assignment a read rather than a guess. Asserted, not assumed. "
             "**FOUR NAMES DO NOT FOLD ONTO COD'S AND NO ALIAS TABLE IS WRITTEN.** "
             "`Koewarasan`/`Kwarasan`, `Moengo Tapoe`/`Moengo Tapu`, "
             "`Marchallkreeek`/`Marechallkreek` (ABS's typo, three e's, transcribed as "
             "printed) and `Coeroeni`/`Coeroenie`. The exact fold runs first, and then a "
             "district with exactly one unmatched census ressort and exactly one unmatched "
             "polygon has them paired **by elimination** — a derivation that re-runs every "
             "time and stops if a future vintage leaves two of either, which is the "
             "§12-safe form of what a frozen alias list does badly. 58 folded, 4 forced, "
             "62/62 both ways. "
             "**AN EXACT PARTITION IN BOTH DIRECTIONS**, integers, no rounding and no "
             "suppression: the six categories sum to each ressort's own total on all 63 "
             "columns, and the 62 ressorten sum to the national column on all 7 rows. "
             "**PLACEMENT IS KONTUR'S 400 m GRID**, 5,689 hexes, and Suriname needs it more "
             "than anywhere: 163,820 km² with ~90% of the people on the coastal strip, and "
             "three interior Sipaliwini ressorten together larger than the Netherlands. "
             "The national ratio is **1.274x**, which is expected rather than alarming — a "
             "2023 grid against a **2004** census, the widest vintage gap on this map — so "
             "the band is asserted at [1.00, 1.60] rather than around 1.0. "
             "**NAME WHERE IT IS WORST.** The per-ressort ratio runs 0.06x to 2.86x, which "
             "is wide because nineteen years of Suriname's growth went to Wanica and "
             "Paramaribo. **Galibi is the worst-placed unit on this country** — 4 hexes and "
             "43 modelled people against a real census population — so its dots sit on "
             "essentially no weighting at all. Galibi is the Kalina (Carib) village area at "
             "the Marowijne mouth, and a building-footprint model reads it as empty. The "
             "dots are still inside Galibi and still the right number (§9t); only the "
             "surface under them is bad.",
    ),
    "bs": dict(
        name="The Bahamas",
        name_in="the Bahamas",
        source="2022 Census of Population and Housing (Bahamas National Statistical "
               "Institute)",
        basis="self-identification",
        view=[-79.60, 20.75, -72.40, 27.45],
        gap=("the smallest religions on 17 of the 18 islands, pooled into one Other religion "
             "cell"),
        note_public=(
            "**The most Baptist country on this map, and it is not close.** One Bahamian in "
            "three — 135,875 people, **35.8% of everyone drawn here**. The next three are "
            "Saint Vincent at 9.3%, the United States at 7.3% and Jamaica at 6.9%, so the "
            "Bahamas is nearly four times the runner-up. Nothing about being Caribbean or "
            "Anglophone predicts it. "
            "**Two islands three kilometres apart are nothing like each other or the "
            "country.** Off the north end of Eleuthera, **Harbour Island is 32.0% Roman "
            "Catholic and 1.2% Baptist** — the only place in the Bahamas where the national "
            "religion is a rounding error — while **Spanish Wells is 23.2% Brethren**, "
            "against 1.5% nationally, a sixteenfold concentration of a church that barely "
            "registers anywhere else, alongside Methodists at 25.6% and almost no Catholics "
            "or Anglicans at all. Spanish Wells is a Loyalist fishing settlement and "
            "Harbour Island's Dunmore Town is the old colonial capital; the two histories "
            "are still legible in the answers. "
            "**Anglicanism is a Family Island religion here, not a Nassau one.** 43.2% of "
            "Long Island, 34.9% of Inagua, 34.7% of the Berry Islands, against 11.2% on "
            "New Providence. Seventh Day Adventists are **29.7% of Crooked Island**. "
            "Mayaguana is **73.9% Baptist**. "
            "**And Rastafari runs the opposite way from every stereotype about it** — "
            "highest on Cat Island (1.02%) and Andros (1.01%), lowest in Nassau (0.24%). "
            "It is the fourth census count of Rastafari on this map, after Jamaica, Saint "
            "Vincent and Trinidad. "
            "**This is the first census outside the United States to count African "
            "Methodists under their own name** — 1,028 people, and 282 of them on "
            "Eleuthera, which is 3.1% of that island against 0.02% of Grand Bahama. "
            "**It is also one of the very few that separates `no religion` from `atheist`**, "
            "and offers both boxes: 24,668 people take the first and 281 the second. Most "
            "censuses collapse the two and this map can then only show the first. "
            "**Three quarters of the country is one unit.** New Providence holds 296,732 of "
            "398,165 people, so for most Bahamians this map shows the composition of a "
            "single island spread across Nassau by where people live, not by what they "
            "answer street by street. The seventeen Family Islands are where the geography "
            "is real. "
            "**On seventeen of the eighteen islands the census pools its smallest answers.** "
            "Only New Providence prints all 24 religions; everywhere else the rarest go "
            "into one `Other religion` cell, with a footnote naming which. It is 372 people "
            "nationally — 0.09% — but 30.4% of Ragged Island and 11.8% of Mayaguana, and on "
            "those two the pooled cell swallowed the `no religion` answers too, so neither "
            "island reports any. "
            "**4.8% stated no religion at all** and are not drawn; that is 23.2% on "
            "Acklins, which the census does not explain."),
        how="census, 2022",
        grain="islands, 22,000 people on average",
        counts=_bs_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "bs" / "bs_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_bs_place_weight,
        note="**THE COUNTRY WAS 'OPEN' BECAUSE THE PUBLISHER'S OWN LISTING HIDES THE "
             "SOURCE.** §11t left the Bahamas as a lead: BNSI's first release — the one "
             "`bnsistats.gov.bs/publications` lists and the one a search finds — carries "
             "religion for **All Bahamas only**, and says so in its preface. The **579-page "
             "All-Island Report**, published June 2026 with a religion table per island, is "
             "absent from that listing and was found on the **CARICOM regional mirror** "
             "(§11v), which turned out to hold the census reports of the whole region. It "
             "IS on BNSI's own CDN, and that is what `sources/bs.py` fetches; the mirror is "
             "how the file becomes findable, not where it has to be cited from. "
             "**THE TABLE NUMBER IS THE CENSUS'S OWN ISLAND CODE.** The 2022 questionnaire "
             "pre-fills `Name of Island` from a numbered list of 18 and Tables 12.1-12.18 "
             "come in exactly that order, so `geo_id` is BNSI's and not invented here. The "
             "printed caption is checked against it on every run. "
             "**THE CATEGORY LIST IS NOT FIXED ACROSS ISLANDS**, which is the structural "
             "surprise and why the parse validates labels against a set instead of walking "
             "a fixed sequence the way `sources/tt.py` does. See `gap` and `taxonomy/"
             "bs2022.py`; the residual is 372 people and every island's footnote names its "
             "contents, which `bs.csv` carries in the `note` column. "
             "**FOUR RECONCILIATIONS, AND THE OUTER ONES ARE CROSS-DOCUMENT.** The seven "
             "age bands sum to each cell's own TOTAL column on all 753 cells — a check on "
             "every figure read rather than on the margins — Male + Female == TOTAL on 251, "
             "the categories sum to each island's own total on 18, and **the 18 islands sum "
             "to the FIRST RELEASE's national Table 6.0 category by category**, with the "
             "shortfall on each category exactly equal to what the island footnotes say was "
             "pooled. Two separately published tabulations agreeing to the person. "
             "**TWO CROSS-DOCUMENT ALIASES, ONE OF THEM BNSI'S TYPO.** The first release "
             "writes `Church of God (including Church of God of Prophecy)` where the "
             "All-Island Report writes `... AND Church of God of Prophecy`, and it spells "
             "the atheist row **`Athiest`**. Both are held in a two-entry table rather than "
             "solved by fuzzy matching, which would happily pair `Church of God` with "
             "`Church of God of Prophecy` if BNSI ever split them. "
             "**NOBODY PUBLISHES THE CENSUS'S OWN TIER, SO IT IS DISSOLVED.** Every "
             "boundary set for the Bahamas — COD-AB and geoBoundaries are the same geometry "
             "— gives the **32 local-government districts**, which nest inside the 18 "
             "islands exactly. Twenty-six carry their island in the name; the six cays that "
             "do not were each checked against BNSI's own publications rather than a map — "
             "**Black Point is enumeration district 420201 in `EXUMA AND CAYS POPULATION BY "
             "SETTLEMENT: 2010`**, and Mangrove Cay is named in the 2010 ANDROS report. "
             "**Harbour Island and Spanish Wells are census islands in their own right**, "
             "not part of Eleuthera, which is where the census tier and the geographic "
             "intuition disagree. The partition is asserted both ways. "
             "**PLACEMENT IS KONTUR'S 400 m GRID, AND 6% OF IT MISSES THE COUNTRY.** A "
             "plain `within` join leaves 836 hexes and 25,095 modelled people outside every "
             "island, because COD-AB's coastline is generalised GDAMS 2009 and a hex is "
             "400 m across. Measured before deciding: **all but three of those people are "
             "within 500 m of an island**, and nine of the twelve heaviest are Nassau's own "
             "waterfront. In a country where the population IS the coastline, dropping them "
             "tilts every island's dots inland — so they are snapped to the nearest island "
             "within **1 km**, a threshold that sits in an empty gap in the measured "
             "distribution rather than through the middle of it. The histogram prints on "
             "every run. "
             "**THE PER-ISLAND KONTUR/CENSUS RATIO RUNS 0.80x TO 1.92x** around a national "
             "1.036x. Exuma at 1.92x and Abaco at 1.32x are second homes and resorts read "
             "as population by a building-footprint model, not a grouping error; only the "
             "within-island shape is used, so no island gets the wrong number of dots "
             "(§9t).",
    ),
    "ky": dict(
        name="Cayman Islands",
        name_in="the Cayman Islands",
        source="2021 Census of Population and Housing (Economics and Statistics Office)",
        basis="self-identification",
        view=[-81.50, 19.20, -79.65, 19.82],
        gap=("the institutional population and a national non-response estimate, 3.7% together, "
             "which are outside every table in the report"),
        note_public=(
            "**The most evenly religious country on this map.** The largest answer reaches "
            "only 19.5%, and the top five are five different things — a Holiness church, no "
            "religion at all, Roman Catholicism, Adventism and non-denominational "
            "Christianity. Nowhere else drawn here is this flat. "
            "**The reason is that over half the residents were born abroad**, and almost "
            "every category's geography is really a map of that. Roman Catholicism is 18.4% "
            "in George Town against 3.6% in North Side — the Filipino and Latin American "
            "workforce in the capital. The Hindu share peaks at 6.6% in East End. "
            "**The Church of God is the exception and is effectively the national church**: "
            "27.2% in North Side, 25.3% in Bodden Town, 23.5% on the Sister Islands, 20.9% "
            "in East End, 18.3% in West Bay, 16.8% in George Town. Nothing else is that "
            "even. It is the **Anderson, Indiana** church — Holiness rather than "
            "Pentecostal, and the same body Jamaica's census counts separately — which "
            "arrived through the Cayman Islands Regional Mission Council. "
            "**And Cayman Brac is a different country.** The Sister Islands are **30.5% "
            "Baptist** against 2.8% to 9.5% in every Grand Cayman district, the sharpest "
            "single contrast in the territory: the old Brac Baptist settlement still "
            "visible through a population that immigration has otherwise remade. They are "
            "also the least Presbyterian place here, at 0.66% against 13.8% in North Side. "
            "**16.7% report no religion**, the second largest answer, and it is highest in "
            "West Bay and East End rather than in the capital. "
            "**3.7% of the territory is outside this map before any of that.** Every table "
            "in the census report runs on what the office calls the *census survey tabular "
            "population count* — 68,811 of the 71,432 people counted. The difference is 327 "
            "people living in institutions and a **2,294-person non-response estimate**, "
            "weighted from household refusals and verified no-contacts, that exists only as "
            "a national figure and so cannot be put anywhere on a map."),
        how="census, 2021",
        grain="districts, 11,500 people on average",
        counts=_ky_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ky" / "ky_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ky_place_weight,
        note="**THE ROW LIST IS NOT FIXED AND NEITHER IS THE COLUMN COUNT.** Two "
             "irregularities, neither announced. **North Side's table has no `Muslim` row "
             "at all** — not a zero, not a dash, the row is absent — and **Table 4.10F "
             "(Sister Islands) has eleven figures per row where the other five have "
             "twelve**, dropping the Non-Caymanian DK/NS column. So the parse can assume "
             "neither a fixed row sequence (the Bahamas' problem, §9ar) nor a fixed column "
             "count: it reads a row as `label, then a run of figures`, takes the first, and "
             "asserts the run length is constant WITHIN a table. "
             "**AN OMITTED ROW IS NOT A ZERO, AND THE NATIONAL TABLE PRICES IT.** The six "
             "districts are short by exactly 3 on `Muslim` and North Side is the only "
             "district omitting it, which pins the missing cell at 3 people. **It is not "
             "added back** (§14.4): the value is implied by a residual rather than "
             "published, and ESO prints a dash for a genuine zero elsewhere in the same "
             "table, so the omission means something it does not say. The map draws North "
             "Side with no Muslims. "
             "**ESO'S TABLES DO NOT INTERNALLY RECONCILE, BY ONE TO FIVE PEOPLE.** A "
             "district's rows fall 1 short of its own printed Total in George Town, Bodden "
             "Town and the Sister Islands and 1 over in East End; North Side is 5 short, of "
             "which 3 is the omitted Muslim row. The NATIONAL table is internally exact, so "
             "the discrepancy is the district tables' and not the category list's. "
             "Two-sided and bounded at five people on a 68,811-person table; the whole "
             "spread prints on every run rather than being absorbed by a tolerance. "
             "**THE BOUNDARIES ARE WRONG AND WERE CHOSEN ANYWAY, ON A MEASUREMENT.** "
             "COD-AB's ADM1 is ESO's six districts exactly — same names, `Sister Islands` "
             "included, 6/6 both ways with the pcodes cross-checked — but its **Bodden Town "
             "is an 8.3 km² coastal strip** and its North Side reaches south across the "
             "island. OSM carries the same six districts at `admin_level=8`, so this was a "
             "choice. Both were tested. **Settlements**: all 69 OSM place nodes located in "
             "both sets, 60 agree and 9 do not — COD wrongly puts six eastern Bodden Town "
             "villages (Breakers, Northward, Frank Sound, Midland Acres, Pease Bay, Belford "
             "Estates) in North Side, and OSM wrongly puts Savannah and Pedro Castle in "
             "George Town. **Population**: Kontur summed per polygon against ESO's own "
             "district counts gives a total absolute error of **11,785 for COD against "
             "25,748 for OSM** — COD's mistakes are on villages, OSM's are on a town. COD "
             "is used, and `sources/ky_geo.py` re-runs the settlement test every build and "
             "asserts the three known failures are exactly those three, so a reissue that "
             "fixes or breaks one stops the run. "
             "**THE VISIBLE SYMPTOM IS THE RATIO TABLE**: North Side reads 2.03x its census "
             "population and Bodden Town 0.86x, West Bay 0.78x. That is the boundary error, "
             "not Kontur — nationally the grid is 1.007x. Only the within-district shape is "
             "used, so no district gets the wrong number of dots (§9t). "
             "**PLACEMENT IS THE THINNEST KONTUR EXTRACT ON THE MAP**, 392 hexes for the "
             "whole country — but the units are 8-89 km² and a hex is 0.67 km², so the grid "
             "is still finer than the tier it weights, which is the test Saint Vincent "
             "failed (§9ac). 6.29% of it lands outside every district and **none of that is "
             "genuinely offshore** — the measured maximum distance is 500 m — so it is "
             "snapped to the nearest district within 1 km, as in the Bahamas.",
    ),
    "bb": dict(
        name="Barbados",
        source="2010 Population and Housing Census (Barbados Statistical Service)",
        basis="self-identification",
        view=[-59.72, 13.02, -59.37, 13.36],
        gap=("the census's own 18% undercount; 49,115 people were never enumerated, unevenly "
             "between parishes"),
        note_public=(
            "**The most Anglican country on this map.** 23.9%, against 13.9% in Saint "
            "Vincent, 11.9% in the Bahamas, 5.7% in Trinidad and 2.8% in Jamaica. "
            "\"Little England\" is still legible in the answers: **St. John is 37.5% "
            "Anglican**, and the parish churches are the oldest institutions on the island. "
            "**The two big answers run opposite to each other across the island.** "
            "Anglicanism peaks in St. John (37.5%) and bottoms in St. Andrew (15.3%); the "
            "Pentecostal churches do the reverse — **29.4% of St. Andrew** against 19.5% "
            "nationally. The established church holds the south and east, the Pentecostals "
            "the rugged, poorer north-centre. "
            "**And this is the only census outside the United States that counts Nazarenes, "
            "Wesleyans and the Salvation Army as three separate answers.** 7,299 "
            "Nazarenes, 7,694 Wesleyans, 878 Salvationists, and a fourth Holiness cell — "
            "`Church of God` — on top: **9.4% of the country in one church family, split "
            "four ways**, which nothing else here does at all. They are not spread evenly. "
            "**St. Lucy is 13.4% Adventist** — more than twice the national 5.9% — and "
            "**2.3% Salvation Army against 0.4%**, six times the national share, in the "
            "northernmost parish. **St. Joseph is 7.2% Wesleyan** and **St. Thomas 5.7% "
            "Moravian**, against 1.2% nationally, which is the old Moravian mission field. "
            "**Roman Catholics are 3.8%** — very low for the Caribbean, and a reminder that "
            "Barbados was never Spanish or French. **20.6% report no religious "
            "affiliation**, the second largest answer, highest in St. Joseph (25.8%) and "
            "St. Michael (23.6%). "
            "**This is 2010, and Barbados has held a census since.** The 2021 census "
            "counted only **136,415 of an estimated 269,090 people — a 48.7% undercount** — "
            "and its own report says of the parish tables that \"most results at that level "
            "would be understated\". So the newer census is the worse map, and the older one "
            "is drawn. "
            "**Even 2010 is missing 18% of Barbados**, and unevenly: the census reached "
            "96.1% of St. John and only 74.6% of St. James. Nothing here scales the "
            "parishes back up, so an under-counted parish shows proportionally fewer dots "
            "than its true population warrants. What the composition inside each parish "
            "shows is unaffected."),
        how="census, 2010",
        grain="parishes, 20,600 people on average",
        counts=_bb_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "bb" / "bb_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_bb_place_weight,
        note="**THE NEWER CENSUS WAS REJECTED, ON THE PUBLISHER'S OWN WARNING.** BSS ran a "
             "2021 census and published the same Table 02.06 — *Total Population by Parish, "
             "Sex and Religion* — in `Census-2020-Tables.xlsx`, with the same 23 categories. "
             "It is not used: **2021 tabulated 136,415 of an estimated 269,090, a 48.7% "
             "undercount**, against 2010's 226,193 of 277,821 and 18%. And the 2021 report "
             "says outright that *\"in most cases, disaggregation by area is not included – "
             "as most results at that level would be understated, considering the "
             "significant size of the undercount\"* — while the workbook ships the parish "
             "cut anyway. The publisher's warning is taken over the publisher's "
             "spreadsheet. "
             "**THE `NO RELIGION` COLUMN HAS NO HEADER AND IT IS 20.6% OF THE COUNTRY.** In "
             "the 2010 sheet, column 23 sits between `Other Non-Christian` and `Not Stated` "
             "and its header cell is **blank** — 46,562 people, the second largest answer. "
             "A header-driven read names it `Unnamed: 23` or drops it. Two things identify "
             "it and `sources/bb.py` asserts both: the categories only sum to each unit's "
             "own `Total` when it is included, on all 36 rows; and **the 2021 workbook** "
             "publishes the same categories in the same relative order with that position "
             "labelled `No Religious Affiliation`. So the 2021 file, rejected as a source, "
             "is the evidence for reading the 2010 one. "
             "**KONTUR INDEPENDENTLY REPRODUCES THE CENSUS'S OWN UNDERCOUNT PATTERN, WHICH "
             "IS THE STRONGEST CHECK ON THIS COUNTRY.** BSS publishes an estimated resident "
             "population per parish, so the census's coverage can be computed: 74.6% of "
             "St. James up to 96.1% of St. John. Kontur's building-footprint grid knows "
             "nothing about any of that, and its modelled-to-tabulated ratio per parish "
             "correlates with the implied undercount factor at **r = +0.863** — a "
             "relationship 20,000 random relabellings of the parishes reproduce 0.05% of "
             "the time. Two unrelated sources agreeing on which parishes were "
             "under-enumerated. It also means the ratio table in `sources/bb_grid.py` is a "
             "COVERAGE read rather than a shape check, and is expected to sit above 1. "
             "**NOTHING IS SCALED UP** (§14.4): correcting the parishes would assume the "
             "missed 18% has the same religion mix as the counted 82%, and nothing "
             "establishes that. "
             "**THE JOIN IS 11/11 BOTH WAYS.** COD-AB's ADM1 is the parish tier exactly, "
             "and the parishes are the only sub-national geography Barbados publishes at "
             "all. Ten of the eleven names differ only as `St.` against `Saint`, handled by "
             "`fold()` rather than an alias table; BSS publishes no code, so `bb.py` carries "
             "COD's pcode by name and `bb_geo.py` asserts the pairing from the other side. "
             "**PLACEMENT IS KONTUR'S 400 m GRID**, 705 hexes. Barbados is the densest "
             "country on this map and its parishes are unusually uniform in area — 23.9 to "
             "62.5 km², a 2.6-fold spread against Trinidad's 71-fold — so the grid is not "
             "here for empty land or unit size. It is here for **St. Michael**, which holds "
             "a third of the country on 40.7 km², nearly all of it in Bridgetown and the "
             "south-west coastal belt.",
    ),
    "lc": dict(
        name="Saint Lucia",
        source="2022 Population and Housing Census, provisional release 2 "
               "(Central Statistics Office)",
        basis="self-identification",
        view=[-61.09, 13.69, -60.86, 14.12],
        note_public=(
            "**The most Catholic country on this map, and the one losing it fastest.** "
            "50.6% Roman Catholic — against 31.5% in Grenada, 21.6% in Trinidad and 3.8% "
            "in Barbados — in an island Britain and France traded fourteen times and where "
            "the French church stayed after the British navy left. "
            "**The census's own back-series is the finding.** Saint Lucia was **92.4% "
            "Catholic in 1960**, 85.6% in 1980, 67.5% in 2001, 61.1% in 2010 and 50.6% in "
            "2022: forty-two points in sixty-two years, and still nine points a decade at "
            "the end. About half of what left went to two churches — Seventh Day "
            "Adventists rose 1.8% to 10.8% and Pentecostals 0.0% to 9.0% over the same "
            "span — and about half went to no church at all. "
            "**And the church held where the roads did not.** Catholicism is 71.7% of "
            "Choiseul, 70.1% of Soufriere and 66.1% of Canaries, the south-west coast, "
            "against **44.1% in Anse La Raye, 44.8% in Castries and 45.8% in Gros Islet** "
            "— the north-west, where nearly two thirds of Saint Lucians now live. The "
            "Adventists run the other way: 18.6% of Anse La Raye and 16.3% of Canaries "
            "against 5.3% of Soufriere. "
            "**This census asks whether you believe in God and whether you belong to "
            "anything, separately, which almost nothing else here does.** 14.1% answer "
            "*no religion but believe in God* and **0.30% answer *do not believe in "
            "God*** — a 47-fold gap, and the strongest evidence on this map that "
            "Caribbean irreligion is lapsed affiliation rather than atheism. The "
            "non-affiliated are the mirror image of the Catholics: 16.2% of Castries "
            "against **4.8% of Choiseul**, the most Catholic district in the country. "
            "**Anglicans are 1.3% here and 23.9% in Barbados**, which is the whole "
            "difference between an island the British kept and an island they only "
            "captured. Where they are is odd and worth a look: Choiseul (3.7%) and Laborie "
            "(2.8%), in the Catholic south rather than the anglophone north. "
            "**One row on this map does not say what the census says.** The report prints "
            "2.2% of the country as `Mennonite`; the census's own questionnaire calls that "
            "option `Evangelical`, and it is drawn as Evangelical here. The note below sets "
            "out why."),
        how="census, 2022",
        grain="districts, 17,200 people on average",
        counts=_lc_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "lc" / "lc_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_lc_place_weight,
        note="**THE `Mennonite` ROW IS THE FORM'S `Evangelical` OPTION, AND IT IS 2.2% OF "
             "THE COUNTRY — 3,760 PEOPLE.** This is the largest reading decision on any "
             "Caribbean source here and it is machine-checked rather than argued. **CSO "
             "publishes the 2022 census instrument** (*St Lucia Census 2022, Version 4*) on "
             "its own site, and question 1.5 offers 22 options. Table D.2's 23 rows are "
             "those 22 options **in the same order**, plus `Not reported`; twenty-two of "
             "the twenty-three match one for one, and the one that does not is **option 6, "
             "`Evangelical` on the form and `Mennonite` in the report**. **The 2010 census "
             "agrees**: its Table 40 has `Evangelical` at the same 2.2%, in the same place "
             "in a similar list, and no Mennonite row at all — while the 2022 report has no "
             "Evangelical row at all. And there is no Mennonite community of 3,760 people "
             "in Saint Lucia; Grenada's 2021 census, whose form has BOTH cells, counts 280 "
             "Mennonites (0.26%) beside 2.36% Evangelical, which is what the real pair "
             "looks like in this region. `sources/lc.py` asserts the whole shape of the "
             "discrepancy on every run and refuses to build if it changes. **The normalised "
             "file still carries the source's own label**; the reading happens in the "
             "mapping module, which is why the two layers are separate. "
             "**THE PUBLISHED FIGURES ARE ALREADY CORRECTED FOR UNDERCOUNT, AND THAT IS "
             "CSO'S DOING RATHER THAN THIS MAP'S.** The enumeration was **23.3% short** and "
             "CSO applied per-district weight factors — 1.107 in Anse La Raye up to 1.507 "
             "in Laborie — to reach *estimated full values*, which are what every table in "
             "the report holds. **That is the exact inverse of Barbados**, where BSS "
             "publishes the raw count, warns that its parish tables are understated, and "
             "this project declines to scale them (§14.4). Nothing here scales anything "
             "either. It is also why the cells miss their own margins by one to three "
             "people — independently rounded estimates — so the drawn total is 171,829 "
             "against a published 171,834. "
             "**THE FORM ASKS ABOUT HINDUISM TWICE.** Options 7 and 19 are `Hindu` and "
             "`Hinduism`, 253 and 66 people, and they are drawn at one node. A duplicated "
             "option, not two religions. "
             "**THE JOIN IS 10/10 BOTH WAYS**, on COD-AB's ADM1, which is the government's "
             "own boundary file and is CSO's district tier exactly. One name differs, by a "
             "hyphen. **COD's district AREAS do not match the census's own** — Dennery is "
             "1.71x CSO's published figure and Soufriere 0.73x — which in Cayman was a "
             "boundary error that moved dots. **Here it is not, and that was measured:** "
             "geoBoundaries' independent set matches CSO's areas within 8.3%, the two sets "
             "agree on 528 of COD's own 547 settlements, and summing Kontur's population "
             "grid inside each gives the same answer district by district — Dennery holds "
             "10,581 people under COD and 10,571 under geoBoundaries **despite COD's "
             "Dennery being 51 km² larger**. The extra land is the Central Forest Reserve "
             "and nobody lives in it. "
             "**PLACEMENT IS KONTUR'S 400 m GRID**, 772 hexes, and it is here because Saint "
             "Lucia is a volcanic ridge with every settlement on the coast road — Castries "
             "alone holds 60,614 people on 87.4 km², nearly all of them between Bois "
             "d'Orange and Cul de Sac. The grid and the census agree nationally (1.07x) and "
             "not per district (0.58x in Laborie to 1.54x in Anse La Raye); only the shape "
             "*within* a district is used, so no district gets the wrong number of dots. "
             "**Both methods find fewer people in the rural south than CSO's correction "
             "asserts**, which is worth recording and is not something this map can settle. "
             "**A FINER GEOGRAPHY EXISTS AND CARRIES NO RELIGION.** COD's ADM2 has 547 "
             "settlements, ~310 people each, which would be among the finest tiers on this "
             "map. Nothing in the report cuts religion below the district, so it is not "
             "used: it would be placement with no counts to place.",
    ),
    "gd": dict(
        name="Grenada",
        source="2021 Housing and Population Census, preliminary results "
               "(Central Statistical Office)",
        basis="self-identification",
        view=[-61.82, 11.97, -61.36, 12.55],
        note_public=(
            "**The most evenly divided country in the Caribbean, on this map.** Grenada's "
            "largest religion is **31.5%** — against Saint Lucia's 50.6% Catholic 150 km "
            "north, the Bahamas' 34.9% Baptist and Barbados's 23.9% Anglican — and four "
            "bodies hold more than 7% each, no two of them the same tradition: Roman "
            "Catholic 31.5%, Pentecostal 19.9%, Seventh Day Adventist 12.3%, Anglican "
            "7.3%. The French and the British each left a church behind, and the "
            "twentieth-century missions landed on top of the pair. "
            "**The Adventists have the north and the Pentecostals the south, and they "
            "barely overlap.** Seventh Day Adventists are **24.1% of St. Andrew and 22.6% "
            "of St. Mark** against **7.3% of St. George**; Pentecostals are **25.9% of "
            "St. David and 23.0% of St. Andrew** against **7.2% of Carriacou**. Both are "
            "national churches with regional hearts. "
            "**Carriacou is a different island in this sense too.** **22.1% Anglican** — "
            "seven times St. David's 3.0% — and 40.6% Roman Catholic, with Pentecostals at "
            "a third of their national share. Three centuries of Scottish and English "
            "settlement in the Grenadines, still legible, and the sharpest single-unit "
            "signal in the country. "
            "**This census names twenty-five answers**, one of the deepest lists in the "
            "region: `Spiritual Baptist`, `Mennonite`, `Lutheran`, `Moravian`, "
            "`Presbyterian`, `Independent Baptiste`, `Evangelical`, `Church of God`, "
            "`Buddhist`, `Bahai`, `Hindu`, `Muslim` and `Rastafarian` all have cells of "
            "their own. Several are one parish each: **Presbyterians are 3.2% of St. Mark** "
            "and 0.02% of Carriacou, the old Scottish mission on the west coast; **Church "
            "of God is 8.5% of St. Andrew** and 0.7% of St. Mark; **Spiritual Baptists are "
            "4.8% of St. Mark**, the Afro-Caribbean tradition that Trinidad banned by "
            "ordinance from 1917 to 1951. "
            "**Grenada also splits disbelief from non-affiliation** — 5.95% report no "
            "religious affiliation and **0.05% report atheism**, a 130-fold gap, the widest "
            "on this map from a census that offered both boxes. Saint Lucia's form asks the "
            "same pair and gets the same answer. "
            "**7.1% did not answer**, and where they are is the most striking thing the map "
            "cannot show: the capital. See the note below."),
        how="census, 2021",
        grain="parishes, 15,500 people on average",
        counts=_gd_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "gd" / "gd_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_gd_place_weight,
        note="**THE COUNTING TIER IS 8 UNITS AND THE MAP DRAWS 7, BECAUSE NOBODY PUBLISHES "
             "A BOUNDARY FOR THE CAPITAL.** The census reports the **Town of St. George** "
             "— 2,681 people — apart from the rest of the parish, and no boundary set has "
             "it: COD-AB's ADM1 is the six parishes plus Carriacou and Petite Martinique, "
             "and OpenStreetMap has the six parishes at `admin_level=6` and, for the town, "
             "only a `place=town` **node**. So the two halves are folded back into one "
             "St. George. **What that hides is this country's sharpest number**: the town "
             "declined the religion question at **15.8%** against 9.9% for the rest of the "
             "parish and 0.91% in St. Mark. `gd.csv` carries BOTH tiers — the census's own "
             "8 units and the 7 drawn ones — so nothing is lost from the record and the "
             "split is already there if a town boundary ever appears. "
             "**CARRIACOU AND PETITE MARTINIQUE ARE ONE UNIT BECAUSE THE CENSUS MAKES THEM "
             "ONE.** COD-AB gives them separate polygons and the census publishes a single "
             "figure, so the polygons are dissolved; splitting one published number between "
             "two islands would be inventing a magnitude (§14.4). After both moves the "
             "tiers agree exactly, 7 on 7, which is also what OSM independently has. "
             "**THE TABLE RECONCILES TO THE PERSON**, in both directions and on every row — "
             "which Saint Lucia's does not and Cayman's does not. **And the universe is "
             "almost the whole country**: 108,279 of a census 109,021, the difference being "
             "690 people in institutions and 52 homeless. 99.3% of Grenada is inside the "
             "table before the refusals come out, against 96.3% in Cayman and 81.4% in "
             "Barbados. "
             "**`NOT STATED` IS 7.1% AND IS NOT AN UNDERCOUNT.** These 7,698 people were "
             "counted and declined the question; they are marked, not filled (§3.5). "
             "**THE REPORT'S TEXT LAYER SUBSTITUTES `Ǫ` FOR `Q`** — `MARTINIǪUE` — which "
             "is invisible on the page and breaks any comparison against a typed name. "
             "`sources/gd.py` folds it back. `MORMOM` and `BAPTISTE` are CSO's own "
             "spellings and are left alone in the data. "
             "**`CHURCH OF GOD` IS FILED AT THE HOLINESS PARENT**, 3.6% of the country, "
             "because the name cannot decide between the Cleveland (Pentecostal) and "
             "Anderson (Holiness) lines and — unlike Cayman — no external evidence names "
             "which body Grenada means. **`SPIRITUAL BAPTIST` IS NOT FILED WITH THE "
             "BAPTISTS**: CSO offers it and `Independent Baptiste` as two separate answers, "
             "so merging them would undo a distinction the source drew. "
             "**PLACEMENT IS KONTUR'S 400 m GRID**, 509 hexes, and it does two different "
             "jobs here — St. George holds 41% of Grenada on 65.8 km², nearly all of it "
             "between St. George's town and Point Salines; and Carriacou and Petite "
             "Martinique are one unit on two islands 2.4 km apart, where scattering by area "
             "would put far too many people on the smaller one.",
    ),
    "th": dict(
        name="Thailand",
        source="2010 Population and Housing Census (NSO), Table 4 and the provincial sheets",
        basis="self-identification",
        view=[97.2, 5.5, 105.8, 20.6],
        note_public=(
            "**Thailand is 93.6% Buddhist and that is the least interesting thing about "
            "this map.** 56 of its 76 provinces are over 95% Buddhist, and everything worth "
            "looking at is in the twenty that are not. "
            "**The Muslim south is a gradient down the peninsula, not a border.** "
            "Narathiwat is 85.9% Muslim, Pattani 84.4%, Yala 76.6% and Satun 67.1% — the "
            "provinces of the old Sultanate of Patani, annexed in 1909, where the everyday "
            "language is Patani Malay rather than Thai and the Islam is Shafi'i Sunni. But "
            "it does not stop there: **Krabi is 34.6%, Songkhla 25.3%, Phangnga 22.1%, "
            "Phuket 16.0% and Phatthalung 11.7%**, thinning steadily northward against 4.9% "
            "nationally. The Andaman coast has been Muslim for as long as the deep south "
            "has, and a map drawn only from the four border provinces misses half of it. "
            "**Christianity is a highland religion here, not an urban one.** The North is "
            "3.05% Christian against 0.50% in the Northeast, because the Karen, Lahu, Lisu "
            "and Akha of the hills were reached by missions from the 1880s while the "
            "lowland Thai were not. Those are the same peoples China draws across the "
            "border in Yunnan, and this is the census that counts them rather than "
            "inferring them from ethnicity. **Mae Hong Son, on the Myanmar border, comes "
            "out the most Christian province in the country** — but see the last paragraph, "
            "because that figure is inferred rather than counted. "
            "**Bangkok is where the small religions are.** It holds most of the country's "
            "Hindus and Sikhs — the Punjabi merchant community of Phahurat and the Tamil "
            "community around Silom — and at 4.6% Muslim it has more Muslims than any "
            "province outside the south. "
            "**What the census cannot show, and it is most of Thai religious practice.** "
            "The form asks which religion you belong to, and in a country where Buddhist "
            "identity is close to civic default that question does not reach the spirit "
            "houses outside every building, the Brahmanical court ritual, the Chinese "
            "temple practice of the Thai Chinese, or the phi that the same household "
            "attends to alongside the wat. **0.07% of Thailand answers `no religion`, the "
            "smallest such share on this map**, and that is a fact about the question "
            "rather than about the country. "
            "**And only two of the nine categories are counted where you see them.** "
            "Buddhist and Muslim are published for every province; Christian, Hindu, "
            "Confucian, Sikh, other and none are published only for five regions, and each "
            "province's share of them here is its own region's. Turn on the `inferred dots` "
            "control to see which is which — 98.5% of Thailand stays. **Mae Hong Son is "
            "where that assumption is doing the most work**: a quarter of the province is "
            "neither Buddhist nor Muslim, and this map calls almost all of that Christian "
            "because almost all of the North's is. Its Karen and Lahu villages hold both "
            "churches and older traditions, and nothing published separates them."),
        how="census, 2010",
        fill="from the same census at region level",
        grain="provinces, 868,000 people on average",
        counts=_th_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "th" / "th_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_th_place_weight,
        note="**EVERY CENSUS FILE COMES OUT OF THE WAYBACK MACHINE, AND THAT IS THE FIND.** "
             "sources.md recorded Thailand as *'data existed and the server is gone'*: "
             "`statbbi.nso.go.th` and `web.nso.go.th` no longer resolve. What that missed "
             "is that **`www.nso.go.th` is alive and answers 418 to a bare curl and 200 to "
             "a browser User-Agent** — the office moved hosts, as Nepal's had — and that "
             "its old `/sites/2014/Documents/` tree, now 404, was archived wholesale. The "
             "live site's new CKAN (`catalog.nso.go.th`, keyless) carries only a 6-region "
             "3-religion survey table and is not used. "
             "**The two halves are §3.10:** nine categories at five regions from the "
             "regional volumes' Table 4, and Buddhist/Muslim percentages at 76 provinces "
             "from the `kpi_stat` indicator sheets, reunited by `allocate.py --within 1`. "
             "100% of the census population is drawn, 98.5% of it `measured`. "
             "**The check is the residual.** Each region's province residuals, summed, "
             "against that region's own Table 4 non-Buddhist non-Muslim total: -1.8% "
             "Bangkok, -0.8% Central, +4.0% North, +2.2% Northeast, -0.3% South. Those are "
             "two documents that never reference each other agreeing to within a few "
             "percent, and they are the only evidence that the percentages and the counts "
             "describe the same population. "
             "**The rounding is real and is stated**: the indicator sheets give shares to "
             "one decimal, so a province's Buddhist figure carries about ±0.05% and the "
             "residual twice that. The denominator does not compound it — province totals "
             "come from Table 1 at full precision. **A share too small to print appears as "
             "`a`**, NSO's *'less than half the last digit shown'*, and reading that as a "
             "number put Lampang's household-registration rate into its Muslim row before "
             "the check caught it. "
             "**Boundaries are geoBoundaries ADM1, 77 polygons, joined on the TIS 1099 "
             "code through OCHA's COD attribute table** — geoBoundaries has English names "
             "only, the census Thai only, and seven provinces begin `Nakhon`. **Bueng Kan "
             "is dissolved back into Nong Khai** (spec §8.1): it was carved out in March "
             "2011, seven months after the census, so the tables have 76 changwat and the "
             "boundary file has 77. "
             "**Kanchanaburi's indicator sheet was never archived** and takes its region's "
             "shares — 848,000 people, the one province whose headline figures are not its "
             "own.",
    ),
}

# spec §7c: the header block is plain text, and the em dash is the thing it must not contain.
# Anita, 2026-09-07 — the first draft of these rows was correct and read as machine-written, and
# the dash doing the work of a comma, a semicolon and a bracket at once was most of the reason.
# Checked here rather than trusted to review: the fields are edited one country at a time, months
# apart, and a rule this easy to forget is a rule worth failing the import over.
for _cc, _m in COUNTRIES.items():
    for _f in ("how", "fill", "grain", "gap"):
        _v = _m.get(_f, "")
        assert "\u2014" not in _v, f"{_cc}.{_f} has an em dash; use a comma, a semicolon or a bracket"
        assert not (set("<`*") & set(_v)), (
            f"{_cc}.{_f} has markup (< ` or *); these fields are escaped, not rendered")
