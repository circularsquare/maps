# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _uk_counts():
    """The UK's three drawn censuses, unioned onto one unit namespace.

    England and Wales (ONS, 2021) and Northern Ireland (NISRA, 2021) arrive allocated;
    Scotland (NRS, 2022) publishes its 13 categories at Output Area already and needs no
    allocation, so it is read straight from uk.csv and every Scottish row is `measured`.

    The three code namespaces are disjoint — E00/W00, S00, N20 — so they share one `unit`
    column without a prefix (sources/uk_geo.py checks this rather than assuming it).

    NISRA's second question, religion brought up in, is a different variable and is not
    read here at all (sources/uk.md §1).

    **ENGLAND'S `Christian` ROWS ARE REPLACED, NOT SUPPLEMENTED** (spec §3.5a, uk_split.py).
    uk_split.csv holds five denominational rows plus a reduced `Christian` remainder for
    every English Output Area, and they sum to exactly the census figure they replace. So
    the England rows whose category is `Christian` are dropped from the allocated file
    before uk_split's are concatenated in — adding them would double England's Christians,
    and the check below is what makes that impossible to do quietly. Wales's `Christian`
    rows are replaced the same way by uk_split_wales.csv (2026-10-04), whose reduced
    `Christian` remainder is 8.2%: the churches too thin in the Welsh sample to draw (the
    unnamed share is spread like England's since Anita's call the same day).
    """
    import uk2021

    frames = []
    for stem in ("uk_ew", "uk_ni"):
        d = pd.read_csv(HERE / "data" / "normalized" / f"{stem}_allocated.csv",
                        dtype={"geo_id": str}, low_memory=False)
        if stem == "uk_ew":
            england = d["geo_id"].str.startswith("E")
            replaced = d.loc[england & (d["source_category"] == "Christian"), "count"].sum()
            d = d[~(england & (d["source_category"] == "Christian"))]
            split = pd.read_csv(HERE / "data" / "normalized" / "uk_split.csv",
                                dtype={"geo_id": str}, low_memory=False)
            drawn = split["count"].sum()
            if abs(drawn - replaced) > 1.0:
                raise SystemExit(
                    f"uk_split.csv draws {drawn:,.0f} English Christians against the "
                    f"{replaced:,.0f} it replaces; re-run uk_split.py")
            # Wales, the same way, from uk_split_wales.py (2026-10-04): a survey-named church
            # for 70% of Welsh Christians, the rest left on the census's `Christian`.
            wales = d["geo_id"].str.startswith("W")
            w_replaced = d.loc[wales & (d["source_category"] == "Christian"), "count"].sum()
            d = d[~(wales & (d["source_category"] == "Christian"))]
            w_split = pd.read_csv(HERE / "data" / "normalized" / "uk_split_wales.csv",
                                  dtype={"geo_id": str}, low_memory=False)
            if abs(w_split["count"].sum() - w_replaced) > 1.0:
                raise SystemExit(
                    f"uk_split_wales.csv draws {w_split['count'].sum():,.0f} Welsh Christians "
                    f"against the {w_replaced:,.0f} it replaces; re-run uk_split_wales.py")
            d = pd.concat([d, split, w_split], ignore_index=True)
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


ENTRY = {
    "uk": dict(
        name="United Kingdom",
        name_in="the United Kingdom",
        source="Censuses of 2021 and 2022 (ONS, NRS, NISRA)",
        basis="self-identification, voluntary question in England and Wales",
        view=[-8.7, 49.8, 2.0, 61.0],
        note_public=(
            "Three censuses, three agencies, three category lists. England and Wales "
            "publish no Christian denomination at all, at any geography, for 27.5 million "
            "people. The write-in detail there is everything outside Christianity: Pagan, "
            "Alevi, Jain, Ravidassia, Yazidi, Vodun.\n\n"
            "England is drawn split anyway. Its 26.2 million Christians are divided into "
            "Anglican, Roman Catholic, Methodist, Baptist and Reformed using three sources "
            "that are not the census: the Church of England's own register of its churches, "
            "together with OpenStreetMap, for where each denomination is; the English Church "
            "Census of 2005 for how large its congregations are; and the British Election "
            "Study for how many people belong to each. The census keeps every total, and "
            "none of those five categories was ever counted, so they are marked as inferred "
            "and fold back into one Christian figure if you switch inferred dots off.\n\n"
            "Orthodoxy is placed differently, from where people born in Romania, Bulgaria, "
            "Greece, Cyprus and Moldova live, each origin weighted by how Orthodox that "
            "country is. A church census from 2005 found 49 Orthodox churches and answered "
            "on the Sunday after Orthodox Easter, when many were shut; England now has "
            "around 567,000 Orthodox Christians, more than it has Baptists. Nothing counted "
            "them, so those dots are modelled rather than inferred.\n\n"
            "About one English Christian in seventeen stays on the plain Christian figure. "
            "Nothing available places the Pentecostal and independent evangelical "
            "congregations, which mostly meet in rented halls and are missing from both a "
            "2005 church census and a map of church buildings. Ethnicity was tried as a way "
            "in and dropped: Black African England is heavily Anglican and Catholic too, so "
            "it would say where the Black-majority congregations are rather than where the "
            "Pentecostals are.\n\n"
            "Wales is split too, more thinly. No church census has been taken there since "
            "1995, so the British Election Study does both jobs: 2,970 Welsh Christians, each "
            "counted once across its waves, name their church, and the mix is read for twelve "
            "groups of council areas. Anglican, Methodist, Baptist and Reformed differ between "
            "those areas consistently enough to place; Catholics do not, so they are drawn at "
            "the Welsh share everywhere. As in England, the census Christians the survey names "
            "no church for, about 30%, are given the survey's mix; the survey has no box for a "
            "Christian of no particular church. The Presbyterian Church "
            "of Wales has no box of its own either, and many of its members seem to have "
            "answered Methodist, which is why Methodism peaks in Anglesey and Gwynedd. The "
            "Union of Welsh Independents has no box at all.\n\n"
            "Scotland names the Church of Scotland and the Roman Catholics and stops. "
            "Northern Ireland, where the denomination is the political fact, names "
            "twenty-two Christian bodies including four kinds of Presbyterian, and is the "
            "only agency here that counts people as Mixed Catholic / Protestant."),
        how="censuses, 2021 and 2022, voluntary in England and Wales",
        fill=("from the same censuses at a coarser geography; Christian denominations from "
              "church registers, a 2005 English church census and the British Election Study"),
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
}
