# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _kw_place_weight(place):
    """countries.py hook. `place` is Kontur's footprint cut to the 143 drawn units.

    Kontur's counts are far off in Kuwait (Sabah Al-Salem 0.03 of its census share, Wafra Farms
    23x), so `sources/kw_grid.py` keeps only where Kontur puts anybody and weights that by area:
    uniform inside a residential area, the settled part of a desert or farm unit.
    """
    return _kontur_place_weight(place, "kw_hexes.gpkg", "sources/kw_grid.py")


def _kw_counts():
    """2021 census areas; PACI's June 2014 religion by nationality group and sex, carried by each
    area's mix of groups.

    EVERY ROW IS `derived` (spec §7): PACI's register counted every resident's religion, but only
    for the whole country by nationality group and sex, and each area's rows are that count
    carried by the area's own people of each group and sex (sources/kw.py). Kuwaitis all Muslim
    (99.978% in PACI's count), on `islam.shia` and `islam.sunni` at one national share (Arab
    Barometer III, 2026-10-03); the Asian `Other-Not Stated` split by nationality through Pew. No
    row rolls up (below), the sect split included: it rolls nowhere rather than to `islam`,
    because PACI's `islam` is no more measured per area than the split is. 157 census areas on 143 OSM polygons (sources/kw_geo.py).
    """
    from kw2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "kw.csv", keep_default_na=False,
                     na_values=[""])
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"kw.csv categories with no node: {unmapped}")
    lut = pd.read_csv(HERE / "data" / "geo" / "kw" / "kw_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"kw rows with no unit: {missing}; re-run sources/kw_geo.py")
    if df["unit"].nunique() != 143:
        raise SystemExit(f"{df['unit'].nunique()} units, expected 143")
    df = df[df["count"] > 0]
    df = df.groupby(["unit", "node"], as_index=False)["count"].sum()
    df["tier"] = "derived"
    df["congregations"] = 0
    # spec §3.10: a carried count cannot establish that anyone is present
    df["may_ring"] = False
    # spec §7a-i-1: a roll target must be a column measured AT THE SAME UNIT, and PACI's columns
    # were counted for the whole country only, so nothing rolls and Kuwait empties under
    # `inferred dots: not shown`, as Switzerland, China, Saudi Arabia and Oman do. NOWHERE as
    # Angola and Uganda. taxonomy/kw2021.py's COLUMNS (the PACI column each row came out of) is
    # kept as a record and deliberately not attached. Reviewer fafd1067-rev7, 2026-10-03; the
    # published counts.json already carried an empty roll table for kw, so nothing shipped changes.
    from rollup import NOWHERE
    df["roll"] = NOWHERE
    return df[["unit", "node", "count", "congregations", "tier", "roll", "may_ring"]]


ENTRY = {
    "kw": dict(
        name="Kuwait",
        source="Public Authority for Civil Information, religion by nationality group and sex, "
               "June 2014 (its table builder, as archived); carried to the 2021 census's areas by "
               "their Kuwaitis and non-Kuwaitis by sex (Central Statistical Bureau, Table 52) and "
               "each area's mix of nationality groups (PACI's December 2014 locality table, as "
               "mirrored by the Gulf Labour Markets, Migration and Population programme); "
               "Kuwaitis' sect from Arab Barometer wave III (2014)",
        basis="Religion as the civil register records it, counted for the whole country and "
              "placed by nationality group",
        note_public=(
            "**Kuwait's civil register records a religion for everyone, but its tables give it "
            "only for the whole country.** In June 2014 the Public Authority for Civil Information "
            "counted Muslims, Christians and people of another or no stated religion among "
            "Kuwaitis and among seven groups of foreign residents (Arab, Asian, African, European, "
            "North American, South American, Australian), each by sex. No table found crosses "
            "religion with any place inside Kuwait. So each group's shares are carried to the "
            "157 areas of the 2021 census by the number of people of that group and sex living "
            "in each. The census gives each area's Kuwaitis and non-Kuwaitis, but the nationality "
            "groups only for each governorate, so the mix of groups inside an area is the "
            "register's own from December 2014, adjusted to the 2021 governorate totals. Every "
            "dot was counted for the country and placed this way, so all of them "
            "disappear when inferred dots are turned off. "
            "**The 1,488,435 Kuwaitis are drawn as Muslim.** The register counts **99.98%** of "
            "them as Muslim; the few hundred Christian citizens are not drawn, because nothing "
            "says where they live. They are split into Shia and Sunni at one share for the whole "
            "country, **17.7%** Shia, so every area has the same mix among its Kuwaitis. The share "
            "is from the Arab Barometer's 2014 survey of Kuwaiti citizens, in which the "
            "interviewers recorded the sect they took each respondent to belong to; they could "
            "not tell for 19%, and those are split at the same ratio. The figure most often "
            "quoted, about 30% Shia among citizens (US State Department, from NGOs and the "
            "media), is higher and gives no method. Non-Kuwaiti Muslims are not split. "
            "**Where the Muslim share falls, it is where Asian workers live.** In the register, "
            "Arab residents are 95% Muslim and Asian residents 46% Muslim, 39% Christian and 15% "
            "another or no stated religion, with Asian women 56% Christian. In the areas of more "
            "than 50,000 people, the share drawn as Muslim runs from 61% in Al-Mahbula and 64% in "
            "Jleeb Al-Shuyoukh to over 90% in Taima and Al-Sulaibiya. Kuwaiti suburbs show Christians too, at about a tenth: most of their "
            "foreign residents are Asian, and close to half are Asian women. Of the 2,892,704 "
            "non-Kuwaitis the map draws "
            "**66.3%** as Muslim and **24.9%** as Christian; the register's own figures for 2023, "
            "as the US State Department quotes them, are 62.7% and 24.5%. "
            "**The Asian residents' other religions are split by nationality.** That part of the "
            "register's count is mostly Hindus, Buddhists and Sikhs, and it is drawn on those "
            "religions by the number of Indians, Bangladeshis, Filipinos, Pakistanis, Sri Lankans "
            "and Nepalis in Kuwait and Pew Research Center's estimate for each home country. That "
            "puts 216,323 people on Hinduism, 20,379 on Buddhism and 5,268 on Sikhism. Community "
            "estimates quoted by the State Department put Kuwait's Buddhists nearer 100,000, so "
            "Buddhists are probably under-drawn and Hindus over-drawn. Christians are not split "
            "into churches: the register's count of Asian Christians is about ten times what the "
            "home countries' shares would give, so those shares say nothing about which churches "
            "they belong to."),
        how="population register, 2014; Kuwaitis' sect from a 2014 survey",
        fill="from the register's national count for each nationality group and sex",
        grain="areas, 31,000 people on average",
        gap="the 4,578 people (0.1%) whose area the 2021 census did not state",
        # 4,578 of 4,385,717: Table 1's `Not Stated` row, in no area and so in no unit
        gap_share=0.00104,
        counts=_kw_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "kw" / "kw_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_kw_place_weight,
        note="REOPENED 2026-10-03 on Anita's priority line, after sources.md §11ao and "
             "§scout-2026-09-14-asia-oceania left it blocked on PACI's table builder (times out "
             "from here). sources/kw.md is the record. RELIGION: PACI's own tableQuery JSON, "
             "June 2014, Wayback 20141015093707 (Muslim / Christian / Other-Not Stated by "
             "Kuwaiti and 7 non-Kuwaiti groups by sex; 4,039,445). POPULATION: census 2021 "
             "Table 52 (157 areas x Kuwaiti/non-Kuwaiti x sex), Table 6 (governorate x group x "
             "sex), Table 1; area group mix from GLMM's copy of PACI's December 2014 locality "
             "table, raked per governorate and sex to 2021 (moves 2.1% of non-Kuwaitis). "
             "Kuwaitis all Muslim (99.978%), split islam.shia/islam.sunni at ONE NATIONAL SHARE, "
             "17.69% Shia (Arab Barometer III q2005kw, the field team's opinion, weighted, of "
             "those placed; Anita's ruling 2026-10-03, sources.md §kw-2026-10-03b). Asian "
             "Other-Not Stated split by GLMM 2018's six named Asian nationalities x Pew 2020. "
             "ROLL: NOWHERE for every row (nothing measured at an area; reviewer 2026-10-03). "
             "Christians unsplit. WITNESS: non-Kuwaitis 66.3/24.9/8.8 against "
             "PACI 2023 62.7/24.5/12.8 (State Dept 2023), band 5 points. GEOGRAPHY: OSM "
             "admin_level 6 (q8maps), 115 areas on the Arabic name, 42 by hand, 143 units; OSM "
             "population tags witness 13 areas at 0.80-1.36. PLACEMENT: Kontur's counts fail "
             "(p10 0.12, p90 15.6 per unit); its footprint at 50/km2 is used, by area.",
    ),
}
