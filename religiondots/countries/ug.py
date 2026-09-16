# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _ug_place_weight(place):
    """countries.py hook. `place` is data/geo/ug/ug2024_hexes.gpkg, Kontur 2023-11 hexes on the
    2,200 drawn units (sources/ug_2024_geo.py).

    A within-unit weight only: every unit's people come from the census workbook. Median unit
    57 km2, about 75 hexes, and every unit has a populated hex, so the grid floor is not near.
    """
    return _kontur_place_weight(place, "ug2024_hexes.gpkg", "sources/ug_2024_geo.py")


def _ug_counts():
    """NPHC 2024 on 2,200 drawn subcounty units: 12 answers, shares from UBOS's 10% sample.

    sources/ug_2024.py writes one row per unit and answer, already composed and scaled to the
    unit's full-count household population; its note says the tier the answer was drawn at.
    `tier=subcounty` rows are the unit's own sample share and are `measured`; `tier=county` and
    `tier=district` rows (Orthodox and Jehovah's Witnesses; Bahai and Buddhist) take a parent's
    share because the split-half test did not support them finer, so they are `derived`, may not
    ring, and roll NOWHERE: nothing measured them at the subcounty.
    """
    from rollup import NOWHERE
    from ug2024 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "ug.csv", dtype={"geo_id": str},
                     low_memory=False, keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "subcounty_2024"].copy()
    if df["geo_id"].nunique() != 2200:
        raise SystemExit(f"ug.csv has {df['geo_id'].nunique()} units, expected 2,200 -- re-run "
                         "sources/ug_2024.py")
    df["unit"] = df["geo_id"]
    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)].copy()
    tier = df["note"].str.extract(r"tier=(\w+)")[0]
    if tier.isna().any() or not set(tier) <= {"subcounty", "county", "district"}:
        raise SystemExit(f"ug.csv rows without a known tier: {sorted(set(tier.dropna()))}")
    df["tier"] = tier.map({"subcounty": "measured", "county": "derived", "district": "derived"})
    df["may_ring"] = df["tier"] == "measured"
    df["roll"] = df["tier"].map({"derived": NOWHERE, "measured": None})
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "may_ring", "tier", "roll"]]


ENTRY = {
    "ug": dict(
        name="Uganda",
        source="National Population and Housing Census 2024, 10% population sample and "
               "subcounty profiles (Uganda Bureau of Statistics)",
        basis="self-identification, household population",
        view=[29.4, -1.6, 35.2, 4.4],
        note_public=(
            "**Uganda is drawn from its 2024 census, read from the Bureau of Statistics' own "
            "sample of one household in ten.** The Bureau publishes religion only for the "
            "whole country, but its sample file carries each person's answer beside their "
            "district, subcounty and parish: 4.5 million people in 1.07 million households. "
            "Each subcounty's mix comes from that sample and its population from the full "
            "count, and added up the four largest answers land within 1% of the Bureau's "
            "published national figures. A split-half test decided how finely each answer "
            "can be drawn. The large answers, traditional religion, no religion and `Others` "
            "hold up down to the parish, but no parish boundaries are published, so the map "
            "stops at 2,200 subcounties of about 20,000 people. The Orthodox and Jehovah's "
            "Witnesses are drawn at their county's share and the Baha'is and Buddhists at "
            "their district's. "
            "**The line between Catholic and Anglican Uganda still follows the missions of "
            "the 1890s.** Catholicism is the largest answer in 80 of the 146 districts and "
            "the Church of Uganda in 53. The Anglican south-west reaches **59.1%** in Sheema "
            "and 58.7% in Rukiga; the Catholic north reaches 81.9% in Moyo. "
            "**Muslim Uganda is a handful of places.** Yumbe district is **70.9%** Muslim and "
            "its Aringa subcounties reach 98%; Bugweri and Butambala are just over half, Mbale "
            "City 46.4% and Koboko 45.0%, while Lamwo, Agago and Abim are under half a percent. "
            "**Pentecostal and evangelical churches are the third answer, at 14.7%, and their "
            "stronghold is Mount Elgon.** Bukwo is **66.6%** Pentecostal or evangelical and "
            "several of its subcounties are over 80%; Kween is 42.5% and Namayingo 38.8%. "
            "The 2002 census's narrower Pentecostal box was 4.6% of the country. "
            "**The Seventh-day Adventists are a Rwenzori church.** Ntoroko is 19.2% Adventist, "
            "Bundibugyo 14.6% and Bunyangabu 11.9%, and Mabere subcounty in Bundibugyo 68.4%. "
            "**Karamoja answers the opposite way from 2002.** The 2002 form had no box for "
            "traditional religion, and Kotido put 28.2% in `Other` and 11.9% in no religion. "
            "In 2024 Kotido is **91.7%** Catholic, Karenga 91.2% and Kaabong 89.7%. Traditional "
            "religion is 0.13% of the country, highest in Kaabong at 2.7% and in its Lotim "
            "subcounty at 11.3%. No religion peaks on Mount Elgon instead, in Bukwo (2.6%) and "
            "Kween (2.1%), where Benet subcounty is 6.5%. "
            "**`Others` is 1.5% of the country and its centre is Kagadi and Kibaale.** It is "
            "14.8% of Kagadi, 12.7% of Kibaale and 11.4% of Kyenjojo, and 34.7% of Muhorro "
            "Town Council; the Bureau names the Faith of Unity (Ow'obushobozi), which began "
            "there, among the contents of the box. "
            "**Bidi Bidi is drawn inside its host subcounties.** The refugee settlement in "
            "Yumbe, 122,000 people in households, is three census subcounties with no "
            "boundary of their own, so each is drawn together with the subcounties that "
            "hold its zones. That is why Anglican and Catholic dots sit among Muslim "
            "villages in eastern Yumbe."),
        how="census, 2024, 10% household sample",
        grain="subcounties, 20,000 people on average",
        fill="from the same sample at county and district level",
        gap_share=0.033257,
        gap="3.33% of the country: the 1,517,205 people counted outside households, in "
            "boarding schools, barracks, prisons, hospitals and transit centres, whose sample "
            "records carry no religion; and Apaa's 9,456 people, counted as a unit of their "
            "own with no sample record",
        counts=_ug_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ug" / "ug2024_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ug_place_weight,
        note="REBUILT 2026-09-15 FROM THE 2024 CENSUS, REPLACING THE 2002 BUILD (56 districts, "
             "Table B7; sources/ug.py, taxonomy/ug2002.py and data/normalized/ug2002.csv are "
             "kept). Anita registered with UBOS and downloaded the NPHC 2024 population file "
             "(ask 008): a 10.22% sample, 4,693,190 person records, no weight, religion in "
             "twelve codes, district, county, subcounty and parish on every record. "
             "SHARES FROM THE SAMPLE, PEOPLE FROM THE FULL COUNT: joined by name to the "
             "Subcounty Profiles workbook (all 10,852 sample parishes join; every subcounty's "
             "sampling fraction is 8.2% to 12.5%), each subcounty's composition is scaled to "
             "its Table 2 household population. Catholic, Anglican, Islam and Pentecostal land "
             "within 0.5% of Final Report Table 3.1. "
             "GRAIN BY THE STABILITY TEST, NOT THE FILE: households dealt into six waves; "
             "district against the nation by stability.median_rho and wave_null; each finer "
             "tier against its parent (departure correlation between halves, within-parent "
             "shuffle null, 2,000 permutations) with a bar of median half-sample Pearson 1/3, "
             "where the unit's own share starts to beat its parent's in expected squared error. "
             "Eight answers pass to parish; Orthodox and Witnesses to county; Bahai and "
             "Buddhist to district. No 2024 parish polygon exists (COD-AB is the 2020 edition; "
             "the UBOS portal's map stops at subcounty), so the unit is the subcounty, from "
             "the portal's own polygons fetched county by county. "
             "BIDI BIDI: three census subcounties with no polygon, each merged with the host "
             "subcounties of its county whose Kontur people exceed the census count 1.2 times "
             "(the surplus is 0.65-0.73 of each camp), giving 2,200 drawn units from 2,207. "
             "PENTECOSTAL/EVANGELICALS goes to christianity.evangelical under the one-node "
             "ruling, as mz2017 does. sources/ug.md has the full record.",
    ),
}
