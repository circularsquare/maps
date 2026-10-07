# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _az_place_weight(place):
    """countries.py hook. `place` is the 400 m hex layer scatter.py has read.

    Kontur AZ, with every hex inside the area outside government control in 2019 masked (Natural
    Earth 4.1.0's Nagorno-Karabakh), so Aghdam's and Fuzuli's census people are not placed on the
    ruins east of the 1994 line (sources/az_grid.py).
    """
    return _kontur_place_weight(place, "az_hexes.gpkg", "sources/az_grid.py")


def _az_counts():
    """The 2019 census's existing population in 66 rayons and cities, as an ethnicity model.

    EVERY ROW IS `modelled` (§7b). Eight nationalities are placed on the 2009 census's nationality
    by unit, scaled to 2019, and given a religion (Russians, Ukrainians and Tatars at Kazakhstan's
    2021 census shares); everyone else is on Islam. The eight units with no existing population in
    2019 (the districts outside government control) have no rows. sources/az.py and sources/az.md.
    """
    from az2019 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "az.csv", dtype={"geo_id": str},
                     keep_default_na=False, na_values=[""])
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"az.csv categories with no node: {unmapped}")
    if df["geo_id"].nunique() != 66:
        raise SystemExit(f"{df['geo_id'].nunique()} populated units in az.csv, expected 66")
    df["unit"] = df["geo_id"]
    df = df[df["count"] > 0].copy()
    df = df.groupby(["unit", "node"], as_index=False)["count"].sum()
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "az": dict(
        name="Azerbaijan",
        source="No census or survey asks religion with a place; nationality by district from the "
               "2009 and 2019 censuses (State Statistical Committee), with Kazakhstan's 2021 census "
               "shares for Russians, Ukrainians and Tatars",
        basis="modelled from census nationality; nobody was asked their religion",
        note_public=(
            "**Nobody in Azerbaijan is asked their religion in a way that says where they live.** "
            "The 2009 and 2019 censuses asked nationality and mother tongue, not religion. Surveys "
            "that asked religion and recorded where people were interviewed include the "
            "Demographic and Health Survey in 2006, the EBRD's Life in Transition Survey in 2016 "
            "and the European Values Study in 2017. In all three almost everyone answered Muslim, and in "
            "the Life in Transition Survey all **1,510** respondents did, the three Russians "
            "among them. So the map is built from the census's nationalities instead. Nobody "
            "counted these dots, so they disappear when inferred dots are turned off. "
            "**Every nationality except eight is drawn as Muslim.** That covers Azerbaijanis, "
            "Lezgins, Talysh, Avars, Turks, Tats, Tsakhurs and the smaller peoples of the north. "
            "The eight are placed where the 2009 census found them, scaled to their 2019 national "
            "counts, because the 2019 census gives nationality only for the whole country. "
            "Georgians are drawn as Georgian Orthodox, most of them in Gakh in the north-west; "
            "Udins as Christians of the Udi church, in Nij in Gabala and in Oghuz; Jews "
            "as Jewish, in Baku and in Quba's Red Town; and the 178 Armenians the 2019 census "
            "counted as Armenian Apostolic. No source gives a religion for Russians, Ukrainians "
            "or Tatars in Azerbaijan, so they take the shares Kazakhstan's 2021 census gives the "
            "same nationalities, leaving out those who refused to answer: **92.7%** of Russians "
            "Orthodox, 2.1% Muslim and 5.1% non-believers. The non-believers are the weakest part "
            "of this map, because Kazakhstan's own figures show non-belief changing from place to "
            "place inside every nationality. The Russians of Ismayilli and Gadabay are mostly "
            "Molokans, a Spiritual Christian community, and are drawn as Orthodox. Ingiloys and "
            "the census's other nationalities, 6,849 people, are drawn as religion not known. "
            "That makes **95,717** Christians and **5,099** Jews, and 98.9% of the people drawn "
            "Muslim. "
            "**The published estimates differ.** Pew Research Center, working from the European "
            "Values Study, puts Azerbaijan at 94.7% Muslim and 4.8% unaffiliated, with 42,730 "
            "Christians and 8,580 Jews. Its unaffiliated are people who said they belong to no "
            "religious denomination; the Demographic and Health Survey found 0.7% of women "
            "Christian, of no religion or of another religion together, and the Life in "
            "Transition Survey found none. No source says where the unaffiliated live, so they "
            "are drawn as Muslim. Shia and Sunni Muslims are drawn as one: in Pew's 2011 survey "
            "37% of Azerbaijan's Muslims called themselves Shia, 16% Sunni and 45% just Muslim, "
            "with no figure for any district. "
            "**This is Azerbaijan as the 2019 census found it.** People are drawn in the district "
            "where they were counted, not the one they are registered in, so the people displaced "
            "from the districts around Nagorno-Karabakh are drawn in Baku, Sumgayit and the other "
            "places they lived in 2019. The census did not enumerate the territory outside the "
            "government's control from 1994 to 2020, so that area is empty here. More than "
            "**100,000** refugees from Karabakh arrived in Armenia in September 2023, in UNHCR's "
            "count, and they are on neither country's map, since Armenia's census was taken in "
            "2022. The President's office says more than 48,000 people now live and work in the "
            "districts that returned to Azerbaijani control in 2020 and 2023; they are drawn where "
            "they lived in 2019."),
        how="no source asks; modelled from census nationality",
        grain="rayons and cities, 151,000 people on average",
        gap="the territory outside government control in 2019, which the census did not "
            "enumerate; and Azerbaijanis of no religion, whom no source places and who are drawn "
            "as Muslim",
        counts=_az_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "az" / "az_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_az_place_weight,
        note="REOPENED ON ANITA'S PRIORITY (ask/RULINGS.md 2026-09-15) AND THE MAURITANIA RULING "
             "(2026-09-16). sources/az.md is the record. NOTHING ASKS WITH A PLACE: 2009 and 2019 "
             "census forms (nationality, mother tongue); LiTS III 1,510 of 1,510 MUSLIM incl. 3 "
             "Russians (instrument zero, playbooks/lits.md); DHS 2006 99.2% Muslim (report only); "
             "EVS 2017 behind GESIS (Pew 2020's source). MODEL (sources/az.py): 2019 census Vol A "
             "Table 3 EXISTING population (de facto; 9,943,958) in 66 populated COD-AB units; 8 "
             "groups at their 2019 national counts (Vol B Table 28) spread on the 2009 census's "
             "nationality by unit (Vol XIX via pop-stat.mashke.org, checked against tables 1.11 "
             "and 1.17); Russians, Ukrainians, Tatars at Kazakhstan 2021's religion-by-nationality "
             "(kz_model.py), refusals and the small answers dropped; Georgians Georgian Orthodox, "
             "Udins christianity.oriental, Jews judaism, Armenians Armenian Apostolic (Karabakh "
             "units excluded), Ingiloys and other nationalities unknown; rest islam. NOT DRAWN: "
             "Pew's 4.76% unaffiliated (EVS 'no denomination'; DHS and LiTS say under 1%); no "
             "Shia/Sunni (om/sa ruling). KARABAKH: 8 units empty in 2019 draw nothing; placement "
             "masks Natural Earth 4.1.0's 1994-2020 Nagorno-Karabakh (sources/az_grid.py); ask "
             "filed on the 2023 refugees and resettlement (spec §14).",
    ),
}
