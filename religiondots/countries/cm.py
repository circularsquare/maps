# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _cm_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    12 units over 466,000 km2. Est is 110,000 km2 and Mfoundi 289; the regions hold their people
    along roads, in the Grassfields and around a few towns (sources/cm_geo.py).
    """
    return _kontur_place_weight(place, "cm_hexes.gpkg", "sources/cm_geo.py")


def _cm_counts():
    """Five pooled Afrobarometer rounds on COD-PS 2025: 7 nodes, 12 units, and EVERY ROW IS
    `modelled` IN §7.

    CAMEROON PRINTED ITS 2005 CENSUS RELIGION FOR THE WHOLE COUNTRY ONLY, so there is no measured
    tier. Each unit is drawn at the mix its own respondents gave (Christian, Presbyterian, Baptist,
    Muslim and None pass the split-half) and at BUCREP's projected 2025 population; Traditional and
    Other are the residual under spec §12's small-category rule. sources/cm.py has the construction
    and why two churches are drawn and the rest are not.

    YAOUNDÉ (MFOUNDI) AND DOUALA (WOURI) ARE UNITS OF THEIR OWN, because every round samples them
    apart from Centre and Littoral.
    """
    from cm2025 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "cm.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    lut = pd.read_csv(HERE / "data" / "geo" / "cm" / "cm_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"cm.csv units with no polygon: {missing} -- re-run "
                         "sources/cm_geo.py, the lookup is stale")
    if df["unit"].nunique() != 12:
        raise SystemExit(f"{df['unit'].nunique()} units, expected 12")

    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"cm.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    # EVERY row, without exception -- there is no measured tier in this country (§7).
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "cm": dict(
        name="Cameroon",
        source="five pooled rounds of the Afrobarometer, 2013 to 2022, on the 2025 population "
               "projection of the Bureau Central des Recensements et des Etudes de Population "
               "(BUCREP)",
        basis="self-identification, whole population",
        note_public=(
            "**Cameroon last counted religion in its 2005 census, and printed the answer only "
            "for the whole country.** That count found 38.4% Catholic, 26.3% Protestant, 20.9% "
            "Muslim and 5.6% following traditional religion. A new census was taken in 2026 "
            "and nothing from it is out yet. "
            "**So this map's Cameroon comes from a survey.** Five rounds of the Afrobarometer "
            "are pooled, **5,949** adults interviewed between March 2013 and April 2022, and "
            "every dot is desaturated because nobody counted it. Each unit is drawn at the mix "
            "its own respondents gave and at BUCREP's projected population for 2025, which puts "
            "the country at **74.4%** Christian, **20.1%** Muslim and 3.8% with no religion. "
            "Yaoundé and Douala are drawn apart from the Centre and Littoral regions around them, "
            "because the survey interviews them separately. "
            "**Islam is the religion of the north.** Adamaoua is **63.1%** Muslim and Nord and "
            "Extrême-Nord about 40%, and no other region reaches 18%. The 2005 census put the "
            "same three regions at the top. "
            "**Presbyterians and Baptists are drawn as churches of their own.** Presbyterians "
            "are **27.2%** of Nord-Ouest and about 22% of Sud-Ouest and Sud, and Baptists about "
            "10% of Nord-Ouest and Sud-Ouest; both hold their share from round to round. None "
            "of the 309 people interviewed in Adamaoua was a Baptist, so none is drawn there. "
            "The rest of Christianity is one colour, Catholics included. The share who say just "
            "Christian instead of naming a church doubles across the five rounds, and it comes "
            "mostly out of the Catholic answer, which moves between 26% and 41%. Lutherans hold "
            "their share too, but they live in the north, where up to a third of Christians name "
            "no church, so they are left inside Christianity. "
            "**Traditional religion is 0.55% on this map**, against 5.6% in the 2005 census. "
            "The survey offers it as one answer beside Christianity and Islam, so it counts only "
            "people who give it as their religion. Traditional and other religions are spread "
            "in each unit by what its measured religions leave over."),
        how="a pooled survey, 2013 to 2022, on 2025 projected region populations",
        grain="regions, with Yaoundé and Douala apart; 2.5 million people on average",
        counts=_cm_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "cm" / "cm_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_cm_place_weight,
        note="CAMEROON'S 2005 CENSUS PRINTED RELIGION NATIONALLY ONLY (Tableau 5.8, BUCREP "
             "Volume II Tome 01), and Cameroon is ABSENT from the UNSD oracle. Drawn from "
             "Afrobarometer R5-R9 on COD-PS 2025, on the construction of tz and ng: every row "
             "modelled, the national level computed rather than fitted. 12 UNITS: Mfoundi and "
             "Wouri are cut out of Centre and Littoral because every round samples them apart. "
             "PRESBYTERIAN AND BAPTIST ARE DRAWN, Catholic, Evangelical and Lutheran are folded "
             "into christianity (sources/cm.py docstring: level by round, and the unnamed share "
             "where each church lives). Traditional and Other are the residual. The 2005 census "
             "is a witness, re-read from the PDF on every build; REOPEN on the 4th RGPH.",
    ),
}
