# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _ne_place_weight(place):
    """countries.py hook. `place` is the Kontur 400m hex layer scatter.py has read.

    Eight régions over 1.19 million km2, from Niamey (557 km2 on COD-AB, 1,026,848 people in
    2012) to Agadez (622,000 km2, 487,620); in Agadez, Diffa and northern Tahoua people live in
    towns, oases and along the wadis, so the grid matters most there (sources/ne_geo.py).
    """
    return _kontur_place_weight(place, "ne_hexes.gpkg", "sources/ne_geo.py")


def _ne_counts():
    """Niger RGP/H 2012 at région: 5 nodes on 8 units, every row `measured`.

    Tableau A 11 of the structure volume, in counts, every resident (sources/ne.py). The 42,608
    with no religion recorded (`ND`) are read and not drawn.
    """
    from ne2012 import EXCLUDED, resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "ne.csv",
                     low_memory=False, keep_default_na=False, na_values=[""])
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != 8:
        raise SystemExit(f"{df['geo_id'].nunique()} régions in ne.csv, expected 8 -- re-run "
                         "sources/ne.py")
    df = df[~df["source_category"].isin(EXCLUDED)].copy()
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if unmapped:
        raise SystemExit(f"ne.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0].rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["may_ring"] = True
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "may_ring", "tier"]]


ENTRY = {
    "ne": dict(
        name="Niger",
        source="RGP/H 2012, État et structure de la population du Niger en 2012 (Institut "
               "National de la Statistique), Tableau A 11",
        basis="self-identification, resident population",
        note_public=(
            "**Niger's 2012 census published religion for its eight régions, in counts.** Of "
            "all 17.1 million residents, **99.1%** answered Muslim, 0.33% Christian, 0.20% "
            "animist and 0.13% no religion, and 0.25% have no answer recorded. Young children "
            "were given the religion of their father or mother. "
            "**Christians live mostly in Niamey and Tillabéri.** Niamey is **1.4%** Christian "
            "and Tillabéri, the région around it, 0.78%; the two hold 63% of the country's "
            "Christians, and no other région is above 0.25%. The census report says "
            "Christianity in Niger mostly concerns foreigners, but it prints no table of "
            "religion by nationality. "
            "**Animism is highest in Dosso and Niamey, at 0.34% each.** Zinder holds the most "
            "animists, 9,053. The form offered animism as its own answer, so these are people "
            "who gave it as their one religion."),
        how="census, 2012",
        grain="régions, 2.1 million people on average",
        gap="0.25% whose religion was not recorded",
        gap_share=0.0025,
        counts=_ne_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ne" / "ne_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ne_place_weight,
        note="RGP/H 2012 STRUCTURE VOLUME, Tableau A 11 (région x religion, COUNTS), on "
             "stat-niger.org, pinned. Every column equals UNSD table 28 to the person; the "
             "régions equal Tableau 3's 2012 populations; Tableau 20's shares are of those who "
             "stated a religion, with three cells forced so rows close. The form (C07) has 0 Sans "
             "religion, 1 Musulmane, 2 Chrétienne, 3 Animiste, 9 Autre à préciser and no "
             "non-response code; `ND` (42,608) is excluded. RÉGIONS ARE COD-AB v02 ADM1 by "
             "pcode, witnessed by Tableau 7's density-implied areas (0.96-1.02 of the national "
             "ratio) and Kontur. COD's Niamey is 557 km2 against INS's 255, and its densest 255 "
             "km2 hold 96.4% of its Kontur people. geoBoundaries NER ADM1 is six merged units and "
             "was not used. sources/ne.md has the record.",
    ),
}
