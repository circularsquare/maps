# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _dj_place_weight(place):
    """countries.py hook. `place` is the Kontur 400m hex layer scatter.py has read.

    Six régions over 22,400 km2, from Djibouti-Ville (196 km2 on COD-AB, 728,010 in the table)
    to Dikhil and Tadjourah (6,700 km2 each, about 60,000 people). Kontur's région shares are
    2009's, not 2024's, which moves no count, since dots are placed inside each région only
    (sources/dj_geo.py).
    """
    return _kontur_place_weight(place, "dj_hexes.gpkg", "sources/dj_geo.py")


def _dj_counts():
    """Djibouti RGPH-3 2024 at région: 4 nodes on 6 units, every row `measured`.

    Tome 4 Tableau 42, in counts, residents of ordinary and nomadic households (sources/dj.py).
    """
    from dj2024 import EXCLUDED, resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "dj.csv",
                     low_memory=False, keep_default_na=False, na_values=[""])
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != 6:
        raise SystemExit(f"{df['geo_id'].nunique()} régions in dj.csv, expected 6 -- re-run "
                         "sources/dj.py")
    df = df[~df["source_category"].isin(EXCLUDED)].copy()
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if unmapped:
        raise SystemExit(f"dj.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0].rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["may_ring"] = True
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "may_ring", "tier"]]


ENTRY = {
    "dj": dict(
        name="Djibouti",
        source="RGPH-3 2024, Tome 4: Caractéristiques socioculturelles de la population "
               "(Institut National de la Statistique de Djibouti), Tableau 42",
        basis="self-identification, residents of ordinary and nomadic households",
        note_public=(
            "**Djibouti's 2024 census asked every resident's religion and published it for the "
            "six régions, in counts.** Of the 1,003,800 people in ordinary and nomadic "
            "households, **99.45%** answered Islam, 0.44% Christianity, 0.09% no religion and "
            "0.02% another religion. The question had separate answers for Catholics, Protestants, "
            "Orthodox Christians, animists and atheists; the published tables merge them into "
            "these four. "
            "**Christians live mostly in Djibouti-Ville.** The capital holds 3,860 of the "
            "country's 4,455 Christians, 0.53% of its people, and Ali-Sabieh is 0.48% Christian; "
            "no other région is above 0.16%. Of all Christians, 1,934 are Djiboutian citizens "
            "and 1,807 are Ethiopian. "
            "**The 63,009 residents outside ordinary households are not on this map.** They are "
            "the 30,351 people the census counted as homeless and the 32,658 in collective "
            "households such as barracks, boarding schools and hospitals, together 5.9% of the "
            "1,066,809 residents, and no table gives their religion. In Obock, which the "
            "statistics institute names as the main transit point for migrants heading to "
            "Yemen, they are a quarter of the population."),
        how="census, 2024",
        grain="régions, 167,000 people on average",
        gap="5.9% of residents, the homeless and people in collective households, whose "
            "religion was not tabulated",
        gap_share=0.0591,
        counts=_dj_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "dj" / "dj_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_dj_place_weight,
        note="RGPH-3 2024, TOME 4 Tableau 42 (région x religion, COUNTS, four answers), from "
             "INSTAD's Firebase storage via the API behind instad.dj, pinned. Universe: the "
             "1,003,800 in ordinary and nomadic households (Tableau 2: P12_RELIGION, none "
             "missing); = final report Tableau 14 by région; 30,351 homeless + 32,658 collective "
             "(Tableau 7) are the gap. The question had eight codes, the tables four. RÉGIONS "
             "ARE COD-AB (GADM 2022) ADM1 by name (`Djiboutii` and `Tadjoura` pinned); Kontur DJ "
             "per région is nearer the 2009 census's shares than 2024's (Tadjourah 2.47 over the "
             "national ratio, pinned), placement inside each région only. Not in Afrobarometer "
             "or Arab Barometer (waves I-VIII). sources/dj.md has the record.",
    ),
}
