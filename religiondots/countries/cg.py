# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _cg_place_weight(place):
    """countries.py hook. `place` is the Kontur 400m hex layer scatter.py has read.

    Twelve départements over 342,000 km2, and Brazzaville and Pointe-Noire hold 2.09 of the
    3.70 million people in 454 km2 between them, while Likouala and Sangha are 120,000 km2 of
    forest and swamp with their people in a few river towns (sources/cg_geo.py).
    """
    return _kontur_place_weight(place, "cg_hexes.gpkg", "sources/cg_geo.py")


def _cg_counts():
    """Republic of the Congo RGPH 2007 at département: 9 nodes on 12 units, every row
    `measured` and may ring.

    Tableau 11 of *Le RGPH-2007 en quelques chiffres* in counts, which close to the person on
    Tableau 1's département populations and on the whole resident population, 3,697,490
    (sources/cg.py). No rescale: the table is counts, not shares.
    """
    from cg2007 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "cg.csv",
                     low_memory=False, keep_default_na=False, na_values=[""])
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != 12:
        raise SystemExit(f"{df['geo_id'].nunique()} départements in cg.csv, expected 12 -- "
                         "re-run sources/cg.py")
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if unmapped:
        raise SystemExit(f"cg.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0].rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["may_ring"] = True
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "may_ring", "tier"]]


ENTRY = {
    "cg": dict(
        name="Republic of the Congo",
        name_in="the Republic of the Congo",
        source="RGPH 2007, Le RGPH-2007 en quelques chiffres (CNSEE), Tableau 11",
        basis="self-identification, whole resident population",
        note_public=(
            "**Congo's 2007 census published religion by département, in counts.** Its form "
            "offered nine answers, with separate boxes for the Salvation Army, the Kimbanguist "
            "Church and the revival churches (églises de réveil), which are drawn here as "
            "Pentecostal. "
            "**The revival churches lead in the north.** They are 22.3% of the country, and "
            "**38.8%** of Sangha, 38.7% of Likouala and 37.8% of Cuvette, against about 12% in "
            "Niari, Lékoumou, Bouenza and Pool, where Catholics and the mission Protestant "
            "churches are larger. "
            "**A third of Plateaux gave no religion.** The share is **33.9%** there and over a "
            "quarter in Cuvette, Cuvette-Ouest and Sangha, while the animist box on the same "
            "form is under 2% in every département. "
            "**Kouilou's largest answer is other.** 29.3% of the département gave it, and the "
            "census brochure does not say what it holds. "
            "**This is the 2007 map.** The 2023 census has published no religion figures, and "
            "the twelve départements drawn are the 2007 ones; Congo created three more out of "
            "Plateaux, Pool, Cuvette and Likouala in October 2024."),
        how="census, 2007",
        grain="départements, 308,000 people on average",
        counts=_cg_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "cg" / "cg_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_cg_place_weight,
        note="CITE AND FETCH ONLY THE WAYBACK COPY: cnsee.org is squatted and serves the same "
             "brochure with spam links injected (sources.md §11aq). Tableau 11 is nine answers "
             "x twelve départements in COUNTS, closing to the person on Tableau 1's populations "
             "and on 3,697,490. The 2007 form's P13 codes (CA PR SA KI MU ER AN AU SR) are the "
             "table's column order. UNSD table 28 has no Congo row. The twelve are the 2007 "
             "départements, not the fifteen of the October 2024 laws. sources/cg.md has the "
             "record.",
    ),
}
