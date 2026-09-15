# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _ml_place_weight(place):
    """countries.py hook. `place` is the Kontur 400m hex layer scatter.py has read.

    The régions run from Bamako (733 km2, 4.2 million people) to Taoudenni (293,000 km2,
    100,358), and the northern régions have their people along the Niger and in a few towns
    (sources/ml_geo.py).
    """
    return _kontur_place_weight(place, "ml_hexes.gpkg", "sources/ml_geo.py")


def _ml_counts():
    """Mali RGPH5 2022 at région: 7 nodes on 20 units, every row `measured`, may ring.

    Tableau 2.03's two-decimal shares for five groups and Tableau 6.13's Christian split, on
    Tableau 2.9's ordinary-household populations, raked to annex A01's national counts with its
    non-response prorated (sources/ml.py). `Total` is carried in ml.csv and EXCLUDED in
    taxonomy/ml2022.py.
    """
    from ml2022 import EXCLUDED, resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "ml.csv",
                     low_memory=False, keep_default_na=False, na_values=[""])
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != 20:
        raise SystemExit(f"{df['geo_id'].nunique()} régions in ml.csv, expected 20 -- re-run "
                         "sources/ml.py")
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"]) - set(EXCLUDED))
    if unmapped:
        raise SystemExit(f"ml.csv categories with no node: {unmapped}")
    df = df[df["node"].notna() & (df["count"] > 0)].rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["may_ring"] = True
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "may_ring", "tier"]]


ENTRY = {
    "ml": dict(
        name="Mali",
        source="RGPH5 2022 (INSTAT), Caractéristiques culturelles Tableau 2.03 and État et "
               "structure Tableau 6.13",
        basis="self-identification, ordinary households",
        note_public=(
            "**Mali's 2022 census published religion for its 20 régions.** The table covers "
            "the 21.3 million people living in ordinary households. **96.4%** of them are "
            "Muslim, and 14 of the 20 régions are over 98% Muslim. "
            "**The exception is San, in south-central Mali.**It is 70.1% Muslim, **17.0%** Christian "
            "(9.5% Catholic and 7.3% Protestant), 8.3% animist and 3.7% with no religion. With "
            "3.8% of Mali's people it has a quarter of its Catholics, a third of its Protestants "
            "and almost half of its animists. Koutiala, beside it, is 4.3% Christian and 3.1% "
            "animist, and Bandiagara, on the Dogon plateau, is 6.0% Christian. "
            "**About a million people are not in the table.** 941,335 lived in areas the "
            "census could not reach because of insecurity, most of them in the régions of "
            "Tombouctou, Ségou, Ménaka and the old Mopti région, and their number was estimated "
            "from satellite images of buildings. Another 106,567 lived in collective households "
            "or had no home. "
            "**The religion question had no box for no answer.** The 0.2% left blank were "
            "spread over the religions in proportion, as the census's own tables do. Animism "
            "was an alternative to the other answers, so a Muslim or Christian who also keeps a "
            "traditional practice is counted once; the 2009 census counted 2.0% animists, "
            "against 0.7% here."),
        how="census, 2022",
        grain="régions, 1.07 million people on average",
        gap="4.68% not in the religion table: 4.20% in areas the census could not enumerate "
            "because of insecurity, whose number was modelled, and 0.48% in collective "
            "households or homeless",
        gap_share=0.0468,
        counts=_ml_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ml" / "ml_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ml_place_weight,
        note="RGPH5 2022: Caractéristiques culturelles Tableau 2.03 (région x 5 religions, two "
             "decimals) with État et structure Tableau 6.13 (7 answers, one decimal, rows "
             "forced to 100) splitting the Christians; Tableau 2.9 populations; raked to annex "
             "A01's national counts with its 48,746 Non Déclaré prorated, which is what Tableau "
             "2.01 prints. Universe ordinary households, 21,347,587 of 22,395,489. Boundaries "
             "COD-AB v03 (20 régions, 160 cercles, cercle counts equal Tableau 1.1). "
             "sources/ml.md has the record.",
    ),
}
