# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _bf_place_weight(place):
    """countries.py hook. `place` is the Kontur 400m hex layer scatter.py has read.

    Kadiogo is 2,869 km2 holding 1,727,390 people (Ouagadougou) and Kompienga 6,967 km2
    holding 75,867; the northern and eastern provinces have their people in villages along
    seasonal rivers (sources/bf_geo.py).
    """
    return _kontur_place_weight(place, "bf_hexes.gpkg", "sources/bf_geo.py")


def _bf_counts():
    """Burkina Faso RGPH 2006 at province: 6 nodes on 45 units, every row `measured`, may ring.

    Tableau A5.6 of the structure volume, in counts, every resident (sources/bf.py). `Total`
    is carried in bf.csv and EXCLUDED in taxonomy/bf2006.py.
    """
    from bf2006 import EXCLUDED, resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "bf.csv",
                     low_memory=False, keep_default_na=False, na_values=[""])
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != 45:
        raise SystemExit(f"{df['geo_id'].nunique()} provinces in bf.csv, expected 45 -- re-run "
                         "sources/bf.py")
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"]) - set(EXCLUDED))
    if unmapped:
        raise SystemExit(f"bf.csv categories with no node: {unmapped}")
    df = df[df["node"].notna() & (df["count"] > 0)].rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["may_ring"] = True
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "may_ring", "tier"]]


ENTRY = {
    "bf": dict(
        name="Burkina Faso",
        source="RGPH 2006, État et structure de la population (INSD), Tableau A5.6",
        basis="self-identification, resident population",
        note_public=(
            "**Burkina Faso's 2006 census published religion for its 45 provinces, in counts.** "
            "The table covers all 14.0 million residents, and children under six were given "
            "their mother's religion. "
            "**The north is almost entirely Muslim.** Oudalan is **98.0%** Muslim, and Loroum, "
            "Séno and Soum are each over 96%. "
            "**Animism is largest in the southwest and the east.** Poni is **74.8%** animist and "
            "Noumbiel 72.5%; in the east, Tapoa is 57.2%. "
            "**Catholics are most concentrated in the centre-west, around the capital and in Ioba "
            "in the southwest.** Sanguié is 44.4% Catholic, Ioba 40.3% and Kadiogo, the province "
            "of Ouagadougou, 36.2%.Gnagna, "
            "in the east, is 18.7% Protestant. "
            "**The counts date from December 2006.** The 2019 census counted **9.0%** animists "
            "nationally, against 15.3% here, and its Sud-Ouest région fell from 64.9% animist "
            "to 48.1%."),
        how="census, 2006",
        grain="provinces, 310,000 people on average",
        counts=_bf_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "bf" / "bf_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_bf_place_weight,
        note="RGPH 2006 structure volume, Tableau A5.6 (province x religion, COUNTS), only on "
             "insd.bf's retired trees via Wayback. Every column equals UNSD table 28 to the "
             "person; provinces summed by région equal A5.5, totals equal A 3.1 bis. P15's six "
             "printed codes are the columns, animist separate from none, no non-response code. "
             "`Autre` probably holds some blank answers (A5.7). Boundaries geoBoundaries ADM2 "
             "(COD-AB is the 2025 reform). sources/bf.md has the record.",
    ),
}
