# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _gm_place_weight(place):
    """countries.py hook. `place` is the Kontur 400m hex layer scatter.py has read.

    Eight LGAs over 10,700 km2, from Banjul (9 km2 on COD-AB, 31,054 people in 2013) to Kerewan
    (2,253 km2); Brikama holds the sprawl west of Kanifing and a rural hinterland to the Senegal
    border (sources/gm_geo.py).
    """
    return _kontur_place_weight(place, "gm_hexes.gpkg", "sources/gm_geo.py")


def _gm_counts():
    """The Gambia 2013 census at LGA: 4 nodes on 8 units, every row `measured`.

    Annex H of the Spatial Distribution Report in counts, Kerewan as its male plus female tables
    (sources/gm.py). The 970 whose religion was left blank are read and not drawn.
    """
    from gm2013 import EXCLUDED, resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "gm.csv",
                     low_memory=False, keep_default_na=False, na_values=[""])
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != 8:
        raise SystemExit(f"{df['geo_id'].nunique()} LGAs in gm.csv, expected 8 -- re-run "
                         "sources/gm.py")
    df = df[~df["source_category"].isin(EXCLUDED)].copy()
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if unmapped:
        raise SystemExit(f"gm.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0].rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["may_ring"] = True
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "may_ring", "tier"]]


ENTRY = {
    "gm": dict(
        name="Gambia",
        name_in="The Gambia",
        source="2013 Population and Housing Census, Spatial Distribution Report (Gambia Bureau "
               "of Statistics), Annex H",
        basis="self-identification, total population",
        note_public=(
            "**The Gambia's 2013 census publishes religion for its eight Local Government "
            "Areas.** Of the people who gave a religion, **96.0%** answered Islam and 3.8% "
            "Christianity. Christians live mostly around the capital: Kanifing, the municipality "
            "next to Banjul, is **7.7%** Christian, Banjul and Brikama 4.8% each, and those three "
            "hold nine in ten of the country's Christians. Every LGA further up the river is "
            "under 1.3% Christian, and Basse is 0.5%. "
            "**The form had four answers: Islam, Christianity, Traditional and Other.** It had "
            "no answer for no religion, so anyone who gave none was recorded as Other or left "
            "blank, and no table names a church or a Muslim order. Traditional religion is "
            "0.06%, which counts only people who gave it as their one religion."),
        how="census, 2013",
        grain="Local Government Areas, 232,000 people on average",
        gap="0.05% whose religion was left blank on the form",
        gap_share=0.0005,
        counts=_gm_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "gm" / "gm_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_gm_place_weight,
        note="ANNEX H OF THE SPATIAL DISTRIBUTION REPORT is age x religion in counts for the "
             "nation and each LGA, by sex and by urban and rural; nothing finer crosses religion. "
             "Every table closes on itself. H.28 (Kerewan both sexes) REPRINTS H.31 (Kerewan "
             "urban, 50,188), so Kerewan is H.29 + H.30 = H.31 + H.34 = 220,080, cell by cell, "
             "and with it the eight LGAs equal H.1 in every cell and Table B.1's populations. The "
             "form (Form A Part 2, column 7) has four codes and no code for no religion or no "
             "answer; `Not stated` (970) is excluded. 1,106 of Kanifing's 1,996 `Other` have no "
             "age recorded. LGAS ARE COD-AB v01 ADM1 by pcode (region names), witnessed by "
             "geoBoundaries' LGA-named ADM1; the two draw Kanifing at 93.7 and 52.9 km2 against "
             "the census's 75.55, and Kontur holds the same 479,700 people in either, so the "
             "choice moves nobody. sources/gm.md has the record.",
    ),
}
