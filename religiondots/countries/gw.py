# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _gw_place_weight(place):
    """countries.py hook. `place` is the Kontur 400m hex layer scatter.py has read.

    Nine units, one of them Bissau (SAB, 362,699 nationals on under 80 km2) and one of them
    the Bijagós archipelago; Oio, Bafatá and Gabú are thousands of km2 of villages
    (sources/gw_geo.py).
    """
    return _kontur_place_weight(place, "gw_hexes.gpkg", "sources/gw_geo.py")


def _gw_counts():
    """Guinea-Bissau RGPH 2009 at região: 5 nodes on 9 units, every row `measured`, may ring.

    Anexo Quadro 3 of the socio-cultural volume, in counts, Guinean nationals in ordinary
    households (sources/gw.py). `Total` and `ND` (228,718 who did not answer) are carried in
    gw.csv and are EXCLUDED in taxonomy/gw2009.py.
    """
    from gw2009 import EXCLUDED, resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "gw.csv",
                     low_memory=False, keep_default_na=False, na_values=[""])
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != 9:
        raise SystemExit(f"{df['geo_id'].nunique()} regiões in gw.csv, expected 9 -- re-run "
                         "sources/gw.py")
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"]) - set(EXCLUDED))
    if unmapped:
        raise SystemExit(f"gw.csv categories with no node: {unmapped}")
    df = df[df["node"].notna() & (df["count"] > 0)].rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["may_ring"] = True
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "may_ring", "tier"]]


ENTRY = {
    "gw": dict(
        name="Guinea-Bissau",
        source="RGPH 2009, Características socioculturais (INE Guiné-Bissau), Anexo Quadro 3",
        basis="self-identification, Guinean nationals in ordinary households",
        note_public=(
            "**Guinea-Bissau's 2009 census published religion for its nine regions, in "
            "counts.** The table covers Guinean nationals, and the 15.9% of them who gave no "
            "answer are not drawn. "
            "**The east is mostly Muslim.** Gabú is **86.5%** Muslim and Bafatá 77.1%. "
            "**Animism is largest in Biombo and Cacheu.** Biombo is **40.1%** animist and "
            "Cacheu 34.0%, against 0.3% in Gabú. The census recorded one religion per person, "
            "and its report notes that many people practise two, so this does not count "
            "Christians or Muslims who also follow a traditional religion. "
            "**Christians are concentrated in Bissau.** The capital is **40.2%** Christian and "
            "holds 46% of the country's Christians. The published tables do not divide them "
            "by church."),
        how="census, 2009",
        grain="regions, 160,000 people on average",
        # Both parts as shares of the 1,452,926 people enumerated (sources/gw.py prints the
        # arithmetic): ND, 228,718, is a column the census printed; the 10,699 outside the
        # table (1,933 foreign nationals, 5,070 with no nationality recorded, 3,696 in
        # collective households) are in no religion table at all.
        gap="16.5% of the people counted: 15.7% who did not answer the religion question, "
            "and 0.7% outside the religion table (foreign nationals, people with no "
            "nationality recorded, and residents of collective households)",
        gap_share=0.1648,
        counts=_gw_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "gw" / "gw_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_gw_place_weight,
        note="§11w READ THE NINE REGIONAL BOOKLETS; THE TABLE IS IN THE SOCIO-CULTURAL VOLUME'S "
             "ANNEX (sources.md §11aq). Anexo Quadro 3 is região x religion in COUNTS for "
             "Guinean nationals, every column equal to UNSD table 28 to the person, ND being "
             "UNSD's Unknown. The body prose swaps Gabú and Bafatá; every table, and a "
             "prediction from the annex's região x etnia table, has them as drawn. P.14 is a "
             "write-in, so no answer box names animism. Counts are not corrected for the 4.6% "
             "post-enumeration omission. sources/gw.md has the record.",
    ),
}
