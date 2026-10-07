# Peru. Censo 2017 mother tongue (C5P11, aged 3+) per district from INEI's REDATAM server
# (sources/pe_censo.py), on religiondots' Kontur hexes for Peru's districts, read-only.
from _shared import *  # noqa: F401,F403


def _counts():
    import pe2017
    df = pd.read_csv(NORM / "pe.csv", dtype={"geo_id": str})
    df["node"] = df["source_category"].map(pe2017.resolve)
    df = df[df["node"].notna()].copy()        # no answer, does not hear or speak: in `gap`
    # religiondots joins the six-digit ubigeo to COD-AB's adm3 p-codes; Mazamari and Pangoa
    # (Satipo, Junin) share one COD polygon, PE120699, so both land there and are summed
    lut = pd.read_csv(RD_GEO / "pe" / "pe_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    if df["unit"].isna().any():
        raise SystemExit(f"pe: {df.loc[df['unit'].isna(), 'geo_id'].nunique()} districts "
                         "missing from religiondots' pe_lookup.csv")
    return by_unit(df)


ENTRY = dict(
    name="Peru",
    source="Censos Nacionales 2017 (INEI), variable C5P11, the language learned in childhood, "
           "tabulated per district from INEI's own REDATAM server",
    how="census, 2017, mother tongue, aged 3 and over",
    parts=[dict(covers="Everyone aged 3 and over",
                source="2017 census, language learned in childhood", rest=True)],
    grain="1,874 districts, 14,900 people aged 3 and over on average",
    gap="children under 3, 1.4 million (4.9%), whom the census does not ask, and 230,064 people "
        "(0.8%) who gave no answer or neither hear nor speak",
    view=[-81.5, -18.5, -68.5, 0.1],
    counts=_counts,
    mappings=["pe2017"],
    place=RD_GEO / "pe" / "pe_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The census asked everyone aged 3 or over which language they first learned to speak "
        "as a child; 39 native languages are drawn. Quechua is one category, although it "
        "covers many regional varieties that are not all mutually intelligible; Kichwa, the "
        "Quechua of Loreto and San Martín, is counted apart. Inside a district every "
        "language's dots are spread by population alone, so in an Amazonian district a native "
        "community's language may be drawn in the district town."),
)
