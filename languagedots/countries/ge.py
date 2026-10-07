# Georgia. 2024 census native language by self-governed unit (sources/ge_census.py), on Kontur hexes
# keyed to COD-AB's municipalities and cities (sources/ge_geo.py). sources/ge.md is the record.
# Abkhazia (2011 census, nationality read as language) and South Ossetia (2015 census, native
# language), which the 2024 census did not enumerate, are extra units AB-* and SO-* from
# sources/ge_breakaway.py, as religiondots keeps them inside Georgia; sources/ge.md, "Abkhazia and South Ossetia".
from _shared import *  # noqa: F401,F403

_EXPECTED_UNITS = 64
_AB, _SO = 240_705, 53_439       # Abkhazia's census population; South Ossetia's, language stated


def _counts():
    import ge2015_breakaway
    import ge2024
    df = pd.read_csv(NORM / "ge.csv")
    df = df[df["geo_level"] == "unit"].copy()
    if df["geo_id"].nunique() != _EXPECTED_UNITS:
        raise SystemExit(f"ge: {df['geo_id'].nunique()} units, expected {_EXPECTED_UNITS}; "
                         "re-run sources/ge_census.py")
    df["node"] = df["source_category"].map(ge2024.resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    br = pd.read_csv(NORM / "ge_breakaway.csv")
    for terr, n in (("Abkhazia", _AB), ("South Ossetia", _SO)):
        got = br.loc[br["territory"] == terr, "count"].sum()
        if abs(got - n) > 1:
            raise SystemExit(f"ge_breakaway.csv: {terr} {got:,.0f}, expected {n:,}; "
                             "re-run sources/ge_breakaway.py")
    br["node"] = br["source_category"].map(ge2015_breakaway.resolve)
    df = pd.concat([df[["geo_id", "node", "count", "tier"]], br[["geo_id", "node", "count", "tier"]]])
    df = df.rename(columns={"geo_id": "unit"})
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Georgia",
    source=("2024 Population and Agricultural Census, native language and Georgian language "
            "knowledge level (National Statistics Office of Georgia); Abkhazia's 2011 census, "
            "nationality; South Ossetia's 2015 census, native language"),
    how=("census, 2024, native language; Abkhazia from its own 2011 census of nationality, "
         "South Ossetia from its own 2015 census of native language"),
    parts=[
        dict(covers="Abkhazia",
             source="2011 census of Abkhazia, nationality by district, read as language",
             people=_AB),
        dict(covers="South Ossetia",
             source="2015 census of South Ossetia, native language by district",
             people=_SO),
        dict(covers="The rest of Georgia", source="2024 census, native language", rest=True),
    ],
    grain=("64 municipalities and cities, 61,000 people on average; Abkhazia's 8 districts and "
           "South Ossetia's 5"),
    gap=("the 1.2% who did not state a native language; and the 93 people in South Ossetia who "
         "did not"),
    view=[40.0, 41.0, 46.8, 43.6],
    # Natural Earth breakaway areas this entry now draws: not_drawn.py stops hatching them whole
    drawn_named=("Abkhazia", "South Ossetia"),
    counts=_counts,
    mappings=["ge2024", "ge2015_breakaway"],
    place=GEO / "ge" / "ge_plus_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The 2024 census asked each person's native language, which in the countries of the "
        "former Soviet Union leans towards identity rather than everyday use. 85.1% named "
        "Georgian, 6.8% Azerbaijani, 3.5% Armenian and 1.4% Russian. Mingrelian and Svan have "
        "no answer of their own on the form, so their speakers are drawn as Georgian. The other "
        "languages, 1.9%, are drawn as one colour, among them Kurmanji, Chechen in Pankisi and "
        "the languages of foreign residents. "
        "Abkhazia and South Ossetia, outside the government's control and not in the 2024 "
        "census, are drawn from censuses their own authorities took, which Georgia does not "
        "recognise. Abkhazia's, in 2011, counted nationality only; each nationality is drawn as "
        "its language, less an estimated share speaking Russian, and the Georgians of Gal as "
        "Georgian though most speak Mingrelian at home. South Ossetia's, in 2015, asked native "
        "language: 91% named Ossetian and 7% Georgian."),
)
