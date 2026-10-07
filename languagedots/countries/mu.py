# Mauritius. 2022 Housing and Population Census, Volume II Table D9 (sources/mu_hpc.py), language
# usually spoken at home, on religiondots' Kontur hexes for its 182 wards and village council
# areas (read-only). The 183 census rows carry religiondots' own geo_ids (joined by name to its
# Table D6 rows, totals equal per unit), so its mu_lookup.csv applies unchanged; Vacoas-Phoenix
# Ward 5 and Ward 6-West share one polygon there and are summed here.
from _shared import *  # noqa: F401,F403


def _counts():
    import mu2022
    df = pd.read_csv(NORM / "mu.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "unit"].copy()
    lut = pd.read_csv(RD_GEO / "mu" / "mu_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    if df["unit"].isna().any():
        raise SystemExit(f"mu: {sorted(df.loc[df['unit'].isna(), 'geo_id'].unique())} missing "
                         "from religiondots' mu_lookup.csv")
    if df["unit"].nunique() != 182:
        raise SystemExit(f"mu: {df['unit'].nunique()} units, expected 182")
    df["node"] = df["source_category"].map(mu2022.resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    if df["count"].sum() != 1_233_097:
        raise SystemExit(f"mu: drawn {df['count'].sum():,}, expected 1,233,097")
    return by_unit(df)


ENTRY = dict(
    name="Mauritius",
    source="2022 Housing and Population Census, Volume II Table D9 (Statistics Mauritius)",
    how="census, 2022, home language",
    parts=[dict(covers="Everyone", source="2022 census, language usually spoken at home",
                rest=True)],
    grain="182 wards and village council areas, 6,800 people on average",
    gap="Agalega's few hundred residents, whom the published tables leave out",
    # The main island only, as religiondots frames it: a box holding Rodrigues too (600 km east)
    # is mostly ocean. note_public says Rodrigues is there.
    view=[57.25, -20.55, 57.85, -19.95],
    counts=_counts,
    mappings=["mu2022"],
    place=RD_GEO / "mu" / "mu_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Statistics Mauritius asks which language a person usually speaks at home and accepts "
        "two. Its table by ward and village council area puts a home that speaks two under the "
        "first one the office lists. So Bhojpuri here includes the 63,101 people in homes that "
        "speak Bhojpuri and Creole; homes that speak Bhojpuri alone hold 29,827 people, not "
        "106,583. Most Hindi and Bangla speakers are men, which fits foreign contract workers "
        "rather than Mauritian families. Creole on Rodrigues, 600 km east of the main island "
        "and outside the opening view, is Rodrigues Creole, a variety of Mauritian Creole."),
)
