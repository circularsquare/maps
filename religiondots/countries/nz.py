# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _nz_counts():
    """Stats NZ at SA2, allocated to 159 categories.

    NOTE: these are RESPONSES, not people — the census allows up to four religions per
    person, so the categories sum to 5,003,112 against an SA2 population of 4,993,920.
    A New Zealand dot is a response; every other country's dot is a person (nz2023.py).
    """
    import nz2023
    return _allocated_counts("nz", "sa2", nz2023)


def _nz_place_unit(g):
    """SA1 -> its SA2, via the concordance nz_geo.md §5 derived spatially.

    Stats NZ publishes no SA1->SA2 lookup reachable without a datafinder key, and neither
    the SA1 boundary service nor the meshblock service carries an SA2 column. SA1s nest
    inside SA2s by construction, so the spatial join that produced the CSV is exact.

    LANDWATER 21 is Inland Water — 71 SA1s holding six people between them. Mapped to NaN
    so `groupby` drops them and no dot is ever placed in a lake.
    """
    lut = pd.read_csv(HERE / "data" / "geo" / "nz" / "sa1_2023_to_sa2_2023.csv", dtype=str)
    sa2 = dict(zip(lut["SA12023_V1_00"], lut["SA22023_V1_00"]))
    unit = g["SA12023_V1_00"].astype(str).map(sa2)
    return unit.where(g["LANDWATER"].astype(str) != "21")


ENTRY = {
    "nz": dict(
        name="New Zealand",
        source="Census 2023 (Stats NZ), 2018 structure",
        basis="self-identification, up to 4 responses per person",
        note_public=(
            "Two things here are unlike the rest of the map. The census lets a person give "
            "up to four religions, so a dot is a response rather than a person — about "
            "9,000 people are drawn twice. And the denominations are five years older than "
            "the totals: Stats NZ published 166 categories in 2018 and only 13 by area in "
            "2023, so the fine ones are 2018 shares applied to 2023 counts. Ratana and "
            "Ringatu, the churches founded by Maori prophets, are counted separately here "
            "and almost nowhere else. So is Jedi, at 22,605 — more than Baha'i, Jain, "
            "Taoist and Zoroastrian combined."),
        how="census, 2023 totals with 2018 denominations",
        fill="from the 2018 census",
        grain="statistical areas, 2,000 people on average",
        counts=_nz_counts,
        # SA1 2023, clipped to the coastline. Median 150 people, IQR 120-183 — the tightest
        # placement layer on the map, fourteen times finer than the SA2 the counts are on
        # and much tighter than a US census tract (spec §8.2).
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "nz" / "sa1_2023_clipped.geojson",
        place_unit=_nz_place_unit,
        note="Stats NZ is self_id with multiple response; categories below the national "
             "level are allocated from the 2018 table (spec §3.9 and §3.4 at once).",
    ),
}
