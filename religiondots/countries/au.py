# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _au_counts():
    """ABS 2021 at SA2, allocated to 148 categories (§2.4, branch level)."""
    import au2021
    return _allocated_counts("au", "sa2", au2021)


ENTRY = {
    "au": dict(
        name="Australia",
        source="Census of Population and Housing 2021 (ABS)",
        basis="self-identification, voluntary question",
        view=[112.0, -44.0, 154.5, -9.5],
        note_public=(
            "The religion question is the only voluntary one on the Australian census, and "
            "about 7% left it blank; those people are not drawn. What is drawn is the "
            "deepest list on this map outside the United States — 148 groups, including "
            "three separate Orthodox communions that most sources collapse into one, and "
            "the Mandaeans, of whom Australia now holds more than Iraq does. Groups below "
            "the state level are derived: the ABS publishes 150 religions nationally and "
            "34 by SA2, so the fine ones are split out proportionally and can show "
            "composition but not presence."),
        how="census, 2021, voluntary question (7% left it blank)",
        fill="from the same census's national table",
        grain="statistical areas, 9,700 people on average",
        counts=_au_counts,
        # Counts are on SA2; SA1s carry their parent's code, so no spatial join is needed.
        # SA1s are built to about 406 people, which is the cleanest §8.2 case in the
        # project — finer than a US tract and 25x finer than the SA2 the counts are on.
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "au" / "SA1_2021_AUST_GDA2020" /
              "SA1_2021_AUST_GDA2020.shp",
        place_unit=lambda g: g["SA2_CODE21"].astype(str),
        note="ABS is self_id on the census's only voluntary question; categories below "
             "state level are allocated (spec §3.9).",
    ),
}
