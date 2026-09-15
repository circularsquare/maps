# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _lk_counts():
    """DCS CPH 2024 at GN division: 6 categories on 14,003 units.

    ONE level, no allocation, nothing modelled — the census publishes these categories at
    this geography and the map draws exactly that. 14,003 units for 21.8M people is about
    1,555 people each, the finest grain in the project outside the US tracts and the German
    grid, and it arrives without any of the machinery those two needed.

    THE WHOLE POPULATION IS DRAWN, which is true of no other country here and is not the
    compliment it sounds like. DCS's six categories partition all 21,781,800 people with no
    'not stated', no 'no religion' and no refusal line at all, so everyone is assigned a
    religion whether or not they profess one. §3.5's "undercounting is marked, not filled"
    has nothing to mark; the thing to say instead is that Sri Lankan irreligion is not
    absent from this map, it is invisible on it, distributed among the six.

    PLACEMENT IS COARSER FOR 53 UNITS. The only boundary set that reaches GN divisions is
    OCHA's COD, valid 2022, and 53 of the 2024 census's divisions have no 2022 polygon.
    Their dots are placed in the unmatched remainder of their DS division instead — 95,641
    people, 0.44%. The counts are still measured; it is where inside the map they sit that
    is looser, so `tier` stays `measured` and sources/lk_geo.py carries the detail.
    """
    from lk2024 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "lk.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"] == "gnd"].copy()

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "lk": dict(
        name="Sri Lanka",
        source="Census of Population and Housing 2024 (Department of Census and Statistics)",
        basis="self-identification",
        view=[79.4, 5.7, 82.1, 10.0],
        note_public=(
            "The finest map here, and the shallowest. Sri Lanka's 2024 census publishes "
            "religion for all 14,003 **Grama Niladhari divisions** — about 1,550 people "
            "each, so a dot sits in something the size of a few streets or one village — "
            "and it offers only six answers to put in them: Buddhist, Hindu, Islam, Roman "
            "Catholic, other Christian, other. Nowhere else on this map is the geography "
            "this good and the religion this coarse. Buddhists are 69.8% of the country "
            "and arrive with no school attached, though the island is one of Theravada's "
            "historic homes; Muslims are 10.7% with no branch given. What the resolution "
            "does buy is that the four traditions sit in visibly different places rather "
            "than blended into district averages: the Hindu north around Jaffna and, "
            "separately, the Hindu hill country in the tea districts — Sri Lankan Tamils "
            "and Indian-origin Tamils, two populations the religion question cannot tell "
            "apart but the map can; Muslims along the eastern coast from Trincomalee to "
            "Kalmunai and in pockets inland; and a Roman Catholic coastal strip running "
            "north from Negombo through Chilaw, which is the sharpest religious boundary "
            "in the country and is invisible at any coarser grain. "
            "**Everyone is drawn, and that is a fact about the question rather than the "
            "country.** The six categories account for all 21,781,800 people: there is no "
            "'no religion', no 'not stated' and no refusal line anywhere in this census. "
            "Irreligious Sri Lankans are not missing from this map, they are counted "
            "inside one of the six, and no total here can be read as a measure of belief. "
            "One more thing the census does to itself: where a religion has fewer than ten "
            "people in a GN division its count is moved into 'other', so the small groups "
            "— the Bahá'ís, the Parsis, the Malay Muslims — cannot be recovered at this "
            "grain, and 'other' holds an unknown mixture of them and that suppressed "
            "tail."),
        how="census, 2024",
        grain="village divisions, 1,550 people on average",
        counts=_lk_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "lk" / "lk_gnd.gpkg",
        place_unit=lambda g: g["gnd"].astype(str),
        note="The census codes and the COD boundary codes are the same shape and disagree: "
             "13 DS divisions were renumbered between COD's 2022 vintage and the 2024 "
             "census, so joining on the pcode places 762,824 people in the wrong division "
             "with no symptom. sources/lk_geo.py aligns DS divisions by name first and "
             "only then matches GN codes inside a pair. 53 GN divisions have no 2022 "
             "polygon and are placed in their DS division's unmatched remainder.",
    ),
}
