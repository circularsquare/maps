# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _ie_counts():
    """CSO 2022 at Small Area, allocated to 24 categories."""
    import ie2022
    return _allocated_counts("ie", "small_area", ie2022)


ENTRY = {
    "ie": dict(
        name="Ireland",
        source="Census 2022 (CSO)",
        basis="self-identification",
        note_public=(
            "The finest geography on this map: 18,919 Small Areas, about 90 households "
            "each, so the dots sit where the people actually are rather than being spread "
            "across a county. The categories are the other way round — CSO publishes five "
            "at Small Area and 24 by county, so everything below Catholic, no religion and "
            "not stated is derived. One row reads 'Orthodox (Greek, Coptic, Russian)', "
            "which welds two churches that separated in 451 into a single number."),
        how="census, 2022",
        fill="from the same census at county level",
        grain="Small Areas, 250 people on average",
        counts=_ie_counts,
        # The counts are already ON the finest unit, as in Czechia — no placement layer.
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ie" / "smallareas2022" / "SMALL_AREA_2022.shp",
        place_unit=lambda g: g["SA_GUID__1"].astype(str),
        note="CSO is self_id; categories below county level are allocated (spec §3.9).",
    ),
}
