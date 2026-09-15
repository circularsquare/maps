# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


ENTRY = {
    "bm": dict(
        name="Bermuda",
        source="Census 2010 (UN Demographic Yearbook, table 28)",
        basis="self-identification",
        view=[-64.90, 32.23, -64.62, 32.40],
        gap_share=0.022,
        gap="2.2%, who did not state a religion",
        note_public=(
            "**Twenty-three categories for 64,237 people, the deepest religion list of any "
            "small country drawn here.** "
            "**The African Methodist Episcopal Church holds 8.6%, the highest AME share "
            "anywhere on this map**, including the United States. It arrived in 1870 and is "
            "the island's historically Black Methodist tradition, counted separately from "
            "the Methodists below it. "
            "**And the largest single answer is no religion at all**, 11,466 people or "
            "17.9%. That is far above Montserrat's 2.6%, Antigua's 4.1% and Dominica's "
            "6.1%; on this measure Bermuda reads less like the Caribbean than like a North "
            "Atlantic offshore financial centre, which is what it is. "
            "**One caution about the residual.** The published table prints both an *other "
            "religions* column and an *other* column, 5,926 people between them, and does "
            "not say how they differ. Both are drawn as one colour here. Since the Muslims, "
            "Jews, Baha'is, Rastafari and Ethiopian Orthodox all appear separately in the "
            "same table, whatever the distinction was, it is not one of those."),
        how="a census question, 2010",
        grain="the country as one unit, 64,200 people",
        counts=lambda: _micro_counts("bm", "bm2010"),
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "bm" / "bm_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_micro_place_weight("bm"),
        note="queue.md: *religion by sex only, and the 2016 census does not ask at all*, "
             "which is why 2010 is drawn. Kontur 64,223 against a census 64,237 is the "
             "closest agreement in this tier and is a coincidence of a stable population "
             "rather than evidence about either.",
    ),
}
