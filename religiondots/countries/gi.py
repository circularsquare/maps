# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


ENTRY = {
    # ---- GIBRALTAR (sources/gi.py, sources/gi.md) ------------------------------------------
    # The microstate tier's shape (one unit, Kontur hexes) from an office's own table. The
    # same table prints seven residential areas, which are not drawn: sources/gi.md §5.
    "gi": dict(
        name="Gibraltar",
        source="Census of Gibraltar 2022, Table 42 (HM Government of Gibraltar)",
        basis="self-identification",
        view=[-5.38, 36.10, -5.33, 36.165],
        how="a census question, 2022",
        grain="the territory as one unit, 37,900 people",
        # No gap. The form has no not-stated box and the report prints blanks inside
        # `Other/Not stated`, which is drawn on other.gi (taxonomy/gi2022.py REVIEW, the shape
        # of ask 023's ruling for Iran), so nobody in the table is left off the map.
        note_public=(
            "**The census asks religion with eight boxes and prints it in counts.** Roman "
            "Catholics are **63.5%** of Gibraltar's 37,936 usual residents, down from 72.1% in "
            "2012, and people with no religion **14.1%**, up from 7.1%. The Church of England "
            "is 6.7%, Muslims 5.0%, other Christians 4.0%, Jews 2.8% and Hindus 1.8%. "
            "**Other religion and not stated are one figure.** The 2022 form has an Other box "
            "and no box for not stating a religion, and the report prints the two together, "
            "779 people (2.1%), drawn here as other religion. The 2012 report printed them "
            "apart: 365 other and 44 not stated. "
            "**The territory is drawn as one unit.** The census also prints religion for seven "
            "residential areas, but publishes no map of them, and at one dot per thousand "
            "people Gibraltar draws 34 dots, so where they land follows where people live. The "
            "areas do differ. Town Area, the old town around Main Street, holds **490** of the "
            "1,070 Jews and **658** of the 1,909 Muslims, 13.0% and 17.4% of its 3,783 "
            "residents, and the Reclamation Areas hold 380 of the 693 Hindus."),
        counts=lambda: _micro_counts("gi", "gi2022"),
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "gi" / "gi_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_micro_place_weight("gi", "sources/gi.py"),
        note="CENSUS OF GIBRALTAR 2022, TABLE 42, IN COUNTS, newer than UNSD's 2012 row. Drawn "
             "as one unit: the seven residential areas are defined only as lists of "
             "enumeration areas, which are lists of streets, with no map; and Kontur's 20 "
             "hexes (two of them over 6,800 people) cannot separate them. CHECKED: Table 42 "
             "parsed off the page equals the transcription and closes in every cell; the "
             "report's 1970-2022 series has the same 2022 column, and its 2012 and 2001 columns "
             "equal UNSD table 28; Table 43's 78 enumeration areas rebuild every area through "
             "Appendix 9, which also shows EA 85 (70 people) tabulated in South District rather "
             "than Institutions and one cell 4 people apart, pinned. Other/Not stated is on "
             "other.gi. sources/gi.md has the record.",
    ),
}
