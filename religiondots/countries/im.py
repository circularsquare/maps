# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


ENTRY = {
    # ---- ISLE OF MAN (sources/im.py, sources/im.md) -----------------------------------------
    # The microstate tier's shape (one unit, Kontur hexes) from the office's own table, which is
    # island-wide only. No finer religion table is published.
    "im": dict(
        name="Isle of Man",
        name_in="the Isle of Man",
        source="2021 Isle of Man Census, Report Part I, Table 2.12 (Statistics Isle of Man)",
        basis="self-identification",
        view=[-4.85, 54.03, -4.30, 54.42],
        how="a voluntary census question, 2021",
        grain="the island as one unit, 84,100 people",
        gap="11.4% of residents, who did not answer a voluntary question",
        gap_share=0.114,
        note_public=(
            "**The 2021 census was the first on the island to ask religion, and the question "
            "was voluntary.** Of the 84,069 residents, 74,487 answered. Christians are "
            "**54.7%** of those who answered and people with no religion **43.8%**; Muslims "
            "and Buddhists are 0.5% each, Hindus 0.4% and Jews 0.2%. The 9,582 who did not "
            "answer are not drawn. "
            "**The form offered more boxes than the table prints.** It had one box for all "
            "Christians, so no church can be shown, and it also offered Sikhism and a "
            "write-in. The published table has no Sikh row and prints other religions as 0, "
            "and the report does not say where those answers went. "
            "**The island is drawn as one unit.** The census publishes religion only for the "
            "whole island, so where the dots land follows where people live, not where each "
            "religion is."),
        counts=lambda: _micro_counts("im", "im2021"),
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "im" / "im_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_micro_place_weight("im", "sources/im.py"),
        note="2021 ISLE OF MAN CENSUS, REPORT PART I, TABLE 2.12, IN COUNTS; not in UNSD "
             "table 28. Island-wide only, drawn as one unit on Kontur's 567 hexes (84,591 "
             "people against 84,069 residents). CHECKED: both reports pinned by digest; Table "
             "2.12 parsed off the page equals the transcription and closes by age band; Table "
             "2.1's total row gives 84,069 and every band has answered <= residents; p.13's "
             "figures recompute; the Part II form's Q8 offers Sikhism and Other, which the "
             "table does not print. The 9,582 non-answers are gap. sources/im.md has the record.",
    ),
}
