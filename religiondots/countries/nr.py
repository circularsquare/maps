# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


ENTRY = {
    # ---- NAURU (sources/nr.py) ----------------------------------------------------------
    # The microstate tier's tenth country and the only one not built from the Yearbook:
    # Nauru is not in UNSD table 28 at all, so this comes from the Nauru Bureau of
    # Statistics' own 2021 census workbook and is nineteen categories deep.
    "nr": dict(
        name="Nauru",
        source="2021 Population and Housing Census (Nauru Bureau of Statistics)",
        basis="self-identification",
        view=[166.885, -0.573, 166.980, -0.484],
        how="a census question, 2021",
        grain="the country as one unit, 11,700 people",
        # gap_share.py reports 0.49% and declines to WRITE it, because Nauru is a "rows only"
        # case: the residual is a printed column and there is no separately stated universe
        # to check it against. Written by hand because here those are the same number — the
        # nineteen categories sum to the census's own 11,680 with a difference of zero, so
        # 57/11,680 is exact rather than a candidate. `--check` passes on it.
        gap_share=0.00488,
        gap="the 57 people, 0.49% of the country, who did not wish to answer the question",
        note_public=(
            "**Two churches hold two thirds of Nauru and they are 42 people apart.** The "
            "Nauru Congregational Church is **4,001** people and the Roman Catholics "
            "**3,959**, out of 11,680. Protestant work on the island began with a Gilbertese "
            "teacher in 1887 and the Catholic mission followed soon after, and Nauru has been "
            "split between the two ever since. The gap is closing: in 2011 it was 35.7% "
            "against 33.0%. "
            "**The census offered ten answers and printed nineteen.** Question 307 of the "
            "2021 form listed no religion, seven named churches, a box for people who did not "
            "wish to answer, and other religion with a line to write on. The Bureau of "
            "Statistics then read the write-ins and gave nine of them rows of their own, "
            "from the Shalosh Pentecostal Church at 186 people down to six Hindus, so the "
            "published table is deeper than the question was. The 98 people still under "
            "other religion are the ones it did not classify. "
            "**The largest change since 2011 is one church appearing and another emptying.** "
            "Pacific Light House has **706** people here, 6.0% of the country, and had no "
            "cell at all in the 2011 census; the Nauru Independent Church went from 945 "
            "people to **410** over the same ten years. Nothing published connects the two "
            "movements. "
            "**The whole island is one unit, and that is the source rather than a "
            "shortcut.** At one dot per thousand people Nauru draws eight dots and eleven rings, so where "
            "on the island they land is decided by where people live and asserts nothing "
            "about religion. The census does publish religion by district, but only for "
            "Nauruan citizens and with the smaller churches folded back together, and "
            "eight dots over fifteen districts could not have shown it."),
        counts=lambda: _micro_counts("nr", "nr2021"),
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "nr" / "nr_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_micro_place_weight("nr", "sources/nr.py"),
        note="**THE FIRST COUNTRY TO JOIN THE MICROSTATE TIER FROM AN OFFICE**, and it is "
             "deeper than the Yearbook is for any of the other nine: nineteen categories "
             "against Palau's nine and the Marshall Islands' four. Nauru is not in UNSD "
             "table 28 at all. "
             "**THE FILES ARE BEHIND WP FILE DOWNLOAD, NOT BEHIND A WALL.** "
             "`nauru.prism.spc.int` 301-redirects to `stats.gov.nr`, which runs the same "
             "plugin as Fiji (§9bd) and PNG (§11ab) with the same unauthenticated AJAX "
             "route; `id=0` returns all 131 files ten to a page. A previous session swept "
             "`wp/v2/media`, found one non-image file and concluded the documents were not "
             "on the site. "
             "**READING THE QUESTIONNAIRE IS WHAT MAKES THE TABLE INTERPRETABLE.** Nine of "
             "the nineteen rows are the office's back-coding of free text, so they are "
             "respondents' own words rather than an office classification, and the residual "
             "is a coding tail rather than a sampling one. "
             "**THE PARTITION IS EXACT**, nineteen categories to 11,680 with a difference of "
             "zero, and `sources/nr.py` pins the total and the category count so a "
             "re-publication fails the build. **The citizen-only district table reconciles "
             "to the person** against G-7, which is how it is known to be a clean subset; "
             "`sources/nr.md` carries its shares.",
    ),
}
