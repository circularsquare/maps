"""
Arab Barometer `Q1012` crossed with the logged sect `Q1012A` (Yemen, wave V, 2018-2019) ->
religiondots taxonomy.

Named for the drawn wave's last fieldwork year, the registry convention. `sources/ye.py` and
`sources/ye.md` have the construction; wave III (2013) is read there as a witness only.

    56.86%  Sunni                           -> islam.sunni
    24.42%  Just a Muslim                   -> islam
    18.19%  Zaydi                           -> islam.shia
     0.43%  Muslim, denomination not given  -> islam
     0.10%  Muslim, other denomination      -> islam

Shares are wave V's weighted respondents; the map's shares are these per governorate against
the Population Task Force's 2025 populations. Every row is `modelled` (§7b): nobody in Yemen has
published a count of religion or sect.

No Christian, Jewish, Bahá'í or Hindu answer is in the drawn wave, so nothing else is mapped.
The State Department puts all four together at under 1% of Yemen, and a household survey in
wartime reaches none of them.
"""

EXCLUDED = {}

REVIEW = {
    "Zaydi":
        "-> islam.shia. The file's code 14, labelled `Alawi` on Arab Barometer's cross-country "
        "list, which has no Zaydi code; no Yemeni is logged under code 6 `Shia`. It is read as "
        "Zaydi on its geography, which `sources/ye.py::zaydi_geography` asserts: 34% of the "
        "highland governorates, 0.12% of the southern and eastern nine, 1 of 120 in Hadramawt "
        "(so not the Ba 'Alawi sayyids), highest in Sa'dah, and the same places wave III's "
        "`Shia` answer lands on a different card. Zaydism is the Fiver branch of Shia Islam, so "
        "the node is exact at the depth it is drawn. **Not a new `islam.shia.zaydi` node**: it "
        "would be a legend row only Yemen uses, which AGENT_BRIEF §3 sends to Anita, and the map "
        "loses nothing at depth 2. Adding it later is this line and a branches.py entry.",
    "Sunni":
        "-> islam.sunni, carrying 815 respondents logged `Shafi'i`, folded in `sources/ye.py`. "
        "`branches.py` has `islam.sunni.shafii` and the fold is still right, because the split "
        "between the two boxes is a logging habit: Lahj is 100 Sunni and 0 Shafi'i, Al Maharah "
        "40 and 0, Ad Dali' 58 and 2, while Aden next door is 9 and 57 and Ibb 4 and 139. "
        "Yemen's Sunnis are Shafi'i by school almost throughout, so the fold loses no fact.",
    "Just a Muslim":
        "-> islam, the parent, as Iraq. 24.4% of wave V weighted, nothing shares it out "
        "(§14.4 rule 1). **Its geography is partly the fieldwork's**: eight of 21 governorates "
        "have nobody at all logged in this box, Ta'iz 0 of 260 and Lahj 0 of 100, which reads "
        "as how interviewers there recorded a plain `Muslim`. It passes the PSU split-half "
        "(+0.846), which cannot see a governorate team's habit, and the interviewer column is "
        "blank, so it cannot be tested. The note says so.",
    "Muslim, denomination not given":
        "-> islam. Authored label: 11 `refused` and 1 `don't know` on the sect item from people "
        "who said Muslim, as Iraq. Under the 1% floor; drawn at its governorate shares only "
        "because it shares the node with `Just a Muslim` (`sources/ye.py::SAME_NODE`).",
    "Muslim, other denomination":
        "-> islam. Authored label for the sect item's `other`, 2 respondents. Same treatment and "
        "reason as the line above.",
}

MAP = {
    "Sunni": "islam.sunni",
    "Zaydi": "islam.shia",
    "Just a Muslim": "islam",
    "Muslim, denomination not given": "islam",
    "Muslim, other denomination": "islam",
}


def resolve(category):
    """religiondots branch for a composed Arab Barometer answer, or None if off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
