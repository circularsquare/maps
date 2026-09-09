"""
Arab Barometer `Q1012` crossed with `Q1012A` (Iraq, waves V, VI-3, VII and VIII) ->
religiondots taxonomy.

The file is named for the last wave in the pool, which is the registry convention
(`taxonomy/<cc><YYYY>.py`); wave VIII's fieldwork ran September 2023 to July 2024 and wave V's
was 2018-2019. `sources/iq.py` and `sources/iq.md` have the construction.

    45.62%  Shia                            -> islam.shia
    29.25%  Sunni                           -> islam.sunni
    22.78%  Just a Muslim                   -> islam
     0.93%  Muslim, other denomination      -> islam
     0.82%  Muslim, denomination not given  -> islam
     0.31%  Christian                       -> christianity
     0.29%  Other religion                  -> other.iq

Shares are as drawn, which is the pooled weighted governorate shares against the 2024 census's
governorate populations; every row is `modelled` in §7's sense, because nobody has published a
count of religion in Iraq since 1987.

**THREE ANSWERS SIT ON THE BARE `islam` NODE AND THEY ARE A QUARTER OF THE COUNTRY.** That is
the substantive decision in this file and it is a refusal rather than an omission: 24.5% of
Iraqis in this pool are Muslims who did not put themselves in a branch, and nothing here
shares them out between Sunni and Shia. Russia is the precedent and it is exact —
`branches.py`'s note on `islam.shia` records that 4.66% of Russia answered *"I profess Islam,
but am neither Sunni nor Shia"* and that those people stay on the parent. §14.4 rule 1 is the
rule: splitting them would invent a magnitude no source publishes, and it would invent it on
the one quantity in this file that most obviously moves with the fieldwork rather than with
the country (17.7%, 42.7%, 27.1%, 21.2% across the four waves).

**AND THEY ARE NOT SPREAD EVENLY, WHICH IS THE THING WORTH KNOWING ABOUT THEM.** Declining to
name a sect has a geography of its own and it passed the split-half (+0.766): it runs 45.9% in
Diyala, 40.1% in Salah al-Din, 39.6% in Nineveh, 33.2% in Kirkuk and 30.5% in Anbar against
2.6% in Erbil and Duhok, 6.1% in Sulaymaniyah and 7.0% in Najaf. The places where people
answer are the places where one answer is overwhelming; the places where they do not are
Baghdad and the mixed belt north and east of it. So the `islam` dots on this map are not noise
distributed at random, they are where the two communities actually meet, and a reader should
take Baghdad's drawn Sunni share in particular as a floor.
"""

EXCLUDED = {}

REVIEW = {
    "Just a Muslim":
        "-> islam, the PARENT, not islam.sunni and not islam.shia. `Q1012A`'s own wording for "
        "a Muslim who declines a branch, 22.78% of Iraq. See the module docstring for why "
        "nothing shares it out. The one thing worth adding here is what the alternative would "
        "have been: apportioning these people in each governorate at that governorate's "
        "Sunni:Shia ratio among those who did answer. It is a tempting operation because it "
        "preserves every total and produces a tidier map, and it is exactly §14.4 rule 1's "
        "prohibition — it would state, in Baghdad, that the people who would not name a sect "
        "divide 81:19 the way the people who would do, which is the least likely thing to be "
        "true about them.",
    "Muslim, denomination not given":
        "-> islam. This is an AUTHORED label for a composed answer, not a wording the survey "
        "prints, and `sources/iq.py`'s `COMPOSED` says so: it is the respondent who answered "
        "Muslim to `Q1012` and then refused or did not know the follow-up, plus the handful "
        "with a missing sect value. 83 people. "
        "**They are kept in the universe where Jordan's refusals were dropped, and the "
        "difference is which question was refused.** Jordan's eleven refused the religion "
        "question itself, so nothing was known about them and spreading them would have drawn "
        "9,600 Jordanians on the strength of eleven non-answers (§9cq). These people have "
        "already said they are Muslim; only the branch is missing, which is the same thing "
        "`Just a Muslim` says in a different tone of voice.",
    "Muslim, other denomination":
        "-> islam. `Q1012A`'s `Other` box, 89 people, 0.93%. Also an authored label, and it "
        "exists because `Other` is a box on BOTH questionnaires — the sect card and the "
        "religion card — and those are different answers from different people. Left with "
        "their printed wording the two merge into one category with every total still adding "
        "up, which is the failure `ab.assert_one_wording` exists to catch and which it cannot "
        "see here, because the two spellings are identical rather than merely similar. "
        "It goes to `islam` rather than to `other.iq` for the obvious reason: these "
        "respondents said Muslim to the religion question. Iraq's Sufi orders, its Salafis, "
        "and anyone who names a school the card does not offer are in here.",
    "Shia":
        "-> islam.shia, and it carries 221 respondents who answered `Ja'fari` and two who "
        "answered `Alawi`, folded in `sources/iq.py` rather than mapped here. **`branches.py` "
        "HAS `islam.shia.jaafari`**, added with Türkiye, so the fold is a choice and not a "
        "gap in the tree. The reason is that this instrument does not measure a school: "
        "Ja'fari runs 1.6%, 2.8%, 1.9% and 4.2% across the four waves, and §11af rejected "
        "this same column as a madhhab layer for the Maghreb. Türkiye's schools come from the "
        "Diyanet's own survey, which asks the madhhab outright and does not put Sunni and "
        "Shia on the same card. The `Alawi` fold is two people and is argued at `SECT_FOLD`.",
    "Sunni":
        "-> islam.sunni, and it carries 105 `Shafi'i`, 19 `Hanbali` and 2 `Maliki`, folded "
        "the same way and for the same reason. The Shafi'i fold is the one with a "
        "consequence: Iraq's Shafi'is are overwhelmingly Kurds, and only five of the 105 are "
        "in the early wave half, so left as their own category they would fail the split-half "
        "on noise and `ab.build` would spread Kurdish Shafi'is across Basra at the national "
        "rate. Folded, they are Sunnis in Erbil and Sulaymaniyah, where they were counted.",
    "Christian":
        "-> christianity, the bare family node, NOT `christianity.eastern`. 0.31% of the "
        "pool, 25 respondents, drawn as **143,000 people** — which is close to the 150,000 "
        "usually quoted as what is left of a community that was roughly 1.5 million in 2003. "
        "The overwhelming majority are Chaldean Catholics, Syriac Orthodox, Syriac Catholics "
        "and Assyrian Church of the East, so `christianity.eastern` would be wrong for the "
        "largest group of them and `christianity.catholic` would be wrong for the second. "
        "Jordan (taxonomy/jo2024.py) declined the same deepening on the same reasoning and "
        "this follows it. Iraq's case is sharper in one way: the Chaldean Church is in "
        "communion with Rome while its rite is East Syriac, so even a correct split needs a "
        "published proportion, and no Iraqi source has published one this century. "
        "**Its geography is not drawn either**, being under §11ad's 1% floor, so these dots "
        "are spread at the national rate rather than placed in Nineveh and Baghdad where the "
        "community actually is. That is the §14.4 rule 2 answer for Iraq and it is deliberate.",
    "Other religion":
        "-> other.iq. `Q1012`'s `Other` (wave VII) and `Something else: SPECIFY_______` "
        "(wave VI-3) are one box under two wordings, and `sources/iq.py`'s `COMPOSED` merges "
        "them under an authored label, because the label has to be qualified anyway to keep "
        "it apart from the sect card's `Other`. 27 respondents, 0.29%. "
        "The node's own text names what is inside it — Yazidis, Sabean-Mandaeans, Kaka'i, "
        "Zoroastrians, Bahá'ís — and says why the figure is a floor of roughly a third: the "
        "Yazidis alone are put at 400,000 to 500,000, and a household survey does not reach a "
        "population that has been in camps since 2014. It is NOT mapped to bare `other`, "
        "which `branches.py` says nothing should ever be mapped to directly.",
}

MAP = {
    "Shia": "islam.shia",
    "Sunni": "islam.sunni",
    "Just a Muslim": "islam",
    "Muslim, other denomination": "islam",
    "Muslim, denomination not given": "islam",
    "Christian": "christianity",
    "Other religion": "other.iq",
}

# No COLUMNS dict (spec §7a-i-1). Every row in this country is `modelled` rather than
# `derived`, and the roll-up is about where a DERIVED row was actually counted. Nothing here
# was counted anywhere: Iraq's last published religion tabulation is the 1987 census.


def resolve(category):
    """religiondots branch for an Arab Barometer answer, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
