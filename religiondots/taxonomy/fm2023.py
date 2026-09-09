"""Micronesia — FSM 2023 census Table B6 -> religiondots taxonomy.

Eleven categories, an exact partition of the enumerated population at every unit, drawn on
**33 units at two tiers**: Yap's 20 municipalities and Pohnpei's 11 from their own per-state
workbooks, Chuuk and Kosrae whole because theirs publish no religion table (`sources/fm.py`).
**No new nodes beyond `other.fm`.** Shares below are of the 75,576 people drawn.

    55.5%  Roman Catholic           -> christianity.catholic.latin
    37.1%  Congregation/Protestant  -> christianity.protestant
     1.5%  Other religion           -> other.fm                      <- new
     1.3%  Mormon                   -> christianity.latterday
     1.2%  Baptist                  -> christianity.baptist
     0.7%  Assembly of God          -> christianity.pentecostal.trinitarian
     0.7%  No religion/Refused      -> unaffiliated
     0.6%  Apostolic                -> christianity.pentecostal
     0.6%  SDA                      -> christianity.adventist
     0.4%  Pentecostal              -> christianity.pentecostal
     0.3%  Jehovah's Witness        -> christianity.witnesses

**THE COUNTRY IS ONE MISSION BOUNDARY AND THE MUNICIPALITIES DRAW IT PROPERLY.** Catholic and
Congregational are 92.6% of Micronesia between them, and they invert completely across it:
Kosrae is **88.7%** Congregational and 1.7% Catholic, Yap **79.6%** Catholic and 4.6%
Congregational. Six of Yap's outer islands are 100% Catholic with no Congregational cell at all
(Eauripik, Elato, Faraulep, Ifalik, Lamotrek, Ngulu); four of Pohnpei's outer atolls are 100%
Congregational with no Catholic cell (Kapingamarangi, Mwoakilloa, Nukuoro, Pingelap). Inside
Pohnpei alone the Catholic share runs from **86.3% in Nett to 39.3% in Sokehs**, a spread the
four-state tier flattens to one number.

**AND THE SMALL DENOMINATIONS ARE ISLAND-SIZED, WHICH IS ONLY VISIBLE AT MUNICIPALITY.** Fais,
463 people in Yap's outer islands, is **29.4% Assembly of God and 28.5% Baptist** against a
national 0.7% and 1.2%; Sapwuahfik in Pohnpei is **20.3%** Adventist against 0.6%; Sokehs holds
282 of the country's 463 Apostolic. Nine categories out of eleven appear in fewer than half the
33 units, so most of this map is two colours and a handful of very local third ones.

EXCLUDED holds categories that are deliberately not on the tree. It is empty here.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {}

REVIEW = {
    "Congregation/Protestant":
        "-> christianity.protestant, the 'named no body' node, for 28,063 people and "
        "**37.1% of the country**. This is the biggest call in the file and it goes the same "
        "way as the two neighbours. **The body is known and is deliberately not asserted**: "
        "the Congregational churches of Micronesia descend from the American Board mission "
        "that reached Kosrae and Pohnpei in 1852 and Chuuk in 1879, and they are the direct "
        "siblings of the national churches this map draws separately for Samoa, Tuvalu, Niue, "
        "the Cook Islands and Kiribati. Three things say not to file them there. "
        "First, `mh1999.py` and `pw2005.py` send the SAME CHURCH to "
        "`christianity.protestant` in the Marshall Islands (54.8%) and Palau (23.2%), and "
        "sending Micronesia's to `christianity.reformed.congregational` would draw one "
        "mission's congregations in two different colours across three adjacent countries. "
        "Second, the census's string is `Congregation/Protestant`, a MERGED cell with a "
        "slash, and Table B6 has **no `other Christian` cell at all**, so every Protestant "
        "the other ten categories do not name is also inside it. "
        "Third, the Congregational reading is only safe for about half of it: Kosrae (4,515) "
        "and Pohnpei (8,894) are American Board ground, but Chuuk and Yap (14,654 between "
        "them) were handed to the German Liebenzell Mission around 1907, which is pietist "
        "and evangelical rather than Congregational, and **Chuuk is the one state with no "
        "municipality table**, so the half where the mix is live is a single polygon. "
        "A cell whose larger half cannot be resolved is not one to put a name on. "
        "The under-claim is real and is stated in `note_public` rather than hidden. If FSM "
        "ever prints the bodies apart, this is the row that changes.",
    "No religion/Refused":
        "-> unaffiliated, and **the 2010 census of the same country asked them apart, which "
        "is what settles it**. Table B09 of the 2010 FSM Basic Tables prints `No religion` "
        "723 and `Refused` 72, so 90.9% of a merged cell of this kind in Micronesia is "
        "people with no religion; in Yap, which carries most of it, the 2010 split is 281 "
        "against 12, or 95.9%. This is Palau's method (`pw2005.py` used the 1995 split of "
        "1,577 against 7) applied to the neighbour. "
        "**The reason it is worth the evidence is that the municipality tier makes this cell "
        "visible.** It is 0.7% of the country but 28.9% of Kanifay and 33.3% of Rumung, both "
        "in Yap, against 0.1% of Chuuk. A range that wide would ordinarily read as "
        "enumeration behaviour rather than belief; 2010 says otherwise, because it found the "
        "same shape (Yap 2.5% no-religion against Chuuk 0.05%) with the refusal cell taken "
        "out separately. So the concentration is Yapese and it is real. The residual 9% "
        "who refused are drawn as unaffiliated with them and cannot be separated.",
    "Apostolic":
        "-> christianity.pentecostal, the parent, following `ck2011.py`, which faces the "
        "identical bare cell in the Cook Islands. `Apostolic` names a body in some sources "
        "and a Oneness family in others, and the census says nothing more; the parent is the "
        "node that asserts neither. 463 people, and **282 of them are in Sokehs** on "
        "Pohnpei, which is 5.8% of that municipality against 0.6% of the country.",
    "Assembly of God":
        "-> christianity.pentecostal.trinitarian, named unambiguously, following `ck2011.py` "
        "and `mh1999.py`. Small nationally at 0.7% but **29.4% of Fais**, a single outer "
        "island of 463 people in Yap; the Assemblies of God are a real Micronesian feature "
        "rather than a coding artefact, and the Marshall Islands next door is 25.8%.",
    "Other religion":
        "-> other.fm rather than christianity.other. The column header says *religion* and "
        "not *church*, so reading it as a Christian tail would assert something the census "
        "does not (the Samoa and Vanuatu test). 1,167 people, 1.5%, and its geography is "
        "not flat: 4.4% of Yap against 0.9% of Pohnpei, and 12.3% of Fanif.",
    "Roman Catholic":
        "-> christianity.catholic.latin rather than the `christianity.catholic` parent. The "
        "category says Roman, the Caroline Islands are a Latin-rite territory and there is "
        "no Eastern Catholic presence for the node to misdescribe. Same call as `ki2015.py` "
        "for the same wording in the same region.",
}

COLUMNS = {
    # Every category is measured at the unit it is drawn on -- the municipality in Yap and
    # Pohnpei, the state in Chuuk and Kosrae -- so no row is derived and nothing rolls up.
    # Recorded here per COMMANDS.txt's check_rollup note rather than left blank by accident.
}

MAP = {
    "Roman Catholic":          "christianity.catholic.latin",
    "Congregation/Protestant": "christianity.protestant",
    "Assembly of God":         "christianity.pentecostal.trinitarian",
    "Pentecostal":             "christianity.pentecostal",
    "Apostolic":               "christianity.pentecostal",
    "Baptist":                 "christianity.baptist",
    "SDA":                     "christianity.adventist",
    "Mormon":                  "christianity.latterday",
    "Jehovah's Witness":       "christianity.witnesses",
    "Other religion":          "other.fm",
    "No religion/Refused":     "unaffiliated",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id. Micronesia excludes nothing, so never None."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
