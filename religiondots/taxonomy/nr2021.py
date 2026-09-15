"""
Nauru Bureau of Statistics, 2021 Population and Housing Census, Tables Vol 1, sheet G-7
(*Total population by religious affiliation and sex*) -> religiondots taxonomy.

**Nineteen categories, a flat partition of the whole enumerated population, summing to the
census's own 11,680 with a difference of zero.** One geography, which is the country; see
`sources/nr.py` for why that is complete rather than coarse at 11,680 people.

    Nauruan Congregational        4,001  34.26%  -> christianity.reformed.congregational.ncc
    Catholic                      3,959  33.90%  -> christianity.catholic.latin
    Assemblies of God (AOG)       1,365  11.69%  -> christianity.pentecostal.trinitarian
    Pacific Light House             706   6.04%  -> christianity.pentecostal.charismatic
    Nauru Independent               410   3.51%  -> christianity.nondenominational
    Shalosh Pentecostal Church      186   1.59%  -> christianity.pentecostal
    Baptist                         175   1.50%  -> christianity.baptist
    Seven Day Adventist             168   1.44%  -> christianity.adventist
    No Religion                     157   1.34%  -> unaffiliated
    Protestant                      126   1.08%  -> christianity.protestant
    Other religion                   98   0.84%  -> other.nr
    FOM Pentecostal Church           81   0.69%  -> christianity.pentecostal
    Do not wish to answer            57   0.49%  -> EXCLUDED
    Fishers of Men Church            57   0.49%  -> christianity.pentecostal
    Christ Embassy                   48   0.41%  -> christianity.pentecostal.charismatic
    Brethren Church                  47   0.40%  -> christianity.plymouth
    Methodist Church                 18   0.15%  -> christianity.methodist
    Fundamental Christian Church     15   0.13%  -> christianity.nondenominational
    Hinduism                          6   0.05%  -> hinduism

**NINE OF THE NINETEEN ROWS ARE WRITE-INS THE OFFICE CODED, AND THE QUESTIONNAIRE IS WHAT
SAYS SO.** Question 307 offers ten pre-coded answers — the first nine above other than Shalosh Pentecostal Church plus `Do not wish
to answer` and `Other religion` — and a free-text `307_oth` reached only from `Other religion`.
Protestant, Shalosh Pentecostal, Fishers of Men, Brethren, FOM Pentecostal, Christ Embassy,
Hinduism, Fundamental Christian and Methodist are all back-coded out of that box, and the 98
left in it are what the office did not code. Two things follow that matter for this mapping.
The nine are **verbatim self-descriptions**, so a name like `FOM Pentecostal Church` is the
respondent's own word and not an office classification; and **the tail is a coding tail, not a
sampling tail**, so `Other religion` at 0.84% is smaller than the same cell would be anywhere
that did not code its write-ins.

**CATHOLIC IS THE ROMAN CATHOLIC CHURCH AND THE REPORT SAYS SO.** G-7 prints the cell as bare
`Catholic`; the Analytical Report's table 25 prints the same 3,959 people as `Roman Catholic`.
Nauru is in the Latin-rite Diocese of Tarawa and Nauru.

**THE NAURU INDEPENDENT CHURCH LOST 57% OF ITS MEMBERS IN TEN YEARS AND PACIFIC LIGHT HOUSE
APPEARED.** 945 people in 2011 (9.5%) against 410 in 2021 (3.5%), while Pacific Light House
goes from no cell at all to 706 (6.0%). Those are the two largest movements in the table by
far; everything else is within a point of where it was. Nothing published states that the one
fed the other and this file does not claim it, but the arithmetic is close enough to be worth
the next person's attention.
"""

EXCLUDED = {
    "Do not wish to answer":
        "57 people, 0.49%, and the census's refusal cell rather than a religion. The office "
        "agrees: the Analytical Report's table 25 prints the same 57 people under the header "
        "`Not stated`, beside 109 for 2011. It is excluded here rather than dropped in "
        "`sources/nr.py` so that `tools/gap_share.py` can compute the hole from the "
        "normalized file, and it is the whole of Nauru's gap: every other one of the "
        "nineteen categories is a religion or the absence of one.",
}

REVIEW = {
    "Nauruan Congregational":
        "-> christianity.reformed.congregational.ncc, a node added for it. **4,001 people, "
        "34.3%, the largest body in the country.** The Nauru Congregational Church is the "
        "sixth of the Pacific Congregational set on this map and belongs with the other "
        "five rather than at the parent: **the Council for World Mission lists it (NCC) "
        "among its Pacific member churches**, in the same list as CCCS, CICC, EKT and the "
        "Kiribati Uniting Church, and CWM is the London Missionary Society's successor "
        "body. So the test that put `.cccs`, `.cicc`, `.ekt`, `.kpc` and `.niue` on nodes "
        "of their own is met by the same institution's own membership roll.\n\n"
        "**Its mission line runs through the Gilberts rather than from Samoa**, which makes "
        "it Kiribati's daughter rather than Samoa's: Protestant work on Nauru begins with a "
        "Gilbertese teacher in 1887 and the American Board sent Philip Delaporte in 1899. "
        "The 1887 landing is the one Nauru itself keeps — the centenary was celebrated in "
        "1987 — and the Gale encyclopedia's entry calls the arrivals London Missionary "
        "Society rather than American Board. The node is unaffected either way: `.kpc`'s "
        "note already records that the Kiribati church has both parents and that both are "
        "Congregational.\n\n"
        "**The US State Department's 2023 religious freedom report says the NCC `includes "
        "the Nauru Protestant Church`**, which is a live question for the separate "
        "`Protestant` cell below and is why that cell is not merged into this one here.",

    "Nauru Independent":
        "-> christianity.nondenominational, and this is the weakest call in the file. **410 "
        "people, 3.5%, the fifth-largest body and one of the five churches the government "
        "has actually registered** (US State Department 2023 report, alongside the "
        "Catholics, the Congregationalists, the Assemblies of God and the Adventists), so "
        "it is a real institution and not a write-in. **What nothing published says is what "
        "family it belongs to.** The two characterisations that exist disagree: Operation "
        "World calls it *the largest evangelical group* in Nauru, and the Gale "
        "encyclopedia's Nauru entry says *a breakaway Protestant church was formed in 1977 "
        "under the American Pentecostal church* without naming it. Neither is a founding "
        "history.\n\n"
        "It goes to `nondenominational` on the reading that `Independent` plus evangelical "
        "is the category Australia and New Zealand print as `Independent Evangelical "
        "Churches`, which both map here (`au2021.py`, `nz2023.py`). `christianity."
        "evangelical` was considered and rejected because that node holds an ANSWER a "
        "source collected, by its own definition, and `Nauru Independent` is a named church "
        "rather than the answer *evangelical*. **What would flip it**: a founding history "
        "showing it seceded from the Congregational church, which would put it under "
        "`christianity.reformed.congregational`, or one confirming the 1977 Pentecostal "
        "origin, which would put it under `christianity.pentecostal`. Neither was findable.",

    "Pacific Light House":
        "-> christianity.pentecostal.charismatic. **706 people, 6.0%, the fourth-largest "
        "body**, and it did not exist as a census cell in 2011. It is a founder-led local "
        "congregation in the born-again and revival stream rather than a branch of a "
        "classical Pentecostal denomination, which is where Samoa's Worship Centre, Voice "
        "of Christ and Peace Chapel sit (`ws2021.py`) and the Philippines' precedent "
        "before them. The only published description found calls it the **Pacific Light "
        "House Church (Born Again Christian Church)**, opened in Boe district on 5 "
        "September 2019; the Nauru Bureau of Statistics prints only the name. It is "
        "**not** among the churches the government has registered, which is consistent "
        "with a body younger than the 750-member registration rule can accommodate "
        "quickly, and not with it being a branch of the Assemblies of God, which is "
        "registered and printed separately at 1,365.",

    "Shalosh Pentecostal Church":
        "-> christianity.pentecostal, the family and no further. **186 people.** The "
        "respondent's own write-in says Pentecostal, so the family is not in doubt; what is "
        "in doubt is Trinitarian against Oneness, which is the deepest division inside "
        "Pentecostalism and the one thing the parent node exists to leave open. Nothing "
        "published identifies the body beyond church directories listing a `Church of God "
        "Shalosh Pentecostal` in Nauru, and `Church of God` names at least two unrelated "
        "families (`ki2015.py`). Same treatment as Tonga's `Other Pentecostal` and the "
        "Solomons' `Pentecostal` (`to2021.py`, `sb2019.py`).",

    "FOM Pentecostal Church":
        "-> christianity.pentecostal. **81 people**, and `FOM` is Fishers of Men: the "
        "office printed `Fishers of Men Church` as a separate 57-person row in the same "
        "table. **They are almost certainly one body written two ways** and are mapped to "
        "the same node, so the map is unaffected either way; they are not merged in "
        "`sources/nr.py` because that would edit the office's own partition. Filed at the "
        "family rather than at `trinitarian` for the same reason as Shalosh: the write-in "
        "says Pentecostal and nothing says which side of 1916.",

    "Fishers of Men Church":
        "-> christianity.pentecostal, with `FOM Pentecostal Church` above. **57 people.** "
        "Church directories list a `Fishers of Men Pentecostal` in Nauru, which is what "
        "ties the bare name to the family; the two rows together are 138 people, 1.2%.",

    "Christ Embassy":
        "-> christianity.pentecostal.charismatic. **48 people.** Christ Embassy is the "
        "public name of Believers' LoveWorld Inc., the Nigerian neo-charismatic church "
        "founded by Chris Oyakhilome in 1987, which plants branch congregations worldwide "
        "under exactly that name; no other body uses it. Neo-charismatic rather than "
        "classical Pentecostal, which is what `charismatic` is for. **Nauru counting it at "
        "all is the notable part**: 48 people is 0.4% of the country, and this is the only "
        "census on this map that prints a West African charismatic franchise as its own "
        "cell.",

    "Brethren Church":
        "-> christianity.plymouth. **47 people.** A bare `Brethren` cell is ambiguous "
        "between the Anabaptist Brethren family (Church of the Brethren, Brethren in "
        "Christ) and the Christian or Plymouth Brethren assemblies, and this map resolves "
        "it the same way in eight countries already: Antigua, Australia, Barbados, Bermuda, "
        "Jamaica, St Lucia, the UK and **Tuvalu**, whose `Brethren Assembly` is the nearest "
        "Pacific precedent there is. The Anabaptist Brethren are a North American body with "
        "no Pacific presence; the Open Brethren assemblies have one.",

    "Protestant":
        "-> christianity.protestant, the node that holds the ANSWER rather than a church. "
        "**126 people, 1.1%.** This is a write-in, so it is someone who wrote `Protestant` "
        "when the pre-coded list already offered them Nauruan Congregational, Assemblies of "
        "God, Nauru Independent, Pacific Light House, Seventh Day Adventist and Baptist by "
        "name.\n\n"
        "**It was NOT merged into the Congregational church, and that is a decision.** The "
        "US State Department's 2023 report describes the largest Christian group as *the "
        "Nauru Congregational Church (which includes the Nauru Protestant Church)*, which "
        "would make this cell part of the 4,001. Against that: the census prints the two "
        "apart; the figures are of a different order (4,001 against 126); and the State "
        "Department's sentence is about institutional structure, not about how 126 people "
        "answered a census question. Drawing an unspecified Protestant colour describes "
        "what they wrote. At 0.13 of a dot it changes nothing on the map either way.",

    "Fundamental Christian Church":
        "-> christianity.nondenominational. **15 people, 0.13%**, and a write-in. The name "
        "is a self-description in the independent fundamentalist idiom and names no family "
        "the tree has a branch for; `nondenominational` is where a church that claims no "
        "denomination goes. Draws no dot at 1 dot = 1,000 people.",

    "Other religion":
        "-> other.nr, the country's unclassified cell, and NOT `christianity.other`. **98 "
        "people, 0.84%.** Two reasons it stays out of Christianity. The header says "
        "*religion*, not *churches*, which is the wording test Samoa's `OTHER CHURCHES` "
        "passed and Vanuatu's `Other churches` failed (§9bk, §9bg). And Hinduism is printed "
        "as a row of its own at 6 people, so the cell is demonstrably not where the "
        "non-Christian religions were put; whatever is in it was too varied or too "
        "illegible for the office to code, having already coded nine other write-in "
        "bodies out of the same box.",
}

COLUMNS = {
    # Every category is measured at the country, which is the unit drawn, so no row is
    # derived and nothing rolls up. Recorded per COMMANDS.txt's check_rollup note.
}

MAP = {
    "No Religion":                  "unaffiliated",
    "Nauruan Congregational":       "christianity.reformed.congregational.ncc",
    "Catholic":                     "christianity.catholic.latin",
    "Assemblies of God (AOG)":      "christianity.pentecostal.trinitarian",
    "Nauru Independent":            "christianity.nondenominational",
    "Pacific Light House":          "christianity.pentecostal.charismatic",
    "Seven Day Adventist":          "christianity.adventist",
    "Baptist":                      "christianity.baptist",
    "Protestant":                   "christianity.protestant",
    "Shalosh Pentecostal Church":   "christianity.pentecostal",
    "Fishers of Men Church":        "christianity.pentecostal",
    "Brethren Church":              "christianity.plymouth",
    "FOM Pentecostal Church":       "christianity.pentecostal",
    "Christ Embassy":               "christianity.pentecostal.charismatic",
    "Hinduism":                     "hinduism",
    "Fundamental Christian Church": "christianity.nondenominational",
    "Methodist Church":             "christianity.methodist",
    "Other religion":               "other.nr",
}


def resolve(category):
    """Source category -> node, or None for the one excluded cell."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
