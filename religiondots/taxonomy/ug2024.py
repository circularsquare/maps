"""Uganda 2024 census religion (NPHC 2024, `HH_P9`, the 10% population sample) -> religiondots taxonomy.

Twelve answers, the questionnaire's codes 11-21 and 96, labels verbatim from the Stata file's
`HH_P9_VS1` (including its spelling `Budhist`). The Final Report's Table 3.1 prints ten; the file
also separates Bahai and Buddhist. sources/ug_2024.py composes them on 2,204 drawn subcounty units.

    household records in the sample, national
    37.40%  Roman Catholic                          -> christianity.catholic
    30.01%  Anglican / Church of Uganda             -> christianity.anglican
    14.71%  Pentecostal / Evangelicals (Born Again) -> christianity.evangelical
    13.68%  Islam                                   -> islam
     2.04%  Seventh Day Adventist                   -> christianity.adventist.sda
     1.54%  Others                                  -> other.ug
     0.19%  No Religion                             -> unaffiliated
     0.14%  Orthodox                                -> christianity.orthodox
     0.13%  Traditional                             -> indigenous.african
     0.11%  Jehovah's witness                       -> christianity.witnesses
     0.02%  Bahai                                   -> bahai
     0.02%  Budhist                                 -> buddhism

EXCLUDED holds the two population rows the normalized file carries per unit.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Total population":
        "the unit's whole census population, NPHC 2024 Subcounty Profiles workbook Table 1, "
        "not a category. It is the universe the gap is measured against.",
    "Not in a household":
        "Table 1 minus Table 2's household population: the people counted in institutions and "
        "other non-household quarters (boarding schools, barracks, prisons, hospitals, transit "
        "centres). Their records in the sample file have a blank religion (152,385 of 152,385), "
        "so they are not drawn. 1,517,205 people, 3.31% of the 146 districts; highest in "
        "Kisoro (24% of its sample records), Ntoroko, Kampala and Kalangala (16-17%) and "
        "Wakiso (13%). Which way it leans is unknown; the full census asked them.",
}

REVIEW = {
    "Pentecostal / Evangelicals (Born Again)":
        "-> christianity.evangelical, not christianity.pentecostal. The box names both, and "
        "Anita's ruling of 2026-09-14 keeps `Evangelical, unspecified` as one node because "
        "sources on both continents print evangelicals and Pentecostals in one cell; "
        "mz2017.py sends Mozambique's `Evangélica/Pentecostal` to the same node. In Uganda "
        "`Born Again` is mostly the Pentecostal and charismatic churches, so a Pentecostal "
        "reading would be close, but the census did not separate them and the 2002 build's "
        "`Pentecostal` cell was a different, narrower box. 14.7% of household records, the "
        "third-largest answer; its geography is Sebei and Bukedi (Bukwo, Kween) and the "
        "towns, and it is the answer that grew most since 2002 (4.6% then).",
    "Anglican / Church of Uganda":
        "-> christianity.anglican. The label names one church, the Church of Uganda, a "
        "province of the Anglican Communion; the form has separate boxes for the Pentecostals, "
        "the Adventists, the Orthodox and the Witnesses and puts other Protestants in `Others`, "
        "so this is not a Protestant family cell (the same reasoning as ug2002.py).",
    "Seventh Day Adventist":
        "-> christianity.adventist.sda, the child, because the label names the body.",
    "Roman Catholic":
        "-> christianity.catholic, the parent; Latin rite throughout and the census does not "
        "say so (the call ke2019.py, rw2022.py and ug2002.py make).",
    "Islam":
        "-> islam, the root. Overwhelmingly Sunni; the census asks no branch.",
    "Traditional":
        "-> indigenous.african. The form offers `Traditional` beside `No Religion`, so the "
        "no-religion box is not a lumped one (tools/check_no_religion.py's `separate`). 0.13% "
        "of household records, largest in Karamoja (Kaabong 2.7%, Moroto 1.7%). In 2002 the "
        "same Karamoja districts put about a quarter of their people in `Other` and a tenth in "
        "`None` on a card with no traditional box; in 2024 they answer Catholic.",
    "No Religion":
        "-> unaffiliated. One box, no separate atheist answer, so nothing to `secular`.",
    "Others":
        "-> other.ug. 1.54% of household records. Its peak is Kagadi (14.8%), Kibaale (12.7%) "
        "and Kyenjojo (11.4%), the area of the Faith of Unity (Ow'obushobozi), which UBOS names "
        "among the contents of this cell; elsewhere it holds the Protestant churches the form "
        "does not name (Baptist, Presbyterian, Methodist, the Salvation Army) and smaller "
        "movements. Drawn whole per spec §3.11.",
    "Orthodox":
        "-> christianity.orthodox. 0.14%. Drawn at its COUNTY's share inside each subcounty "
        "(the stability test's finest passing tier); 31% of the sample's Orthodox are "
        "refugees (HH_P27), largely in Adjumani, Obongi and the West Nile settlements, and "
        "Kampala has the largest settled community.",
    "Jehovah's witness":
        "-> christianity.witnesses. 0.11%, drawn at its county's share; 15% refugees.",
    "Bahai":
        "-> bahai. 0.02%, 975 sample persons; drawn at its DISTRICT's share, the finest tier "
        "it passes.",
    "Budhist":
        "-> buddhism. The file's spelling kept. 0.02%, 869 sample persons, drawn at its "
        "district's share.",
}

MAP = {
    "Roman Catholic": "christianity.catholic",
    "Anglican / Church of Uganda": "christianity.anglican",
    "Seventh Day Adventist": "christianity.adventist.sda",
    "Islam": "islam",
    "Pentecostal / Evangelicals (Born Again)": "christianity.evangelical",
    "Orthodox": "christianity.orthodox",
    "Bahai": "bahai",
    "Budhist": "buddhism",
    "Jehovah's witness": "christianity.witnesses",
    "Traditional": "indigenous.african",
    "No Religion": "unaffiliated",
    "Others": "other.ug",
}

# spec §7a-i-1: each node is the source's own column. Eight are measured at the drawn unit; the
# four drawn at a county or district share are `derived` and carry roll=NOWHERE in countries/ug.py.
COLUMNS = {v: v for v in MAP.values()}


def _key(cat):
    return " ".join(str(cat).split())


_FOLDED = {_key(k): v for k, v in MAP.items()}
_FOLDED_EXCLUDED = {_key(k) for k in EXCLUDED}


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = str(cat)
    if c in EXCLUDED or _key(c) in _FOLDED_EXCLUDED:
        return None
    if c in MAP:
        return MAP[c]
    return _FOLDED.get(_key(c))
