"""ZamStats 2022 census, Series B religion volume -> religiondots taxonomy.

Nine religions and twenty-five Christian rows, every one of them `derived` at the constituency
(sources/zm.py). **The deepest African census list on this map by Christian body**: the
Adventists, the New Apostolic Church, the United Church of Zambia, the Reformed Church in
Zambia, the Brethren of the CMML missions, the Evangelical Church in Zambia, the Brethren in
Christ, the Salvation Army and the Wesleyans are each their own row, where Zimbabwe and Malawi
pool them.

Shares below are of the de facto 18,340,343 the religion tables count, and a province share is
of that province's own de facto total in Table B.1.

TWO B.1 COLUMNS ARE FILED BY THE OFFICE'S ANALYTICAL REPORT AND NOT BY THEIR PRINTED HEADERS.
See sources/zm.py's docstring for the evidence; `check_relabel()` asserts it on every build.

EXCLUDED holds categories that are deliberately not on the tree (none: zm.csv carries only
drawn rows, and the two measured parent columns are named in each row's note).
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {}

REVIEW = {
    # ---- the relabel
    "Judaism":
        "-> indigenous.african, NOT judaism. 30,502 people, 0.17%. Table B.1 heads this column "
        "`Judaism`; ZamStats' own 2022 National Analytical Report (Aug 2025, Figure 3.13) has "
        "no Judaism and gives African Traditional religion as 0.2%, which only this column can "
        "make, and says it is 0.3% rural and 0.1% urban. Its geography is the traditional "
        "religion's and not a Jewish community's: **15,625, half of it, in Eastern Province** "
        "(0.67%), the Chewa and Ngoni country, against 1,037 on the Copperbelt. Zimbabwe's "
        "`Judaism` cell (zw2022.py) was left at judaism because nothing contradicted its label; "
        "here the office's second publication does. A floor for the reason zw2022.py and "
        "mw2018.py give: the answer is exclusive of the churches.",
    "Other Religious Groups (B.1 column)":
        "-> unaffiliated, NOT other.zm. 233,260 people, 1.27%. The analytical report's key "
        "findings say *1.3 percent reported no religious affiliation*, and only this column is "
        "1.3%. Its Figure 3.14 puts that answer at **1.8% of men and 0.8% of women**, the "
        "male skew irreligion has here (2010 census, Lusaka Province: `None` 26,603 men to "
        "8,676 women, `Other` 23,182 to 24,077). Highest in North-Western 3.13% and Luapula "
        "2.55%, lowest in Muchinga 0.31%. One cell, so nothing goes to `secular` (ke2019.py, "
        "mw2018.py, zw2022.py make the same call).",
    "Non-Religious":
        "-> other.zm, NOT unaffiliated. 9,238 people, 0.05%. Under the analytical report's "
        "reading this column is part of its `Other` 0.1% (the six small columns sum to 21,401, "
        "0.12%), so its printed header cannot be taken at its word once `Other Religious "
        "Groups` is the no-religion answer. What it actually holds is not recoverable from "
        "anything published.",
    "African Traditional Religion":
        "-> other.zm, NOT indigenous.african. 463 people, 0.003%, which no reading makes the "
        "traditional religion: the report's 0.2% is 37,000 people. Filed with `Non-Religious` "
        "in the residual for the same reason. If the two swapped headers are a symmetric "
        "mislabel this is the Judaism column; nothing published says so.",
    # ---- the other religions, as printed
    "Islam":
        "-> islam, no branch (the report says Sunni and Shia were asked and B.1 does not split "
        "them). 88,803 people, 0.48%. Eastern 1.27% and Lusaka 1.05%, and 241 people in all "
        "of Western Province.",
    "Hinduism":
        "-> hinduism. 2,106 people, 428 in Lusaka Province. Low for Zambia's Indian community, "
        "and B.1's headers are not reliable, so read it as what the column says and no more.",
    "Buddhism":
        "-> buddhism. 7,531 people, 54% in Lusaka Province.",
    "Bahai Faith":
        "-> bahai. 910 people.",
    "Sikhism":
        "-> sikhism. 1,153 people, 433 of them in North-Western.",
    # ---- Christian rows that are an answer and not a body
    "Christianity Denomination: None":
        "-> christianity, the `other or unspecified` row (ie2022's `Christian (Not Specified)`, "
        "nz2023's `Christian nfd`). 335,271 people, 1.83%. Christians who named no "
        "denomination, and **68.4% of them are men** (Table B.12: 229,305 to 105,966) where "
        "Christians overall are 48.3% men. That is the no-religion answer's profile, and it "
        "ranks the provinces the same way: North-Western first on both (3.88% here, 3.13% "
        "no religion), Luapula second. So some of this row is very likely people with no "
        "religion who were recorded as Christian. Filed as printed, per §2.7; the note says it.",
    "Christianity Denomination: Other":
        "-> christianity, not christianity.other. 516,861 people, 2.82%. A residual, and "
        "branches.py's christianity.other is explicitly not one (the reason ca2021.COLUMNS and "
        "hu2022.COLUMNS give). zw2022.py and mw2018.py filed their `Other Christian` at "
        "christianity.other before that was written down. **Eastern Province is 8.22%** and "
        "Muchinga 6.86%, so it carries something regional the list does not name.",
    "Episcopal":
        "-> christianity. 12,787 people, 0.07%, spread evenly over all ten provinces. B.6 lists "
        "it apart from `Anglican`, so it is not the Anglican Church in Zambia, and it could be "
        "the African Methodist Episcopal Church or another episcopal body; the census does not "
        "say, so no branch is inferred.",
    "Restoration":
        "-> christianity. 9,142 people, 0.05%. Listed apart from `Church of Christ`, so not the "
        "Stone-Campbell churches that row already holds; which body it is, the census does not "
        "say.",
    # ---- named bodies filed at their family
    "Catholic":
        "-> christianity.catholic, the parent (ke2019.py, mw2018.py, zw2022.py). 3,220,214, "
        "17.56%. **Northern Province 34.0%**, Eastern 25.4%, Luapula 22.9%; Southern 7.8%.",
    "Seventh Day Adventist (SDA)":
        "-> christianity.adventist.sda, the node for the body (§2.7; ca2021.py). 3,183,685, "
        "17.36%, level with the Catholics. **Southern Province 42.1%** and Central 29.7%, "
        "Eastern 3.3%. 20.3% of rural Christians and 14.4% of urban ones.",
    "Pentecostal":
        "-> christianity.pentecostal. 3,142,575, 17.13%. The urban church: 24.7% of urban "
        "Christians and 11.7% of rural ones; Lusaka Province 26.8%, Copperbelt 24.2%.",
    "United Church Of Zambia (UCZ)":
        "-> christianity.united (jm2011, sb2019, au2021). 1,727,783, 9.42%. The 1965 union of "
        "the London Missionary Society and Church of Scotland congregations, the Methodists and "
        "the Paris Mission's Barotseland church. Muchinga 26.4%, Northern 19.6%; North-Western "
        "2.6%.",
    "New Apostolic":
        "-> christianity.newapostolic (ao2024, st2012). 1,513,225, 8.25%. **Western Province "
        "38.0%**, the largest single-body share in any Zambian province; Southern 12.3%, "
        "North-Western 11.7%, Northern 1.8%. Angola's Moxico border figure (ao2024.py) is the "
        "other side of the same field.",
    "Jehovah's Witness (Watchtower)":
        "-> christianity.witnesses. 1,009,578, 5.50%. Central 9.4%, Copperbelt 7.6%; Southern "
        "1.5%.",
    "Baptist":
        "-> christianity.baptist. 651,111, 3.55%. Copperbelt 7.4%, Eastern 6.5%; Luapula 0.75%.",
    "Reformed Church In Zambia (RCZ/Dutch)":
        "-> christianity.reformed.continental (za2016's `Reformed church`). 464,053, 2.53%. The "
        "Dutch Reformed Church's Eastern Province mission, and **68.5% of it is still in "
        "Eastern** (13.5% of the province). Malawi's Nkhoma synod (mw2018.py) is the same "
        "mission across the border, filed there inside CCAP.",
    "Christian Missions in Many Lands (CMML)":
        "-> christianity.plymouth. 459,852, 2.51%. The Open Brethren missionary society "
        "(Garenganze, F. S. Arnot, 1886). **Luapula 14.9%**, holding 46.2% of the body.",
    "Evangelical Church in Zambia":
        "-> christianity.evangelical, on sb2019.py's and ke2019.py's precedent for a church "
        "descended from an interdenominational faith mission with no confessional family "
        "(the South Africa General Mission). 349,827, 1.91%. **North-Western 13.9%** and "
        "Western 7.2%, 74.3% of the church between them.",
    "Apostolic Faith Mission":
        "-> christianity.pentecostal. 325,553, 1.78%. The classical Pentecostal body founded in "
        "Johannesburg in 1908; a named church with no node of its own, filed at its family "
        "beside the `Pentecostal` answer. Western 3.4%, Southern 3.1%.",
    "Church of Christ":
        "-> christianity.restorationist (fj2007, ph2020). 296,697, 1.62%. Southern 5.2%, "
        "North-Western 3.4%.",
    "Anglican":
        "-> christianity.anglican. 237,597, 1.30%. Eastern 3.3%, holding a third of the body.",
    "Methodist":
        "-> christianity.methodist. 134,382, 0.73%. North-Western 2.4%, Luapula 2.2%. Most of "
        "Zambia's Methodist inheritance is inside the UCZ, so this is what stayed out.",
    "Brethren in Christ":
        "-> christianity.anabaptist.brethren, which branches.py names it under. 113,070, "
        "0.62%, 48.7% in Southern Province.",
    "Salvation Army":
        "-> christianity.holiness (au2021, ph2020). 100,479, 0.55%, 59.9% in Southern.",
    "Presbyterian":
        "-> christianity.reformed.presbyterian. 72,056, 0.39%, 68.2% in Eastern, the CCAP "
        "side of the Malawi border.",
    "Wesleyan":
        "-> christianity.holiness.wesleyan (bb2010, ca2021). 54,230, 0.30%, **87.5% in "
        "Southern Province**: the Wesleyan Church's Choma mission.",
    "Lutheran":
        "-> christianity.lutheran. 23,864.",
    "Orthodox":
        "-> christianity.orthodox, the parent; the census does not say which church. 7,218.",
    "Latter-Day Saints":
        "-> christianity.latterday. 5,267.",
}

MAP = {
    # religions (Table B.1), filed by the analytical report's reading where it differs
    "Islam": "islam",
    "Judaism": "indigenous.african",
    "Hinduism": "hinduism",
    "Buddhism": "buddhism",
    "Bahai Faith": "bahai",
    "Sikhism": "sikhism",
    "African Traditional Religion": "other.zm",
    "Non-Religious": "other.zm",
    "Other Religious Groups (B.1 column)": "unaffiliated",
    # Christian rows (Tables B.6, B.9, B.10)
    "Anglican": "christianity.anglican",
    "Apostolic Faith Mission": "christianity.pentecostal",
    "Baptist": "christianity.baptist",
    "Brethren in Christ": "christianity.anabaptist.brethren",
    "Catholic": "christianity.catholic",
    "Christian Missions in Many Lands (CMML)": "christianity.plymouth",
    "Church of Christ": "christianity.restorationist",
    "Episcopal": "christianity",
    "Evangelical Church in Zambia": "christianity.evangelical",
    "Jehovah's Witness (Watchtower)": "christianity.witnesses",
    "Latter-Day Saints": "christianity.latterday",
    "Lutheran": "christianity.lutheran",
    "Methodist": "christianity.methodist",
    "New Apostolic": "christianity.newapostolic",
    "Orthodox": "christianity.orthodox",
    "Pentecostal": "christianity.pentecostal",
    "Presbyterian": "christianity.reformed.presbyterian",
    "Reformed Church In Zambia (RCZ/Dutch)": "christianity.reformed.continental",
    "Restoration": "christianity",
    "Salvation Army": "christianity.holiness",
    "Seventh Day Adventist (SDA)": "christianity.adventist.sda",
    "Wesleyan": "christianity.holiness.wesleyan",
    "United Church Of Zambia (UCZ)": "christianity.united",
    "Christianity Denomination: None": "christianity",
    "Christianity Denomination: Other": "christianity",
}

# spec §7a-i-1: the level this source COUNTED each derived row at. Every Christian row came out
# of Table B.5's constituency `Christianity`, so `inferred dots: not shown` redraws that.
#
# THE NON-CHRISTIAN PARENT DELIBERATELY GETS NO COLUMN. B.5's other measured column is every
# non-Christian at once (Islam, no religion, the traditional religion and the rest together),
# and no node means that; rolling a Muslim dot up to `other.zm` or to a root would draw a
# figure nobody counted. With none, those 373,966 disappear under the toggle, which is
# ao2024.py's honest behaviour, and check_rollup lists them as still gone.
COLUMNS = {
    "Christianity": "christianity",
}


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
