"""
American Samoa 2015 Household Income and Expenditure Survey, Table 1.6 -> religiondots taxonomy.

**Sixteen rows, a flat partition of the weighted household population at every county.** The
survey took a write-in religion for every member of 1,838 sampled households (9,578 people, one
flat weight of 5.99668) and the Department of Commerce coded the answers into these rows. The
source's own string travels in `source_category` (§2.4); `Baha’i` keeps the report's curly
apostrophe.

    CCCAS              19,147  33.34%  -> christianity.reformed.congregational.cccas
    Catholic           10,410  18.12%  -> christianity.catholic.latin
    LDS _ Mormons       9,091  15.83%  -> christianity.latterday
    Assembly of God     5,451   9.49%  -> christianity.pentecostal.trinitarian
    Methodist           4,342   7.56%  -> christianity.methodist
    Other religion      3,178   5.53%  -> other.as
    SDA                 1,661   2.89%  -> christianity.adventist
    Baptist               774   1.35%  -> christianity.baptist
    Full Gospel           762   1.33%  -> christianity.pentecostal.trinitarian
    Jehovah's Witness     714   1.24%  -> christianity.witnesses
    No religion           702   1.22%  -> unaffiliated
    Pentecostal           558   0.97%  -> christianity.pentecostal
    Baha’i                294   0.51%  -> bahai
    Nazarene              246   0.43%  -> christianity.holiness
    Jewish                 84   0.15%  -> judaism
    Orthodox               24   0.04%  -> christianity.orthodox

**NOTHING IS EXCLUDED.** Table 4.3 of the same report prints a not-reported row (`NR`) of 0 for
both sexes, so there is no non-response to leave out, and the census counts the dots are laid on
include everyone.

**OWN_GEOGRAPHY is the verdict of `sources/as.py::geography_test`**, which asserts it matches this
list on every run. There is no microdata (the SPC catalogue entry answered 403 to the scout) and
only one round, so the split-half of §9bi/§9bl cannot be run. The test used instead: under the
null that a category is spread evenly, simulate each county's sampled households drawing it at
the territory rate, and compare the 2 x 10 chi-square on households with the observed one, where
a household counts once because a household mostly shares one church. 20,000 draws, a fixed seed,
the house 95% bar per category (ask 007) and no multiplicity correction (ask 012 left that open).
A category that fails shares each county's remainder at its territory-wide proportion
(`countries/as.py::_as_counts`), as Puerto Rico's and Taiwan's failing categories do.
"""

EXCLUDED = {}

# Categories drawn on their own county shares. Everything else in MAP shares each county's
# remainder at the territory-wide proportions. Pinned; sources/as.py re-runs the test and stops if
# its verdict differs.
OWN_GEOGRAPHY = (
    "CCCAS",
    "Catholic",
    "Methodist",
    "SDA",
    "LDS _ Mormons",
    "Jehovah's Witness",
    "Nazarene",
    "Other religion",
)

REVIEW = {
    "CCCAS":
        "-> christianity.reformed.congregational.cccas, a node added for it. 19,147 weighted "
        "people, 33.3%, the largest body in every county but Maoputasi (Catholic 25.9% against "
        "25.5%) and 88.3% of Manu'a. The Congregational Christian Church in American Samoa "
        "(Ekalesia Fa'apotopotoga Kerisiano i Amerika Samoa) is not Samoa's CCCS: the World "
        "Council of Churches' member page says an independent assembly for American Samoa was "
        "sought from 1964 and constituted in 1980, and the two churches declared reconciliation "
        "in 1982. It is a WCC member in its own right. So it goes beside `.cccs`, `.cicc`, "
        "`.ekt`, `.kpc`, `.ncc` and `.niue` as the seventh national church of the Pacific "
        "Congregational set rather than onto `.cccs`, which would merge two churches the "
        "churches themselves keep apart.",
    "Other religion":
        "-> other.as. 3,178 weighted people, 5.5%. The write-ins the office did not code into "
        "fifteen named rows; the report does not list them. Unlike Samoa's `OTHER CHURCHES` "
        "(on `christianity.other`), the row says religion rather than church, so it is not "
        "assumed Christian. Table 3.3 puts 1,943 of them born in American Samoa, 588 in Samoa "
        "and 138 in Asia, so it is not mainly a migrants' answer. 11.6% of Maoputasi and 6.8% "
        "of Tualauta, none recorded in Saole, Vaifanua or Manu'a; it passes the county test "
        "and is drawn on its own shares.",
    "Orthodox":
        "-> christianity.orthodox (Eastern Orthodox). 24 weighted people, four sampled persons. "
        "The report does not say which Orthodox church; Eastern is the default reading of a bare "
        "'Orthodox' write-in, and at four people nothing on the map turns on it.",
    "Full Gospel":
        "-> christianity.pentecostal.trinitarian, with Assembly of God, following Samoa's "
        "`FIRST FULL GOSPEL PENTECOSTAL CHURCH IN SAMOA` (`ws2021.py`). The report prints only "
        "`Full Gospel`.",
    "Pentecostal":
        "-> christianity.pentecostal, the parent. A write-in that named no church; kept apart "
        "from Assembly of God and Full Gospel because the office kept it apart.",
    "Jewish":
        "-> judaism. 84 weighted people, fourteen sampled persons in six counties. Kept as the "
        "source prints it; drawn at the territory-wide rate.",
    "Nazarene":
        "-> christianity.holiness. 246 weighted people, 41 sampled persons, 23 of them in Ituau "
        "and 18 in Tualauta. It passes the county test (p about 0.03) on what is perhaps eight "
        "households, which is what a congregation looks like in a one-in-six sample; drawn on "
        "its own shares because that is the test as written, and it moves at most a fraction "
        "of a dot.",
    "LDS _ Mormons":
        "-> christianity.latterday. The label is the report's own, underscore included. 25.0% "
        "of Tualauta against 8.4% to 17.9% elsewhere; Tualauta holds 1,397 of the survey's 1,607 "
        "Tongans (Table 1.5), and Tonga is the most Latter-day Saint country the UN has a figure "
        "for.",
}

COLUMNS = {
    # Every row is `modelled` (survey shares on census counts), which never rolls up, so nothing
    # needs a source column here.
}

MAP = {
    "CCCAS":             "christianity.reformed.congregational.cccas",
    "Catholic":          "christianity.catholic.latin",
    "Methodist":         "christianity.methodist",
    "SDA":               "christianity.adventist",
    "LDS _ Mormons":     "christianity.latterday",
    "Assembly of God":   "christianity.pentecostal.trinitarian",
    "Baha’i":       "bahai",
    "Full Gospel":       "christianity.pentecostal.trinitarian",
    "Jehovah's Witness": "christianity.witnesses",
    "Orthodox":          "christianity.orthodox",
    "Jewish":            "judaism",
    "Pentecostal":       "christianity.pentecostal",
    "Nazarene":          "christianity.holiness",
    "Baptist":           "christianity.baptist",
    "Other religion":    "other.as",
    "No religion":       "unaffiliated",
}

assert set(OWN_GEOGRAPHY) <= set(MAP), "OWN_GEOGRAPHY names a category MAP does not"


def resolve(category):
    """Source category -> node. Nothing is excluded, so this is None only for an unknown label."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
