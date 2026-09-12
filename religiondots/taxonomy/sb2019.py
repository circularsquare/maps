"""
SINSO 2019 Census of the Solomon Islands, Vol 2 Table P8.3 -> religiondots taxonomy.

**Branch-level mapping, no nested universe and no leaves.** P8.3's eighteen columns are a flat
partition of the enumerated population at every one of 183 wards, and they sum to the printed
total exactly. The source's own string travels with the row in `source_category` (§2.4).

    Church of Melanesia          232,041  32.19%  -> christianity.anglican
    Roman Catholic               144,078  19.98%  -> christianity.catholic.latin
    South Sea Evangelical Church 124,506  17.27%  -> christianity.evangelical
    Seventh Day Adventist         83,452  11.58%  -> christianity.adventist
    United Church                 66,915   9.28%  -> christianity.united
    Christian Fellowship Church   16,179   2.24%  -> christianity.melanesianindependent.cfc
    Other religions               14,953   2.07%  -> other.sb
    Jehovah's Witness             14,624   2.03%  -> christianity.witnesses
    Christian OutReach Church      5,582   0.77%  -> christianity.pentecostal.charismatic
    Custom Beliefs or Animism      4,115   0.57%  -> indigenous.solomon
    Assembly of God                3,756   0.52%  -> christianity.pentecostal.trinitarian
    Bahai Faith                    3,104   0.43%  -> bahai
    Pentecostal                    3,019   0.42%  -> christianity.pentecostal
    Baptist Church                 2,172   0.30%  -> christianity.baptist
    No Religion or Atheism         1,227   0.17%  -> unaffiliated
    Muslim                         1,100   0.15%  -> islam
    -------------------------------------------------------------------------------------
    Religion Faith/Refuse to Answer  133   0.02%  §3.5 residual, EXCLUDED
    Total                        720,956          universe, EXCLUDED

**THE MISSIONS DIVIDED THE ISLANDS AND THE CENSUS STILL SHOWS THE LINE.** The Church of
Melanesia — the Anglican province of Melanesia — is **89.1% of Isabel, 85.1% of Temotu and
82.3% of Central**, against 2.0% of Choiseul and 4.9% of Western, which the Methodists took
and where the United Church is 53.5% and 38.8%. The Catholics hold Guadalcanal (36.2%). No
province is like its neighbour.

**THE SOUTH SEA EVANGELICAL CHURCH IS THE LABOUR TRADE COMING HOME.** 124,506 people, 17.3%,
and it exists because the Queensland Kanaka Mission evangelised Solomon Islanders working the
Queensland cane fields from 1886 and they took it back with them. **Malaita, which sent most
of those labourers, is 28.1% South Sea Evangelical; Isabel is 0.4%.** Nothing else on this map
records a church that was founded among migrant workers abroad and then became a national body
at home.

**AND THE CHRISTIAN FELLOWSHIP CHURCH IS ONE ISLAND GROUP.** Silas Eto's 1960 breakaway from
the Methodist mission on New Georgia: 16,179 people, of whom 13,629 are in Western Province,
and **77.2% of Kusaghe ward**.
"""

EXCLUDED = {
    "Total": "the ward's own enumerated population, not a religion category.",
    "Religion Faith/Refuse to Answer":
        "133 people, **0.02%**, and the smallest residual anywhere on this map. Vol 1's "
        "text calls them people who *\"refused to provide any information\"*. §3.5: marked, "
        "not filled. The Solomon Islands are 99.98% drawn.",
}

REVIEW = {
    "South Sea Evangelical Church":
        "-> christianity.evangelical, and this is the call worth a second opinion. 124,506 "
        "people, **17.3%, the third-largest body in the country**, and the tree has no "
        "family that contains it: the SSEC descends from the **Queensland Kanaka Mission** "
        "(Florence Young, 1886), an interdenominational faith mission, so it has no Baptist, "
        "Methodist, Reformed or Pentecostal parent, and it has been an autonomous Solomon "
        "Islands church since 1964. A named child was considered and is not possible — "
        "`christianity.evangelical` is deliberately not a parent of anything. **The "
        "precedent for putting it on the node itself is Kenya**, where KNBS's `Evangelical "
        "Churches` cell carries the Africa Inland Church, which is the same kind of body: "
        "faith-mission-descended, now autonomous, belonging to no confessional family. "
        "It is NOT filed under `christianity.melanesianindependent`, because that node is "
        "for churches founded outside the missions and this one came from one.",
    "Christian Fellowship Church":
        "-> christianity.melanesianindependent.cfc, a node added for it. 16,179 people. "
        "Silas Eto, the **Holy Mama**, broke with the Methodist mission on New Georgia in "
        "1960; the church kept a Methodist shape and added a devotion to Eto that the "
        "mission would not have. Not under `christianity.methodist`, because containment is "
        "a fact about people now (§2.1) and its members are not Methodists. **13,629 of the "
        "16,179 are in Western Province and it is 77.2% of Kusaghe ward.**",
    "Church of Melanesia":
        "-> christianity.anglican. 232,041 people, **32.2% and the largest body in the "
        "country**. This is the **Anglican Church of Melanesia**, a province of the Anglican "
        "Communion in its own right since 1975, out of the Melanesian Mission that worked "
        "from Norfolk Island from the 1850s. The census's own name is kept in "
        "`source_category` per §2.4. Its geography is the sharpest thing in this file: "
        "89.1% of Isabel and 85.1% of Temotu against 0.3% of nothing at all in Choiseul.",
    "United Church":
        "-> christianity.united. 66,915 people, 9.3%. The **United Church in Papua New "
        "Guinea and Solomon Islands**, formed in 1968 out of the Methodist mission in the "
        "west of the country and the LMS churches. It is 53.5% of Choiseul and 38.8% of "
        "Western and effectively absent everywhere east of them. `christianity.united` is "
        "for exactly this kind of 20th-century union and holds the same body in PNG if that "
        "country is ever drawn.",
    "Christian OutReach Church":
        "-> christianity.pentecostal.charismatic. 5,582 people, 0.77%. The **Christian "
        "Outreach Centre**, founded in Brisbane in 1974 and in the Solomons since 1988, now "
        "part of the International Network of Churches: mainline Pentecostal doctrine with a "
        "charismatic emphasis, and independent of the classical Pentecostal denominations, "
        "which is what `charismatic` is for as against `trinitarian`. An Australian import "
        "rather than a local foundation, so not under "
        "`christianity.melanesianindependent`.",
    "Custom Beliefs or Animism":
        "-> indigenous.solomon, a node added for it. 4,115 people, 0.57%, and a printed "
        "census category rather than an outsider's residual. Concentrated in the Kwaio "
        "interior of Malaita and the Weather Coast of Guadalcanal: `Waneagu/Taelanasina` "
        "21.5%, `Tetekanji` 21.2%. Read as a floor, for the reason `indigenous.african` "
        "carries.",
    "Other religions":
        "-> other.sb. 14,953 people, 2.1%. Unusually, this is a genuine tail rather than a "
        "place where the non-Christian religions were hidden: the same table names the "
        "**Baha'i Faith (3,104)** and **Muslims (1,100)** as their own cells, which most "
        "censuses on this map do not.",
    "Bahai Faith":
        "-> bahai. 3,104 people, 0.43%, and **a census that names them at ward level**. "
        "Very few do.",
    "Pentecostal":
        "-> christianity.pentecostal, the family node. 3,019 people, and it is the "
        "unaffiliated-Pentecostal remainder: this table already names Assembly of God and "
        "Christian OutReach separately, so what is left has no denomination recorded and "
        "belongs on the family (§14.4).",
}

MAP = {
    "Church of Melanesia": "christianity.anglican",
    "Roman Catholic": "christianity.catholic.latin",
    "South Sea Evangelical Church": "christianity.evangelical",
    "Seventh Day Adventist": "christianity.adventist",
    "United Church": "christianity.united",
    "Christian Fellowship Church": "christianity.melanesianindependent.cfc",
    "Christian OutReach Church": "christianity.pentecostal.charismatic",
    "Pentecostal": "christianity.pentecostal",
    "Jehovah's Witness": "christianity.witnesses",
    "Bahai Faith": "bahai",
    "Assembly of God": "christianity.pentecostal.trinitarian",
    "Muslim": "islam",
    "Baptist Church": "christianity.baptist",
    "Other religions": "other.sb",
    "Custom Beliefs or Animism": "indigenous.solomon",
    "No Religion or Atheism": "unaffiliated",
}

# No COLUMNS dict (spec §7a-i-1): every category is measured at the ward it is drawn on, so
# no row is `derived` and nothing ever needs to roll up.


def resolve(category):
    """religiondots branch for a SINSO category, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
