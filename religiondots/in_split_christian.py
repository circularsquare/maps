"""India: split the 2011 census `Christian` column by church wherever a source reaches: Kerala, Mizoram, and
Pew's 2021 East and South regions.

Writes data/normalized/in_split_christian.csv: every sub-district's `Christian` count, replaced by the
church rows a second source names plus the census's own `Christian` for the rest, summing to the census
figure to the person. countries/in.py::_in_counts swaps these in for the allocated file's `Christian`
rows, exactly as in_split.py's file replaces `Muslim`.

WHY THIS EXISTS
---------------
India draws 27.8M Christians and until 2026-09-15 every one sat on the bare `christianity` node: the
census asks religion, never church, and its Annexure's 8,399 Catholics are write-ins (sources/in.md §4).
Anita asked for a Christian breakdown by name (queue.md, "D. Splits a drawn country is missing"). The
scout (sources.md §scout-2026-09-15-india-christians) found two states reachable from public sources and
all of India reachable only through Pew's microdata, which Anita then downloaded (ask 032).

The construction is spec §2.7a's, the Muslim split's: a named share on the church node, everything the
source does not name on the parent, `derived` because the census counted these Christians at the
sub-district and only the church comes from elsewhere.

KERALA: THE CATHOLICS ONLY, AND WHY NOT THE TEN CHURCHES THE TABLE PRINTS
--------------------------------------------------------------------------
Source: K.C. Zachariah, *Religious Denominations of Kerala*, CDS Working Paper 468, April 2016
(`https://cds.edu/wp-content/uploads/WP468.pdf`), p. 17 of the PDF (printed pp. 32-33), Table 6,
"Percent Distribution of District Population by Christians Denominations, 2008-2014", compiled from
the Kerala Migration Surveys (KMS), CDS's household survey of about 15,000 households a round. Ten
answers per district: Syro-Malabar, Syro-Malankara, Latin Catholics, Jacobite, Orthodox, Mar Thoma,
CSI, Dalit Christian, Pentecost (with Church of God and Brethren), Others.

The same paper's Muslim table puts Pathanamthitta's Muslims at 60.9% Shia, which is not credible, so
the Christian codes were checked before any of them was drawn. Three checks, all run by `build()`:

1. THE TABLE IS INTERNALLY CONSISTENT, and that shows what its Kerala row is. Table 5 prints the same
   cells as column shares (each church across districts) with a `Total` column. Table 6 x that Total
   column reproduces Table 5 to rounding. But Table 5's Total column is NOT the census distribution of
   Kerala's Christians: Thrissur is 4.9% of it against 12.3% of the census, Kozhikode 6.0 against 2.1,
   Wayanad 8.1 against 2.8. So the Total column is the sample's own spread, the table is unweighted, and
   Table 6's `KERALA` row (and Table 2's state totals, which are that row times 6,141,269) weights each
   district by how many Christians happened to be sampled there. That is why the scout found Table 6's
   Kerala row equal to Table 2's shares, and why this file applies each DISTRICT row to that district's
   census Christians and never uses the Kerala row.

2. THE THREE CATHOLIC RITES DISAGREE WITH THE DIOCESES' OWN ROLLS, AND THEIR SUM DOES NOT. Witness:
   catholic-hierarchy.org's all-diocese table for India (`country/scin1.html`, Annuario Pontificio 2005,
   2004 data), the 26 jurisdictions seated in Kerala, each put in the district of its see city (SEAT).
   A roll runs above self-identification by a margin, so the test is whether roll / KMS is steady from
   place to place, not whether it is 1. It is steady for Catholics as a whole and wildly unsteady for the
   rites:

     Kannur + Kasaragod, whose Syro-Malabar and Latin sees (Tellicherry, Kannur) both cover exactly
     these two districts: rolls say 279,200 Syro-Malabar and 32,540 Latin; Table 6 on the census says
     about 126,000 and 120,000. A roll margin cannot make one rite 2.2 times its survey figure and the
     other 0.27 times in the same two districts.
     Ernakulam: the only Syro-Malankara see seated there, Muvattupuzha, rolls 11,067; Table 6 gives
     the district about 106,000 Syro-Malankara Catholics.

   Catholics summed over the three rites run 0.7 to 1.7 times the survey across six groups of seat
   districts, and that range is mostly see territory crossing group lines (Changanacherry, seated in
   Kottayam, covers Thiruvananthapuram and Kollam; Palakkad's Latin Catholics sit under Coimbatore, a
   Tamil Nadu see). The rites run from 0.08 to over 4 in the same groups (spread: Syro-Malankara 51,
   Latin 9.2). Syro-Malabar on its own spreads only 2.2, because it is two-thirds of the total; but the
   rites are a partition, so a respondent filed under the wrong rite is wrong in two of them, and the
   three pass or fail together. The reading: KMS respondents or
   coders swap the rites (Syro-Malabar and Syro-Malankara differ by two letters, and four of Kerala's
   churches carry "Malankara" in their formal names), and the swaps stay inside the Catholic total.

   CATHOLIC_SPREAD_BAR = 3.0 is the guard that fails the build if a re-read changes this. It was set on
   2026-09-15 after a first rough look at the numbers, which is post hoc; the gap between the Catholic
   total (about 2.3) and the rites (tens) is wide enough that any bar from 3 to 20 decides the same way.

3. NOTHING INDEPENDENT REACHES THE NON-CATHOLIC CHURCHES. No roll by diocese was found for the Jacobite,
   Orthodox, Mar Thoma or CSI churches (the scout found CSI diocese figures only on Wikipedia, in mixed
   years), and three of the four carry the same "Malankara" name that the rite check shows being confused.
   Unchecked is not passed, so they stay on `christianity` with Dalit Christians (a caste, not a church),
   Pentecost/Church of God/Brethren and Others.

So Kerala draws ONE named share per district: Syro-Malabar + Syro-Malankara + Latin, on
`christianity.catholic`. It is a floor: Catholics who answered `Dalit Christian` or `Others` stay on the
parent. Known cost, spec §3.10's: the share is uniform inside a district, and the district cells in
Malabar rest on few sampled localities (point 1), which the six-group check cannot see inside.

MIZORAM: THE STATE'S CHURCH ROLLS, ONE SHARE FOR THE WHOLE STATE
-----------------------------------------------------------------
Source: Directorate of Economics and Statistics, Government of Mizoram, Mizoram Statistical Database,
NGO > churches (`http://crsmizo.mizoram.gov.in/ngo/index.php?page=ngo_<body>_year`, POST
district=State, from_year=2000, to_year=2023), members by year for each church body. `--fetch` saves
the ten pages that carry members under data/raw/in/mizoram_churches/.

YEAR = 2010-11. The census's reference date, 1 March 2011, falls in the financial year 2010-11, so that
column is the one read. The scout's figures were 2011-12, which sum to 106.2% of the census's 956,331
Christians; 2010-11 sums to 100.5%. Most of that difference is the Salvation Army's step from 36,395 to
55,791 between the two years, an unexplained level shift in its series. Both years are printed by
`build()`; no body's share moves by more than 1.8 points between them.

Each body's share of the ten rolls' sum divides each sub-district's census Christians: a roll splitting
a self-identification column, which spec §3.1 allows. The rolls sum close to the census, so scaling moves
no body by more than half a percent of itself. Isua Krista Kohhran (1.2%) is not placed on any family,
because nothing read here says which it belongs to, so it joins the census `Christian` remainder.

State only. The district-wise form returns empty cells (scout, 2011, 2016, 2020), so ONE SHARE COVERS
ALL 29 SUB-DISTRICTS, and that is the known cost here: the Evangelical Church of Maraland is the church
of the Mara in Saiha district and the Lairam Isua Krista Baptist Kohhran that of the Lai in Lawngtlai,
and both are spread across Aizawl like everything else.

THE REST OF INDIA: PEW'S 2021 RESPONDENT FILE, THREE CHURCHES, EAST AND SOUTH (added 2026-09-15)
----------------------------------------------------------------------------------------------
Source: Pew Research Center, *India Survey Dataset* (Neha Sahgal and Jonathan Evans, 2021,
doi:10.58094/rfte-a185), the respondent file behind *Religion in India: Tolerance and Segregation*
(29 June 2021). Downloaded by Anita with her Pew account on 2026-09-15 (ask 032) and copied to
data/raw/in/pew_india_2021.dta; it cannot be fetched here. `ICPSR_38489-V1.zip` beside it is NOT this
study: ICPSR 38489 is the East Asian Social Survey 2018 (its manifest and codebook say so), so its terms
do not govern this file. Pew's own Terms of Use do, §13 and §14, read 2026-09-15: a licence to publish
derivatives, publication of the data limited to excerpts, attribution to the Center, no attempt to
identify respondents, and a fixed disclaimer that "you must include ... with your use of any Data",
which is why countries/in.py's note carries it word for word.

1. THE PUBLIC FILE RECODES THE CARD. The topline (p. 23) prints sixteen QDENOM answers for India; the
   file's `qdenomrec` keeps seven: Catholic, Baptist, Presbyterian, no denomination, all other
   denominations, don't know, refused. Church of North India (7%), Church of South India (7%), Orthodox,
   Pentecostal, Lutheran, Methodist, Adventist and the rest are inside `all other` (30%) and cannot be
   drawn. check_pew() re-reads the topline's India row off the PDF and requires the weighted file to
   reproduce each cell within rounding.

2. REGION, AND NOTHING FINER. `region` is Pew's six zonal-council regions (in_split.py's STATES). The file
   has no state, no sampling point and no usable stand-in: Q85AREC says only whether the respondent was
   raised in the current state, qmlangrec only whether the interview was in Hindi, and qrid is a
   respondent number. So one share per region (spec §3.10's cost), and Kerala's and Mizoram's respondents
   cannot be taken out of the South and Northeast pools that are applied to their neighbours.

3. A REGION IS DRAWN ONLY WITH 100 CHRISTIAN RESPONDENTS OR MORE. Christians per region: South 472,
   Northeast 326, East 123, West 56, North 24, Central 10. Pew drew 138 sampling clusters with about twelve
   interviews to a village (report p. 224), so a cell of ten or twenty-four Christians can be one or two
   villages, and the file carries no cluster id to check that (stability.py's CELL_CAP cannot be run). Pew
   printed QDENOM for India only, while it printed QSECT by region down to Central's 202 Muslims.
   MIN_RESPONDENTS = 100 is Italy's `N_FLOOR` (sources/it.py, sources/it.md), where 100 respondents put
   the standard error on a Catholic share near 4.3 points. It was chosen after the answers were
   tabulated, so it is not blind; it is a number borrowed, not one fitted here. It is a minimum and not a
   guarantee: under Pew's median design effect for Christians (3.7) the East's 123 are worth about 33.
   Italy's thin units take their parent's share instead. Here the only parent is all of India, whose
   Baptists and Presbyterians Pew found only in the Northeast and South and whose Catholic share (37%) is
   below all three thin regions' readings (52 to 62%), so West, North and Central stay on `christianity`,
   with their shares printed so a later session can argue for them.

   THE NORTHEAST IS NOT DRAWN EITHER, though it has 326 respondents (NOT_DRAWN; withdrawn 2026-09-15 on
   the review in sources/in.md §11, recorded in §12). Its churches follow the tribe and so the state, and
   one regional share fits none of them. Check (d) below, the Catholic rolls by state, reads roll / drawn
   2.21 in Assam with Arunachal Pradesh and 0.18 in Nagaland, a spread of 12 against CATHOLIC_SPREAD_BAR;
   the same share put about 553,000 Presbyterians in Nagaland, whose churches are Baptist associations.
   Mizoram's respondents are also inside the share, while Mizoram itself draws from its own rolls. So the
   Northeast's Christians outside Mizoram stay on `christianity` until a source gives churches by state
   (queue.md, D); do not re-derive them from Pew's region.

4. WHAT IS DRAWN. `Catholic` -> christianity.catholic (the card's word, every rite); `Baptist` ->
   christianity.baptist; `Presbyterian` -> christianity.reformed.presbyterian. The last two were
   volunteered (DO NOT READ). Shares are weighted (`weight`) and are shares of all Christians, don't know
   and refused included, as the topline's are. Applied to each sub-district's census Christians in the
   drawn regions, except Kerala and Mizoram (built above, and asserted unchanged against the file on disk)
   and the places Pew interviewed nobody (in_split.py's lists: Manipur, Sikkim, and the rest).

5. CHECKS, printed by --dry-run. (a) Codes, labels and counts asserted; the topline reproduced. (b) The
   spatial chi-square of each named answer over the drawn regions, on unweighted counts, raw and with the
   statistic divided by Pew's median design effect for Christians, 3.7 (report p. 228), since no cluster
   is in the file. An answer that differs at 0.05 after that keeps one share per drawn region. One that
   does not is POOLED: every drawn region takes the weighted share of all their Christian respondents
   together, the standard treatment for an answer whose regions cannot be told apart. Since the
   Northeast went, Catholics over the East and South (37.4 and 39.4, p 0.27 after the design effect) are
   drawn at the pooled 39.0 in both; Baptist still differs (p 0.026) and keeps its regional shares.
   (c) Catholics by region against catholic-
   hierarchy's 149 diocesan rolls (2004), each see put in its see city's state (SEE_STATE): roll / the
   share drawn, per region, whose spread over the drawn regions has to stay under CATHOLIC_SPREAD_BAR, the bar the
   Kerala check set a day earlier. (d) The same rolls by state, inside every region with 100 respondents:
   each state's rolls against Pew's regional Catholic share applied to its surveyed census Christians,
   Kerala and Mizoram left out, Arunachal Pradesh counted with Assam (no see of its own in 2004) and
   Puducherry with Tamil Nadu (ROLL_STATE_MERGE). Printed, not a guard: it is the check that withdrew the
   Northeast (spread 12), and the East, kept, spreads 3.8 because of Bihar's small Christian count
   (sources/in.md §12).

THE KNOWN COST. The South's share includes Kerala's respondents and is applied to Tamil Nadu, Karnataka,
Andhra Pradesh and Puducherry alike; the East's has no Baptist or Presbyterian answer at all, and the
pooled Catholic share reads roll / drawn from 0.97 in Odisha to 3.66 in Bihar (check (d)). The Northeast's
cost was larger, one share over Meghalaya (Presbyterian and Catholic in the Khasi and Jaintia hills,
Baptist in the Garo hills), Nagaland (Baptist), Assam, Arunachal Pradesh and Tripura, and it is no longer
drawn (point 3). With it went every Presbyterian this file drew from Pew; PEW_NAMED still names the label,
and no row carries it.

TIER
----
Named rows are `derived` and roll back to `christianity` through in2011.COLUMNS (`parent_column=
Christian`); the remainder is the census's own `Christian`, `measured`, as in_split.py's REMAINDER is.

Usage:
    python in_split_christian.py --fetch     download the paper, the diocese table and the Mizoram pages
                                             (not Pew's file, which needs Anita's account)
    python in_split_christian.py             build data/normalized/in_split_christian.csv
    python in_split_christian.py --dry-run   run every check and print the numbers, write nothing
"""

import os
import re
import sys
import urllib.parse
import urllib.request
from html import unescape

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import numpy as np
import pandas as pd

import in_split   # Pew's regions and the places it did not survey, shared with the Muslim split

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
NORM = os.path.join(HERE, "data", "normalized")
RAW = os.path.join(HERE, "data", "raw", "in")
SRC_ALLOC = os.path.join(NORM, "in_subdistrict_allocated.csv")
SRC_CENSUS = os.path.join(NORM, "in.csv")
WP468 = os.path.join(RAW, "zachariah_2016", "WP468.pdf")
WP468_URL = "https://cds.edu/wp-content/uploads/WP468.pdf"
CH = os.path.join(RAW, "catholic_hierarchy", "scin1.html")
CH_URL = "https://www.catholic-hierarchy.org/country/scin1.html"
MIZO_DIR = os.path.join(RAW, "mizoram_churches")
MIZO_URL = "http://crsmizo.mizoram.gov.in/ngo/index.php?page=ngo_{key}_year&title="
OUT = os.path.join(NORM, "in_split_christian.csv")
UA = {"User-Agent": "Mozilla/5.0"}

COL = "Christian"
REMAINDER = COL

# ------------------------------------------------------------------------------------------ Kerala
DENOMS = ["Syro-Malabar", "Syro-Malankara", "Latin Catholics", "Jacobite", "Orthodox", "Mar Thoma",
          "CSI", "Dalit Christian", "Pentecost", "Others"]
RITES = ["Syro-Malabar", "Syro-Malankara", "Latin Catholics"]
# Table 6 as printed (PDF p. 17), % of each district's Christians, DENOMS order. check_tables() re-reads
# every number off the PDF and fails on a mismatch.
TABLE6 = {
    "Thiruvananthapuram": [21.6, 2.3, 23.4, 2.8, 10.9, 3.6, 17.1, 2.7, 6.2, 9.5],
    "Kollam":             [19.7, 11.3, 26.1, 15.8, 8.4, 8.7, 1.9, 1.3, 4.7, 2.0],
    "Pathanamthitta":     [28.6, 5.4, 1.6, 7.2, 15.1, 18.5, 4.2, 4.9, 12.7, 1.7],
    "Alappuzha":          [48.4, 4.6, 19.1, 2.1, 8.0, 4.3, 4.8, 5.0, 1.9, 1.7],
    "Kottayam":           [54.7, 6.9, 4.4, 8.8, 5.5, 3.4, 4.1, 6.3, 1.8, 4.1],
    "Idukki":             [50.7, 4.4, 11.6, 5.2, 7.9, 4.9, 4.9, 3.1, 3.7, 3.6],
    "Ernakulam":          [28.4, 8.5, 24.7, 13.6, 5.9, 1.9, 4.2, 0.9, 0.9, 10.9],
    "Thrissur":           [81.4, 3.5, 3.6, 4.1, 2.5, 1.3, 1.1, 0.0, 1.4, 1.0],
    "Palakkad":           [58.1, 8.2, 11.9, 1.6, 4.1, 7.0, 1.6, 2.9, 1.8, 2.6],
    "Malappuram":         [33.9, 34.5, 9.5, 0.7, 4.8, 7.7, 1.7, 0.2, 3.8, 3.1],
    "Kozhikode":          [9.4, 16.5, 9.7, 6.8, 16.0, 22.3, 3.3, 3.0, 0.0, 13.0],
    "Wayanad":            [53.2, 4.1, 9.2, 6.9, 4.9, 0.1, 2.4, 1.5, 1.6, 16.1],
    "Kannur":             [30.8, 5.0, 40.8, 5.0, 3.7, 3.9, 7.0, 1.2, 0.0, 2.4],
    "Kasaragod":          [51.6, 11.4, 14.0, 8.9, 7.1, 0.8, 1.4, 0.5, 0.5, 3.8],
    "KERALA":             [38.2, 7.6, 15.2, 7.9, 8.0, 6.6, 4.5, 2.6, 3.5, 5.9],
}
# 2011 census district codes, names asserted against in.csv.
KERALA = {
    "32588": "Kasaragod", "32589": "Kannur", "32590": "Wayanad", "32591": "Kozhikode",
    "32592": "Malappuram", "32593": "Palakkad", "32594": "Thrissur", "32595": "Ernakulam",
    "32596": "Idukki", "32597": "Kottayam", "32598": "Alappuzha", "32599": "Pathanamthitta",
    "32600": "Kollam", "32601": "Thiruvananthapuram",
}
KERALA_STATE = "32"
KERALA_LABEL = "Christian: Catholic, three rites (KMS 2008-2014)"

# The witness. catholic-hierarchy's name for each Kerala-seated jurisdiction -> the district of its see
# city. Territories cross district lines; that is why the comparison is by group, below.
SEAT = {
    "Ernakulam-Angamaly (Syro-Malabarese)": "Ernakulam",
    "Trichur (Syro-Malabarese)": "Thrissur",
    "Changanacherry (Syro-Malabarese)": "Kottayam",
    "Palai (Syro-Malabarese)": "Kottayam",
    "Tellicherry (Syro-Malabarese)": "Kannur",
    "Irinjalakuda (Syro-Malabarese)": "Thrissur",
    "Idukki (Syro-Malabarese)": "Idukki",
    "Kothamangalam (Syro-Malabarese)": "Ernakulam",
    "Kanjirapally (Syro-Malabarese)": "Kottayam",
    "Mananthavady (Syro-Malabarese)": "Wayanad",
    "Thamarasserry (Syro-Malabarese)": "Kozhikode",
    "Palghat (Syro-Malabarese)": "Palakkad",
    "Verapoly": "Ernakulam",
    "Trivandrum": "Thiruvananthapuram",
    "Quilon": "Kollam",
    "Alleppey": "Alappuzha",
    "Cochin": "Ernakulam",
    "Neyyattinkara": "Thiruvananthapuram",
    "Kottapuram": "Thrissur",
    "Vijayapuram": "Kottayam",
    "Punalur": "Kollam",
    "Calicut": "Kozhikode",
    "Kannur": "Kannur",
    "Trivandrum (Malankarese)": "Thiruvananthapuram",
    "Tiruvalla (Malankarese)": "Pathanamthitta",
    "Muvattupuzha (Malankarese)": "Ernakulam",
}
# Absent from the table and so from the witness: the Syro-Malabar Archeparchy of Kottayam (Knanaya, a
# personal jurisdiction). Sees erected after 2004 (Mavelikara, Pathanamthitta, Sultanpet) did not exist
# yet. The Syro-Malankara Eparchy of Bathery IS in the table, spelled `Battery (Malankarese)` (25,512),
# and is not in SEAT; found 2026-09-15 by SEE_STATE below, not added here, because the rites already
# fail and adding a Syro-Malankara roll to Malappuram, Kozhikode and Wayanad only widens that spread.
GROUPS = {
    "Thiruvananthapuram, Kollam": ["Thiruvananthapuram", "Kollam"],
    "Pathanamthitta, Alappuzha, Kottayam, Idukki": ["Pathanamthitta", "Alappuzha", "Kottayam", "Idukki"],
    "Ernakulam, Thrissur": ["Ernakulam", "Thrissur"],
    "Palakkad": ["Palakkad"],
    "Malappuram, Kozhikode, Wayanad": ["Malappuram", "Kozhikode", "Wayanad"],
    "Kannur, Kasaragod": ["Kannur", "Kasaragod"],
}
CATHOLIC_SPREAD_BAR = 3.0

# ----------------------------------------------------------------------------------------- Mizoram
MIZORAM_STATE = "15"
YEAR = "2010"            # the column headed "2010 - 2011"; see the docstring
ALT_YEAR = "2011"
# key on the database -> (the name written into source_category, members 2010-11 as fetched 2026-09-15)
BODIES = {
    "pres":      ("Presbyterian Church of India", 550560),
    "baptist":   ("Baptist Church of Mizoram", 146331),
    "upcn":      ("United Pentecostal Church (North East India)", 70497),
    "upcm":      ("United Pentecostal Church (Mizoram)", 45471),
    "ecm":       ("Evangelical Church of Maraland", 37383),
    "salvation": ("Salvation Army", 36395),
    "likbk":     ("Lairam Isua Krista Baptist Kohhran", 24795),
    "seventh":   ("Seventh-day Adventist", 19235),
    "catholic":  ("Roman Catholic", 18890),
    "ikk":       ("Isua Krista Kohhran", 11257),
}
# Named, and still written to the remainder: nothing read here says which family it belongs to.
TO_REMAINDER = {"ikk"}
MIZO_LABEL = "Christian: {name} (Mizoram roll 2010-11)"

# -------------------------------------------------------------------------------- Pew, rest of India
PEW_DTA = os.path.join(RAW, "pew_india_2021.dta")
TOPLINE = os.path.join(RAW, "pew_2021", "PF_06.29.21_India_topline.pdf")
PEW_N = 29999
PEW_CHRISTIAN = 3                        # qrelsing
PEW_N_CHRISTIAN = 1011
PEW_DENOM = {1: "Catholic", 6: "No denomination or church in particular", 9: "Baptist",
             12: "Presbyterian (DO NOT READ)", 97: "All other denominations", 98: "Don't know",
             99: "Refused (DO NOT KNOW)"}
PEW_REGIONS = {1: "Northeast", 2: "North", 3: "Central", 4: "East", 5: "West", 6: "South"}
# qdenomrec code -> the category written into source_category. Everything else stays `Christian`.
PEW_NAMED = {1: "Christian: Catholic (Pew 2021)", 9: "Christian: Baptist (Pew 2021)",
             12: "Christian: Presbyterian (Pew 2021)"}
# Topline p. 23, QDENOM, the India row as printed: sixteen answers, Total, N.
QDENOM_TOPLINE = [37, 7, 7, 3, 2, 4, 0, 13, 2, 0, 5, 1, 5, 2, 1, 11, 100, 1011]
# Which printed cells each of the file's codes holds. 97 is Church of North India, Church of South India,
# Orthodox, some other, Jehovah's Witness, Adventist, Unitarian, Methodist, Pentecostal, Lutheran,
# Protestant not specified; the topline prints don't know and refused as one cell.
TOPLINE_CELLS = {1: [0], 9: [7], 12: [10], 6: [5], 97: [1, 2, 3, 4, 6, 8, 9, 11, 12, 13, 14], (98, 99): [15]}
MIN_RESPONDENTS = 100    # Christian respondents a region needs before its shares are drawn; docstring 3
# Regions over the floor that are still not drawn, with the reason written into their rows' note (no
# semicolons: the note is `;`-separated). Docstring point 3, sources/in.md §11 and §12.
NOT_DRAWN = {
    "Northeast": "one share fails the Catholic rolls state by state and puts Presbyterians in Nagaland "
                 "(sources/in.md §12)",
}
# Check (d) only: a state whose Catholics the 2004 rolls count under a neighbour's sees -> that neighbour.
# Arunachal Pradesh had no see of its own (Tezpur and Dibrugarh, in Assam); Pondicherry and Cuddalore
# reaches into Tamil Nadu.
ROLL_STATE_MERGE = {"12": "18", "34": "33"}
PEW_DEFF_CHRISTIANS = 3.7   # report p. 228, median design effect for Christians over 155 estimates
CHI_ALPHA = 0.05
# An answer whose drawn regions do not differ at CHI_ALPHA after the design effect is drawn at one pooled
# share in all of them (docstring 5(b)). This replaced, 2026-09-15, a post-hoc rule that let regional
# shares stand within 4.3 points of the pooled one (sources/in.md §12, §13).

# The witness for Catholics by region: every see in catholic-hierarchy's table (names folded to ASCII)
# -> the 2011 census code of the state its see city is in. Territories cross state lines here and there
# (Pondicherry and Cuddalore reaches into Tamil Nadu, Simla and Chandigarh into Punjab, Jammu-Srinagar
# over the Kashmir Valley); the comparison is by region, where few of those crossings leave a region.
# Arunachal Pradesh has no see of its own in 2004 (it was under Tezpur and Dibrugarh).
SEE_STATE = {
    # Kerala
    "Ernakulam-Angamaly (Syro-Malabarese)": "32", "Trichur (Syro-Malabarese)": "32",
    "Changanacherry (Syro-Malabarese)": "32", "Palai (Syro-Malabarese)": "32",
    "Tellicherry (Syro-Malabarese)": "32", "Verapoly": "32", "Trivandrum (Malankarese)": "32",
    "Irinjalakuda (Syro-Malabarese)": "32", "Idukki (Syro-Malabarese)": "32", "Trivandrum": "32",
    "Quilon": "32", "Kothamangalam (Syro-Malabarese)": "32", "Kanjirapally (Syro-Malabarese)": "32",
    "Mananthavady (Syro-Malabarese)": "32", "Alleppey": "32", "Cochin": "32", "Neyyattinkara": "32",
    "Thamarasserry (Syro-Malabarese)": "32", "Kottapuram": "32", "Vijayapuram": "32",
    "Palghat (Syro-Malabarese)": "32", "Punalur": "32", "Tiruvalla (Malankarese)": "32", "Calicut": "32",
    "Kannur": "32", "Battery (Malankarese)": "32", "Muvattupuzha (Malankarese)": "32",
    # Tamil Nadu, Puducherry
    "Kottar": "33", "Tiruchirapalli": "33", "Tuticorin": "33", "Madras and Mylapore (Meliapor)": "33",
    "Coimbatore": "33", "Sivagangai": "33", "Kumbakonam": "33", "Tanjore": "33", "Madurai": "33",
    "Vellore": "33", "Palayamkottai": "33", "Chinglepet": "33", "Dindigul": "33", "Ootacamund": "33",
    "Salem": "33", "Dharmapuri": "33", "Marthandom (Malankarese)": "33",
    "Thuckalay (Syro-Malabarese)": "33", "Pondicherry and Cuddalore": "34",
    # Karnataka
    "Bangalore": "29", "Mangalore": "29", "Mysore": "29", "Karwar": "29", "Chikmagalur": "29",
    "Belgaum": "29", "Bellary": "29", "Belthangady (Syro-Malabarese)": "29", "Shimoga": "29",
    "Gulbarga": "29",
    # Andhra Pradesh (with Telangana, as in 2011)
    "Vijayawada": "28", "Eluru": "28", "Visakhapatnam": "28", "Guntur": "28", "Khammam": "28",
    "Hyderabad": "28", "Nellore": "28", "Cuddapah": "28", "Kurnool": "28", "Nalgonda": "28",
    "Warangal": "28", "Srikakulam": "28", "Adilabad (Syro-Malabarese)": "28",
    # Maharashtra, Goa, Gujarat
    "Bombay": "27", "Vasai": "27", "Kalyan (Syro-Malabarese)": "27", "Nashik": "27", "Poona": "27",
    "Sindhudurg": "27", "Nagpur": "27", "Aurangabad": "27", "Chanda (Syro-Malabarese)": "27",
    "Amravati": "27", "Goa e Damo": "30", "Baroda": "24", "Ahmedabad": "24", "Gandhinagar": "24",
    "Rajkot (Syro-Malabarese)": "24",
    # Northeast
    "Shillong": "17", "Tura": "17", "Tezpur": "18", "Dibrugarh": "18", "Bongaigaon": "18",
    "Guwahati": "18", "Diphu": "18", "Kohima": "13", "Imphal": "14", "Aizawl": "15", "Agartala": "16",
    # East
    "Simdega": "20", "Gumla": "20", "Ranchi": "20", "Khunti": "20", "Dumka": "20", "Jamshedpur": "20",
    "Daltonganj": "20", "Hazaribag": "20", "Rourkela": "21", "Berhampur": "21",
    "Cuttack-Bhubaneswar": "21", "Sambalpur": "21", "Balasore": "21", "Calcutta": "19",
    "Jalpaiguri": "19", "Raiganj": "19", "Krishnagar": "19", "Baruipur": "19", "Bagdogra": "19",
    "Darjeeling": "19", "Asansol": "19", "Bhagalpur": "10", "Patna": "10", "Purnea": "10",
    "Bettiah": "10", "Muzaffarpur": "10",
    # Central
    "Raigarh": "22", "Ambikapur": "22", "Raipur": "22", "Jagdalpur (Syro-Malabarese)": "22",
    "Jhabua": "23", "Khandwa": "23", "Jabalpur": "23", "Indore": "23", "Bhopal": "23",
    "Sagar (Syro-Malabarese)": "23", "Gwalior": "23", "Ujjain (Syro-Malabarese)": "23",
    "Satna (Syro-Malabarese)": "23", "Meerut": "09", "Varanasi": "09", "Allahabad": "09", "Agra": "09",
    "Lucknow": "09", "Bareilly": "09", "Jhansi": "09", "Gorakhpur (Syro-Malabarese)": "09",
    "Bijnor (Syro-Malabarese)": "09",
    # North
    "Jullundur": "03", "Delhi": "07", "Simla and Chandigarh": "02", "Jammu-Srinagar": "01",
    "Udaipur": "08",
    # outside every region (Andaman and Nicobar)
    "Port Blair": "35",
}
SCIN1_ROWS = 149

NOTE_KERALA = ("level=leaf; derivation=survey_share; structure=zachariah_2016_wp468_table6 (Syro-Malabar "
               "+ Syro-Malankara + Latin); structure_geo=district:{district}; share={share:.1f}%; "
               "parent_column=Christian")
NOTE_KERALA_REST = ("level=leaf; cat=Christian; derivation=exact_single_child; church not drawn "
                    "(Jacobite, Orthodox, Mar Thoma, CSI, Dalit Christian, Pentecost, Others: unchecked; "
                    "WP468 Table 6, {district}); parent_column=Christian")
NOTE_MIZO = ("level=leaf; derivation=roll_share; structure=mizoram_des_church_members_2010_11; "
             "structure_geo=state; share={share:.2f}%; parent_column=Christian")
NOTE_MIZO_REST = ("level=leaf; cat=Christian; derivation=exact_single_child; Isua Krista Kohhran, family "
                  "not identified ({share:.2f}% of the Mizoram rolls); parent_column=Christian")
NOTE_PEW = ("level=leaf; derivation=survey_share; structure=pew_india_2021_qdenomrec; "
            "structure_geo=pew_region:{region}; share={share:.2f}%; parent_column=Christian")
NOTE_PEW_REST = ("level=leaf; cat=Christian; derivation=exact_single_child; church not named (all other "
                 "denominations, no denomination, don't know, refused; Pew 2021 {region}); "
                 "parent_column=Christian")
NOTE_PEW_THIN = ("level=leaf; cat=Christian; derivation=exact_single_child; church not split, Pew 2021 "
                 "{region} has {n} Christian respondents, under {floor}; parent_column=Christian")
NOTE_PEW_NOT_DRAWN = ("level=leaf; cat=Christian; derivation=exact_single_child; church not split, Pew 2021 "
                      "{region} ({n} Christian respondents) not drawn: {why}; parent_column=Christian")
NOTE_PEW_UNSURVEYED = ("level=leaf; cat=Christian; derivation=exact_single_child; church not split, "
                       "{why}; parent_column=Christian")
OUT_COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count", "basis",
               "year", "source_id", "tier", "note"]


# ----------------------------------------------------------------------------------------- fetching
def _get(url, data=None):
    req = urllib.request.Request(url, data=data, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _save(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        fh.write(payload)
    os.replace(tmp, path)


def fetch():
    pdf = _get(WP468_URL)
    if not pdf.startswith(b"%PDF") or b"%%EOF" not in pdf[-2048:]:
        raise SystemExit("WP468 did not come back as a whole PDF")   # [[reference_pdf_truncated_at_source]]
    _save(WP468, pdf)
    html = _get(CH_URL)
    if b"Ernakulam-Angamaly" not in html:
        raise SystemExit("catholic-hierarchy scin1 came back without the Kerala sees")
    _save(CH, html)
    for key in BODIES:
        data = urllib.parse.urlencode({"district": "State", "from_year": "2000", "to_year": "2023",
                                       "catch": ""}).encode()
        page = _get(MIZO_URL.format(key=key), data)
        if b"No. of Members" not in page:
            raise SystemExit(f"Mizoram {key}: no members table in the reply")
        _save(os.path.join(MIZO_DIR, f"{key}.html"), page)
    print(f"  fetched WP468 ({len(pdf):,} bytes), scin1 ({len(html):,} bytes), {len(BODIES)} Mizoram pages")
    print(f"  NOT fetched: {PEW_DTA}, Pew's India Survey Dataset, which needs a Pew account (sources/in.md §10)")


# ------------------------------------------------------------------------------------------ reading
def _rows_after(lines, names, width, start=0):
    out, k = {}, start
    for name in names:
        try:
            k = lines.index(name, k)
        except ValueError:
            raise SystemExit(f"WP468: row {name!r} not found")
        nums, m = [], k + 1
        while len(nums) < width:
            nums += [float(x) for x in re.findall(r"\d+\.\d", lines[m])]
            m += 1
        out[name] = nums[:width]
        k = m
    return out


def check_tables():
    """Re-read Tables 6 and 5 off the PDF; Table 6 must equal TABLE6 exactly."""
    if not os.path.exists(WP468):
        raise SystemExit(f"missing {WP468}; run --fetch")
    import fitz
    doc = fitz.open(WP468)
    if doc.page_count != 29:
        raise SystemExit(f"WP468 has {doc.page_count} pages, expected 29")
    text = doc[16].get_text()
    i6 = text.find("Table 6:")
    i5 = text.find("Table 5: Percent Distribution of Christian Denominations by Districts")
    if i6 < 0 or i5 < i6:
        raise SystemExit("Tables 6 and 5 not found on WP468 p. 17")
    names = list(TABLE6)
    t6 = _rows_after([l.strip() for l in text[i6:i5].splitlines()], names, 11)
    t5 = _rows_after([l.strip() for l in text[i5:].splitlines()], names, 11)
    for name in names:
        if t6[name][:10] != TABLE6[name]:
            raise SystemExit(f"Table 6 {name}: PDF {t6[name][:10]} != here {TABLE6[name]}")
        if t6[name][10] != 100.0 or abs(sum(TABLE6[name]) - 100) > 0.25:
            raise SystemExit(f"Table 6 {name}: row sums to {sum(TABLE6[name]):.1f}")
    # Table 5 is used only by the checks. As printed, its `Others` column sums to 101.4, a slip in the
    # paper; every other column is within 0.3 of 100.
    cols = [sum(t5[n][j] for n in names[:-1]) for j in range(11)]
    for j, col in enumerate(cols):
        if abs(col - 100) > 1.5 or t5["KERALA"][j] != 100.0:
            raise SystemExit(f"Table 5 column {j}: districts sum to {col:.1f}")
    worst = max(range(11), key=lambda j: abs(cols[j] - 100))
    print(f"  OK Table 6 re-read off WP468 p. 17, all {len(names) * 10} cells match; Table 5's columns "
          f"sum to 100 within 1.5 (worst: {(DENOMS + ['Total'])[worst]}, {cols[worst]:.1f})")
    return t5


_SCIN1_ROW = re.compile(r"<tr align=right><td>(\d+)<td>([\d,]*)<td>([\d,]*)<td>([\d.]*)%?<td align=left>"
                        r"<a [^>]*>([^<]+)</a>([^<]*)<td>(\d{4})?<td align=left>(\S*)")


def read_rolls():
    if not os.path.exists(CH):
        raise SystemExit(f"missing {CH}; run --fetch")
    text = open(CH, encoding="utf-8", errors="replace").read()
    got = {}
    for _, cath, _pop, _pct, name, _kind, year, _src in _SCIN1_ROW.findall(text):
        name = name.strip()
        if name in SEAT:
            if name in got:
                raise SystemExit(f"scin1: {name!r} appears twice")
            got[name] = (int(cath.replace(",", "") or 0), year)
    missing = set(SEAT) - set(got)
    if missing:
        raise SystemExit(f"scin1: Kerala sees not found: {sorted(missing)}")
    return got


def read_all_rolls():
    """Every see in scin1.html -> Catholics, names folded to ASCII (Goa's `Damão` is mis-encoded there)."""
    if not os.path.exists(CH):
        raise SystemExit(f"missing {CH}; run --fetch")
    text = open(CH, encoding="utf-8", errors="replace").read()
    got = {}
    for _, cath, _pop, _pct, name, _kind, _year, _src in _SCIN1_ROW.findall(text):
        key = name.strip().encode("ascii", "ignore").decode()
        if key in got:
            raise SystemExit(f"scin1: {key!r} appears twice")
        got[key] = int(cath.replace(",", "") or 0)
    if len(got) != SCIN1_ROWS:
        raise SystemExit(f"scin1: {len(got)} rows, expected {SCIN1_ROWS}")
    if set(got) != set(SEE_STATE):
        raise SystemExit(f"scin1 and SEE_STATE disagree: not mapped {sorted(set(got) - set(SEE_STATE))}, "
                         f"mapped but absent {sorted(set(SEE_STATE) - set(got))}")
    return got


def _rite(name):
    return ("Syro-Malankara" if "Malankarese" in name else
            "Syro-Malabar" if "Syro-Malabarese" in name else "Latin Catholics")


def _mizo_table(html):
    body = html[html.find("print_one"):]
    trs = re.findall(r"<tr>(.*?)</tr>", body, flags=re.S)
    rows = [[" ".join(unescape(re.sub(r"<[^>]+>", " ", c)).split())
             for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.S)] for tr in trs]
    years = next((r[2:] for r in rows if len(r) > 2 and r[1].startswith("Parameters")), [])
    seen = False
    for r in rows:
        if any("No. of Members" in c for c in r):
            seen = True
            continue
        if seen and len(r) > 2 and r[1] == "Total":
            return {y[:4]: int(t.replace(",", "")) for y, t in zip(years, r[2:]) if t.strip()}
    return {}


def read_mizoram():
    series = {}
    for key, (name, expected) in BODIES.items():
        path = os.path.join(MIZO_DIR, f"{key}.html")
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}; run --fetch")
        s = _mizo_table(open(path, encoding="utf-8", errors="replace").read())
        if s.get(YEAR) != expected:
            raise SystemExit(f"Mizoram {key}: {YEAR}-11 reads {s.get(YEAR)}, expected {expected}")
        series[key] = s
    print(f"  OK all {len(BODIES)} Mizoram rolls for 2010-11 match the figures recorded here")
    return series


def check_topline():
    """Re-read QDENOM's India row off topline p. 23 and compare it with QDENOM_TOPLINE."""
    if not os.path.exists(TOPLINE):
        raise SystemExit(f"missing {TOPLINE}; see sources/in.md §8 for the URL")
    import fitz
    doc = fitz.open(TOPLINE)
    text = doc[22].get_text()
    i, j = text.find("QDENOM."), text.find("ASK IF SIKH")
    if i < 0 or j < i:
        raise SystemExit("QDENOM block not found on topline p. 23")
    block = text[i:j]
    nums = [int(x) for x in re.findall(r"\b\d+\b", block[block.rfind("Christian"):])]
    if nums != QDENOM_TOPLINE:
        raise SystemExit(f"QDENOM on the PDF does not match:\n  pdf  {nums}\n  here {QDENOM_TOPLINE}")
    print(f"  OK QDENOM's India row re-read off topline p. 23, all {len(nums)} numbers match")


def read_pew():
    """Pew's respondent file -> {region: dict(n, counts, share)} over Christians, after the checks."""
    if not os.path.exists(PEW_DTA):
        raise SystemExit(f"missing {PEW_DTA}: Pew's India Survey Dataset needs a Pew account; "
                         f"see sources/in.md §10")
    with pd.io.stata.StataReader(PEW_DTA) as r:
        labs = r.value_labels()
    df = pd.read_stata(PEW_DTA, convert_categoricals=False,
                       columns=["qrelsing", "qdenomrec", "region", "weight"])
    if len(df) != PEW_N:
        raise SystemExit(f"Pew file has {len(df):,} rows, expected {PEW_N:,}")
    if labs.get("QRELSING", {}).get(PEW_CHRISTIAN) != "Christian":
        raise SystemExit(f"qrelsing {PEW_CHRISTIAN} is not labelled Christian")
    if dict(labs.get("QDENOMREC", {})) != PEW_DENOM:
        raise SystemExit(f"qdenomrec's labels changed: {labs.get('QDENOMREC')}")
    if dict(labs.get("REGION", {})) != PEW_REGIONS:
        raise SystemExit(f"region's labels changed: {labs.get('REGION')}")
    ch = df[df["qrelsing"] == PEW_CHRISTIAN]
    if (len(ch) != PEW_N_CHRISTIAN or ch["qdenomrec"].isna().any()
            or df["qdenomrec"].notna().sum() != PEW_N_CHRISTIAN):
        raise SystemExit("qdenomrec is not asked of exactly the 1,011 Christians")
    if not set(ch["qdenomrec"].astype(int)) <= set(PEW_DENOM) or ch["weight"].le(0).any():
        raise SystemExit("an unlabelled qdenomrec code, or a weight that is not positive")
    print(f"  OK Pew file: {len(df):,} respondents, {len(ch):,} Christians, qdenomrec's seven codes and "
          f"the six regions labelled as expected")

    # (a) the weighted file against the printed India row
    w = ch.groupby(ch["qdenomrec"].astype(int))["weight"].sum()
    pct = 100 * w / w.sum()
    worst = 0.0
    for code, cells in TOPLINE_CELLS.items():
        got = sum(pct.get(c, 0.0) for c in (code if isinstance(code, tuple) else (code,)))
        printed = sum(QDENOM_TOPLINE[c] for c in cells)
        if abs(got - printed) > 0.5 * len(cells) + 1e-9:     # each printed cell is rounded on its own
            raise SystemExit(f"qdenomrec {code}: weighted {got:.2f}% against the topline's {printed}")
        worst = max(worst, abs(got - printed) / len(cells))
    print(f"  OK the weighted file reproduces the topline's India row (largest gap per printed cell "
          f"{worst:.2f} points); Catholic {pct[1]:.1f}, Baptist {pct[9]:.1f}, Presbyterian {pct[12]:.1f}, "
          f"all other {pct[97]:.1f}")

    regions = {}
    for code, name in PEW_REGIONS.items():
        s = ch[ch["region"] == code]
        c = s["qdenomrec"].astype(int)
        ws = s.groupby(c)["weight"].sum()
        regions[name] = dict(n=len(s), weight=float(ws.sum()),
                             counts={k: int((c == k).sum()) for k in PEW_DENOM},
                             share={k: float(ws.get(k, 0.0) / ws.sum()) for k in PEW_DENOM})
    if sum(r["n"] for r in regions.values()) != PEW_N_CHRISTIAN:
        raise SystemExit("a Christian respondent has no region")
    return regions


# ----------------------------------------------------------------------------------------- checks
def check_kerala(t5, census_d):
    """The three checks in the docstring. Returns each district's Catholic share (%)."""
    names = [n for n in TABLE6 if n != "KERALA"]
    total_c = sum(census_d.values())

    # 1. internal consistency, and what Table 5's Total column is
    worst = 0.0
    for j, d in enumerate(DENOMS):
        mass = {n: TABLE6[n][j] * t5[n][10] for n in names}
        s = sum(mass.values())
        for n in names:
            if s > 0:
                worst = max(worst, abs(100 * mass[n] / s - t5[n][j]))
    print(f"\n  1. Table 6 x Table 5's Total column reproduces Table 5: worst cell off by {worst:.2f} points")
    if worst > 1.5:
        raise SystemExit("Tables 5 and 6 no longer agree; re-read the paper")
    print("     Table 5's Total column (the sample) against the census's own Christians, % by district:")
    for n in names:
        print(f"       {n:<20} sample {t5[n][10]:5.1f}   census {100 * census_d[n] / total_c:5.1f}")

    # applied to the census
    applied = {n: {d: TABLE6[n][j] / 100 * census_d[n] for j, d in enumerate(DENOMS)} for n in names}
    print("\n     Kerala on census weights against Table 2 (which applies the sample-weighted row):")
    for j, d in enumerate(DENOMS):
        c = sum(applied[n][d] for n in names)
        print(f"       {d:<16} {c:>10,.0f}  {100 * c / total_c:5.1f}%   Table 2 {TABLE6['KERALA'][j]:5.1f}%")

    # 2. the witness
    rolls = read_rolls()
    years = sorted({y for _, y in rolls.values()})
    print(f"\n  2. roll / survey by group of seat districts (catholic-hierarchy, {len(rolls)} Kerala sees, "
          f"data years {years})")
    cats = RITES + ["Catholic"]
    spread = {}
    print(f"     {'group':<46}" + "".join(f"{c:>17}" for c in cats))
    ratios = {c: [] for c in cats}
    for g, members in GROUPS.items():
        cells = []
        for c in cats:
            want = RITES if c == "Catholic" else [c]
            roll = sum(n for s, (n, _) in rolls.items() if SEAT[s] in members and _rite(s) in want)
            kms = sum(applied[m][d] for m in members for d in want)
            r = roll / kms if kms else float("nan")
            if roll > 0 and kms > 0:
                ratios[c].append(r)
            cells.append(f"{r:>8.2f} ({roll / 1000:>4.0f}k)" if roll else f"{'no see':>17}")
        print(f"     {g:<46}" + "".join(cells))
    for c in cats:
        spread[c] = max(ratios[c]) / min(ratios[c])
    print("     spread, largest ratio over smallest: " +
          ", ".join(f"{c} {spread[c]:.1f}" for c in cats))
    if spread["Catholic"] > CATHOLIC_SPREAD_BAR:
        raise SystemExit(f"Catholic roll/survey spread {spread['Catholic']:.2f} is over the bar "
                         f"{CATHOLIC_SPREAD_BAR}; the Catholic total no longer passes")
    # The rites partition the Catholic total, so a person the survey filed under the wrong rite is wrong in
    # two rites at once: they pass or fail together. Syro-Malabar alone spreads about 2.2, under the bar,
    # only because it is two-thirds of the total; in Kannur and Kasaragod, where one Syro-Malabar see and
    # one Latin see cover the same two districts, it reads 2.22 where Latin reads 0.27.
    if max(spread[r] for r in RITES) <= CATHOLIC_SPREAD_BAR:
        print("  !! all three rites now pass the bar; reconsider drawing them (docstring, point 2)")
    print(f"  OK Catholics as a whole pass (spread {spread['Catholic']:.1f} <= {CATHOLIC_SPREAD_BAR}); "
          f"the rites do not, and are not drawn")
    print("  3. Jacobite, Orthodox, Mar Thoma, CSI, Pentecost: no independent figure; not drawn")
    return {n: sum(TABLE6[n][:3]) for n in names}


def check_mizoram(series, census_christians):
    for year in (YEAR, ALT_YEAR):
        s = sum(series[k].get(year, 0) for k in BODIES)
        print(f"  Mizoram rolls {year}-{int(year[2:]) + 1:02d}: {s:,} = {100 * s / census_christians:.1f}% "
              f"of the census's {census_christians:,} Christians")
    total = sum(series[k][YEAR] for k in BODIES)
    if not 0.95 <= total / census_christians <= 1.10:
        raise SystemExit("Mizoram rolls no longer sum near the census")
    alt = sum(series[k][ALT_YEAR] for k in BODIES)
    print(f"  {'body':<46}{'2009-10':>9}{'2010-11':>9}{'2011-12':>9}  share  share 2011-12")
    worst = 0.0
    shares = {}
    for k, (name, _) in BODIES.items():
        s = series[k]
        shares[k] = s[YEAR] / total
        alt_share = s[ALT_YEAR] / alt
        worst = max(worst, abs(shares[k] - alt_share))
        flag = ""
        for y in ("2009", ALT_YEAR):
            if y in s and abs(s[y] - s[YEAR]) / s[YEAR] > 0.15:
                flag = "  !! >15% from a neighbouring year"
        print(f"  {name:<46}{s.get('2009', 0):>9,}{s[YEAR]:>9,}{s[ALT_YEAR]:>9,}  {100 * shares[k]:5.2f}"
              f"  {100 * alt_share:5.2f}{flag}")
    print(f"  largest share change if 2011-12 were read instead: {100 * worst:.2f} points")
    return shares


def _pew_place(geo_id):
    """(region, why-not-surveyed) for a sub-district, from in_split.py's lists."""
    st, dcode = geo_id[:2], geo_id[:5]
    if st not in in_split.STATES:
        raise SystemExit(f"state code {st} is not in in_split.STATES")
    why = in_split.NOT_SURVEYED_DISTRICT.get(dcode) or in_split.NOT_SURVEYED_STATE.get(st)
    return in_split.STATES[st][1], why


def check_pew(regions, ch):
    """Checks 5(b) to 5(d) of the docstring. Returns the list of regions drawn."""
    from scipy.stats import chi2, chi2_contingency

    names = list(PEW_REGIONS.values())
    drawn = [r for r in names if regions[r]["n"] >= MIN_RESPONDENTS and r not in NOT_DRAWN]
    order = [1, 9, 12, 6, 97, 98, 99]
    short = {1: "Catholic", 9: "Baptist", 12: "Presb.", 6: "no denom.", 97: "all other", 98: "DK", 99: "ref."}
    print(f"\n  Pew 2021, Christians by region: respondents, and weighted % of each region's Christians")
    print(f"     {'region':<11}{'n':>5}" + "".join(f"{short[c]:>11}" for c in order) + "   drawn")
    for r in names:
        g = regions[r]
        print(f"     {r:<11}{g['n']:>5}" + "".join(f"{100 * g['share'][c]:>11.1f}" for c in order)
              + ("   yes" if r in drawn else "   no, NOT_DRAWN (docstring 3)" if r in NOT_DRAWN
                 else f"   no, under {MIN_RESPONDENTS} respondents"))
    if drawn != ["East", "South"]:
        print(f"  !! the drawn regions are now {drawn}; re-read the docstring's point 3 and the note")

    # (b) spatial chi-square on unweighted counts, raw and divided by the design effect. `drawn_share` is
    # what the build applies: the region's own share, or the pooled one where the regions do not differ.
    for r in names:
        regions[r]["drawn_share"] = dict(regions[r]["share"])
    pooled = {}
    print(f"\n  (b) chi-square of each named answer against the rest, across regions (unweighted; "
          f"'deff' divides the statistic by {PEW_DEFF_CHRISTIANS})")
    for code, label in PEW_NAMED.items():
        if not any(regions[r]["counts"][code] for r in drawn):
            # Presbyterian since the Northeast was withdrawn: no drawn region has one answer, so nothing
            # is drawn and there is nothing to test (a zero row would fail chi2_contingency).
            print(f"     {label:<36}no answer in the drawn regions; nothing drawn")
            continue
        cells = []
        for units, tag in ((drawn, "drawn regions"), (names, "all six")):
            hits = np.array([regions[r]["counts"][code] for r in units])
            tots = np.array([regions[r]["n"] for r in units])
            stat, p, dof, _ = chi2_contingency(np.array([hits, tots - hits]))
            p_deff = float(chi2.sf(stat / PEW_DEFF_CHRISTIANS, dof))
            cells.append((tag, p, p_deff))
        print(f"     {label:<36}" + "   ".join(f"{t}: p {p:.1e}, deff p {pd_:.1e}" for t, p, pd_ in cells))
        if cells[0][2] >= CHI_ALPHA:
            # The drawn regions cannot be told apart for this answer, so it is drawn at one pooled share:
            # the weighted share over all their Christian respondents together (docstring 5(b)).
            tot = sum(regions[r]["weight"] for r in drawn)
            share = sum(regions[r]["share"][code] * regions[r]["weight"] for r in drawn) / tot
            pooled[code] = share
            for r in drawn:
                regions[r]["drawn_share"][code] = share
            print(f"  -> {label}: the drawn regions do not differ at {CHI_ALPHA} after the design effect, so "
                  f"POOLED at {100 * share:.1f}% in each ("
                  + ", ".join(f"{r} {100 * regions[r]['share'][code]:.1f}" for r in drawn) + " on their own)")
        else:
            print(f"  -> {label}: regional shares, one per drawn region")

    # census Christians per region, over the places Pew surveyed
    census = {r: 0 for r in names}
    for geo_id, n in zip(ch["geo_id"], ch["count"]):
        region, why = _pew_place(geo_id)
        if region is not None and why is None:
            census[region] += int(n)

    # (c) Catholics by region against the diocesan rolls
    rolls = read_all_rolls()
    by_region = {r: 0 for r in names}
    for see, n in rolls.items():
        region = in_split.STATES[SEE_STATE[see]][1]
        if region is not None:              # Imphal (Manipur) and Port Blair: Pew surveyed neither
            by_region[region] += n
    print(f"\n  (c) Catholics: catholic-hierarchy's {len(rolls)} sees (2004 rolls) against the Pew share drawn "
          f"(pooled where (b) pooled it) on each region's surveyed census Christians")
    print(f"     {'region':<11}{'census Chr.':>13}{'Pew %':>8}{'survey':>12}{'rolls':>12}{'roll/survey':>13}")
    ratio = {}
    for r in names:
        survey = regions[r]["drawn_share"][1] * census[r]
        ratio[r] = by_region[r] / survey
        print(f"     {r:<11}{census[r]:>13,}{100 * regions[r]['drawn_share'][1]:>8.1f}{survey:>12,.0f}"
              f"{by_region[r]:>12,}{ratio[r]:>13.2f}" + ("" if r in drawn else "   (not drawn)"))
    sd = max(ratio[r] for r in drawn) / min(ratio[r] for r in drawn)
    sa = max(ratio.values()) / min(ratio.values())
    print(f"     spread over the drawn regions {sd:.2f}, over all six {sa:.2f} (bar {CATHOLIC_SPREAD_BAR})")
    if sd > CATHOLIC_SPREAD_BAR:
        raise SystemExit(f"Catholic roll/survey spread over the drawn regions is {sd:.2f}, over "
                         f"{CATHOLIC_SPREAD_BAR}")

    # (d) the same rolls by state, inside every region with enough respondents. Printed, not a guard: the
    # East, kept, spreads over the bar on Bihar (docstring point 5).
    st_census, st_rolls = {}, {}
    for geo_id, n in zip(ch["geo_id"], ch["count"]):
        st = geo_id[:2]
        region, why = _pew_place(geo_id)
        if st in (KERALA_STATE, MIZORAM_STATE) or region is None or why is not None:
            continue
        st = ROLL_STATE_MERGE.get(st, st)
        st_census[st] = st_census.get(st, 0) + int(n)
    for see, n in rolls.items():
        st = ROLL_STATE_MERGE.get(SEE_STATE[see], SEE_STATE[see])
        st_rolls[st] = st_rolls.get(st, 0) + n
    merged = {v: [k for k in ROLL_STATE_MERGE if ROLL_STATE_MERGE[k] == v] for v in ROLL_STATE_MERGE.values()}
    print(f"\n  (d) Catholics by state: rolls against Pew's regional share applied to each state's surveyed "
          f"census Christians (Kerala and Mizoram left out)")
    for r in names:
        if regions[r]["n"] < MIN_RESPONDENTS:
            continue
        share = regions[r]["drawn_share"][1]
        tag = ("   (pooled)" if r in drawn and 1 in pooled else "") if r in drawn else "   (not drawn)"
        print(f"     {r}, {100 * share:.1f}% Catholic{tag}")
        print(f"       {'state':<42}{'census Chr.':>13}{'drawn':>11}{'rolls':>11}{'roll/drawn':>12}")
        ratios = []
        for st in sorted(st_census, key=lambda s: -st_census[s]):
            if in_split.STATES[st][1] != r:
                continue
            name = " + ".join([in_split.STATES[st][0]] + [in_split.STATES[m][0] for m in merged.get(st, [])])
            got = share * st_census[st]
            roll = st_rolls.get(st, 0)
            cell = f"{roll / got:>12.2f}" if roll and got else f"{'no see':>12}"
            if roll and got:
                ratios.append(roll / got)
            print(f"       {name.title():<42}{st_census[st]:>13,}{got:>11,.0f}{roll:>11,}{cell}")
        if len(ratios) > 1:
            print(f"       spread {max(ratios) / min(ratios):.1f} (bar {CATHOLIC_SPREAD_BAR})")
    return drawn, census, pooled


# ------------------------------------------------------------------------------------------- build
def _largest_remainder(shares, total):
    """Integer split of `total` by fractional shares summing to 1, summing to `total` exactly."""
    exact = [total * s for s in shares]
    base = [int(x) for x in exact]
    short = total - sum(base)
    order = sorted(range(len(exact)), key=lambda i: -(exact[i] - base[i]))
    for i in order[:short]:
        base[i] += 1
    return base


def _kept(frame):
    k = frame[frame["geo_id"].str[:2].isin([KERALA_STATE, MIZORAM_STATE])]
    return (k[["geo_id", "source_category", "count", "tier", "note"]]
            .sort_values(["geo_id", "source_category"]).reset_index(drop=True))


def build():
    t5 = check_tables()
    check_topline()
    census = pd.read_csv(SRC_CENSUS, dtype={"geo_id": str}, low_memory=False,
                         usecols=["geo_id", "geo_level", "geo_name", "source_category", "count"])
    cc = census[census["source_category"] == COL]
    dist = cc[cc["geo_level"] == "district"].set_index("geo_id")
    for code, name in KERALA.items():
        if dist.loc[code, "geo_name"] != name:
            raise SystemExit(f"district {code}: in.csv says {dist.loc[code, 'geo_name']!r}, here {name!r}")
    state = cc[cc["geo_level"] == "state"].set_index("geo_id")
    if state.loc[MIZORAM_STATE, "geo_name"] != "MIZORAM" or state.loc[KERALA_STATE, "geo_name"] != "KERALA":
        raise SystemExit("state codes 15 and 32 are not Mizoram and Kerala in in.csv")
    print("  OK 14 Kerala district codes and the two state codes match in.csv by name")
    in_split._check_names()
    census_d = {name: int(dist.loc[code, "count"]) for code, name in KERALA.items()}
    if sum(census_d.values()) != int(state.loc[KERALA_STATE, "count"]):
        raise SystemExit("Kerala's districts do not sum to its state Christians")

    catholic = check_kerala(t5, census_d)
    print()
    mizo_christians = int(state.loc[MIZORAM_STATE, "count"])
    shares = check_mizoram(read_mizoram(), mizo_christians)

    df = pd.read_csv(SRC_ALLOC, dtype={"geo_id": str}, low_memory=False)
    ch = df[df["source_category"] == COL].copy()
    if (ch["tier"] != "measured").any() or ch["geo_id"].duplicated().any():
        raise SystemExit("the allocated `Christian` rows changed shape")
    if ((ch["count"] - ch["count"].round()).abs() > 1e-9).any():
        raise SystemExit("a census `Christian` count is not a whole number")
    ch["count"] = ch["count"].round().astype("int64")
    ch = ch[ch["count"] > 0]

    print()
    regions = read_pew()
    drawn, census_r, pooled = check_pew(regions, ch)

    keys = list(BODIES)
    rows, tally = [], {}
    pew = {r: {"census": 0, **{lab: 0 for lab in PEW_NAMED.values()}, REMAINDER: 0}
           for r in list(PEW_REGIONS.values())}

    def add(key, n):
        tally[key] = tally.get(key, 0) + n

    for r in ch.itertuples(index=False):
        base = dict(geo_id=r.geo_id, geo_level=r.geo_level, geo_name=r.geo_name,
                    basis=r.basis, year=r.year, source_id=r.source_id)
        st, dcode = r.geo_id[:2], r.geo_id[:5]
        n = int(r.count)
        if st == KERALA_STATE:
            district = KERALA[dcode]
            share = catholic[district]
            cath, rest = _largest_remainder([share / 100, 1 - share / 100], n)
            if cath:
                rows.append(dict(base, source_category=KERALA_LABEL, count=cath, tier="derived",
                                 note=NOTE_KERALA.format(district=district, share=share)))
            if rest:
                rows.append(dict(base, source_category=REMAINDER, count=rest, tier="measured",
                                 note=NOTE_KERALA_REST.format(district=district)))
            add(KERALA_LABEL, cath)
            add("Kerala remainder", rest)
        elif st == MIZORAM_STATE:
            parts = _largest_remainder([shares[k] for k in keys], n)
            rest = 0
            for k, p in zip(keys, parts):
                if k in TO_REMAINDER:
                    rest += p
                    continue
                label = MIZO_LABEL.format(name=BODIES[k][0])
                add(label, p)
                if p:
                    rows.append(dict(base, source_category=label, count=p, tier="derived",
                                     note=NOTE_MIZO.format(share=100 * shares[k])))
            if rest:
                rows.append(dict(base, source_category=REMAINDER, count=rest, tier="measured",
                                 note=NOTE_MIZO_REST.format(share=100 * sum(shares[k] for k in TO_REMAINDER))))
            add("Mizoram remainder", rest)
        else:
            region, why = _pew_place(r.geo_id)
            if why is not None:
                rows.append(dict(base, source_category=REMAINDER, count=n, tier="measured",
                                 note=NOTE_PEW_UNSURVEYED.format(why=why)))
                add("not surveyed by Pew", n)
            elif region in NOT_DRAWN:
                rows.append(dict(base, source_category=REMAINDER, count=n, tier="measured",
                                 note=NOTE_PEW_NOT_DRAWN.format(region=region, n=regions[region]["n"],
                                                                why=NOT_DRAWN[region])))
                add(f"{region} outside Mizoram, not drawn", n)
            elif region not in drawn:
                rows.append(dict(base, source_category=REMAINDER, count=n, tier="measured",
                                 note=NOTE_PEW_THIN.format(region=region, n=regions[region]["n"],
                                                           floor=MIN_RESPONDENTS)))
                add(f"{region}, under the respondent floor", n)
            else:
                sh = regions[region]["drawn_share"]
                named = [sh[c] for c in PEW_NAMED]
                parts = _largest_remainder(named + [1 - sum(named)], n)
                pew[region]["census"] += n
                for (code, label), p in zip(PEW_NAMED.items(), parts):
                    pew[region][label] += p
                    add(label, p)
                    if p:
                        geo = "+".join(drawn) if code in pooled else region
                        rows.append(dict(base, source_category=label, count=p, tier="derived",
                                         note=NOTE_PEW.format(region=geo, share=100 * sh[code])))
                pew[region][REMAINDER] += parts[-1]
                add("Pew regions remainder", parts[-1])
                if parts[-1]:
                    rows.append(dict(base, source_category=REMAINDER, count=parts[-1], tier="measured",
                                     note=NOTE_PEW_REST.format(region=region)))

    out = pd.DataFrame(rows, columns=OUT_COLUMNS)
    per_unit = out.groupby("geo_id")["count"].sum()
    measured = ch.set_index("geo_id")["count"]
    if set(per_unit.index) != set(measured.index) or (per_unit.reindex(measured.index) - measured).abs().max():
        raise SystemExit("in_split_christian does not conserve sub-district Christians")
    total = int(measured.sum())
    print(f"\n  OK every one of {len(measured):,} sub-districts' rows sum to its census `Christian` count, "
          f"{total:,} in all")
    if os.path.exists(OUT):
        if not _kept(pd.read_csv(OUT, dtype={"geo_id": str}, low_memory=False)).equals(_kept(out)):
            raise SystemExit("Kerala's or Mizoram's rows differ from the file on disk; they stay as built")
        print("  OK Kerala's and Mizoram's rows are identical to the file on disk")

    print("\n  Kerala, by district: census Christians, Catholic share, Catholics drawn")
    kc = out[out["source_category"] == KERALA_LABEL].assign(d=lambda x: x["geo_id"].str[:5])
    got = kc.groupby("d")["count"].sum()
    for code, name in KERALA.items():
        print(f"    {name:<20}{census_d[name]:>11,}{catholic[name]:>7.1f}%{int(got.get(code, 0)):>11,}")

    labels = list(PEW_NAMED.values())
    print("\n  Pew regions drawn, outside Kerala and Mizoram: census Christians, then each named church")
    print(f"    {'region':<11}{'Christians':>12}" + "".join(f"{l.split(': ')[1].split(' (')[0]:>14}" for l in labels)
          + f"{'unnamed':>12}")
    for r in drawn:
        g = pew[r]
        print(f"    {r:<11}{g['census']:>12,}" + "".join(f"{g[l]:>14,}" for l in labels) + f"{g[REMAINDER]:>12,}")
    print("\n  every Christian, by what carries them")
    for label, n in tally.items():
        print(f"    {label:<66}{n:>12,}{100 * n / total:>7.2f}%")
    named_total = int(out.loc[out["tier"] == "derived", "count"].sum())   # every church row is `derived`
    print(f"    {'on a church node':<66}{named_total:>12,}{100 * named_total / total:>7.2f}%")
    return out


def main():
    if "--fetch" in sys.argv:
        fetch()
        return
    out = build()
    if "--dry-run" in sys.argv:
        print("  --dry-run: nothing written")
        return
    tmp = OUT + ".part"
    out.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, OUT)
    print(f"  wrote {OUT} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
