# India — Census of India 2011, table C-01

Rebuilt by `sources/in.py`. Boundaries by `sources/in_geo.py`, documented in
`sources/in_geo.md`.

**1,210,854,977 people on 5,988 sub-districts** — more than the thirteen countries drawn
before it put together, and the largest single acquisition the project will ever make.

## 1. What was downloaded, and from where

| file | what | catalogue |
|---|---|---|
| `DDW{SS}C-01 MDDS.XLS` × 35 | C-01 per state: 8 categories × state/district/sub-district/town | NADA 11362–11396 |
| `DDW00C-01 MDDS.XLS` | the same 8 categories for all 35 states, as a summary | NADA 11361 |
| `DDW00C-01 Appendix MDDS.xlsx` | 83 named religions inside `Other religions and persuasions`, India + state | NADA 11398 |
| `DDW00C-01 Annexure MDDS.xlsx` | 47 write-in sects inside the six religions, India + state | NADA 11397 |

`https://censusindia.gov.in/nada/index.php/catalog/<id>` for each; the download link
carries a per-file resource id that is **not derivable from the state code**, so `in.py`
scrapes it off the catalogue page rather than constructing it. ~14MB in total.

Re-fetch: `python sources/in.py --fetch`.

### The TLS failure is the server's, and is the same defect as Poland's

`censusindia.gov.in` serves an incomplete certificate chain, so curl, `requests` and
`certifi` all fail identically with *unable to get local issuer certificate*. This is not a
bad URL, not a proxy and not bot protection — `stat.gov.pl` does exactly the same thing
(`sources.md` §5a). Verification is disabled **for this host only** and the payload is
validated structurally instead: OLE2 or zip magic, minimum size, the expected sheet, and
the expected header text at the expected column. We cannot authenticate the server, so we
authenticate the bytes.

## 2. The shape of the source

C-01 is the spine. Eight columns:

    Total | Hindu | Muslim | Christian | Sikh | Buddhist | Jain
          | Other religions and persuasions | Religion not stated

at four nested levels in one column set, distinguished only by which code is non-zero:

    state       s != 00, d == 000,   sd == 00000, town == 000000
    district    d != 000,            sd == 00000, town == 000000
    sub-district                     sd != 00000, town == 000000
    town                                          town != 000000

**Town rows are urban-only subsets of their sub-district**, and this is the single easiest
way to double-count India. A town's people are already inside the sub-district row above
it. They happen to carry `Total/Rural/Urban == Urban` and never `Total`, so filtering to
`Total` excludes them — but that is a coincidence of the layout rather than a rule, so
`in.py` **asserts** it and fails if a town ever carries a `Total` row.

### Five things that had to be got right

1. **`00` is text in one file and a number in another.** The state files store state,
   district and sub-district codes as text (`"00"`); the Appendix stores the same codes as
   numbers, so `str(cell)` gives `"0"`. India's own row was therefore read as a 36th state
   and the entire Appendix was counted twice — 15,725,800 people against a 7,937,734
   bucket. `_code(value, width)` normalises once, and every code goes through it.

2. **The same category is spelled two ways in two tables of the same census.** C-01 writes
   `Other religions and persuasions`; the Appendix writes `Other Religions and
   Persuasions`, capital R and P, as the parent row repeated inside every state block.
   Matching the parent by name silently failed to recognise it, so each state's bucket
   total was added as though it were a named religion — the same doubling as above, by a
   different route. **The parent is matched on its code (`700000`)**, not its label.

3. **The `Other` column header is not constant across the 35 files.** Seven of the eight
   headers are identical everywhere; the eighth is `Other religions and persuasions
   (incl.Unclassified Sect.)` and, in some states, with a trailing ` - 2011`. The header
   check compares on the leading text, so that variation passes while a genuine column swap
   still fails.

4. **`DDW00C-01` contains no India row.** It is 35 states × 3 (T/R/U) and nothing else, so
   it cannot check the national total. What it can do is better: it is an independently
   published copy of every state's eight figures, and comparing it state by state would
   catch a column-offset error in any single state file — which summing to the national
   total cannot, because a swapped Sikh/Buddhist pair leaves the `Total` column correct.

5. **The Appendix names a religion only at 100+ adherents nationally**, so every unit has
   an unnamed remainder — 149,668 people, 1.9% of the bucket. `in.py` emits that remainder
   as its own category rather than letting the allocation normalise it away, which would
   have inflated every Adivasi religion by about 2%.

### What reconciles, and exactly

- The 35 state files sum to **1,210,854,977**, the published national total, **exactly**.
  India neither rounds nor suppresses, so there is no band to compute and none is allowed.
- All **315** (state × category) figures agree between `DDW00C-01` and the 35 state files.
- The Appendix's named religions sum to **7,788,066**, 98.1% of the `Other religions and
  persuasions` bucket, and never exceed it in any unit.

## 3. `Religion not stated` is 0.24%, and that is not good news

2,867,303 people, the smallest such residual of any country on the map — against Poland's
20.5% refusal and Romania's 14% absent variable. The reason is structural rather than
creditable: **the religion question was answered by the head of household for the whole
household**, so it is not a personal self-identification and nobody had the opportunity to
decline on their own behalf. `basis` is `self_id` because that is the nearest of the
project's categories, and §3.1's warning applies with more force here than anywhere: a
household head's answer and a personal answer are not the same measurement, and India's
figures should not be read as though 1.2 billion people each answered for themselves.

**India has no `no religion` box at all.** There is no `unaffiliated` figure for India from
any source here — the 33,304 `Atheist` write-ins in the Appendix are the whole of it. On
the map India will therefore appear to be a country with essentially no irreligion, and
that is a property of the question and not of the country.

## 4. The Annexure looks like a sect breakdown and is not — read this before using it

This cost the better part of an hour and would have cost a day if it had been believed.

The C-01 Annexure is titled *Details of sects/religions clubbed under specific religious
communities*. It is **arithmetically a true partition**: for each state and each of the six
religions, `Religion:X` = an unspecified remainder + the named sects, to within a few
hundred people nationally. Everything about its structure says it is usable for splitting
Hindus into Lingayats or Christians into Catholics, exactly as the Appendix is used for
splitting the `Other` bucket.

It is not usable, and the numbers say so plainly:

| religion | total | what the Annexure names |
|---|---|---|
| Muslim | 172,204,810 | **573 Shia**, 267 Sunni, 33,460 Bohra, 5,929 Agakhani, 119 Ahmadia |
| Christian | 27,806,028 | **8,399 Catholic**, 603 Protestant, 191 Orthodox, 146 Jacobite |
| Jain | 4,447,575 | 3,269 Digambar, 275 Swetamber |
| Buddhist | 8,407,065 | 34,123 Nav Buddhist, 998 Hinayana, 697 Bon |
| Hindu | 962,970,404 | **2,663,229 Lingayat**, 245,954 Bathou, 101,740 Satnami |

Nobody believes India has 573 Shia Muslims or 8,399 Catholics. **What the Annexure counts
is people who wrote a SECT where the form asked for a religion** — a measure of insistence,
not of membership. Every figure in it is an undercount of the real community by one to
three orders of magnitude, and the plausible-looking entries are no exception: Lingayat's
2.66M is against a Karnataka community usually put near 10 million.

The trap is that the large ones look real. Lingayat is 2.66M and 99% in Karnataka, which
would draw beautifully and would be wrong by a factor of four. **Nothing from the Annexure
is mapped.** It is normalised into `in.csv` with a `Sect: ` prefix and a note, all 47
entries are listed by name in `taxonomy/in2011.py`'s `EXCLUDED` so that a new one in a
reissue fails the check rather than being swallowed, and it is drawn nowhere.

What it is genuinely good for: **names**. It is a published list of the sects Indians
volunteer, which is a lead generator for a future source that counts them properly.

## 5. The Appendix, which is the reason India is worth drawing

83 named religions, 7,788,066 people, at state level. Nearly all Adivasi, and no other
census on earth names them.

| religion | people | where |
|---|---|---|
| Sarna | 4,957,467 | Jharkhand 83%, Odisha 8%, West Bengal 8% |
| Gond / Gondi | 1,026,344 | Madhya Pradesh 57%, Chhattisgarh 36% |
| Sari Dharma | 506,369 | West Bengal 100% |
| Doni Polo / Sidonyi Polo | 331,370 | Arunachal Pradesh 98% |
| Sanamahi | 222,422 | Manipur 100% |
| Khasi | 138,512 | Meghalaya 100% |
| Niamtre | 84,276 | Meghalaya 100% |
| Parsi/Zorastrian | 57,264 | Maharashtra 78%, Gujarat 17% |
| Atheist | 33,304 | Maharashtra 29% |
| Bahai | 4,572 | scattered |
| Jews / Judaism | 4,429 | Manipur 46% — the Bnei Menashe |

**These are floors, not measurements.** India's census form lists six religions and every
one of these 7.9 million people had to be written in under `Other`. Many more Adivasi are
recorded as Hindu and some as Christian, and the boundary is politically live: the demand
for a `Sarna` code on the census form has been running since 1951 and was refused again for
2011.

The concentration is the reason `allocate.py --within` had to be written — see
`sources/in_geo.md` and `spec.md` §3.10.

## 6. Judgement calls

Recorded in `taxonomy/in2011.py`'s `REVIEW`. The three that could most easily be wrong:

- **`Sari Dharma` → `indigenous.indian.sarna`**, merging 506,369 people into Sarna rather
  than giving them a node. Sari/Sarna Dharam is the Santal and Oraon name for the same
  sacred-grove religion, and the perfect state-complementarity — Sarna 83% Jharkhand, Sari
  Dharma 100% West Bengal — is evidence for one religion under two regional spellings. If
  that is wrong, half a million people are on the wrong node.
- **`Pagan` → `indigenous.indian.khasi`, not `paganism`**, and **`Animist` →
  `indigenous.indian`, not `paganism`** — while `cz2021.py` sends its `animismus` to
  `paganism`. Same word, opposite meaning: a Czech write-in is a Western neo-pagan
  self-description, a Meghalaya write-in is the colonial-era label for the traditional
  religion. Decided by where it was written.
- **`ADI DHARM` → `indigenous.indian`, not `ravidassia`.** 82,255 people, and 65% of them
  in **Odisha** rather than Punjab — which is the giveaway that this is the generic `Adi
  Dharam` ('original religion') write-in of the Adivasi belt and not the Ad Dharm movement
  of Punjab's Ravidassia Dalits. `ravidassia` was the obvious wrong answer.

## 7. What is not done

- **No 2021 figures, and there will be none for some time.** The census was postponed from
  2021 and has not been held. §3.4's "structure from the detailed source, totals from the
  recent one" is unavailable because there is no recent one.
- **No sect detail for the six religions**, and none is obtainable from this census by any
  route — see §4. India's Shia and Sunni, its Syro-Malabar and Latin Catholics, its
  Digambar and Swetambar Jains are inside single categories and no Indian census table
  separates them. This is the largest single R2 gap on the map and it will need a
  non-census source. For Christians, the non-census sources were scouted on 2026-09-15:
  `sources.md §scout-2026-09-15-india-christians` (Kerala by district from the Kerala
  Migration Surveys, Mizoram from the state's church rolls, all of India only from Pew's
  microdata).
- **Town-level splitting of the 23 residual units** is available for free and not taken;
  see `sources/in_geo.md` §3.

## 8. Muslim branches from Pew's *Religion in India* (2021), added 2026-09-14

Built by `in_split.py`, which writes `data/normalized/in_split.csv`; `countries.py::_in_counts`
swaps it in for the allocated file's `Muslim` rows. Anita's ruling is spec §2.7a. This closes
§7's "no sect detail" for Islam only.

### The source

| | |
|---|---|
| publisher | Pew Research Center, *Religion in India: Tolerance and Segregation*, 29 June 2021 |
| topline | `https://www.pewresearch.org/wp-content/uploads/sites/20/2021/06/PF_06.29.21_India_topline.pdf`, printed p. 23 |
| report | `https://www.pewresearch.org/religion/2021/06/29/religion-in-india-tolerance-and-segregation/`, full PDF on disk |
| on disk | `data/raw/in/pew_2021/PF_06.29.21_India_topline.pdf` and `PF_06.29.21_India_full_report.pdf` |
| item | QSECT *"Are you ...?"*, asked of Muslims; `Ahmadiyya` is a volunteered code (DO NOT READ) |
| fieldwork | 17 Nov 2019 to 23 Mar 2020, face to face (CAPI), RTI International; 3,336 Muslims answered QSECT |
| geography | six regions, the zonal councils |

% of Muslims, as printed. `in_split.py` re-reads all 56 numbers off the PDF and fails on a mismatch.

| region | N | Sunni | Shi'a | some other sect | no sect in particular | Ahmadiyya | DK/refused |
|---|---:|---:|---:|---:|---:|---:|---:|
| India | 3,336 | 55 | 6 | 2 | 14 | 1 | 22 |
| Northeast | 512 | 32 | 5 | 4 | 20 | 0 | 38 |
| North | 655 | 80 | 7 | 1 | 6 | 0 | 6 |
| Central | 202 | 79 | 5 | 1 | 2 | 0 | 12 |
| East | 1,016 | 43 | 2 | 1 | 18 | 0 | 35 |
| West | 579 | 50 | 11 | 4 | 17 | 1 | 17 |
| South | 372 | 38 | 6 | 7 | 24 | 4 | 22 |

**Four rows do not sum to 100.** Northeast, Central and East print to 99 and South to 101, with
`Total` printed as 100 on each; each cell is rounded on its own. Nothing is re-rounded: the three
named shares are used as printed and the rounding falls on the unspecified remainder.

### The regions, read at source

PDF page numbers (the printed folio is one lower). The p. 16 map is titled *How regions of India
are defined in this report* and says the regions *"reflect zonal council divisions"*; p. 224
footnote 26 names the States Re-organisation Act 1956 and the North Eastern Council Acts of 1972
and 2002.

| region | states and UTs | where it is stated |
|---|---|---|
| North | Chandigarh, Delhi, Haryana, Himachal Pradesh, Jammu and Kashmir, Ladakh, Punjab, Rajasthan | p. 123 text |
| Central | Chhattisgarh, Madhya Pradesh, Uttar Pradesh, Uttarakhand | **p. 16 map only**; no text list anywhere in the report |
| East | Bihar, Jharkhand, Odisha, West Bengal | p. 33 text |
| Northeast | Arunachal Pradesh, Assam, Manipur, Meghalaya, Mizoram, Nagaland, Sikkim, Tripura | **p. 16 map only** |
| West | Goa, Gujarat, Maharashtra (p. 43 text); the map also draws Dadra and Nagar Haveli and Daman and Diu inside it | p. 43, p. 16 |
| South | Andhra Pradesh, Karnataka, Kerala, Tamil Nadu, Telangana, Puducherry | p. 49 text |

The Central and Northeast lists were read off the rendered map's region boundaries, and they are
the Central Zonal Council and the North Eastern Council exactly. The 2011 census has Andhra
Pradesh undivided; both halves are South.

### What stays undivided, and why

Anita's line is whether anything measured the place (Ecuador's Galápagos, `queue.md`). The grain
is the state or UT, which is Pew's stratum (p. 224). **7,196,283 Muslims, 4.18%**, stay on
`islam`:

| place (2011 census units) | Muslims | Pew says |
|---|---:|---|
| Kashmir Valley: Kupwara, Badgam, Baramula, Bandipore, Srinagar, Ganderbal, Pulwama, Shupiyan, Anantnag, Kulgam | 6,640,957 | *"Fieldwork could not be conducted in the Kashmir Valley due to security concerns"* (p. 16); *"Kashmir districts"* dropped after sampling (p. 227); the 480 planned interviews moved to Jammu, Haryana and West Bengal (p. 229) |
| Ladakh: Leh (Ladakh), Kargil | 127,296 | *"No locations in ... Ladakh were selected"* (p. 16), and hatched on the map |
| Manipur, Sikkim | 249,703 | no interviews, COVID-19 (p. 16, p. 230) |
| Chandigarh, Dadra and Nagar Haveli, Daman and Diu | 83,646 | *"No locations ... were selected"* (p. 16) |
| Lakshadweep, Andaman and Nicobar Islands | 94,681 | outside the sample design (p. 224, footnote 27) |

**"Kashmir Valley" is read as the 2011 Kashmir division.** Pew names no districts. Jammu division,
including Doda, Ramban and Kishtwar, was surveyed (fieldwork dates for *Jammu & Kashmir*, p. 222)
and gets the North's shares. p. 25 also drops *"a few districts elsewhere"* for security without
naming them; nothing can be done about those.

Ladakh, Chandigarh and the two western UTs were in Pew's frame and drew no location. They are
left undivided anyway, since nothing measured them. For Ladakh that also avoids a known wrong
answer: Kargil's Muslims are mostly Shia and the North's shares would have made 80% of them Sunni.

### Result

| region | census Muslims | Sunni | Shi'a | Ahmadiyya | unspecified |
|---|---:|---:|---:|---:|---:|
| Northeast | 11,216,626 | 3,589,311 | 560,829 | 0 | 7,066,486 |
| North | 12,640,005 | 10,112,012 | 884,809 | 0 | 1,643,184 |
| Central | 45,180,485 | 35,692,583 | 2,259,021 | 0 | 7,228,881 |
| East | 47,918,298 | 20,604,887 | 958,355 | 0 | 26,355,056 |
| West | 18,939,477 | 9,469,746 | 2,083,345 | 189,380 | 7,197,006 |
| South | 29,153,984 | 11,078,534 | 1,749,216 | 1,166,145 | 15,160,089 |
| not surveyed | 7,196,283 | 0 | 0 | 0 | 7,196,283 |
| **India** | **172,245,158** | **90,547,073** | **8,495,575** | **1,355,525** | **71,846,985** |

Checks: every one of 5,974 sub-districts' rows sums to its census `Muslim` count; 35 state codes
and the 12 unsurveyed district codes are asserted against `in.csv` by name. Census-weighted over
the surveyed units, Sunni is 54.9% against Pew's national 55, Shi'a 5.1 against 6, Ahmadiyya 0.8
against 1. Those are weighted differently (Pew by adult Muslims in its frame), so this is a
relationship and not an identity.

### Calls made, all in `taxonomy/in2011.py`'s `REVIEW`

- **Shi'a goes to `islam.shia`, not `.jaafari`.** The card names no school, and India's Bohras and
  Khojas are not Twelvers.
- **The split is `derived`, not `modelled`, unlike Türkiye.** The brief asked for `modelled`. The
  census counted these Muslims at the sub-district, so this is spec §7a-i's Israel case: only which
  branch is inferred. It also decides the viewer: `index.html` rolls `derived` dots up to their
  column and removes `modelled` dots outright, so `modelled` would make `inferred dots: not shown`
  delete about 100 million counted Muslims instead of redrawing them as `islam`. `uk_split.py`'s
  England is the same shape and is `derived`. The remainder and the unsurveyed units stay
  `measured`, as `uk_split.py`'s `Christian` remainder does.
- **Ahmadiyya is drawn, and is the weakest number here.** South's 4% of 372 respondents is roughly
  fifteen people and becomes 1.17M once applied to the census; the North, which holds Qadian,
  reads 0. No outside figure was checked. First thing to revisit.

### The known cost

One share per region (spec §3.10). Lucknow's Shia get Central's 5% like the rest of Uttar
Pradesh; Hyderabad's get the South's 6%; the Dawoodi Bohras of Gujarat and Maharashtra are spread
over the whole West at 11%. The East's 35% DK/refused and the Northeast's 38% make those two
regions mostly unspecified, which §2.7a accepts. Pew's own warning (p. 25) applies: the survey
*"cannot speak to the experiences and views of Kashmiri Muslims"*, and the Muslim sample is 11%
of respondents against about 13% of adults.

### Ahmadiyya folded back into `islam`, 2026-09-14

Anita's call, the same day: *"lets move ahmadiya back for now"*. The 1,355,525 people who were on
`islam.ahmadiyya` are back on the census's own `Muslim` (`islam`, `measured`). `in_split.py` still
splits Pew's Ahmadiyya share out as its own part and then adds it to the remainder, so **Sunni and
Shi'a are unchanged to the person**: India draws 90,547,073 Sunni, 8,495,575 Shi'a and 73,202,510
on `islam`. The result table above and the Ahmadiyya bullet under *Calls made* are the state before
this change.

Why, from the check in `sources/branches.md` (Decided, "India's Ahmadis, checked 2026-09-14"): no
census of independent India counts Ahmadis, and the only national figure with any standing is
about 150,000, from the US State Department's religious freedom reports for 2021 to 2023, which
cite unnamed media reports. Pew's pattern also fails where the answer is known: the East reads 0
although Odisha has organised Ahmadi villages (Kerang, Soro, Bhadrak), and the South's 1.17M is 8
to 10 times the whole national estimate, most likely one or two sampling points. No Ahmadiyya row
was added to the national estimate layer.

## 9. Christian churches in Kerala and Mizoram, added 2026-09-15

Built by `in_split_christian.py`, which writes `data/normalized/in_split_christian.csv`;
`countries/in.py::_in_counts` swaps it in for the allocated file's `Christian` rows, as it does
`in_split.csv` for `Muslim`. Queue item `queue.md` D, scouted in `sources.md
§scout-2026-09-15-india-christians`. Spec §2.7a's construction; named rows `derived`, rolling back to
`christianity` through `in2011.COLUMNS`. Session `d743fc47-inchr`.

**Drawn:** 3,926,477 Catholics in Kerala's 14 districts, and 945,127 Christians on nine church rows in
Mizoram. **Undivided:** 20,721,988 Christians outside those two states (74.5%), plus 2,214,792 in
Kerala and 11,204 in Mizoram, on `christianity`.

### Sources, all under `data/raw/in/`, all refetched by `--fetch`

| file | what |
|---|---|
| `zachariah_2016/WP468.pdf` | K.C. Zachariah, *Religious Denominations of Kerala*, CDS Working Paper 468, April 2016, `cds.edu/wp-content/uploads/WP468.pdf`, 29 pp. Table 6 (district rows) and Table 5 (column shares), PDF p. 17 |
| `catholic_hierarchy/scin1.html` | catholic-hierarchy.org's all-diocese table for India, Annuario Pontificio 2005 (2004 data), the witness |
| `mizoram_churches/<body>.html` | Mizoram Statistical Database, NGO > churches, year-wise form, district `State`, 2000-2023, ten bodies |

### Kerala: the Zachariah table was checked, and most of it failed

The same paper's Muslim table has Pathanamthitta 60.9% Shia, so nothing was drawn before three checks.
Every number below is printed by `python in_split_christian.py --dry-run`.

1. **The table is unweighted.** Table 6 times Table 5's `Total` column reproduces Table 5 to 0.55 points,
   so the tables agree; but that `Total` column is the sample's spread of Christians, not the census's
   (Thrissur 4.9% of the sample against 12.3% of the census, Kozhikode 6.0 against 2.1, Wayanad 8.1 against
   2.8, Kasaragod 4.9 against 1.4). Table 6's `KERALA` row, and Table 2's state totals which are that row
   times 6,141,269, weight districts by who was sampled. That is why Table 2 equals Table 6's Kerala row
   (the scout's open question). The build applies each district row to that district's census Christians;
   on census weights Kerala is 41.9% Syro-Malabar against Table 2's 38.2%.
2. **The three Catholic rites fail against the dioceses' rolls; their sum passes.** The 26 Kerala-seated
   jurisdictions in `scin1.html`, each placed in its see city's district, compared with Table 6 on the
   census, by six groups of districts. A roll runs above self-identification, so the test is whether
   roll / survey is steady between places:

   | group | Syro-Malabar | Syro-Malankara | Latin | all Catholics |
   |---|---:|---:|---:|---:|
   | Thiruvananthapuram, Kollam | no see | 4.26 | 2.52 | 1.69 |
   | Pathanamthitta, Alappuzha, Kottayam, Idukki | 1.10 | 0.29 | 1.35 | 1.06 |
   | Ernakulam, Thrissur | 1.47 | 0.08 | 1.54 | 1.36 |
   | Palakkad | 0.99 | no see | no see | 0.74 |
   | Malappuram, Kozhikode, Wayanad | 2.19 | no see | 0.96 | 1.44 |
   | Kannur, Kasaragod | 2.22 | no see | 0.27 | 1.16 |
   | largest over smallest | 2.2 | 51.1 | 9.2 | 2.3 |

   The cleanest cell is Kannur and Kasaragod, where Tellicherry (Syro-Malabar) and Kannur (Latin) cover
   the same two districts: rolls 279,200 and 32,540, survey about 126,000 and 120,000. No roll margin makes
   one rite 2.2 and the other 0.27 in one place. In Ernakulam the only Syro-Malankara see rolls 11,067
   against about 106,000 in the survey. The rites partition the Catholic total, so a respondent filed under
   the wrong rite is wrong in two of them; Syro-Malabar's 2.2 is under the bar only because it is two
   thirds of the total, and the three pass or fail together. Most of the Catholic total's 2.3 is see
   territory crossing group lines (Changanacherry, seated in Kottayam, covers Thiruvananthapuram and
   Kollam; Palakkad's Latin Catholics are under Coimbatore, a Tamil Nadu see not counted here). The bar,
   `CATHOLIC_SPREAD_BAR = 3.0`, was set after a first look, so it is post hoc; any bar from 3 to 20 gives
   the same answer. Likely cause: Syro-Malabar and Syro-Malankara differ by two letters, and four of
   Kerala's churches have "Malankara" in their formal names. Absent from the witness: the Knanaya
   Archeparchy of Kottayam and the Syro-Malankara Eparchy of Bathery (no rows in the table).
3. **Nothing independent reaches the other churches.** No diocesan roll was found for the Jacobite,
   Orthodox, Mar Thoma or CSI churches, and three of them carry the "Malankara" name. Not searched beyond
   the scout's pass: each church's own diocesan membership. The J.B. Koshy Commission on Kerala's
   Christian minorities had its release reported on 28 February 2026, with no count by church in the
   coverage; the scout did not find the report online.

So Kerala draws one share per district, the three rites summed, on `christianity.catholic`: Thrissur
88.5%, Palakkad 78.2, Malappuram 77.9, Kasaragod 77.0, Kannur 76.6, Alappuzha 72.1, Idukki 66.7, Wayanad
66.5, Kottayam 66.0, Ernakulam 61.6, Kollam 57.1, Thiruvananthapuram 47.3, Kozhikode 35.6, Pathanamthitta
35.6. It is a floor (Catholics who answered Dalit Christian or Others stay on the parent). **Kozhikode's
35.6% is the number to distrust first:** its cells put 54.8% of its Christians in the three "Malankara"
churches, and Thamarassery's 124,664 Syro-Malabar Catholics are seated there. The six-group check
cannot see inside Malappuram, Kozhikode and Wayanad.

Table 5 as printed has an `Others` column summing to 101.4, a slip in the paper; it is used only by the
checks.

### Mizoram: the state's church rolls for 2010-11

Ten bodies' members (the eleventh, "Association", has none). **The year is 2010-11**, because the
census date, 1 March 2011, is in that financial year; it sums to 960,814, 100.5% of the census's 956,331
Christians. The scout's 2011-12 sums to 1,016,033 (106.2%), mostly because the Salvation Army steps from
36,395 to 55,791 that year; reading it instead would move no share by more than 1.7 points.

| body | 2010-11 | share | node |
|---|---:|---:|---|
| Presbyterian Church of India (Mizoram Synod) | 550,560 | 57.30 | `christianity.reformed.presbyterian` |
| Baptist Church of Mizoram | 146,331 | 15.23 | `christianity.baptist` |
| United Pentecostal Church (North East India) | 70,497 | 7.34 | `christianity.pentecostal.oneness` |
| United Pentecostal Church (Mizoram) | 45,471 | 4.73 | `christianity.pentecostal.oneness` |
| Evangelical Church of Maraland | 37,383 | 3.89 | `christianity.other` |
| Salvation Army | 36,395 | 3.79 | `christianity.holiness.salvation-army` |
| Lairam Isua Krista Baptist Kohhran | 24,795 | 2.58 | `christianity.baptist` |
| Seventh-day Adventist | 19,235 | 2.00 | `christianity.adventist` |
| Roman Catholic | 18,890 | 1.97 | `christianity.catholic.latin` |
| Isua Krista Kohhran | 11,257 | 1.17 | stays on `christianity`, family not identified |

Each share divides every Mizoram sub-district's census Christians. **One share for the state is the
known cost:** the Evangelical Church of Maraland (Saiha) and the Lairam Isua Krista Baptist Kohhran
(Lawngtlai) are spread across Aizawl, and the Baptist Church of Mizoram's southern weight is lost. The
district form returns empty cells. Flagged series: Salvation Army (the 2011-12 step), Seventh-day
Adventist (26,858 in 2009-10, 12,542 in 2013-14), Presbyterian 2014-15 (325,214, an entry error in a year
not read).

### Calls, all in `taxonomy/in2011.py`'s `REVIEW`

- Kerala's Catholics on `christianity.catholic`, not the rite nodes, for point 2.
- Kerala's non-Catholic answers stay on `christianity`: unchecked is not drawn.
- The Evangelical Church of Maraland on `christianity.other`, since `christianity.evangelical` holds an
  answer and not a body. The United Pentecostal Church (Mizoram) on Oneness by its name alone.
- No new nodes. Mar Thoma, Malankara Orthodox and Jacobite would have needed ASARB's leaves promoted
  (`christianity.oriental.marthoma`, `.malankara-orthodox`) and were not drawn.

## 10. Christian churches for the rest of India from Pew's respondent file, added 2026-09-15

Built by `in_split_christian.py`, the §9 script extended; session `cb8b206e-in`, under a supervisor.
`python in_split_christian.py --dry-run` prints every number below. Kerala and Mizoram are unchanged,
asserted row for row against the file on disk before it is overwritten. The Muslim split (§8) is not
touched.

**Drawn:** 4,848,532 Catholics, 2,793,795 Baptists and 1,810,899 Presbyterians in Pew's Northeast, East
and South, outside Kerala and Mizoram. With §9, **14,324,830 of India's 27,819,588 Christians (51.5%) are
on a church node**, up from 4,871,604 (17.5%).

### The file, and whose terms govern it

| | |
|---|---|
| publisher | Pew Research Center, *India Survey Dataset*, Neha Sahgal and Jonathan Evans, 2021, doi:10.58094/rfte-a185 |
| behind | *Religion in India: Tolerance and Segregation*, 29 June 2021 (§8) |
| on disk | `data/raw/in/pew_india_2021.dta`, 10,235,100 bytes, Stata, stamped 24 May 2023; Anita's download `Pew India Survey Dataset.DTA` from her Pew account (ask 032). Not fetchable by a script |
| size | 29,999 respondents, 312 variables; 1,011 Christians |
| item | `qdenomrec`, QDENOM recoded, *"Please tell me which denomination or church, if any, you identify with MOST CLOSELY?"*, asked if `qrelsing` = 3 (Christian) |
| geography | `region`, Pew's six zonal-council regions (§8); nothing finer |
| weight | `weight`, summing to 29,999 |

**`ICPSR_38489-V1.zip` beside it is another study.** ICPSR 38489 is the *East Asian Social Survey (EASS),
Cross-National Survey Data Sets: Culture and Globalization in East Asia, 2018* (its manifest, its
codebook's cover, its description page). `ask/RULINGS.md` (ask 032) calls it this file's codebook and
terms; it is neither, and its terms (research use, no redistribution) do not govern the Pew file. Nothing
here reads it. The Pew file came with no documentation; the topline and report in `pew_2021/` stand in.

**Pew's own Terms of Use govern it** (`pewresearch.org/about/terms-and-conditions/`, read 2026-09-15
through WebFetch's extraction, twice, quoting; the dataset page asks for them to be accepted at download
and is behind the account). §13, *Additional Survey Dataset Terms and Conditions*: a licence to "publish,
modify, create derivatives of, or otherwise exploit the survey datasets"; publication of the Data
"limited to excerpts" and never "in full or substantially in full"; nothing implying a policy or lobbying
position of the Center; attribution to the Center; and *"you must include the following disclaimer with
your use of any Data: 'Pew Research Center bears no responsibility for the analyses or interpretations of
the data presented here. The opinions expressed herein, including any implications for policy, are those
of the author and not of Pew Research Center.'"* §14: no attempt to identify respondents, or to link
records with other data to identify them. Regional shares applied to census counts are a derivative that
publishes no respondent record and identifies nobody, so this is not an ask. **The disclaimer closes
India's `note_public`, word for word**, and `source` names the dataset. Not read closely enough to rule
on: whether §13 allows a sold print (`[[reference_poster_commercial_licences]]`); the extraction quoted no
commercial clause, which is not the same as there being none.

### The public file recodes the card

Topline p. 23 prints QDENOM for India in sixteen answers. The file keeps seven codes:

| code | label | Christians | weighted % | topline cells it holds |
|---|---|---:|---:|---|
| 1 | Catholic | 374 | 37.1 | Catholic 37 |
| 9 | Baptist | 181 | 13.2 | Baptist (DO NOT READ) 13 |
| 12 | Presbyterian (DO NOT READ) | 96 | 5.1 | Presbyterian 5 |
| 6 | No denomination or church in particular | 23 | 3.5 | 4 |
| 97 | All other denominations | 246 | 30.4 | Church of North India 7, Church of South India 7, Orthodox 3, some other 2, Jehovah's Witness 0, Adventist 2, Unitarian 0, Methodist 1, Pentecostal 5, Lutheran 2, Protestant not specified 1 |
| 98, 99 | Don't know; Refused | 76; 15 | 9.3; 1.2 | DK/Refused 11 |

Every row reproduces within its rounding (largest gap 0.47 points per printed cell), which confirms the
codes and the weight; `check_topline()` re-reads the printed row off the PDF. **The Church of North India
and Church of South India, 14% of India's Christians between them, are inside code 97 and cannot be
drawn from this file.** Nor can the Orthodox or the Pentecostals.

### Nothing below region

`region` is the only geography. There is no state, district or sampling-point variable. `Q85AREC` ("In
what state or union territory were you raised?") is recoded to same state / different state / don't know;
`qmlangrec` is Hindi / not Hindi; `qrid` is a respondent number from 23 to 34,929 with no cluster in it.
So the Kerala and Mizoram respondents cannot be taken out of the South and Northeast pools that are applied
to their neighbours.

Pew's design (report pp. 223-227): 30 strata built from states, 138 PSUs (groups of districts), six
sub-districts per PSU, four villages or census blocks per sub-district, twelve households per village; the
Northeast allocated more than its share; Christians targeted through a composite measure of size (731
expected without it, 1,011 achieved). Median design effect for Christians 3.7, margin of error 5.9 points
(p. 228).

### Christians by region, weighted % of each region's Christians

| region | respondents | Catholic | Baptist | Presbyterian | no denomination | all other | don't know | refused | drawn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Northeast | 326 | 16.2 | 33.6 | 31.8 | 1.3 | 14.4 | 1.1 | 1.6 | yes |
| North | 24 | 52.4 | 0 | 0 | 18.4 | 16.9 | 3.7 | 8.6 | no |
| Central | 10 | 62.2 | 0 | 0 | 3.7 | 8.4 | 25.7 | 0 | no |
| East | 123 | 37.4 | 0 | 0 | 0 | 37.7 | 24.9 | 0 | yes |
| West | 56 | 52.5 | 0 | 0 | 3.3 | 37.3 | 4.9 | 2.0 | no |
| South | 472 | 39.4 | 13.0 | 0 | 4.6 | 34.3 | 7.5 | 1.2 | yes |
| India | 1,011 | 37.1 | 13.2 | 5.1 | 3.5 | 30.4 | 9.3 | 1.2 | |

Unweighted: Catholic 62, 13, 6, 43, 32, 218; Baptist 114 in the Northeast and 67 in the South; all 96
Presbyterians in the Northeast.

### Which regions are drawn: 100 Christian respondents or more

`MIN_RESPONDENTS = 100` is Italy's `N_FLOOR` (`sources/it.py`), where 100 respondents put the standard
error on a Catholic share near 4.3 points. It was picked after the table above was printed, so it is not
blind; it is a borrowed number, not one fitted to these answers. Pew printed QDENOM for India only, though
it printed QSECT by region down to Central's 202 Muslims, and with twelve interviews to a village the
North's 24 and Central's 10 Christians may be one or two villages; the file has no cluster id to check
(`stability.py`'s `CELL_CAP` cannot run). The floor is a minimum, not a guarantee: at the design effect
of 3.7 the East's 123 are worth about 33.

Italy's thin units take their parent's share. Here the only parent is all of India, whose Baptists and
Presbyterians Pew found only in the Northeast and South, and whose 37% Catholic is below all three thin
regions' readings (52 to 62%). So the North, Central and West stay on `christianity` rather than take a
national share that their own respondents contradict.

**No split-half.** The survey is one wave with no cluster variable, and six regions give a rank test no
power anyway (`playbooks/cab.md`, Turkmenistan). What stands in:

- **(b) The spatial chi-square** of each named answer against the rest over the drawn regions, on
  unweighted counts, raw and with the statistic divided by the design effect 3.7: Catholic p 2.7e-14 (2.1e-04
  after), Baptist 1.0e-19 (7.4e-06), Presbyterian 3.4e-43 (3.3e-12). The build stops if one is 0.05 or
  more after the division. A clustered answer can have a design effect above the median, so this is not a
  cluster check.
- **(c) Catholics against catholic-hierarchy's 149 diocesan rolls** (`scin1.html`, 2004), each see put in
  its see city's state (`SEE_STATE`), against Pew's share of the region's surveyed census Christians:

  | region | census Christians surveyed | Pew Catholic % | survey | rolls | roll / survey |
  |---|---:|---:|---:|---:|---:|
  | Northeast | 6,653,490 | 16.2 | 1,078,110 | 1,258,335 | 1.17 |
  | North | 676,264 | 52.4 | 354,347 | 259,518 | 0.73 (not drawn) |
  | Central | 1,098,053 | 62.2 | 682,786 | 601,324 | 0.88 (not drawn) |
  | East | 3,368,181 | 37.4 | 1,260,149 | 1,941,686 | 1.54 |
  | West | 1,762,381 | 52.5 | 925,890 | 1,585,489 | 1.71 (not drawn) |
  | South | 12,910,581 | 39.4 | 5,083,139 | 11,307,772 | 2.22 |

  Spread over the drawn regions 1.91, under `CATHOLIC_SPREAD_BAR` (3.0, set by §9 the same morning); over
  all six 3.04. A roll runs above self-identification, so the North's 0.73, the only ratio under 1, is a
  sign its 52% is high, and it is not drawn. The South's 2.22 is the highest drawn ratio; since Kerala's
  own rolls run 1.28 times its KMS Catholics (§9), the rest of the South reads about 2.35, so Tamil Nadu's
  and Andhra Pradesh's Catholics are probably under-drawn at 39%.
- **(d) The Northeast's named Protestants against Mizoram's rolls.** Pew implies 2,114,860 Presbyterians
  in the surveyed Northeast; Mizoram's roll holds 550,560 (26%), leaving 1,564,300 for the other states.
  Baptists 2,233,378, of which Mizoram's two Baptist rolls hold 171,126 (8%). The Nagaland Baptist Church
  Council's 716,495 baptised members (the scout's reading of the Baptist World Alliance page, undated) sit
  inside the second figure comfortably. Neither is a check with power.

### Result

| drawn region, census Christians split | Christians | Catholic | Baptist | Presbyterian | unnamed, on `christianity` |
|---|---:|---:|---:|---:|---:|
| Northeast, without Mizoram, Manipur and Sikkim | 5,697,159 | 923,139 | 1,912,361 | 1,810,899 | 1,050,760 |
| East | 3,368,181 | 1,260,157 | 0 | 0 | 2,108,024 |
| South, without Kerala, Lakshadweep, Andaman and Nicobar | 6,769,312 | 2,665,236 | 881,434 | 0 | 3,222,642 |

Undivided and `measured`: North 676,264, Central 1,098,053, West 1,762,381 (under the floor); 1,350,638
where Pew interviewed nobody (Manipur, Sikkim, the Kashmir Valley, Ladakh, Chandigarh, Dadra and Nagar
Haveli, Daman and Diu, Lakshadweep, Andaman and Nicobar; `in_split.py`'s lists, imported). On a church
node: Pew 9,453,226, Kerala 3,926,477, Mizoram 945,127. Every one of 5,974 sub-districts sums to its census
`Christian` count.

### Calls, all in `taxonomy/in2011.py`'s `REVIEW`

- **One Northeast share, Presbyterians in Nagaland.** About 550,000 Presbyterians are drawn in Nagaland,
  whose churches are Baptist associations, and Meghalaya, Assam, Arunachal Pradesh and Tripura take the
  same Baptist share. Kept because the Muslim split draws Pew's regions the same way (spec §3.10) and the
  region's total holds. **The first call to reverse**; the alternative is the Northeast's Baptists and
  Presbyterians on `christianity`.
- `Catholic` on `christianity.catholic`, not `.latin`: the card names no rite.
- North, Central and West undivided under the floor, above; their shares are printed by `--dry-run`.
- Shares of all Christians, don't know and refused included, as the topline prints them.
- The disclaimer in `note_public`, because §13 requires it with any use of the data.

### Found on the way

- **`scin1.html` does list the Syro-Malankara Eparchy of Bathery**, as `Battery (Malankarese)`, 25,512
  Catholics. §9's docstring called it absent. It would sit in the Malappuram, Kozhikode and Wayanad group,
  where Syro-Malankara already has no see, so it adds a ratio where §9 had none; the rites already fail,
  and §9's check was not re-run with it. Recorded in the docstring beside `SEAT`.
- The ICPSR zip, above.

## 11. Review, 2026-09-15 (session `cb8b206e-rev2`)

A full review of §9 and §10. `check_md.py` is clean, `built_countries.py --check` passes, and `check_rollup.py in`
shows 121,305,212 derived people, all rolling up and none orphaned. A screenshot at the country's `view` shows no
dots in the sea, no blank state and nothing that looks broken. Not rebuilt; put back in `queue.md` (D, the `in` item).

**The one Northeast share fails §10's own Catholic test inside the region.** Check (c) compares
catholic-hierarchy's 2004 rolls with Pew's share applied to census Christians, by region, and passes at a
spread of 1.91 under `CATHOLIC_SPREAD_BAR` = 3.0. I ran the same comparison by state inside each drawn region,
on the same `scin1.html` and `SEE_STATE` (read-only scratch script, nothing written):

| Northeast, drawn at 16.2% Catholic | census Christians | rolls | roll / drawn |
|---|---:|---:|---:|
| Assam with Arunachal Pradesh (no see of its own in 2004) | 1,584,599 | 567,217 | 2.21 |
| Meghalaya (Shillong, Tura) | 2,213,027 | 588,465 | 1.64 |
| Tripura (Agartala) | 159,882 | 21,162 | 0.82 |
| Nagaland (Kohima) | 1,739,651 | 50,873 | 0.18 |

That is a spread of 12.2. A roll runs above self-identification, which is why §10 reads the North's 0.73 as a
sign of a share that is too high. By the same reading, Nagaland's Catholics are drawn at about five times
the Diocese of Kohima's own roll: about 282,000 drawn against 50,873 on its books. Meghalaya and Assam are
under-drawn by that same share. For comparison, the East spreads 3.8 (Bihar 3.81 on 129,247 Christians,
Jharkhand 1.42, Odisha 1.01) and the South outside Kerala spreads 1.7 (Andhra Pradesh 3.49, Karnataka 2.23,
Tamil Nadu with Puducherry 2.09). So the South holds, the East is marginal, and the Northeast fails at four
times the bar.

This is the Presbyterian problem in §10's calls (about 553,000 drawn in Nagaland) showing up in the one church
that has an outside figure to check against. The error has a direction; it is not noise. Northeast churches
follow the tribe, and so the state: Baptist in Nagaland and the Garo hills, Presbyterian and Catholic in the
Khasi and Jaintia hills. A regional average matches none of them. Nagaland is drawn 33.6% Baptist (about
585,000), while the Nagaland Baptist Church Council alone has 716,495 baptised members, a count that leaves
out children.

**The region's total does not hold either, though `REVIEW` says it does.** Mizoram draws from its own rolls,
while the rest of the Northeast still takes the full regional share, which includes Mizoram's respondents.
Here is the Northeast as drawn (Mizoram's rolls plus the Pew share elsewhere) against the totals Pew implies
for the same 6,653,490 surveyed Christians:

| church | drawn | Pew implies | difference |
|---|---:|---:|---:|
| Presbyterian | about 2.36M | 2.11M | +12% |
| Baptist | about 2.08M | 2.23M | -7% |
| Catholic | about 0.94M | 1.08M | -13% |

Splitting what is left after Mizoram instead would put Presbyterians at 27.5% rather than 31.8%. That fixes the
totals but still puts about 480,000 Presbyterians in Nagaland.

**What I would do:** take the Northeast out of the Pew split and leave its Christians outside Mizoram on
`christianity`, as North, Central and West already are, and keep the East and South. That removes 923,139
Catholics, 1,912,361 Baptists and all 1,810,899 Pew Presbyterians. The share of India's Christians on a church
would fall from 51.5% to 34.8%. The note's Northeast figures and the Nagaland sentence would go with them.

Smaller points, recorded and not fixed:

- `taxonomy/in2011.py`: the `REVIEW` text for `Pagan` and `Non Christians` says `indigenous.indian`, but `MAP`
  files both on `indigenous.indian.khasi`. That text is older than this upgrade. The comment above `COLUMNS`
  still says only Kerala and Mizoram get their churches from a second source.
- Kozhikode's **36%** Catholic, printed in bold in the note, is a soft doubt. The two sees seated in the
  district (Thamarassery 124,664, Calicut 35,123) hold 3.4 times the 46,820 Catholics that KMS gives its 131,516
  Christians, against 1.28 for Kerala as a whole. Both sees also reach into Malappuram and Wayanad, and §9's
  check on that three-district group passes, so this is a doubt about one printed figure, not a failure.
- Voice: the new Christian paragraphs read plainly. They add no em dashes and bold only figures and the
  paragraph openers. The closing disclaimer is Pew's required text.

## 12. The Northeast taken off the Pew split, 2026-09-15 (session `cb8b206e-in2`)

A supervisor's call on §11, built under a supervisor. `python in_split_christian.py --dry-run` prints every
number below.

**Changed.** Pew's Northeast (326 Christian respondents) is no longer drawn. Its 5,697,159 Christians outside
Mizoram, Manipur and Sikkim are back on `christianity`, `measured`, with a note giving the reason
(`NOT_DRAWN`). That removes 923,139 Catholics, 1,912,361 Baptists and all 1,810,899 Pew Presbyterians. The
East and South are unchanged (1,260,157 and 2,665,236 Catholics, 881,434 Baptists), and Kerala and Mizoram
are identical to the file on disk row for row. **9,678,431 of India's 27,819,588 Christians (34.8%) are on a
church node**, down from 51.5%. `note_public`, `gap`, `fill` and `note` in `countries/in.py`, and the
Catholic, Baptist, Presbyterian and `Christian` entries in `taxonomy/in2011.py`'s `REVIEW`, say so.

**Why.** §11's by-state roll test is now check (d) in the script, printed on every run and not a guard:

| Northeast, 16.2% Catholic | census Christians | drawn | rolls | roll / drawn |
|---|---:|---:|---:|---:|
| Meghalaya | 2,213,027 | 358,592 | 588,465 | 1.64 |
| Nagaland | 1,739,651 | 281,887 | 50,873 | 0.18 |
| Assam with Arunachal Pradesh | 1,584,599 | 256,763 | 567,217 | 2.21 |
| Tripura | 159,882 | 25,907 | 21,162 | 0.82 |

A spread of 12.2 against `CATHOLIC_SPREAD_BAR` (3.0); the same share drew about 553,000 Presbyterians in
Nagaland, and Mizoram's respondents sat inside a share applied only to its neighbours.

**A Northeast split needs a source by state**, not Pew's region again: the church bodies' own rolls state by
state, or the church each tribe follows applied to a count of that tribe, after
`[[feedback_proxy_residual_nameable]]`'s test. Recorded in `queue.md` D.

**Two things the withdrawal exposed.**

- **Check (b) stopped the build on Catholics.** Over the East and South alone the chi-square gives p 0.033
  raw and 0.27 after the design effect: 37.4% and 39.4% cannot be told from one pooled 39.0%. Pooling would
  move the East by 1.6 points, in a split the call keeps as built. The guard now lets regional shares stand
  when no drawn region is further than `SE_AT_FLOOR` from the pooled share: 4.3 points, the standard error
  at the 100-respondent floor from `sources/it.py`'s `N_FLOOR` note. It was added after the failure, so it
  is post hoc. Baptist still passes (p 0.026 after the design effect); Presbyterian has no answer in the
  drawn regions and is skipped. **Replaced the same day by pooling the East and South (§13).**
- **The East by state spreads 3.8, over the bar:** West Bengal 2.29, Jharkhand 1.42, Odisha 1.01, and
  Bihar 3.81 on 129,247 Christians; 2.3 without Bihar. Kept on the supervisor's call, which is why check (d)
  prints and does not stop. The South spreads 1.7 (Andhra Pradesh 3.50, Karnataka 2.23, Tamil Nadu with
  Puducherry 2.09).

**Also fixed.** `taxonomy/in2011.py`'s `REVIEW` text for `Pagan` and `Non Christians` named
`indigenous.indian`; `MAP` files both on `indigenous.indian.khasi`, and the text now says so (`Pagan` adds
that its 38% outside Meghalaya take that node too). The comment above `COLUMNS` names all three Christian
sources. Kozhikode's 36% (§11) is untouched.

**Rebuilt through step 9.** `build_tree.py` clean (728 nodes), `check_mapping.py in` clean, rescattered: 1,207,976
dots at 1:1,000 and 120,785 at 1:10,000, no rings. `check_rollup.py in`: 116,658,813 derived, all rolling
up (4,646,399 fewer than §11, the Northeast's three churches). `coverage.py` ok over 186 countries,
`built_countries.py --check` ok, `check_md.py` clean. Waiting for the supervisor's build tail.

## 13. East and South pooled for Catholics, 2026-09-15 (session `cb8b206e-fixes`)

A supervisor's call on §12, built under a supervisor. `python in_split_christian.py --dry-run` prints every
number below.

**Changed.** §12's post-hoc rule is gone. It had let the East's and South's Catholic shares stand apart
because neither sat more than 4.3 points (`SE_AT_FLOOR`) from their pooled share. Check (b) now applies
the standard treatment. If an answer's drawn regions differ at 0.05 after Pew's design effect (3.7), each
region keeps its own share. If they do not, every drawn region takes one pooled share: the weighted share
of all their Christian respondents together. Catholics over the East and South (37.4 and 39.4 on their
own; p 0.033 raw, 0.27 after the design effect) are now drawn at **39.0%** in both. Baptist still differs
(p 0.026 after the design effect), so it stays regional: 13.0 in the South, and none in the East.
Presbyterian has no answer in either region. Kerala and Mizoram are identical to the file on disk, row for
row. The pooled rows' note reads `structure_geo=pew_region:East+South`.

| drawn region | census Christians | Catholic in §12 | Catholic now | Baptist |
|---|---:|---:|---:|---:|
| East | 3,368,181 | 1,260,157 | 1,314,139 | 0 |
| South, outside Kerala, Lakshadweep, Andaman and Nicobar | 6,769,312 | 2,665,236 | 2,641,132 | 881,460 |

The South's Baptists moved by 26 (from 881,434), from the rounding done alongside the Catholic share.
**9,708,335 of India's 27,819,588 Christians (34.90%) are on a church node**, up from 9,678,431 (34.79%).
The note's rounded 35% holds, and its sentence now reads 39% Catholic in both regions.

**The checks, run on the share drawn.** (c) Roll / survey is 1.48 in the East and 2.24 in the South, a
spread of 1.52 (1.44 before; the bar is 3.0). (d) By state, the East still spreads 3.8: Odisha 0.97,
Jharkhand 1.37, West Bengal 2.19, Bihar 3.66. The South spreads 1.7: Tamil Nadu with Puducherry 2.11,
Karnataka 2.26, Andhra Pradesh 3.53.

**Rebuilt through step 9.** `build_tree.py` clean (729 nodes), `check_mapping.py in` clean. Rescattered:
1,207,976 dots at 1:1,000 and 120,785 at 1:10,000, no rings. `check_rollup.py in`: 116,688,717 derived, all
rolling up (29,904 more than §12). `built_countries.py --check` ok over 187, `check_md.py` clean.
`coverage.py` reports one problem, and it is not India's: `ug christianity.pentecostal` draws dots but is
missing from Uganda's coverage line, while `cb8b206e-ug` is mid-upgrade. Waiting for the supervisor's build
tail.
