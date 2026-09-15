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
  non-census source.
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
