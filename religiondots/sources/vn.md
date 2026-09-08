# Vietnam — General Statistics Office, Population and Housing Census

`sources/vn.py` -> `data/normalized/vn.csv`. Boundaries and placement: `sources/vn_geo.md`.
Taxonomy: `taxonomy/vn2009.py`. Scouting note: `sources.md` §11i.

## 1. What is drawn

**The 2009 census, Biểu 7** — *Dân số chia theo thành thị/nông thôn, giới tính, tôn giáo, các
vùng kinh tế - xã hội và tỉnh/thành phố* — 13 named religions plus a not-stated row, over the
6 socio-economic regions and the **63 provinces**. 15,651,467 people, **18.2% of Vietnam** —
plus a computed residual for the other 81.8%, which lands on `unknown`. See §3.

| | |
|---|---|
| publication | *Kết quả toàn bộ Tổng điều tra dân số và nhà ở Việt Nam năm 2009* |
| file | `nso.gov.vn/wp-content/uploads/2019/03/KQ-toan-bo.pdf`, 3.8 MB, 901 pages |
| table | Biểu 7, publication pages 281–312 (PDF pages 289–320) |
| geography | province (63); regions and the nation also published |
| basis | `self_id` |
| tier | `measured` on every row |

## 2. Two censuses, and why the older one is the map

**Both censuses ask. Only the older one says where.**

| | 2019 | 2009 |
|---|---|---|
| table | Biểu 3 | Biểu 7 |
| geography | **national only** | **province (63)** |
| categories | 16 | 13 |
| extent | 1 page of 842 | 32 pages of 901 |
| religious | 13,162,879 of 96,208,984 — 13.7% | 15,651,467 of 85,846,997 — 18.2% |

The 2019 volume gives **ethnicity** a full province × 54-group tabulation over 167 pages and
gives religion one page. Nothing in it says the 2009 volume was finer, and every secondary
compilation cites the newer one — see §5.

**2009 is drawn as it stands and is NOT rescaled to 2019 totals**, which is spec §3.4's escape
clause rather than a departure from it. §3.4's Brazil case had structure and totals at the same
geography, so the rescale was per município and each place kept its own shape. Here the recent
source has no geography at all, so a rescale is **one national factor per religion applied to
all 63 provinces** — it would preserve 2009's spatial pattern exactly and relabel the year.

And the factors are not credible as history:

| | 2009 | 2019 | |
|---|---|---|---|
| Buddhist | 6,802,318 | 4,606,543 | **−32.3%** |
| Hòa Hảo | 1,433,252 | 983,079 | **−31.4%** |
| Cao Đài | 807,915 | 556,234 | **−31.2%** |
| Catholic | 5,677,086 | 5,866,169 | +3.3% |
| Protestant | 734,168 | 960,558 | +30.8% |
| population | 85,846,997 | 96,208,984 | +12.1% |

**Three unrelated traditions falling by the same third in the same decade is an instrument, not
a history.** Nobody has published what changed — the plausible candidates are the enumerator
instruction, the position of the religion question on the form, and a tightening of what
"belonging to" a recognised organisation means — and none of them is documented. Catholicism,
which has an institutional membership record of its own, barely moves; Protestantism, which is
growing, grows. The three that fall are precisely the ones whose adherents have no register to
be on.

**What that costs the map.** Every count drawn is a 2009 count and the map says 2009. India
(2011) and Russia (2012) are drawn the same way, and §3.4 says outright that an old census is a
measurement and its tier is not reduced for age.

**Three bodies recognised after 2009 are therefore in the data and not on the map**: the
Seventh-day Adventists (11,830, recognised 2008 but not tabulated until 2019), the Latter-day
Saints (4,281, recognised 2016) and Phật giáo Hiếu Nghĩa Tà Lơn (401, recognised 2010). They
have a national figure and no geography anywhere.

## 3. The 81.8%, which is the largest fact about this country

**Biểu 7's `Tổng số` row is the religious population, not the population.** An Giang's is
2,025,015 against a census population of 2,142,709. There is no row anywhere, in either census,
at any geography, for the **70.2 million people who answered `Không theo tôn giáo`**.

That answer is not irreligion, and reading it as irreligion would be the biggest error available
on this map. Vietnam recognises a fixed list of religious organisations and the census asks
which of them a person belongs to. Ancestor veneration is close to universal; the village đình,
đạo Mẫu and the mother-goddess rites are everywhere; and the great majority of people who would
describe themselves as Buddhist in conversation are inside the blank rather than in the 4.8
million the 2019 census counts, in a country usually described as around 45% Buddhist by
practice.

**Those 70.2 million ARE drawn, on a node that says we do not know what they are** — Anita's
call, 2026-09-06. `sources/vn.py` computes each province's row as its **Biểu 1 population minus
its Biểu 7 total** and it resolves to **`unknown`** ("Religion unknown"), the seventh member of
spec §6.3a's grey family, added for this and specified by §14.7 for China before there was a
country to hang it on. See spec §6.3a-ii.

**The number is not an estimate.** Both inputs are published figures of the same census on the
same universe, so the residual is the complement of a published partition: the religions plus the
not-stated cell plus the residual equal the population in all 63 provinces to the person, and the
provinces sum to the national 70,195,530. Tier `measured`, like everything else here.

**What it must not be is `unaffiliated`.** §6.3a's grey ramp is a set of *answers*, and answering
"none of your sixteen registered organisations" is not the answer "no religion". Nor is it
`unrecorded`, which is for a source that never asked — these people were asked, and the answer
set was too narrow for their answer to mean what it says. That distinction is the whole reason
the family gained a member rather than reusing one.

**And it is why the country reads at all.** Drawn at 18.2%, Vietnam was 15,646 dots on an empty
outline and §6.12 could label that blank without fixing it. Drawn at 100%, the grey is the
country's own settlement pattern — the two deltas, the coastal strip, the empty uplands — and
every concentration sits legibly on top of it.

## 4. Reconciliation

Everything closes. `sources/vn.py`'s `check()` runs all of it on every build.

| | |
|---|---|
| Biểu 1's 63 provinces sum to | 85,846,997 — the published census population, exactly |
| Biểu 7's national total | 15,651,467 — the printed figure, exactly |
| every province's categories vs its own printed total | **63/63 exact** |
| the 63 provinces vs the national row, per category | **14/14 exact** |
| each socio-economic region vs its provinces, per category | **6/6 exact, all 14 categories** |
| no province's religious total exceeds its population | 63/63 |
| the 2019 table's 17 rows sum to | 96,208,984 — exactly |
| 2019 internal identities (total = M+F, total = urban+rural) | 34/34 |
| religions + not-stated + computed residual = population | **63/63 exact** |
| the province residuals sum to | 70,195,530 — the national figure, exactly |
| every province's residual is non-negative | 63/63, smallest 117,694 — An Giang, at 94.5% affiliated |

**Non-response is 30 people nationally** — the smallest on this map by three orders of
magnitude, and small enough that it is worth saying it is real rather than a rounding artefact:
30 individual people in **9 provinces**, the largest concentration being 13. Reported, not
filled, not drawn (§3.5).

**The region check is the one worth having and it was not free.** The census prints the nation,
then all six regions, then all 63 provinces — three flat blocks with **no marker of which
province is in which region**, so §12's parent/child test needs the composition from somewhere
else. `vn.py`'s `REGION_OF` is GSO's published six-region standard written out by hand, and the
arithmetic is what verifies it: a province in the wrong region breaks 28 equations at once.
It reconciles to the person, which is evidence that the transcription is right.

## 5. Four traps, and the third is the nastiest

### 5a. The two volumes use different thousands separators, and one of them is unparseable

2009 writes `85.846.997`. 2019 writes `96 208 984`. Same office, same series title, same layout
— and the space-separated one **cannot be read from the text layer at all**, because the digit
groups are indistinguishable from separate numbers.

`Tôn giáo Baha'i 2 153 1 089 1 064 841 419 422 1 312 670 642` has a second reading in which
`841`, `419` and `422` are three values rather than one, and an anchored nine-number regex finds
it: the greedy parse fails on the count, backtracks, and lands on the wrong split with no error.
So `read_2019()` reads **geometry** instead — the nine columns are right-aligned on a 57pt grid,
digit groups inside one number sit 2.6pt apart and adjacent columns 11.8pt apart — and
calibrates the grid off the national row rather than hard-coding it.

Two details that cost time inside that: the columns are right-aligned, so a **two-digit cell
starts 5pt further right than a three-digit one** and a left-edge grid misses it; and the label
cutoff has to be 275pt and not 250, because the rightmost label word on the page is `đạo`, the
last syllable of `Giáo hội Phật đường Nam Tông Minh Sư đạo`, at x=253.

### 5b. A wrapped label has fragments on BOTH sides of its figures

The Latter-day Saints row prints `Giáo hội Các thành hữu Ngày sau của` above its numbers and
`Chúa Giê su Ky tô Việt Nam (Mormon)` below them. Attaching trailing fragments — gy.py's rule —
gives the previous category the LDS opening and the next category the LDS closing, so **three
labels come out wrong from one wrapped row**, all of them plausible-looking strings. Each
fragment goes to the nearest record by y instead; the two halves are 5pt from their own figures
and 18pt from their neighbours', so the assignment is not close.

### 5c. THE PDF'S TEXT LAYER IS NOT IN THE SAME UNICODE NORMAL FORM AS THE SOURCE FILE

`Giá o hội Cơ đố c Phục lâm Việt Nam` comes back with **`á` and `ố` decomposed** — base letter
plus combining acute — while every other Vietnamese label on the same page is precomposed. The
two affected syllables are exactly the two the typesetter split across glyph runs.

Compared as bytes, that string is unequal to a visually identical literal in `vn2009.py`. So the
category resolves to nothing, `countries.py` drops it, and **the string is identical in a
terminal, in a diff, and in code review**. `tools/check_mapping.py` caught it, which is what that
tool is for; nothing else would have.

Fixed in two places on purpose: `vn.py::_txt` normalises to NFC on write so the CSV is canonical,
and `vn2009.py::_key` normalises on read so a future re-import is safe. **This will recur** — any
office whose PDF pipeline splits a glyph run can produce it, and it is invisible until a mapping
silently loses a category.

### 5d. The census spells its own provinces two ways

Vietnamese admits two tone-mark placements on `oa`/`oe`/`uy`, and GSO uses both: Biểu 1 writes
`Hoà Bình`, `Thanh Hoá` and `Khánh Hoà` where geoBoundaries writes `Hòa Bình`, `Thanh Hóa`,
`Khánh Hóa`. An exact-string join fails on exactly three provinces and looks like a vintage
problem. See `vn_geo.md`.

## 6. What Vietnam is worth drawing for

**The Mekong Delta produced its own Buddhist movements and the census counts four of them
separately.** Bửu Sơn Kỳ Hương (1849), Tứ Ân Hiếu Nghĩa (1867), Hòa Hảo (1939) and Hiếu Nghĩa
Tà Lơn are **one documented descent**, and this is the first source anywhere on this map to hand
`branches.py` a whole lineage rather than one more member of an existing group — which is why
they are flat siblings under `buddhism` in BRANCHES and a lineage group in LINEAGE (spec §2.1's
two relations, and the cleanest example of the split in the file).

**And they are the most concentrated religions on the map.** An Giang alone holds 65% of the
country's Hòa Hảo, 84% of its Tứ Ân Hiếu Nghĩa and 76% of its Bửu Sơn Kỳ Hương, and is **94.5%
religious** against a national 18.2% — the largest gap between a province and its country
anywhere here. Sơn La is 0.4%.

**Caodaism arrives at 807,915** against the 677 Australians the node was created for, and Tây
Ninh is 35.6% Caodaist.

**Cham Balamon gets a node**: 56,427 people, 72% of them in Ninh Thuận, and the last living Hindu
tradition of Southeast Asia's Indianised kingdoms.

**Two Minh đạo bodies** — Minh Sư Đạo (709) and Minh Lý Đạo (366) — the Vietnamese Xiantiandao
line, and among the traditions Caodaism drew its practice from. Both are rings, not dots.

### What the categories cannot show

- **Buddhism is one word and holds two schools.** Trà Vinh is 49.7% Buddhist and Sóc Trăng 25.7%,
  and those are Khmer Krom, whose Buddhism is Theravada; the northern and urban Buddhism is
  Mahayana. Perhaps a million people. Mapping the cell to `buddhism.mahayana` would be right
  about most of it and would erase them, so it sits on the undivided parent (§2.4, lk2024.py's
  call for Sri Lanka).
- **Islam holds two Cham communities that are not the same religion to the people in them.** The
  Sunni Cham of An Giang, Tây Ninh and Saigon, and the **Bani** of Ninh Thuận and Bình Thuận,
  whose practice is a thousand-year-old Cham localisation with a hereditary priesthood and a
  three-day Ramadan. The census gives Balamon its own row and folds Bani into Islam, so one half
  of the Cham religious system has a colour and the other does not.
- **Protestantism is one cell and is two geographies.** Đắk Nông 10.3%, Gia Lai and Đắk Lắk 8.6%
  is Montagnard; Điện Biên 7.5% and Lai Châu 7.1% is Hmong. The census counts registered
  congregations, and the unregistered house churches — the ones actually under pressure — are
  not in it.

## 7. §14, and where the line was drawn

Vietnam is a State Department Country of Particular Concern. The persecuted communities are
independent Cao Đài, independent Hòa Hảo, the Unified Buddhist Church, Khmer Krom Buddhists, and
Hmong and Montagnard Protestants.

**Drawn at province, which is the state's own published resolution, and no finer.** §9t's
Pakistan reasoning transfers: omitting a persecuted group performs the erasure it means to
prevent, and this census's Protestant row is the *registered* churches — the house churches
being raided are the people this instrument does not see, so drawing it neither reveals them nor
claims to count them. §14.2's reflect-vs-reveal test passes on the source: every number is
Hanoi's own published table about its own territory, and religious organisations in Vietnam are
registered by the state.

**§14.4's resolution clause is what fixes the tier.** *"For a persecuted group, no resolution
finer than the state's own publication."* Vietnam publishes religion by province in 2009 and
nationally in 2019, so province is inside the rule.

**The district route is outside it and is not taken.** IPUMS International holds the 2019 census
as an 8.5% sample — 8,236,773 person records — with `RELIGION` and `GEO2_VN`, which is district,
about 700 units. That is two orders of magnitude finer than anything Vietnam has published, and
whether a state's microdata release to a foreign archive counts as "the state's own publication"
is a question this project has not had to answer. It is also blocked on §10a's dead IPUMS
account. **Both facts should stay true independently**: if the account ever opens, the tier
question is already settled rather than settled by whatever arrives.

## 8. Access

Both volumes are one GET each and need no account, no key and no browser.

```
python sources/vn.py --fetch      # 13.3 MB, two PDFs
python sources/vn.py              # parse and reconcile
python sources/vn_geo.py --fetch  # geoBoundaries ADM1 + Kontur VN, ~17 MB
python sources/vn_geo.py
```

**`gso.gov.vn` times out on connect and `nso.gov.vn` serves the identical paths.** The office
renamed and kept the WordPress install. `sources.md` §9q's sibling-host rule, fifth sighting, and
the reason §11d wrote Vietnam off in the first place.

**The search surface is the WordPress REST API**, not the site's own search:
`GET /wp-json/wp/v2/search?search=kết quả toàn bộ&per_page=50` returned both census volumes as
its top two hits after the site search and two web searches had missed the 2009 one. Search the
publication's name, not the variable — `search=tôn giáo` returns nothing useful.
