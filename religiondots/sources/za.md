# South Africa — Stats SA, Community Survey 2016 person microdata, 213 local municipalities

**Redrawn 2026-09-09** from 9 provinces to **213 local and metropolitan municipalities**, from
the CS 2016 person file, DataFirst catalogue 611: 3,328,867 records, 55,653,654 weighted
people, 25 drawn categories, ~258,000 people per unit. `sources/za.py`, `sources/za_geo.py`,
`sources/za_grid.py`, `sources/za_profiles.py`, `taxonomy/za2016.py`, and the `za` entry in
`countries.py`.

**Drawn 2026-09-08** at nine provinces from the nine published CS 2016 provincial profile
reports. That build is not deleted and it is not wasted: `sources/za_profiles.py` is its
parser, and the nine published tables are now **the reconciliation**. `sources/za.py` refuses
to write `data/normalized/za.csv` unless every published province cell reproduces from the
microdata, which 215 of 216 do to within a person. §4 is the 216th.

The headline is unchanged and its geography is not. `christianity.africaninstituted` holds
**14,158,461** South Africans, 54.67% of that node across seven countries. At nine provinces
that ran 15.83% of Western Cape's Christians to 50.77% of Limpopo's. At 213 municipalities it
runs **4.64% of Kai !Garib's to 75.56% of Mthonjaneni's**, and in Big Five Hlabisa and Mfolozi
it is **61.98% and 61.62% of every answer given**, not of Christians.

Every figure in this file reproduces from `data/normalized/za.csv`. Percentages described as
*of Christians* are of that unit's fourteen denominations plus its not-reported row; all others
are of that unit's drawn answers. Superlatives are people, not dots.

---

## 1. Why the survey and not the census

`queue.md` had `za` closed on *"Christianity undivided; behind a DataFirst account"*, and
sources.md §11ag had already corrected half of that: **Census 2022 is not walled.** Statistical
release P0301.4 section 2.9 table 2.10 gives religion for all nine provinces over eleven
categories, openly downloadable. What it does not give is any split of `Christianity`, which is
83.6% of the country in one cell, and it gives nothing below province.

**Community Survey 2016 has both, and now at 213 units.** The `ReligionBelief` question carries
the eleven substantive categories; a second question, `Christianity`, asked only of Christians,
splits them fourteen ways: Catholic, Anglican/Episcopalian, Baptist, Lutheran, Methodist,
Presbyterian, Pentecostal/Evangelistic, African Independent Church/African Initiated Church,
Jehovah's Witness, Seventh Day Adventist, Mormon, Reformed church, Just a Christian/
non-denominational, Other. Several carry an exemplar list, preserved in the CSV's note column,
and the African Independent row's names *Zion Christian Church; Apostolic Church; African
Nazareth Baptist Church/Shembe*.

So 24 published categories against the census's 12, at 213 units against the census's 9. The
cost is six years and a survey rather than a census. §3.1 forbids mixing them and §3.9 says
category detail and spatial detail trade off inside one source; here they did not have to,
which is unusual.

**One thing the ask got wrong and it is worth writing down.** The ask, and the brief that came
back from it, said the microdata's `Christianity` variable has **15** substantive denominations
against the 14 published, adding *Just a christian/non-denominational*. It does not. The
variable's 17 value labels are the **fourteen published ones**, which already include *Just a
Christian/non-denominational*, plus `Do not know` (15), `Not applicable` (88) and `Unspecified`
(99). There is no new category, no new node, and no mapping decision to make; `taxonomy/
za2016.py`'s `MAP` is character for character what the province build shipped. `sources/za.py`
asserts that each codebook label folds onto a published one, so a re-release that added a
fifteenth would fail the build rather than pass unnoticed.

---

## 2. The municipal tier: what it takes, what it costs, and what the survey says about it

### 2.1 Sampling adequacy, in CS 2016's own words

A survey drawn at a fine tier has to say why that is not over-reach, and the answer here is the
sample design rather than an assertion. Report 03-01-07 §1.2.2:

> The sample design for CS 2016 was a stratified single-stage sample design. At enumeration
> area (EA) level, **all in-scope EAs were included in the sample** and a sample of dwelling
> units was taken within each EA (i.e. there was no subsampling of EAs).

Every enumeration area in the country is in the sample. That is the fact that matters: no
municipality is represented by a neighbour's households, and there is no cluster-selection
stage whose variance would have to be argued about. §1.2 of the same report says what the
survey is for, and it is this tier: *"This household-based survey is one of the few available
data sources providing data at municipal level."* §1.2.3 says why: *"Through the consultation
process, it became clear that there is an increased demand for data at municipal level."*

### 2.2 How thin it gets, stated rather than assumed

| | |
|---|---|
| unweighted records per municipality | min **529** (Prince Albert), median 7,890, max 224,976 (Johannesburg) |
| Kish effective n per municipality | min 449, median 7,102, max 186,817 |
| weighted people per municipality | min 8,895 (Laingsburg), median 123,419, mean 261,285, max 4,949,347 |
| non-empty (municipality, category) cells | 4,472 |
| of those, resting on fewer than 10 records | 882, holding **0.107%** of the people drawn |
| heaviest cell resting on ≤5 records | 317 people (Stellenbosch, Judaism, 2 records) |

So the weight artefact this map has to watch for elsewhere is absent: no cell of any size is
one household multiplied up. **Every row of `data/normalized/za.csv` carries `cell_n`**, the
unweighted records behind that cell, and `unit_n` for its municipality. That is there so a
reviewer can check any figure quoted in `taxonomy/za2016.py`, in `note_public` or here without
opening the 498 MB .dta. Superlatives in `note_public` are taken from the 175 municipalities
over 50,000 people; `za2016.py` quotes some smaller ones and gives `n=` every time.

### 2.3 Two municipalities return something the province tables hid, and neither is corrected

The fine tier exposes what looks like fieldwork rather than geography. Both are large, both
rest on hundreds of interviews, and nothing here can establish or repair either.

**uPhongolo (KZN262), rural KwaZulu-Natal, 141,248 people.** `Atheism` is **4.76% of its
answers** against 0.13% in KwaZulu-Natal and 0.096% nationally, on **384 of its 8,121
records**. That is 6,630 people, **12.6% of every atheist counted in South Africa**, in a
municipality holding 0.25% of the population. Cape Town holds 30.6% and Johannesburg 15.6%,
which is what a reader expects; uPhongolo third is not. Its Catholic (1.35% against a
provincial 9.20%), Methodist (0.28% against 3.41%) and Anglican (0.40% against 1.82%) cells are
all far below the province at the same time, so the excess is drawn from the named
denominations rather than from the no-religion cell. *(§12 contradicts this last sentence
from the file: uPhongolo's no-religion cell is short by more than its atheism cell is long,
and its overall Christianity share is above the province's, not below.)*

**Swellendam (WC034), Western Cape, 40,209 people.** `Just a Christian/non-denominational` is
**57.34% of its answers** against 7.06% of its province's answers, on **965 of its 1,685
records**. Its Pentecostal (4.08% against 15.90%), African Independent (3.62% against 12.94%),
Catholic (0.78% against 6.59%), Methodist (0.34% against 5.29%) and Muslim (0.30% against
5.64%) cells are all far below the provincial rate at once.

Both have the shape of one enumeration team's habit with a CAPI answer list: a cell far above
its province, with the categories adjacent to it on the card far below by about as much. Both
are **drawn as returned**. §14.4 rule 1 forbids inventing a magnitude and §3.5 says undercounts
are marked rather than filled; there is no third option that is not a guess about which
households were mis-coded. They are named in `note_public`, in `taxonomy/za2016.py` under
`Religion: Atheism` and `Christian: Just a Christian/non-denominational`, and here.

**The province tabulation could not have shown either**, which is the general point: KwaZulu-
Natal's provincial atheism figure is 0.13% and looks unremarkable, and Western Cape's
non-denominational figure is 8.64% of Christians and looks like a city effect. Averaging over
six million people hides a fieldwork artefact as effectively as it hides a real place.

### 2.4 Which vintage, and the trap under it

**Every record carries its geography twice.** `PR_CODE_2011` / `DC_MDB_C_2011` /
`MN_CODE_2011` are the demarcation in force when the survey was enumerated (6 March 2016) and
`*_2016` are Stats SA's recode of the same households onto the **August 2016** demarcation,
which merged municipalities and cut the local tier from 226 to 205. So `MN_CODE_2011` has 234
municipalities and `MN_CODE_2016` has 213.

**The 2016 set is drawn**, for one reason that decides it: OCHA's COD-AB ADM3 for South Africa
is that same 213, `valid_on` 2020-11-09, and it joins to the microdata **213/213 both ways on
the MDB municipality code**. There is no boundary layer for the 2011 set on disk, and the 2011
set's only advantage is 10% more units. If someone ever wants the finer one, note that
COD-AB's older ZAF releases would have to be found and that this build's three independent
join checks (§6) would have to be rewritten, because the 2011 district codes do not match the
current ones everywhere.

**And there is a silent failure sitting right beside that choice.** Stata truncates value-label
set names to eight characters, so the file carries label sets called `MN_CODE` (234 entries)
and `MN_COD_A` (213), and **neither is named after the variable it belongs to**. Pick the wrong
one and 213 codes still resolve, to *different municipalities*, and every national and
provincial total still reconciles, because the codes are a subset either way. `sources/za.py`
picks by size and then confirms on the label **text**, which differs: the 2016 set writes
`WC011 : Matzikama` with a space before the colon and the 2011 set writes `WC011: Matzikama`
without one. It refuses to build if the two ever stop being distinguishable that way.

### 2.5 Nine provinces was the open ceiling, and that record stands

None of the sweep below is superseded; it is why the account was worth asking for. Everything
in it was opened and read, not inferred from a search result.

| what | verdict |
|---|---|
| **Report 03-01-84, *Cultural dynamics in South Africa*** — named in §11ag as the likely home of a finer cut | **Province-only and COARSER.** Religion is chapter 4; table 4.1 is *by religious denomination and province*, censuses 1996/2001/2022, over **8** categories. Its own chapter 2 goes finer for language (table 2.3 is by metropolitan area), so the report had the geography and did not use it for religion. |
| **Census 2011** | **Asked no religion question at all.** Report 03-01-84 p.48 says so in terms. That closes Wazimap and every Census 2011 municipal product, and it is why the UNSD oracle jumps 2001 → 2022. |
| **Census 2022 provincial profiles** | Religion is one table, **province total only**, 11 categories, Christianity undivided. |
| **Census 2022 Municipal Fact Sheet** | 41 pages, zero matches for religion, Christian, Islam or Hindu. |
| **Census 2001 *Census in Brief*** | 111 pages, zero matches for religion, Christian, Zion, Muslim or Hindu. |
| **Stats SA's Census 2022 dissemination API** — keyless, no login | **Live and useful, and religion is not on it.** It serves 24 topics down to **Main Place**. Grepping the whole 19 MB front-end bundle for `religio` gives **0 matches**. |
| **HDX**, **USCB country geodatabases** | No South African religion dataset. |
| **SuperWEB2**, `superweb.statssa.gov.za` | Stats SA's own cross-tab tool, geography reportedly to ward. **Still unverified**: the host is behind the Imperva wall for both curl and WebFetch and needs a registered account. It is the one open question in this table, and it now matters much less. |

**The pattern was the finding**: in the Census 2022 provincial profiles essentially every other
variable is tabulated *"by district and local municipality"* and religion alone is not. That
reads as a decision about what to *print*, and the microdata shows it was only that: the
municipality codes were on the same records all along.

### 2.6 What the account actually cost

Recorded because `[[reference_gated_data_last_resort]]` says a wall should be priced rather
than described. DataFirst registration turned out to be **self-service**: name, institution,
email, password, no approval step, and catalogue 611 is a public-use file needing only a login
plus a signed confidentiality declaration. It is not the Korean-ID wall and not IPUMS. The
download stalled twice at ~83 MB and completed on a later attempt. This is the cheap tier of
wall, and the lesson for the next country is that "behind an account" in `queue.md` is worth
one look before it is worth a year.

Census 2022's 10% sample (DataFirst 982) is behind the same account and is more recent, but
leaves Christianity undivided, so it would trade the whole denominational split for six years
of currency. 611 is the file to take.

---

## 3. The two Stats SA releases disagree past what six years explains

Unchanged from the province build and still the most important thing here for anyone who wants
to update the country. The category *labels* match verbatim between CS 2016 and Census 2022
table 2.10, which makes them look interchangeable. The distributions do not:

| category | CS 2016, % of answers | Census 2022, % of answers |
|---|---|---|
| Christianity | 79.0 | 85.3 |
| **No religious affiliation/belief** | **10.9** | **2.9** |
| **Traditional African religion** | **4.5** | **7.8** |
| Other | 2.7 | 1.0 |
| Islam | 1.6 | 1.6 |
| Hinduism | 1.0 | 1.1 |

Islam and Hinduism, the two categories a respondent is least likely to be unsure about, are
stable to a tenth of a point. The three that move are the ones whose boundary is a matter of
how the question is put. **That is an instrument difference, not a trend**, and it is §3.1a's
finding. Census 2022 figures appear in this project only as the contrast above, in
`taxonomy/za2016.py`'s REVIEW entries and in one sentence of `note_public`.

### 3.5 Which way the hole leans, re-run at 213 units

707,295 people, **1.2709%** of the survey, answered `Do not know` (704,355) or nothing at all
(2,940) and are not drawn. That is not new; what is new is that it is now **two rows per
municipality in `data/normalized/za.csv`** rather than eight printed footnotes plus one
inferred by difference, so `tools/gap_share.py za` computes the share off the file (it refused
this country before, correctly) and the §3.5 lean can be measured where it is drawn.

| | province tier, n=9 | **municipality tier, n=213** |
|---|---|---|
| non-response range | 0.417% (Northern Cape) to 2.017% (Gauteng) | **0.000% (five municipalities) to 8.350% (eDumbe)**, median 0.672% |
| r vs `No religious affiliation/belief` share | +0.7183 | **+0.3063** |
| exact / permutation p | 0.0280 | **0.00005** (20,000 permutations) |
| leave-one-out range | not computed | **+0.2649 to +0.3273** |
| r vs Christianity share | −0.6273 | **−0.2552**, p = 0.0002 |

**The correlation halved and the evidence got very much stronger, which is the expected shape
and is worth stating plainly** ([[reference_check_needs_power]]). Nine points give a large,
noisy r that one province could move; 213 points give a smaller one that no single unit moves
by more than 0.02. Weighting by population rather than by unit puts it at **+0.3986**, so the
lean is larger among the people actually drawn than among the municipalities.

The direction is Serbia's, unchanged: the people dropped are disproportionately from the least
religious places, so **every share drawn for South Africa is slightly more religious than South
Africa is**, including the 79.0% Christian and the 10.86% with no affiliation. Nothing corrects
for it; `note_public` says so.

---

## 4. North West's Report 03-01-11 is defective, the microdata proves it, and 336,482 people move

This is the one published cell that does **not** reproduce, and it is the reason the
reconciliation is worth running rather than merely worth claiming.

**The defect.** Report 03-01-11's table 2.10b prints fourteen denomination rows summing to
3,072,039 against its own printed total of 3,408,521, and its percentages sum to 90.1 rather
than 100.0. So 336,482 people, 9.9% of that province's Christians, are in no denomination row.
Its `Other` cell reads **21 873**, character for character the `Do not know` figure in that
same table's own footnote.

**The province build had two stories and neither closed, so it guessed at neither.** One: the
`Other` row was mis-set to the footnote's figure. Two: North West is the only province whose
footnote lists a third exclusion, `Not applicable` (318,029), which is 94.5% of the shortfall,
and if the rows were built excluding it while the total was not, these people would not be
Christians at all. The 336,482 were parked on bare `christianity` as `Denomination not
reported`, where a reader could see them.

**The microdata settles it to the person.** North West's `Christian: Other` comes out at
**358,355**, which is 21,873 + 336,482 exactly. Story one is right; story two is wrong. Those
people are Christians of another denomination and are now drawn as such. `sources/za.py`
asserts the difference is exactly `NW_SHORTFALL` and `sources/za_profiles.py` asserts the
printed `Other` cell is still 21,873, so a reissued report fails the build instead of quietly
changing the map.

**What moved.** North West's `Other` goes from 0.64% of its Christians to 10.45%, inside the
4.43%–17.25% the other eight run, and its `Denomination not reported` goes from 10.45% to
**0.64%**, inside a national range of 0.20% (Limpopo) to 0.91% (Western Cape). Both cells were
wrong in opposite directions and both are right now. Nationally `Christian: Other` goes
3,509,156 → **3,845,643** and `Christian: Denomination not reported` 567,039 → **230,558**.

### 4.1 `Denomination not reported` is now a measurement rather than a subtraction

It used to be table 2.10a's Christianity cell less table 2.10b's rows. It is now the
`Christianity` question's own `Do not know` (227,585) plus `Unspecified` (2,976), counted per
person: 230,558 after rounding. The province build's careful footnote arithmetic — exact in
four provinces, within one person in three, unverifiable in Western Cape because Report
03-01-07 prints no note — was right everywhere it could be checked, and it is now redundant.

### 4.2 The `Christianity` question is nested and that is asserted

Code 88, `Not applicable`, is **12,229,937** people and is exactly the non-Christian
population (55,653,654 − 43,423,717). It is not emitted; doing so would count every
non-Christian twice. `sources/za.py` asserts both that identity and that no non-Christian
record carries a denomination.

---

## 5. Access — the wall is real and it is not where §11ag put it

`statssa.gov.za` and its subdomains sit behind Imperva. **HTML pages are unreachable**:
`?page_id=` and `?p=` listings return a ~1 KB `_Incapsula_Resource` stub for curl *and* for
WebFetch, and a browser User-Agent does not help.

**But the PDFs themselves are not walled, and §11ag's account of why curl fails is wrong.**
What stops a plain `curl` on `cs2016.statssa.gov.za` is the **TLS chain**, not the bot wall:
the host presents a self-signed intermediate and curl exits 60 before it sends the request,
which reads exactly like a dead host. Relax the certificate check and the same URL returns the
real PDF at full size. `sources/za_profiles.py --fetch` does exactly that and then checks the
`%PDF` magic and the `%%EOF` trailer, because a 200 is not a download
(`[[reference_pdf_truncated_at_source]]`).

The nine profiles are static WordPress uploads at
`https://cs2016.statssa.gov.za/wp-content/uploads/2018/07/<Province>.pdf`, where `<Province>` is
`WesternCape`, `EasternCape`, `NorthernCape`, `FreeState`, **`KZN`** (not `KwaZulu-Natal`,
which 404s), `NorthWest`, `Gauteng`, `Mpumalanga`, `Limpopo`.

**The microdata cannot be fetched by a script at all** and `sources/za.py` says so rather than
pretending: catalogue 611 needs a DataFirst login and a signed declaration. `data/raw/za/cs-
2016-person.dta` is 497,704,785 bytes and the script asserts its record count.

---

## 6. Boundaries and placement

**The join is a CODE join and that is the single biggest structural improvement over the
province build.** The CS 2016 microdata and OCHA COD-AB carry the same MDB municipality codes
(`WC011`, `CPT`, `LIM345`), so `sources/za_geo.py` matches 213/213 both ways with nothing spare
on either side. The province build joined on nine strings and needed an area rank test to have
any independent evidence at all, because every province's figures reconciled whichever polygon
they were paired with.

A code join still gets independent evidence, because its failure mode is not a missed match but
a match to the wrong polygon after a file is reissued with codes shifted. Three quantities, none
of them the join key:

1. **District**, 213/213 over 52 districts. Stats SA's `DC_MDB_C_2016` is carried through the
   CSV's note column and checked against COD's `adm2_name1`.
2. **Province**, 213/213 over 9. COD misspells one: `adm1_name` reads **`Nothern Cape`**, no
   `r`. An explicit alias handles it rather than a fuzzy matcher.
3. **Name**, 210 of 211 testable. The one that differs is **MP326**, `City of Mbombela` against
   COD's `Mbombela`, the same place under its official and its short name, and the exception
   list is asserted to contain exactly that.

Together they close the within-district swap that (1) and (2) would miss.

**Two names come from COD rather than from Stats SA, against §12's Chile rule, and both are
written out in `sources/za.py`'s `NAME_OVERRIDE` with the reason.** `LIM345`'s Stats SA label
is the literal word **`New`**: the municipality was created in the August 2016 demarcation out
of Thulamela and Makhado and had no name when the file was coded. It is **Collins Chabane**.
`NC067`'s label reads `Kh+ói-Ma`, and that is not an encoding artefact of the reader — those
are the bytes in the .dta. It is **Khâi-Ma**, in the Namakwa district. Both are recorded in the
`note` column of every row of theirs, so `data/normalized/za.csv` shows a reader that a name
was changed.

**Placement is Kontur's 400 m population grid** (`kontur_population_ZA_20231101`), and it
matters less at 213 units than it did at 9, which is the right way round. It is still needed:
Dawid Kruiper is 44,231 km² with 107,161 people and Mandeni is 545 km² with 147,808. 1,760
hexes (124,999 people, 0.207%) fall outside every municipality — the border overrun plus, mostly,
the Lesotho enclave and the Eswatini salient. Dropped and reported.

**And at 213 units the join can be tested rather than assumed**, which nine polygons could not
support. Log-log correlation between Kontur's population per municipality and the people drawn
there is **r = 0.9815 against a best of 0.2253 over 500 shuffles** of the same numbers over the
same polygons. The per-unit ratio band is 0.53 to 8.80, median 1.02, with 2 of 213 outside
0.5–2.0.

### 6.1 Kontur is badly wrong in two municipalities and it changes nothing

Worth writing down because the ratio band prints it and it looks alarming. **Matatiele (EC441)
reads 8.80x** — Kontur models 1,919,300 people in a 4,352 km² municipality where the survey
counts 218,000 — and **Mtubatuba (KZN275) reads 3.59x**. Checked and not a bug on this side:
the Kontur file has no duplicate H3 cells, the polygons are the right size and shape (the 213
sum to 1,220,813 km² against South Africa's 1,221,037), Matatiele's polygon has one part and no
Lesotho hole to leak through, and the dense hexes sit on Matatiele town and Maluti, where the
people actually are. Kontur simply overestimates there.

**It does not reach the map.** `_KonturHexWeighter` normalises inside each unit: a
municipality's dot count comes from the survey and the grid only decides where inside it the
dots land. An inflated unit total cancels. What it does mean is that this layer must not be read
as a population estimate for those two municipalities.

---

## 7. What the map shows

79.03% of answers are Christian and the fourteen-way split is the point, but the geography is
now the story rather than the categories.

| | national | of Christians | low (units over 50k) | high |
|---|---|---|---|---|
| African Independent Church | 14,158,461 | 32.61% | Kai !Garib 4.64% | **Mthonjaneni 75.56%** |
| Pentecostal/Evangelistic | 8,483,686 | 19.54% | Mthonjaneni 5.85% | Greater Giyani 47.71% |
| Other Christian | 3,845,643 | 8.86% | Umhlabuyalingana 0.38% | Dawid Kruiper 33.49% |
| Catholic | 3,778,328 | 8.70% | Bergrivier 0.69% | Dr Nkosazana Dlamini Zuma 51.94% |
| Methodist | 2,777,934 | 6.40% | Greater Giyani 0.16% | Ntabankulu 33.82% |
| Just a Christian | 2,501,373 | 5.76% | Walter Sisulu 0.23% | *Swellendam 57.34% of answers, §2.3* |
| Reformed church | 2,350,855 | 5.41% | **Mfolozi 0.21%** | **Bergrivier 41.89%** |
| Anglican/Episcopalian | 1,765,279 | 4.07% | Maphumulo 0.02% | Hessequa 33.00% |

**The African Independent Churches are the country and the municipal tier says so differently.**
At province the range was a factor of 3.2 and could be read as a homeland/settler gradient. At
municipality it is a factor of 16 of Christians, and as a share of *all* answers it reaches
**61.98% in Big Five Hlabisa and 61.62% in Mfolozi**: in those places the AICs are not the
largest denomination, they are most of the population. The clusters are northern KwaZulu-Natal,
which is Shembe's Nazaretha country, and Limpopo, where the Zion Christian Church has its
headquarters at Moria in the Capricorn district.

**The mission denominations are a nineteenth-century map and nine provinces averaged it away.**
Methodists are 33.82% of Ntabankulu's Christians and 33.41% of Umzimvubu's, both in the old
Transkei, against 0.16% in Limpopo's Greater Giyani; the Eastern Cape as a whole reads 15.18%,
which is that Methodist Transkei averaged with a coast that is Anglican and Reformed. The Dutch
Reformed family is 41.89% of Bergrivier's Christians and 39.11% of Matzikama's, the Swartland
and the Olifants River valley, against 0.21% in Mfolozi: a factor of **200** where the province
tier showed 9. The Scottish Presbyterian mission is 11.13% of Raymond Mhlaba, which contains
Alice and Lovedale, and effectively zero across three KwaZulu-Natal municipalities.

**Pentecostal/Evangelistic was the flattest large category at province, 15.96% to 25.69%, and
it is not flat at all.** 5.85% of Mthonjaneni to 47.71% of Greater Giyani, with Collins Chabane
42.02% beside it. A factor of 8 rather than 1.6. What survives is the comparison: 8 is still
much less than the AIC's 16 or the Reformed row's 200, so it is genuinely the most evenly
spread of the large Christian answers.

**The non-Christian tail is cities and one coast.** Hinduism is 9.23% of KwaDukuza's answers
and 8.98% of eThekwini's and near zero across most of the country, which is the population
descended from the indentured labourers brought to the Natal sugar estates from 1860. Islam is
8.41% of Cape Town, 3.59% of eThekwini and 3.34% of Johannesburg against nothing at all in much
of the Eastern Cape. Judaism is 0.48% of Johannesburg and 0.32% of Cape Town, and those two
hold 73.0% of every Jewish answer in the country (corrected from 78% by review 2026-09-09,
§12; eThekwini's 7.3% takes the top three to 80.2%).

**No religious affiliation is 10.86% and its geography is still not the one a reader expects.**
Okhahlamba 32.67%, Ephraim Mogale 27.66%, Blouberg 26.78%, all rural, against Johannesburg
14.73% and Cape Town 7.79%. Limpopo and KwaZulu-Natal are both the most African-Independent
provinces and among the least affiliated, so this is not a secularisation gradient. The reading
offered at province still stands and still is not resolved: where church membership is the
dominant idiom, someone outside a particular church may answer `none` rather than name an
ancestral practice.

**African traditional religion is 4.47% and is a floor**, for the reason sources.md §11b gives
for the continent. The floor is much higher in the KwaZulu-Natal midlands than nine provinces
could show: 31.35% of Maphumulo, 30.78% of Mkhambathini, 26.03% of Msinga, against a KwaZulu-
Natal figure of 7.36%.

**The most and least Christian municipalities are worth one line**, because a dot map reads as
composition: Kareeberg is 99.68% Christian and Msinga 46.61%.

### 7.1 One taxonomy node, and it was added by the province build

`other.za`, *Other religion (South Africa)*, 1,482,207 people, 2.70%. A genuine tail rather
than a store cupboard: Islam, Hinduism, Judaism, Buddhism, Bahaism and Traditional African
religion all have their own boxes. At province it was unusually flat and nothing was inferred
from it; at municipality it runs 12.07% of Nqutu's answers to 0.14% of Mfolozi's, which is a
real geography that still names nothing (§9r's Chittagong rule). Nothing else was added by
either build.

---

## 8. Ethics (§14)

**No new ask, and the resolution question was already Anita's to decide and was decided.**
`ask/answered/002-za` put the tier to her and her ruling on 2026-09-09 was that South Africa is
to be redrawn at local-municipality level. Recorded here because a reviewer will and should ask
whether §14.4 rule 2 was cleared.

The argument, for the record. Religion is asked on South Africa's own survey, published by
Stats SA itself, and no group on this card is one the South African state or anyone else
persecutes. §14.4 rule 2's ceiling is the state's own publication, and the question is whether
tabulating a variable at a geography the office released on the same record but did not itself
print is finer than that ceiling. It is not, and the file's own design says why: Stats SA
released `ReligionBelief`, `Christianity` and `MN_CODE_2016` on the same person record under a
public-use licence, its own report calls the survey a source of *data at municipal level*, and
**it withheld the enumeration area**. The finest geography in catalogue 611 is the
municipality; `EA_GTYPE_C` is a settlement type (urban, tribal/traditional, farm), not an
identifier. The office made the resolution decision and drew the line where this map now
draws it.

At 1 dot = 1,000 people, the smallest categories draw as scattered single dots in large cities
and nothing on this card identifies a locatable minority at a grain the state has not already
released.

---

## 9. Not done

- **The 1996 and 2001 censuses split the African Independent Churches and nothing here uses
  it.** UNSD's oracle carries 2001 at **26** categories including *Zion Christian Church*
  4,971,932, *Other Apostolic Churches* 5,609,070, *Other Zionist Churches* 1,887,147,
  *Ethiopian Churches* 880,414, *Ibandlalama Nazaretha* 248,824 and *Other African Independent
  Churches* 656,644; and **1996 at 64 categories**, which is the richest religion card any
  African census has ever run. Both are national-only in the oracle and §2.5 found no open
  subnational table for either. **This is now the single largest improvement available to this
  country**, and it is larger than it was: a §3.4 rescale splitting the 2016 AIC cell by 2001
  shares would land those churches on 213 units rather than 9. Note the answer sets are close
  enough to make it tractable, unlike the 2016/2022 pair in §3. The 2001 census microdata is on
  DataFirst too and behind the same account that is now open.
- **The 2011-vintage 234-municipality build**, if a 2011 COD-AB release can be found. 10% more
  units; §2.4 says what it would cost.
- **SuperWEB2** could not be reached and its Census 2022 religion content is unverified either
  way. It is the one open question in §2.5's table and it matters much less now.
- **uPhongolo and Swellendam.** §2.3. Nothing here can establish what happened, and a Stats SA
  query is an email rather than a download.
- **Whether CS 2016's `no religion` or Census 2022's is closer to the truth.** Neither is
  checkable from here and the gap is a factor of four. Anyone updating this country to 2022 is
  changing that number by 4.2 million people and should say so in the note.

---

## 10. Review of the PROVINCE build, 2026-09-08 — kept, and superseded on 2026-09-09

*This section reviewed the nine-province build. The country it describes no longer exists; it is
kept because most of what it established is still load-bearing, and because two of its findings
were acted on by the rebuild rather than merely carried over.*

A second pass, not the builder's. `check_md.py`, `built_countries.py --check`,
`check_rollup.py za` (54,946,360, all measured, nothing derived, nothing orphaned) and
`check_mapping.py za` were all clean, and the screenshot was clean: dots inside the border,
Gauteng dense and Northern Cape sparse, Lesotho and the Eswatini salient blank, nothing in the
sea.

**The node arithmetic reproduced exactly**, recomputed from `data/normalized/*.csv` through each
country's own `resolve()`. *A trap for whoever recomputes this next: `ke` carries `country` and
`county` rows and `bj` carries three tiers, so a naive sum over the normalised file
double-counts Kenya and triples Benin. Group by `geo_level` first.*

**The fourteen denominations agree with every precedent that shares a body**, and that is
unchanged by the rebuild because not one node moved. Against `sz2017.py`, which is the closest
card: Catholic, Anglican, Lutheran, Methodist (parent, not `.african`), Jehovah's Witness,
Pentecostal, Seventh Day Adventist (parent, not `.sda`) and `Other` all land on the same nodes,
and Zionists/Apostles and African Independent Church both land on `christianity.africaninstituted`.
`zw2022.py` and `ke2019.py` agree on Catholic, Pentecostal, Other Christian, the AIC cell and
every non-Christian row. `Mormon` -> `christianity.latterday` matches bb, bs, gd, ee, fi, lc and
pe; `Reformed church` -> `christianity.reformed.continental` matches au2021 and ca2021. **The
AIC call is not close and the exemplar list settles it**: the printed row reads *Zion Christian
Church; Apostolic Church; African Nazareth Baptist Church/Shembe*, and the Pentecostal row
separately prints *Apostolic Faith Mission*, so the indigenous Apostolic bodies and the
Pentecostal mission body each have their own box.

**One divergence, and it is Eswatini's rather than South Africa's.** `sz2017.py` sends
`Christian: Not Stated` to `christianity.other`; this country sends `Denomination not reported`
to bare `christianity`, with at2001, au2021, bd2011, bg2021, bs2022 and ie2022 behind it. Bare
`christianity` is the better call and Eswatini's cell is thirteen people, so nothing is worth
changing; recorded so the next reader does not read the pair as a rule.

### 10.1 The published gap, and where it went

The review found that 707,294 people, 1.27%, answered no religion question at all and that
`countries.py` carried no `gap`. It established the figure two independent ways: the eight
printed footnotes sum to 640,083 and Western Cape by difference is 67,213, giving 707,295;
table 2.1's national 55,653,654 less the nine table 2.10a totals is 707,294. It noted that
`tools/gap_share.py za` refused the country ("the mapping excludes nothing"), correctly, because
those people were in no row of any table the parse read.

**The microdata reproduces both routes and closes the tool's refusal.** Summed per person the
two excluded codes are 704,355 + 2,940 = **707,295**, Western Cape's is **67,213** on the nose
against the review's inferred figure, Gauteng's is 270,324 against its footnote's 269,397 +
927, and Free State's is 25,979 against 25,846 + 133. The rows are now in
`data/normalized/za.csv` and `gap_share.py` computes 1.27% from the file. `gap_share=0.0127` is
unchanged.

### 10.2 The §3.5 lean, and what happened to it

The review computed r = +0.7183 with the no-religion share over nine provinces (two-sided
permutation p = 0.0280) and r = −0.6273 with Christianity, and put the finding in
`note_public`. §3.5 above re-runs it at 213 units with leave-one-out. The direction holds, the
correlation halves and the p-value falls by three orders of magnitude, and the province-level
non-response shares it printed (Northern Cape 0.417% … Gauteng 2.017%) are reproduced by the
microdata exactly.

### 10.3 The `africaninstituted` node note

Rewritten 2026-09-08 with the composition recomputed through each country's own `counts()`, and
carrying one correction worth keeping: **`BRANCHES`'s third element is the maintainer note and
is never rendered.** `taxonomy/build_tree.py` puts it on the node as `note`; the reader-facing
tooltip is `PUBLIC_NOTE[bid]`, which has two entries and no `africaninstituted`. The audience
for that note is the next builder deciding where to file an AIC cell.

**Checked again after the rebuild, 2026-09-09.** The node is 25,899,977 over seven countries,
za 54.67%, zw 23.60%, ke 12.71%, ao 4.24%, bj 2.61%, sz 1.62%, ci 0.54%; 95.21% on the bare
parent and 4.79% on four named-church children. Every share in the note is still right to the
decimal it prints; two integers moved by eight people (25,899,969 → 25,899,977 and 14,158,453 →
14,158,461) and were updated in place.

---

## 11. Verification of the rebuild, 2026-09-09

Run by the builder, so it is not a review; §10 is the province build's review and this country
has not had one at its new tier.

| check | result |
|---|---|
| `sources/za.py` | 3,328,867 records, 55,653,654 weighted, 213 municipalities, **215 of 216 published province cells agree to within 0.5 people**, the 216th is §4 |
| `sources/za_geo.py` | 213/213 both ways; district 213/213 over 52, province 213/213 over 9, name 210 of 211 with `MP326` the only expected exception |
| `sources/za_grid.py` | 501,054 hexes kept, 1,760 dropped (0.207%), every municipality has 479 to 7,765; log-log r **0.9815** vs a best shuffle of 0.2253 |
| `tools/check_mapping.py za` | 27 categories, 0 unmapped and not EXCLUDED, 24 nodes, all on the tree |
| `tools/check_rollup.py za` | 54,946,339 people, **all measured**, nothing derived, nothing orphaned |
| `tools/gap_share.py za` | computes **1.27%** off the normalised file and agrees with the authored `gap_share=0.0127`; it refused this country before the rebuild |
| `coverage.py` | 121 countries, every drawn node covered |
| `tools/check_md.py` | clean |
| `tools/built_countries.py --check` | 121 countries, both editions, nothing missing |
| `tools/check_tiles.py` | 1,227 tiles, 0 mismatched, identical to the reference |
| screenshot | 54,933 dots at 1:1,000 across 28,109 polygons, no rings. Dots inside the border, Lesotho and the Eswatini salient blank, nothing in the sea, Gauteng and the KwaZulu-Natal coast dense and the Northern Cape sparse. The header block reads *household survey, 2016, read per person* / *local municipalities, 258,000 people on average* / *0% modelled* / the `not drawn` row. |

**One number moved that a reader might notice.** The country's drawn total is 54,946,339 against
the province build's 54,946,360, a difference of 21 people. That is rounding: 213 x 27 float
cells are rounded independently where nine provinces used Stats SA's own printed integers. The
worst province x category drift from rounding is 4.7 people and `sources/za.py` asserts it stays
under 12.

---

## 12. Review of the 213-municipality build, 2026-09-09

A second pass on the rebuild, not the builder's. §10 reviewed the nine-province country that no
longer exists; this reviews the one that replaced it. Everything below was recomputed from
`data/normalized/za.csv`, from the nine PDFs in `data/raw/za/` with a parser that shares nothing
with `sources/za_profiles.py`, and from the `.dta`'s own value labels, rather than read out of
§§1-11.

`check_md.py`, `built_countries.py --check` (121 countries, both editions), `check_rollup.py za`
(54,946,339, all measured, nothing derived, nothing orphaned), `check_mapping.py za` (27
categories, 0 unmapped, 24 nodes all on the tree) and `gap_share.py za --check` are clean, and
the `countries.py` entry carries no em dash, no markup in `how`/`grain`/`gap` and no `fill`,
which is right for a country with no derived rows. The screenshot is clean: dots inside the
border, Lesotho and the Eswatini salient blank, nothing in the sea, Gauteng and the KwaZulu-Natal
coast dense, the Northern Cape sparse.

### 12.1 The reconciliation holds, and §4 is right to the person

**Reproduced independently.** All nine profile PDFs were re-parsed here with a different parser,
which had to cope with the table being numbered 2.10a/b in five provinces, 2.9a/b in Eastern Cape
and Gauteng (and written `Table 2.9 a:`, with a space, in Gauteng), 2.7/2.8 in Limpopo and
2.11a/b in Western Cape. 221 published cells were compared against province sums of the
normalised file. **Exactly one differs by more than 12 people, and it is North West's `Other`.**
Every other province's fourteen denomination rows sum to its own printed total within two
people; North West's fall 336,482 short.

**§4's arithmetic is exact.** Report 03-01-11 table 2.10b's fourteen rows sum to 3,072,039
against a printed total of 3,408,521, its percentages sum to 90.1, its `Other` cell reads
`21 873`, and its footnote reads *Excludes 'Do not know' (21 873), 'Unspecified' (12) and 'Not
applicable' (318 029)* — so the `Other` cell is character for character the footnote's `Do not
know`, and North West is the only one of the nine whose footnote carries a third exclusion.
21,873 + 336,482 = **358,355**, and the normalised file puts North West's `Christian: Other` at
358,356, one person off from independent per-municipality rounding. Two further identities close
it: 2.10a's Christianity 3,430,406 less 2.10b's total 3,408,521 is 21,885, which is that table's
own `Do not know` 21,873 plus `Unspecified` 12; and the rival hypothesis needs the shortfall to
be `Not applicable` 318,029, which misses by 18,453. **The finding is clean.** It is the one
place on this map where a published cell has been shown to be mis-set rather than merely
doubted, and moving 336,482 people from bare `christianity` to `christianity.other` is right.

### 12.2 There is no fifteenth denomination, confirmed from the file

Read straight off the `.dta`'s value labels. `CHRISTIA` carries **17** labels: codes 1-14 are the
fourteen published denominations, with 13 being `Just a christian/non-denominational`, and then
15 `Do not know`, 88 `Not applicable`, 99 `Unspecified`. `RELIGION` carries 13: the eleven
published categories plus `Do not know` and `Unspecified`. **The 15 in `ask/answered/002-za` and
in `sources.md` §9bw was wrong**; §1 here and the ask's own *Carried out* section already say so,
and §9bw has now been corrected in place, so the count no longer stands anywhere unqualified.
`sources/za.py` folds every codebook label onto a published one and refuses otherwise, so a
re-release that added a real fifteenth would fail the build.

### 12.3 The label-set trap: right set, and better protected than §2.4 claims

Confirmed from the file. `MN_CODE` has 234 labels written `WC011: Matzikama`; `MN_COD_A` has 213
written `WC011 : Matzikama`, and `LIM345 : New` is in the 213 set, which is itself evidence that
it is the August 2016 demarcation, since Collins Chabane did not exist before it.
`_label_sets()` takes the smaller set, which is the right one, asserts the two sizes are exactly
213 and 234, and asserts the spacing still tells them apart. Because it picks by **size and not
by name**, a release that swapped the two truncated names would be handled silently and
correctly, which is the case §2.4 worries about.

One correction to §2.4, and it is in the build's favour: **in this file the wrong pick would not
be silent.** The two sets' numeric keys are disjoint, `MN_CODE` running 160 to 987 and
`MN_COD_A` 1011 to 9032, so resolving `MN_CODE_2016` through the 2011 set raises `KeyError` on
every record rather than producing wrong municipalities. `sources/za_geo.py`'s 213/213 code join
to COD-AB is a third layer behind that. The trap §2.4 describes is real as a general Stata hazard
and worth keeping in spec §12; it is not live in this particular file, and saying so is worth
more than leaving a reader to think one assertion is all that stands between the map and 213
wrong places.

### 12.4 The §3.5 lean reproduces exactly, and `note_public` quotes the new figure

Recomputed from the normalised file: r = **+0.3063** with the no-religion share and **−0.2552**
with the Christian share; two-sided permutation p = **0.00005** over 20,000 permutations, with
zero draws at or above the observed value; leave-one-out **+0.2649 to +0.3273**; non-response
0.0000% in five municipalities to **8.3504%** in eDumbe, median 0.6720%; population-weighted r
**+0.3986** when the weight is each municipality's survey total, which is what §3.5 quotes.
707,295 undrawn, 1.2709%, matching `gap_share=0.0127` and the `gap` sentence. `note_public` says
*"r = +0.31, permutation p under 0.0001"*, which is the 213-unit figure and not the old +0.72.

### 12.5 Reader-facing figures

Every figure in `note_public` was recomputed and all reproduce: African Independent Churches
14,158,461, 25.77% of answers and 32.61% of Christians; Big Five Hlabisa 61.98% and Mfolozi
61.62% of answers; Mthonjaneni 75.56% and Kai !Garib 4.64% of Christians; Methodists 33.82% of
Ntabankulu, 33.41% of Umzimvubu, 0.16% of Greater Giyani, 15.18% of the Eastern Cape; Reformed
41.89% of Bergrivier, 39.11% of Matzikama, 0.21% of Mfolozi; Presbyterian 11.13% of Raymond
Mhlaba; no affiliation 5,964,889 and 10.86%, Okhahlamba 32.67%, Ephraim Mogale 27.66%,
Johannesburg 14.73%, Cape Town 7.79%; Hinduism 9.23% of KwaDukuza and 8.98% of eThekwini; Islam
8.41% of Cape Town; traditional religion 4.47%, Maphumulo 31.35%, Mkhambathini 30.78%; uPhongolo
4.76% atheist on 384 records and 12.61% of the national cell, with Cape Town 30.6% and
Johannesburg 15.6%; Swellendam 57.34% on 965 records. §2.2's sample figures reproduce as well:
4,472 non-empty drawn cells, 882 of them on fewer than ten records holding 0.1066%, heaviest
five-record-or-fewer cell 317 people in Stellenbosch's Judaism on two records, smallest sample
529 in Prince Albert, median 7,890. Kareeberg 99.68% and Msinga 46.61% are right. **No figure in
`note_public` rests on a thin cell**; the smallest sample behind any quoted municipal share is
Greater Giyani's 21 Methodist records, which is in §7 and not in the note.

Two corrections, both made in place above:

- **§7's Judaism sentence said Johannesburg and Cape Town hold 78% of the country's Jewish
  answers. They hold 73.0%** (23,420 and 12,672 of 49,467). eThekwini's 3,599 takes the top three
  to 80.2%, which is probably where the 78 came from. Not in `note_public`, so nothing
  reader-facing was wrong.
- **§2.3 described Western Cape's 7.06% as *of its province's Christians*. It is of its
  province's answers**; of its Christians the figure is 8.64%, which the same paragraph quotes
  correctly two sentences later. Every other comparison in that paragraph is answers against
  answers. `note_public` says *"its province reports 7.1%"* with no base named and is fine.

§7's `Just a Christian` row puts *Swellendam 57.34% of answers* in a column whose other rows are
of-Christians over the 175 municipalities above 50,000 people. Swellendam has 38,742 and is not
in that set; the genuine high there is **Steve Tshwete at 17.03% of Christians**. The cell is
italicised and cross-referenced to §2.3 so it is not passing itself off as the column's answer,
but the real maximum is now on the record.

### 12.6 The two anomalies: drawn as returned is right, and the reason given for one is wrong

**Drawing them is right and I would not change it.** §14.4 rule 1 forbids inventing a magnitude
and §3.5 says mark rather than fill, and there is no third option that is not a guess about which
households were mis-coded. Beyond the rules, three things settle it. The blast radius is tiny:
uPhongolo's atheists are 6,630 people, 0.012% of the country, and Swellendam's non-denominational
excess is about 19,500, 0.036%, so neither moves a national share by a tenth of a point and
neither touches `christianity.africaninstituted`, which is what this country is for. The
disclosure is as strong as this map's grammar allows, both named in `note_public` with their own
figures. And the precedent runs the other way: declining to draw a returned cell because it is
surprising would license editing any cell a reviewer dislikes, when the whole point of the
microdata tier is that it shows what the province tabulation averaged away, fieldwork included.
The map is showing an interviewer in two places out of 213, it says so, and that is better than a
smoothed country that says nothing.

**But §2.3's account of uPhongolo does not survive the file, and the correction strengthens the
artefact reading rather than weakening it.** Recomputed against KwaZulu-Natal, all shares of
answers:

| | uPhongolo | KwaZulu-Natal | diff |
|---|---|---|---|
| Atheism | 4.76% | 0.13% | **+4.63** |
| No religious affiliation/belief | 7.35% | 12.87% | **−5.52** |
| African Independent Church | 40.82% | 30.56% | +10.26 |
| Pentecostal/Evangelistic | 19.91% | 11.35% | +8.56 |
| Catholic | 1.35% | 9.20% | −7.84 |
| Christianity, all fifteen rows | **75.13%** | **71.11%** | +4.02 |

§2.3 concludes *"the excess is drawn from the named denominations rather than from the
no-religion cell."* The file says the opposite. The no-religion cell is short by 5.52 points,
more than atheism is long, and it is the second-largest negative deviation in the municipality.
uPhongolo is **more** Christian than its province, not less; its Catholic, Methodist and Anglican
cells are low because it is an African-Independent and Pentecostal municipality, which is
ordinary northern-KwaZulu-Natal geography and is not evidence about atheism at all. The tidier
reading is the one §2.3 rules out: the movement is inside one block of the religion card, from
code 10 to code 8 with `Agnosticism` at 9 between them. That matters for what a reader loses,
because it means uPhongolo's `unaffiliated` node is understated by about as much as `secular` is
overstated, and those two sit beside each other in the legend, so the visible damage is smaller
than §2.3 implies rather than larger.

Swellendam is a different shape and the record's account of it holds better, though not its
mechanism. `Just a Christian` is +50.28 points; the deficit is spread across Pentecostal
(−11.82), African Independent (−9.32), `Other Christian` (−6.44), Catholic (−5.81), Islam
(−5.33), no religion (−5.13) and Methodist (−4.96), so it drew from both question cards and its
overall Christianity share is 95.63% against the province's 81.73%. A second anomaly sits in the
same municipality and is not mentioned anywhere: its `Denomination not reported` is **5.28%
against a provincial 0.74%**, the highest of the 213.

**Two wording points follow.** First, *"the categories adjacent to it on the card"* (§2.3, and
the same phrase in `sources.md` §9cs) is not supported by the code order in either case:
uPhongolo's deficit is two rows away with `Agnosticism` between, and Swellendam's is scattered
across both cards including one category that is up. `note_public`'s own wording, *"the other
categories fall short by about as much as those rise"*, is exactly right and needs nothing.
Second, §2.3 opens *"Both are large"*; Swellendam is 38,742 drawn people against a median of
122,104, and 1,685 records against a median of 7,890, which puts it near the bottom of the
distribution on both. The defensible claim is the one `note_public` already makes, that neither
rests on one household.

None of that changes what is drawn. `note_public` is accurate as it stands; the corrections are
to §2.3's reasoning, which is where the next person updating this country will start.

### 12.7 Nothing else

The mapping is unchanged from the province build and §10 already checked all twenty entries
against precedent; `other.za` is the only node either build added and §10 cleared it. No new
taxonomy node, no new ask, no §14 question. §8's reading that Stats SA released the municipality
code on the same public-use record while withholding the enumeration area is sound, and
`ask/answered/002-za` is Anita's ruling on the tier.

**One thing worth carrying elsewhere.** The reconciliation only worked because the parser
survived the table being numbered four different ways across nine reports of the same series,
one of them with a space inside the number. That is the same failure mode as
`[[reference_korail_yearbook]]`: a publication series renames and re-punctuates between volumes,
and a parser anchored on the table's title rather than its number is what makes an old build
usable as a new build's check.
