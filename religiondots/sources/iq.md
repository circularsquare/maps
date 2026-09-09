# Iraq — Arab Barometer, waves V, VI-3, VII and VIII (2018–2024)

Built 2026-09-09. The third country on this map drawn from the Arab Barometer, after Egypt
(`sources/eg.md`, `sources.md` §9bz) and Jordan (`sources/jo.md`, §9cq), and **the first
Sunni/Shia geography on the map**. `sources.md` §11af assesses the source across the region,
§11r closed Iraq on the state, and §11al is Lebanon, which is the country this one has to be
told apart from.

**46,118,793 people over 18 governorates**, five drawn nodes, every row `modelled`.

## 1. The state route, closed by §11r and re-checked here rather than assumed

Iraq is not a country that failed to count itself. **The 2024 census is real, recent and
complete**: enumerated 20–21 November 2024, 46,118,793 people, final results 24 February 2025,
the first full count since 1987. COSIT is reachable and so is KRSO for the Kurdistan Region.

**What it published was read directly.** `cosit.gov.iq/documents/AAS2024/02.pdf` is the Annual
Statistical Abstract's census chapter, 39 pages, and this build uses two of its tables as the
denominator and one from chapter 01 as a witness. Every table in the census chapter is
governorate crossed with urban/rural, sex and age. **There is no religion table and no sect
table.** §11r scanned the same file for the Arabic strings for religion, sect, Christian,
Sabean, Yazidi, Muslim and ethnicity and found none of them; the only hits for *al-din* are the
Salah al-Din governorate name.

**Whether the question was asked at all is genuinely disputed and does not matter.** The
consistent account is that sect and ethnicity were deliberately excluded from the 2024 form to
avoid re-litigating the disputed territories, with some reporting that a plain religion field
survived. Nothing was published either way, so both accounts have the same consequence here.

Everything else was checked by §11r and is not re-opened: Iraq is **absent from the UNSD
Demographic Yearbook's religion table** (`tools/oracle.py`), and **the USCB country-geodatabase
seam does not rescue it** — the Iraq geodatabase exists and its tabular layers are households,
ICT, health, mortality and people-in-need, with no religion and no ethnicity. The last Iraqi
census to publish religion at all is 1987.

## 2. What is drawn

| source category | node | share as drawn | people |
|---|---|---:|---:|
| Shia | `islam.shia` | 45.15% | 20,824,066 |
| Sunni | `islam.sunni` | 30.56% | 14,095,685 |
| Just a Muslim | `islam` | 21.91% | 10,103,174 |
| Muslim, other denomination | `islam` | 0.94% | 433,924 |
| Muslim, denomination not given | `islam` | 0.83% | 382,411 |
| Christian | `christianity` | 0.31% | 144,260 |
| Other religion | `other.iq` | 0.29% | 135,273 |

`taxonomy/iq2024.py` carries the mapping and its `REVIEW` argues each of the seven.

## 3. The instrument, and why the column §11af rejected is the one this country is for

`Q1012` is *"What is your religion?"* and in Iraq it answers almost nothing: 99.3% Muslim.
Drawn on its own, Iraq would be one flat colour, which is what closed Morocco, Algeria,
Tunisia, Libya and Sudan.

The content is the follow-up. **`Q1012A` (waves V and VI-3) and `Q1012A_MUSLIM` (VII and VIII)
ask a Muslim's denomination**, offering Sunni, Shia and *just a Muslim* beside Hanbali,
Shafi'i, Maliki, Ja'fari, Druze, Ibadi and Ahmadiyya.

**§11af tested that exact column across five North African countries and rejected it**, and it
was right to: *just a Muslim* is the modal answer in all five (44.9% in Morocco, 81.8% in
Sudan) and wholly-Maliki Morocco returns 16.2% Maliki, so there the variable records which
label a person volunteers. §11af explicitly left open *"whether the same variable behaves
better in Iraq and Lebanon, where sect is a salient public identity rather than an unmarked
default"*.

**In Iraq it does, and three measurements say so rather than one argument.**

1. **The undifferentiated share is 24.5%**, against 44.9–81.8% across the five.
2. **Of the Iraqis who name a branch, 60.9% say Shia.** The CIA World Factbook's band is 61–64%
   Shia and 29–34% Sunni of the whole population, which is 64–68% of the named; the range of
   serious estimates runs from roughly 55% to 70%. Iraq's own state publishes none.
3. **The geography reproduces Iraq's, without being told it.** `sources/iq.py`'s
   `sect_geography` asserts rather than admires: all nine southern and mid-Euphrates
   governorates come out majority Shia among the branch-namers and all six western and Kurdish
   ones majority Sunni, **98.2% against 1.5%**, with Baghdad, Diyala and Kirkuk in between at
   68.1%. A survey measuring something other than sect could not do that.

### The Shia share of branch-namers, by governorate

| governorate | % Shia of those naming a branch | n |
|---|---:|---:|
| Maysan | 100.0 | 202 |
| Najaf | 100.0 | 311 |
| Dhi Qar | 99.8 | 439 |
| Qadisiyyah | 99.6 | 237 |
| Karbala | 99.6 | 234 |
| Muthanna | 99.0 | 99 |
| Babil | 98.2 | 386 |
| Wasit | 96.3 | 267 |
| Basra | 94.6 | 499 |
| **Baghdad** | **81.5** | 1,336 |
| **Diyala** | **38.4** | 177 |
| **Kirkuk** | **10.9** | 220 |
| Nineveh | 4.2 | 472 |
| Salah al-Din | 2.3 | 214 |
| Anbar | 0.8 | 265 |
| Sulaymaniyah | 0.2 | 409 |
| Duhok | 0.0 | 58 |
| Erbil | 0.0 | 399 |

## 4. The quarter of Iraq that names no branch, which is drawn as itself

**23.7% of the country as drawn sits on the bare `islam` node** and nothing shares it out. That
is `Just a Muslim` (21.9%), the sect card's `Other` (0.94%) and the people who refused the
follow-up after answering Muslim (0.83%). Russia is the precedent and it is exact:
`branches.py`'s note on `islam.shia` records that 4.66% of Russia answered *"I profess Islam,
but am neither Sunni nor Shia"* and that those people stay on the parent.

**Why nothing is apportioned.** The obvious operation is to divide them in each governorate at
that governorate's observed Sunni:Shia ratio. It preserves every total and gives a tidier map,
and it is exactly §14.4 rule 1: it would state, in Baghdad, that the people who would not name
a sect divide 81:19 the way the people who would do — the least likely thing to be true about
them.

**And the level moves with the fieldwork**, which is the second reason and the harder one:

| wave | fielded | names no branch |
|---|---|---:|
| V | 2018–2019 | 17.7% |
| VI-3 | March–April 2021 | 42.7% |
| VII | Oct 2021 – Jul 2022 | 27.1% |
| VIII | Sep 2023 – Jul 2024 | 21.2% |

A factor of 2.4 across six years is not a change in Iraq. Anything built on the *level* of the
undifferentiated share would be measuring which fieldwork a respondent fell into.

**What is not noise is where they are.** `Just a Muslim` cleared the split-half in its own
right at **+0.766**, and it runs 45.9% in Diyala, 40.1% in Salah al-Din, 39.6% in Nineveh,
33.2% in Kirkuk and 30.5% in Anbar against 2.6% in Erbil and Duhok, 6.1% in Sulaymaniyah and
7.0% in Najaf. **Declining to name a sect is a Baghdad and mixed-belt behaviour**, rare where
one branch holds nearly everybody. So the drawn Sunni share in Baghdad, Diyala, Kirkuk and
Nineveh is a floor rather than an estimate, and the `islam` dots on this map are not scattered
noise, they are where the two communities actually meet.

## 5. The four readings of the questionnaire this build makes

Every one is in `sources/iq.py` beside its reason; they are collected here so a reviewer does
not have to find them.

1. **Waves II and III are omitted on the card, not on the answers.** The files offer Iraq a
   religion answer in six waves. Wave II (2011, 1,234 Iraqis) has **no `q1012a` column at
   all**; wave III (2013, 1,215) has one and it is **empty for every Iraqi in it**. Pooling
   them would put 2,449 Iraqi Muslims into the undifferentiated bucket because of the
   questionnaire they were handed, and that bucket is the single most instrument-sensitive
   quantity in the file. They are named in `omit=` with the reason, so the pool is not quietly
   narrower than the files.
2. **The madhhab answers are folded up into their branch.** 105 `Shafi'i`, 19 `Hanbali` and 2
   `Maliki` to Sunni; 221 `Ja'fari` and 2 `Alawi` to Shia. **`branches.py` has
   `islam.sunni.shafii` and `islam.shia.jaafari`**, added with Türkiye, so this is a choice
   rather than a gap in the tree. It is made because this instrument does not measure a
   madhhab: Shafi'i runs 0.1%, 0.2%, 1.7% and 2.6% across the four waves and Ja'fari 1.6% to
   4.2%. Türkiye's schools come from the Diyanet's own survey, which asks the madhhab outright
   and does not put Sunni and Shia on the same card. And left unfolded they are the wrong shape
   for the machinery: Iraq's Shafi'is are overwhelmingly Kurds, five of the 105 are in the
   early wave half, so the split-half would rank them on noise, fail them, and `ab.build` would
   spread Kurdish Shafi'is across Basra.
   **`Alawi` is the one entry here that is a judgement and it is two people.** 0.019% weighted.
   Iraq has no Alawite community of any size and in an Iraqi Shia setting the word is
   ordinarily a claim of descent from Ali. Left as its own answer it falls under the
   eligibility floor and gets drawn as roughly 8,600 Alawites who are not there.
3. **A sect refusal stays in the universe; a religion refusal does not.** Jordan's eleven
   refusals were refusals of the religion question, so nothing was known about them and
   dropping them was right. Here 83 people have already said Muslim and then declined the
   follow-up, which is a Muslim of unstated denomination and not a person of unknown religion.
   They are composed into `Muslim, denomination not given` and drawn on `islam`. The six who
   refused `Q1012` itself are dropped, and so are the two who answered `Atheist` on the one
   card of four that offers the box (Egypt dropped its two for the same reason, §9bz).
4. **Two boxes spelled `Other` are kept apart by hand.** `Other` is an answer on the religion
   card and an answer on the sect card, and they are different answers from different people.
   Left with their printed wording they merge into one category **with every total still adding
   up**, which is `assert_one_wording`'s failure one column across and which that function
   cannot catch, because the two spellings are identical rather than merely similar. They are
   composed into `Muslim, other denomination` and `Other religion`.

Plus one recode that is mechanical rather than a reading: waves V, VII and VIII print `Ja’fari`
and `Shafi’i` with U+2019 and wave VI-3 prints `Ja'fari` with the ASCII apostrophe, which
`ab.fold`'s NFKC normalisation does not merge. Without the recode the pool carries each answer
twice with every total still adding up. `Shafi'i'` is wave VI-3's own trailing-apostrophe typo,
three people, and `Malki` is the survey's spelling of Maliki.

## 6. The geography, and why it is COD-AB rather than geoBoundaries

`sources/iq_geo.py`. **This is the first country on this map where geoBoundaries was tried and
rejected on a measurement.**

`gbOpen/IRQ/ADM1` is eighteen features with ISO 3166-2 codes and is the easier download. Its
**Baghdad polygon is 912 km² against Iraq's own published 4,555 and COD-AB's 5,100**: it draws
something close to the built-up city and hands the rest of the governorate to Babil (1.55× its
statute area in that file), Salah al-Din (1.08×) and Diyala (1.13×). **Baghdad is 21.2% of
Iraq's population.** That would have crammed a fifth of the country's dots into a fifth of the
right polygon and drawn the overflow in three neighbours it does not belong to.

COD-AB's `irq_admin1.shp` is the Central Statistical Organisation's own layer, valid from
2019-06-03, eighteen features carrying the official `IQG01`–`IQG18` p-codes and each
governorate's Arabic name. Its polygon areas reproduce COSIT's published table to a few per
cent everywhere except the Najaf/Anbar desert boundary (Najaf 1.39×, Anbar 0.90×), which is a
real disagreement between two Iraqi authorities rather than a defect in either file.

**The join is the p-code and it is checked four ways**, because Iraqi governorate names
romanise badly enough that only 9 of 18 survive a letters-only fold between two English
renderings of the same Arabic:

1. the authored table covers all eighteen p-codes and both COSIT name sets exactly;
2. **COSIT's published area against COD-AB's own geometry**, ρ = **+0.994**, and **0 of 5,000**
   random pairings reach it. Two authorities and two operations: the Ministry of Water
   Resources surveyed the areas and the CSO drew the polygons, and they span a factor of 30;
3. **the shape of the population on the ground.** The three sparsest come out Anbar 16/km²,
   Muthanna 20 and Najaf 49 against a next-sparsest of 71, and the densest is Baghdad at
   1,918/km² against Babil's 465. Both are asserted;
4. the name fold, 9 of 18, printed as the weak witness it is. A permutation would leave 0 or 1.

### Two traps in COSIT's PDFs

**The area table is printed three times in three vintages on consecutive pages** — `FOR 2024`,
then `FOR 2022`, then `FOR 2023`. Reading "the area table" by page number or by the first
caption match gets one of them at random. `AREA_YEAR` anchors it.
[[reference_pdf_table_geometry]] with the year in the header.

**And the census page's column header extracts as `Total الذكور/`**, which folds to `total` and
is read as the table's own Total row with one value in front of it. `_rows` only accepts a line
as a row label if it carries **no Arabic and no digits**; every real label on these pages is a
bare Latin string on its own line and every real value line either ends in a digit or is glued
to the Arabic name.

### Halabja

Iraq's parliament made Halabja a governorate in its own right in 2014, splitting it from
Sulaymaniyah. **None of the three inputs carries it**: COD-AB's ADM1 is eighteen features, the
2024 census tabulates eighteen, and the survey's `Q1` offers eighteen. So all three tiers agree
and nothing is lost; drawing a nineteenth unit no input carries would be the error.

## 7. The checks

**The label harmonisation against the codes.** Iraq's `Q1` code means the same thing in all
four waves, which is not true of this survey anywhere else — Jordan's means three different
things across nine. It is `70000 + n` in waves V, VII and VIII and `7000 + n` in VI-3, with the
same `n` throughout. It is still not the pooling key; it is the witness on the names, and it
agrees on **8,335 respondents with zero disagreements**. Three labels need reading rather than
transliterating:

* **`Diwaniyah` is Al-Qadisiyyah**, named for its capital, in wave VI-3 alone. No spelling of
  Qadisiyah reaches it. This is `[[reference_name_join_wrong_neighbour]]`'s case in the form
  where the name is not wrong, it is a different name.
* **`Dhi War`** is wave V's typo for Dhi Qar, 163 respondents.
* **`Salahaddin`** is Salah al-Din with the article run into the name.

**The held-out decode**, which touches nothing in the religion column: the survey's governorate
shares of respondents against the census's governorate populations, **r = +0.985**, and **0 of
20,000** random pairings reach it.

**Lebanon's quota check, run here.** §11al found that Arab Barometer's Lebanese sample is a
fixed sect-by-governorate quota and that §14.16's split-half passes it at its strongest, so
`ab.assert_not_quota` now runs before `ab.stability` for every country. Iraq is clean: **no
wave pair returns an identical composition in any free cell**, worst pair V against VII with
0 of 34, Bonferroni p = 1 against a bar of 0.001, where Lebanon comes in at 1.4e-4.

**The split-half (§14.16)**, bar **+0.4014** at 95% on eighteen units:

| answer | national | Spearman | verdict |
|---|---:|---:|---|
| Shia | 45.62% | **+0.917** | own geography |
| Sunni | 29.25% | **+0.965** | own geography |
| Just a Muslim | 22.78% | **+0.766** | own geography |
| Muslim, other denomination | 0.93% | — | under the 1% floor |
| Muslim, denomination not given | 0.82% | — | under the 1% floor |
| Christian | 0.31% | — | under the 1% floor |
| Other religion | 0.29% | — | under the 1% floor |

Leave-one-out never comes near the bar: Shia +0.901 at worst (without Diyala), Sunni +0.958
(without Anbar), Just a Muslim +0.737 (without Anbar), against +0.4142 at seventeen units.
This is the least fragile survey pass on this map; Jordan's was +0.617 against +0.5035.

**Duhok is the one governorate to read carefully.** It is sampled at **0.15× its population
share**, the thinnest of the eighteen, on 58 branch-namers, and wave V and wave VIII do not
sample it at all. Its 92.5% Sunni is not in doubt as a fact about Duhok; the precision is.

## 8. §14, and why this country did not need an ask

`queue.md`'s Iraq row said *"Yazidis and Christians make §14.4 rule 2 acute"*, and it is right
that they would if they were drawn with a geography. **They are not, and that falls out of the
instrument rather than from a choice.**

* **The resolution question was already ruled on.** `ask/answered/001-eg` ruled that a
  religious minority may be drawn at governorate, on the ground that governorates are big, and
  §11af recorded before any of these four were built that *"Jordan, Lebanon, Iraq and Yemen
  inherit whatever Anita rules on `ask/001-eg`"*. Iraq's eighteen average **2.56 million**
  people, more than twice Egypt's twenty-seven. Nothing finer exists anyway: the survey cuts by
  governorate and carries nothing below it.
* **The groups rule 2 is about are placed nowhere.** 25 Christians and 27 others in 8,335
  respondents are both far under §11ad's 1% eligibility floor, so they fail the test that
  decides whether a category carries its own geography and `ab.build` spreads them at the
  national rate over all eighteen governorates. The map says these communities exist and says
  nothing at all about where they live. That is a stronger answer to rule 2 than any resolution
  choice could have been, and it is worth stating plainly that it is luck rather than design.
* **What IS drawn with a geography is the Sunni/Shia distribution**, which is among the most
  published facts about Iraq: it is the organising fact of the country's politics, it is in
  every International Crisis Group report and every news map of the disputed territories, and
  the survey behind it is public. §14.2's reflect-versus-reveal line puts that on the reflect
  side without much argument.

**The `other.iq` figure is a floor and the node's own text says by how much.** 27 respondents
draw 135,000 people, where the Yazidis alone are usually put at 400,000 to 500,000 and are the
largest of the four communities inside the cell (Yazidi, Sabean-Mandaean, Kaka'i, Zoroastrian
and Bahá'í). A general-population household survey does not reach a population that has been in
camps in Duhok, in Europe or unreachable since 2014.

## 9. Terms

Unchanged from §11af and re-read before anything was downloaded. Arab Barometer's download page
renders every file as `href="#"` behind a name/email form, which reads as a wall and is not
one: the real URLs are in the page's own HTML. Its FAQ says *"Anyone can download the Arab
Barometer data for analysis at no cost"* and that the data are *"publicly available and free of
charge"*. No retrieval restriction, no redistribution clause, no stated citation requirement.

COSIT's Annual Statistical Abstract is a public government publication served unauthenticated
from `cosit.gov.iq`. COD-AB is OCHA's, CC-BY on HDX. Kontur's population grid is CC-BY.

## 10. What is on disk

```
data/raw/iq/irq_admin_boundaries.shp.zip          COD-AB, 1.2 MB
data/raw/iq/AAS2024_01.pdf                        COSIT abstract chapter 1 (areas)
data/raw/iq/AAS2024_02.pdf                        COSIT abstract chapter 2 (census)
data/raw/iq/kontur_population_IQ_20231101.gpkg.gz 8 MB, the placement grid
data/geo/iq/iq_governorates.gpkg                  18 polygons
data/geo/iq/iq_lookup.csv                         18 rows, name, Arabic name, census pop
data/geo/iq/iq_hexes.gpkg                         109,239 hexes keyed to governorate
data/normalized/iq.csv                            126 rows, 46,118,793 people
```

The Arab Barometer waves live in `data/raw/arabbarometer/` and are shared with Egypt and
Jordan.
