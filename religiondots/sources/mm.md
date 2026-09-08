# Myanmar — DOP, 2014 Census Report Volume 2-C (Religion), Table 1

Wired 2026-09-07. 51,486,253 people, 15 States/Regions, 7 religion categories **plus the
non-enumerated**, 100% drawn.

| | |
|---|---|
| source | Department of Population, **2014 Myanmar Population and Housing Census, Census Report Volume 2-C: Religion**, **Table 1**, page 3 of 17 |
| basis | `self_id`, enumerated census population, plus DOP's own estimate of who it did not enumerate |
| geography | **15 States/Regions** — ~3.4M people each, the second coarsest counting geography on this map after Zimbabwe's ten |
| categories | **7** religions, plus `Estimated Non-enumerated population`, plus the universe total |
| drawn | **51,486,253** = 50,279,900 enumerated + 1,206,353 non-enumerated. The 7 religions are `measured`; the non-enumerated are `modelled` |
| licence | DOP publication, distributed by UNFPA; free to download and cite |
| cross-check | MIMU's independent p-coded transcription of the same table |

**Rakhine State is 34% not-enumerated**, and that single fact is why the country is worth
drawing and why it needed a decision before it needed any code.

---

## 1. What exists, which is very little, and it is not a §14 ceiling

**The entire published religion output of the 2014 census is two tables in a 17-page report.**
Table 1 is Union plus 15 States/Regions; Table 2 is a 1973/1983/2014 national time series.
That is all.

So 15 units is **not** §14 rule 2 being applied — it is everything DOP published, and §3.9b is
what makes it drawable. The check was done rather than assumed:

* the 17-page Vol 2-C is the whole religion release, and its own List of Tables names two;
* MIMU — which holds and republishes essentially every 2014 census product, including the
  per-State/Region table workbooks in Excel — carries religion **only** as the p-coded version
  of this same Union/State table;
* the general census tables MIMU ships per State/Region (`Census_<State>_Tables_Eng_2015.xlsx`)
  predate the religion release by a year and do not contain it.

**Read Myanmar as composition, never as location.** A state here averages 3.4 million people.

## 2. The non-enumerated, and the decision

DOP publishes an eighth column, `Estimated Non-enumerated population`: **1,206,353 people,
of whom Rakhine is 1,090,000, Kayin 69,753 and Kachin 46,600.** It says what they are:

> *"In Rakhine, an estimated 1.09 million people were not enumerated in the Census because
> they were not allowed to self-identify using a name not recognized by the Government. It is
> assumed that the non-enumerated population in Rakhine is mainly affiliated with the Islamic
> faith."*

**Anita's call, 2026-09-07: draw them, as their own category.** They map to `unenumerated`, a
new eighth member of §6.3a's grey family — see spec §6.3a-iii for the full argument.

**What is deliberately NOT done is applying DOP's own assumption.** The report takes it to the
Union level and publishes a second national figure on that basis (Islam 2.3% → 4.3%). It
publishes no such breakdown by State, and this map draws States. Taking the assumption down a
level would be estimating a magnitude at a finer resolution than the source publishes it — §14
rule 1, the one that has never moved. So the dots say *these people were not counted* and
nothing about what they believe, and the assumption is quoted in `note_public` so the reader
learns what the state itself concluded.

**The alternative was not drawing them at all, and that is much worse.** Enumerated Rakhine is
2,098,807 people of whom 28,731 are Muslim, so a map built from the enumerated columns alone
renders Rakhine **96.2% Buddhist**. That is not a modelling slip; it is the census's own
exclusion reproduced as a finding, and it is §14.2's second risk exactly.

**Every non-enumerated row is `modelled`** (§7), because nobody was counted and 1,090,000 is a
round number in the source because it is an estimate. `unenumerated` is a root with nothing
measured above it, so the viewer's rolled-up view removes it outright — which is the honest
test of this country and worth performing rather than describing.

## 3. The parse, and the two checks that earn their place

The text layer is clean: the row label, the literal `Number`, that row's figures one per line,
the literal `%`, then the percentages. **The non-enumerated cell is BLANK for the eleven states
that have none**, so a Number row is 8 or 9 figures and the count is what says which.

**Every identity inside Table 1 survives a consistent column permutation** — the seven
religions summing to each Total, the fifteen rows summing to the Union — which is §12's
Zimbabwe warning. Two things do not:

* **DOP prints a percentage under every count**, so `count / total` must reproduce it on all
  110 cells. A swapped pair of religion columns fails this instantly and no sum would notice.
* **MIMU published an independent transcription** of the same table, p-coded. Every cell is
  compared and must agree exactly, and **the state rows are paired to MIMU's BY THEIR FIGURES**
  rather than by name or position — the data proves its own pairing, which is just as well,
  because DOP writes `Ayeyawady` and MIMU and OCHA write `Ayeyarwady`.

### DOP rounds inconsistently, and one cell of the report is simply wrong

Measured over all 110 cells: **53 where rounding and truncation agree, 51 that match rounding
only, 5 that match truncation only** (Union/Buddhist 89.8678 → 89.8, Sagaing/Christian 6.5606 →
6.5, Bago/Hindu 2.0579 → 2.0, Shan/Hindu 0.093 → 0.0, Ayeyawady/Buddhist 92.1556 → 92.1). So
the check accepts either, as a rule rather than as a tolerance picked to pass — and it keeps
enormous margin, because a column swap moves a percentage by whole points.

**And one cell is neither.** Kachin's Hindu count is 5,738 of 1,642,841 = **0.3493%**, and DOP
prints **0.4**. It is not a parse artefact: the seven Kachin counts sum to its Total exactly,
MIMU's transcription carries the same 5,738 and the same 0.4, and no denominator in the table
yields 0.4 — including the total with the non-enumerated added back. **The count is what this
map draws, so nothing is affected.** It is listed in `PCT_EXCEPTIONS` rather than tolerated, so
that a second such cell fails the build, and asserted still-present so that a corrected file
also fails and the exception gets removed.

## 4. The `Total` row is not this country's universe

`Total` is the **enumerated** population, 50,279,900, and the non-enumerated sit outside it.
The universe Myanmar draws is 50,279,900 + 1,206,353 = **51,486,253**, which is the report's
own overall figure on page 4. That is asserted rather than assumed, and it is the sort of thing
that would otherwise silently make a country 2.3% short.

## 5. What the map shows

**Christianity is an upland religion and the boundary is the sharpest in the country.** Chin is
**85.4% Christian**, Kayah 45.8%, Kachin 33.8% — against 1.1% in Magway and 1.1% in Nay Pyi
Taw. That is the American Baptist mission field from Adoniram Judson's arrival in 1813 onward,
plus Catholics and Anglicans; the census names no body, so nothing on the map says Baptist.
Chin and Kachin are the only states where something other than Buddhism holds a plurality or
comes close.

**Almost all of the country's traditional religion is in one state.** Shan holds **383,072 of
the 408,045 Animists, 93.9%**, and is 6.6% Animist against no other state above 1.9%. Read it
as a floor: nat propitiation is close to universal in Myanmar and normally accompanies Buddhism
rather than replacing it, and a census allowing one religion per person counts those people as
Buddhist.

**Islam is 2.3% of the enumerated population and that number needs its caveat every time.** The
enumerated Muslims are several unlike communities the census cannot separate — the Rohingya of
northern Rakhine who *were* enumerated, the Kaman, the Panthay of Shan State, and the
Indian-descended communities of Yangon and Mandalay. The 1,090,000 who were not enumerated are
counted separately and not as Muslims, for the reason in §2.

**No religion is 0.06% — the smallest such share of any country on this map by a wide margin**
— and 80% of it is in Shan State, the same state that holds almost all the Animists. Some of it
is likely traditional practice reported as "no religion" rather than irreligion, which is
Benin's `Aucune` warning again. The census does not say.

## 6. Boundaries and placement

`sources/mm_geo.md` has the detail. Two things worth having here:

**The counting geography is not the administrative geography, and the feature count says so
before any join.** COD's ADM1 has **18** features because the standard p-codes split Bago in
two and Shan in three; the census reports both whole. They are dissolved by rule, MIMU's own
aggregate codes (`MMR111`, `MMR222`) confirm the intent, and every multi-member group is
asserted contiguous.

**The placement grid is 2023 and the census is 2014, and in one state that difference is
visible in the diagnostic.** Kontur reads **0.58×** on Rakhine against the population drawn
there and **0.89×** against the enumerated count alone, in line with the other fourteen states.
The gap is the ~750,000 Rohingya who left for Bangladesh in 2017. So Rakhine's `unenumerated`
dots are weighted towards where people live now, which is south of the northern townships those
people actually lived in. Nothing published can fix it, a uniform spread would put them in the
Arakan mountains, and inventing a northern concentration would be §14.4.

## 7. Ethics (§14)

Myanmar is on §14.3's short list of genuinely dangerous countries, and the conversation happened
before the build rather than during it ([[feedback_flag_ethics_for_discussion]]).

**It is the *reflect* case of §14.2's reflect-vs-reveal test, and unusually clearly so.** The
only input is the state's own published table about its own territory, at the only resolution
that table has. No compilation of a government's published figures can tell that government
something it does not have, and 15 units of ~3.4 million people locate nobody. §14 rule 2 — no
resolution finer than the state's own publication — is satisfied by construction, because the
state published exactly one.

**The live question was not whether to draw the country but what to say about the people it
left out**, and the answer is the narrowest true thing: they exist, there are about this many
of them, and this map does not know what they believe. Anita's call.

## 8. Not done

* **Township or district religion.** Does not exist — §1.
* **DOP's own Islamic assumption**, taken to State level. §14 rule 1 — see §2.
* **The 2024 census**, conducted by the State Administration Council with partial coverage.
  Nothing on religion has been published from it and its enumeration is a different object
  from 2014's.
* **A current picture.** This is 2014: three years before the 2017 expulsions and seven before
  the 2021 coup. `note_public` says so.
