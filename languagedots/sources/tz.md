# Tanzania: Afrobarometer R4 and R6-R9 home language, by region, placed by district

Drawn 2026-10-05 (session edd42a8c-tz). 61,741,120 people (NBS 2022 census region totals), 30
units (31 regions, Mbeya and Songwe as one), 90 nodes, every row `modelled`. 61,695 dots at
1:1000, 1 ring. Inside each unit the dots follow the survey's own respondents by COD-AB
district (170).

```
python sources/tz_afro.py --fetch   # Tanzania's rows from religiondots' merged .sav files
python sources/tz_place.py          # religiondots' hexes + each hex's district
python taxonomy/build.py
python tools/check_country.py tz
python scatter.py --country tz
```

## 1. Why a survey

Tanzania's census has asked neither language nor ethnicity since independence (the coverage
sweep: IPUMS 1988/2002/2012 hold only an English-literacy item; religiondots' `sources/tz.md`
on the policy of not counting). The Afrobarometer asks home language in every round and covers
every region, so this is the survey route of AGENT_BRIEF §2.

**The Languages of Tanzania atlas** (LOT project, University of Dar es Salaam, *Atlasi ya Lugha
za Tanzania*, 2009, ISBN 9789987691265) has first-language counts by region and district (papers
citing it quote, e.g., Luguru 22.9% of Morogoro region and 73.5% of Morogoro district). It is a printed book with no open release: WorldCat and Stanford list it,
ResearchGate has only papers citing it, and the one full PDF online (ebin.pub) is an
unauthorised copy, not used. What it gives here is one figure, through Malin Petzell, "The linguistic
situation in Tanzania" (University of Gothenburg; pdfs.semanticscholar.org/5635/0ab9b297cac9e92e59bd26c130fc03f36f30.pdf): **2,379,294 first-language Swahili speakers**,
about 7% of the 2002 mainland population. A scanned copy from a library would give a district
table that could replace or check everything below; it would need typing in.

## 2. How the counts are made (`sources/tz_afro.py`)

Each unit's count of a language = the weighted share of its pooled respondents naming it,
times its 2022 census population (religiondots' `tz_lookup.csv`, 61,741,120). Within-country
weights, asserted to average 1. 3 non-answers dropped.

**Rounds and units.** R4 2008 (1,208), R6 2014 (2,386), R7 2016, R8 2019, R9 2021 (2,400 each,
R8 2,398). **R5 (2012) is left out**, as in religiondots: its 26 pre-2012 regions cannot place
anyone in the five regions split in March 2012 and it has no district column. R4 is placed by
its district (religiondots' rule against COD-AB 2018 names; every district lands in its old
region or one carved from it; 136 respondents moved to a new region; Magu's 16 dropped, split
across regions). R8's merged file has REGION codes and no labels for Tanzania; decoded by code
from the other rounds (every code names one unit everywhere; R8's sample shares per unit
against R7/R9's: r 0.990). R6, R7, R9 districts agree with their region label for every
matched respondent. Respondents per unit: 104 (Kusini Unguja) to 1,201 (Dar es Salaam), median
324.

**Swahili (the call ask 018 may switch).** The question changed with R7: R4-R6 "Which language
is your home language?", R7-R9 "Language spoken in home". Swahili, weighted share of each
round: R4 9.0%, R5 3.4%, R6 5.5%, **R7 73.0%, R8 61.4%, R9 70.4%**. Inside groups the jump is
the same: Sukuma respondents naming Swahili 0-3% in R4-R6, 29-49% in R7-R9; Nyamwezi 1-2% then
66-81%. The early wording reads as a first language (and matches LOT's 7%); the later as the
language the household uses. Instructed (supervisor, 2026-10-05) to draw Swahili as given, so
**each unit's Swahili share comes from R7-R9** (`SW_ROUNDS = [7, 8, 9]`), and every other
answer's share from all five rounds among the non-Swahili answers, scaled to what Swahili
leaves. As drawn: Swahili 65.8% nationally; Dar es Salaam 97%, Zanzibar's five 98-100%, Simiyu
25%, Shinyanga 31%, Geita 36%, Kagera 40%, Mara 41%. **`SW_ROUNDS = [4, 6]`** (Kenya's choice)
gives Swahili 6.3%, Sukuma 20.5%, Ha 4.7%, Gogo 4.1%, Dar es Salaam 13%, Mjini Magharibi 87%.

**Card labels.** "Ki-" is the Swahili language prefix; each label is the language of that name.
Iraqw comes in four spellings plus Kimbulu (the Swahili exonym, "Wambulu"), merged. Kimeru is
Rwa (Mount Meru, Arusha), not Kenya's Meru; Kingoni is Tanzanian Ngoni, not the Nyanja-like
Ngoni of Zambia. Kichaga is one answer for Glottolog's Chaga subgroup and is drawn as one
language (Kenya's Luhya precedent). Kiarusha (Arusha Maa) is a sibling of Maasai, Kishirazi
(Zanzibar Swahili under its speakers' name) a sibling of Swahili, as Bajuni in Kenya. Kimanyema
has no Glottolog entry and is its own Bantu leaf.

**Free text** (`VERBATIM`, 433 answers). Spellings of card languages merged. Swahili varieties
(Tumbatu, Makunduchi, Pemba) counted in Swahili; Gunya on Bajuni (its own name). Digo on
Kenya's Mijikenda leaf. Ruri on Kwaya, Sizaki on Ikizu (Glottolog dialects). Barabaig, Taturu,
Mang'ati on Datooga. Kichasi/Chasi/Kiasi on Alagwa (the Wasi; R8's five gave ethnic group
"Mwasi"). 27 words not identifiable as a language (Kikine, 7 in Mara, the largest) and 9
languages named by a single respondent (Borana, Doe, Isanzu, Kisi, Lambya, Manda, Nyankole,
Taita, Vidunda) on "Other African language", 142,000 people. "Kihindi" (1) on `other`.

As drawn, nationally: Swahili 65.8%, Sukuma 12.1%, Gogo 1.7%, Haya 1.4%, Fipa 1.3%, Ha 1.2%,
Nyamwezi 1.1%, Jita 0.9%, Iraqw 0.9%, Nyaturu 0.8%, Nyakyusa 0.8%; 79 more under 0.7%.
(Before §2a. Now: Sukuma 20.0%, Ha 4.9%, Swahili 4.3%, Gogo 4.2%, Chaga 3.2%, Nyamwezi 3.1%.)

## 2a. Swahili at R7's mother tongue (2026-10-05, session edd42a8c-r7e)

Anita's ruling on ask 018: lingua francas are drawn at Afrobarometer R7's **mother tongue**
question (Q2A). R7's extract column is now Q2A, not Q2B ("language spoken in home"), and
`SW_ROUNDS = [7]`: each unit's Swahili share is R7's mother-tongue share there (13 respondents
in Kusini Unguja, median 69); every other answer still pools R4, R6-R9 (R7 now on its
mother-tongue answers), scaled to what Swahili leaves. Districts follow the same rule.

**Two interviewers left out.** R7 Q2A Swahili by interviewer: TAN15 99% (98 of 100), TAN25 85%,
while the other interviewers on the same teams, in the same regions and often the same
districts, recorded 1-20%. Their Swahili "mother tongue" respondents named Gogo, Ha, Chaga,
Zigua as their ethnic group, and Swahili landed on exactly a quarter of every district they
worked (Dodoma, Kigoma, Kilimanjaro, Manyara, Morogoro, Arusha, Tanga): one interviewer of four.
Their 196 R7 respondents are dropped (`DROP_INTERVIEWERS`, asserted in `drop_interviewers`).
With them, R7 Q2A Swahili is 11.9% (weighted); without, about 7% (157 of 2,204). The Zanzibar interviewers' 17-47% is real.

**Free text.** R7's mother-tongue "other" (15% of answers) often names the speaker, not the
language (Mzaramo, Muha, Mshirazi): `verbatim_key` reads it as the Ki- form. ~60 more spellings
are in `VERBATIM`, each checked against the respondent's ethnic group. New answers with two or
three respondents: Mpoto, Isanzu, Kisi, Lambya (new leaves in tz.txt), Arabic (the shared node).
The card's Kidigo, Mzigua and Kitumbatu (counted in Swahili, as the free-text Tumbatu was).

| | before (R7-R9 home, Q2B) | after (R7 mother tongue, Q2A) |
|---|---|---|
| Swahili, national | 65.8% (40.6M) | 4.3% (2.65M) |
| Swahili, mainland | ~65% | 2.3% (1.39M) |
| Dar es Salaam | 97% | 13% |
| Zanzibar's five | 98-100% | 38-100% (Shirazi drawn beside it, 0.4% nationally) |
| Sukuma, national | 12.1% | 20.0% |

**Check against LOT 2009** (2,379,294 first-language Swahili speakers, ~7% of the 2002
**mainland** population): the mainland as drawn is 2.3%, a third of LOT's share; R4/R6's wording
gave 6.3%. R7's mother-tongue item reads lower than LOT, and with ~70 respondents a unit many
inland units have no Swahili mother-tongue answer at all (0% in 15 units). Drawn as ruled;
flagged to the supervisor.

## 3. Placement

religiondots' `tz_hexes.gpkg` (Kontur 2023, 505,415 hexes), copied with each hex's COD-AB 2018
district (`sources/tz_place.py`: every hex's district in its own unit, all 170 hit, population
equal to religiondots'). 8,367 of 8,375 district-labelled respondents (R4, R6, R7, R9) join a
district of their unit by name (Swahili urban/rural words folded; aliases in `DIST_ALIAS`); R4's
8 "Kaskazini" (A or B unsaid) do not. 163 districts sampled. Per district: Swahili from R7/R9
respondents, other languages from all four rounds' non-Swahili respondents, each shrunk to the
unit share with 8 respondents' weight; the 7 unsampled districts borrow the inverse-square
distance mean of their 3 nearest sampled ones (`data/normalized/tz_district.csv`). A placement
weight only. It reads right where it can be checked: Mara's Rorya 56% Luo, Tarime 50% Kuria,
Butiama 22% Zanaki, Musoma rural 63% Jita.

## 4. Checks

| check | result |
|---|---|
| extract | question and ethnicity labels asserted per round; weights average 1.000 |
| units | 30 units, census total 61,741,120 exact; R8 code decode r 0.990; R6/R7/R9 districts 0 disagreements with region |
| split-half, R4+R6 vs R7-R9, non-Swahili answers, r across 28 units | Sukuma +0.93, Ha +0.99, Gogo +0.97, Haya +0.99, Iraqw +0.97, Chaga +0.98, Makonde +0.98, Fipa +0.87, Nyakyusa +0.91, Jita +0.92, Shambala +0.91, Nyaturu +0.96, Zigua +0.96. **Nyamwezi +0.27**: Tabora's Nyamwezi answered Swahili in R7-R9 (6, 24, 10 respondents left), so the later half barely sees it |
| LOT 2009 | Swahili first language 2.38M, ~7%: agrees with the R4/R6 wording (6.3%), not with what is drawn |

No retention check is needed: the language is asked, not inferred from ethnicity.

## 5. Mapping and tree (`taxonomy/tz2022.py`, `taxonomy/tree.d/tz.txt`)

Every branch checked against Glottolog; the fragment's header lists glottocodes. Bantu leaves
flat under Bantu as Kenya did, except Fipa, Pimbwe, Rungwa, Bungu, Safwa, Malila, which join
zm.txt's Mambwe-Nyiha (Glottolog's Mbozi holds them all), and Nyasa (Mwera of Lake Nyasa,
Nyanjaic) under Nyanja-Sena. New: Arusha and Datooga under Nilotic; a South Cushitic group
(Iraqw, Gorowa, Alagwa; colour 0.68 0.14 15); Sandawe on the existing isolate. Sukuma,
Nyamwezi and Nyiha, bare in other fragments, are coloured here. About 45 Tanzanian nodes
hand-picked for neighbours (fragment header); not yet looked at on the map.

## 6. Calls someone might reverse

- Swahili from R7's mother-tongue question (`SW_ROUNDS = [7]`, §2a, Anita's ruling); [4, 6]
  puts it near LOT's first-language figure, [7, 8, 9] with Q2B back in R7 gives the home reading.
- Two R7 interviewers dropped (`DROP_INTERVIEWERS`, §2a).
- R5 left out; pooling 2008-2022.
- Chaga as one language; Digo on Mijikenda; Kishirazi and Kiarusha as siblings.
- Adult survey shares applied to whole populations, children included.

## 7. Room for improvement

The LOT atlas's district table (library copy, typed in) would replace the survey for first
languages. Tanzania's DHS records language of interview only. The 2022 census did not ask.

## Terms

Afrobarometer data: free download, citation requested ("Afrobarometer Data, Tanzania, Rounds 4,
6-9, 2008-2022, available at http://www.afrobarometer.org"). NBS 2022 PHC: free publication.
COD-AB Tanzania (OCHA, CC BY-IGO); Kontur CC BY 4.0; Glottolog CC BY.

## Cross-border groups (2026-10-06, 5d7dac7e-xb)

Ha and Hangaza are in a Rwanda-Rundi group with Kinyarwanda and Kirundi; Nyakyusa is a group holding Malawi's Nkhonde. Groups only (taxonomy/regroup.txt): no label's node or count changed, each keeps its leaf and colour; dots rewritten in place. The full table of cases is in followups.md (2026-10-06, languages that stop at a border).
