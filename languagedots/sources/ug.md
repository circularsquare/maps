# Uganda: 2024 census tribe by subcounty, read as home language through Afrobarometer

Built 2026-10-05 (session edd42a8c-ug). 44,378,756 people (the 2024 household population),
2,200 subcounty units, 50 answers, every row `modelled`. 44,356 dots at 1:1000. Tier D
(ethnicity only), built under AGENT_BRIEF §2's 2026-10-05 ruling with the retention check.

```
python sources/ug_nphc_extract.py   # ~25 min, one core: tribe counts by parish out of Anita's RAR
python sources/ug_afro.py           # Uganda's Afrobarometer rows (religiondots' .sav, read-only)
python sources/ug_census.py         # -> data/normalized/ug.csv (+ data/raw/ug/ug_retention.csv)
python taxonomy/build.py
python tools/check_country.py ug
python scatter.py --country ug
```

## 1. What was searched

- **No language question in 2024.** The 10% person file's 161 variables (names and labels in
  `data/raw/ug/ug2024_variables.txt`) hold religion (P9) and **tribe/nationality (P10)**,
  nothing on language. The coverage sweep found none in the reports either.
- **Ethnicity below national.** The published 2024 and 2014 reports give tribe nationally
  only (2014 Main Report Appendix A8; 2024 Final Report); the 2024 sub-region profiles carry
  no tribe table. The 2014 district reports were not searched further, because a finer
  source was already on disk: **the 2024 10% sample** that Anita downloaded from UBOS for
  religiondots (ask 008 there: "extract what is needed, keep the compact aggregate"). Its P10
  answer is known for every household member, by parish. `ug_nphc_extract.py` streams the RAR
  through UnRAR and keeps only parish x P10 counts (`data/raw/ug/ug2024_parish_ethnic.csv`,
  6 MB); nothing else is written.
- So the census + retention-from-survey route (Ghana's) is used at subcounty grain. IPUMS
  (district ethnicity 1991-2014) is not needed and the account is blocked anyway.

## 2. The census side

P10's codes: 500 "Uganda" (no tribe; 0 household members in the sample), 511-583 the 73 tribes
on the form, 701+ nationalities, 998 Unknown. 4,540,805 sampled household members, every one
with a P10 answer. Units: religiondots' 2,200 drawn units (2,207 subcounties; Bidi Bidi's three
camp subcounties merged with their hosts, its `CAMP_HOSTS` copied). Each unit's tribe shares come
from its own sample (about 2,000 people a unit) times its full-count household population
(religiondots' `ug.csv` Total population minus Not in a household). Checks: sample rows
4,693,190 and household members 4,540,805 equal religiondots' pass; sample units = base units
both ways; units sum to 44,378,756; drawn rows sum back to it. No split-half test was run on
tribe: the tribes are strongly local and dots at 1:1000 hide subcounty sampling noise in small
groups, but a test like religiondots' would say how far it holds.

Tribe shares (sample): Baganda 15.7%, Banyankore 9.4%, Basoga 8.4%, Iteso 7.1%, Bakiga 6.7%,
Lango 6.1%, Bagisu 4.7%, Acholi 4.4%, Banyoro 2.8%, Lugbara 2.7%; South Sudanese 1.0%.

## 3. The retention model (`sources/ug_census.py`)

Afrobarometer R4-R9 Uganda (12,031 respondents), tribe and home language. Each tribe's count
in a unit is shared over P(answer | tribe, region), five regions (Central, Kampala, East,
North, West; census district code's first digit, Kampala apart). The tribe-region share is
shrunk to the tribe's national share with K = 30 respondents, the national share to the
tribe's own language with K0 = 10, so a tribe the survey never met is drawn on its own
language. R7's REGION column is scrambled (its "Acholi" holds Buikwe and Kayunga; 44%
agreement), so R7 and R9 respondents are placed by their district.

**Lingua francas (ask 018; superseded by the ruling below).** As in Kenya and Tanzania the wording changed with R7.
Uganda by round: English 0.0, 0.0, 0.1% (R4-R6), 1.8, 1.6, 2.7% (R7-R9); non-Baganda naming
Luganda 0.0, 1.1, 0.8%, then 10.2, 3.5, 10.1%; in Buganda non-Baganda naming Luganda went from
6-8% (R5, R6) to 78% (R9). So, until Anita rules, **English, Swahili, and Luganda for anyone
not Muganda take their shares from R4-R6** (`LF_ROUNDS = [4, 5, 6]`; a regional adjustment
keeps Buganda's higher R4-R6 Luganda rate), every other answer from all rounds, scaled to what
the lingua francas leave. Drawn: Luganda 7.26M (16.4%), English 18,700, Swahili 44,000.
`LF_ROUNDS = [7, 8, 9]` gives Luganda 10.7M (24.2%), English 1.06M (2.4%), Swahili 124,000,
and Kinyarwanda falls from 540,000 to 98,000.

**Ruling, 2026-10-05 (session edd42a8c-r7e).** Anita ruled on ask 018: lingua francas at
Afrobarometer R7's **mother tongue** question (Q2A). `sources/ug_afro.py` now extracts Q2A for
R7 (not Q2B, "language spoken in home"), and `LF_ROUNDS = [7]`: English, Swahili and Luganda
for anyone not Muganda take their tribe-by-region shares from R7's 1,200 mother-tongue answers,
shrunk as before (K = 30 to the tribe's national share, K0 = 10 to the national rate). Every
other answer still pools all six rounds, R7 now on its mother-tongue answers. No R7 interviewer
recorded English or Swahili for more than 3 of their respondents.

| drawn | before (R4-R6 wording) | after (R7 mother tongue) |
|---|---|---|
| Luganda | 7.26M (16.4%) | 7.82M (17.6%) |
| English | 18,700 | 210,000 (0.47%) |
| Swahili | 44,000 | 73,000 (0.16%) |
| Kinyarwanda | 540,000 | 521,000 |

R7's mother-tongue answers put more non-Baganda on Luganda than R4-R6 did (retention in Central
now 0.63-0.86 for most groups, Bagwere 0.63-0.73 everywhere), and English at 0.5% where the
earlier rounds had almost none.

Retention found (own language, by region, `data/raw/ug/ug_retention.csv`): the big groups
90-99% (Lango, Alur, Basoga, Iteso, Acholi, Banyankore); lower: Banyole about 0.70 (Lusoga,
Luganda), Banyarwanda 0.72-0.80, Bafumbira 0.75 (Rukiga), Basamia about 0.77 (Lusoga,
Luganda), Bagwere about 0.80 (Lusoga), Kumam 0.74-0.81, Aringa about 0.80 (Lugbara; Aringa
was on the card only in R5). Baganda living in the West and East: 0.73-0.75 (Runyankore,
Lusoga).

## 4. Calls

- **Survey answers.** R5's "Luo" (120 of 123 Acholi) is the respondent's Luo language; R5's
  "Lunyoro" beside "Runyoro" (19 of 20 Banyole) is Lunyole. An "Other" with no identifiable
  verbatim goes on the respondent's own tribe's language (most such tribes, Kumam, Kakwa,
  Bagungu, were not on the card that round).
- **Tribe -> language** (`OWN`): dialect groups on their language per Glottolog (Bahororo,
  Banyabutumbi, Banyaruguru on Nyankole; Batuku, Banyabindi on Tooro; Jonam on Alur; Dodoth,
  Jie, Ngikutio, Mening on Karamojong; Bagabu on Soga; Bagwe on Saamia; Babukusu, Maragoli on
  Kenya's Luhya; Baziba on Haya). **Tepeth, Nyangia, Napore on Karamojong**: Soo is "moribund"
  and Nyang'i "nearly extinct" in Glottolog's endangerment field. Batwa on Rufumbira in Kisoro,
  Chiga elsewhere. Aliba on Ma'di (the survey's 9 Aliba: Madi 7, Lugbara 2). Ethur on Labwor.
  Gimara, Reli, Shana, Vonoma, Bakingwe (language not identified) and "Other Ugandan" on
  "Other Ugandan language" (`africa_other`), 406,000.
- **Non-Ugandans by nationality**: Rwanda, Burundi, Somalia on their languages; every other
  African nationality, South Sudanese (about 445,000) and Congolese (300,000) included, on
  "Other African language" (878,000); others on `other` (39,000). "Unknown" nationality shared
  like the unit's Ugandans.
- Rufumbira and Labwor drawn as siblings of Kinyarwanda and Acholi; Tagwenda, Songora, Nyara
  (no Glottolog entry) as Bantu leaves; Chope as a Nilotic leaf; Ik under a new Kuliak group in
  Nilo-Saharan; Nubi under Arabic.
- **Colours**: every new node hand-picked off the generated grid; a build without ug.txt gives
  every other node the same colour (Nyankole, pl.txt's, kept generated for that reason).
  Not yet looked at on the map.

## 5. Room for improvement

- Refugees: a nationality is not a language. UNHCR's settlement data by ethnicity or language
  would replace "Other African language" in West Nile and the south-west settlements.
- A split-half test of tribe shares at subcounty (religiondots' method) would say which small
  tribes should be drawn at county or district share.
- A census language question (never asked) would replace the survey's retention shares, which
  rest on 12,000 adults and are applied to children too.

## Terms

UBOS NPHC 2024 10% sample: Anita's registered download (religiondots ask 008); only aggregates
kept. Afrobarometer: free download, citation requested ("Afrobarometer Data, Uganda, Rounds
4-9, 2008-2022, available at http://www.afrobarometer.org"). Kontur CC BY 4.0 (religiondots'
hexes, read-only); Glottolog CC BY.

## Cross-border groups (2026-10-06, 5d7dac7e-xb)

Pokot and Kupsabiny are in the Kalenjin group with Kenya's; Teso and Karamojong in Ateker; Aringa under Lugbara; Konzo with the DRC's Nande; Kinyarwanda and Rufumbira in Rwanda-Rundi. Groups only (taxonomy/regroup.txt): no label's node or count changed, each keeps its leaf and colour; dots rewritten in place. The full table of cases is in followups.md (2026-10-06, languages that stop at a border).
