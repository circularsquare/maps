# Sweden (se): the record

Drawn 2026-10-05 by session edd42a8c-se. 10,602,611 people (SCB, 31 Dec 2025; the 1,121 with
an unknown birth country not drawn), 290 kommuner, 129 languages, 10,550 dots. Every row is
`derived`.

Build: `python sources/se_scb.py --fetch` (SCB PxWeb API, no key, five JSON files into
`data/raw/se/`), `python sources/se_geo.py --fetch` (SCB 1 km grid, `data/geo/se/se_grid1km.gpkg`),
`python sources/se_build.py` (writes `data/normalized/se.csv`, prints every check below).
Mapping `taxonomy/se2025.py`, nodes `taxonomy/tree.d/se.txt`, entry `countries/se.py`.
Research downloads (Parkvall 2009 PDF, the Skolverket table, the 2025 Sami electoral roll) are
in the session scratchpad (`edd42a8c-se/research/`), not copied to the repo.

## 1. The rule and the method

Sweden has no language statistics of any kind. Anita's 2026-10-05 ruling for rich countries with
no language question (AGENT_BRIEF §2) applies: Swedish, plus regional/minority languages from
cited estimates, plus immigrant languages by origin, all derived.

The method is **Mikael Parkvall's**, the one Swedish source that estimates first-language
speakers language by language: *Sveriges språk – vem talar vad och var?* (Rapporter från
Institutionen för lingvistik vid Stockholms universitet 1, 2009, data about 2006, open on
DiVA: diva2:225395) and *Sveriges språk i siffror* (Språkrådet/Morfem 2016, data 2012; seen
only as the summary box Språktidningen published, 29 Mar 2016). He credits the foreign-born with
their origin language and the second generation by two Swedish studies, which is what this
build does with 2025 registers. It measures **first language (mother tongue)**, not the
home-use share France and the Netherlands used: no Swedish survey asks home language by
origin (SCB's integration reports, ULF/SILC and PIAAC were checked), so TeO was not borrowed.

## 2. Immigrants

**First generation.** SCB `FolkmRegFlandKCKM`: 290 kommuner x 187 named birth countries,
31 Dec 2025. SCB suppresses small kommun cells into "övriga födelseländer" (60,425 people, of
whom only 112 are countries SCB names nowhere); `unsuppress` spreads each kommun's övriga back
over the countries whose cell it published as 0, by IPF to each country's national deficit
(58,604; the 1,700 gap is SCB's cell-key perturbation). Kommuner rebuild to their totals within
52 people, which is SCB's own internal noise.

Country -> language: France's `COUNTRY_LANG` with `SE_OVERRIDES` (Yugoslavia-born ->
Serbo-Croatian, since SCB does not know the successor state; Serbia-and-Montenegro -> Serbian;
Soviet Union, Belarus -> Russian; Czechoslovakia -> Czech; Hong Kong -> Cantonese; Belgium ->
Dutch, Switzerland -> German, Canada -> English, unlike France).

**Splits** (`splits()`, printed): Parkvall 2009's 2006 speaker counts, less his share born in
Sweden, over SCB's 2006 birth-country stocks:

| born in | split |
|---|---|
| Iraq | Arabic 48.5%, Kurdish 32.8%, Assyrian Neo-Aramaic 16.7%, Iraqi Turkmen 2.0% |
| Syria (2006 cohort only) | Arabic 41.0%, Kurdish 27.8%, Turoyo 31.2% |
| Turkey | Turkish 56.1%, Kurdish 21.5%, Turoyo 22.4% |
| Iran | Persian 72.3%, Kurdish 20.4%, Azerbaijani 7.3% |
| Ethiopia | Amharic 72%, Oromo 28% |

Assumptions, each the build's own: Turkish 90% from Turkey ("nearly 90%"); Persian includes
Dari, so the 2006 Afghanistan-born come off before Iran; Azerbaijani all Iranian ("mainly");
Aramaic half Iraq (Parkvall), the other half Turkey:Syria **3:2** ("then Turkey, then Syria",
ratio chosen here); Turkmen all Iraqi; Kurdish takes Iran's and Turkey's remainders and the rest
of Parkvall's Kurdish total is shared by Iraq and Syria's 2006 cohort pro rata. Iraq's Aramaic
goes to Assyrian Neo-Aramaic, Turkey's and Syria's to Turoyo (Tur Abdin). Iraq's shares apply to
the whole 2025 stock (141,451). **Syria-born above the 2006 stock (176,838 of 194,606), mostly
2013-16 refugees, are drawn Arabic**: no figure splits their Kurds. Khayati (Linköping) reports
the Kurdistan list won >10,000 of 18,000 Iraqi votes in Sweden in 2005, so 33% Kurdish for Iraq
may be low.

**Kept / not kept, first generation**: Finland 75% Finnish, 25% Swedish (Parkvall: Swedish-
speaking Finns "more than a quarter" of the Finland-born); South Korea 19.3% (Parkvall's 1,900
Korean speakers over 9,862 Korea-born in 2006: adoptees); Ethiopia 37.8% (Amharic + Oromo
foreign-born speakers over the 2006 stock: adoptees). Everyone else 100%. Adoptees from India,
Sri Lanka, Colombia, Chile and China are not taken off (no per-language figure isolates them).

**Second generation.** `UtlSvBakgFinCKM` gives each kommun's Sweden-born with two (726,200) and
one (845,076) foreign-born parent; `FolkmForUrspCKMv2` gives their parents' countries nationally
(mixed-country couples half under each). Each kommun's second generation gets countries by IPF
seeded on its own foreign-born mix. They keep the parents' language at **78% (two parents) and
14% (one)**: Parkvall 2009 pp. 83-84, from Boyd 1985 (23.7% with Swedish as only mother tongue)
and Nekby & Özcan 2006 (20.9%); 86.5/84.6% for one Swedish parent. Third generation: Swedish.

## 3. National minority languages (carved out of Swedish)

- **Finnish**: no layer; generations 1-2 give 149,365. Parkvall 2012 175,000 (excluding
  Meänkieli; 200,000 with it); ISOF ~200,000. Lower because the Finland-born fell from 180,906
  (2006) to 122,462 and the third generation is drawn Swedish.
- **Meänkieli 30,000**: the middle of Parkvall 2009's 15,000-45,000 grew-up-as-active-users
  (p. 48); ISOF 2024 gives 50,000-75,000 in a wider sense, Parkvall calls 40,000 an upper limit
  for native speakers. 15,000 on the Five Kommuner (Haparanda, Övertorneå, Pajala, Kiruna,
  Gällivare; his "perhaps 15,000 native speakers" in the old Finnish-majority villages), 15,000
  elsewhere split 19:42 rest of Norrbotten : rest of Sweden by his phone-book surname count, by
  population inside each.
- **Sami 6,000**: ISOF Vanliga frågor (2024) and Parkvall 2016 both "about 6,000"; split
  North 4,500 / Lule 900 / South 600 by Parkvall 2009 p. 53 (three quarters, ~15%, ~a tenth).
  Placed by Parkvall pp. 54-55 (Sameutredningen, early 1970s): the ten kommuner with most
  speakers (Kiruna 1,449 ... Sorsele 177) and the län remainders, the latter on each län's Sami
  administrative-area kommuner (minoritet.se, 2025) by population. Kiruna, Gällivare and the
  rest of Norrbotten are North; Jokkmokk, Arvidsjaur, Arjeplog Lule; Västerbotten, Jämtland,
  Västernorrland and Dalarna South. 20.5% (Stockholm, Luleå, Uppsala, the south) is dispersed
  across all three. Ume and Pite Sami are not drawn (near extinct). The 2025 Sami electoral
  roll (9,755, by län only) is a check on the län pattern: Norrbotten 45%, Västerbotten 23%,
  Stockholm 8%, Jämtland 6%.
- **Romani 11,000**: Parkvall 2016 (2012), first-language speakers. ISOF's "about 80,000 speak
  Romani" counts use, not first language, and was not used. 16% born in Sweden (Parkvall 2009)
  comes from Swedish by population; the rest out of the first-generation rows of Finland,
  ex-Yugoslavia, Romania, Bulgaria, Poland, Hungary and Slovakia, placed by those
  countries' foreign-born per kommun.
- **Yiddish 1,000**: Parkvall 2009 (ISOF 2024: 750-1,500; "others say 3,000-4,000"; Parkvall
  2016: 3,000). On Stockholm, Göteborg and Malmö by population.

## 4. Geography

`data/geo/se/se_grid1km.gpkg`: SCB's open 1 km population grid, 31 Dec 2025 (`stat:befolkning_
1km_2025`, CC0; 103,464 populated squares, 10,578,118 residents), each square to the kommun its
centre falls in (religiondots' `se_lau.gpkg`, GISCO LAU 2021, read-only; 1,928 centres in the
sea go to the nearest kommun). Grid/table per kommun: p1 0.964, median 0.998, p99 1.047; worst
Sundbyberg 0.644 (a small kommun whose border squares centre in Solna and Stockholm) and Solna
1.118. Chosen over Kontur for Finland's reason (summer houses). Placement is population only.

## 5. Checks and result

check_country ok; scatter 10,550 dots on 6,147 squares, 38 rings, 52,611 people (0.50%) under
one dot. National: Swedish 73.1%, Arabic 4.3%, Finnish 1.4%, Spanish 1.2%, Polish 1.1%,
Kurdish 1.1%, Serbo-Croatian 0.9%, Somali 0.9%, English 0.8%, German 0.8%, Dari 0.8%, Persian
0.75%, Bosnian 0.74%; Assyrian 33,606, Turoyo 26,788, Meänkieli 30,000.

Against Skolverket's 2025/26 mother-tongue tuition eligibility (Tabell 8B, grundskola; 314,432
of 1,089,267 pupils eligible, ten largest languages only), as ratios to Arabic: Kurdish 0.21
(this map 0.25), Somali 0.25 (0.21), Tigrinya 0.14 (0.14), Persian 0.20 (0.17, Dari apart),
Polish 0.13 (0.25), BKS 0.22 (0.45), English 0.29 (0.20). The Polish and Yugoslav gaps are age
(older migrations, few school-age children), English the reverse. Against Parkvall 2012:
Arabic 155,000 then, 460,000 here (Syrian arrivals since), Kurdish 84,000 / 113,600, Aramaic
52,000 / 60,400, Turkish 45,000 / 50,000, Persian 74,000 / 79,500 + Dari 83,200.

## 6. Colours

New: Meänkieli (darker blue-teal than Finland's Finnish), North, Lule and South Sami spread
around Finland's Sami sea-green, Turoyo hand-set darker than the generated Assyrian (pale cyan,
0.03 apart before; both in Södertälje). Finnish, Swedish and Sami keep fi.txt's colours.

## 7. Room for improvement

- Parkvall's 2016 book has a 100-language appendix with all 290 kommuner; only its summary box
  was reachable. It would replace the 2006 splits and give Meänkieli and Sami by kommun.
- A split of the post-2006 Syrian arrivals (Kurds, Syriacs) and of the Iraq-born by cohort.
- Adoptees from India, Sri Lanka, Colombia, Chile and China drawn on the origin language.
- Skolverket's full per-language table (top ten only published) would check the second
  generation by language.
- Inside a kommun, everything follows population; Södertälje's Syriacs, Rinkeby's Somalis and
  the mountain Sami are not placed by group.
