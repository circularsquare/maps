# Origin -> language: one shared table (sources/origin_mix.py)

Session edd42a8c-imm, 2026-10-05. Before this, each European build mapped immigrant origins in its
own way (India was Hindi in fr/es, Punjabi in it/pt/gr; Mozambique Emakhuwa in fr, Portuguese in
pt; Morocco's Berber share 22% fr, 20% es, 3% be, 43% nl). Now `fr es it be nl pt gr se kr jp`
all take `origin_mix.mix(iso, dest)`. Each build keeps its own retention step and placement.

```
python sources/origin_mix.py MA be          # one mix and where it came from
python sources/origin_mix.py --fragment fr  # regenerate fr's borrowed-node block in tree.d
```

## 1. The default: the home mix

Saudi Arabia's method (sources/sa.md §3): the origin's own drawn counts on this map, summed
nationally, languages under 1% left out, the rest scaled to 100%. Origins not on the map take
France's main-language table (`fr_build.COUNTRY_LANG`); small territories and dissolved states
(Yugoslavia: Serbo-Croatian; USSR: Russian; Czechoslovakia: Czech) are in `TERRITORY`/`PSEUDO`.

**Immigration countries as origins** (`NATIVE`): their home mix keeps only their own languages,
since the languages their immigrants brought are not what people born there speak. The Gulf
states keep only their citizens' Arabic; Germany and Austria German; France French and its
regional languages; and so on. Calls someone might reverse:
- the US keeps Spanish (13.6% of its home mix, much of it US-born), so Americans abroad are
  15% Spanish; dropping it is one word in `NATIVE`;
- Estonia and Latvia keep Russian, Cyprus keeps Turkish, Finland keeps Swedish (Sweden's build
  overrides Finland, below).

Group nodes in a home mix stay as they are: Afghanistan's map has 25% on the Iranian group
("language not named"), so Afghans abroad carry that too.

**Latin America** (session edd42a8c-latn, 2026-10-05): `mx cr pa hn ni` take `mix(iso, cc)` through
`sources/latam_immig.py`, which keeps Spanish-speaking origins on Spanish, drops indigenous-American
nodes and applies TeO2 retention (`sources/mx.md`, "Immigrant and settler languages").

## 2. Overrides (a diaspora known to differ, with a figure)

| origin | in | mix | source |
|---|---|---|---|
| Morocco | nl | Darija 54 / Tarifit 46 | NIDI, Hogendoorn, Demos 39(4) 2023, p. 6 (SIM/SING, first generation, language understood best: Arabic 51, Berber 43) |
| Suriname | nl | Sranan 67 / Sarnami 31 / Javanese 2 | same (64 / 29 / 2) |
| Iraq | nl | Iraqi Arabic 61 / Kurdish 39 | same (55 / 35) |
| Afghanistan | nl | Dari 83 / Pashto 17 | same (79 / 16) |
| Turkey | nl | Turkish 96 / Kurdish 4 | same (94 / 4) |
| Morocco | be | Tarifit 40, the rest at Morocco's mix without Tarifit | Reniers, "On the History and Selectivity of Turkish and Moroccan Migration to Belgium", International Migration 37(4), 1999: "More than 40 per cent of Moroccans living in Belgium reported having passed their youth in one of the two provinces of the Rif. Almost all immigrants from this region speak Tarifit." (King Baudouin Foundation 2009: half of Morocco-born Belgian Moroccans from the eastern region.) |
| Morocco | es | Arabic 80 / Berber group 20 | Idescat EULP 2023, Catalonia, first language 15+: Tamazight 45,600, Arabic 179,600 (es2021's figure, moved here) |
| Algeria | fr | Kabyle 30, the rest at Algeria's mix without Berber | Salem Chaker (Inalco/CRB): 30-40% of the Algerian-origin population in France Berber-speaking, mostly Kabyle; the low end (fr.md's figure, moved here) |
| Turkey | fr, be | Turkish 94 / Kurdish 6 | Center for American Progress, "The Turkish Diaspora in Europe" (2020), DATA4U survey Nov 2019 - Jan 2020, 2,357 people of Turkish origin in Germany, Austria, France, Netherlands: 6% self-identify primarily as Kurds (identity, used as language). Belgium also: Reniers 1999, Belgium's Turks mostly from central Anatolia |
| India | it | 70% at Punjab's 2011 census mix, 30% at India's | Barbara Bertolani (University of Trento) via The Tribune, 2019, "70% Indian migrants in Italy from Punjab"; CeSPI Brief 2/2023 also gives 70% |
| Suriname | fr | Ndyuka (the Maroon node) | INSEE, Pratiques culturelles en Guyane 2019-20: Maroon languages 8% of Guyane (23,520), against 28,062 Suriname-born there; nearly all Suriname-born in France live in Guyane |
| China | pa | Hakka 80 / Cantonese 10 / Mandarin 10 | Wikipedia, "Chinese people in Panama", Demographics: "Around 80% of this population are of Hakka origin, with the rest being Cantonese and Mandarin speakers" (origin used as language; its reference not checked). Session edd42a8c-latn |
| Iraq, Syria (2006 cohort), Turkey, Iran, Ethiopia; Finland; South Korea | se | Parkvall's splits and keep shares | stay in `se_build.py` (`splits`, `keep`, FINLAND_SWEDISH: 25% Finland-Swedes), since they need SCB's 2006 stocks. ROMANI_FROM countries lose their home-mix Romani in se, which Parkvall's Romani total already counts |

## 2b. Uncited overrides (Anita, 2026-10-05: obvious ones allowed without a figure)

Session edd42a8c-fix. Each is **uncited**: a judgement, no source gives the share.

| origin | in | mix | why |
|---|---|---|---|
| Belgium | fr | French | Belgians in France are overwhelmingly francophone (home mix was Dutch 59 / French 40) |
| Belgium | nl | Dutch | Flemish next door (home mix had French 40) |
| Switzerland | fr | French | Romands (home mix German 70) |
| Switzerland | it | Italian | Ticinesi (home mix German 70) |
| DR Congo | be, fr | French 40, the rest at Kinshasa's drawn mix (countries/cd.py CD1000), so Lingala ~40 / French 40 / Kikongo varieties, Luba and others ~20 | Kinshasa-heavy, French-speaking diaspora; replaces DR Congo's national mix (Luba-Kasai 14, Lingala 10) |
| India | pt, gr | as Italy's: 70% Punjab's 2011 mix, 30% India's | Punjabi Sikh farm labour by every account (§2's India/it row) |
| South Africa | nl | Afrikaans 75 / English 25 | home mix was Zulu 23 / Xhosa 16 / Afrikaans 14 |
| Canada | everywhere | English | home mix was English 74 / French 26 |
| Canada | fr, be | French | Québécois |
| Brazil | everywhere | Portuguese | already its home mix; made explicit |
| Venezuela | ar cl co br pe ec | Spanish | home mix had Wayuu 1.1% (Zulia, on the Colombian border); the millions who emigrated across South America are drawn as Spanish speakers (session edd42a8c-lats) |

**More immigration countries in `NATIVE`** (their home mix carried their own immigrants):
Denmark, Norway, Iceland (Polish 4%), Maldives (Bengali 15%), Dominican Republic (Haitian 6%),
Monaco, Bahamas (Haitian 11%), Cayman (Jamaican 25%), Turks and Caicos, Antigua, St Kitts,
Montserrat, Anguilla, BVI, Palau, Guam.

**The Gulf and Jordan builds** (`gulf_mix.origin_mix`, `sa_census.nationality_mixes`) did not use
this module for European and American origins: they took France's main-language table (Canadians
French, Swiss and Belgians French, South Africans Zulu) or, above 20,000 people, the origin's
whole drawn map (Americans with every immigrant language of the US). Now any origin with an
override here or in `NATIVE` (the Gulf states aside) goes through `origin_mix.gulf_route` ->
`mix()`. South Africans in the Gulf are still Zulu (France's table): no call made.

**Morocco in France** stays at the home mix (Berber 18%): Chaker puts Moroccan emigration to France
at 40-50% Berber but as an estimate; a footnote of his cites Tribalat's INED survey (MGIS 1992) at
28%, but it is unclear whether for Moroccans or all Maghrebis. Reniers says Riffians went to the
Benelux and Germany rather than France.

**Dropped, no figure behind them** (now the home mix; the obvious ones came back as uncited
overrides on 2026-10-05, §2b): India -> Punjabi in pt and gr; Lusophone
Africa -> Portuguese in pt; Belgium -> Dutch, South Africa -> Afrikaans in nl; Sri Lanka ->
Sinhala, Switzerland -> German, Belgium -> Dutch in it; Belgium, Switzerland, Canada, Cameroon,
Gabon, Côte d'Ivoire -> French in fr; Swiss/Belgian/Canadian/Ukrainian fixed splits in es/pt;
Peru, Bolivia -> Spanish in jp. Searched without a usable figure: Indians in Portugal (35,000 Sikhs
against about 100,000 of Indian origin, nothing by nationality), Moroccans in France (above).

## 3. Mechanics

- Each build writes origin rows as node ids; `fr2023 be2021 gr2021 it2025 nl2026 se2025 pt2021`
  `resolve()` pass any lower-case label through as a node, `es2021` does for its
  "Otra | nationality: <node>" rows. es folds every Arabic variety onto "Árabe" and Moldovan onto
  "Rumano" before dropping the languages a province's table already names (`es_ecepov.es_mix`).
- The host language's share of a mix goes to the host language without retention (Belgium's
  Dutch in nl, Mozambique's Portuguese in pt). nl's calibration of the common rate now runs on
  the non-Dutch part of each origin's mix.
- it, nl, gr, se dropped rows under half a person; with the long tails that lost up to 8,000
  people (nl), so those rows now go onto the host language.
- Every listed country's fragment ends with a generated block of borrowed nodes
  (`--fragment <cc>`), for the build tail's `--only`.
- Node counts grew (fr 135 -> 589 languages, es 73 -> 356, it 142 -> 542, be 122 -> 570, nl 129
  -> 446, pt 123 -> 591, gr 113 -> 423, se 129 -> 514): every origin now brings its home tail.
  Most are a handful of people and draw no dot.

## 4. Before / after, national totals of the languages that moved most

- **fr**: Arabic 747k -> 45k, now Algerian Arabic 392k, Tunisian Arabic 201k, Levantine 43k,
  Darija 393k -> 410k; Chinese group 77k -> 0 (Mandarin 52k, Wu, Cantonese...); Dutch 26k -> 76k
  and German 79k -> 110k (Belgians and Swiss off French); French 61.32M -> 61.17M; Tamil 40k ->
  13k (Sri Lanka now Sinhala/Tamil 75/25); Wolof 72k -> 40k; Turkish 189k -> 180k.
- **es** (only foreign nationals' "Otra" moves): Chinese group 194k -> 0 (Mandarin 110k, Wu 16k);
  Punjabi 83k -> 28k, Hindi 43k -> 12k; Wolof 66k -> 33k; Paraguayan Guarani 0 -> 40k, Quechua 0 ->
  20k; Hungarian 10k -> 38k (Romania's); Berber 140k -> 130k; share of "Otra" placed by
  nationality 96.8% -> 99.1%.
- **it**: Arabic 250k -> 8k (Egyptian 96k, Tunisian 92k, Darija 250k); Chinese group 239k -> 0
  (Mandarin 155k); Punjabi 252k -> 130k (70% Punjab override); Romanian 856k -> 748k (Moldovans
  now Moldovan, Romania's Hungarians); Ukrainian 211k -> 155k, Russian 39k -> 104k; Hungarian 6k
  -> 57k; Tagalog 113k -> 40k (Philippines' mix).
- **be**: Tarifit 5k -> 69k, Darija 137k -> 87k; Arabic 95k -> 31k (Algerian, Iraqi, Levantine,
  Tunisian split out); Lingala 48k -> 6k (DR Congo's mix); Italian 84k -> 67k (Italy's regional
  languages).
- **nl**: Moroccan, Surinamese, Iraqi, Afghan, Turkish splits unchanged (NIDI, now in
  origin_mix); Arabic 237k -> 185k; Ukrainian 125k -> 90k, Russian 41k -> 78k; Afrikaans 10k ->
  1k (South Africa's mix); Chinese group 21k -> 0 (Mandarin 15k).
- **pt**: Punjabi 13k -> 1k (India's mix); Guinea-Bissau Kriol 8k -> 0 (Fula, Balanta,
  Mandinka...); Chinese group 11k -> 0; Angolans now Umbundu 3k, Kimbundu 2k beside Portuguese;
  Portuguese 10.148M -> 10.136M.
- **gr**: Punjabi 29k -> 8k (India's mix, and Pakistan's now Punjabi/Pashto/Sindhi); Arabic 21k
  -> 7k; Pashto 0 -> 8k; Russian 9k -> 12k.
- **se**: Arabic 460k -> 329k, Levantine 0 -> 69k; Dari 83k -> 23k, Pashto 0 -> 40k, Iranian
  group 0 -> 21k (Afghanistan's mix); Hindi 70k -> 22k, Punjabi 40k -> 17k; Somali 95k -> 59k,
  Maay 0 -> 19k; Bosnian 79k -> 44k, Serbian 33k -> 55k (Bosnia's mix); Chinese 46k -> 0.
- **kr**: English 82k -> 70k, Spanish 3k -> 12k (Americans now 15% Spanish), French 0 -> 4k
  (Canadians); otherwise unchanged (it used home mixes already).
- **jp**: Spanish 55k -> 47k, Quechua 0 -> 5k (Peru's mix); Arabic 5k -> 1k; small moves.

## 5. Room for improvement

- Every §2b row wants a figure: a survey of Congolese in Belgium by home language, Indians in
  Portugal and Greece by state, South Africans in the Netherlands by language.
- Moroccans in France: a published MGIS or TeO table of Berber by country of birth.
