# Unspecified Protestant / Evangelical splits: Latin America

Research 2026-10-16 (slug `latam`), then **built the same day on Anita's ruling**: Pew 2014
Pentecostal everywhere except Brazil, Baptist only in El Salvador and Panama, the unplaced rest
left on the bucket. The build is `sources/pew_latam.py` (fetches the Pew file into
`data/raw/pew_latam/` and writes `data/normalized/pew_latam.csv`) and `latam_split.py`; the record
is sources.md §latam-2026-10-16. All eighteen countries are wired. The
tables below were computed before the build; the drawn figures are in sources.md.

## 1. What sits on the unspecified nodes now

Read from `data/processed/counts.json` (thousands of people; the share is of the country's drawn
population). "Bucket" = `christianity.protestant` + `christianity.evangelical`.

| country | source | protestant | evangelical | bucket % | named Protestant rows already drawn |
|---|---|---|---|---|---|
| Mexico | INEGI 2020 | 9,494 | | 7.6 | Pentecostal 1,179, LLDM 190, Adventist 791, Presbyterian 429, Baptist 143 |
| Brazil | IBGE 2010 | 10,183 | | 5.8 | everything: AdD, CCB, IURD, Baptist, Adventist... (Evangélica não determinada 9.2M + Missão outras) |
| Argentina | ENCREA 2019 | 7,133 | | 15.5 | none |
| Colombia | LAPOP | 3,651 | 4,142 | 15.0 | none |
| Guatemala | LAPOP | 977 | 6,221 | 40.4 | none |
| Honduras | hn2019 | | 4,122 | 41.9 | Adventist 66 |
| Venezuela | LAPOP | 908 | 2,577 | 13.2 | none |
| Peru | INEI 2017 | 3,264 | | 14.1 | Adventist 353 (+381 "Cristiano" on bare `christianity`) |
| Chile | INE 2024 | 2,466 | | 16.3 | none |
| Dominican Rep. | do2019 | | 2,412 | 22.4 | Adventist 212 |
| El Salvador | LAPOP | 506 | 1,876 | 37.6 | none |
| Ecuador | LAPOP | 438 | 1,869 | 13.7 | none |
| Bolivia | LAPOP | 836 | 1,354 | 19.3 | none |
| Costa Rica | LAPOP | 463 | 696 | 23.0 | none |
| Panama | LAPOP | 130 | 896 | 25.5 | none |
| Nicaragua | INIDE 2005 | 981 | | 21.6 | Moravian |
| Uruguay | ENHA 2006 | 325 | | 10.1 | none |
| Puerto Rico | pr2018 | 297 | | 9.4 | none (+662 "Otros" on bare `christianity`) |
| Paraguay | DGEEC 2002 | | 186 | 4.8 | many, census is detailed |

About **69M people** in total. Belize, Guyana, Suriname and Haiti already name their churches
(Haiti draws Baptist and Pentecostal), so they are not in scope. **Cuba** has 630K non-Catholic
Christians on bare `christianity`, and no survey below covers Cuba.

The Dominican Republic sits next to Haiti, which is already drawn as Baptist / Pentecostal, so that
border is the starkest version of the Mexico/US problem.

## 2. Sources checked

### 2a. Pew Research Center, *Religion in Latin America* (2014). The usable one.

- Fieldwork Oct 2013 to Feb 2014, face to face, ~1,500-2,000 adults per country, 18 countries plus
  Puerto Rico. Protestants per country: 116 (Paraguay) to 638 (Honduras).
- `QPROT`, asked of every Protestant: *"Do you belong to (1) a historical Protestant church, for
  example Baptist, Seventh Day Adventist, Methodist, Lutheran or Presbyterian, (2) a Pentecostal
  church, for example Assemblies of God, or (3) another Protestant church?"* Then `QPENT<cc>` (which
  Pentecostal church, with a country card naming the main bodies) and `QDENOM<cc>` (which other
  church, with a country card: Baptist, Adventist, Methodist and so on, plus "none in particular" and
  "just a Protestant").
- Pew's "Protestant" covers historic, Pentecostal and other Protestants, Adventists included.
  Jehovah's Witnesses and Mormons are outside it, the same as on every source here. So Pew's
  universe is the map's bucket plus whatever named Protestant rows a country already has.
- **Access.** Pew's own download needs a free account. An open mirror with the identical .sav
  and codebook is on opendata.com.pk (6.2 MB, no account):
  https://opendata.com.pk/dataset/pew-research-center-religion-in-latin-america-data-set-survey-conducted-oct-2013-feb-2014
  It is the real file: weighted, it reproduces Pew's published chart "% of Protestants who belong to
  a Pentecostal denomination" for every country I checked (Dominican Rep. 71.6 vs 72, Brazil 59.7
  vs 60, Panama 58.4 vs 58, Mexico 46.1 vs 46, Guatemala 57.0 vs 57; identity Q28a, Dominican
  Rep., 65.9 vs 66). ARDA also lists it as `RELLA13`. Report PDF:
  https://www.pewresearch.org/wp-content/uploads/sites/7/2014/11/Religion-in-Latin-America-11-12-PM-full-PDF.pdf
  (chart on p.63, topline from p.164).
- **No geography in the public file** (columns are only `Country`, `CASEID`, `WEIGHT` besides the
  questions). So this gives **national shares only**.

Pentecostal-church share of Protestants (`QPROT`=2), weighted, from the file:

| DO 71.6 | AR 62.0 | NI 61.0 | BR 59.7 | PA 58.4 | GT 57.0 | EC 55.6 | PR 50.0 | HN 48.1 | CO 46.7 |
|---|---|---|---|---|---|---|---|---|---|
| **SV 46.3** | **MX 46.1** | **CR 45.5** | **CL 44.7** | **PY 44.3** | **BO 42.6** | **UY 41.8** | **PE 41.3** | **VE 39.2** | |

Named historic churches, % of Protestants (only where the country card offered them): Baptist
SV 17.2, PA 12.1, BR 10.8, HN 9.2, PR 8.8, CR 7.2, NI 6.3, MX 6.0, BO 5.8, AR 5.0, PE 4.8, CL 4.7,
VE 4.6, GT 3.9, DO 3.7. Adventist PE 8.4, BO 8.1, DO 5.5, BR 4.0, PR 3.7, SV 3.5, MX 3.4, PA 3.1,
GT 2.6, NI 2.1. Methodist CL 8.9, PA 4.2, CR 3.2, PR 3.0. Presbyterian MX 3.3, BR 3.2, GT 1.7,
HN 1.1. Anglican PA 4.5. "Nondenominational church" HN 8.7, VE 7.4, PR 5.5, GT 1.5 (offered in
those four only). Full table: scratchpad `latam/pew_composition.csv`.

The named Pentecostal bodies are all in the file as well: Assemblies of God everywhere (11.5-38% of
Protestants), Iglesia Pentecostal Unida in CO 11.2, EC 13.6, VE 8.0, NI 5.1 and SV 3.7 (Oneness),
La Luz del Mundo VE 7.6 (Oneness), Foursquare PA 17.7 and EC 5.7, Iglesia de Dios Pentecostal PR
31.5 and NI 12.2, Príncipe de Paz GT 9.8, Methodist Pentecostal CL 8.5+3.8, and so on. This would
support an optional second step: Assemblies of God and the Church of God bodies to
`pentecostal.trinitarian`, IPU and LLDM to `pentecostal.oneness`, the rest left on `pentecostal`.

### 2b. Checks against what the censuses already count

Pew's family share × the country's Protestant universe, against the census's own named row:

| check | Pew-implied | census | |
|---|---|---|---|
| Brazil Pentecostal | 28.3M | 28.7M (2010) | agrees |
| Brazil Adventist | 1.90M | 1.81M | agrees |
| Brazil Presbyterian | 1.52M | 1.15M | Pew high |
| Brazil Baptist | 5.12M | 4.12M | Pew high |
| Mexico Presbyterian | 403K | 429K | agrees |
| Mexico Adventist | 416K | 791K | Pew low (n=211 Protestants) |
| Peru Adventist | 304K | 353K | agrees |
| Dominican Rep. Adventist | 144K | 212K | Pew low |

Brazil is the strong test, because IBGE codes the church name itself: **Pew's Pentecostal share of
Protestants (59.7%) matches the census's Pentecostal share of evangelicals (60.0%)**, four years
apart. So Pew's `QPROT` answer behaves like a census church name, and it is the measure to use.
This is what makes the borrowed shape checkable, in the sense of the Bulgaria tests.

### 2c. Global Flourishing Study (on disk, CC BY), `REL3_Y1`: Mexico, Brazil, Argentina only

Card: Catholic, Orthodox, Anglican, Presbyterian/Reformed, Lutheran, Methodist, Baptist,
Pentecostal/Charismatic, Independent/Holiness/Evangelical, LDS, JW, Adventist, AIC, other, none in
particular. Weighted shares of non-Catholic, non-JW/LDS Christians:

| | Pentecostal | Indep./Evangelical | Baptist | Adventist | Presb. | none in particular | n |
|---|---|---|---|---|---|---|---|
| Brazil | 49.7 | 12.5 | 12.6 | 4.2 | 2.4 | 12.2 | 4,796 |
| Mexico | 16.1 | 9.8 | 7.6 | 4.4 | 3.8 | 45.9 | 1,244 |
| Argentina | 15.7 | 35.2 | 2.6 | 2.4 | 0.7 | 35.3 | 1,178 |

The GFS levels do not match the censuses (Mexico 58% Catholic against INEGI's 78%), and its "none
in particular" pool mixes nominal Catholics with Protestants, so only its shape is usable. **It
contradicts Pew for Argentina** (Pentecostal 16% of the pool against Pew's 62%). The likely cause is
the card: GFS offers "Independent/Evangelical", and Argentina's independent charismatic churches
take it, while Pew's three-way question puts them in "Pentecostal church". In Brazil the two agree
more closely (50 + 12.5 against 60) and the census sides with Pew. Mexico's three GFS regions
(n 278-521) are the only subnational cut, and they are too thin to draw.

### 2d. Latinobarómetro 2003-2024 (on disk, open). A dead end for denominations.

Its card has had Baptist, Methodist, Pentecostal, Adventist, Protestant and "Evangélica sin
especificar" boxes since 2003, and its `ciudad` code would give geography. But in practice
**80-97% of Protestants are coded "sin especificar"** in 15 of 18 countries (GT 94%, HN 91%, MX
66%, AR 90%, VE 96%), and the coding drifts between waves (Brazil Pentecostal 43% of Protestants in
2003-11, 18% in 2013-24; Colombia "Protestante" 49% then 26%). That is interviewer coding of
"soy evangélico", not denomination. Not usable, even pooled.

### 2e. LAPOP

Its card has only the two boxes (`Evangélica y Pentecostal`, `Protestante tradicional`), and
`q3cn`'s country codes add nothing Protestant (`sources/lapop.py` CATEGORY). No help with the split.

### 2f. INEGI 2020 (Mexico's own detail)

The 2020 classification has 46 denominations (local `clasificaciones_cpv2020_702825198701.pdf`
p.9-10): Asambleas de Dios 1312, Apostólica 1313, Iglesia de Dios 1314, ... 1318 other Pentecostal,
then 1341 **Cristiana**, 1342 **Evangélica**, 1343 Pentecostal, 1344 Protestante as generic answers.
The published table's "Pentecostal" row (1.18M) already contains the named Pentecostal bodies. The
bucket is 6.78M `Cristiana` + 2.39M `Evangélica` + 0.33M `Otro`, so 97% of it is people who
answered with a generic word. **The microdata cannot split them, because they named no church.** A
finer INEGI table would only re-divide the 1.18M already drawn.

### 2g. Rejected: the census's own named shape

Spreading the generic answers in proportion to the churches people did name (Brazil within a
municipio, Mexico within an entidad) fails the check. In Mexico's census-named Protestants,
Adventists are 26% and Presbyterians 14%, against Pew's 3.4% and 3.3% of all Protestants: members
of historic churches name their church, and the people saying "cristiano" are a different
population. In Brazil, Pew leaves no Pentecostal residual at all for the 9.2M undetermined (see
§3).

## 3. Proposal

**National Pew 2014 shares, applied with a residual rule**, spread flat inside each unit's counted
bucket (so the bucket's own measured geography still carries the pattern):

> for each family f: add = max(0, Pew share_f × U − already-named_f), where U is the bucket plus the
> country's named non-JW/LDS Protestant rows. The bucket is the ceiling. Whatever Pew could not place
> ("another church", "something else", "none in particular", "just a Protestant", don't know)
> **stays on the parent**.

Result (thousands; from scratchpad `latam/proposal.csv`):

| country | bucket | → Pentecostal | → Baptist | → other named | stays on parent |
|---|---|---|---|---|---|
| Mexico | 9,494 | 4,267 | 590 | 0 (Presb./Advent. already ≥ Pew) | 4,637 |
| Guatemala | 7,198 | 4,103 | 281 | Advent. 187, Presb. 122 | 2,397 (+108 nondenom., see Q) |
| Argentina | 7,133 | 4,422 | 357 | | 2,354 |
| Colombia | 7,793 | 3,639 | (not on card) | | 4,154 |
| Honduras | 4,122 | 2,014 | 385 | Presb. 46 | 1,312 (+364 nondenom.) |
| Dominican Rep. | 2,412 | 1,879 | 97 | | 436 |
| Venezuela | 3,485 | 1,366 | 160 | | 1,701 (+258 nondenom.) |
| Peru | 3,264 | 1,494 | 174 | | 1,597 |
| El Salvador | 2,382 | 1,103 | 410 | Advent. 83 | 786 |
| Ecuador | 2,307 | 1,283 | (not on card) | | 1,024 |
| Chile | 2,466 | 1,102 | 116 | Methodist 219 | 1,028 |
| Bolivia | 2,190 | 933 | 127 | Advent. 177 | 953 |
| Nicaragua | 981 | 643 | 66 | Advent. 22 | 250 |
| Costa Rica | 1,159 | 527 | 83 | Methodist 37 | 511 |
| Panama | 1,026 | 599 | 124 | Advent. 32, Meth. 43, Angl. 46 | 182 |
| Uruguay | 325 | 136 | | | 189 |
| Puerto Rico | 297 | 148 | 26 | small | 86 |
| Paraguay | 186 | 86 | | | 100 |
| **Brazil** | 10,183 | **0** | 998 | Presb. 368, Advent. 86 | 8,732 |

**Brazil is the exception and should not be split into Pentecostal.** The 2010 census already counts
as many Pentecostals as Pew finds (28.7M against 28.3M implied). The 9.2M "Evangélica não
determinada" line up with Pew's 17.6% "another church / don't know" (7.4M implied), so they are the
people who answer "evangélico" without a church, not hidden Pentecostals. Leave them on the parent,
or (see questions) treat them as Non-denominational. Brazil's small Baptist and Presbyterian residual
is within Pew's sampling noise, so I would not draw it.

**Tier.** For the census and census-like countries (MX, PE, CL, NI, HN, DO, UY, AR, BR), a survey
share applied inside a counted column is `derived`, not `modelled` (spec §2.7a: "inferred dots: not
shown" must roll these back to the counted bucket, not delete them). The LAPOP countries are already
`modelled` throughout, so nothing changes there.

**What would contradict it.**
- The census/Pew agreement table in §2b. Adding it as a build assertion would keep the check live:
  Brazil Pentecostal within ~5%, Mexico Presbyterian and Peru Adventist within ~25%.
- GFS for Mexico, Brazil and Argentina. It already disagrees on Argentina (§2c).
- A newer Pew round: a search summary (not checked against Pew's page) says a newer Pew Latin
  America update puts Brazil's Pentecostal share of Protestants at 65%, near 2014's 60%. If that
  round's file goes open, it is the second year.

**Caveats.**
- National shares only. **No open subnational denominational source was found anywhere in the
  region.** The dots will have the bucket's geography, with a flat mix inside it.
- Pentecostal is defined by church, not practice. Pew's "identify as Pentecostal" (Q28a) differs:
  for example Bolivia 21% identify against 43% by church, Brazil 71% against 60%. The proposal uses
  church, which is what a census codes.
- Sample sizes: Paraguay (116 Protestants), Argentina (205), Mexico (211) and Colombia (211) are
  thin. Their Pentecostal shares carry about ±7 points.
- Country cards differ: Baptist was not offered in Colombia, Ecuador, Paraguay or Uruguay, and
  "Nondenominational church" only in GT, HN, VE and PR. Drawing those families where the card had
  them would put a questionnaire border on the map (spec §3.1a). Pentecostal comes from the same
  question in all 19, so it is the safe family to draw.
- Peru's 381K "Cristiano" and Puerto Rico's 662K "Otros" on bare `christianity` could join their
  buckets, but neither is clearly Protestant, so I left them out.

## 4. Ranked by visual payoff (people moved × how jarring the border is)

1. **Guatemala**: 4.1M to Pentecostal, 23% of the whole country. It is in the middle of the
   Central American block.
2. **Mexico**: 4.3M, on the US border. The case Anita named.
3. **Dominican Republic**: 1.9M, 18% of the country, and it borders Haiti, which is already split.
4. **Honduras**: 2.0M, 21%.
5. **Argentina**: 4.4M, but the GFS conflict makes it the least certain of the big ones.
6. **Colombia**: 3.6M, then Venezuela (1.4M) and Ecuador (1.3M), with Oneness IPU a visible part of
   the CO/EC split.
7. **El Salvador** (1.1M + 0.4M Baptist), **Chile** (1.1M), **Peru** (1.5M), **Nicaragua** (0.64M,
   29% of its bucket).
8. Bolivia, Costa Rica, Panama, Uruguay, Puerto Rico, Paraguay: small.
9. **Brazil**: leave it (see §3).

## 5. Decisions only Anita can make

1. **Adopt Pew 2014's national Pentecostal share inside the counted buckets** (`derived` in census
   countries)? I'd do it for all 18, Brazil excepted.
2. **Pentecostal only, or Baptist / Adventist / Methodist too?** Pentecostal comes from one question
   asked the same way everywhere. The others depend on each country's card (no Baptist box in
   Colombia or Ecuador), so drawing them makes a questionnaire border. I'd draw Pentecostal only, and
   Baptist only in El Salvador and Panama where it is big (17% and 12%).
3. **What about the ~45% Pew could not place** ("another church", "just a Protestant", "none in
   particular")? Either leave it on "Protestant, unspecified" (honest; the flat colour shrinks by
   about half but does not go away) or send it to `christianity.nondenominational`, which is how the
   US draws its independent congregations. The second makes the US/Mexico palettes match better but
   asserts more. Brazil's 9.2M "não determinada" is the same choice.
4. **Argentina**: draw on Pew (62% Pentecostal) despite GFS saying 16%? I'd draw it, because Pew's
   measure is the one that matches Brazil's census.
5. **Second step**: split Pentecostal into Trinitarian (Assemblies of God etc.) and Oneness (IPU,
   La Luz del Mundo) where Pew names the body? It is visible in Colombia and Ecuador, and optional.

## Sources

- Pew Research Center, *Religion in Latin America* (2014), report: https://www.pewresearch.org/religion/2014/11/13/religion-in-latin-america/ ; chapter 4: https://www.pewresearch.org/religion/2014/11/13/chapter-4-pentecostalism/ ; full PDF above; dataset page https://www.pewresearch.org/religion/dataset/religion-in-latin-america/ (account); open mirror on opendata.com.pk (above); ARDA `RELLA13` https://www.thearda.com/data-archive?fid=RELLA13
- Global Flourishing Study wave 1, `data/raw/gfs/gfs_all_countries_wave2.csv` (OSF c8hbk, CC BY); REL3 codes as in `sources/tz.py`
- Latinobarómetro 2003-2024, `data/raw/latinobarometro/` (read with `sources/latinobarometro.py`)
- INEGI, *Clasificaciones del Censo de Población y Vivienda 2020*, `data/raw/mx/clasificaciones_cpv2020_702825198701.pdf`; religion tabulado `data/raw/mx/cpv2020_b_eum_12_religion.xlsx`
- CEIL-CONICET, *Primera Encuesta sobre Creencias y Actitudes Religiosas en Argentina* (2008), https://sociologiac.net/sociology/wp-content/uploads/2008/08/1encrel.pdf: it gives evangelicals at 9% and no Pentecostal split. The 2019 second round (15.3% evangelical, 2,421 cases) is published without one either (https://chequeado.com/el-explicador/creencias-en-la-argentina-hay-menos-catolicos-y-mas-evangelistas-y-personas-sin-religion/).
