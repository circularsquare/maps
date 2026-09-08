# Ecuador — LAPOP AmericasBarometer, waves 2010–2023

Built 2026-09-08, the **third** country drawn from the AmericasBarometer after Guatemala and
El Salvador. `sources.md` §11ad assesses the source across all nine countries it could serve;
`sources/gt.md` carries the arguments all three share; `sources/sv.md` is the one whose join
is hard. Read §11ad, then this file's §3, before doing a fourth — **§3 is a defect in the two
countries already built.**

## 1. Why a survey at all, and a stronger negative than §11x's

`sources.md` §11x closed Ecuador along with Panama, Guatemala, Honduras and El Salvador for
having no census religion question. That was right, and this build corroborates it **from
INEC's own metadata rather than from reputation**, which is the standard §11ac set when it
closed Venezuela and Colombia on their own variable dictionaries.

The 2022 census person file in INEC's ANDA catalogue is
`CPV_Población_2022_Nacional_Sector Editado` and it publishes **88 variables**. None is
religion. The list runs `P01` household relationship, `P02` sex at birth, `P03` age, `P0701`–
`P0706` functional difficulty, `P08` place of birth, `P09` residence five years ago, `P1001`–
`P1005` languages spoken, `P11R` **self-identification by culture and customs**, `P12`
**indigenous nationality or people**, `P13` parents' indigenous language, `P15`–`P20`
education, `P21` device use, `P22`–`P30` work, `P31` marital status, `P32`–`P35` fertility,
then the derived columns. Ecuador asks what people call themselves, what they speak and whom
they descend from. It does not ask what they believe, and it never has.

Confirmed twice: the same 88 appear in `4_DICCIONARIO_DE_VARIABLES_CPV_2022.xlsx` on the
census site, which is a different publication of the same fact.

## 2. The files

| file | what it does |
|---|---|
| `sources/lapop.py` | the construction, the split-half and the category list, shared with `gt.py` and `sv.py` |
| `sources/ec.py` | LAPOP `.dta` → `data/normalized/ec.csv`, this country's decode, and the four provinces held out |
| `sources/ec_geo.py` | COD-AB ADM1 + **INEC's 2022 census** → `data/geo/ec/`, joined on the name with the code join asserted to agree |
| `sources/ec_grid.py` | Kontur 400 m hexes → `data/geo/ec/ec_hexes.gpkg` |
| `taxonomy/ec2023.py` | the eleven answers → the tree |

Acquisition of the `.dta` is Guatemala's and `sources/gt.md` has the walk-through. The two
Ecuadorian files are open: COD-AB from HDX (**736 MB**, because `cod-ab-ecu` ships ADM0 through
ADM4 and ADM4 is census sectors — only the seven `adm1` members are ever unpacked), and the
census tabulado from the census site's WordPress media library.

**The census tabulado was found through `wp-json/wp/v2/media`, not through a page.**
`www.censoecuador.gob.ec` lists **924 media items over 10 pages**, 86 of them spreadsheets,
and `01_2022_CPV_Estructura_poblacional.xlsx` is among them. That is the WordPress media API
habit paying off a third time. Note the host serves an **incomplete certificate chain**, so
`ec_geo.py` relaxes TLS for that one hostname and asserts the national total instead.

## 3. THE ANSWER CARD CHANGED AFTER 2016 — AND GUATEMALA AND EL SALVADOR ARE AFFECTED

**This is the finding worth carrying out of Ecuador and it is not about Ecuador.** Two of the
eleven answers are not available in every wave. Counting raw respondents across the whole
merge, all 28 countries together:

| code | answer | 2010 | 2012 | 2014 | 2016 | 2018 | 2023 |
|---|---|---:|---:|---:|---:|---:|---:|
| 77 | `Otro` | **0** | **0** | **0** | 812 | 431 | 1702 |
| 6 | Mormons | 105 | 80 | 153 | 145 | **0** | **0** |
| 10 | Jewish | 16 | 15 | 24 | 14 | **0** | **0** |
| 12 | Jehovah's Witnesses | 315 | 290 | 623 | 251 | **0** | **0** |
| | *valid answers that wave* | 29,374 | 27,254 | 36,725 | 18,203 | 15,107 | 25,649 |

**Zero Jehovah's Witnesses among 25,649 Latin Americans is not a measurement, it is a missing
box.** So is zero `Otro` among 29,374. LAPOP restructured the card between the 2016 and 2018
rounds: the named small denominations lost their own answers and fall into `Otro`.

What it does to a pooled figure:

* **`Testigos de Jehová` is a FLOOR.** Ecuador's own wave series is 1.80%, 1.85%, 2.08% and
  then a structural zero, and the pooled 1.43% is those three diluted by a fourth that could
  not record them.
* **`Otro` is absent at the early end and inflated at the late end**, 0%, 0%, 3.22%, 5.02%,
  and the 2023 figure contains the Witnesses and Mormons with nowhere else to go. The
  2016→2023 rise is 1.80 points against the 1.81% those three cells hold, which is suggestive
  on n=1,545 and is not a decomposition.
* The same for `Iglesia de los Santos de los Últimos Días` and `Judío`.

**Nothing leaves the partition.** Every respondent is drawn and the country total is right;
what moves is about half a percent of Ecuador between `christianity.witnesses` and
`other.ec`. It is **stated rather than corrected**, because correcting it means deciding how
the 2023 `Otro` decomposes and nothing published says. A share-within-offering-waves
renormalisation was considered and rejected: it double-counts, because the 2023 `Otro`
already contains the Witnesses whose own rate would be re-estimated beside it.

**Guatemala and El Salvador pool the same waves and neither `gt2023.py` nor `sv2023.py`
mentions this.** El Salvador's `Testigos de Jehová` (0.74%) and `Otro` (0.93%) and Guatemala's
(0.5% and 1.14%) carry the same distortion. Nothing has been changed in either country from
here — it is a judgement about two finished builds and it belongs to whoever revisits them.

## 4. The population is a census count, not COD-PS, and that is the first time in this set

Guatemala and El Salvador use OCHA COD-PS projections because neither country has counted
recently. **Ecuador counted 16,938,986 people in November 2022** and INEC publishes them by
province, canton, parroquia and five-year age band.

    COD-PS 2020 (projected off the 2010 census)   17,510,643
    INEC 2022 census (counted)                    16,938,986      −3.4%

**And the error is uneven, so it is a distortion of shape and not only of level:**

| province | COD-PS 2020 | census 2022 | |
|---|---:|---:|---:|
| Galápagos | 33,042 | 28,583 | −13.5% |
| Loja | 521,154 | 485,421 | −6.9% |
| Bolívar | 209,933 | 199,078 | −5.2% |
| Pichincha | 3,228,233 | 3,089,473 | −4.3% |
| Guayas | 4,387,434 | 4,391,923 | +0.1% |
| Manabí | 1,562,079 | 1,592,840 | +2.0% |

Pichincha down 4.3% while Manabí is up 2.0% is a six-point swing between the second and third
provinces of the country. §14.4 rule 1's promise — *every person drawn is a person somebody
counted in that unit* — is only true if the somebody counted them.

**It also fixes a unit with no polygon.** COD-PS 2020 carries **25 rows**: there is an `EC90`
*Zona no delimitada* holding 41,907 people, and COD-AB 2024 has no such boundary, because
Ecuador assigned those inter-provincial disputed zones by referendum in 2015–16. On COD-PS
those people would have had to be dropped or given a home by hand. The census has 24
provinces summing exactly to the national total, one per polygon.

The age bands come from the same workbook's sheet 2 and are asserted to sum to sheet 1's
totals on all 24 provinces with **no residual**, which is the check that the two sheets are
one tabulation rather than two universes.

## 5. The join is Guatemala's, and both keys are required

    LAPOP prov   901..924  =  900 + INEC's official province number
    COD-AB       EC01..EC24 =  the same number
    INEC census   the same names

So the code join and the name join are two independent signals over the same rows, and
`ec_geo.py` requires **both** — `gt_geo.py`'s construction, for `ni_geo.py`'s reason. El
Salvador is the counter-example and the reason this is checked rather than assumed.

**LAPOP labels its own province codes**, which is a better witness than either: `prov_es` in
the Grand Merge names 901 Azuay through 924 Santa Elena. **One alias is needed** — LAPOP
abbreviates 923 to `S.D. De los Tsáchilas` where COD and INEC write `Santo Domingo de los
Tsáchilas`.

**There is no code 920.** Galápagos is not merely unsampled; it is absent from the value label
set, so no Ecuadorian respondent could ever have been placed there. `ec_geo.py` asserts that
Galápagos is the *only* polygon with no LAPOP label.

## 6. The held-out check is the strongest in the set

LAPOP's weighted province distribution against the census population distribution:
**r = +0.994 over 23 provinces, and none of 20,000 random pairings reaches it** (best random
+0.954). The permutation is the test; the correlation on its own would say little with Guayas
and Pichincha dominating.

The age comparison is printed beside it and **decides nothing**, as in both earlier countries:
between-province variance in mean adult age over sampling variance is **F = 0.35**, so
Ecuador's twenty-three provinces are indistinguishable from twenty-three draws on one
distribution. `lapop.held_out`'s docstring has the argument; it was asserted on once, in
Guatemala, and withdrawn in El Salvador.

## 7. Which categories carry their own geography

The split-half runs on the **20 provinces present in both halves**, so the bar is
1.96/√19 = **+0.45**:

    Católico                       75.54%   +0.71   own geography
    Evangélica y Pentecostal       10.95%   +0.48   own geography — clears by 0.03
    Ninguna (creyente)              5.97%   +0.63   own geography
    ------------------------------------------------------------------
    Protestante Tradicional         2.60%   +0.16   national rate
    Otro                            2.05%   undef.  national rate — see §3
    Testigos de Jehová              1.43%   +0.34   national rate
    the five under 1%                               national rate (§11ad)

**Three categories, as in El Salvador**, and `Ninguna (creyente)` passes here at +0.63 where
Guatemala's failed at +0.21. So Ecuador is the second country in this set whose no-religion
geography is a measurement rather than a national rate spread flat, and it is the more
striking one: **12.7% of Esmeraldas and 10.3% of El Oro against 0.3% of Zamora Chinchipe**,
a spread of forty to one.

**`Evangélica y Pentecostal` clears the bar by 0.03**, which is the mirror image of El
Salvador's `Protestante Tradicional` missing it by 0.02. Both were left where the arithmetic
put them. A bar that moves when a value lands near it is not a bar.

**`Otro`'s "undefined" verdict is right for the wrong reason.** `lapop.stability` calls an
all-zero half "the strongest possible failure of the test", and here the early half is zero
because the answer did not exist in 2010 or 2012. Same verdict, different reason, and §3 is
why it matters.

## 8. Three provinces are assumed, one is left blank

Four provinces cannot be drawn on their own measured shares, and **they do not get the same
treatment, because they are not the same case. The line is whether anything measured the
place at all.**

| province | n | treatment | why |
|---|---:|---|---|
| Carchi | 20 | national rate | 2010 wave only, so the split-half cannot rank it twice |
| Pastaza | 32 | national rate | 2010 wave only |
| Orellana | 55 | national rate | 2010 wave only |
| **Galápagos** | **0** | **NOT DRAWN** | **LAPOP has no code 920. Not sampled: not offered.** |

The three assumed provinces are **466,909 people, 2.76% of Ecuador**, drawn at the national
rate on each province's own census population. The split-half needs a province in both halves
to rank it twice, so these drop out of the test that licenses the other twenty — §14.16's rule
about categories applied to units. **But one wave did measure them**, and that is enough to
anchor an assumption on. Anita, 2026-09-08: *"i feel like carchi is fine to assume and we can
just do it."*

**Galápagos is drawn empty**, 28,583 people and 0.17% of Ecuador, in `countries.py`'s `gap=`
rather than as a §3.5 undercount. There is no code 920 in the `prov_es` label set, so no
Ecuadorian respondent could ever have been recorded there and there is no reading of any kind
to anchor an assumption on. It keeps its polygon and its Kontur hexes; only the religion is
absent. Anita, 2026-09-08: *"galapagos maybe we just leave empty for now. no data."*

**Galápagos is also why the distinction is worth drawing at all.** It is globally famous, so
the national average there is a claim a reader will actually check — and the archipelago is a
migrant population from the coastal provinces, which run 67–72% Catholic against a national
75.3%, so the assumption would have been visibly and checkably wrong in the one place people
zoom to. The general form: **the national average is not a neutral fallback but a positive
claim that a place resembles its country, so it is most wrong exactly where a region is
distinctive — and distinctive regions are the prominent ones.** Its error is anti-correlated
with obscurity.

**Drawing the three on their own shares was the alternative and it is worse three ways at
once, all of them measured:**

1. **No signal to lose.** Their 95% half-widths on a share near 75% are ±19, ±15 and ±11
   points. Carchi reads 88% Catholic against 76% nationally, Pastaza 81%, Orellana 70%: none
   of the three is distinguishable from the national rate to begin with.
2. **A bias, not noise.** They would carry a **2010 level** while every other province carries
   a four-wave average, and Ecuador moved fast: Catholic identification runs 79.70% (2010),
   80.44% (2012), 74.31% (2016), 67.65% (2023). The 2010 wave is **+4.16 points** more
   Catholic than the pool, and all three would be pushed the same direction.
3. **A false zero, eight times over.** **Carchi's twenty interviews contain three of the
   eleven answers**: eighteen Catholics, one `Ninguna`, one agnostic, nothing else. Drawn on
   its own shares it is a province of 172,828 people with **zero Evangelicals in a country
   that is 11% Evangelical**, and hard zeros on eight of the eleven categories. A hard zero on
   a map is a strong claim that twenty interviews cannot make. Pastaza (26 Catholic, 3
   Evangelical, 3 `Ninguna` out of 32) and Orellana (38 / 11 / 3 plus 3 Protestant, out of 55)
   are the same problem one and two notches less severe.

**Guatemala drew Zacapa on n=40 and said so in `note_public`, which is the opposite call.**
The difference is that Zacapa appears in every wave and is therefore *inside* the split-half.
The line is not sample size; it is whether the country's own stability test could see the unit.

### The neighbour-average fallback was tested and it LOST

Before settling on the national rate, the obvious alternative was measured: predict a
held-out province from its **bordering provinces** rather than from the whole country.
Leave-one-out over the twenty measured provinces, mean absolute error:

| | national average | neighbour average | |
|---|---:|---:|---|
| Católico | 7.60 pp | 8.08 pp | 6% worse |
| Evangélica | 5.29 pp | 6.14 pp | 16% worse |
| Ninguna | 3.41 pp | 3.39 pp | 1% better, which is noise |

**Ecuador's religion does not vary smoothly across space.** It jumps at province lines,
because sierra, coast and Amazon interleave: Chimborazo is 85.9% Catholic beside neighbours
averaging 67%, and Tungurahua is 2.4% Evangelical beside neighbours at 10.6%. Spatial
smoothing has nothing to work with. Galápagos has no land neighbours at all, so it could not
have been reached this way in any case.

One gotcha for anyone repeating this: `geopandas`' `touches` returns **nothing** for five of
Ecuador's provinces, because COD's polygon edges do not share exact vertices. A 500 m buffer
on a metric CRS fixes it, and the first run of this test was wrong until it did.

## 9. What was drawn

**16,910,403 people, 23 provinces, 11 categories, 253 rows, 16,905 dots at 1:1,000.** Every
row `modelled` (§7b): nobody counted religion in Ecuador at any level. The drawn population
is 99.83% of the census; the missing 28,583 are Galápagos and are in `gap=` (§8).

    75.31%  Católico                                    christianity.catholic.latin
    11.09%  Evangélica y Pentecostal                    christianity.evangelical
     6.10%  Ninguna (creyente)                          unchurched
     2.58%  Protestante / Protestante Tradicional       christianity.protestant
     2.04%  Otro                                        other.ec
     1.43%  Testigos de Jehová                          christianity.witnesses
     0.67%  Agnóstico o ateo                            secular
     0.35%  Mormones                                    christianity.latterday
     0.35%  Religiones Orientales no Cristianas         other.ec
     0.06%  Religiones Tradicionales                    indigenous
     0.02%  Judío                                       judaism

**The most Catholic country the AmericasBarometer has drawn here**, thirty points above either
Central American one, and the map is about how unevenly that thins: **94.0% of Loja** against
**61.7% of Sucumbíos**. The evangelical map is nearly its negative (29.5% of Sucumbíos, 0.8%
of Loja), and the two big cities separate too — Guayas 67.1% Catholic and 18.7% Evangelical,
Pichincha 74.8% and 9.1%.

The view frames the mainland only. Including Galápagos would make the frame **2.8× wider** for
0.17% of the people, which is Chile's Easter Island decision met again — and here the
archipelago draws nothing anyway (§8), so the frame and the data agree.

## 10. What is claimed and what is not

**`Religiones Tradicionales` at 0.06% is called a floor**, which is Guatemala's treatment and
not El Salvador's. Ecuador is **7.69% indigenous** by its own 2022 census — 1,302,057 people
self-identifying as *indígena*, beside 1,305,000 *montubios* — and 0.06% is nineteen
respondents in 7,387. El Salvador's near-zero was left unqualified because its census agreed
with it at 0.2%; Ecuador's census does not agree, by two orders of magnitude. §11ad measured
this same box reading **0.21×** a census in Suriname.

**`unchurched` and not `unaffiliated`**, for `gt2023.py`'s reason: LAPOP prints *cree en un
Ser Superior pero no pertenece a ninguna religión* on the card. `secular` holds the 0.67% who
say they do not believe in God, and the two are nine times apart.

## 11. Open

* **The non-Christian tail**, as for every LAPOP-only country. `other.ec` is 2.40% across two
  answers and §3 says part of that is misfiled Witnesses. Nothing has been searched for yet.
* **§3 itself, in Guatemala and El Salvador.** Both are built and both carry the distortion.
* **INEC's own religion module: IT EXISTS, THE MICRODATA IS OPEN, AND IT IS NOT CLEAN.**
  The queue recorded it as *~2012, unverified*; it is verified now. `Filiación Religiosa`,
  ENEMDU December 2012, **13,211 respondents aged 16 and over**, weighted to 4,920,719 people.
  Found through the same WordPress route as §2 (`wp/v2/posts?search=religion`), and
  `bdd_filiacion_religiosa.zip` is a 175 KB open download containing `Religión.sav` with
  `RE02` *Cuál es su religión actual*, `CIUDAD`, `fexp` and a devoutness scale. No gate.

  **What makes it worth wanting is the card**, which is INEC's and not a worldwide one:
  Católica, Evangélica, **Islámica, Budismo, Judaísmo, Espiritismo, Religiones
  Afroamericanas**, Testigos de Jehová, Mormona, Pentecostales, Agnósticos, Ateos, Otra.
  That is the non-Christian tail LAPOP's card cannot see, in Ecuador's own words.

  **What stops it being wired in is three things, in order.** It is **five cities** — Quito,
  Guayaquil, Cuenca, Machala, Ambato — and **100% urban**, so it cannot draw a country; at
  best it is a second reading on five provinces. It is 2012, which is the wrong end of a
  period in which Ecuadorian Catholic identification fell twelve points. And **the coding
  looks unstable between cities**: `Otra` runs **19.35% in Cuenca and 10.58% in Quito against
  0.66% in Guayaquil and 0.21% in Machala**, while `Ateos` runs the other way, 1.35% in Cuenca
  against 11.99% in Guayaquil. A ninety-fold spread on a residual that trades off against
  irreligion city by city is a field-coding difference, not a geography — §9r's rule points at
  a missing category and this is the other thing that shape can mean. Anyone using it should
  treat `Otra` + `Ateos` + `Agnósticos` as one bucket until the questionnaire
  (`formulario_religion.pdf`, on the same page) says otherwise.

  So: a real second source for the **national** non-Christian tail, on a card that names five
  traditions LAPOP does not, and not a placement layer. The five PDFs beside it (presentation,
  bulletin, methodology, questionnaire) are unread.
* **A finer geography.** LAPOP's sub-provincial identifiers are a PSU list, not a partition.
  The 2022 census publishes at **1,000-odd parroquias**, which would be a superb placement
  layer if a religion source ever appeared to put on it.
* **Galápagos, which is currently blank.** It needs an Ecuador-specific source; LAPOP will
  never supply one, because the province is not on its card. INEC's 2006 Galápagos census
  (`censo_galapagos_2006.zip`, open on `ecuadorencifras.gob.ec`) was **not** checked for a
  religion question and is the obvious first place to look.
