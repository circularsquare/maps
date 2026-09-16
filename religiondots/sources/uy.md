# Uruguay — INE, Encuesta Nacional de Hogares Ampliada 2006

Built 2026-09-08. `sources.md` §9ce is the short version; `taxonomy/uy2006.py` is the mapping;
`countries.py`'s `uy` entry carries the reader-facing text.

    source      Instituto Nacional de Estadística, ENHA 2006, variable `e29_1`
    tier        19 departamentos, 167,000 drawn people each
    n           230,898 respondents with an answer, 3,252 (Flores) to 80,196 (Montevideo)
    universe    persons aged 7 and over in private households, whole national territory
    population  INE's own estimate at 30 June 2023: 3,496,400, of whom 3,228,150 are 7+ (§12.7)
    drawn       3,228,150 people, 7 categories, every row `modelled`

---

## 1. The queue was wrong about this country, and checking the office is why

`queue.md` §11ad priced Uruguay from the AmericasBarometer: **4,318 respondents pooled over
three waves, 19 of 19 departments**, and flagged it as the standout in the Americas at roughly
48% with no religion. That row is accurate about LAPOP and it is not the best source available.

**INE asked the question itself.** The *Encuesta Nacional de Hogares Ampliada* of 2006 is the
year the continuous household survey was widened from localities of 5,000 and over to the
whole country, small localities and rural areas included, and it carried a religion module
asked of every household member over six. That is **230,898 answers against LAPOP's 4,318** on
the same nineteen departments, collected by the national statistics office.

The census half of the question is genuinely closed and is not being implied:

  * Uruguay is **ABSENT from the UNSD oracle** (`python tools/oracle.py Uruguay`), so no
    census tabulation of religion was ever forwarded.
  * The last Uruguayan census to ask about religion was **1908**, nine years before the
    1917 constitution separated church and state. 2011 and 2023 do not carry it.

So this is a survey standing where no census has stood for over a century, and every row is
`modelled` in §7's sense.

**The general lesson, for the record**: the queue row said what the international survey had.
It did not say what the office had, because nobody had asked. One WebSearch for the office's
own household survey and one read of its microdata catalogue turned a 4,318-respondent build
into a 230,898-respondent one at the same tier.

---

## 2. The files, and INE's terms

`www4.ine.gub.uy` serves an **incomplete certificate chain**, so `WebFetch` and a plain `curl`
both fail with *"unable to verify the first certificate"* and the host reads as dead. It is
not. `curl -k` reaches it and everything below is open, with no login.

    catalogue   www4.ine.gub.uy/Anda5/index.php/catalog/48        (ANDA, i.e. NADA)
    DDI/XML     .../metadata/export/48/ddi                        824 KB, all 527 variables
    terms       .../catalog/48/get-microdata                      POST `accept=Aceptar`
    microdata   .../catalog/48/download/1166                      2006_SAV.rar, 31.8 MB

The terms page is a click-through, not an account. `POST accept=Aceptar` with a session cookie
returns the three download links (`.DAT`, `.DBF`, `.SAV`); the SPSS bundle unpacks to four
files and only `P_2006_TERCEROS.sav` (234 MB, persons) is needed. There is no `unrar` on this
machine; **`bsdtar -xf 2006_SAV.rar P_2006_TERCEROS.sav` works**, because libarchive reads
RAR3 and anaconda ships bsdtar.

**INE's terms, in full, because somebody has to know them.** They do not bar what this map
does, and one clause is an obligation rather than a permission:

  1. The data will not be redistributed or sold without INE's written consent.
  2. Scientific and statistical research only, and **for the presentation of AGGREGATED
     information only**, not for investigating particular individuals or organisations.
  3. No attempt to re-identify respondents.
  4. No linkage between INE datasets or between INE data and others that could identify
     individuals or organisations.
  5. Publications using the data must cite the source as INE requires.
  6. **"Una copia electrónica de todos los informes y publicaciones basados en los datos
     solicitados serán enviados al INE."** An electronic copy of any report or publication
     based on the data is to be sent to INE.

Nothing here redistributes the microdata (`data/` is gitignored) and the output is a
departmental aggregate, which is exactly what clause 2 permits. **Clause 6 is the live one**:
if this map is ever published as a paper or a print product, a copy owes to INE. That is a
note for Anita rather than a blocker, and it is not the Nişanyan case, where the terms banned
the retrieval itself.

Geography and population:

    boundaries  HDX cod-ab-ury, ury_adm_2020_shp.zip           1.0 MB, ADM0-2 + lines
    population  www5.ine.gub.uy .../Revisión 2025/A.1.2 Departamentos.xlsx    0.6 MB
    grid        Kontur kontur_population_UY_20231101.gpkg.gz   3.8 MB, 46,436 hexes

`www5.ine.gub.uy` has the same broken chain and `sources/uy_geo.py` turns verification off for
that host only, the way `ec_geo.py` does for `censoecuador.gob.ec`. Nothing is trusted on the
strength of the transport: the workbook's own national total is asserted.

---

## 3. THE CODE JOIN IS EL SALVADOR'S TRAP AND IT IS HARDER TO SEE

**INE numbers the departments with Montevideo first and the other eighteen alphabetically**:
1 Montevideo, 2 Artigas, 3 Canelones, … 19 Treinta y Tres. That is `dpto` in the microdata and
it is also LAPOP's, whose `prov` for Uruguay is 1400 plus the same number.

**COD-AB's `UY01`..`UY19` are alphabetical with no Montevideo exception.** UY01 is Artigas and
Montevideo is UY10. So `UY{dpto:02d}`:

    INE  1 Montevideo    -> UY01, which COD says is Artigas
    INE  2 Artigas       -> UY02, which COD says is Canelones
    …
    INE 10 Maldonado     -> UY10, which COD says is Montevideo
    INE 11 Paysandú      -> UY11, which is Paysandú        correct, and so are 12 to 19

**Nine of nineteen coincide and they are the last nine.** That is worse than El Salvador's
two-of-fourteen, because a spot check naturally lands on a name near the end of the list:
Salto, Soriano, Tacuarembó and Treinta y Tres all pair correctly. Montevideo's 1.3 million
people, 37% of the country, would have been drawn in Artigas, and a permutation preserves
every total, so no reconciliation would have caught it.

`sources/uy_geo.py` joins on the **name** — all 19 match COD's spellings exactly, no aliases —
and `check_code_join()` asserts that the code join still mispairs exactly ten, so an OCHA
re-cut to INE's order stops the build instead of silently changing which department is which.

**Three witnesses, all required:**

  1. every INE name is a COD name and every COD name is an INE name, with no leftovers;
  2. the code join mispairs, in the exact pattern above;
  3. **LAPOP's own `prov_es` labels** agree with INE's numbering on 18 of 19. The nineteenth,
     `1407`, carries an **empty** label rather than a wrong one, and it is Flores. §11ad
     already noted the blank; this is what it is.

A fourth witness is the population workbook joining on the same names, with Montevideo coming
out both the densest and the largest department, which a permuted population join would break.

---

## 4. The universe is ages 7 and over, and the metadata says otherwise

ANDA's data dictionary for `e29_1` gives the question as **"PARA MAYORES DE 6 AÑOS ¿como se
definiria desde el punto de vista religioso?"** and then glosses the universe as *"solo se
preguntará a personas de 6 años y más"*. Those are two different universes, four thousand
people apart.

**The microdata settles it and the questionnaire is right.** The not-applicable code 0 covers
ages 0 to 6 exactly and completely — 3,136 zero-year-olds, 3,349 one-year-olds, … 4,225
six-year-olds, 25,963 in total — and nobody aged 7 or over carries it. `sources/uy.py` asserts
this in both directions on every run.

So the drawn universe is **7 and over, 92.33% of Uruguay**, and 7.67% is children rather than
a non-response cell (91.01% and 8.99% until §12.7's fix). **There is no non-response cell at all**: every person in the universe
carries one of the seven answers, which is unusual on this map.

The children are in `gap=` with the figure, following Peru's under-twelves rather than Chile's
15-and-over (which states the cut in `basis` and `how` and leaves `gap` for non-response).
Nothing scales the shares up to them. The nearest measured band, ages 7 to 11, is 42.3%
Catholic against 46.0% for the whole universe, so a flat scale-up would have drawn Uruguay's
small children about four points more Catholic than the survey's own youngest cohort.

---

## 5. The population is INE's own, and the 7+ cut needs one interpolation

Uruguay counted in 2023, so §9bn applies and COD-PS is not the neutral choice: its Uruguay
file projects off the 2011 census. INE's *Estimaciones y proyecciones, revisión 2025* is the
post-census series and `A.1.2 Departamentos.xlsx` gives one sheet per department, both sexes
and five-year age bands, 1996 to 2023.

Two years are read, not one:

  * **2023** is the magnitude the map is drawn on: 3,496,400 people, 3,228,150 aged 7+.
  * **2006** is what the survey's own departmental weighting is checked against. Comparing a
    2006 survey's unit shares with a 2023 population would test seventeen years of internal
    migration rather than the join: Montevideo is down 64,657 over that span, Canelones is up
    96,535 and Maldonado up 59,437.

**Nothing INE publishes for Uruguay is by single year of age.** The 2025 revision's
`(100ymas)` files are five-year bands too; the phrase refers to the open top band. So the 7+
cut is `total − (0-4) − 0.4 × (5-9)`, ages 5 and 6 being two fifths of the band (it was 0.6
until §12.7). The 5-9 band is 6.60% of the country; the census 2023 persons file puts
Montevideo's under-7 share at 6.95%, and this cut gives 7.16%. Stated because it is arithmetic
on a published band rather than a published number.

**One trap in the workbook.** Eighteen sheets write the 2023 column header as the integer
`2023` and **Paysandú writes it as the string `2023*`**, a footnote marker nothing in the
workbook explains. A header parser that types the cell drops the year the whole map is drawn
on, for one department out of nineteen, and reports it as a missing column rather than as a
wrong number. `read_pop()` parses the header instead.

---

## 6. Two checks, and the second one is a second survey

**The population check.** The ENHA's own weighted department distribution against INE's 2006
estimate: **r = +1.0000 over 19 departments**, and none of 20,000 random pairings of the same
units reaches it (best random +0.996). The thinnest department relative to its population is
Salto at 0.943x and the fullest Rivera at 1.029x, which is a survey designed to be
representative at exactly this tier. `sources/lapop.py`'s small-country failure mode (its
docstring puts the bite at roughly seven to ten units) is nowhere near: 19! is 1.2e17.

`held_out()` is written in `sources/uy.py` rather than imported from `lapop.py` for one
reason: that module's printout says the word LAPOP, and a line of output naming the wrong
instrument is a false provenance on screen. That is the argument `sources/ec.py` made when it
added `pop_source`. Nothing else about it differs, and nothing in `lapop.py` was changed.

**The cross-check, which is the one no LAPOP-only country can run.** The AmericasBarometer
measured the same nineteen departments four to eight years later with a different instrument,
a different sample and a different sponsor, so its departmental ordering is an outside witness:

    non-Catholic Christian / Ev+Prot    r = +0.86    0 of 20,000 random pairings reach it
    atheist / agnostic                  r = +0.73    3 of 20,000
    no religion, both cells             r = +0.56  118 of 20,000
    Catholic                            r = +0.34  1,574 of 20,000   NOT significant

**The Catholic row failing is a fact about the pair of cards, not about either survey.** LAPOP
offers *Ninguna (cree en un Ser Superior pero no pertenece a ninguna religión)* and INE offers
*Creyente sin confesión*; the same idea, worded differently, and the boundary between that
answer and *Católico* is exactly where wording moves people. LAPOP finds 37.0% Catholic and
32.4% in that cell where INE finds 46.0% and 26.9%. On top of that, LAPOP has **45 respondents
in Rivera against INE's 9,322**, so most of the scatter is LAPOP's sampling error. The two
agree about which departments are Protestant and which are atheist, where neither card is
ambiguous, and that is the part the map draws.

**Read the other way, this is also a measurement of LAPOP** and it is worth keeping: on the
one country where a large national instrument can be laid beside it, the AmericasBarometer's
departmental cut reproduces the ordering of a big categorical difference well and the ordering
of its own ambiguous boundary cell badly.

---

## 7. Which categories carry their own geography

§14.16's split-half, run across the two halves of 2006 rather than across waves. A one-year
survey has no waves, but the ENHA is a monthly panel with its own semester weights, so
January-June (n=115,718) and July-December (n=115,180) are two samples of the same country.
Bar is 1.96/√18 = **+0.46** on 19 departments.

    category                   national    rank   shares   chi-sq p   verdict
    Catolico                     45.97%   +0.92    +0.90    0.0e+00   own geography
    Creyente sin confesion       26.87%   +0.88    +0.87    0.0e+00   own geography
    Ateo/agnostico               15.72%   +0.88    +0.92    0.0e+00   own geography
    Cristiano no catolico        10.07%   +0.85    +0.96    0.0e+00   own geography
    Umbandista/afroamericano      0.64%   +0.78    +0.83   1.4e-170   own geography
    Judio                         0.41%   +0.26    +0.93   7.9e-128   UNDER THE BAR, drawn
    Otra                          0.31%   +0.26    +0.45    6.8e-20   UNDER THE BAR, drawn

**The two that fail are the two that are near zero in eighteen of nineteen departments**, and
for those the rank test is ranking which department happened to catch one respondent in a
half-year. It is not testing the thing the map draws.

`Judio` is the one that matters. **Montevideo is 0.92% Jewish against 0.06% across the other
eighteen departments and holds nine tenths of the cell**, on 80,196 Montevideo respondents and
a 95% half-width of 0.07 points. The two halves of 2006 agree on the *shares* at +0.93 while
disagreeing on the ranking of the near-zeros. Drawing this cell at the national rate would put
two thirds of Uruguay's Jews outside Montevideo, which is a stronger claim than the one the
rank test declines to license, and a false one.

`Otra` is 0.31% and ranges 0.04% (Tacuarembó) to 0.60% (Canelones). Drawn as measured, with
the doubt recorded rather than hidden: this box collects both real minority religions and the
interviewer's give-up answer, so how often it is reached for is partly a property of the
fieldwork team, and those are regional. At three people in a thousand the choice moves nothing
a reader can see.

**Both are in `sources/uy.py`'s `UNDER_BAR` with the reason printed on every run, and the bar
was not moved.** The precondition is the chi-square, per `lapop.stability`'s own docstring: a
category whose departments do not differ significantly has nothing to draw and gets no
exception. Both differ at p < 1e-12, and the module refuses the exception if that ever stops
being true.

So **nothing in Uruguay is drawn at a national rate**, which is not true of any of the three
AmericasBarometer countries.

---

## 8. What was drawn

As drawn, off `data/normalized/uy.csv` (department shares on 2023 populations, so up to a
quarter of a point from the survey's own weighted national shares):

    46.185%   1,469,614   Catolico                    -> christianity.catholic.latin
    26.900%     855,973   Creyente sin confesion      -> unchurched
    15.566%     495,317   Ateo/agnostico              -> secular
    10.024%     318,977   Cristiano no catolico       -> christianity.protestant
     0.626%      19,921   Umbandista/afroamericano    -> afrodiasporic
     0.381%      12,130   Judio                       -> judaism
     0.316%      10,065   Otra                        -> other.uy

**42.47% claim no religious affiliation**, and 63% of those still say they believe in God.
The two halves have opposite maps:

    department        n     Cath  nonCath  believer  atheist   none
    Paysandú      8,950     67.7     11.6      15.5      5.0    20.5
    Colonia      10,026     59.6     18.1      13.6      8.6    22.2
    …
    Montevideo   80,196     43.0      8.2      24.8     21.7    46.5
    Tacuarembó   11,785     35.8     13.8      41.9      8.3    50.2
    Maldonado     8,936     42.3      5.6      34.6     16.6    51.2
    Río Negro     6,212     36.3     11.7      40.5     11.0    51.5
    Treinta y Tres 6,566    38.2      7.6      41.3     12.5    53.9
    Rocha         6,118     33.8      7.5      41.6     16.5    58.1

`secular` is Montevideo and the Atlantic coast: 21.7% of the capital and 16.6% of Maldonado
against 3.9% of Artigas. `unchurched` is the interior and the north: 41.9% of Tacuarembó and
41.6% of Rocha against 13.6% of Colonia. Catholicism runs 67.7% in Paysandú to 30.3% in
Rivera, and non-Catholic Christianity 27.9% in Rivera to 5.6% in Maldonado.

**Uruguay's card gives Judaism and Afro-American religion their own boxes**, which no LAPOP
card does, so `other.uy` is 0.32% where `other.gt`, `other.sv` and `other.ec` run 1.45% to
2.35%. `e29_2` is a free-text follow-up for `Otra` and it is transcribed in the microdata, so
this is the one `other` cell on the map whose contents can be read: a third blank, a fifth
`NO SABE` / `SIN DEFINICION` / `NO DEFINIDO` / `NO SE DEFINE`, then BUDISTA at 6.7% of the
cell, ESPIRITISTA, METAFISICA, PANTEISTA, MORMON, MUSULMAN, TESTIGO DE JEHOVA, BAHAI,
SEICHO-NO-IE, DEISTA, ORTODOXO RUSO. The largest named group is 682 people, under one dot at
every value this map offers, so nothing is split off (§3.11).

---

## 9. What is claimed and what is not

  * **Claimed**: the departmental composition of Uruguay's population aged 7 and over as INE's
    own household survey measured it in 2006, on INE's own 2023 population.
  * **Not claimed**: the level today. Uruguay has moved since 2006, and the AmericasBarometer's
    2010-2014 rounds find 37% Catholic against this survey's 46%, so the map is several points
    more Catholic than the country now is. The `note_public` says so.
  * **Not claimed**: anything about children under 7. They are in `gap=` and nothing scales up
    to them.
  * **Not claimed**: any body inside `Cristiano no catolico`. INE's own definition of that cell
    is *"la iglesia evangélica (incluyendo pentecostal y bautista), los protestantes,
    adventistas y armenios"*, so it is Protestantism of every kind plus the Armenian Apostolic
    Church, and the Waldensians of Colonia Valdense — the oldest Protestant body in the country
    — are invisible inside it.
  * **A floor, probably**: `Umbandista/afroamericano` at 0.63%. Uruguayan practice of Umbanda
    and Batuque runs well ahead of identification with them, as it does in Brazil, and a
    household survey asking a single self-definition question catches only the people who put
    it first.

---

## 10. Open

**The best available improvement is Montevideo, and it is a big one.** The ENHA person file
carries `ccz` (18 Centros Comunales Zonales) and `barrio` / `nombarrio` (62 Montevideo
neighbourhoods) beside `dpto`, both fully populated for all 88,917 Montevideo rows. The barrio
sample sizes are **381 to 4,780, median 1,192**, which is far more than any LAPOP country has
per department. Montevideo is 37% of the drawn country and holds the map's most striking
contrast (21.7% atheist against 3.9% in Artigas), so splitting it into 62 units would change
what this map is worth more than any other single move available here.

Three things to settle before doing it, none of them blockers:

  1. **The weights are calibrated at department and Montevideo-stratum level, not at barrio.**
     `Estrato` has four Montevideo classes (Bajo, Medio-Bajo, Medio-Alto, Alto) plus Periferia,
     and those correlate strongly with barrio, so barrio estimates should be well-behaved, but
     they are not design-representative and the write-up has to say so.
  2. **There is no replication check inside Montevideo** except the two halves of 2006, which
     is the same split this build already uses.
  3. **Polygons.** INE's *Mapas de Unidades Geoestadísticas 2023* are 19 per-department RARs of
     print maps rather than vector data; Intendencia de Montevideo publishes barrio and CCZ
     boundaries as open data and that is the place to look
     (`catalogodatos.gub.uy`). The 62 INE barrio codes are the standard Montevideo set.

**Outside Montevideo the ENHA offers nothing finer than the department.** `locagr` /
`nom_locagr` has 19 values nationwide and they are strata labels rather than places —
*Agrp. Localidades Menores*, *CANELONES Rural*, *Tacuarembó y otras* — so they are not a
partition and cannot carry a map.

**Second: the level is nineteen years old.** Nothing since 2006 has asked. Worth a look, in
order of likelihood:

  * later ENH/ECH rounds for a repeat of the module (2006 is the only one located);
  * the ENAJPD or any INE thematic survey with a beliefs block;
  * Latinobarómetro and the World Values Survey, both of which carry Uruguay with religion but
    at national level only, so they would date the level rather than place it.

**Third, and smaller: `Umbandista/afroamericano` wants a second source.** Uruguay's 2011
census asked about Afro-descent (`ascendencia`) and INE publishes it by department; that is
not religion and must not be used as if it were, but it would say whether the 0.63% is a floor
by a factor of two or of ten. The Atlas Sociodemográfico de la Población Afrodescendiente is
the obvious place.

---

## 11. Review, 2026-09-08 — nothing needs rebuilding, and one sentence in §6 is wrong

A second pass over the primary material rather than over this file. `check_md.py`,
`built_countries.py --check`, `check_rollup.py uy` and `check_mapping.py uy` are all clean;
the map draws, on land, concentrated where Uruguay's people are.

### 11.1 The §3.5 lean check was not run, and the hole does lean

§3 of this file states the size of the gap and §4 states its composition, but nobody had
correlated the excluded residual against the drawn categories, which is what §3.5 asks for.
Run on the ENHA's own code-0 rows: the child share of a department runs **8.69% (Colonia) to
12.06% (Artigas)**, and against the drawn shares,

    Ateo/agnostico             r = -0.51      (the 5% bar on 19 units is 0.46)
    Creyente sin confesion         +0.44
    Judio                          -0.39
    Cristiano no catolico          +0.38
    Catolico                       -0.33

So the excluded children are thinnest exactly where the atheist share is highest, and
dropping them leaves the drawn map very slightly more secular than the country. **The
magnitude is negligible and that is the point of measuring it**: applying each department's
own composition to its whole population instead of its 7+ population moves `secular` from
15.566% to 15.521%, and every other category by under 0.03 points. The age half of the same
lean is larger and §4 already has it: the 7-11 band is 42.3% Catholic, 30.1% believer without
a denomination and 12.1% non-Catholic Christian against 46.2 / 26.9 / 10.0 for the whole
universe, so children resemble the youngest measured cohort and pull away from Catholic
rather than toward it. Nothing corrects for either (§14.4) and neither is worth a sentence in
`note_public` at this size; recorded so the next person does not have to run it.

### 11.2 The cross-check: the mechanism holds, the weak cell is not the Catholic one, and the sampling-error clause is wrong

§6's explanation for the Catholic row was tested rather than accepted, and it survives the
test that could have killed it. If the disagreement is a *transfer across the Católico /
believer-without-a-religion line*, then the two cells should agree badly on their own and
their SUM should agree well. That is exactly what happens:

    INE Catolico              1 | LAPOP 1      r = +0.34   1,574 of 20,000 reach it
    INE Creyente sin confes.  5 | LAPOP 4      r = +0.25   2,987 of 20,000
    the two summed          1,5 | LAPOP 1,4    r = +0.69      12 of 20,000

Two cells that are individually poor and jointly good cannot be produced by noise, which
would degrade the sum as well. Per department the two errors run at **r = -0.84**: where
LAPOP finds fewer Catholics than INE it finds correspondingly more believers without a
religion, from -23.8 / +18.6 points in Florida to +9.9 / -2.1 in Tacuarembó.

**Two corrections to §6, both of which matter more for `gt`, `sv` and `ec` than for Uruguay.**

*First, the failing cell is `Ninguna (cree en un Ser Superior)`, not Católico.* On its own it
comes in at **+0.25, worse than the Catholic pair**, and it is the cell those three countries'
`unchurched` geographies rest on. §6 and `sources.md` §9ce both frame the finding as "the
Catholic row", which points a future reader at the wrong half of the boundary.

*Second, "most of the scatter in the Catholic comparison is LAPOP's sampling error" is not
true, and the arithmetic says the opposite.* Correcting each pair for LAPOP's own binomial
noise (n per department 45 to 1,723, median 118) moves the failing pairs least:

    Catolico            +0.34 -> +0.46        Cristiano no catolico  +0.86 -> +0.93
    Creyente sin conf.  +0.25 -> +0.32        Ateo/agnostico         +0.73 -> +0.79

Noise attenuates every pair, and it attenuates the two that fail *less* in relative terms than
the two that pass, because Catholic has the largest between-department spread of the four
(INE sd 0.103 against a LAPOP noise sd of 0.048). Sampling error therefore cannot explain why
one cell uniquely fails. The Rivera n=45 sentence is a true fact doing no work.

**What this does and does not say about `gt`, `sv` and `ec`.** It does not overturn them. The
cross-instrument correlation conflates the wording difference with four to eight years of real
change, and this comparison cannot separate the two, because secularisation and a moved
question boundary both move people across the same line in the same direction. What it does is
converge with something already in the tree: **Guatemala's own split-half gave that cell +0.21
and it is drawn on `sources/gt.py`'s `OVERRIDE`, under the bar, on Anita's call.** El Salvador
got +0.74 and Ecuador +0.63 for the same cell, so their internal evidence is good. The useful
statement is narrower than §6's and survives both readings: *LAPOP's `Ninguna (creyente)` cell
is the one whose departmental geography is least reproducible, internally in `gt` and against
an outside instrument here, and passing its internal split-half does not establish that it
holds the same people another instrument's version of the answer would hold.*

### 11.3 `UNDER_BAR`: `Judio` is sound, `Otra`'s stated precondition is contaminated

The `Judio` exception is right and I would have made it. Chi-square, a Pearson shares
correlation of +0.93, 80,196 Montevideo respondents, and a substantive fact that is
independently true; drawing it at the national rate would be a stronger and false claim. Keep.

`Otra` is thinner than its own reasoning admits, and the chi-square is the part that does not
hold. Of the 526 `Otra` respondents, **273 are blank or a give-up string, 56.6% of the
weighted cell** — §8's "a third blank, a fifth *NO SABE*" is right and adds to more than half.
The give-up share per department runs **0.007% in Tacuarembó to 0.308% in Canelones, a
forty-fold range**, and Canelones is also the department §7 names as `Otra`'s maximum. So the
chi-square that licenses the exception is significant partly on variation in how often
interviewers wrote nothing down, which is not a fact about religion. The precondition is the
right *kind* of gate — it correctly refuses an exception for a category that is pure noise —
but for this cell it is not the load-bearing argument. The load-bearing argument is the one
already in `UNDER_BAR`: 0.316% of the country, 10,065 people, and the choice moves nothing a
reader can see. Left as drawn on that ground; the chi-square framing is the part to distrust
if this exception is ever cited as precedent.

### 11.4 The join, and the assertion

Verified from the primary material rather than from §3. The `.sav`'s own `dpto` value labels
match `uy_geo.INE_DEPARTMENTS` on all nineteen with no disagreement, so the numbering the
whole country rests on is the file's and not a transcription. `data/geo/uy/uy_lookup.csv` as
built pairs all 19 correctly, and INE 1 Montevideo really does sit on `UY10`.

`check_code_join()` fires and is not vacuous. Called with the real COD names it prints the
ten mispairings and passes; called with an OCHA re-cut to INE's numbering it raises at 19 of
19; called with two pcodes swapped it raises at 7 of 19. It catches both directions.

### 11.5 Figures and terms

Every reader-facing figure in `countries.py` recomputes from `data/normalized/uy.csv`, and
every superlative in it is a genuine extreme of the drawn data: Paysandú highest Catholic and
Rivera lowest, Rivera highest non-Catholic and Maldonado lowest, Montevideo highest atheist
and Artigas lowest, Tacuarembó highest believer-without-religion and Colonia lowest, Rocha
highest and Paysandú lowest on the two combined, Montevideo highest on both Judaism and
`afrodiasporic`. Montevideo holds **90.2%** of the Jewish cell against `note_public`'s "nine
tenths". The one figure to know about is `afrodiasporic`: Montevideo 0.932% and Rivera 0.780%
are the top two as the note says, but **Canelones is 0.751%**, so the note's pairing is true
by 0.03 points rather than by a distance.

INE's terms were re-fetched from `www4.ine.gub.uy/Anda5/index.php/catalog/48/get-microdata`
and §2 quotes clause 6 **verbatim**; the other five clauses are accurate paraphrases, and
clause 2's *"exclusivamente para la presentación de información agregada"* is on the page as
described. The deposit obligation is real and stands.

---

## 12. Montevideo at its 62 barrios, 2026-09-15 (session cb8b206e-uy)

The §10 upgrade, built from the ENHA file already on disk. `sources.md` §uy-2026-09-15 is the
short version. `sources/uy.py`, `uy_geo.py` and `uy_grid.py` carry the checks in code.

### 12.1 What changed, and what did not

- Montevideo's department row is gone; 62 barrio rows replace it (`geo_level` `barrio`,
  `geo_id` `UY10-B01` to `UY10-B62` on INE's own barrio numbers). `tools/check_mapping.py`'s
  `DEFAULT_LEVELS` has `uy: departamento, barrio`.
- **The other 18 departments' 126 rows are identical, row for row**, to the build before
  (checked against a copy taken first). Montevideo's total (1,191,552 people aged 7+) and the
  country's (3,181,997) are unchanged; rounding drift goes into Montevideo's largest cell. (Both
  totals are from before §12.7's fix: 1,207,523 and 3,228,150 after it.)
- National shares move by at most 0.06 points, because the city is now a sum of barrio shares
  on barrio populations: Catholic 46.19% to 46.12%, believer without a religion 26.90% to
  26.96%, atheist 15.57% to 15.54%, non-Catholic Christian 10.02% to 10.07%, Jewish 0.381% to
  0.364%, Umbanda 0.626% to 0.632%, other 0.316% unchanged.

### 12.2 The grain: barrio for six answers, and why not CCZ

The file places every Montevideo respondent in a barrio (62), a CCZ (18) and a sección censal
(25 plus a `99` catch-all of 5,478 respondents, not a place). **CCZ is not a coarser barrio**:
36 of 62 barrios cross a CCZ line, so the two are rival partitions and neither nests in the
other. Secciones were not tested.

**Why the test resamples census segments and not people.** Religion runs in households: the
within-household intraclass correlation is 0.66 for Catholic, 0.48 for atheist and 0.82 for
Jewish, with 2.69 answering people per household, a design effect of about 2. Montevideo's
80,196 respondents are 29,766 households in 1,025 census segments (`secc` with `segm`); no
household spans two months and no segment crosses a barrio line; 6 to 46 segments per barrio,
median 16. The file has no PSU or zona column, so the segment is the finest cluster it names.

`barrio_stability()` is `sources/pr.py`'s construction: 400 random halvings of each unit's
segments, median Spearman of the halves' unit shares, against 2,000 dealings of the segments
into random groups of the same sizes (`stability.cluster_null`), with `chi2_p` and `CELL_CAP`
as vetoes. Unweighted counts. Seed `STAB_SEED`, so it reproduces.

    category                   n      barrio  null95  p       | CCZ     null95  p
    Catolico                   32,949 +0.486  +0.155  <0.001  | +0.709  +0.267  <0.001
    Cristiano no catolico       7,387 +0.444  +0.163  <0.001  | +0.754  +0.271  <0.001
    Judio                         444 +0.462  +0.180  <0.001  | +0.763  +0.288  <0.001
    Umbandista/afroamericano      923 +0.533  +0.171  <0.001  | +0.750  +0.296  <0.001
    Creyente sin confesion     21,349 +0.591  +0.154  <0.001  | +0.804  +0.271  <0.001
    Ateo/agnostico             16,883 +0.406  +0.150  <0.001  | +0.728  +0.263  <0.001
    Otra                          261 +0.020  +0.179   0.465  | +0.231  +0.271   0.085

Every chi-square is under 1e-9 and no segment holds more than 3% of any answer. **Six of seven
carry their own barrio shares; `Otra` fails at both grains** and no barrio tops both halves
in more than 2% of halvings, so it takes Montevideo's weighted share (0.36%) in every barrio
and the six are scaled together to fill the rest, which moves no carried share by more than
0.55 points. Barrio rather than CCZ: every answer that passes at CCZ passes at barrio, and
the barrio is the unit the city and its census use. The medians are lower at barrio (+0.41 to
+0.59 against +0.71 to +0.80), which is the price of the finer grain and is said in the note.

**What this does not fix.** The weights are calibrated to department and to `Estrato` (four
Montevideo classes; 56 of 62 barrios span more than one; `pesoano` runs 7-8 in *Bajo* to 26-27
in *Alto*), not to barrio, so a barrio's share is a domain estimate the design did not target.
The drawn shares are weighted; the test is not.

### 12.3 Witnesses on the barrio join

- The ENHA's own `nombarrio` best-matches the census name of its own code for all 62. The
  .sav writes Ñ as Ð (`BAÐADOS DE CARRASCO`), which `fold_name` handles.
- **Held out**: each barrio's weighted share of the ENHA's Montevideo against its 2023 census
  share, r = +0.941 over 62, best of 20,000 shuffles +0.584. Seventeen years of movement are in
  it: Bañados de Carrasco sits at 0.32x its 2023 share, Jacinto Vera at 1.24x. Not checked
  further.

### 12.4 Populations: the 2023 census, weighted, under INE's own Montevideo total

- **Cuadro 15** (census 2023, *Población por barrios*, weighted): 62 barrios, 1,359 people *sin
  dato de barrio* who live on the street, 1,302,721 in all. It prints names and no numbers.
- **The persons file**, ANDA catalogue 781, `download/1503`, `personas_ext_07_2026.rar`,
  135,339,245 bytes holding one 1.9 GB CSV, streamed through bsdtar and never unpacked. Its
  terms are catalogue 48's seven clauses word for word (§2), so clause 6's deposit covers it
  too. Only a 64-row aggregate is kept (`data/raw/uy/censo2023_mvd_barrio_edad.csv`).
- **Traps in that file**: the CSV is latin-1; the header names are quoted and the values are
  not; the weight is a bare `W` that a search for *peso* or *pond* misses; age is `PERNA01`
  (questionnaire p.4, question 19) and the DDI's range for it is from a small extraction.
- **The weight is not optional.** Weighted, the file reproduces Cuadro 15 within 0.53 of a
  person in every barrio, and 3,499,451 nationally, the published census count; `9898` weights
  to exactly the 1,359. Unweighted it runs 0.78 (Villa García) to 0.99 (Punta Carretas) of the
  table, lowest in the poorest barrios, where the census's omission correction was largest.
- **Why age by barrio matters**: the under-7 share runs 3.97% (Tres Cruces) to 10.68%
  (Casavalle). INE's single Montevideo 7+ ratio would have drawn barrios 3.1% under to 4.2%
  over their own 7+ population, centre against periphery.
- Each barrio takes INE's revision-2025 Montevideo total at 30 June 2023 (1,300,670, of whom
  1,207,525 are 7+ since §12.7's fix, 1,191,553 before) split by its census share; the 7+ figures are rounded by largest remainder
  to the department's own rounded total.

### 12.5 Polygons, and COD's Montevideo is not INE's

- The Intendencia's WFS layer `zon_v_sig_barrios` (435 KB, free use under resolution 640/10)
  against INE's own `ine_barrios_mvd_nbi85` from the 45 MB *mapas vectoriales 2011* zip: the
  same numbers 1-62, IoU 0.990 or better on every barrio. One lineage, so only the small one
  is fetched. Cuadro 15's names pair to it one to one by best match; the narrowest margin is
  Cerro against Cerrito at 0.17.
- **INE's own 2011 department layer draws Montevideo as the union of the barrios (IoU 0.9999);
  COD-AB's Montevideo meets it at IoU 0.835.** 70.2 km2 of INE's Montevideo lies in COD's
  Canelones, holding about 19,700 Kontur people (the north of Villa García and Colón), and
  18.1 km2 of COD's Montevideo is outside every barrio, holding about 12,100 (by Paso Carrasco,
  and across the Santa Lucía). Kontur cannot pick the line: on INE's, Montevideo reads 0.997
  (COD's 0.995), Canelones 0.948 (0.970) and San José 0.983 (0.932). INE tabulates its counts
  on its own line, so placement follows it.
- `uy_grid.py::montevideo_edge` cuts every hex on either line: pieces inside a barrio go to it,
  pieces outside to their COD department, the 127 pieces of COD's Montevideo outside INE's line
  to the nearest other department (Canelones or San José), each hex's people shared over its
  land pieces. Montevideo holds 1,316,475 Kontur people on INE's line against 1,294,683 on
  COD's; the edge's Canelones hexes 38,696 to 29,810, San José 0 to 6,822; 19,728 people in
  river-centred hexes, dropped before, are placed on their land; nobody is dropped.
- **Witness**: Kontur per barrio over census per barrio, normalised to Montevideo, p10 0.90,
  median 1.03, p90 1.13 (Tres Cruces lowest at 0.76, Punta Carretas highest at 1.18); Spearman
  +0.974 against a best of 1,000 shuffles at +0.393, the assertion set before the numbers were
  read. Pieces per barrio: median 14, minimum 4.
- Not checked: COD's other department lines against `ine_depto`. The before-build had
  Montevideo's dots in Paso Carrasco; whether any other department has the same problem is
  one overlay away.

### 12.6 What the barrios show (off `data/normalized/uy.csv`)

- Atheist or agnostic: 29.8% of Palermo and 29.7% of Barrio Sur, against 14.7% of Lezica and
  Melilla and 15.2% of Cerrito.
- Catholic: 63.8% of Carrasco Norte and 61.2% of Carrasco, against 31.8% of La Paloma and
  Tomkinson and 32.8% of Casabó and Pajas Blancas.
- Believer without a religion: 33.5% of La Paloma and Tomkinson, against 9.5% of Carrasco.
- Non-Catholic Christian: 14.9% of Villa García and Manga Rural, 13.8% of Manga and Toledo
  Chico, 13.6% of Casavalle, against 3.1% of Barrio Sur and 4.4% of Pocitos.
- Jewish: 8.4% of Punta Carretas and 8.0% of Pocitos; with Punta Gorda they hold 69.2% of the
  city's 10,546 (10,404 before §12.7's fix). Montevideo holds 89.8% of the country's.
- Umbanda and Afro-American: 2.7% of Las Acacias and 2.5% of Villa García, none drawn in
  Carrasco.
- Both no-religion cells together run 30.3% (Carrasco Norte) to 55.5% (Barrio Sur), inside the
  departments' own range.
- Thinnest barrio sample: La Blanqueada, 358 respondents, about 5 points either way on a share
  near 46% before the design effect. A gap of several points between two barrios can be noise.

### 12.7 The 7+ cut took out the wrong fifths: FIXED 2026-09-15

`uy_geo.py`'s `BAND_5_9_OVER_6 = 0.6` is subtracted from the 5-9 band, but three fifths of that
band is ages 7 to 9, **inside** the universe. The cut wants 0.4 (ages 5 and 6). Nationally the
5-9 band is 230,771 people (6.60%, not §5's 6.3%), so **the drawn 7+ population is 46,154 people
short, 1.32% of Uruguay, in every department in proportion**, and `gap_share` is 0.0899 where
0.0767 is right. §5's "0.05% of Uruguay" is also wrong arithmetic. The census is the witness:
Montevideo's under-7 share is 6.95% in the 2023 persons file, 8.39% as built and 7.16% with 0.4.

Not fixed by the barrio session, which was scoped to Montevideo with the rest of Uruguay as drawn.

**Fixed the same day, session `cb8b206e-fixes2`, on the supervisor's call.** `BAND_5_9_OVER_6`
is 0.4; `uy_geo.py` and `uy.py` rerun, both editions rescattered (3,225 and 321 dots).

- Drawn **3,228,150** aged 7+, 46,154 more; Montevideo 1,207,523, over the same 62 barrio
  shares, which `uy.py` recomputed unchanged. `gap_share` **0.0767** (268,250 under 7), written
  by hand: `tools/gap_share.py uy` refuses an age cut ("the mapping excludes nothing").
- Every department's shares are the survey's and did not move. National shares moved by at
  most 0.007 points (atheist 15.535% to 15.528%, non-Catholic Christian 10.066% to 10.070%),
  because the 5-9 band's weight differs a little between departments. Every figure in
  `note_public` holds at the precision it prints, and the note was not edited.
- Changed text: `gap` 9.0% to 7.7%; `grain` 111,000 to 112,000 per department (the barrio
  average stays 19,000); the city's Jewish residents 10,404 to 10,546 (§12.6); §5's arithmetic.
- Not taken: reading each department's 7+ from the census persons file (`DEPARTAMENTO`,
  `PERNA01`). 0.4 leaves Montevideo 0.21 points above the census's under-7 share, so that
  would move each department by about 0.2%.

### 12.8 Not done

- Whether any ECH after 2006 repeated `e29` (§10, still open).
- Secciones censales as a grain.
- The barrio figures are 2006's; the note says the level is 2006 and that holds here too.
