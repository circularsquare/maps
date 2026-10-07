# Lebanon — DRAWN 2026-10-03 from the 2022 electoral register by caza of family registration (§10). Sections 1-9 are the survey closures that came first

**§10 is the build** (session `fafd1067-lb`, on Anita's ruling in ask 049). Sections 1-9 below
record why no survey could place Lebanon's sects; they still hold, and they are why the register,
with its known flaw, is what is drawn.

*What follows, through §9, was written while Lebanon was closed.*

**Lebanon is not drawn and there is no `sources/lb.py`.** The World Values Survey's wave 7 was
opened on 2026-09-15 and closes the same way, by its own sample design, from the same fieldwork firm
(§9, `sources/lb_wvs.py`). `queue.md` called it *"the one to
want"* and it was right about the prize: the Arab Barometer carries a **sect** item for Lebanon
and nowhere else in this file, with Maronite, Orthodox, Catholic, Armenian, Sunni, Shia and
Druze as separate answers. What closes the country is not the answer column. It is that the
survey firm decides, before fieldwork, how many of each of those it will interview in each
governorate — so the per-governorate composition is the contractor's own assumption about where
Lebanon's sects live, and drawing it would put that assumption on the map with nothing in the
pipeline able to disagree.

`sources.md` §11al is the short version, `sources/arabbarometer.py`'s `quota_agreement` is the
check that now runs on every Arab Barometer country, and spec §12 carries the general lesson.

---

## 1. What the file actually holds for Lebanon

Ten waves are on disk. Lebanon appears in all ten. Two religion columns:

| wave | n | `Q1012` religion | `Q1012A` sect | `Q1` governorates |
|---|---:|---|---|---|
| I | 1,195 | absent (`q711` instead) | — | **none — no subnational variable at all** |
| II | 1,387 | 1,387 | — | 6 |
| III | 1,200 | 1,200 | **1,200** | 6 |
| IV | 1,500 | 1,500 | 615, **Muslims only** | 6 |
| V | 2,400 | 2,400 | **2,400** | 8 |
| VI-1 | 1,000 | **empty** | **1,000** | 9 |
| VI-2 | 1,000 | 1,000 | **1,000** | 9 |
| VI-3 | 1,000 | 1,000 | **1,000** | 9 |
| VII | 2,399 | 2,399 | — | 8 |
| VIII | 2,403 | 2,403 | — | 8 |

`Q1012` is answered by **13,289** Lebanese over eight waves, which is what `queue.md` priced.
`Q1012A` is answered by **6,600** over five, and it is a *complete* religion answer rather than
a follow-up: it partitions every respondent, and it nests exactly inside `Q1012` — Maronite,
Orthodox, Catholic, Armenian and the tiny Chaldean/Anglican/Latin/Assyrian/Syriac cells all sit
under `Christian`; Sunni, Shia and `Just a Muslim` under `Muslim`; **Druze sits under `other`,
`Something else` or `No religion` depending on which wave's card was used**, which is the answer
to how 190 Druze respondents in wave V come back as 190 `other`.

Wave IV cannot enter a sect pool: its `Q1012A` was asked of Muslims only (615 of 1,500,
Sunni/Shia), so pooling it would drop every Christian in the wave.

Three things about the geography, none of which is the reason the country is closed:

* **wave I has no subnational variable of any kind** (181 columns, `country` is the finest),
  which is §9cq's finding about Jordan and holds here. Its `q711` is the one place in the whole
  file that offers `sunni muslim (lebanon & bahrain)`, `shiite muslim (lebanon & bahrain)` and
  `druze (lebanon)` as top-level codes; unweighted it reads 50.1% Christian, 21.9% Sunni, 21.3%
  Shia, 6.7% Druze, and it can never be placed;
* **the unit set changes three times.** Waves II-IV use the six pre-2003 mohafazat; waves V, VII
  and VIII use the eight current ones, Akkar split from North and Baalbek-Hermel from Bekaa;
  waves VI-1 to VI-3 use nine, splitting Kesrwan-Jbeil out of Mount Lebanon on the 2017 decree.
  Harmonising to eight is the obvious choice and costs waves II, III and IV;
* **the exact bar that would have applied.** `spearman_null.critical_rho` gives **+0.8286 at six
  units**, +0.6429 at eight and +0.6000 at nine. Six units is not a country you can draw from a
  split-half; that alone rules out keeping wave III for the sake of its sect column.

Wave II spells Nabatieh **`4506. Nabataean`**, which is not a place; it is the survey's own
rendering and is transcribed rather than repaired.

---

## 2. THE FINDING: the per-governorate composition is set before fieldwork

Christian interviews over total interviews, per governorate, per wave, on `Q1012`:

```
                     II      III       IV        V     VI-2     VI-3      VII     VIII
AKKAR                 -        -        -   30/160   10/ 70   10/ 70   30/160   32/170
BAALBEK-HERMEL        -        -        -    0/150    0/ 60    0/ 60    0/150   10/180
BEIRUT           51/159   50/130   50/130   90/250   40/100   40/100   90/250   74/251
BEKAA            45/201   20/150   20/250   60/150   20/ 60   20/ 60   60/149   48/150
MOUNT LEBANON   355/451  300/480  314/555  650/960  270/400  270/404  650/960  568/851
NABATIYEH         7/107    0/ 70    0/100    0/140    0/ 60    0/ 60    0/140    0/110
NORTH           103/290   80/240   94/315  100/330   40/140   40/140  100/330   75/341
SOUTH            15/179   20/130   20/150   10/260   10/110   10/106   10/260    0/350
```

**Wave V and wave VII are the same numbers in all eight governorates.** They were fielded three
years apart, in 2018-19 and 2021-22, by separate rounds of fieldwork, and they return 30
Christians in 160 Akkar interviews, 90 in 250 Beirut interviews, 650 in 960 Mount Lebanon
interviews and 10 in 260 South interviews, twice. Wave VII's denominator differs from wave V's
by one respondent in one governorate. **Waves VI-2 and VI-3 are a second such pair** on a
half-size grid, and VI-1 shares that grid too.

The sect column says the same thing more sharply, because a quota shows through best where the
true share is near 0 or 1:

* **Kesrwan-Jbeil comes back 100% Christian in all three parts of wave VI** — 60, 60 and 40
  interviews, not one Muslim among them. Kesrwan-Jbeil is a heavily Christian district and a
  free sample of 160 people there would still find some;
* **Akkar is exactly 10 Maronite and 60 Sunni in each part**, with zero Orthodox, zero Shia and
  zero Druze all three times, in a governorate that has all three;
* **Baalbek-Hermel is exactly 60 Shia and nothing else, three times**, and Nabatieh is 0
  Christians in every wave from III on.

The numbers are round — 10, 20, 40, 50, 60, 70, 100, 140, 160 — which is what a filled quota
looks like and is not what a sample of a population looks like.

### The statistic, and what it says

`ab.quota_agreement` compares every pair of waves cell by cell. For each unit both waves
sampled, over the answers both cards offered, dropping the largest answer because the shares
sum to one, it computes the exact probability that two independent samples of those sizes would
land on the same rational share, and reads the number of exact agreements against the
Poisson-binomial tail of those probabilities.

    Q1012,  eight waves, 13,289 respondents:  V vs VII,      4/4  cells identical, p = 6.6e-06
    Q1012A, five waves,   6,600 respondents:  VI-2 vs VI-3,  8/26 cells identical, p = 1.1e-04

Against **Jordan's worst pair at p = 0.888, Egypt's at 0.925 and Iraq's at 1.0** — none of those
three has a single exact agreement outside the degenerate cells, and their Bonferroni products
are above 1. The bar shipped in `sources/arabbarometer.py` is 1e-3 on the Bonferroni-adjusted
minimum, and Lebanon's religion pool comes in at 1.4e-4.

**One nuance worth carrying, because it will catch somebody.** The sect pool on its own gives a
Bonferroni-adjusted 1.1e-3 and would have squeaked *past* the bar — not because the sect column
is cleaner but because the pool that carries it has no wave VII in it, and wave VII is where the
strongest evidence lives. The quota is a property of the **fieldwork**, not of a column, so a
country that fails this on any of its religion columns has failed it. Run the check on every
column the file offers before believing a pass.

### What was ruled out before this was believed

* **Not a panel.** Wave VI-1 and VI-2 share 877 `ID` values, which looks like a re-interview
  until you check it: of those 877, only **165 agree on governorate and 160 on sect**, which is
  chance. `ID` is a within-file serial and the parts are independent samples. Waves V and VII
  are three years apart with 2,400 and 2,399 respondents, which no panel survives.
* **Not the weights.** Every table above is unweighted counts. The weights (0.38 to 3.95 in wave
  V, 511 distinct values) move the per-governorate Christian share by one to three points and
  cannot undo a quota; they re-level it towards whatever frame Arab Barometer used, which is
  itself unpublished.
* **Not an artefact of the name harmonisation.** The agreement is between raw `Q1` labels that
  are character-identical within each wave pair; folding Kesrwan-Jbeil back into Mount Lebanon
  affects only the wave VI rows and the pattern is there without it.

---

## 3. WHY THIS IS FATAL AND THE SPLIT-HALF CANNOT SEE IT

§14.16's stability test asks whether a category's ranking across units replicates between the
early and the late waves. **A quota replicates by construction.** Run on Lebanon's eight
governorates it would return something close to +1.0 for every answer, clear the +0.6429 bar
with room to spare, and license drawing Maronites, Sunnis, Shia and Druze on their own
governorate shares — shares that are the fieldwork specification. Every other guard in the
pipeline agrees with it: `held_out` passes trivially (the governorate allocation is a quota
too, so the survey's unit shares match the population's by design), the totals reconcile, and
`ab.build` produces a closed partition. Nothing anywhere would print a warning.

That is a worse failure than a country that fails a check, and it is the reason
`assert_not_quota` now runs at the top of `ab.stability` rather than being a note in this file.

It is also §14.4 rule 1 in a form the rule did not anticipate. The rule says never estimate a
magnitude a source does not publish. Here the magnitude would be an estimate — the survey
firm's — that the source does not publish *as an estimate*, but ships inside a respondent file
that looks like a measurement.

---

## 4. The state route, and what Lebanon does publish

**No census since 1932**, which is the fact everyone knows, and the reason is the one everyone
gives: the confessional balance determines the distribution of political office under the 1943
National Pact and the 1989 Taif Agreement, and counting would settle it.

* **UNSD Demographic Yearbook table 28: Lebanon is ABSENT** (`tools/oracle.py Lebanon`). That
  is a floor and not a fact, per §11r, but there is nothing behind it here.
* **The Central Administration of Statistics** (`cas.gov.lb`) publishes the 2018-19 Labour Force
  and Household Living Conditions Survey and the 2004 and 2007 household surveys. §6 below is
  §9cd's route run on it in full, and it comes back empty.
* **The Directorate General of Personal Status** (`dgcs.gov.lb`) is the office that holds the
  civil register, in which every Lebanese person's sect is recorded, and it publishes a
  **statistics map at caza level — 26 cazas, years 2009 to 2026** — carrying registered voters
  by sex, plus births, deaths, marriages and divorces. **It does not carry sect.** The page was
  downloaded and searched: 412 KB of HTML, no `api`, no map-server URL, and the sixteen hits on
  `Sect` are all the word `Section` in CSS class names. Its `/arabic/statistics-map/details?q=`
  pages are per-caza and worth one more look by anyone who reopens this.
* **The electoral register is the only sect-by-place data Lebanon has**, and it is real: the
  Ministry of Interior and Municipalities issues registered-voter counts by sect and by caza
  before each election, 3,967,507 voters in 2022 against 3,746,483 in 2018, and Information
  International and L'Orient-Le Jour both publish tabulations built from it. **No open
  machine-readable national file was found.** `github.com/omarabboud/lebanon_elections` has a
  `sect` column but only for Beirut II's polling stations in 2022.

### AND THE REGISTER IS PROBABLY THE WRONG GEOGRAPHY ANYWAY, WHICH IS THE THING TO READ BEFORE CHASING IT

**A Lebanese voter is registered where their family's civil record sits, not where they live.**
That is why the register is organised by *qada al-qayd*, the district of registration, and why
the recurring proposal to let people vote where they live needs its own name (the *megacentres*)
and has never been implemented. So a dot map built on the register would draw Beirut's and
Mount Lebanon's Shia population in Nabatieh and Baalbek-Hermel, and its Sunni population in
Akkar and Tripoli — the villages their grandparents left. Roughly half the country lives in
Greater Beirut. Nothing published converts the register from place of origin to place of
residence, and inventing that conversion is §14.4 rule 1 again.

**This is not a reason to stop looking**, and §12's standing instruction says a negative is a
record of what was tried. It is a reason not to reach for the register as though it were a
census with a different cover.

---

## 5. What would actually draw Lebanon

Ranked, so the next session does not re-derive it:

1. ~~**A CAS release nobody here has opened.**~~ **RUN, 2026-09-09, AND IT IS EMPTY — see §6.**
   The office's whole archived tree was enumerated and carries no religion table of any kind.
2. **A published tabulation of the electoral register by sect and caza, with a residence
   correction that somebody else published.** Both halves are needed. Without the second half
   the map is a map of where families are registered, and it should say so in `grain` if it is
   ever drawn that way, which is a decision for Anita and not for an agent.
3. **The 1932 census**, which is published by mohafaza and is a real count. Ninety-four years
   old, before the Palestinian and Syrian displacements and before the emigration that changed
   the balance. It would be the oldest vintage on this map by half a century.
4. **The Arab Barometer, if a future wave drops the quota.** Wave VIII is the one wave whose
   Lebanese numbers move freely — Beirut 29.5% Christian against 36.0% in V and VII, North
   22.0% against 30.3%, South 0.0%, Baalbek-Hermel 5.6% — so the design may already have
   changed. **One wave cannot carry a split-half**, so this needs wave IX to exist and to be
   free too. Re-run `ab.quota_agreement` on the pool before believing it.

---

## 6. §9cd's route run on CAS, and the denominator a future build would use

§9cd's lesson is that four correct checks on an office's *documents* closed Kazakhstan and
opening its dashboard reopened it. So the route was run here rather than left as a suggestion,
and this is what came back.

**`cas.gov.lb` is behind a Cloudflare interstitial and returns 403 to every client**, curl and
WebFetch alike, on the root and on every sub-path — `Just a moment...`, the challenge page, not
a dead host. Per Anita's standing rule that is a browser job and was not fought.

So the office was enumerated from the **Wayback CDX API** instead
(`[[reference_dead_stats_office]]`), two sweeps, about **6,000 distinct archived URLs with a
200**. **Not one of them mentions religion, sect, confession, denomination, طائفة or مذهب.** The
only hits on the string `sect` are `Financial sector`. The tree is CPI monthly files, the
Statistical Yearbook and Monthly Bulletin chapters (`Excel/SYB/`, `Excel/SMB/`), press releases
and the survey report PDFs.

**CAS ran a DevInfo 7 instance** — `/di7web/`, `/di7uilibservices/diuilib/1.1/` and `1.8/` — which
is exactly the BI engine §9cd says to look for. It is not reachable now, and the archive shows
why nobody should expect it back: the instance was **compromised at some point**, with
`Janissaries.Shell.php`, `modue.php`, `jun.asp;.txt` and SEO-spam pages under
`di7web/stock/users/` all captured with a 200. That is very likely what the Cloudflare wall in
front of the whole host is for.

### The denominator, which exists and is good

**OCHA Lebanon's `lebanon-population-estimates-and-displacement-figures` on HDX**, resource
`05.-2026-lrp-population-package.xlsx`, updated **2026-03-17**. Eight governorates, and it
separates the four populations that a Lebanese map has to keep apart:

    Lebanese     3,864,296       Palestinians   224,791
    Syrians      1,120,000       Migrants       164,097      TOTAL  5,373,184

with sex and five-year age bands per governorate on separate `LEBANESE`, `SYRIAN`, `PALESTINIAN`
and `Migrants` sheets. Its own methodology sheet names **CAS's Labour Force and Household Living
Conditions Survey** as the source for the Lebanese figure, IOM/DTM round 87 for displacement and
UNHCR registration for the Syrians. Its `Within 332` sheet is a facilities count and not a finer
population tier; Lebanon has 1,132 cadastres and this package does not reach them.

So the universe question below has a published answer at governorate: **a Lebanese-only build
would draw 3,864,296 of 5,373,184 people and its `gap_share` is 0.281**, with the gap being
Syrians, Palestinians and migrants, each of which that file counts. Nothing here supplies the
religion column, which is the part Lebanon does not have.

**And `cod-ps-lbn` does not exist** (HDX 404) while **`cod-ab-lbn` does** (200) — the same shape
as Jordan (§9cq): boundaries yes, population COD no. The OCHA package is the better denominator
anyway, for the same reason DOS's own estimate was for Jordan.

## 7. What this cost, and the universe question that was never reached

About half a session. The sect column was found in twenty minutes and the quota in another
forty; everything after that was measurement and writing. **Nothing was downloaded** — all ten
waves were already on disk from §9bz.

The universe question §9cq asks of Lebanon was never reached and is still open, so it is
recorded here rather than lost. Lebanon's resident population includes something on the order of
1.5 million Syrians and several hundred thousand Palestinians, neither of which the Arab
Barometer's Lebanese sample includes, and both of which are much larger relative to the country
than Jordan's non-citizen quarter. Any future Lebanese build has to decide whether it draws
residents or citizens **before** it picks a denominator, and there is no COD-PS figure that
answers it for you.

## 8. Re-checked 2026-09-15: the World Values Survey was never opened

Scout `cb8b206e-scout1`, sources.md §scout-2026-09-15-negatives. §5's list missed a second survey.
**WVS wave 7, Lebanon 2018** (IHSN catalog 12281, data dictionary) has 1,200 respondents with
`Q289CS9` coded Sunni 320, Shia 310, Maronite 297, Druze 100, Orthodox 82, Roman Catholic 64,
Armenian Apostolic 23, other 4, and `N_REGION_ISO` on all eight governorates (Mount Lebanon 480,
North 190, Beirut 130, South 130, Bekaa 100, Nabatieh 70, Akkar 50, Baalbek-Hermel 50). It is
untested for the quota §2 found. The first step is the WVS online tool's crosstab of `Q289CS9` by
`N_REGION_WVS`, set beside §2's grid, before anyone asks Anita for the download; whether the sample
is citizens or residents is also unchecked. Arab Barometer wave IX fielded Lebanon on 3-25 November
2025 and its data is promised for autumn 2026, which is when §5 item 4 can be run.

## 9. WVS wave 7 opened 2026-09-15: the sample design gives every cluster a sect, so it closes too

Session `cb8b206e-lb`, on ask 035's ruling ("get it in and then see", with the rider that a quota
sample does not stand on it). Anita downloaded `F00013081-WVS_Wave_7_Lebanon_Stata_v5.1.zip`
(Stata, 1,200 rows, 451 columns). It is in `data/raw/lb/` with the .dta unzipped beside it, and the
three IHSN related materials for catalogue 12281 are saved there as PDFs: the methodology report
(104800), the team sheet (104801) and the sample design (104802). The zip carries no terms file; the
WVS download licence (`worldvaluessurvey.org/AJDownloadLicense.jsp`, read 2026-09-15) allows
non-profit use, requires a citation in each publication and sending that citation to the WVSA, and
forbids redistributing the files. Wave 6 was not downloaded. `python sources/lb_wvs.py` reruns
everything below and exits 0 while it holds.

**The survey says so itself.** The Sample Design's pp.3-4 are a table, rendered as an image, of
Mohafaza, Kadaa, Sample, Number of PSUs and **Sect** (Christian, Sunni, Shiaa, Druze): 120 PSUs of
10 interviews, each row one sect. The methodology report ticks Yes on quota controls (Q15), gives
the stratification factors as "by governorates / districts / religions" (Q20), and says "the daily
work sheet mentioned also the profile required of the interviewee" (Q14). The fieldwork firm is
Statistics Lebanon Ltd. (co-PI Rabih Haber), which is also the Arab Barometer's Lebanese partner;
Arab Barometer wave V's technical report (`ABV_Methods_Report-1.pdf` p.7) gives Lebanon's strata as
"Governorates and sect", 23 strata, 240 PSUs of 10.

**The file reproduces the design exactly.** `I_PSU` holds 120 clusters of exactly 10, and all 120
hold a single community; the four `Other; nfd` answers sit one each in four El Meten Christian
clusters. Counted by kadaa (`N_TOWN`, where the file's "Saida Villages" is the design's three Shia
PSUs in Saida) and sect, the clusters equal the design table in all 23 kadaa. Inside Christian
clusters the denominations mix in 39 of 47 (Maronite, Orthodox, Catholic, Armenian), which is what
free household selection looks like; Sunni, Shia, Druze and Christian never share a cluster. §8's
marks are all present: Akkar 50 of 50 Sunni, with no Christian, Alawite, Shia or Druze respondent;
Baalbek-Hermel 50 of 50 Shia; Nabatieh 70 of 70 Shia.

Christian over total per governorate, beside §2's grid (WVS Christian includes the four `Other`):

```
                  WVS 7 (2018)      AB V and VII     AB VIII
Akkar               0/50     0%      30/160  19%      32/170  19%
Baalbek-Hermel      0/50     0%       0/150   0%      10/180   6%
Bekaa              20/100   20%      60/150  40%      48/150  32%
Beirut             50/130   38%      90/250  36%      74/251  29%
North              80/190   42%     100/330  30%      75/341  22%
South              20/130   15%      10/260   4%       0/350   0%
Mount Lebanon     300/480   62%     650/960  68%     568/851  67%
Nabatieh            0/70     0%       0/140   0%       0/110   0%
```

The WVS grid is not a copy of the Arab Barometer's, and that is not reassuring: one firm's two
2018 allocations disagree by up to 20 points (Bekaa 20% against 40%, North 42% against 30%), so
the allocation is not even stable as the firm's own estimate. Each is a frame rounded into 5 to 48
clusters per governorate.

**What it means.** The sect composition per governorate, and nationally (Sunni 320, Shia 310 and
Druze 100 are 32, 31 and 10 clusters), is the design's allocation, and nothing in the file measured
it. One wave cannot carry a split-half, and a split-half would have passed a quota anyway (§3); the
check that stands in for it is the one above, which needs only the cluster id and the design
document. Lebanon does not stand on ruling 035 and is not drawn.

**What the file does measure**, in case another source ever supplies the community shares: the
denomination split inside Christian clusters (Maronite 297, Orthodox 82, Roman Catholic 64,
Armenian 23, other 4, of 470). The sample is citizens only: `Q269` is Yes for all 1,200, though the
design's text says "residents" in one sentence and "Lebanese citizens" a few lines later.

**REOPEN** only on a source whose design does not set sect per cluster. Arab Barometer wave IX
(Lebanon fielded November 2025, data promised autumn 2026) is very likely the same firm with the
same strata: read its technical report's strata line before loading the file. WVS wave 6 Lebanon
(2013) was not downloaded and its design was not read; read its IHSN sample design before asking
Anita for it.

## 10. DRAWN 2026-10-03 from the 2022 electoral register, by caza of family registration

Session `fafd1067-lb`, on ask 049 (Anita: draw it from the register by caza, labelled as where
families are registered and not where they live, "its replicating what people usually use. and
would be a big improvement over no granularity"). `python sources/lb.py`, then `lb_geo.py`, then
`lb_grid.py` rebuild everything below and stop on any check that moves.

### 10.1 What was found, and what the 2022 register is published as

* **The Monthly no. 187 (April-May 2022), Information International**, pp. 6-22, "Lebanon's 2022
  voters by sect and district": Table 1 the nation (3,967,507, 17 rows), Tables 2-16 the fifteen
  electoral districts of the 2017 law, each "based on the figures issued by the Ministry of
  Interior and Municipalities", with the 2018 column and the difference beside 2022. A born-digital
  PDF (4,609,114 bytes, 52 pages), `monthlymagazine.com/cms/upload/magazine/630f55c7ec72b382_file.pdf`,
  read off its text layer. **This is the most complete citable 2022 tabulation found, and it is by
  electoral district, not caza.** Nine of the fifteen districts hold two to four cazas.
* **No caza-by-sect table for 2022 was found in a publishable form.** Searched 2026-10-03: The
  Monthly's 2022 issue (districts only); the Wikipedia articles for the 2022 district elections
  (totals only) and the caza articles (2022 percentages credited to L'Orient Today, mixed
  groupings, Koura's Greek Catholics jump from 1.18% to 2.80% between 2018 and 2022, so not
  used); `elections.gov.lb` (a single-page app; its Statistics chunk shows turnout by major
  district and nothing by sect); the DGPS statistics map (§4: no sect).
* **lub-anan.com**, "electoral facts about Lebanon, per the official voter lists issued by the
  Ministry of Interior for 2014" (its disclaimer: the lists the Ministry issued on discs). One
  page per caza with every sect's count by sex: 29 pages (Beirut as the 2008 law's three
  districts, Saida as city and villages), 3,514,588 voters, each page closing on its own block
  subtotals and grand total. Only these aggregate pages were read; the site also has name and
  family pages, which were not.
* **L'Orient Today's Tableau Public workbook "Registered Voters by District"** (Richard Salame
  and Iva Kovic, 2022, author profile `richard.salame`), embedded in their "Mapping Lebanon: Data
  and statistics" page. Its view's own CSV export
  (`public.tableau.com/views/RegisteredVotersbyDistrict/RegisteredVotersbyDistrict.csv?:showVizHome=no`)
  gives registered voters per minor district: 25 rows, three of them caza pairs (West
  Bekaa-Rashaya, Marjayoun-Hasbaya, Baalbek-Hermel), summing to 3,967,507 exactly.

### 10.2 THE PACKAGED WORKBOOK IS THE VOTER ROLL. DO NOT DOWNLOAD IT

`public.tableau.com/workbooks/RegisteredVotersbyDistrict.twb` answers with a 58 MB packaged
workbook whose `.hyper` extract is built from "20220428 district populations.csv", with the
columns `firstname`, `lastname`, `fathersname`, `mothersname`, `dob`, `sex`, `personalsect`,
`regsect`, `regnumber`, `town_neighb`, `cadaa`, `voting_country`: the Ministry's roll, one row per
voter, 3.97 million named people with their sect. It was fetched once on 2026-10-03 while probing
for the view's data, its `.twb` column list was read, and both it and the towns workbook's copy
were deleted without the extract being opened. Nothing in this build comes from it. It would
answer every question here (caza by sect, personal against register sect, town by sect), and it
is still not to be used: it is personal data on a scale this project should not hold, whatever
its availability. If a caza-by-sect table is wanted from it, it is L'Orient's to publish.

### 10.3 Personal sect and register sect are different counts

The roll carries two sects per voter: the personal sect and the sect of the family register
(`مذهب السجل`). lub-anan's 2014 pages count the **personal** sect: its minority sects are mostly
women (El Koura's Greek Catholics 593 women, 98 men; nationally 12,126 of the 13,857 `not stated`
are women), which is what a woman's own sect looks like after her record moves to her husband's
family. The Monthly's 2022 tables are the **register** sect: in Maronite Kesrouan-Jbeil they put
3,423 Greek Orthodox and 2,499 Greek Catholics where lub-anan's 2014 seed has 7,001 and 6,102,
and their 2018 column (3,361, 2,328) agrees with 2022. The map draws the register sect, because
that is what is published for 2022 and what the ruling names; `note_public` says a married woman
is usually counted under her husband's family's sect. The 2014 seed is used only for how a
district's sects split between its cazas, so the two kinds mix only there.

### 10.4 The construction and its checks

1. **2022 by district** (`read_monthly`). Every table's Total row equals the district total printed
   above it and L'Orient's minor districts summed, except Table 9, whose Total (153,974) is a
   misprint for the 153,975 its rows, its header and L'Orient give (`PINNED`). **Nine tables' rows
   miss their total by 1 to 44 voters** (`ROW_SUM_OFF`: North III +44, North II +10); the
   difference column finds three slips (Table 4 Others +99 for +9, Table 12 Others +13 for +15,
   Table 13 Alawite +57 for +58) that do not close their tables, so rows are kept as printed and
   scaled to the total (at most 0.017%). Against Table 1, **ten sects agree to the voter** (Sunni,
   Shia, Maronite, Druze, Armenian Orthodox and Catholic, Alawite, Evangelical, Latin, Assyrian);
   Greek Orthodox is 2,554 higher in the districts, and Table 1 groups the Syriac, Chaldean and
   Others rows differently (its Syriac Orthodox 15,672 against the districts' 6,530 plus 13,047
   printed as one `Syriac` in four districts). Pinned as `NATIONAL_OFF`; the districts are drawn.
2. **Districts to cazas** (`build`). In each multi-caza district, an IPF of the 2014 caza-by-sect
   seed onto the district's 2022 sect rows and L'Orient's 2022 minor-district totals; `Syriac` and
   `Others` rows are expanded on the seed. The 2014 seed against each district's 2022 rows is held
   to a growth band (0.70-1.45 for sects over 2,000), which is what would catch a sect on the wrong
   row; three cells are outside it and pinned with their reasons (`GROWTH_PINNED`: the two in
   §10.3, and North II's Alawites at 1.47, already 1.35 by the 2018 column). The rake moved between
   1,000 and 6,100 voters between minor districts per district. The three caza pairs are split per
   sect on 2014.
3. **Five cazas are whole districts** (Beirut = Beirut I + II, El Meten, Baabda, Zahle, Akkar) and
   are `measured`: 1,305,483 of the 3,864,296 Lebanese dots. The rest are `derived`, with
   `roll = NOWHERE` (rollup.py's same-unit rule: the sect was counted for the district).
4. **Scale.** Registered voters (21 and over, emigrants included) times 3,864,296 / 3,967,507 =
   0.97399, OCHA's resident Lebanese (2026 LRP package, from CAS's 2018-19 LFHLCS). One national
   factor: each caza is drawn with as many people as are REGISTERED there. The alternative,
   OCHA's resident Lebanese per caza at each caza's registered mix, was rejected: it would move
   the national mix (the South and the Bekaa, registered far above their residents, are mostly
   Shia, so national Shia would fall and Mount Lebanon's sects rise) while still colouring the
   southern suburbs with Baabda's register.

The 2022 register as drawn: Sunni 29.51%, Shia 29.32%, Maronite 19.31%, Greek Orthodox 6.65%,
Druze 5.59%, Greek Catholic 4.31%, Armenian Orthodox 2.12%, Alawite 0.97%, Armenian Catholic
0.50%, Evangelical 0.44%, Syriac Orthodox 0.41%, Syriac Catholic 0.32%, Latin 0.28%, Jewish
0.11%, Chaldean 0.06%, Assyrian 0.04%, Others 0.05%.

### 10.5 What registration does to the map, measured

Register share over OCHA's resident-Lebanese share, per caza (`sources/lb.py` prints it):
Marjaayoun 4.87, El Hermel 2.84, Bent Jbeil 2.71, Bcharre 2.39, Jezzine 2.24, **Beirut 2.08**, ...
Akkar 1.02, Chouf 0.99, ... Aley 0.61, **El Meten 0.45, Baabda 0.41, Kesrwane 0.41**. Baabda, El
Meten and Kesrwane hold 26.8% of resident Lebanese and 11.4% of the register. Baabda's register is
35.6% Maronite, 25.9% Shia and 17.6% Druze; Beirut's is 48.9% Sunni and 16.0% Shia. OCHA's
resident split of the south reflects the 2023-24 war's displacement (Marjaayoun's 0.66% of
residents), so the extreme southern ratios are partly that.

**Kontur only places inside a caza and is not calibrated.** It agrees with OCHA's residents
nationally (0.997) and ranks the cazas like OCHA (Spearman +0.847 against a shuffled 99th
percentile of +0.448, the asserted witness), but six cazas fall outside a 0.5-2.0 band: Kontur
puts 1,200,117 people in Baabda against OCHA's 568,296 and 112,226 in Akkar against 484,765 (its
footprint covers Akkar, 912 populated hexes). A caza total Kontur gets wrong moves nobody between
cazas here, so the band is printed, not asserted. Cap blocks (`kontur_cap.csv`): Beirut with the
Dahiyeh (58 hexes, 1.6 million people) and Tripoli's core (5 hexes, 167,487 of the caza's
229,436) real; three hexes east of Hazmieh at the cap among 2,600-9,900/km2 capped. COD-AB's
Hasbaya loses 17.9 km2 to Natural Earth's Israeli-held Shebaa Farms; 6,276 Kontur people in hexes
across the Syrian and Israeli borders are dropped, 64,888 on the coast snapped.

### 10.6 The non-Lebanese layer

OCHA's 2026 package gives every population by caza (the brief expected governorate): Syrians
1,120,000 (UNHCR registration, adjusted for 2025 returns and arrivals), Palestinians 224,791
(UNRWA, refugees from Lebanon and from Syria), migrants 164,097 (IOM's Migrant Presence
Monitoring, quoted by OCHA; no IOM data was downloaded, per the 2026-09-16 ruling). Syrians take
Pew 2020's Syria row (94.2% Muslim, 3.8% Christian) with Muslims on bare `islam` and Christians on
bare `christianity`, as Syria itself is drawn: nothing measures the sects or churches of Syrians
in Lebanon, and the 2025 arrivals after the coastal killings were reported as largely Alawite,
which a Sunni default would get wrong. Palestinians take Pew's Palestinian-territories row,
Muslims on `islam.sunni` (origin_religion's default). Both `modelled`. Migrants are the `gap`
(3.05% of OCHA's 5,373,184): no nationality by caza.

### 10.7 Nodes

Added `christianity.catholic.eastern.maronite` and `.melkite` (depth 4) and `other.lb`. Greek
Orthodox on `christianity.orthodox.canonical.antiochian` (origin_religion's LB and SY rows already
use it); Armenian Orthodox on `christianity.oriental.armenian` (not `.cilicia`: the register does
not name the catholicosate, ask 004's rule); Armenian Catholic, Syriac Catholic and Chaldean on
the Eastern Catholic parent; Evangelical on `christianity.protestant`; Assyrian on
`christianity.churchofeast`; the register's `Israeli` on `judaism`; Alawites on bare `islam`
(origin_religion.py's reasoning for Syria's). Cyprus's `Maronite church` still sits on the parent;
moving it would change a drawn country and was left.

### 10.8 Calls someone might reverse

* Dots per caza follow registration, scaled by one national factor (§10.4 step 4).
* The 2014 personal-sect register splits each district's 2022 register-sect counts between cazas.
* Lebanon's Jews (4,309 registered, almost all Beirut II) are drawn where registered.
* Alawites on bare `islam` rather than a new node.
* Syrians' Muslims unsplit; Palestinians' Sunni.

### 10.9 What would improve it

* A caza-by-sect table for 2022 that someone publishes as a table (L'Orient's dashboard tooltips
  show the top sects per minor district; Information International may print cazas in another
  issue; neither was found).
* The next parliamentary election's register: the same route, one vintage newer. Whether the
  May 2026 vote was held was not checked (a search on 2026-10-03 returned a Wikipedia page for a
  2028 election, unopened); the Directorate of Personal Status lists 4,093,662 voters for 2025,
  totals only.
* Personal sect by caza for 2022, which would be closer to what people call themselves.

## 11. Top text before the 75-word cut, 2026-10-03 (`fafd1067-top75`)

Anita asked for the text at the top of the phone screen to come down to about 75 words
(queue.md, "Cut the country text to about 75 words"). These are the four header fields as
they stood before the cut, verbatim, so nothing they said is lost. The cut versions are in
`countries/lb.py`; `note_public` was not changed.

- `how`: electoral register, 2022, every voter's sect of record, by electoral district
- `fill`: from the 2014 register by caza, inside each 2022 electoral district
- `grain`: cazas of family registration, not of residence; 200,000 people on average
- `gap`: 164,097 migrant workers, 3.1% of the people living in Lebanon, whose nationalities and religions are not known by caza
