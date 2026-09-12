# Lebanon — CLOSED on the Arab Barometer, 2026-09-09. The sect column exists and the geography under it is a fieldwork quota

**Lebanon is not drawn and there is no `sources/lb.py`.** `queue.md` called it *"the one to
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
