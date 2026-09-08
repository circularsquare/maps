# Russia — Sreda «Арена» 2012

Drawn 2026-09-05. `sources/ru.py` rebuilds `data/normalized/ru.csv` from `data/raw/ru/`.
`data/` is gitignored, so this file is the record.

| | |
|---|---|
| source | Sreda, *Атлас религий и национальностей* («Арена»), published August 2012 |
| instrument | MegaFOM omnibus, fieldwork 29 May – 30 June 2012 |
| sample | **56,900 respondents**, 79 of the then 83 federal subjects, ~720 per subject |
| geography | federal subject (ADM1). 79 units, 142,590,384 people |
| categories | **18** — 17 religious positions + "difficult to answer" |
| basis | `self_id`, survey |
| tier | **`modelled` on every row** (spec §7) |
| licence | not stated on the site. Sreda published the atlas for reuse and it is cited everywhere; **terms need reading before anything ships** (sources.md §6) |

## 1. Why this source and not a census

**Russia's census has not asked about religion since 1937**, and that census's results were
suppressed and its organisers shot. Neither 2002, nor 2010, nor the 2020/2021 census asks —
the form carries nationality and language and stops there. So there is no census route at
all, and Russia is the first country on this map whose primary source is a survey. (The US
uses Pew, but on top of ASARB's county-level enumeration; here there is nothing underneath.)

Arena is the only large-sample, all-region religion survey Russia has ever had, and **no
successor exists**. Sreda stopped operating and nothing has replaced it.

## 2. The files

| file in `data/raw/ru/` | what | size |
|---|---|---|
| `arena_statistic_en.xls` | the whole published cross-tabulation, English edition | 207,360 bytes |
| `wp_federal_subjects.wikitext` | 2021 census population per subject (see §3) | ~13 KB |

`https://sreda.org/maps/arena_russia_main/arena_statistic_en.xls` — direct download, no
auth, no bot protection, no session. One `Data` sheet, 138 × 89.

**The sheet is one worksheet holding several different questions and only rows 14–31 are a
partition of anybody.** Rows 33 onward are a separate multi-answer block ("I observe as
many religious precepts as possible", "I have read the gospel", "I support traditional
family foundations") where respondents could tick any number, and below that are
demographics. Summing the sheet, or reading the religion block as rows 14–40, produces
numbers that look plausible and mean nothing. `ru.py` pins the block by asserting that the
18 rows sum to 100 in **all 79** subject columns, which they do.

Columns are three levels interleaved on two header rows: a national column, eight federal
districts, and the 79 subjects. Subjects carry a value on row 11 and the aggregates do not,
which is how `ru.py` separates them — not by position, which would break if a district were
inserted.

## 3. The magnitude comes from the census, and this is the weak link

Arena publishes **shares of respondents**. Turning them into people needs a population per
subject, and spec §3.4 says which one: structure from the detailed source, totals from the
recent one. So 2012 shares × **2021 census** population. That choice is deliberate — the
Muslim republics grew and the Russian oblasts shrank across those nine years, and applying
2012 shares to 2021 populations carries the shift rather than freezing it.

**Rosstat cannot be fetched from here, and it is a TLS wall rather than a bot wall.**
`rosstat.gov.ru` presents a chain no Western trust store can verify; `curl` exits 60,
`urllib` raises `CERTIFICATE_VERIFY_FAILED`, and the WebFetch service fails with "unable to
verify the first certificate". Per §9h's test that is a **server-side** condition — every
client fails identically — rather than the interception §9h's Hungarian case turned out to
be. Russia's national CA is not in any Western root store and will not be.

Disabling verification was not taken for a whole country's magnitudes. The figures are
parsed instead from the English Wikipedia table *List of federal subjects of Russia by
population*, which carries the Rosstat 2021 census column with its citation, fetched as
**wikitext** through the MediaWiki API so the parse is deterministic and a changed table
breaks the run rather than silently relabelling it.

**Two checks make this defensible rather than merely convenient:**

- The 83 parsed subject rows sum to **144,699,673**, which is exactly the national row in
  the same table and exactly Rosstat's published 2021 total.
- Kontur's population surface, built independently from GHSL and building footprints,
  reproduces the same per-subject figures to **0.984× nationally**, with a median subject
  ratio of 1.015 and every one of the 79 inside a factor of two (`ru_geo.md` §3).

**If the primary is ever wanted**, the URLs are in the Wikipedia table's own references:
`https://rosstat.gov.ru/storage/mediabank/PrPopul2025_Site.xlsx` for the current estimate,
and the census volumes under `rosstat.gov.ru/vpn/2020`. Both need a browser — hand them to
Anita rather than turning verification off (§12).

## 4. The category list, and what makes it unusual

Arena names **positions in the first person**, not institutions, and several are defined by
what they exclude. Every other source on this map names bodies and leaves the tree to work
out what they are.

| share | answer | node |
|---|---|---|
| 41.14% | I am Orthodox, and belong to the Russian Orthodox Church | `christianity.orthodox.canonical` |
| 25.16% | I believe in God (in a higher power), but do not profess a particular religion | `unchurched` |
| 13.02% | I do not believe in God | `secular` |
| 5.46% | Difficult to answer | **excluded** |
| 4.66% | I profess Islam, but am neither Sunni nor Shia | `islam` |
| 4.06% | I profess Christianity, but not Orthodox, Catholic, nor Protestant | `christianity` |
| 1.66% | I profess Sunni Islam | `islam.sunni` |
| 1.50% | I am Orthodox, but not the ROC, and not an Old Believer | `christianity.orthodox.other` |
| 1.22% | I practice the traditional religion of my ancestors… | `indigenous.northeurasian` |
| 0.61% | Other | `other.ru` |
| 0.46% | I profess Buddhism | `buddhism` |
| 0.32% | I am Orthodox, and I am an Old Believer | `christianity.orthodox.oldbeliever` |
| 0.21% | I profess the Protestant (Lutheran, Baptist, Evangelical, Anglican) | `christianity.protestant` |
| 0.21% | I profess Shia Islam | `islam.shia` |
| 0.13% | I profess Catholicism | `christianity.catholic` |
| 0.07% | I profess Judaism | `judaism` |
| 0.06% | I follow Eastern religions and spiritual practices (Hinduism, Krishnaism, other) | `other.ru` |
| 0.06% | I profess Pentecostalism | `christianity.pentecostal` |

Reasoning for each call is in `taxonomy/ru2012.py`. Four nodes were added for this source:
`islam.sunni`, `islam.shia`, `indigenous.northeurasian`, `other.ru`.

**`islam` had no children at all before this.** Twenty countries in — Australia's 148
categories, the Philippines' 129, ASARB's 372, the UK's write-in tail that reaches Alevism —
and not one of them asks a Muslim which branch. Arena is the first source on this map that
does.

## 5. Four subjects are missing from Arena — and are now filled from census ethnicity

Arena covers **79 of 83**. Absent: **Chechnya, Ingushetia**, Nenets AO and Chukotka AO —
2,109,289 people, 1.46% of Russia. The two Arctic okrugs are 88,924 people between them and
nobody would notice; Chechnya and Ingushetia are 2,020,365 and are the two most Muslim
republics in the country, so the blank sat exactly where Islam is densest.

**Filled 2026-09-05, Anita's call.** Her reasoning: a blank region on a dot map does not
read as "no data", it reads as "nobody lives here", and it made Russia look less Muslim than
it is. `ru_fill.py` builds it and carries the full argument; the essentials:

- The magnitude comes from the **2021 census ethnic composition** (Rosstat Vol. 5 Tab. 1),
  through a relationship **fitted on Arena's own 79 measured subjects**:

  ```
  islam_answer = 0.724 x muslim_ethnic_share      R² = 0.964   (n = 79)
  ```

- **That slope is the whole point.** Only about seven in ten ethnically Muslim Russians give
  an Islam answer; the rest take "I believe in God but profess no particular religion", "I
  do not believe in God" or "difficult to answer". A naive fill would put 98% of Chechnya on
  Islam and make it the most religiously uniform unit on the entire map — more uniform than
  anything measured anywhere.
- Chechnya and Ingushetia use **Dagestan's own measured ratio, 0.861**, not the national
  slope, because the linear fit underpredicts at the top: Dagestan is 95.9% Muslim by
  ethnicity and answered 82.6%, against 69.4% predicted. Extrapolating a straight line to
  98.5% is exactly where it is weakest, and Dagestan is the nearest republic with the same
  Shafi'i Sufi tradition. The Arctic okrugs use the national slope, where the Muslim share
  is ~2% and the choice changes nothing.
- Everything Islam does not take comes from a **donor subject Arena did measure**,
  renormalised: Dagestan for the two republics, Arkhangelsk Oblast for Nenets AO (which sits
  administratively inside it), Magadan Oblast for Chukotka.
- **All Islam goes to the unspecified `islam` node.** Anita's instruction, and it is right:
  Arena's Sunni/Shia split is a measurement about the regions where it was asked, and there
  is no basis for projecting it into two republics where nobody was.

| subject | Muslim by ethnicity | drawn as Islam | people |
|---|---|---|---|
| Chechnya | 98.48% | **84.8%** | 1,281,536 |
| Ingushetia | 99.12% | **85.4%** | 435,019 |
| Nenets AO | 1.80% | 1.3% | 541 |
| Chukotka AO | 2.41% | 1.75% | 830 |

The fill adds **1,717,926 Muslims and 2,109,289 people**, and takes Russia to all 83
subjects and the full census population of 144,699,673.

**How it is kept separable.** The filled rows carry `source_id = ru_ethnic_fill_2021` and
live in `data/normalized/ru_filled.csv`, which is `ru.csv` plus four subjects; `ru.csv`
itself is untouched and is still exactly what Arena published. `tier` is `modelled` on
every Russian row either way, because a 720-per-region survey already is — so the tier
system cannot mark the fill and `source_id` is what does.

**The limitation worth naming.** Chukotka is 28% Chukchi and Nenets AO 18% Nenets, and both
donors have much smaller indigenous populations, so **traditional religion is understated in
both**. The same ethnic method would fix it; it is 89,000 people, about 89 dots, and a
second fitted model doubles the surface for error to move them. Not done, and recorded here
so it is not rediscovered as a good idea.

### And the transferable find: the Wayback Machine is the way past a TLS-walled office

§3 records that `rosstat.gov.ru` cannot be verified by any Western client. The census
ethnicity workbook was fetched anyway, because **`web.archive.org` holds it and Wayback's
own certificate is fine** — the `if_` suffix on the timestamp serves the original bytes
rather than a rewritten page. The URL came from a Wikipedia citation, which is the general
recipe: an article citing a statistical office usually carries an archive link beside the
dead one. This is now the first thing to try for any office behind a certificate wall.

## 6. What to distrust

**~720 respondents per subject.** A category at 0.3% nationally is about 180 respondents
across 79 regions, so its *national* figure is worth reading and its *map* is close to
noise. The clearest case is the Old Believers: Arena's top subject is Smolensk at 1.5%,
which is not where the historic Old Believer concentrations are. `tier` is `modelled` on
every row for this reason.

**Shares are published to 2 decimal places and many are suspiciously round** — 0.25, 0.40,
0.75, 1.20 recur across subjects, which is what a weighted sample of ~720 produces when one
or two respondents move. Do not read a difference of 0.2pp between two subjects as real.

**Column sums are not exactly 100.** Each share is rounded independently, so a subject can
total 99.6 or 100.4 — Kostroma is 100.37. `ru.py` renormalises before apportioning;
without it the floors overshoot the population and the largest-remainder step goes negative,
which is a bug that only fires in some subjects and passes every national check.

## 7. Vintage

**2012, and there will not be another.** Fourteen years, over a period when Russian religious
identification is one of the things most likely to have moved. Nothing on this map is older
except Brazil's 2010 structure and India's 2011 census, and both of those have a recent total
to rescale to. Russia has the 2021 population and no 2021 religion figure of any kind.
