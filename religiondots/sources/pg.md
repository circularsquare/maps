# Papua New Guinea (`pg`): record

Drawn 2026-10-03 by `fafd1067-pg`, reopening the IPUMS block of sources.md §11ab under Anita's
2026-09-15/16 rulings (a country is drawn on the best available figure with the method disclosed;
`sd`, `mr`, `om`, `sa`). 22 provinces, 10,185,363 people (2024 census), every row `modelled`.
Ask 047 asks whether to register with DHS.

## 1. What PNG publishes, and what was checked this time

§11ab and its 2026-09-09 re-check swept the National Statistical Office's whole library (287 files)
and retired SPC .Stat, PopGIS, HDX, ArcGIS Online and the USCB geodatabases. Their verdict stands:
**no table of religion below the nation exists online.** What they set aside as "one label and one
share per province" is this build's anchor (§2).

Checked 2026-10-03:

- **The 2000 census printed religion by province, on paper.** The *2000 Census Basic Tables*
  series (NSO 2002, three volumes a province: provincial level, rural sector, urban sector; e.g.
  NLA catalogue 1113474 for Enga, Google Books `3u5_rZg0eesC` for the urban-sector parts 4-6) has
  `religion`, `Salvation Army` and `Jehovah's Witness` among Google Books' indexed terms. So a full
  religion table per province exists. **Not online anywhere found:** Google Books is snippet view
  and its search-inside sends this machine to a captcha (`google.com/sorry`); HathiTrust's copies
  are search-only and `babel.hathitrust.org` and `catalog.hathitrust.org/Search` answer a Cloudflare
  challenge; archive.org has nothing; NLA's catalogue answers an Anubis wall. The 20 *Provincial
  Report* volumes (Google Books `IrIUAQAAMAAJ` Morobe, `3LIUAQAAMAAJ` Western) index no religion
  terms. A scan request is in ask 047 as the second route.
- **The old NSO site in the Wayback Machine** (CDX for `nso.gov.pg/*`, to 2016): a
  `2000_Census/census.htm` page (2005) and `census-a-surveys/census-2000/provincial-population-
  division` (a 404 in 2013). Neither could be opened: Wayback returned 429 to this machine and
  WebFetch refuses `web.archive.org`. No PDF under the old site carries census tables.
- **DHS 2016-18 has no religion indicator in its open API.** `api.dhsprogram.com/rest/dhs/
  indicators` (all indicators, 2026-10-03) has religion only as a reason or a provider (contraception,
  nets, violence, circumcision), never a composition. The microdata needs a registration: ask 047.
- **Adventist membership by local mission** (`adventiststatistics.org`) answers a Cloudflare 403.
  A browser job if anyone wants a second church's pattern.
- **Catholic dioceses** (catholic-hierarchy.org `scpg1`, the Annuario Pontificio's figures for
  2004): open, used as a seed (§3).

## 2. The anchor: the Summary Indicators' "Main religion" line

Both National Reports end chapter 2 with Summary Indicators by province, and the last row is
`Main religion (% of population)`: one church abbreviation and its share. 2011 (pp28-29, text
layer, checked by `sources/pg.py::check_2011_text`):

```
Western EA 37.1   Gulf UC 30.1   Central UC 40.0   NCD UC 23.0   Milne Bay UC 54.9   Northern Ang 60.6
SHP RC 19.7   Enga Luth 26.4   WHP Luth 26.0   Chimbu RC 34.4   EHP SDA 39.6   Hela EA 19.7   Jiwaka RC 29.6
Morobe Luth 67.0   Madang Luth 38.4   ESP RC 43.0   WSP RC 40.4
Manus RC 38.5   NIP RC 31.3   ENB RC 42.8   WNB RC 55.3   AROB RC 68.4
```

2000 (pp25-26, a scan, read at 200 dpi): the same church is largest in **17 of 20** provinces, the
shares mostly a few points higher. The three that changed: Southern Highlands (Other Christian
22.3 then, Catholic 19.7 in 2011, Hela split off), Western Highlands (Catholic 31.6 with Jiwaka;
Lutheran 26.0 without it in 2011, Jiwaka Catholic 29.6), New Ireland (United Church 40.3, Catholic
31.3 in 2011).

**Read as a share of Christians**, not of the population as the label says: the national row
(R/Cath 26.0) equals Figure 2.1's 26.0, and Figure 2.1's eleven churches sum to 100.2, so they are
shares of Christians (95.6% of citizens). In 2000 the same line is of the population (Figure 2.1
there includes Non-Christian and No religion). Reading the 2011 province figures the other way
would lower each largest church by 4.4% of itself; it changes no ranking.

## 3. The fit (`sources/pg.py::fit`)

On the 2011 census populations (the 2024 Final Figures' Table 2, whose 2011 column equals COD-PS
2011 by p-code), each province's Christians are 95.6/100.1 of its people. The largest church is
fixed at its printed share. The other ten churches fill the rest by iterative proportional fitting
to two margins: each province's remainder, and each church's national total (Figure 2.1) less what
the provinces where it is largest already hold. Every church's margin is positive (asserted). A
church that is not a province's largest is held 0.1 points under it (4 cells bind: the largest is
small in Western Highlands, Madang and the two Southern Highlands provinces).

Seeds, which move people between provinces but cannot change any total:

- **Catholic**: each diocese's Catholic share (2004) over the 19 dioceses' pooled rate (34.1%).
  Dioceses are provinces almost one for one (CBC PNG/SI's diocese pages); Kavieng is New Ireland
  and Manus, Mount Hagen is Western Highlands and Jiwaka, Mendi is Southern Highlands and Hela,
  Aitape and Vanimo are West Sepik. **Oro is inside Port Moresby archdiocese**: no diocese page says
  so, but Port Moresby's 2004 population (512,386) only fits with it (NCD 248,948 + Central outside
  Bereina about 100,000 + Oro 132,952 in 2000). Central is Bereina (Kairuku and Goilala, 83,733
  people at 81.1%) plus the archdiocese's rate on the rest of Central's 183,805.
- **The 2000 census's largest church where 2011's differs**, at its 2000 share over its 2000
  national share: New Ireland United Church x3.36 (40.3 / 12, the 12 from p31's text), Southern
  Highlands and Hela Other Christian x2.72 (22.3 / 8.2, the 8.2 measured off Figure 2.1's bar, which
  prints no value in 2000).
- Everything else uniform. Kwato Church (0.2%) belongs in Milne Bay but no source places it.

**Witness, Catholics against the dioceses' own share of the population** (printed by the build):
where the census names Catholics as largest, the diocese figure runs 1.06x to 1.54x the census's
(Chimbu 36.5 against 34.4, East Sepik 66.3 against 43.0), which is the usual gap between a
church's roll and self-identification. The exception is Southern Highlands, 13.1 against 19.7,
where Mendi diocese pools it with Hela and the pooled rate hides how the two differ. The fitted provinces sit in the same relation (Western 18.9
against 24.2, Morobe 3.9 against 7.0, Eastern Highlands 3.6 against 3.1), which is the seed doing
its job and not an independent check.

**What the fit gets wrong, knowingly.** A church strong in a few provinces without being largest
in any (Lutherans in the Highlands outside Enga and WHP, Adventists, the United Church in New
Britain and the Highlands, Pentecostals) is uniform outside its largest-provinces. So Lutherans are
5-10% of the Sepik and island provinces as drawn, Anglicans 1-2% everywhere but Oro. The note says
so. It is still more than the national-share build §11ab rejected: every province's largest church
and its size are the census's.

## 4. Mapping (`taxonomy/pg2011.py`)

Eleven churches to existing nodes; `Kwato Church` to `christianity.melanesianindependent` (left
the LMS in 1917); `Evangelical Alliance` to `christianity.evangelical` (Kenya's precedent);
`Other Christian` to `christianity.other`; `Non-Christian` to a new residual `other.pg`
(`taxonomy/branches.py`); `No Religion` (0.0% in 2011) to `unaffiliated`, drawing nothing;
`Not Stated` EXCLUDED, the gap (3.1%, `gap_share` 0.03097 hand-written from the rows).

## 5. Geography (`sources/pg_geo.py`)

COD-AB `cod-ab-png` admin1, 22 provinces with Hela and Jiwaka. The p-code-to-name table is
`sources/pg.py::PROVINCES`; its witness is COD-PS 2011 by p-code equal to the 2024 Final Figures'
2011 column by name for all 22. Population: 2024 Final Figures Table 2 (10,185,363).

## 6. Placement (`sources/pg_grid.py`)

Kontur PG 2023-11-01 (45,723 hexes), centroid join, 1,479 offshore hexes snapped within 2 km, 14
dropped (522 people). The land border with Indonesia is not Bhutan's case (Jaigaon
beside Phuentsholing, sources/bt.md): only 13 outside hexes lie west of 141.2 E, 2,843 people, most of them on the coast at Wutung,
and those whose centroid is west of 141.0 E hold about 200. Kontur over census 1.014; rank witness +0.877 against 20,000 shuffles (best
+0.757). Per province, NCD 0.57 and Western and Gulf 0.73 at the low end, Hela 1.79 and Enga 1.46
at the high end: it only places within a province, so this moves no dot between provinces. Seats:
Kontur within 5 km of every GeoNames seat over 10,000 holds at least 0.96 of its figure (GeoNames'
town figures are old). `kontur_cap.py pg`: no blocks. **Not done:** the 2024 census prints every
district's count (Final Figures, Provincial Snapshots), so Kontur could be calibrated to districts;
worth it if PNG ever gets district religion.

## 7. Reopen

- DHS 2016-18 microdata (ask 047): measured mix in all 22 provinces, adults 15-49, 21% catch-all.
- 2000 Basic Tables (print; §1): census religion by province, maybe district.
- The 2024 census, when it publishes religion: check for a province table first.

## 8. Review, 2026-10-03 (`fafd1067-rev6`)

Full pass. Checks clean, rollup clean, dots on land and dense where the Highlands and the towns
are. Re-summed `data/normalized/pg.csv`: each province's largest church equals pp28-29 as a share
of Christians, the note's Catholic figures (3.5% of Eastern Highlands, 23.9% of Enga) and Lutheran
range (4.9-9.5% of the people in the Sepik and island provinces) hold. National shares drift a
little from Figure 2.1 (Catholic 25.8 of Christians against 26.0) because the fit is on 2011
populations and the rows on 2024's; expected, not worth a note. Mapping matches precedent (Kenya,
`sb2019.py`); Kwato on `christianity.melanesianindependent` is a fair reading of the node's own
definition although Abel was a missionary rather than an islander.

Two wording fixes to `note_public`, then `tiles.py --refresh-meta`: the six named largest churches
now say "of the Christians" (the note said "Morobe is 67.0% Lutheran", and Morobe is drawn 64.0% of
its people Lutheran); and "every other church is drawn at one rate in every province" became "the
remaining churches are split in the same proportions in every province", since the rate itself
varies with what the largest church leaves (Lutherans 5-12% of Christians) and the next sentence
already said so.
