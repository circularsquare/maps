# Solomon Islands — SINSO, 2019 Census, Report Vol 2 Basic Tables, Table P8.3

Wired 2026-09-08. 720,956 people, **183 wards**, 16 drawn categories, **99.98% of the country
drawn**.

| | |
|---|---|
| source | Solomon Islands National Statistics Office, **2019 National Population and Housing Census**, *Report Volume 2, Basic Tables*, Table P8.3 — *Total population by ward and religious denomination* |
| basis | `self_id`, whole enumerated population, no age floor |
| geography | **183 wards**, 3,940 people each — the finest tier of any Pacific country on this map |
| categories | **18 columns**: sixteen drawn bodies, one refusal residual, one total |
| drawn | **720,823 of 720,956, 99.98%**; the 133 not drawn are `Religion Faith/Refuse to Answer` |
| licence | published census table, open download, no account |
| witness | Volume 1's Table 8.3.1 reproduces every figure exactly |

**The table reconciles to the person**, which is rare enough here to lead with: the 183 ward
rows sum to the printed national row exactly on all eighteen columns, the ten province rows do
too, and every ward's own categories sum to its own total. No tolerance is used anywhere.

---

## 1. Access — the WP File Download plugin for the third time

`statistics.gov.sb` is WordPress with an open REST API. Its **media library is a dead end**:
455 items, all uploaded 2025-26 in a site rebuild, and not one census report among them.

What holds the reports is the same **WP File Download** plugin as Fiji (§9bd §1) and PNG
(§11ab), found because a news post linked a file at
`/download/45/press-releases/921/urban-and-rural-distribution.pdf` — the plugin's own URL
shape. `id=0` on its AJAX route returns the whole library, **330 files**, and the census sits
in a category named *Solomon Islands 2019 Population and Housing Census National Report_Volume
1 and 2*:

```
  1207  Vol 2, Basic Tables    5.1 MB  318 pp   <- drawn, table P8.3
  1208  Vol 1, National Report 8.0 MB  305 pp   <- the witness, table 8.3.1
   764  2009 National Report Vol 1
   765  2009 Basic Tables Vol 2
   761  Census1999_Files.zip
```

**Three routes that looked promising and are not:**

* the **ten per-province XLSX files** in the `Population` category are population projections
  by single year of age, 2010-2025. No religion.
* `www.statistics.gov.sb/sinso-documents`, still in the site menu, is a **404**.
* **SPC's PopGIS is live at `solomons.popgis.spc.int`** and has a `p11_religion` dataset in
  its config — but the theme is not in any indicator tree, `GC_listIndics.php` returns an
  empty list for every theme, and `GC_coldata.php` answers **"Unavailable service"** on this
  instance where Fiji's answers with data. So the 2009 religion figures are behind a disabled
  endpoint. It remains a good geography reference: 183 wards, 50 constituencies, and 2009
  enumeration areas.

## 2. The tier — 183 wards, and the categories come with it

P8.3 prints the same eighteen columns at the nation, at each of the ten provinces, and at
**every one of the 183 wards**. 3,940 people a ward. As in Vanuatu (§9bg §2) there is no
trade-off to argue: the deep list and the fine geography are the same table.

## 3. Getting it off the page — the column COUNT is the only usable anchor

Landscape pages, eighteen right-aligned numeric columns, rows running province-then-its-wards.
Three approaches fail before the fourth works:

1. **A fixed x map fails**, because each page sizes its columns to its own widest figure and
   the positions move from page to page.
2. **Pooling every page and clustering fails**, because that same drift merges neighbouring
   columns: pooled, the eighteen come out as seventeen.
3. **Clustering each page with a gap threshold fails on the last page**, which holds only
   twelve wards. With that few samples the within-column spread swallows the between-column
   gaps.
4. **What works is that the column count is known.** Sort a page's right edges and cut at the
   seventeen largest gaps. On pages 156-162 the smallest between-column gap is 21 points
   against a largest within-column gap of 7, so the split is not close.

**And the last page needs bounding, or it is silently wrong.** Page 163 carries the tail of
P8.3 *and the whole of P8.4* (ethnic group by province, 13 columns). P8.4 has **its own
`Province` header**, so taking the last one on the page anchors the parse to the wrong table,
and its rows smear the edge distribution until no split separates. `sources/sb.py` bounds the
table by its own title above and the next `P8.n` title below.

## 4. What is in the table

```
  Church of Melanesia        232,041  32.19%    Christian OutReach Ch.   5,582  0.77%
  Roman Catholic             144,078  19.98%    Custom Beliefs/Animism   4,115  0.57%
  South Sea Evangelical      124,506  17.27%    Assembly of God          3,756  0.52%
  Seventh Day Adventist       83,452  11.58%    Bahai Faith              3,104  0.43%
  United Church               66,915   9.28%    Pentecostal              3,019  0.42%
  Christian Fellowship Ch.    16,179   2.24%    Baptist Church           2,172  0.30%
  Other religions             14,953   2.07%    No Religion or Atheism   1,227  0.17%
  Jehovah's Witness           14,624   2.03%    Muslim                   1,100  0.15%
  --------------------------------------------------------------------------------
  Religion Faith/Refuse to Answer 133   0.02%   §3.5 residual, NOT DRAWN
  Total                      720,956
```

**133 refusals in a country of 721,000 is the smallest residual anywhere on this map** —
0.018%. Vol 1's text says so in as many words: *"a small proportion of people (1,227) claim
that they had no religion, and 133 people refused to provide any information."*

**The census names Baha'is and Muslims as their own cells at ward level**, which very few do.

## 5. The three churches worth the trouble

**Church of Melanesia**, 32.2% and the largest body: the Anglican province of Melanesia, out of
the Melanesian Mission that worked from Norfolk Island from the 1850s. Its geography is the
sharpest thing in the file — **89.1% of Isabel, 85.1% of Temotu, 82.3% of Central**, and 2.0%
of Choiseul.

**South Sea Evangelical Church**, 17.3%: the **Queensland Kanaka Mission**, founded by Florence
Young in 1886, evangelised Solomon Islanders working the Queensland cane fields, and they took
the church home. **Malaita, which sent most of those labourers, is 28.1% SSEC; Isabel is 0.4%.**
No other source on this map records a church founded among migrant workers abroad that became a
national church at home. It is filed on `christianity.evangelical` — see `taxonomy/sb2019.py`
REVIEW, and the Kenya precedent it rests on.

**Christian Fellowship Church**, 2.2%: Silas Eto, the **Holy Mama**, broke with the Methodist
mission on New Georgia in 1960. **13,629 of its 16,179 members are in Western Province**, and
by ward:

```
  Kusaghe          1,935 of 2,508   77.2%      North Rendova    771 of 1,907   40.4%
  Roviana Lagoon   3,259 of 5,206   62.6%      Vonavona       2,029 of 6,398   31.7%
  Kolombaghea      2,039 of 3,565   57.2%
```

New node `christianity.melanesianindependent.cfc`, in the `Locally founded churches` lineage
group beside the Maori, Filipino and African-instituted churches.

## 6. Custom belief, and why the ward tier is what shows it

`Custom Beliefs or Animism`, 4,115 people, **0.57% nationally — and 21.5% of one ward**:

```
  Waneagu/Taelanasina   Malaita        791 of  3,682   21.5%
  Tetekanji             Guadalcanal    340 of  1,602   21.2%
  Gulalofou             Malaita      1,056 of  8,609   12.3%
  Vulolo                Guadalcanal    617 of  6,265    9.8%
```

That is the **Kwaio interior of east Malaita** and the **Weather Coast of Guadalcanal**, the
two parts of the country the missions reached least. **At province level Malaita is 1.2% and
the pattern is invisible**; this is the clearest case on the map of the tier deciding whether
something can be seen at all. New node `indigenous.solomon`.

## 7. The join is on SINSO's own ward id, and the names would have failed

COD-AB Solomon Islands ADM3 is **SINSO's own ward layer** — HDX's description says *"Sourced
from Solomon Islands National Statistics Office (SINSO), 2009 Census"* — and it carries
**`SINSO_WID`** beside the OCHA pcode. The census prints a ward number that restarts at 01
inside each province, and the id is the province number followed by it padded to two digits, so
Choiseul ward 1 is `101` and Honiara ward 1 is `1001`. **183/183, nothing spare either way.**

**A name join would have matched 154 of 183.** Twenty-eight of the misses are orthography:
Solomon Islands English writes prenasalised stops both ways, so the census's `Mbilua`,
`Ndovele`, `Mbuini Tusu` and `Banika` are COD's `Bilua`, `Dovele`, `Buini Tusu` and `Mbanika`.
Apostrophes and separators account for most of the rest.

> **The twenty-ninth is not a spelling. Isabel ward 02 is `Baolo` in the census and `Havulei`
> in COD.** Neither name appears anywhere on the other side and every other Isabel ward pairs
> on both id and name, so it is a renamed ward — but it is precisely the failure a name join
> cannot see, and it is why this one is on the id.

**The witness is the province.** OCHA files each ADM3 under an ADM1; the census files each ward
under the province whose block it prints in. Two organisations, **agreeing on all 183**.

No antimeridian problem: 155.5°E to 169.9°E, asserted anyway.

## 8. Placement — snapping, one atoll, one lagoon and one ward too small to see

`sources/sb_grid.py`, 6,385 Kontur 400 m hexagons, counts 2019 against a 2023 grid.

**1,230 hexes and 126,579 people — 17.1% — fall outside every ward**, higher than Vanuatu's
13% and for the same reason. **99.7% of them are within 500 m of a ward** and are snapped to
the nearest; 38 cells and 403 people (0.054%) are dropped. Vanuatu's rule (§9bg §9): the loss
is seaward and therefore directional, and dropping it would pull every coastal ward's dots
inland.

**Two wards remain outliers and both are the grid's blind spot rather than a bad join:**

```
  Sulufou/Kwarande   census 2,747  vs Kontur   482   0.17x
  Sikaiana           census   359  vs Kontur     7   0.02x
```

**Sulufou/Kwarande is the Lau Lagoon, where people live on artificial islands built of coral**,
and **Sikaiana is a Polynesian atoll 210 km out from Malaita.** A building-footprint model
under-detects both. Together they are 0.43% of the counted population. The check names them and
allows them; more than four such wards would fail the build.

**And one ward is smaller than one hex.** `Naha` in Honiara is 0.08 km² with 464 people, and no
400 m hex centroid falls inside it — the Kontur resolution floor rather than a join failure.
Snapping it to a neighbour's hex would place its dots in the wrong ward, and dropping it would
remove the ward from the map, because the hexes *are* the geography here. It is given its own
polygon as a single cell, which is §8.2's equal share over the unit, and it is excluded from
the band and correlation because it has no measurement to be scored against.

```
  national      Kontur 740,236 vs census 720,956, ratio 1.027
  band          180 of 182 inside a factor of 5
  correlation   r = 0.8974, against a best of 0.2749 over 2,000 random pairings (0 reach it)
```

## 9. What the map shows

**Read it as the mission map of the 1800s, because that is what it is.** Anglicans took the
east and the small islands, the Methodists took the west, the Catholics took Guadalcanal, and
the South Sea Evangelicals came home to Malaita with the labour trade. Three provinces are
over 80% one church and no province resembles its neighbour.

```
                        CoM     RC   SSEC    SDA  United    CFC
  Isabel               89.1%   1.5%   0.4%   2.1%    0.7%   0.1%
  Temotu               85.1%   0.3%   0.4%   2.6%    0.2%   0.3%
  Central              82.3%   9.3%   3.6%   3.7%    0.4%   0.1%
  Makira-Ulawa         45.6%  21.1%  23.9%   3.1%    0.4%   0.1%
  Honiara              31.3%  14.3%  23.3%  15.6%    6.3%   0.9%
  Malaita              28.8%  24.0%  28.1%   5.7%    0.2%   0.2%
  Guadalcanal          26.5%  36.2%  17.7%  11.5%    3.1%   0.3%
  Rennell-Bellona       5.9%   1.4%  37.3%  47.8%    0.5%   0.1%
  Western               4.9%   7.5%   3.1%  26.4%   38.8%  14.5%
  Choiseul              2.0%  22.6%   1.4%  15.9%   53.5%   1.2%
```

**Rennell-Bellona is the odd one**: 47.8% Seventh Day Adventist and 37.3% South Sea
Evangelical, the only province where neither of the two largest national churches is close to
first. It is also the only Polynesian province.

## 10. What else is on this server, unused

- **The 2009 census basic tables** (file 765) and its national report are both open. Religion
  is in them; 2019 is newer at the same ward tier, so there is no reason to prefer 2009.
  A 1999-2009-2019 series exists in Vol 1's Table 8.3.1 and is quoted there.
- **Vol 2 also crosses religion with sex** (P8.2 is by province, and Vol 1's 8.3.2 by sex).
  Not read.
- **P8.4 and P8.5 are ethnicity by province and by sex** — Melanesian 95.5%, Polynesian 2.8%,
  Micronesian 1.2% — and would answer whether Rennell-Bellona's Adventist majority tracks its
  Polynesian population. Not read.
- The **2009 census provincial profiles** (nine PDFs, category 57) were not opened.
