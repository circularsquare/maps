# Iran: census 1395 (November 2016), religion by province

**Drawn 2026-09-14** by session `d743fc47-ir`. 31 provinces, 5 nodes, 79,801,698 people drawn;
124,572 who did not state a religion (0.156%) are `gap`. Every row `measured`, in counts.

- `sources/ir.py` -> `data/normalized/ir.csv` (raw: `data/raw/ir/sci_yearbook1395_ch3_population.pdf`,
  `census1395_tafsili_3-jamiat-k.xls`)
- `sources/ir_geo.py` -> `data/geo/ir/ir_provinces.gpkg`, `ir_hexes.gpkg`, `ir_lookup.csv`,
  `ir_county_lookup.csv` (COD-AB ADM1 and ADM2, COD-PS ADM2 2016, Kontur 400 m)
- `taxonomy/ir2016.py` -> the mapping; `countries/ir.py` -> the entry; `taxonomy/branches.py`
  `other.ir` is the one new node
- `sources.md §ir-2026-09-14` is the summary; `§scout-2026-09-14-iran` and §11n are the earlier
  scouting; `ask/answered/023` is Anita's ruling.

```
python sources/ir.py     --fetch
python sources/ir_geo.py --fetch
```

## 1. What the Statistical Centre of Iran (SCI) publishes

| release | religion | tier |
|---|---|---|
| **Statistical Yearbook 1395, ch. 3, Table 3-18** `جمعیت برحسب دین واستان: آبان ۱۳۹۵` | **province x religion, counts: Muslim, Zoroastrian, Christian, Jewish, other, not stated** | **31 provinces (drawn)** |
| same chapter, Table 3-17 | religion x sex, 1385, 1390, 1395 | national |
| *Amar* no. 21 (Azar-Dey 1395), Elham Fathi, Table 3 | 1390 shares to two decimals; two Christian columns; other merged with not stated | 31 provinces (witness) |
| UNSD Demographic Yearbook table 28 | 1996, 2006, 2011, 2016, six categories | national |
| 1395 detailed results (`Portals/0/census/1395/results/tables/`) | no religion topic in any folder | - |
| 1390 2% public microdata sample | province and county, no religion variable | - |
| amarfact.com | Kermanshah and Semnan 1395, identical to Table 3-18 (a transcription, not a source) | 2 provinces |

The yearbook chapter is read from Syracuse University's Iran Data Portal
(`irandataportal.syr.edu/wp-content/uploads/Population-3.pdf`, 56 pages, 2,228,372 bytes, SHA-1
`WRANG4GEIGWXTSGTMKI7JDZSZP2EHXW4`, pinned). SCI's own copy (`amar.org.ir/Portals/0/Files/fulltext/
1395/n_Salname_95-V3.pdf`) survives in the Wayback Machine truncated at 8.6 MB (§11n), and
`amar.org.ir` resets every connection from here. The table's text layer has all 31 province rows in
Persian digits; the bold national row is a picture and is transcribed. A Wayback CDX query for the
Syracuse URL answered 504 on 2026-09-14, so whether an archived copy exists is **not checked**.

**How it was found.** A Persian search for `سالنامه آماری کشور ۱۳۹۵ جمعیت بر حسب دین به تفکیک استان`
returned the Syracuse PDF; `pdf_find` on NFKC-normalised text found `دین` on pages 35-36. §11n and
the scout had searched the census results tree and SCI's magazine; neither opened the yearbook's
population chapter. The negative that closed Iran on 2026-09-06 was one release's table list.

## 2. Why 1395, when ask 023 named the 1390 shares

Ask 023 was filed on the only province table then known, *Amar* no. 21's 1390 shares, and Anita ruled
(2026-09-14): draw the 31 provinces, and keep other on a coloured node rather than moving it to
`gap`. Table 3-18 is better on every axis the ask raised: counts instead of two-decimal shares
(Jewish read 0.00 in 27 provinces), other and not stated printed apart, and one Christian column
instead of the doubtful "Assyrian or Chaldean" split. The grain and the treatment of other are as
ruled. Not stated, now a column of its own, is handled as in every other census here (`EXCLUDED`,
`gap`); the ruling was about a merged column and does not reach a separate non-response column.

## 3. The checks (all in `sources/ir.py`, all pass)

| check | result |
|---|---|
| Table 3-18 parsed off p159 = transcription | 31 rows identical |
| each province's six cells = its total | 31/31 |
| provinces sum to the national row | all seven columns |
| national row = UNSD table 28, Iran 2016 | all six categories, to the person |
| Table 3-17 (p158): 1395 column = national row; men + women = both; 1390 = UNSD 2011; 1385 = UNSD 2006 | all |
| province totals = 1395 detailed results `3-jamiat-k.xls` (Wayback 2017, pinned digest) | 31/31 to the person; its citizenship columns close |
| *Amar* no. 21 Table 3 (1390): rows sum to 100; national row rounds from UNSD 2011 | within 0.02; exact |
| same leading province in 1390 and 1395 | Zoroastrian Yazd, Jewish Fars, Christian Tehran |

Spearman across provinces, 1395 share against 1390: Zoroastrian +0.80; Christian +0.91 against the
1390 `آشوری یا کلدانی` column, +0.44 against its `مسیحی` column, +0.94 against the two summed.

## 4. The mapping (`taxonomy/ir2016.py`)

| column | people | node |
|---|---:|---|
| `مسلمان` Muslim | 79,598,054 (99.589%) | `islam` |
| `مسیحی` Christian | 130,158 (0.163%) | `christianity` |
| `سایر` other | 40,551 (0.051%) | `other.ir` (new) |
| `زرتشتی` Zoroastrian | 23,109 (0.029%) | `zoroastrianism` |
| `کلیمی` Jewish | 9,826 (0.012%) | `judaism` |
| `اظهارنشده` not stated | 124,572 (0.156%) | `EXCLUDED`, `gap` |

- **The 1390 "Assyrian or Chaldean" column does not name Assyrians.** It was a flat 0.04-0.18% in
  every province, and across provinces it tracks the 1395 Christian share (+0.91) better than 1390's
  own `مسیحی` column does (+0.44). So in 1390 the concentrated part (Tehran, Isfahan) sat under
  `مسیحی` and a thin, even share sat under the Assyrian label. Neither label can put anyone on
  `christianity.oriental`; 1395 has one column and it goes to bare `christianity`.
- **That thin, even share is in 1395 too.** Christians are 0.08% (Golestan) to 0.16% (Bushehr) of
  28 provinces, including Sistan and Baluchestan (4,155) and Ilam (859), and above that only in Tehran
  (0.33%, a third of all Christians), West Azerbaijan (0.23%) and Isfahan (0.17%). The census does not
  say who they are. Drawn as printed; not explained here.
- **Muslims** are one column; the table does not split Shia from Sunni.
- **Other.** The census has no Baháʼí answer. A Baháʼí who answered other is in `other.ir` and one
  who declined is in not stated. Largest in Tehran (9,568), Isfahan (6,014), Fars (4,649), Alborz
  (4,027), Kerman (3,540). Anita's ruling on ask 023 covers this; a Baháʼí-specific source is hers to
  prioritise later (low).
- Oddity, not used: Table 3-17's 1390 not-stated row is 256,321 men and 9,578 women; it closes, so it
  is printed that way.

## 5. Geography and placement (`sources/ir_geo.py`)

**Units.** COD-AB Iran ADM1 (`cod-ab-irn`, valid_on 2019-05-14), 31 provinces, unchanged since
Alborz in 2010. Joined on two keys that must agree: the hand-written English name against
`adm1_name`, and the yearbook's Persian name against COD's own Persian `adm1_name1` (Arabic yeh and
kaf folded). Witness: raw Kontur/census per province has a log-ratio spread of 0.192 against a
minimum of 0.720 over 2,000 shuffled pairings.

**Kontur is badly wrong inside provinces, so the grid is calibrated to the census county.** Kontur
2023-11 is 1.11x the census overall and 0.82x (Ilam) to 1.90x (Fars) by province. Fars's excess is
three rural counties south-east of Shiraz, each drawn as a false city at Kontur's 46,200/km2 limit:

| county | census 1395 | Kontur 2023 |
|---|---:|---:|
| Sarvestan | 38,114 | 1,779,613 |
| Kavar | 83,883 | 1,788,778 |
| Kherameh | 54,864 | 1,677,708 |
| Shiraz | 1,869,001 | 1,238,528 |

The same shape recurs: Torghabe-o-Shandiz outside Mashhad (69,640 against 1,723,708), Bavi in
Khuzestan, Sareyn in Ardabil, Famenin in Hamadan, Malard, Shahr-e Qods and Pishva around Tehran.
`kontur_cap.py ir` listed 78 blocks at the limit, and against `maps/data/worldcities.csv` almost every
one outside central Tehran and Mashhad is a small town holding 3-30 times its figure. **Within their
provinces Kontur puts 14.6 million people, 18.3% of Iran, in the wrong county.** Capping blocks could
not fix that (the ring median around a false city is its own ramp).

COD-PS Iran ADM2 2016 (`cod-ps-irn`, `irn_admpop_adm2_2016_v2.csv`) is the 1395 census by county: its
429 counties sum to every Table 3-18 province total to the person, men plus women close on every row,
and its pcodes are exactly COD-AB's 429 ADM2 polygons under the same provinces. Each hex's `pop` is
Kontur's value times its county's census total over its county's Kontur total; `kontur_pop` keeps the
raw figure. Placement only; counts stay at province. It is spec §12's Haiti rule (scale a false
commune, do not cap it) applied to every county, because here the county figure is a census count.
The calibrated layer exceeds Kontur's limit where a real core was under-weighted, so `kontur_cap.py`
skips it as not raw Kontur (as for cn, kr, bg), and no `kontur_cap.csv` rows were added. What remains
is Kontur's shape inside a county.

**Checked on the scattered dots (1:1,000, 79,800 dots).** Shiraz county has 38.5% of Fars's dots
(census share 38.5%, raw Kontur 13.4%); Sarvestan, Kavar and Kherameh 0.8%, 1.7%, 1.2% (census 0.8%,
1.7%, 1.1%; raw Kontur 19.3%, 19.4%, 18.2%). Across all 429 counties the largest gap between a county's
share of its province's dots and its census share is 0.83 points (Khorramabad).

## 6. Searched and empty (this session, 2026-09-14; the scout's list is in `sources.md`)

- WebSearch, Persian: `جمعیت بر حسب دین استان سرشماری ۱۳۹۵` (found Tehran's provincial planning
  office news, iranopendata age-sex tables, Wikipedia pages citing Fars and Yazd 1395 figures);
  `amarfact دین استان` (two province pages and a national page, no index of the rest).
- The yearbook search is what found Table 3-18 (section 1).

## 7. Open

- A Baháʼí-specific source: Anita, low priority (ask 023).
- The 1395 questionnaire: not found (the scout's list). Table 3-18's columns are the only evidence of
  the 1395 answers.
- The thin Christian share in 28 provinces: unexplained.
- County religion: nothing seen below province in any release; not searched beyond the yearbook
  chapter and the scout's census tree.

## 8. Review, 2026-09-14 (`d743fc47-rev2`, full pass)

`check_md` clean, `check_rollup ir` all measured, both dot editions present. Screenshot at Iran's
bbox: dots follow Tehran, Mashhad, Isfahan, Shiraz, the Caspian coast and the Zagros, the deserts
are empty, nothing is in the sea, and Shiraz is a city with no false cities south-east of it.

- **1395 over the ruling's 1390 shares: agreed.** The vintage is the builder's (`AGENT_BRIEF.md` §2),
  and Table 3-18 settles both data problems ask 023 left to the builder. Grain and other's node are
  as ruled.
- **Not stated to `gap`: agreed.** The ruling was about a merged column. The 1390 standout it worried
  about, Bushehr's 2.95% other or not stated (30,473), is not in 1395 in either column (other 300,
  not stated 2,455), so moving not stated off the map hides no province standout.
- **County calibration: agreed.** The county figure is the same census, and §5's check on the
  scattered dots is the right one. `other.ir` follows `playbooks/census_table.md`'s `other.<cc>` rule.
- **The thin Christian share behaves like a recording error, not a community.** §4 leaves it
  unexplained. From `data/normalized/ir.csv`:
  - It is highest where no church is known and lowest where one is: Bushehr 0.158%, Sistan and
    Baluchestan 0.150%, Ilam 0.148%, against East Azerbaijan 0.108% (Tabriz's Armenian churches) and
    Gilan 0.087%.
  - Across the 25 provinces outside both the Christian cores (Tehran, West Azerbaijan, Isfahan) and
    the Zoroastrian ones (Tehran, Yazd, Kerman, Isfahan, Alborz), the Christian and Zoroastrian shares
    rank together: Spearman +0.61, one-sided permutation p = 0.001 on 5,000 shuffles. Against not
    stated +0.11, against province population -0.15. Two minorities rising and falling together in
    provinces where neither has a community points to a shared recording mechanism. Kyrgyzstan's
    `BUDDHIST` cell (`sources/kg.md` §4.2) had the same signature: highest in the most uniformly
    Muslim units.
  - It is 69,896 of the 130,158 Christians drawn (53.7%), so most of Iran's Christian dots are this
    share. The same floor probably sits under Zoroastrians: 8,048 of 23,109 (35%) are outside those
    five provinces, including 473 in Sistan and Baluchestan and 137 in Ilam.
  - **Not moved, and I would not move it.** Subtracting a floor would be a model laid over printed
    counts. What I would change is `note_public`'s "in the other 28 the census counts Christians at
    0.08% to 0.16%", which reads as a Christian geography. One sentence saying the share is highest
    in provinces with no known church (Bushehr, Sistan and Baluchestan, Ilam) and that the census
    does not say who these people are would be honest. Not edited; the note is the builder's.
- **"There is one Christian answer"** in `note_public`, and "The answers are Muslim, Zoroastrian,
  Christian and Jewish": §7 says the 1395 form was not found, and the 1390 and 1385 forms had three
  Christian sub-answers. What is known is that the yearbook prints one Christian column. Small; not
  edited.
