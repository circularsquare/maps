# Central African Republic — RGPH03 (2003), religion by commune

**Drawn 2026-09-07.** `sources/cf.py` -> `data/normalized/cf.csv`.
Boundaries and placement: `sources/cf_geo.md`. Mapping: `taxonomy/cf2003.py`.

| | |
|---|---|
| source | *Troisième Recensement Général de la Population et de l'Habitation* (RGPH03), 2003 census, published 2005 |
| publisher | Institut Centrafricain des Statistiques et des Etudes Economiques et Sociales (ICASEES) |
| route | U.S. Census Bureau tabulation on HDX — `sources.md` §11h |
| drawn tier | **177 communes** (ADM3) |
| people | **3,836,736** in the religion table; **3,895,139** in the census |
| coverage | **98.50%** |
| categories | **5** |
| basis | self-identification |

## 1. Why it was drawn, and why it took two days rather than one

§11h enumerated the USCB series on 2026-09-05 and listed CAR in its table: 177 communes,
3.8M people, 5 categories. Then Ethiopia, Pakistan, Bangladesh, Jamaica and Saint Vincent
were taken from that series and CAR was not, because five categories looked thin.

The 2026-09-07 Africa sweep is what changed the answer, and it did so by **exhausting the
alternatives rather than by finding something new about CAR**. UNSD table 28 (§11r's oracle)
says 23 African countries have ever produced a census religion tabulation; seven are drawn;
and of the sixteen that remain, the ones with better categories than CAR are each blocked:

* **Togo** (15 categories) publishes three RGPH-5 booklets and religion got none of them.
* **Eswatini** (20 categories, Zion Christian Church at 33.6%) — `eswatinistats.org.sz` does
  not answer from here and the Wayback CDX has no PDFs for it.
* **Côte d'Ivoire** (11 categories, including Harriste and Celestial) — `ins.ci` is a parked
  cPanel page and the office renamed to ANStat, which is behind Cloudflare.
* **Mozambique** (Saio/Zione at 16.3%) — the catalogue is reachable again, and the religion
  table in it is national and urban/rural only.
* **Uganda** (10 categories) — national only, established in §11b.

CAR is what is left, and the thing that makes it worth having is not the category list. It is
**177 units for 3.8 million people — 21,677 each**, which is finer per head than every other
African country on this map except Mauritius. Kenya is 1.01 million per unit, Zimbabwe 1.52
million, Malawi 549,000, Ethiopia 99,900.

## 2. The route

Two GETs, ~3.4 MB, no wall of any kind.

```
https://data.humdata.org/dataset/car-subnational-boundaries-and-tabular-data
  central_african_republic.gdb.zip            3,043,043 bytes
  central_african_republic_uscb_202303.xlsx     401,471 bytes
```

The workbook is the read and the geodatabase is the cross-check, as in `et.py`. USCB's own
citation, from the Metadata sheet:

> *Troisième Recensement Général de la Population et de l'Habitation (RGPH03), 2005,
> Réligion. Institut Centrafricain des Statistiques et des Etudes Economiques et Sociales
> (ICASEES). Available online at
> `http://108.60.219.85/redbin/RpWebEngine.exe/Portal?BASE=RGPH03FRA&lang=fra`
> (accessed November 17, 2022).*

So the underlying source is ICASEES's own REDATAM server, as it is for Bangladesh, Saint
Vincent and CAR's neighbour in the series. That host was not tested here; the USCB copy is
the one used, and it is what `sources.md` §11h's third point is about — the seam sidesteps
the office.

## 3. The five categories, and the one that is missing

| USCB column | census label | people | share |
|---|---|---:|---:|
| `RLG_PRO` | `Protestante` | 2,004,583 | 52.25% |
| `RLG_CAT` | `Catholique` | 1,122,899 | 29.27% |
| `RLG_MUS` | `Musulmane` | 400,962 | 10.45% |
| `RLG_OTHR` | `Autre réligion` | 171,441 | 4.47% |
| `RLG_NR` | `Sans réligion` | 136,850 | 3.57% |

**There is no traditional or animist box.** Ghana, Kenya, Ethiopia, Malawi and Benin all have
one; CAR does not. §11b's continental rule is that an exclusive traditional box undercounts,
because the practice commonly accompanies a Christian or Muslim affiliation rather than
replacing it. CAR is the sharper case: there is no box at all, so the practice is distributed
across `Autre réligion`, `Sans réligion` and the three named religions with no way to
separate any of it.

**The residuals' geography is the evidence, and it is strong.** `Autre réligion` reaches
23.7% in Topia, 21.0% in Moboma and Baleloko, 20.2% in Carnot and 19.5% in Lésse. `Sans
réligion` reaches 17.7% in Basse-Batouri, 17.2% in Basse-Kadéi and 16.7% in Mongoumba. Those
are the same two prefectures — **Lobaye and Mambéré-Kadéï, the south-western forest**, which
is the Aka homeland and Gbaya and Ngbaka country. A fivefold concentration against a 4.47%
national figure, in exactly the place traditional practice is strongest, in *both* residual
cells at once.

`taxonomy/cf2003.py` argues at length why this still does not license mapping `Autre
réligion` to `indigenous.african`: the cell demonstrably also holds CAR's Bahá'ís, Jehovah's
Witnesses, Kimbanguists and the Orthodox merchants of Bangui, and at the same time the
tradition's real size is *larger* than the cell because whoever also answered `Catholique` is
invisible. The mapping would be an overclaim and an undercount simultaneously. **CAR's
traditional religion is not drawn, and the map says so.**

## 4. The universe is 98.50% of the census, and only the metadata says so

This is §9y's shape — the universe, not the response rate, is the problem, and it is stated
nowhere in the data.

The data dictionary calls `RLG_BTOTL` *"Total population reporting a religion or belief
system"*. The anchor that turns that phrase into a number is **the `Ethnicity` sheet of the
same workbook**: same census, same 2003 geography, and its national total is **3,895,139 —
the published RGPH03 population**. Religion covers 3,836,736 of it.

> **58,403 people (1.50%) were counted by the census and are not in the religion table, and
> there is no cell that says so.** `Sans réligion` is not it — that is a separate published
> category of 136,850.

Per commune the coverage runs **92.6% to 99.9%, median 98.8%**, with only Lobaye (92.6%) and
Nola (94.9%) below 95%. Evenly spread, so it is ordinary non-response rather than a
structural hole in one part of the country. Reported, not filled (spec §3.5).

### The trap next to it, which would have shipped a wrong number

**Do not use the `Age-Sex` sheet as the denominator.** Its national total is **5,052,901**,
because it is a **2016 estimate** — the HDX dataset notes say *"total population for 2016
(estimates) and 2021 (estimates), five-year age group and sex (2016 only)"* and the sheet
itself says nothing. It is 31% above the census. Using it would have turned a 98.5%-covered
country into a 76%-covered one and produced a large invented gap.

**One workbook can hold two vintages and label neither in the data.** The GEOG1/GEOG2 split
in the boundary layers is the same fact wearing a different disguise (`cf_geo.md` §2).

## 5. The checks

`cf.py` prints all of these on every run.

1. **No `-999` sentinel.** Ethiopia's is in-band and arithmetic-safe (§9u), so its *absence*
   is asserted rather than noticed. 0 negative cells in this release.
2. **Row counts per level** — 1 country, 17 prefectures, 72 sous-préfectures, 177 communes.
3. **The partition is exact to rounding and two-sided.** Categories minus total over all 267
   rows: `{-2: 2, -1: 66, 0: 131, 1: 64, 2: 4}`. The national row is −1 on 3.8 million; the
   177 communes sum to 3,836,741 against it, five people over. **Nothing is one-sided**,
   which is what independently rounded published figures look like (§9at, §9ak) rather than
   a dropped category. The whole distribution prints rather than vanishing into a tolerance.
4. **Every level partitions the country**, category by category, within the rounding bound.
5. **The cross-table anchor** — the Ethnicity sheet's national total *is* the published
   RGPH03 population, asserted equal to 3,895,139.
6. **Per-commune coverage** — all 177 communes in both sheets, min/median/max reported and
   the sub-95% ones named.
7. **The workbook and the geodatabase agree** on all 1,335 religion cells. Same publisher and
   release, so this catches a misread rather than being independent (§9u's wording kept).

## 6. What it shows

* **The most Protestant country on this map** — 52.3% against 29.3% Catholic. That is the
  map of four mission fields that divided the country in the 1920s: Baptist Mid-Missions and
  the Swedish Örebro mission north and centre, the American Grace Brethren around Bangui, and
  the Africa Inland and Sudan United missions east. The census names none of them, so this is
  one undivided colour where Malawi next door shows three.
* **A hard Muslim north-east.** Vakaga 87.4%, Bamingui-Bangoran 44.5%, against 2.1% in
  Nana-Grébizi and 2.3% in Ouham. Ouandja commune is 96.1%. Islam is 10.5% nationally and the
  top 20 communes of 177 hold 57% of it.
* **A Catholic east and centre.** Haut-Mbomou is 50.6% Catholic and Ouaka 42.2%, against
  15.5% in Nana-Mambéré — roughly the old Spiritan and Capuchin mission territories.
* **And the forest south-west, which is where both residuals live** (§3 above).

## 7. The limit, and it is the vintage

**RGPH03 is CAR's last census.** A fourth was attempted repeatedly from 2013 onward and has
not happened. So this is a twenty-three-year-old table and there is nothing newer to draw.

That matters more here than for almost any other country on this map, because of what
happened in between. **The Séléka and anti-balaka conflict from 2013 displaced a large part
of the Muslim population of the west and centre** — Bangui's PK5 quarter, Bossangoa, Bouar,
Carnot — and much of it left the country or has not returned. Roughly a fifth of the
population has been displaced at some point since.

So the Muslim geography drawn here is **where CAR's Muslims were in 2003, ten years before
the event that moved them**. `note_public` says this outright rather than leaving a reader to
assume a current map. It is not corrected, because correcting it would mean inventing a
magnitude (§14.4) — and there is no source that would support one at commune level.

## 8. Access notes

* No wall, no key, no account. Two GETs from `data.humdata.org`.
* Both files are zip containers (`PK\x03\x04`), so the magic-byte check in `fetch()` is the
  same for the `.gdb.zip` and the `.xlsx` (§5a, §11d).
* The dataset was last modified 2023-04-05 and the CKAN `package_show` route gives the
  resource URLs, which are UUIDs and are not guessable — re-derive them rather than editing
  the constants if the dataset is ever re-released.
* `pyogrio` is needed for check 7 and for `cf_geo.py`; `cf.py` degrades to a printed note
  without it.
