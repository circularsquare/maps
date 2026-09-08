# Ethiopia — 2007 Population and Housing Census, via the U.S. Census Bureau

`sources/et.py` → `data/normalized/et.csv`. 73,750,932 people, 738 woredas, 6 categories,
**100% of the published tabulation drawn**.

| | |
|---|---|
| source | `ET_RELIGION_2007census_uscb_202308`, a layer of the USCB Ethiopia geodatabase |
| publisher | **U.S. Census Bureau**, not the Ethiopian Statistics Service |
| url | `https://data.humdata.org/dataset/ethiopia-subnational-boundaries-and-tabular-data` |
| geography | woreda (738 counted, 743 listed) — 99,900 people each |
| categories | 6: Orthodox, Protestant, Catholic, Islam, Traditional, Other |
| basis | `self_id` |
| year | 2007 |
| access | open, no login, no key, no bot wall. Two GETs, 5.4 MB total. |
| licence | HDX, U.S. Census Bureau — public domain as a US Government work |

`sources/et_geo.md` is the boundary and placement half. `sources.md` §11h is the general
finding about this publisher, which is the part that matters beyond Ethiopia.

---

## 1. §11b priced this country wrong, and the reason is worth keeping

`sources.md` §11b, written 2026-09-05, ranked Ethiopia 2007 **the largest untouched African
source** — "woreda level, ~670 units, 74M people, six categories" — and priced it at *eleven
regional PDF volumes on a website that has since moved*. Both halves of that were true about
the Ethiopian statistical agency and neither was the binding constraint, because **somebody
else had already done the work and published it with the boundaries attached.**

The search that found it was not an Ethiopian search at all. It was a CKAN query for
`religion` against `data.humdata.org`, which returned ten datasets, eight of them from one
publisher and one of them already in use in this repo for something else entirely
(`sources/ph_geo.md` reads the Philippine geodatabase for its geography and had never looked
at what other layers were in the file).

**The rule this adds to §11b's "ask how big the table is":** ask *who else* publishes a
country's census. A national statistical office is the obvious source and it is not the only
one. Humanitarian and defence data programmes re-publish census tabulations, and they do it
in formats built for joining rather than for reading.

## 2. What is in the file

The geodatabase carries fourteen layers — four geography, ten tabular — all keyed on
`GEO_MATCH`, a positional code of the form `ETH_08_07_03` (country, region, zone, woreda).

```
ET_GEOG_ADM0_2021_uscb_202308            1 polygon
ET_GEOG_ADM1_2021_uscb_202308           24 polygons     13 regions + water
ET_GEOG_ADM2_2021_uscb_202308          104 polygons     94 zones + water
ET_GEOG_ADM3_2021_uscb_202308          756 polygons    738 counted woredas + 18 others
ET_RELIGION_2007census_uscb_202308     851 rows        1 + 13 + 94 + 743
ET_AGE_SEX / ECONOMY / DISABILITY / HOUSEHOLD / HOUSING / ETHNICITY / LANGUAGE /
HEALTH_2007census, AGRICULTURE_survey2021
```

The same tables are also served as a 1.8 MB `.xlsx` with the names spelled out and an
explicit `ADM_LEVEL` column, which is what `et.py` reads; the geodatabase layer is read as a
cross-check and agrees on all 5,106 cells. **That is a read check and not an independent
one** — same publisher, same release — and `et.py` says so where it prints.

The religion layer is 18 count columns: six categories × `_B` both sexes, `_F` female, `_M`
male. Only `_B` is drawn. The other twelve are read anyway, because `M + F == B` is 5,106
free equalities and it holds exactly.

## 3. `-999` is the sentinel and it parses as a number

Four woredas carry `-999` in **every** category instead of a value:

```
ETH_02_01_01  ADEAR WEREDA    Āfar     "Population data not available"
ETH_02_02_03  BEDU WEREDA     Āfar     "Population data not available"
ETH_02_02_09  NO NAME         Āfar     "Population data not available"
ETH_08_03_02  BELTU WEREDA    Oromīya  (no comment)
```

24 cells. Summed naively they take **23,976 people off the national total — 0.0325%**. That
is `spec` §5a's rule in a new disguise and a nastier one than most: the defect is in-band,
arithmetic-safe, and the resulting error is *small enough to be mistaken for rounding or for
a genuinely incomplete census*. Nothing raises, nothing looks wrong, and the woreda tier
quietly fails to reconcile by an amount nobody would chase.

`mask(< 0)` before summing, and the discrepancy vanishes completely — which is the thing that
proves the sentinel was the whole of it. `et.py` asserts the sentinel **count** (24) rather
than merely masking, so a future release that changes the convention fails loudly instead of
silently drawing fewer people.

A fifth woreda, `ETH_08_21_01 FINFINNE ZURIA SPECIAL ZONE`, is all-null and has no polygon
either; USCB says so in its own comment (`No polygon included in FGDB`).

## 4. The partition is exact, in both directions

After masking, at every level:

| level | units with data | sums to national? |
|---|---|---|
| region (ADM1) | 13 | **yes, all 6 categories** |
| zone (ADM2) | 93 of 94 | **yes, all 6 categories** |
| woreda (ADM3) | 738 of 743 | **yes, all 6 categories** |

and the national figure is **73,750,932**, the published 2007 census population — the one
number in this whole ingest that comes from outside the USCB file, and therefore the only
genuinely independent anchor the reconciliation has.

**The five null woredas cost nothing, and the reason is the interesting part.** The 738
woredas with data sum to the national total *exactly*, so the national total does not include
the null ones either. That is the tell that these areas were never counted by the 2007 census
rather than lost by USCB — parts of Āfar and Somali were not fully enumerated, which is a
known limitation of that census and is why three of the five carry USCB's *"Population data
not available"*.

## 5. There is no non-response category at all

Every one of the six cells is a religion. There is no *not stated*, no *don't know*, no
refusal column — and the six sum to the census population exactly, so there is no room for
one either.

This is unusual here. `spec` §3.5 normally has something to say about every country, and the
range across this map runs from Serbia's two separate non-response cells to Hungary's 40.1%
*did not wish to answer*. Ethiopia has none.

**It does not mean nobody refused.** It means the 2007 tabulation distributed non-response
into the categories, or never published it, and the published table cannot be taken apart to
find out which. That is the Guyana situation (§9r) with the footnote missing: there, the
Bureau of Statistics said in print that it had prorated non-response in; here nothing says
anything. Recorded rather than corrected (§14.4), and `note_public` says the map's 100%
coverage is a fact about the tabulation and not about Ethiopia.

## 6. The counts are 2007 and the boundaries are 2021

The layer names say it outright and it is easy to read past. **Sidama is a separate ADM1 here
and was part of SNNPR in 2007** (it separated in 2020), so the re-cutting is real and reaches
the top level.

At woreda level:

- **418 of 743** carry a `USCBCMNT` reading `Formed from part of <census-era unit>`
- **70 census-era woredas are split across two to four modern ones** — `Adolana (no longer
  exists)` into four, `Bench (no longer exists)` into four, `Limo` into four, and so on
- **every count is an integer**, so nothing was apportioned by area into fractions
- the partition is exact, so nothing is duplicated or dropped

What that establishes is that the arithmetic is sound. What it does **not** establish is that
any individual split is right, and this project does not check it: the per-unit match is
USCB's work. So `et.py` carries the comment verbatim into the `note` column of every row —
`uscb=Formed from part of Adami Tulu Jido Kombolcha` — because it is the only record of how a
2021 woreda relates to the census-era unit its figures came from, and it exists per unit and
nowhere else. `sources/ph_geo.md` learned the same habit with `USCBCMNT` for the BARMM.

**The census is 2007 and there is no successor.** The 2017 census was postponed four times
and then abandoned; Ethiopia has not counted itself in eighteen years and has roughly doubled
since. This is the shape of Ethiopian religion, not its current size.

## 7. The categories, national

| category | count | share | node |
|---|---|---|---|
| Orthodox | 32,092,182 | 43.51% | `christianity.oriental.ethiopian` |
| Islam | 25,037,646 | 33.95% | `islam` |
| Protestant | 13,661,588 | 18.52% | `christianity.protestant` |
| Traditional | 1,956,647 | 2.65% | `indigenous.african` |
| Catholic | 532,187 | 0.72% | `christianity.catholic` |
| Other | 470,682 | 0.64% | `other.et` |

`taxonomy/et2007.py` argues each one. Two are worth flagging here.

**`Orthodox` → `christianity.oriental.ethiopian`, not `christianity.orthodox`.** The
Ethiopian Orthodox Tewahedo Church is *Oriental* Orthodox — non-Chalcedonian, out of
communion with Constantinople since 451 — and it is by a wide margin the largest church in
that communion anywhere. `ke2019.py` deliberately sent Kenya's bare `Orthodox` to the parent
node because Kenya's cell genuinely mixes both communions; **Ethiopia is the opposite case and
takes the opposite call**, because there is no ambiguity to preserve. The leaf already
existed: `usrc2020.py` created it for an ASARB count of about 66,000 US diaspora. Ethiopia
arrives on the same node with four hundred times as many people.

**`Protestant` means P'ent'ay and is wider than Kenya's identically-spelled cell.** In
Ethiopia it is the evangelical and Pentecostal churches together — Mekane Yesus, Kale Heywet,
Mulu Wongel. In Kenya, `Protestant` *excludes* the evangelicals, who have a cell of their own.
Same word, different sets, which is exactly why `source_category` is kept verbatim (§2.4).

## 8. What the map shows

- **The escarpment is a religious boundary and it is nearly total.** Tigray 95.6% Orthodox,
  Amhara 82.5%; Somali 98.4% Muslim, Āfar 95.3%. **104 of 738 woredas are over 99% one
  religion** — 50 Orthodox, 54 Muslim. Nothing else on this map is that binary over that
  much ground.
- **The Protestant south.** Sidama 84.4%, Gambella 70.1%, old SNNPR 48.4%, against 18.5%
  nationally. Bensa woreda 92.8%, Chere 96.6%.
- **Oromia is the pivot** — 27.0 million people and no majority: 47.6% Muslim, 30.4%
  Orthodox, 17.7% Protestant. The most mixed woredas in the country are here.
- **Traditional religion is South Omo and the Borana and nowhere else.** Surima 96.3%, Hamer
  91.3%, Dasenech 81.9%, Bena Tsemay 74.5%, Dire 75.6%. Half of everyone counted Traditional
  lives in 25 of the 738 woredas.
- **Catholics have one homeland**: Erob in Tigray, 40.6%, with nothing else close; then a
  Wolayta cluster (Damot Pulasa 17.1%).
- **Addis Ababa has an internal gradient** — 74.7% Orthodox overall, but Addis Ketema is
  30.6% Muslim and Kolfe Keraniyo 27.7% against Yeka's 6.8%.

## 9. What the map cannot show

- **Six categories is shallow.** No division of Islam (Ethiopia's Muslims are overwhelmingly
  Sunni Shafi'i with a large Sufi presence, and the Harari and Argobba are distinct
  communities); no division of Protestantism, which folds Lutheran-rooted Mekane Yesus
  together with Pentecostal churches; no separate cell for the Beta Israel remnant, the
  Jehovah's Witnesses or the Bahá'ís, all of which are inside `Other`.
- **`Traditional` is a floor, by an unknown amount.** The box is exclusive of the Christian
  and Muslim ones and Ethiopian traditional practice — Waaqeffanna above all — commonly
  accompanies one of them. `sources.md` §11b's continental rule.
- **And there is direct evidence of that in the residual.** `Other` is 0.64% nationally but
  **21.8% in Bore woreda** and 19.1% in Girja, both in Guji, Oromia — a concentration a
  six-cell question cannot otherwise explain, and most likely Waaqeffanna landing in `Other`
  rather than in `Traditional`. Not corrected: the census does not say, and moving 45,905
  people between two cells on an inference is exactly what §14.4 forbids.

## 10. Not done

- **The 1994 census** is the same shape and would give a change map over thirteen years,
  which almost nothing on this map has. Whether USCB published it is unchecked.
- **The zone tier (94 units) is not used** and does not need to be: the woreda tier is finer
  and reconciles exactly.
- **Ethnicity and language layers are in the same file** and are not read. `spec` §14.5
  permits a derived layer only where there is no religion count; Ethiopia has one, so there
  is nothing to derive and no reason to touch them.
