# Georgia — Geostat, 2014 General Population Census

Ingested 2026-09-06. `sources/ge.py`, `sources/ge_geo.py`, `taxonomy/ge2014.py`.
Drawn at **region**: 11 units, 10 nodes, 3,669,896 of 3,713,804 enumerated.

Summary: the coarsest geography on this map, and it earns its place on people per unit rather
than on unit count. The interesting sections are §1 (a live PxWeb nothing links to, behind
three separate departures from the PxWeb contract), §4 (a suppression convention that gives
its own bound away) and §5 (two territories that were not enumerated, handled two different
ways because the boundary files handle them two different ways).

---

## 1. The host is a default IIS page and the data is behind it

The scouting pass (§11k) found Geostat's religion table as a single 33 KB `.xls` linked from
the census page, at region level, and recorded that as the whole of what Georgia publishes.
It is not. There is a live PxWeb with the entire census in it, and nothing on geostat.ge
links to it.

What the obvious probes return:

| host | result |
|---|---|
| `census.geostat.ge` | **404**, and the Wayback CDX has **zero** captures — so §11f's grep-the-archived-bundle trick has nothing to work on |
| `api.geostat.ge` | **200 with a zero-byte body** |
| `pc-axis.geostat.ge/` | **200, the default IIS Windows Server splash page** |
| **`pc-axis.geostat.ge/PXWeb/api/v1/en/`** | **200, `[{"dbid":"Database","text":"Database"}]`** |

**A default IIS page is not a dead host. It is a host with nothing mounted at `/`.** That is
now the second shape of "looks empty, is not" in this project, next to §9h's JSON-404 test.

### Three departures from the PxWeb contract, on one server

Each one alone reads as "the table is not there":

1. **The root returns `dbid`, not `id`.** A catalogue walker written against any other PxWeb
   here descends into nothing and reports an empty server. Third instance, after Kosovo and
   Moldova (§11k); this is now a *rule* rather than a curiosity.
2. **An empty `{"query": []}` POST 404s.** Kosovo's PxWeb accepts it and returns every cell.
   This one requires every value of every dimension listed explicitly.
3. **`json-stat2` 404s; `json-stat` v1 works.** Different response envelope —
   `{"dataset": {"dimension": …, "value": […]}}` — and the dimension order lives in
   `dataset.dimension.id`.

The table: `Database/Population Census 2014/Demographic And Social Characteristics/22_Population by regions and religion.px`.
4 KB. No key, no login, no wall.

## 2. Eleven regions, twelve answers

`Regions × Urban/Rural × Religion` = 13 × 3 × 13. Only the `Total` urban/rural slice is drawn;
urban/rural is a stratum, not a geography, and drawing it would double-count.

The twelve answers are unusually well chosen for a form this short — **Armenian Apostolic,
Yazidi and Jewish all get cells of their own**, which is why 0.04% categories survive:

```
83.41%  Orthodox            0.52%  Catholic          0.23%  Yazidis
10.74%  Muslim              0.51%  None              0.07%  Protestant
 2.94%  Armenian apostolic  0.33%  Jehovah's Wit.    0.04%  Other
 0.92%  Not stated          0.26%  Refusal           0.04%  Judaism
```

**Refusal and Not stated are separate cells**, which most sources do not do — 0.26% against
0.92%. Both are excluded (§3.5); together 1.18%, small by the standard of any voluntary
religion question here.

## 3. The one category the map cannot show, and it matters

**`Muslim` is one cell for two unrelated populations.**

| region | Muslim | share | who |
|---|---|---|---|
| Kvemo Kartli | 182,216 | **43.0%** | Azerbaijanis, largely Shia, on the Azerbaijani border |
| Adjara | 132,852 | **39.8%** | **Georgian-speaking Sunnis**, converted under Ottoman rule |
| Kakheti | 38,683 | 12.1% | Azerbaijanis of the lowlands |

A Sunni/Shia and Georgian/Azeri split running the length of the country, in one box. The map
will show it as two separate places with the same colour. Splitting it from ethnicity would
be §14.5's derivation applied to a religiously mixed category, which is exactly the case
§14.5 forbids.

## 4. The suppression gives away its own bound

Withheld cells are marked `...` (38) and `..` (40) — two markers, same meaning. What makes
Georgia unusual is that the **legend is inside the dimension**: the `Regions` codelist has a
thirteenth value which is not a region at all but the string `… is less or equal to 10`.

So a withheld cell is **0-10 people, not unknown**, and `check()` can assert something
stronger than the usual "report the shortfall":

> for every category, (national − sum of the 11 regions) ≤ 10 × its number of withheld cells

It holds on all twelve. **7 withheld cells in the drawn slice and 22 people unaccounted for
in 3.7 million — 0.0006%.** The largest single gap is Judaism's 12 across 2 withheld cells.

Two things to carry forward: **a legend value can be sitting in a dimension pretending to be
a category** (dropping it silently is the easy mistake, so `ge.py` counts it and asserts there
is exactly one), and **a bounded suppression is worth far more than an unbounded one** — it
converts §3.8's "we cannot know" into an arithmetic guarantee.

## 5. Two territories were not enumerated, and they need two different treatments

The 2014 census covers territory under Georgian government control. Abkhazia and the
Tskhinvali region (South Ossetia) have **no rows at all** — a hole, not a wrong number, which
is the opposite of Kosovo's problem (§9w) and much easier to draw honestly.

The boundary files do not treat them alike, so neither does the build:

- **Abkhazia is its own geoBoundaries ADM1.** No census row, so it is dropped from the units
  layer and draws nothing. The join check asserts that Abkhazia is the *only* spare polygon,
  which is how a future boundary revision gets caught.
- **South Ossetia has no ADM1 of its own.** Its municipalities sit inside Shida Kartli
  (**Java**) and Mtskheta-Mtianeti (**Akhalgori**), so those two ADM2 polygons are subtracted
  from the placement grid. Without that, Shida Kartli's census dots would spread over ground
  the census did not count.

The correction removes **551 hexes and 6,093 modelled people** — small, because Kontur's
Georgian extract barely covers South Ossetia (Tskhinvali and Znauri are absent from
geoBoundaries ADM2 entirely). *A correction that turns out small is still worth making*: the
alternative is a map quietly claiming Tskhinvali was counted.

## 6. The city/ring pair, in a fourth post-Soviet country

§9q's Lithuanian check recurs. Kontur 2023 / census 2014 by region:

| | ratio |
|---|---|
| the ten regions outside the pair | median **0.87x**, range 0.85–1.19 |
| C. Tbilisi | 0.85x — inside the band, asserted with the rest |
| **Mtskheta-Mtianeti** | **1.66x** |

geoBoundaries' Tbilisi polygon is **249 km² against the city's ~500**, so outer Tbilisi sits
inside its ring region here. The city loses population to the ring and stays plausible; the
ring gains a whole city's suburbs and does not. Reported, not asserted — and it is a
*placement* fact rather than a count fact, because every region's dot total comes from the
census regardless.

The overall 0.87 median is Kontur 2023 against a 2014 census in a country whose population
has fallen, which is the expected direction.

## 7. `None` is a category name

§9m's trap, third country. Georgia's irreligious answer is the literal string `None`, and
`pandas.read_csv` turns it into `NaN` — so a default read **silently deletes 19,080 people**
and draws a country with no unaffiliated population at all. `countries.py` reads `ge.csv` with
`keep_default_na=False, na_values=[""]`. This was caught by a scratch script failing with
`KeyError: "['None'] not in index"`, which is the lucky version; the unlucky version is a
map that just looks slightly wrong.

## 8. What the source is worth

- **11 regions, 334,000 people each.** Coarse, and the right test is people per unit against
  the rest of the map: Russia is drawn at 1.8 million per federal subject. Nothing finer
  exists — the census database publishes municipalities for marital status and only regions
  for religion.
- **Samtskhe-Javakheti is a different country religiously**: 39.9% Armenian Apostolic
  (Akhalkalaki, Ninotsminda), **9.4% Catholic — 78% of all Georgia's Catholics**, the
  Armenian Catholics of Akhaltsikhe — and Orthodoxy a minority at 45.2%.
- **Two Muslim populations, 39.8% of Adjara and 43.0% of Kvemo Kartli**, in different corners
  of the country and for entirely different historical reasons.
- **95% of Georgia's Yazidis are in Tbilisi** — 8,124 of 8,591. This more than doubles the
  `yazidism` node's population on the map, which previously came from Australia.
- **0.5% report no religion**, the second-lowest on this map after Kosovo, in a country that
  spent seventy years in the Soviet Union. Adjara is the outlier at 2.8%.

## 9. What is left

The 2014 vintage is now eleven years old and Georgia has run no census since. `Armenian
apostolic` moved from `christianity.oriental` to `christianity.oriental.armenian` on
2026-09-08, when Armenia's 2,793,041 reopened the question as `ask/004-am` and Anita ruled
that a source is recorded as it answers. Australia and Estonia were re-pointed at the same
time, along with seven others; spec §2.7 and `taxonomy/ge2014.py` REVIEW have it.
