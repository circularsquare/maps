# Cayman Islands — 2021 census, via ESO's own report

`sources/ky.py` -> `data/normalized/ky.csv`. Boundaries: `sources/ky_geo.py`, placement:
`sources/ky_grid.py`. Taxonomy: `taxonomy/ky2021.py`.

**68,811 people on 6 districts, 17 named categories, 98.58% of that universe drawn.** The
most evenly religious country on this map — the largest answer reaches 19.5% — and the one
where nearly every category's geography is really a map of where people were born.

| | |
|---|---|
| counting geography | **district, 6 units** — 11,500 people each |
| placement | Kontur H3 r8 hexes, 392 of them, snapped to the coastline |
| basis | self-identification, census |
| tier | `measured` throughout |
| vintage | census 2021 |

---

## 1. Why it is here, and where it was found

§11t listed the Cayman Islands as reporting religion in 1999, 2010 and **2021** and left it
open. §11v found the report on the CARICOM mirror; ESO serves a byte-identical copy itself,
and that is what `ky.py` fetches. Nothing was hidden here — unlike the Bahamas, the report is
linked from `eso.ky` — but the page lists it as one 376-page PDF with no table of contents
worth searching, and the religion tables are `4.9A` and `4.10A`–`4.10F` on pages 125–132.

## 2. THE UNIVERSE IS SMALLER THAN THE CENSUS, IN TWO STEPS

ESO publishes four population figures for 2021 and every table uses the smallest:

```
  71,432   total population counted
    -327   institutional population (prisons, dorms, retirement homes)
  71,105   non-institutional — what ESO itself calls "the total population"
  -2,294   the census NON-RESPONSE ESTIMATE, from household refusals and verified
           no-contacts, weighted by the district tabular population distribution
  68,811   the census survey tabular population count — every table, religion included
```

So **3.67% of the territory is outside this map before the `DK/NS` cell is even reached**,
and the two reasons are different: 327 people are real and counted but excluded from
tabulation, and 2,294 are an estimate that exists only as a national number and therefore
cannot be put anywhere on a map at all. Nothing is scaled up (§14.4).

> This is the cleanest statement of a distinction the map keeps meeting: **a published
> national total is not a drawable one.** Trinidad's institutional population (§9ak) is the
> same shape; Cayman is unusual in publishing the ladder explicitly.

## 3. THE BOUNDARIES ARE WRONG, AND CHOOSING BETWEEN TWO WRONG SETS TOOK TWO TESTS

COD-AB's ADM1 is ESO's six districts — same names, `Sister Islands` included for Cayman Brac
and Little Cayman, 6/6 both ways with the pcodes cross-checked from the boundary side. **And
its geometry is not the districts.** COD's Bodden Town is **8.3 km²**, a strip about two
kilometres deep along the south coast, while its North Side is 88.9 km² and reaches down
across the island.

OSM carries the same six districts at `admin_level=8` under the same six names, so this was a
choice rather than a complaint. Both were tested.

**Test 1 — settlements.** Every OSM `place` node in the country (69) located in both sets.
**60 agree, 9 do not**, and neither set wins:

| settlement | COD-AB says | OSM says | which is right |
|---|---|---|---|
| Breakers, Northward, Frank Sound, Midland Acres, Pease Bay, Belford Estates | North Side | Bodden Town | **OSM** |
| Savannah, Saint James Pedro Castle | Bodden Town | George Town | **COD-AB** |
| North Sound Estates | North Side | George Town | — |

**Test 2 — population.** Kontur's grid summed per polygon against ESO's own district counts,
which weights each disagreement by how many people it moves:

| district | COD-AB | error | OSM | error |
|---|---|---|---|---|
| Bodden Town | 11,823 | −2,575 | 8,103 | −6,295 |
| East End | 850 | −908 | 776 | −982 |
| George Town | 36,211 | +2,313 | 44,064 | +10,166 |
| North Side | 3,506 | +1,649 | 495 | −1,362 |
| Sister Islands | 1,736 | −379 | 1,566 | −549 |
| West Bay | 10,823 | −3,961 | 8,390 | −6,394 |
| **total \|error\|** | | **11,785** | | **25,748** |

**COD-AB is less wrong by better than two to one, so COD is used.** Its mistakes are on six
villages; OSM's are on Savannah, which is a town.

> **When two boundary sets disagree, the settlement test says WHICH is wrong and the
> population test says HOW MUCH.** Neither decides alone: here the first came out 6–3 against
> COD and the second 2:1 for it.

What that costs is not hidden. Bodden Town's eastern villages are inside COD's North Side
polygon, so a few hundred people are drawn in the wrong district and North Side's 1,857 dots
spread over more ground than the district really covers — which is exactly what the 2.03x in
§7's ratio table is. `ky_geo.py` re-runs the settlement test on every build and asserts the
failures are **exactly** those three, so a COD reissue that fixes one, or breaks another,
stops the run instead of passing quietly.

## 4. Two irregularities in the tables, neither announced

**North Side's table has no `Muslim` row.** Not a zero, not a dash — the row is absent, and
every other district prints eighteen rows where North Side prints seventeen. **An omitted row
is not a zero**: ESO prints a dash for a genuine zero elsewhere in the same table (North
Side's Judaism cell is one).

The national table prices it exactly. The six districts sum 3 short of Table 4.9A on `Muslim`
and North Side is the only district omitting it, so **the missing cell holds 3 people**. It is
**not added back** (§14.4) — the value is implied by a residual rather than published — and the
map draws North Side with no Muslims.

**Table 4.10F (Sister Islands) has eleven figures per row where the other five have twelve**,
dropping the Non-Caymanian `DK/NS` column. So the parse can assume neither a fixed row list
(the Bahamas' problem, §9ar) nor a fixed column count. It reads a row as *label tokens, then a
run of figures*, takes the first figure, and asserts the run length is constant **within** a
table — which tolerates both of the above and still catches a genuinely mis-columned page.

## 5. ESO's tables do not internally reconcile, by one to five people

| district | rows sum to | printed Total | |
|---|---|---|---|
| George Town | 33,897 | 33,898 | −1 |
| West Bay | 14,784 | 14,784 | 0 |
| Bodden Town | 14,397 | 14,398 | −1 |
| **North Side** | **1,852** | **1,857** | **−5** |
| East End | 1,759 | 1,758 | +1 |
| Sister Islands | 2,114 | 2,115 | −1 |

**The national table IS internally exact**, so the discrepancy belongs to the district tables
rather than to the category list. Three of North Side's five are the omitted Muslim row; the
rest is two-sided and bounded at one person, which is what independently rounded figures look
like and is nothing like a parse error. ESO offers no note. The whole spread prints on every
run rather than disappearing into a tolerance.

## 6. What the country shows

* **The largest answer is 19.5%.** Church of God, then no religion (16.7%), Roman Catholic
  (13.6%), Seventh-day Adventist (8.7%), non-denominational (8.3%). Five different things and
  none dominant — no other country drawn here is this flat.
* **The Church of God is the national church and the flattest category**: 27.2% North Side,
  25.3% Bodden Town, 23.5% Sister Islands, 20.9% East End, 18.3% West Bay, 16.8% George Town.
  It is the **Anderson, Indiana** body — Holiness, explicitly not Pentecostal — which reached
  the territory through the Cayman Islands Regional Mission Council. See `ky2021.py`; this is
  the biggest single mapping call in the file and the name alone could not decide it.
* **Everything that varies, varies with foreign birth.** Over half the residents were born
  abroad. Roman Catholicism is 18.4% in George Town against 3.6% in North Side — the Filipino
  and Latin American workforce in the capital — and the Hindu share peaks at 6.6% in East End.
* **Cayman Brac is a different country.** The Sister Islands are **30.5% Baptist** against
  2.8–9.5% in every Grand Cayman district, and **0.66% Presbyterian/United** against 13.8% in
  North Side. The old Brac Baptist settlement, still visible through a remade population.
* **The United Church is 5.7%** — the United Church in Jamaica and the Cayman Islands, the
  same body `jm2011.py` maps from the other side of the same 1965 union.
* **16.7% report no religion**, and it is highest in West Bay (20.3%) and East End (19.5%)
  rather than in the capital.

## 7. Placement: the thinnest grid on the map

**392 Kontur hexes for the whole country** — the smallest extract used here. It still works,
because the units are 8–89 km² and a hex is 0.67 km², so there are 16 to 119 hexes per
district and the grid is **finer than the tier it weights**. That is the test Saint Vincent
failed (§9ac, where the grid was coarser than the enumeration districts and was dropped).

**6.29% of it lands outside every district and none of it is genuinely offshore** — the
measured maximum distance to a district is 500 m:

```
  <100 m       27 hexes   1,404 people
  100-250 m    43 hexes   2,685 people
  250-500 m    27 hexes     272 people
  beyond        0 hexes       0 people
```

COD's coastline against a 400 m hex, and Caymanian settlement *is* the coast, so it is snapped
to the nearest district within 1 km exactly as in the Bahamas (§9ar).

**The per-district ratio is wide and it is the boundaries, not Kontur** — North Side 2.03x,
Bodden Town 0.86x, West Bay 0.78x, against a national 1.007x. §3's boundary error is the whole
explanation. Only the within-district shape is used, so no district gets the wrong *number* of
dots (§9t).

## 8. What is in the report and not used

Male/female and Caymanian/non-Caymanian splits on every religion cell — the status split is
the more interesting one and would support a born-here / born-abroad comparison this map has
no way to draw. Then age, education, marital and union status, fertility, employment, housing,
mortality, emigration, crime, agriculture and food security, all by district.
