# Switzerland — `sources/ch.py`, `ch_rescale.py`, `sources/ch_geo.py`, `taxonomy/ch2000.py`

Drawn 2026-09-06. **2,196 communes, 17 nodes, 7,438,908 people, 99.1% of the survey's
universe.**

| | |
|---|---|
| counting geography | **commune, 2,196 units** — 3,400 people each, the fourth finest on this map |
| placement | Kontur H3 r8 hexes, 40,641 of them, clipped to the commune |
| basis | self-identification; **current magnitudes on 2000 composition** |
| tier | **`derived` throughout** — see §3 |
| vintage | structure 2000, magnitude 2024 |

---

## 1. The crawl §11c priced does not work, and the catalogue answers instead

`sources.md` §11c looked at BFS PxWeb, found **650 databases with opaque ids** (`px-x-0103010000_101`)
and the human title only inside each per-database listing, and concluded that finding the
religion table meant enumerating all 650 — *"an hour of wall clock and no risk"*.

**There is a risk and it is that the crawl returns nothing.** Run politely at one request per
second with backoff, **55 of 55 requests failed**. BFS rate-limits its own API hard enough
that a full enumeration is not available at any speed a script will tolerate.

    ckan.opendata.swiss/api/3/action/package_search?q=Religion

answers in one call, with the px id in the resource URLs. opendata.swiss is the national
portal and it *mirrors BFS's own catalogue*, so the thing §11c wanted to enumerate is already
indexed somewhere searchable.

> **When an office rate-limits its own API, ask the national open-data portal that mirrors
> it.** This extends §11k's catalogue lesson rather than repeating it: there the trap was a
> catalogue endpoint with a wrong mode, here it is a catalogue that cannot be walked at all.

The same call returned **Liechtenstein** — `etab.llv.li`, an open PxWeb with religion by
Gemeinde — because opendata.swiss carries the Liechtenstein statistics office alongside the
Swiss one. *A national portal may index a neighbour.*

## 2. The two instruments, and why neither is enough alone

| | census 2000 | Strukturerhebung |
|---|---|---|
| table | `px-x-4003000000_122` | `je-d-01.08.02.02` |
| geography | **2,896 communes** | **26 cantons** |
| categories | **19** | 8 |
| universe | everybody, 7,287,357 | 15+ in private households |
| vintage | 2000 | 2010–2024, annual |

**2000 is the last time Switzerland asked everybody.** Nothing since 2010 asks below the
canton, and the catalogue was checked for this specifically: the only geographic religion
tables BFS publishes are the censuses of 1970, 1990 and 2000.

**The 19 categories are why a source this old is worth using.** Seven Protestant cells —
Reformed, Methodist, *Neupietistisch-evangelikale Gemeinden*, Pentecostal, New Apostolic,
Jehovah's Witnesses, and a named Protestant residual — plus **Christ Catholic separately from
Roman Catholic**, and Orthodox, Jewish, Islamic, Buddhist and Hindu. No other European census
on this map can express half of it.

## 3. The rescale, and where it is a stronger claim than Brazil's

`ch_rescale.py` implements §3.4: current canton magnitudes on 2000 commune structure. That is
Brazil (§9-br) — 2022 totals on 2010's 56 denominations — **with one extra step, and the step
must not be glossed.**

Brazil's rescale changes the **categories** and keeps the geography, so a 2022 município total
is a measured number for that município and 103.6M Brazilians pass through untouched as
`measured`. **Switzerland's changes the geography too**: the measured quantity is a *canton*
total being spread over that canton's communes. **Every drawn row here is `derived`** and
§3.10 keeps all of it out of the presence ring.

### Two margins, not one scale factor

A plain Brazil-style rescale holds each commune's *share of its canton's* Reformed population
fixed since 2000. That is wrong in a visible way: Switzerland grew from 7.29M to about 9M
between 2000 and 2021, very unevenly — the Zurich and Geneva belts gained, alpine communes
lost — so a 2000 share puts dots in emptying valleys and starves the places people moved to.
On a dot map that is not a rounding error.

So it is an **iterative proportional fit** on two margins:

| | |
|---|---|
| rows | each commune's current population, GISCO's `POP_2021`, scaled within its canton to the survey's 15+ total |
| columns | each canton's eight Strukturerhebung category totals |
| seed | the 2000 census counts, aggregated to the same eight groups |

Both come out exact (worst residual 1e-9). **The 2000 census supplies only the association** —
which commune within a canton is the Catholic one — and contributes no magnitude at all. Each
fitted (commune, group) cell is then split back into the 19 census leaves by that commune's own
2000 within-group proportions, and that is the step that recovers Pentecostals and Old
Catholics from a survey that only knows "other Christian".

`POP_2021` is **in the boundary file already**, so the row margin cost no new download.

### What it cannot do, in three parts

1. **It cannot see a commune that changed religion.** If one village secularised faster than
   its canton, the fit does not know. The canton is the finest unit at which anything has been
   measured since 2000.
2. **The composition inside a group is 2000's, always.** Within *other Christian* the split
   between Pentecostals and Orthodox is a 2000 fact carried forward 24 years, and Switzerland's
   Orthodox population has grown much faster than its Pentecostal one — so the Orthodox cell is
   if anything understated and the free-church cells overstated.
3. **The universe changes.** The survey asks people **aged 15+ in private households**, so
   children, and everyone in a collective household, plus diplomats and international civil
   servants, are outside it. Not scaled back up (§14.4); `countries.py` declares it.

### The scale of what the rescale moves

| | 2000 | 2024 |
|---|---|---|
| Keine Zugehörigkeit | 11.11% | **36.82%** |
| Römisch-katholisch | 41.82% | 30.00% |
| Evangelisch-reformiert | 33.04% | 18.74% |
| Islamische Gemeinschaften | 4.27% | 6.03% |
| Christlich-orthodox | 1.81% | 2.46% |

**That is the fastest religious change on this map**, and it is the argument for doing the
rescale rather than drawing 2000: an unrescaled Switzerland would be wrong about its largest
single category by a factor of three.

## 4. The commune vintage, which was the real work

The census counts **2,896** communes; GISCO LAU 2021 carries **2,242**. A naive join on the
BFS number matches 2,076 and loses **820 communes and 606,086 people**.

BFS publishes the correspondence as an open keyless API:

    https://www.agvchapp.bfs.admin.ch/api/communes/correspondances
        ?includeUnmodified=true&includeTerritoryExchange=false
        &startPeriod=06-12-2000&endPeriod=01-01-2020

Three things about using it:

- **`startPeriod` is the census date, 6 December 2000.** A 1 January 2001 boundary loses that
  year's own mutations — 34 communes and 21,022 people, all Fribourg villages that merged into
  Villorsonnens.
- **`includeTerritoryExchange=false`.** With exchanges a commune maps to more than one
  successor and the join stops being a function. They are boundary adjustments of a few
  hectares.
- **GISCO's "LAU 2021" for Switzerland is the 1 January 2020 commune state.** Asking for 2021
  leaves 15 communes pointing at codes the shapefile lacks — Welschenrohr-Gänsbrunnen and
  Bois-d'Amont, both created 1.1.2021. Asking for 2020 leaves 5, the Verzasca valley communes
  merged in 2020. **Both are requested and whichever target exists wins**, resolving all 2,896.

> **A boundary file named for a year is not necessarily that year's state. Test it against the
> register rather than trusting the filename.**

**45 of the 2,242 LAU features are not communes at all**: the lake surfaces, which BFS numbers
in the 9xxx block and apportions to no municipality, and the Ticino and Graubünden
*comunanze*, common land held jointly by several communes. 44 are empty; `5399` holds 802
people. Dropped against BFS's own register rather than by a code range, so a renumbering fails
loudly. What is left is 2,197, of which 2,196 carry counts.

## 5. The PxWeb trap, which is silent

    {"query": [], "response": {"format": "json-stat2"}}

returns **HTTP 200, valid json-stat2, 1.6 KB, and no geography dimension at all** — just
`Wohnsitztyp` and `Religion`. An empty query does not mean "everything" on this server; it
means "no selection", and the geography is dropped without a word. `sources/ch.py` asserts the
dimension survived the round trip.

`Wohnsitztyp` also has two values that are different populations — *zivilrechtlicher* (where
you are registered) against *wirtschaftlicher* (where you actually live), which differ for
students and weekly commuters. The civil-law figure is used: it is the one the 7,287,357
national total refers to.

## 6. What it is worth

- **A five-hundred-year-old confessional map that has barely moved.** Uri and Appenzell
  Innerrhoden 68% Catholic, Valais 62%, against Bern 14% and Basel-Stadt 13%. At commune level
  Muotathal is 84% Catholic and Poschiavo 82%, against Sumiswald and Lützelflüh at 63%
  Reformed. Appenzell was partitioned into two half-cantons over religion in 1597 and the two
  halves still read that way.
- **Irreligion is urban and it went furthest in Protestant cantons** — Basel-Stadt 60%,
  Neuchâtel 57%, Geneva 50%, against Uri 22% and Appenzell Innerrhoden 18%.
- **Islam is industrial rather than metropolitan** — Böttstein 24%, Gerlafingen 23%,
  St. Margrethen 21%: small towns on the Aare and the Rhine, not the big cities.
- **The Hindus are Tamil and outnumber the Jews and Buddhists combined** — 46,000, from the Sri
  Lankan asylum migration, showing up in Solothurn and Emmental factory towns.
- **Möhlin and Magden in the Fricktal are 15% Christ Catholic**, which exists almost nowhere
  else on earth at that share.

## 7. Left undone

1. **The 1970 and 1990 censuses are on the same endpoint** (`px-x-4001000000_122`,
   `px-x-4002000000_122`) at the same geography. A three-census series at commune level would
   let the rescale's central assumption — that a commune's share of its canton's category is
   stable — be *measured* over 1970→2000 rather than assumed over 2000→2024. That is the single
   best check available to this country and nobody has run it.
2. **Bezirk-level religion is in the same table** (184 districts) and unused.
3. The Strukturerhebung publishes confidence intervals per canton per category; they are read
   past. A canton whose Jewish cell is `X` (four observations or fewer) contributes zero, and
   79 such cells exist across the 26 cantons.
