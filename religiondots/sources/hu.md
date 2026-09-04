# Hungary — KSH, Népszámlálás 2022

Wired 2026-09-04. 9,603,634 people, 3,177 settlements, 28 categories after allocation.

| | |
|---|---|
| source | Központi Statisztikai Hivatal, 2022 census database, tables **WBS003** and **WBS008** |
| basis | `self_id` — the census asked, and answering was **voluntary** |
| geography | 3,177 settlements (Budapest arrives as its 23 kerület) and 20 vármegye |
| categories | 11 at settlement, 29 at vármegye, 28 drawn |
| drawn | **5,746,804 people, 59.8% of the country** — the rest did not answer |
| licence | KSH data are freely reusable with attribution |

---

## 1. The headline number is the non-response

**40.1% of Hungary — 3,852,533 people — did not answer the religion question.** That is the
largest non-response of any source in this project, and it is not stable: 2001 was 10.8%,
2011 was 27.2%, 2022 is 40.1%. The question is voluntary and has been since it returned in
2001, and the share declining it has risen at every census since.

This matters more than it would elsewhere, because the naive reading of the map — "Hungary
is emptying of religion" — is not what the data says. Irreligion is its **own** category
(`Nem vallásos`, 1,549,610, 16.1%) and it *fell* slightly from 2011's 1,806,409 while
non-response rose by 1.15 million. The people who left the named churches between 2011 and
2022 overwhelmingly moved into "no answer", not into "no religion". Nothing here says which
they are, and §3.5 says the map marks the gap rather than filling it.

`countries.py`'s `note_public` leads with this. It has to.

## 2. Where the files came from, and what is and is not fetchable

`data/raw/hu/` holds five files. `data/` is gitignored, so this is the record.

| file | how |
|---|---|
| `WBS003_settlement.csv` | **by hand**, from the census database UI |
| `WBS003_settlement_with_total.csv` | the same export with the total in place of `RE_NA`; a cross-check |
| `WBS008_county.csv` | **by hand**, same |
| `hu_structure_WBS003.json` | `sources/hu.py --fetch` |
| `hu_structure_WBS008.json` | `sources/hu.py --fetch` |
| `nsz2022-1.1.7-eng.xlsx` | the national religion table, used once to pin the codes |
| `nepszamlalas_vallas.pdf` | KSH's 2024 spatial monograph on census religion; context, not data |
| `geoBoundaries-HUN-ADM2.geojson` | `sources/hu_geo.py --fetch` |

**§11a recorded that the census database is "a JavaScript app with no static endpoint
found". That was wrong, and the way it was wrong generalises.** The app is
`/adatbazis/app.js`, 1.68 MB, and its four API routes are plain string templates inside it:

```
GET /api/version                        -> {"version": "V67"}
GET /api/index/{version}/{lang}         -> the catalogue, 45 dataflows
GET /api/structure/{dataflow}/{version} -> SDMX structure: dimensions AND codelists
GET /api/dataflows/{dataflow}/{version}[/s/{dims}|/d/{filter}]   -> the data
```

It is an SDMX backend (Spring Boot). `/s/DIM1,DIM2` means "every code of these dimensions";
`/d/DIM:code+code` names a subset; `/s/DIM:<bitmask>` is the subset form used when the
explicit list would exceed 1,400 characters.

**The structures are fetched and the data is not, for one measurable reason.** WBS008's full
cube is 195 KB and comes back cleanly. WBS003's is not 195 KB, because `/s/TEL_SZ_ADAT`
selects all **149** of that dimension's variables — age, marital status, education, the lot
— and not just the eleven religion ones. The response passes 60 MB and the server truncates
it mid-body on every attempt (three tries, `IncompleteRead` then two invalid JSON tails at
52.8 MB and 60.7 MB). A `/s/TEL_SZ_ADAT:<bitmask>,TERUL_GEO5` request selecting only the
eleven would be roughly 4 MB and would almost certainly work; implementing KSH's bitmask
encoder is the one piece of work that would make Hungary fetchable end to end, and it was
not worth it against two CSVs already on disk and reconciling exactly.

**Two things that looked like walls and were not.** §11a recorded a hard 403 to scripted
clients; a browser User-Agent gets 200 and always did by the time this was written. And
`urllib.request` cannot do TLS to `ksh.hu` on this machine at all — `CERTIFICATE_VERIFY_FAILED,
self signed certificate in certificate chain` — while `curl` and `requests` both verify it
fine. That is a **local** trust store, not a server misconfiguration, and it is emphatically
not the `stat.gov.pl` case: copying pl.py's `verify=False` here would have disabled
verification to work around a problem that does not exist. `sources/hu.py` uses `requests`
with verification on.

## 3. The codes are not labels, and the obvious readings are wrong

Both exports carry **category codes only**. `§11a` flagged the WBS008 half as the open
question and guessed that "the order is guessable". It is not, and this is the part of
Hungary most likely to have ended up quietly wrong on the map:

| code | looks like | actually is |
|---|---|---|
| `RE_CA` | Catholic | **Calvinist** (`Református`) — Catholic is `RE_C` |
| `RE_CO` | Coptic? | **Other Christian** (`Más keresztény felekezet`), 54,981 |
| `RE_OU` | — | **Ukrainian Orthodox** — a jurisdiction absent from KSH's own prose list of the five Orthodox churches in Hungary (`nepszamlalas_vallas.pdf` p. 73 names Serbian, Constantinople, Bulgarian, Romanian and Russian) |
| `RE_CD` | — | **Other Christian denomination**, 141,197 |
| `RE_OCD` | same as `RE_CD`? | **the non-Christian remainder**, 29,977 — a different bucket entirely |
| `RE_NA` / `RE_NOT` | — | `RE_NA` is *did not answer*, `RE_NOT` is *no religion*. Swapping them would swap Hungary's two largest non-church numbers. |

**Every code is pinned by arithmetic before a row is written.** `sources/hu.py` reads the
labels from KSH's own SDMX codelists — never transcribed — and then `check()` re-derives
each one against `nsz2022-1.1.7-eng.xlsx`'s published 2022 national figures. All eleven
match exactly. A renamed code, a reordered codelist or a swapped pair fails the run rather
than relabelling the map.

The two buckets were separated the same way, before the labels were in hand:

```
RE_B + RE_M + RE_H + RE_OCD2 = 11,042 + 7,983 + 3,307 + 7,645 = 29,977 = RE_OCD  exactly
the other nine WBS008 Christian bodies                        = 141,197 = RE_CD  exactly
the seven Orthodox jurisdictions                              =  15,578 = RE_OC  exactly
```

Those three identities are what `taxonomy/hierarchy/hu.csv` encodes, and `check()` asserts
all three on every run. Unlike Ireland's and Mexico's hierarchy files, Hungary's needed no
judgement at all — the arithmetic forces it.

## 4. The Catholic remainder has to be derived, and it has to exist twice

KSH publishes `Katolikus` (2,886,619) and, labelled explicitly as subsets, `Katolikuson
belül római katolikus` (2,643,855) and `Katolikuson belül görögkatolikus` (165,135). It
never publishes the difference — **77,629 people who answered Catholic and named no rite.**

Drawing the parent *and* the children double-counts 2.8 million people. Drawing only the
children drops 77,629. So `sources/hu.py` emits `Catholic, rite not stated` as a derived
category, per unit, and `taxonomy/hu2022.py` excludes the parent. This is §12's
publication-floor rule in a form it had not taken before: the remainder is not below a
threshold, it is simply never printed.

Two consequences worth knowing:

- **At settlement level the derived residual is 78,544, not 77,629.** Where a unit's Greek
  Catholic count is suppressed for disclosure (657 settlements) the residual absorbs those
  people. The inflation is 915 people — 1.2% of this category, 0.01% of the country — and
  it goes in the only direction that does not invent a rite for somebody.
- **It must be derived at the vármegye level too, and that is not cosmetic.** `allocate.py`
  carries a fine column forward only when some coarse category lands on it. Emitting the
  residual at settlement alone would have silently dropped all 78,544 people at the
  allocation step, and the run would still have reported success on every other column.

## 5. Why it was worth two tables

The §3.9 trade is unusually kind here. WBS003 gives fine geography and shallow categories;
WBS008 gives 29 categories on 20 units; and they reconcile **exactly**, which India's did
not. So `allocate.py --within 5` splits three of the eleven settlement columns and leaves
the other eight untouched:

```
11 categories at settlement -> 28,  100.0% of the population allocated,  in == out
184,147 people (1.9%) derived;  160 (vármegye, column) pairs exact rather than allocated
```

**98.1% of the Hungarian map is measured at the settlement itself**, including every Roman
Catholic, Calvinist, Lutheran and Jewish dot. Only the Orthodox jurisdictions, the nine
other Christian bodies and the three non-Christian religions are placed from vármegye
structure — and `--within` matters for exactly the reason India's did: Hungary's minorities
are regional, not national. Romanian Orthodox along the Romanian border, Serbian Orthodox
around Szentendre and Lórév, Greek Catholics overwhelmingly in Szabolcs-Szatmár-Bereg. A
pooled national composition would have smeared each of them evenly over the country.

## 6. What the map shows

- **The Calvinist east.** 943,982 people, concentrated beyond the Tisza where the
  Reformation held and the Counter-Reformation did not reach. Measured off the drawn
  polygons: **10.2% of answers west of the Danube, 33.5% east of the Tisza**, against 54.8%
  and 19.5% for Roman Catholics — the cleanest east/west signal on the map. (No European
  ranking is claimed here; Switzerland's Reformed population is larger and the Netherlands'
  is too, so "largest outside X" formulations are best avoided.) Mapped to
  `christianity.reformed`, the parent —
  matching `ro2021.py`'s call for the same church across the Romanian border, which the two
  countries must agree on or the Partium reads as a different religion on either side of a
  line it is not divided by.
- **The Greek Catholic north-east.** 165,135 people, almost all in one county. This is the
  sharpest regional signal Hungary has and it is the reason the rite split was worth
  deriving a residual for.
- **Budapest as 23 districts**, not one shape — see `sources/hu_geo.md`. The capital is
  17.9% of the country and 34.6% of its answers are "no religion" against 25.5% elsewhere,
  but it holds only **20.9%** of the country's irreligion, so "where the irreligion is" is
  the wrong way to say it. The districts differ more than the claim would: 26.8% in the
  Castle district, 40.2% in Csepel. The highest shares in the country are not in Budapest
  at all but in the eastern Great Plain market towns — Szeghalom and Túrkeve are both 74%.
- **A very small non-Christian population.** Muslims 7,983 and Buddhists 11,042 in a country
  of 9.6 million: Buddhists outnumber Muslims, which is true of almost nowhere else on this
  map, and both are an order of magnitude below the Western European countries drawn here.

## 7. What is not done

- The 2001 and 2011 columns exist in the same database tables and are not read; spec §13
  rules out a time slider.
- WBS008 also carries a settlement-**type** dimension (`CL_TERUL_TELTIP2`, capital / county
  town / town / large village / village) which is a genuinely different cut — religion by
  settlement size, nationally — and is not ingested.
- The bitmask data fetch of §2, which would remove the two hand-exported CSVs.
