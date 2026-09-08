# Saint Vincent and the Grenadines — 2012 census, via the U.S. Census Bureau

`sources/vc.py` -> `data/normalized/vc.csv`. Boundaries: `sources/vc_geo.md`.
Taxonomy: `taxonomy/vc2012.py`.

**109,188 people on 221 enumeration districts, 18 religion categories, 95.3% drawn.** The
finest geography in the project per head, the smallest country on the map, and the source that
makes presence rings earn their place.

---

## 1. Why it is here

§11j, on the four USCB leftovers:

> **Saint Vincent is the finest geography in the project and one of the smallest countries.**
> 221 enumeration districts over 109,188 people is **~494 people per unit** … **Build it as a
> companion to Jamaica or not at all**; alone it is a rounding error, and beside Jamaica it is
> the second half of a Caribbean Protestant-denominational picture that exists nowhere else on
> the map.

Jamaica was built on 2026-09-06 (§9ab). This is the companion.

## 2. The route

The USCB series on HDX (§11h) — the SVG Statistical Office's own REDATAM tabulation,
transcribed, with the boundaries in the same file:

| resource | bytes |
|---|---|
| `saint_vincent_and_the_grenadines.gdb.zip` | 1,400,042 |
| `saint_vincent_and_the_grenadines_uscb_202109.xlsx` | 384,613 |

Nine sheets; **`Ethnicity and Religion`** is the one read. 235 rows — 1 country, 13 census
divisions, 221 enumeration districts — and 38 columns, of which 8 are ethnicity and 18
religion.

The Metadata sheet cites the office's REDATAM server directly
(`redatam.org/binsvg/RpWebStats.exe/Frequency?BASE=SVG2012`).

## 3. THE RECONCILIATION IS CROSS-TABLE, AND IT IS BETTER THAN JAMAICA'S

**This file publishes no religion total.** There is no `RLG_RTOTL` the way Jamaica has one, so
the usual within-table check — categories against their own published total — is unavailable.

What it has instead is `ETH_TPOP`, the **ethnicity** universe, in the same sheet. And:

> the 18 religion cells sum to `ETH_TPOP` **exactly, on all 235 rows, at every level**.

That is two independently tabulated questions agreeing **to the person** on 221 enumeration
districts. It is a stronger check than a column summing to its own neighbour, because a
transcription slip in either question breaks it and there is no shared derivation to hide in.

It also settles the universe. Religion here is asked of **everybody** — not of a 15+ subset the
way Portugal's (§9v) and Chile's (§9k) are — so no share on this country needs a denominator
caveat.

The levels nest exactly too: the 13 census divisions and the 221 enumeration districts each sum
to the national row, category by category, with zero discrepancy.

**And there is no Jamaica-style omission.** The Metadata sheet was read first (§11h's rule, and
§9ab is why), and it carries nothing like STATIN's *"Baha'i, Hinduism, Islam and Judaism were
not included in parish tables"*. Every religion the census counted is in the enumeration-district
tables.

## 4. What the categories are

| category | people | share |
|---|---|---|
| Pentecostal | 30,108 | 27.57% |
| Anglican | 15,175 | 13.90% |
| Seventh Day Adventist | 12,710 | 11.64% |
| Baptist | 9,675 | 8.86% |
| Methodist | 9,458 | 8.66% |
| Without religion | 8,147 | 7.46% |
| Roman Catholic | 6,877 | 6.30% |
| Not stated | 5,095 | 4.67% |
| Other religion | 4,672 | 4.28% |
| Evangelical Christian | 4,119 | 3.77% |
| **Rastafarian** | 1,181 | 1.08% |
| Jehovah's Witness | 909 | 0.83% |
| Presbyterian | 294 | 0.27% |
| Salvation Army | 287 | 0.26% |
| Mormon | 207 | 0.19% |
| Muslim | 111 | 0.10% |
| Hindu | 89 | 0.08% |
| **Traditional** | 74 | 0.07% |

**Six categories are under 400 people, and that is the point of the country.** At 1:1,000 none
of them draws a dot. §4.3's presence rings put them on the map instead — **the scatter produces
exactly six rings at 1:1,000 and thirteen at 1:10,000** — and this is the source that argues
for having built that machinery. A census publishing single people on units of 415 is what
makes an 89-person religion visible at all.

## 5. `Traditional` is not the Kalinago, and that was tested

74 people in 25 of 221 districts. **The obvious reading is the indigenous population** — Saint
Vincent has the largest surviving Kalinago (Carib) community in the eastern Caribbean, at Sandy
Bay in the north-east, and the census's own ethnicity question counts 3,280 Indigenous people.

**The data rules it out.** Ethnicity and religion sit in the same sheet, so §12's Philippines
co-location technique applies directly. Across the 219 populated districts:

* Traditional's share correlates with the Indigenous share at **r = −0.03** — nothing;
* **every one of the eight most-indigenous districts has ZERO Traditional**, including the
  three Sandy Bay districts at 79%, 83% and 86% Indigenous;
* Traditional's largest cells are in Georgetown, the Northern Grenadines and Kingstown, which
  are 75–83% Black and 0–1% Indigenous.

**What it is instead cannot be established from this source.** The plausible readings are
African-derived practice and the **Spiritual Baptist / "Converted"** tradition, for which Saint
Vincent is the home and which was criminalised here from 1912 to 1965 — but the form's separate
`Baptist` cell may already absorb the Shakers, and 74 people across 25 districts carries no
signal either way.

So it goes into `other.vc`, with `afrodiasporic` recorded as the node that would be wanted if a
source ever resolved it (§2.4). Not asserted now (§14.4).

**The general point: a cell whose name suggests an indigenous population is worth testing
against an ethnicity column when the source ships one, and the test can come back negative.**

## 6. What the map shows, and the Jamaica comparison is most of it

Two Caribbean countries 160 km apart, drawn from the same publisher, with different questions:

* **Saint Vincent is 27.6% Pentecostal — the highest share of any country on this map.** The
  Anglican and Methodist inheritance (13.9% and 8.7%) is the British colonial church; the
  Pentecostal majority arrived in the twentieth century and overtook it.
* **Irreligion is 7.5% here against 21.4% in Jamaica.** That is the sharpest irreligion contrast
  between neighbours anywhere on this map, and neither number is a small sample.
* **Rastafari is 1.08% here and 1.08% in Jamaica** — two independently designed censuses
  arriving at the same share. But the *distribution* differs: it is present in **186 of 221
  districts** here, where Hindu is in 26 and Traditional in 25. It is the most evenly spread of
  the small religions rather than a pocket.
* **Almost all of the East Indian population is Christian.** 1,199 East Indians (1.10%,
  descendants of post-emancipation indenture) against 89 Hindus. Visible only because the two
  questions share a sheet.

## 7. Two administrative notes

* **The ADM1 tier is census divisions, not parishes.** The Metadata sheet warns: *"2012 census
  materials utilized a 13 census division ADM1 structure that differs from the 6 parish ADM1
  structure generally used in maps of Saint Vincent"*. A `PARISH` column exists on the ADM2
  layer for anyone who wants the six. Only ADM2 is drawn, so this is a note rather than a
  decision.
* **Two enumeration districts have no people**, returning zero in every column. They are kept
  as zero rows and simply draw nothing, so a vintage that populates them needs no change here.
  `check()` asserts 219 of 221 rather than tolerating whatever it finds.

## 8. What is in the file and not used

Eight ethnicity cells keyed identically (Black 71.2%, Mixed 23.0%, Indigenous 3.0%, East Indian
1.1%, White 0.8%, Portuguese 0.7%, Other), and six further tables — age-sex, education and
occupation, households, buildings, access to services, ICT. The ethnicity columns were used
once, for §5's test, and are not drawn.
