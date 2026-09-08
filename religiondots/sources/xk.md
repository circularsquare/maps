# Kosovo — ASK, Census 2024

Ingested 2026-09-06. `sources/xk.py`, `sources/xk_geo.py`, `taxonomy/xk2024.py`.
Drawn at **municipality**: 38 units, 5 nodes, 1,561,848 drawn of 1,585,566 enumerated.

Summary: technically the easiest ingest here after Portugal — an open PxWeb table, 10 KB, exact
reconciliation — and substantively the most compromised source on the map. **The counts are
clean and the country was not enumerated.** §4 is the section that matters.

---

## 1. The route

`askdata.rks-gov.net` is a plain PxWeb v1 endpoint, no key, no login, no Cloudflare.

```
GET  https://askdata.rks-gov.net/api/v1/en/ASKdata/Census population/
     1_Demographic_Characteristics/census2024_10.px
POST the same URL, {"query": [], "response": {"format": "json-stat2"}}
```

*Population by religion and sex at country and municipal level for the years 2011 and 2024* —
39 territories × 2 years × 3 sexes × 7 categories = 1,638 cells, 12 KB.

**The catalogue walk nearly missed it for a dull reason worth recording.** askdata's PxWeb root
returns **`dbid`** where every other PxWeb in this repo returns `id`, so a generic walker keyed
on `id` sees a list of nameless nodes and descends into none of them. Two offices scouted the
same day did this — Kosovo and Moldova — and both looked empty for the same wrong reason. A
PxWeb root that returns entries with a null `id` has not failed; it is using the other key.

The settlement tier exists in the same database (`6_Sipas vendbanimeve`, one folder per
municipality) and carries **ethnicity but not religion**, so municipality is the ceiling.

## 2. The blanks are true zeros, and the partition is the proof

162 of the 1,638 cells come back with `status: ":"` and no value — 21 of them in the drawn
slice. PxWeb's `:` is "not available", and on this map that usually means disclosure control:
Lithuania (§9q) suppresses 298 municipality cells and reading those as zero would delete people.

Here it cannot be suppression, and the argument is arithmetic rather than editorial. Reading
every blank as zero:

- the six categories sum to each unit's own published total, exactly;
- the 38 municipalities sum to the published national row, **category by category, gap zero**.

A suppressed positive value would break both. `check()` asserts it every run rather than
assuming it, so a vintage that starts genuinely withholding fails instead of quietly losing
people. **The same sentinel means opposite things in two neighbouring countries, and the way to
tell is to test the partition rather than to read the documentation.**

## 3. The boundaries — Kosovo is the hole in the GISCO file

GISCO LAU 2021 covers the EU27 *plus* the candidates: AL, BG, CH, IS, LI, MK, NO and RS are all
in a file this repo already had. **Kosovo is the one Balkan country it does not carry**, because
the EU has no agreed status for it. So §9e's "boundaries are free in Europe" has an exception
and this is it.

geoBoundaries **XKX ADM2** supplies the 38. The join is by name, and Albanian nouns have
definite and indefinite forms — ASK writes `Gjakovë`, `Klinë`, `Pejë`; geoBoundaries writes
`Gjakova`, `Klina`, `Peja`:

| stage | matched |
|---|---|
| fold accents, strip `Municipality of ` | 23 / 38 |
| + strip a trailing `a`/`e` (the definite/indefinite ending) | 34 / 38 |
| + five explicit aliases | **38 / 38** |

The five: `Drenas`→`Gllogoc` (alternative name), `Pristina`→`Prishtinë` (anglicisation),
`North Mitrovica`→`Mitrovicë e Veriut`, `Han i Elezit`→`Hani i Elezit` (one extra vowel),
`Zveçan`→`Zveqan` (ç against q, not a diacritic). Both sides' stems are asserted unique before
the merge, so a future rename collides loudly instead of joining two units silently.

Total area comes out at 10,898 km² against Kosovo's 10,887 km², which is the free sanity check.

## 4. THE NORTH BOYCOTTED THE CENSUS, AND THE PUBLISHED COMPOSITION IS INVERTED

Kosovo's Serb population largely refused enumeration in 2024. In the four northern
Serb-majority municipalities the result is not an undercount but a **different population**:
whoever answered was disproportionately not Serb.

| municipality | enumerated | Islam | Orthodox | % Orthodox |
|---|---|---|---|---|
| Leposaviq | 3,185 | 436 | 2,680 | 84% |
| **Zubin Potok** | **763** | **681** | 82 | **11%** |
| **Zveqan** | **434** | **354** | 40 | **9%** |
| **Mitrovicë e Veriut** | **2,326** | **2,045** | 249 | **11%** |

**Three of the four come out majority Muslim.** They are not. 6,708 people were enumerated
across all four — 0.42% of the country — in municipalities usually put at several thousand each.
In 2011 the same four were **not enumerated at all** and the table is null for them, which is at
least legible; 2024 replaces a hole with a wrong number.

### The independent check measures it rather than asserting it

§9p verifies a name-join by requiring every unit's (other population estimate / census count) to
sit in a tight band, because a scrambled join scatters that ratio. Kosovo cannot pass that test
and **should not** — Kontur's 2023 surface knows about people the 2024 census did not reach:

| | Kontur / census |
|---|---|
| the 34 enumerated municipalities | median **1.06x**, range 0.53–1.82 |
| Leposaviq | **3.2x** |
| Zubin Potok | **6.6x** |
| Zveqan | **12.4x** |
| Mitrovicë e Veriut | 1.11x — see below |

So the band is asserted on the 34 and *reported* on the four, and the boycott is confirmed by a
second, unrelated source instead of by a news story. The band is 0.5–2.0 rather than Serbia's
0.8–1.3, because Kontur is a model rather than an estimate and Kosovo's units run down to 434
people; it still discriminates, and the three above are the proof that it does.

**North Mitrovica is the one this check cannot see**, and that is a limit rather than a reprieve:
geoBoundaries splits Mitrovica along the Ibar, through the middle of one continuous built-up
city, so hexes on the north bank fall to the southern municipality — which reports 1.10x on
70,971 modelled people. **A boundary that cuts a city in half defeats a per-unit population check
for both halves**, and no widening of the band would recover it.

### What was done about it — CORRECTED 2026-09-06, Anita's call

The first build drew the four as published, on §3.5's rule: draw what the source counted and
say what is missing. **That was wrong, and the reason is worth keeping.** §3.5 assumes the
gap is a hole. Here it is an *inversion* — the map itself said Zubin Potok was 89% Muslim —
and no `note_public` repairs a claim the dots are making. **When the missing people would have
changed a unit's majority, "draw what was counted and warn" stops being the conservative
option.**

**ASK publishes the correction itself.** `census2024_63.px` — *Population by ethnicity and sex
at country and municipal level … (with estimation)* — is identical to the enumerated table
everywhere except these four, where it restores the boycotters:

| | enumerated | estimated | Serb, enum. | Serb, est. | added | of which Serb |
|---|---|---|---|---|---|---|
| Leposaviq | 3,185 | 9,485 | 2,677 | 8,648 | 6,300 | 5,971 |
| Zubin Potok | 763 | 3,385 | 80 | 2,702 | 2,622 | 2,622 |
| Zveqan | 434 | 2,867 | 76 | 2,505 | 2,433 | 2,429 |
| Mitrovicë e Veriut | 2,326 | 7,920 | 247 | 5,594 | 5,594 | 5,347 |
| **KOSOVA** | 1,585,566 | 1,602,515 | 36,652 | **53,021** | 16,949 | **16,369 (96.6%)** |

There is **no religion-with-estimation table**, so using it means one ethnic category implying
one religion: **Serb → Orthodox, spec §14.5**. Kosovo passes its three conditions more cleanly
than China does:

1. **The category was constituted religiously.** The Serb/Croat/Bosniak distinction is a
   religious boundary (Orthodox / Catholic / Muslim) drawn over a common language — that is
   *why* it is a boundary. China's nationalities meet this test unevenly; this one squarely.
2. **The group is not religiously mixed.** Kosovo's Serbs are Serbian Orthodox to a degree no
   serious source disputes. `sources/xk.py` asserts the addition stays above 90% one
   ethnicity, so a future revision that spreads it fails rather than passing quietly.
3. **No finer than the ethnicity is published at.** Both tables are per municipality and the
   derived rows land on the same four. Nothing is spread.

**Three things it deliberately does not do.** It does not touch the enumerated rows — the
derivation *adds* the missing Orthodox rather than restating what was counted, so a
municipality carries both a `measured` and a `derived` Orthodox row. It does not place the
**580 non-Serb people** in the estimate: nothing says what they are, and the enumerated
composition of these four is precisely the thing that is not representative (§3.5). And it
does not launder itself — every added row is `tier="derived"`, so §7a's control strips all
16,369 in one click and the raw table is one control away.

What it does to the map:

| | as published | corrected |
|---|---|---|
| Zubin Potok | 89% Muslim | **79.9% Orthodox**, 20.1% Muslim |
| Zveqan | 82% Muslim | **87.4% Orthodox**, 12.5% Muslim |
| Mitrovicë e Veriut | 88% Muslim | **73.0% Orthodox**, 26.7% Muslim |
| Leposaviq | 84% Orthodox | 95.2% Orthodox |

### And no earlier census helps

Asked and answered 2026-09-06. **2011 is strictly worse**: the same four are *null* — not
enumerated at all — and the national Orthodox count is 25,837 against 2024's 36,683.
**1991** was boycotted from the other side, by Kosovo's Albanians. **1981** was the last
census with full participation (13.2% Serb, against 3.4% in 2011 and 2.3% in 2024), but
Yugoslav censuses asked *nationality*, not religion, on pre-2008 municipal boundaries.

## 5. What the source is worth

- **93.5% Muslim, the highest share in Europe.** Hanafi Sunni; the census names no school and no
  Sufi order, so the Bektashi, Halveti, Rufai and Sa'di tekkes of Gjakovë, Prizren and Rahovec
  are inside the one cell. No split of this category would have found them — a Kosovar dervish
  answers `Islam` on a census form.
- **Catholics who are Albanian rather than foreign, and old.** 1.75% nationally but **16.8% of
  Klinë and 14.6% of Gjakovë** — the Dukagjin plain, which never converted under Ottoman rule —
  against 0.07% in Gjilan and 0.01% in Dragash. A minority worth drawing entirely because of
  where it is.
- **The Orthodox enclaves, which are real measurements.** Partesh 99.5%, Ranillug 94.5%,
  Shtërpcë 75.1%, Graçanicë 46.5% — municipalities created after 2008, a few thousand people
  each, inside an otherwise Muslim country.
- **0.50% no religion — the lowest of any country on this map**, and less than a third of the
  number who declined to answer.

## 6. Non-response

`Prefers not to answer` is an explicit cell: 23,718 people, 1.50%, excluded and not drawn (§3.5).
**40% of it is in Prishtinë** (4.19% of the capital), which is the urban refusal pattern Czechia
and Hungary have at twenty times the magnitude. The one municipality above the capital is
**Zveçan at 8.5%** — a different thing, and another trace of the boycott inside the small
population that did answer.

## 7. Both censuses are in the table

2011 and 2024, in one cube. Only 2024 is drawn. The 2011 column is useful for exactly one thing
and it is in §4: the nulls there are the same boycott, stated honestly.
