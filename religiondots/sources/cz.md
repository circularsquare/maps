# Czechia — ČSÚ Sčítání 2021 (SLDB), religious belief

Ingested 2026-09-02. `sources/cz.py` rebuilds `data/normalized/cz.csv` from
`data/raw/cz/`. All of `data/` is gitignored, so this file is the record.

**One sentence: this is the best subnational religion source in the project after ASARB,
and on category-versus-geography it is better than ASARB.** 78 named categories published
at municipality level for all 6,254 municipalities, with no suppression, no rounding, and
an exact partition in every single unit.

---

## 1. Why it matters — it does not make the §3.9 trade

sources.md §9a called it "the single loudest pattern": every one of the seven countries
acquired 2026-08-27 splits category depth from spatial depth. Australia 150 vs 34, Canada
168 vs 25, New Zealand 163 vs 11, Mexico 24 vs 4, Ireland 25 vs 5. Brazil, added the same
day as this file, is 65 vs 9 across twelve years.

Czechia publishes **78 categories at its finest geography**. There is no coarse table and
no fine table; there is one table.

Consequences:

- **No `allocate.py` step.** Nothing is derived, so nothing carries the `derived` tier.
- **Every row may become a ring** (spec §3.10). Canada's cannot, because allocation only
  spreads a total and a ring asserts presence. Czech rows are measured presence.
- The `may_ring` column is `True` throughout.

## 2. No suppression, no rounding — verified, not assumed

`cz.py` checks that the categories sum to the unit's own published total in **every unit at
every level: 6,724 of 6,724, zero mismatched.** That is the check that would catch a
suppressed cell, and nothing is suppressed.

The smallest published figure in the country is **1**: Společenství buddhismu v České
republice has one adherent, in one municipality, and ČSÚ says so. Církev Svatého Řehoře
Osvětitele has 3, across 2 municipalities.

Set against the rest of the project this is unusual and worth stating plainly:

| country | what happens to a small cell |
|---|---|
| **Czechia** | published as-is, down to 1 |
| Canada | base-5 random rounding; 0 of 321,757 counts is not a multiple of 5 (§3.8) |
| New Zealand | `-999` in-band "Confidential" sentinel |
| England & Wales | perturbation, +0.0006% drift |
| Scotland | perturbation, +0.068% drift |

## 3. The geography, including a level finer than municipality

The file carries the **whole territorial hierarchy in one CSV**, keyed by `uzemi_cis`.
Summing it as delivered counts the country eight times. `geo_level` keeps them apart and
nothing downstream may mix two levels.

| `uzemi_cis` | `geo_level` | units | population | note |
|---|---|---|---|---|
| 43 | `municipality` | 6,254 | 10,524,167 | obec — **the one to use** |
| 44 | `city_district` | 142 | 2,495,217 | městská část / městský obvod — **finer than obec** |
| 65 | `orp` | 206 | 10,524,167 | obec s rozšířenou působností |
| 72 | `prague_sso` | 22 | 1,301,432 | Prague's správní obvody |
| 101 | `okres` | 77 | 10,524,167 | district |
| 100 | `kraj` | 14 | 10,524,167 | region |
| 99 | `nuts2` | 8 | 10,524,167 | oblast |
| 97 | `country` | 1 | 10,524,167 | |

**`city_district` is the useful surprise.** Prague as one obec is 1.3M people in a single
polygon, which is the worst case for a dot map; at level 44 it is 57 units. Brno, Ostrava,
Plzeň and the other statutory cities are likewise subdivided, 142 in total.

**It is an alternative to its parent obec, not a child of it for drawing.** Use
`city_district` where it exists and `municipality` everywhere else, or the statutory cities
get drawn twice.

**Levels 44 and 72 cover only the statutory cities**, so they do NOT sum to the national
population — 2,495,217 and 1,301,432. Checking them against 10,524,167 is what first looked
like a missing unit when nothing was missing; `cz.py` now carries each level's own expected
population so that cannot recur.

## 4. Exact URL and re-fetch

    python sources/cz.py --fetch

One file, no login, no key:

    https://csu.gov.cz/docs/107508/4250766c-69e6-3845-0eb4-580f7a692558/sldb2021_vira.csv

Listed in the national open data catalogue as dataset
`https://data.gov.cz/zdroj/datové-sady/00025593/d48571d456d56aa11a4f3488eeba47ec`
("Sčítání 2021 — Obyvatelstvo podle náboženské víry"), whose distribution URL is the above.
A schema sidecar sits at `.../1fa30e44-fccb-f0f4-4e74-8d7f5efe8541/sldb2021_vira-metadata.json`.

**Expected size 80,039,394 bytes, 494,066 rows.** `cz.py` refuses anything under 50 MB —
sources.md §5a, a 200 is not a download. This one bit for real: a first attempt with curl
stopped at 55.5 MB on a full disk, returned HTTP 200, and left a file that parsed fine and
ended mid-row with 4,669 of 6,254 municipalities. Nothing in the CSV said so.

Columns: `idhod, hodnota, ukaz_kod, vira_cis, vira_kod, uzemi_cis, uzemi_kod, sldb_rok,
sldb_datum, ukaz_txt, vira_txt, uzemi_txt`. A blank `vira_kod` is the unit's own total, not
a category.

**Console encoding:** the category names are Czech and the reconciliation prints them, so a
cp1252 Windows console kills the run at the `print` with a `UnicodeEncodeError` that reads
like a data problem and is not one. `cz.py` reconfigures stdout to UTF-8 itself.

## 5. The category list is not a taxonomy — four different kinds of thing (§2.3)

All 78 are left verbatim per §2.4. Mapping them is a separate job, and it is not
mechanical, because the list mixes:

1. **Registered churches under their legal names** — Církev římskokatolická,
   Českobratrská církev evangelická, Církev československá husitská, down to Církev Oáza
   and Kněžské bratrstvo svatého Pia X.
2. **Bare tradition names** — islám, buddhismus, hinduismus, judaismus, sikhismus,
   taoismus, šintoismus, konfucianismus, zoroastrismus.
3. **Positions and currents** — ateismus, agnosticismus, deismus, esoterismus, pohanství,
   satanismus, animismus, Hnutí Nového věku.
4. **Joke and protest answers** — Jedi, Sith, pastafariánství.

### The tradition/institution overlap is the sharp one

Category 2 and category 1 **describe the same people twice under different headings**, and
the tradition wins by a lot:

| tradition (write-in) | institution (registered body) |
|---|---|
| islám **5,132** | Ústředí muslimských obcí **112** |
| judaismus **1,427** | Federace židovských obcí v ČR **474** |
| buddhismus **5,049** | Buddhismus Diamantové cesty **653**, Théravádový **54**, Společenství buddhismu **1** |
| hinduismus **1,226** | Česká hinduistická náboženská společnost **93**, Hare Krišna **455** |
| katolická víra (katolík) **235,834** | Církev římskokatolická **741,019** |

These must not be summed into a single "Muslim" node without a decision about what the two
rows mean. A respondent writing "islám" and one writing "Ústředí muslimských obcí" are
answering different questions.

### Jedi is not a rounding error

**Jedi is the 13th largest category in the country at 21,023 people, present in 2,512 of
6,254 municipalities** — more than Jehovah's Witnesses (13,298), more than Church of the
Brethren, more than Greek Catholics. Pastafarianism is 2,696 across 747 municipalities;
Sith 516.

ČSÚ tabulated them because people wrote them in, and at that size they will be visible on
the map. `cz.py` leaves them verbatim and does not decide. The honest options are a
"not a religion" node, a `joke` flag, or exclusion — and that is a taxonomy call.

## 6. The dominant fact: the question was voluntary and 30% did not answer

**Neuvedeno = 3,162,540, or 30.05% of the country.** It is the second largest category,
larger than every church combined (all named churches sum to ~0.95M).

It is also very unevenly spread. Across municipalities of ≥500 people the not-stated share
runs **11.4% to 81.3%, median 30.3%** — a 7× spread. Any map of Czech religion is
substantially a map of who answered.

Three more residual categories are large and mean different things from each other and from
Neuvedeno. None of them is "no religion", which is its own separate category:

| category | n | means |
|---|---|---|
| Bez náboženské víry | 5,027,141 | no religious belief |
| Neuvedeno | 3,162,540 | did not answer the (voluntary) question |
| věřící — nehlásící se k žádné církvi | 960,201 | believer, no church |
| věřící — hlásící se k církvi, název neuveden | 65,567 | believer, church not named |
| křesťanství | 71,089 | "Christianity", denomination unstated |
| katolická víra (katolík) | 235,834 | "Catholic", not the registered church |
| protestantská/evangelická víra | 27,149 | "Protestant", body unstated |

So the write-in residuals (`křesťanství` + `katolická víra` + `protestantská víra` +
`věřící` ×2) come to **1,359,840 — larger than every named church put together.** Czechia's
problem is not category depth, it is that most believers did not use the categories.

## 7. Presence, and what it means for rings

| category | n | municipalities where present |
|---|---|---|
| Církev římskokatolická | 741,019 | 6,111 of 6,254 |
| Jedi | 21,023 | 2,512 |
| Svědkové Jehovovi | 13,298 | 1,528 |
| pastafariánství | 2,696 | 747 |
| islám | 5,132 | 626 |
| judaismus | 1,427 | 353 |
| Společenství Josefa Zezulky | 1,053 | 298 |
| Kněžské bratrstvo sv. Pia X. | 156 | 46 |
| Církev Svatého Řehoře Osvětitele | 3 | 2 |
| Společenství buddhismu v ČR | 1 | 1 |

**Distinct categories per municipality: median 10, max 76, min 3.** Only 5 categories are
present in fewer than 10 municipalities and only 1 in exactly one.

Compare the US: median 20 distinct groups per county, max 171 (Los Angeles). Czech
municipalities are much smaller units than US counties, so median 10 out of 78 is dense.

## 8. Licence

ČSÚ open data, published through data.gov.cz under the national open data terms. Attribution
to Český statistický úřad. **To read properly before anything ships** — the catalogue entry
states the licence and it has not been checked against a redistribution of derived dots.

## 9. Surprises, collected

- 78 categories at the finest geography, breaking the §3.9 pattern that held for all seven
  earlier countries.
- Counts of 1 published without suppression or rounding.
- A geography level *finer* than municipality hiding in the same file, covering exactly the
  big cities where a dot map most needs it.
- Two levels in the file are partial, not national, and look like missing data if checked
  against the national total.
- Jedi outranks Jehovah's Witnesses.
- The tradition/institution double-listing, where "islám" is 46× "Ústředí muslimských obcí".
- The voluntary question's 30% non-response has a 7× spread between municipalities.
- Write-in residuals outnumber all named churches combined.
