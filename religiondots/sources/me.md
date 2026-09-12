# Montenegro — `sources/me.py`, `sources/me_geo.py`, `taxonomy/me2023.py`

Drawn 2026-09-06. **23 units, 10 nodes, 581,388 people, 93.39% of the counted country.**

| | |
|---|---|
| counting geography | **municipality, 23 units** — 27,000 people each, between North Macedonia's 21,000 and Serbia's 36,000 |
| published geography | **settlement, 1,462 units, 425 people each** — unusable, see §3 |
| placement | Kontur H3 r8 hexes, 9,821 of them, clipped to the municipality |
| basis | self-identification, census |
| tier | `measured` throughout |
| vintage | census 2023 |

---

## 1. Two sweeps called this country empty and the folder name is why

`sources.md` §11c walked nine MONSTAT census subpages and concluded *"none carries an
XLSX"*; §11k repeated it as *"unchanged from §11c"*. **Both are wrong, and the file was
never hidden.**

Montenegro's 2023 census was first scheduled for 2021, postponed twice, and finally taken in
December 2023. **MONSTAT never renamed the folder.** Every result file sits under

    https://www.monstat.org/uploads/files/popis 2021/...

so a path search for `2023`, `popis2023` or `census2023` returns nothing, and guessing
`uploads/files/popis2023/` returns 404 — which reads exactly like an office that has not
published. The landing page `page.php?id=1992` links no data files at all; it links
**fourteen subpages**, and the tables are on one of them (`id=2342`).

> **When an office's folder names disagree with its publication titles, enumerate its pages
> rather than guessing its paths.** A sweep of `page.php?id=` over a few hundred ids found
> the workbook in one pass, after two sweeps that checked the obvious landing pages had
> concluded the country was closed.

The same sweep found only three religion files in total, which is also the answer to "is
there a municipality-level table" — there is not. The settlement workbook is everything.

## 2. What it is

**`naselja vjera popis 2023..xlsx`** — *Broj stanovnika po naseljima i vjeri, Popis 2023.
godine*. 124 KB, one GET, no key and no wall. The filename really does carry two dots.

One sheet, `vjera`, 1,470 rows. A two-row merged header: `Opština | Naselje | Ukupno |
Hrišćanstvo{Pravoslavna, Katolička, Protestanti, Jehovini svjedoci, Ostale hrišćanske} |
Agnostik | Ateista | Budisti | Islamska | Ne želi da se izjasni | Ostale vjere | Ostalo`.
Twelve categories, read positionally — a merged two-row header does not survive pandas'
header inference and reading it by name silently loses the five Christian columns.

## 3. The settlement geography is better than anything on this map and cannot be used

1,462 settlements over 622,537 people is **425 people per unit**. Estonia's municipalities
are 11,000, Portugal's freguesias 2,800, Ireland's Small Areas 250. This would be the second
finest counting geography in the project.

**There is no settlement geometry to put it on.** geoBoundaries publishes MNE ADM0 and ADM1
and nothing below; ADM2 and ADM3 are 404. And the GISCO LAU file — §9e's "European
boundaries are free" file, already on disk since Poland — **has zero Montenegrin features**.
That was checked directly rather than inferred, and it is worth recording what it *does*
carry, because the neighbours are all there: Albania 61, Serbia 169, North Macedonia 80,
Switzerland 2,242, Liechtenstein 11, Iceland 69. Montenegro and Bosnia are the two absences,
and unlike Kosovo's (§9w) neither is about political status.

So the workbook is aggregated to its `Opština` column and the map is drawn on 23
municipalities. **The counts for a much better Montenegro are on disk and waiting for a
boundary file** — an OSM `admin_level=8` extract would do it, and would also solve §5.

## 4. The `z` sentinel, which is the whole caveat

The workbook's last three rows are its own legend:

    Simboli:
    "-" nema pojave                 no occurrence — a true zero
    "z" zaštićen podatak            protected data
    Shodno Zakonu o zvaničnoj statistici …   under the Law on Official Statistics

**So this is Lithuania's case (§9q) and not Kosovo's (§9w)**, and the two are
indistinguishable from the cell alone — Kosovo's blanks were true zeros proven so by an
exact partition, Lithuania's were suppression and reading them as zero would have deleted
1,683 people. Here reading them as zero deletes **29,225**. `me.py` asserts the legend is
still in the sheet, because it is the only evidence for the reading.

### It cannot be differenced out

The obvious attack: a settlement's total is published, so a lone suppressed cell equals the
total less the others. **Of the 745 settlements that publish a total and carry a `z`, exactly
zero have only one.** MONSTAT's complementary suppression is properly implemented. `check()`
asserts that count stays at zero, so a sloppier vintage is noticed rather than quietly
exploited.

### The threshold is ten, and it does not bound the total

No value published anywhere in either 2023 settlement workbook is below **10** — not one
settlement total, not one category cell. So a *primary* suppression is a number in 1–9.

**But most of the hidden mass is not primary.** 223 settlements have a gap larger than 9 ×
their number of `z` cells, so at least one of those cells is 10 or more: complementary
suppression, blanked to protect a neighbour rather than because it is small. Those 223 hold
**23,932 of the 29,225**. The threshold bounds a cell and not the country.

### And the total column is suppressed too

**This build assumed otherwise first**, and it surfaced as the categories out-summing the
country by exactly 31 people — those settlements' unsuppressed cells being counted against a
total that had coerced to NaN. 74 settlements publish `-` throughout and are genuinely
uninhabited; **219 publish `z` as their own population**. Where no denominator exists no
residual can be computed, so those 219 are dropped whole rather than half-drawn, and the
31 people in their unsuppressed cells go with them. At the threshold they hold between 219
and 1,971 people. The same 219 are suppressed in `naselja popis 2023.xlsx`, so there is no
second table to recover them from.

### Report it per category, never as a headline

§3.8's rule, from Lithuania: 4.69% reads as nothing. Per category, against MONSTAT's own
published national figures:

| category | drawn | published | hidden | | municipalities affected |
|---|---|---|---|---|---|
| Pravoslavna | 431,526 | 443,394 | 11,868 | **2.7%** | 21 of 23 |
| Islamska | 111,497 | 124,668 | 13,171 | **10.6%** | 23 of 23 |
| Katolička | 19,085 | 20,408 | 1,323 | 6.5% | 22 of 23 |
| Ateista | 13,402 | 14,260 | 858 | 6.0% | 23 of 23 |

**Islam loses four times the share Orthodoxy does**, and that is the same fact as the
geography: suppression protects small counts, a small count is a *local* minority, and
Montenegro's local minorities are disproportionately Muslim. By municipality the loss runs
**Petnjica 29.4%, Šavnik 25.4%, Rožaje 21.3%, Žabljak 19.5%, Ulcinj 15.0%** against Tivat
0.8% and Podgorica 0.9%. Every unit on this map under-shows whichever religion is its own
minority.

MONSTAT publishes no national figure for the other eight categories, so for those the loss
can only be stated as the number of affected municipalities. It is certainly proportionally
worse: `Ostale vjere` draws 310 people and is withheld in 17 of 23.

## 5. Tuzi and Zeta, and what the boundary vintage costs

Montenegro has been splitting municipalities for a decade — **Petnjica** off Berane in 2013,
**Gusinje** off Plav in 2014, **Tuzi** off Podgorica in 2018, **Zeta** in 2022. The
geoBoundaries cut has the first two and not the last two: 23 polygons against the census's
25 municipalities.

So Tuzi and Zeta are folded back into Podgorica, which is where the polygon still puts them.
**The cost is specific and it is not small: Tuzi is Montenegro's Albanian municipality**,
Catholic and Muslim in a country that is neither, and merging it into a capital of 208,000
averages away exactly the contrast this map exists to show. Ulcinj still carries the southern
Albanian population, so the phenomenon is on the map; Tuzi is not.

**The check that says the merge is right** is Podgorica's Kontur ratio: 0.95x against a
country median of 1.08x. If Tuzi and Zeta had been dropped instead of merged, or merged into
the wrong parent, Podgorica is the one unit where it would show.

## 6. What it is worth

- **The sharpest religious boundary in Europe over the shortest distance.** Gusinje 84%
  Muslim, Rožaje 77%, Plav 72%, Petnjica 70% — against Mojkovac 94% Orthodox and Nikšić 92%,
  an hour away.
- **Two unrelated Catholic communities**, and the map separates them by geography rather than
  by category: Tivat 16.3% and Kotor 10.0% are the Croats of the Bay of Kotor, Venetian and
  six centuries old; Ulcinj 9.2% is Albanian.
- **One Orthodox box and two churches claiming it** — see `taxonomy/me2023.py`, which files
  it on the parent for mk2021.py's reason.
- **No `unaffiliated` cell exists**, so Montenegro is unlit for it and its 2.5% secular is a
  floor rather than a measurement.

## 7. Left undone

1. **Settlement geometry.** The single biggest upgrade available, and it would restore Tuzi
   and Zeta at the same time. OSM `admin_level=8`.
2. **The 2011 census is a cross-check nobody has run.** `Tabela O5.xls` on `popis2011` is
   religion by municipality with **no suppression at all** — 21 units, 13 categories,
   summing to 620,029 exactly. It is 21 units on the old boundaries so it is not a better
   map, but differencing the two vintages per municipality would bound where the 2023
   suppression is hiding people, which nothing else here can do.
3. The 2011 file splits `Islamska` from `Muslimanska`; 2023 asks once. Worth a note if that
   comparison is ever made.
