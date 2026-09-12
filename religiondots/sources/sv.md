# El Salvador — LAPOP AmericasBarometer, waves 2010–2023

Built 2026-09-08, the **second** country drawn from the AmericasBarometer after Guatemala.
`sources.md` §11ad assesses the source across all nine countries it could serve, §9bl is this
build's write-up, and `sources/gt.md` carries the arguments both countries share. Read §11ad
before doing a third.

## Why a survey at all

`sources.md` §11x opened ONEC/BCR's WordPress library, swept **777 media items**, found one
census file and **zero religion tables**, and closed the country. The UNSD oracle has no
Salvadoran row at all — not an old one, none. Nothing here reopens that.

## The files

| file | what it does |
|---|---|
| `sources/lapop.py` | the construction, the split-half and the category list, shared with `gt.py` and asserted byte-identical on Guatemala's output when it was factored out |
| `sources/sv.py` | LAPOP `.dta` → `data/normalized/sv.csv`, plus this country's decode |
| `sources/sv_geo.py` | COD-AB ADM1 → `data/geo/sv/`, **joined on the name**, with the code join asserted to still mispair |
| `sources/sv_grid.py` | Kontur 400 m hexes → `data/geo/sv/sv_hexes.gpkg` |
| `taxonomy/sv2023.py` | the eleven answers → the tree |

Acquisition is the same as Guatemala's and `sources/gt.md` has the walk-through: the `.dta` is
a manual click-through download on `lapopsurveys.org`, and `python sources/sv.py --fetch`
slims it.

## THE CODE JOIN IS A TRAP HERE, AND IT IS GUATEMALA'S INVERTED

Guatemala joins on the pcode, because LAPOP's `prov` is 200 plus the official department
number and COD's pcode is `GT` plus the same number. **El Salvador looks identical and is
not.**

    LAPOP prov   301..314  =  300 + the official WEST-TO-EAST department number
    COD-AB       SV01..SV14 =  ALPHABETICAL BY NAME

    LAPOP 302 Santa Ana     ->  SV02 is Cabañas
    LAPOP 303 Sonsonate     ->  SV03 is Chalatenango
    LAPOP 306 San Salvador  ->  SV06 is La Paz
    LAPOP 309 Cabañas       ->  SV09 is San Miguel

**Two of the fourteen coincide** — Ahuachapán at 01 and La Libertad at 05 — which is exactly
enough for a spot check to pass. The other twelve are wrong. **A permutation preserves every
total**, so San Salvador's 1.7 million people would have been drawn in La Paz and no
reconciliation, national figure or row count would have shown it.

So the join is on the **name**: all fourteen match with no aliases, names are unique on both
sides, and `sv_geo.py::check_code_join()` asserts that the code join *still* mispairs twelve.
If OCHA ever re-cuts these pcodes to the official order, the build stops and a person decides.

This is `sources/ni_geo.py`'s Waspám finding met a second time, in a second country, and it is
the argument for that file's existence.

## The held-out check, and the one that was withdrawn

**The test is the permutation, not the correlation.** LAPOP's weighted department distribution
tracks COD-PS's population distribution at **r = +0.968**, and **none of 20,000 random pairings
of the same 14 units reaches it** — the best random pairing manages **+0.964**. With fourteen
units and one dominant department, a raw correlation of 0.968 would have been worth almost
nothing on its own; the permutation is what turns it into evidence.

**The age comparison is printed and never asserted, because it has no power.** An earlier
version of `lapop.held_out()` asserted on department mean adult age. El Salvador returned
**r = −0.11** and the question became whether the join was broken or the test was useless. It
is the test:

    Guatemala     between-unit variance 0.887  vs  mean sampling variance 1.013   F = 0.88
    El Salvador                         0.218                            0.609    F = 0.36

F below 1 means the departments' mean ages are indistinguishable from that many draws on one
distribution. Guatemala's r = +0.52 was luck and had been written up as a passing witness;
that has been corrected in `sources/gt.md` and §9bi. **A check that has never been shown to
have power is not a check.**

## Which categories carry their own geography

The split-half (§14.16): rank the 14 departments on 2010–2014, rank them again on 2016–2023,
correlate. The bar is the exact null's smallest attainable value at p ≤ 0.05, **+0.4637** on
fourteen units, higher than Guatemala's +0.3608 because there are fewer units.
`sources/spearman_null.py` computes it. **It was 1.96/√(n−1) = +0.5436 until 2026-09-09**, and
that is the one thing about this country that changed; see the note below the table.

| category | national | spearman | exact p | |
|---|---:|---:|---:|---|
| Católico | 46.65% | **+0.88** | 0.000 | own geography |
| Evangélica y Pentecostal | 29.05% | **+0.82** | 0.000 | own geography |
| Ninguna (creyente) | 12.44% | **+0.74** | 0.002 | own geography |
| Protestante Tradicional | 7.97% | **+0.52** | 0.031 | own geography, from 2026-09-09 |
| Religiones Orientales | 1.42% | +0.45 | 0.054 | national rate |
| the six under 1% | | | | national rate (§11ad) |

**Every value is higher than Guatemala's equivalent**, which is what 647 respondents a
department buys over 405: Catholic +0.88 against +0.57, and `Ninguna` passing at +0.74 where
Guatemala's failed at +0.21. **So El Salvador is the first country in this set that can draw
its grey ramp where the survey found it**, and the cell is 796,542 people.

### `Protestante Tradicional` missed by 0.02 until the bar was corrected, and now clears it

It was left alone when this country was built, on the stated rule that a bar is not moved to
make something pass. Costa Rica hit the same bar from the other side — its `Católico`, 63% of
the country, at +0.7857 against +0.8002 — looked at what the bar actually was, and filed
`ask/007-cr`. **`1.96/sqrt(n-1)` is the null's standard deviation, not its 95th percentile**, and
at fourteen units it is a 0.023-level test rather than the 0.05 the docstring claimed. Anita
ruled on 2026-09-09 to replace it with the exact null, on the ground that the fix for a wrong
arithmetic claim is to make the claim true rather than to keep an accidental strictness.

`Protestante Tradicional`'s +0.5165 has an exact one-sided p of **0.031** and is drawn on its own
department shares. It and Costa Rica's `Católico` are the only two categories in the five LAPOP
countries that moved; `gt.csv`, `ec.csv` and `pa.csv` are byte-identical after the change, as are
`eg.csv` and `jo.csv` in the Arab Barometer module that had copied the same line.

**It is still the closest call in this country.** Nothing about the evidence changed; what
changed is that the test now rejects at the level it always said it did. `Religiones Orientales`
at +0.4472 is the next one down at p=0.054, one lattice step outside, and it stays out — it is
also the most heavily tied series here, twelve of its fourteen early-half shares being equal, and
against the conditional null that ties call for it goes to p=0.071, further out rather than in.

## What was drawn

    2,925,226  christianity.catholic.latin      46.1%
    1,876,197  christianity.evangelical         29.5%
      796,542  unchurched                       12.5%
      506,196  christianity.protestant           8.0%
      149,028  other.sv                          2.3%
       46,782  christianity.witnesses            0.7%
       25,771  christianity.latterday            0.4%
       22,460  secular                           0.4%
        2,080  indigenous                        0.03%
          687  judaism                           0.01%

Re-run 2026-09-09 with `Protestante Tradicional` on its own department shares; the national
totals move by at most a few hundred people, because the four measured categories and the tail
they leave still have to close on the same population.

6,347 dots at 1:1,000 over 21,384 Kontur hexes. **Santa Ana is the only department where
Evangelicals outnumber Catholics** (37.6% against 32.7%); San Vicente is 76.6% Catholic and
11.0% Evangelical. `unchurched` runs 18.3% of Usulután to 6.2% of La Paz.

## What is NOT claimed here, unlike Guatemala

**The traditional-religion cell is not called a floor.** It is 0.03%, and El Salvador's 2007
census counted 0.2% indigenous after the 1932 *matanza* made Nahua-Pipil identity dangerous to
state. §11ad's Suriname finding — LAPOP reading 0.21× a census on this cell — is about a card
with no local option **in a country with a large indigenous population**, and the second half
does not hold here. Drawn as given.

## Open

- **A second source for the non-Christian tail**, as for every LAPOP-only country. `other.sv`
  is 2.35% and nearly two thirds of it is the Eastern-religions box; El Salvador's Palestinian
  and Lebanese communities are the obvious thing to chase and nothing has been searched for.
- **Municipio geography.** LAPOP samples ~52 municipalities a wave and `municipio` is a PSU
  list rather than a partition. COD-AB ships ADM2 (2019 vintage, 262 municipalities) and
  El Salvador consolidated to 44 in 2023, so a future finer source would also need a decision
  about which vintage it is on.
- ~~**`Protestante Tradicional`'s +0.52.** If a later wave lands and it clears +0.54, the
  country gains a fourth measured layer.~~ Closed 2026-09-09, not by a new wave but by the bar
  being corrected to the exact null: +0.5165 against +0.4637 is p=0.031 and the fourth measured
  layer is drawn. `Religiones Orientales` at +0.4472, p=0.054, is now the row to watch.
