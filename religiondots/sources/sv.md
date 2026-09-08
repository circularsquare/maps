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
correlate. The bar is 1.96/√(n−1) = **+0.54**, higher than Guatemala's +0.43 because there are
fewer units.

| category | national | spearman | |
|---|---:|---:|---|
| Católico | 46.65% | **+0.88** | own geography |
| Evangélica y Pentecostal | 29.05% | **+0.82** | own geography |
| Ninguna (creyente) | 12.44% | **+0.74** | own geography |
| Protestante Tradicional | 7.97% | **+0.52** | national rate — misses by 0.02 |
| Religiones Orientales | 1.42% | +0.45 | national rate |
| the six under 1% | | | national rate (§11ad) |

**Every value is higher than Guatemala's equivalent**, which is what 647 respondents a
department buys over 405: Catholic +0.88 against +0.57, and `Ninguna` passing at +0.74 where
Guatemala's failed at +0.21. **So El Salvador is the first country in this set that can draw
its grey ramp where the survey found it**, and the cell is 796,542 people.

**`Protestante Tradicional` misses by 0.02 and was left alone.** The bar is what it takes to be
distinguishable from zero at 95%; moving it because a value landed just underneath is fitting
the test to the answer. Recorded so the next reader knows the call was close rather than clear.

## What was drawn

    2,925,233  christianity.catholic.latin      46.1%
    1,876,197  christianity.evangelical         29.5%
      796,542  unchurched                       12.5%
      505,958  christianity.protestant           8.0%
      149,171  other.sv                          2.4%
       46,825  christianity.witnesses            0.7%
       25,797  christianity.latterday            0.4%
       22,481  secular                           0.4%
        2,079  indigenous                        0.03%
          686  judaism                           0.01%

6,346 dots at 1:1,000 over 21,384 Kontur hexes. **Santa Ana is the only department where
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
- **`Protestante Tradicional`'s +0.52.** If a later wave lands and it clears +0.54, the country
  gains a fourth measured layer.
