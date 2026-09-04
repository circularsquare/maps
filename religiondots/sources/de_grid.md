# Germany — the 1km grid as a measured placement layer

Built 2026-09-04. Rebuild with `python sources/de_grid.py --fetch`.

Writes `data/geo/de/de_grid_1km.gpkg`, layer `grid1km`, **209,154 cells**, EPSG:4326. This
replaces `de_gemeinden.gpkg` as the country's `place` layer; the counts still come from the
Gemeinde table.

---

## 1. Why

Berlin is **3,596,999 people in one Gemeinde polygon** of 891 km². Placed on that polygon,
its 3,596 dots scattered uniformly over the whole city — including the Grunewald, the
Müggelsee and Tempelhofer Feld — and Neukölln and Zehlendorf came out identical. 78 Gemeinden
hold 31.6% of Germany and every one had the same problem (`de_geo.md` §4).

## 2. What makes this different from every other country here

spec §8.2 separates two jobs: the **magnitude** per unit, and the **placement** within it.
Everywhere else on this map the placement layer is a *proxy*:

| country | placement weight | what it is |
|---|---|---|
| US, AU, CA, IE | equal share per fine polygon | units engineered to a population target (§8.2) |
| US, additionally | a **fitted** demographic model | §8.4 — guesses where a denomination sits inside a county |
| NZ | SA1 population | a real population, but of everybody, not of the religion |

Germany needs none of them, because destatis publishes **the same three categories on the
grid itself**:

```
GITTER_ID_1km;x_mp_1km;y_mp_1km;Insgesamt_Bevoelkerung;
Roemisch_katholisch;Evangelisch;Sonstige_keine_ohneAngabe
```

So the weight for `christianity.catholic` inside Munich is **Munich's own per-cell Catholic
count**. Germany is the only country on this map whose within-unit placement is a
*measurement* rather than a model or an assumption, and §8.2's approximation is dropped
outright rather than bounded. Nothing here is fitted, so §7 has no confidence to mark: a
Catholic dot in Neukölln is there because the register put Catholics in that square kilometre.

The run confirms it end to end — **17,215 of 17,215 (unit, node) rows placed on the
religion's own grid counts**, with no fallback used at all.

## 3. Why 1km and not the 100m file

The 100m grid has 3,088,036 populated cells; Germany draws 82,710 dots. That is 37 cells per
dot — a placement layer far finer than anything that can be shown, at 14× the bytes. At 1km:

| | |
|---|---|
| cells | 210,556 published, **209,154** kept |
| median cell | about 390 people — finer than every count unit here except UK Output Areas |
| Berlin | **799 cells**, for ~3,596 dots |
| Hamburg / Munich / Cologne | 655 / 305 / 343 |

**The dot value is the binding constraint, not the grid.** Berlin draws ~4 dots per km², so
1km cells are already finer than the dots they carry.

## 4. The two approximations, both stated rather than hidden

- **A cell is assigned to the Gemeinde containing its centre.** destatis assigns each address
  to the cell holding its coordinate and never clips, so a boundary cell's people may belong
  to either side. The effect is bounded by the cell size and moves only *weight within* a
  Gemeinde, never a count.
- **Each cell is then clipped to its assigned Gemeinde**, so a dot can never leave its own
  unit. Without this the coastal and border squares hang over the edge and dots land in the
  Baltic — which looks like a bug in the data rather than in the geometry.

## 5. Reconciliation — reported, not asserted

The grid and the Gemeinde table are perturbed **independently** by the Cell-Key method
(`de.md` §3), so they are not expected to agree exactly and only relative shares within a
Gemeinde are used.

| | grid | Gemeinde table | |
|---|---|---|---|
| Catholic | 20,704,243 | 20,746,759 | −0.205% |
| Protestant | 19,085,379 | 19,127,130 | −0.218% |
| Sonstige | 42,751,976 | 42,837,219 | −0.199% |
| population | 82,547,068 | 82,711,282 | −0.199% |

**The agreement is the check.** All four differ by the same −0.2%, which is what a shared
cause looks like: 8,766 people at 277 addresses that destatis excludes from the grid but not
from the Gemeinde totals, plus the 159,392 people in the 1,436 cells whose centre falls in no
Gemeinde. A column read into the wrong slot would show up here as one row disagreeing with
the other three.

The cell id is verified against the file's own centre columns — `CRS3035RES1000mN…E…` parses
to `x_mp_1km`/`y_mp_1km` with **max off-by 0m** across all 210,556 rows — so the geometry is
derived from the id rather than trusted from a column, and a reissue in another projection
fails loudly.

## 6. Completeness

**34 Gemeinden holding 8,199 people (0.0099%) have no 1km cell at all** — they are small
enough that no cell centre lands inside them. Each gets its own Gemeinde polygon added as a
single cell, carrying its own census counts, so the layer covers all 10,786 units.

That matters more than the number suggests: `scatter.py` looks placement geometry up by unit,
and a unit with none would have its dots silently dropped. `de_grid.py` asserts the coverage
rather than leaving it to be noticed later.

## 7. Reuse

Any country publishing a population or attribute grid can use this shape, and several do —
the INSPIRE 1km grid is a European standard and Austria, the Nordics and the Netherlands all
publish on it. The pattern is: **grid cell → containing admin unit → clip → per-node weight
column**, with the counts still coming from the admin table.

What does *not* generalise is the good part. Most grids carry population only, which makes
them a better §8.2 proxy but still a proxy. Germany is unusual in publishing the **same
variable** on the grid as in the table, and that is what turns the placement from an
assumption into a measurement.
