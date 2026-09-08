# Brazil — the setor placement layer

Built 2026-09-05. `sources/br_setores.py` → `data/geo/br/br_setores_2022.gpkg`.

| | |
|---|---|
| geometry | IBGE **malha de setores censitários 2022**, 27 per-state gpkg, ~1.5 GB |
| population | IBGE **Agregados por Setor**, `Agregados_por_setores_basico_BR_20260520.zip`, 15 MB |
| output | **466,996 setores**, `setor` / `unit` / `pop`, 377 MB |
| population carried | **203,080,756** — Brazil's whole 2022 population |

**This changes where dots sit, and nothing else.** Every município's total is exactly what
IBGE published, before and after. §14.4 permits refining placement and forbids inventing
magnitude, and this is the first half only.

---

## 1. Why Brazil needed it and the US does not

§8.2's argument is that a statistical agency designs its fine unit to a population target,
so an equal share per polygon is already a population weighting. That holds for US tracts
(~4,000 people each) and it is why the US needs no population file.

**Brazil was placing on its count layer**, which §8.2a already named as the failure case:
5,570 municípios, median ~38,000 people, median area 1,527 km². **São Paulo is one polygon
holding 11.5M people**, so its ~11,500 dots were spread uniformly over the Serra da
Cantareira, the Billings and Guarapiranga reservoirs and the built-up city alike. Rio,
Manaus, Brasília and Belém were the same. It is the most conspicuous thing wrong with the
country at city zoom and invisible at national zoom.

## 2. Setores, not the 1 km grid

IBGE publishes both. Setores won on one decisive property:

**`CD_SETOR[:7]` IS `CD_MUN`, on every row of all 27 states** — asserted in the script, not
assumed. So assigning a setor to its município is a string slice: no spatial join, no cells
straddling a boundary, no clip, no slivers. That is precisely the work `de_grid.py` has to do
for Germany's 1 km grid, and it disappears here. Setores are also the units the census was
collected in.

The cost is size: 1.5 GB of source geometry against the grid's 416 MB.

## 3. Equal shares would NOT have been good enough

The tempting shortcut was to skip the population file and split each município's dots equally
across its setores, on the US tract argument. **That is wrong for Brazil.** Setores are built
to roughly 300 households in cities and fewer in the country, and rural setores cover enormous
areas — an equal split would have pulled Brazil's dots systematically into the countryside,
which is a smaller version of the very error being fixed.

So `v0001`, resident population per setor, is joined from the agregados file and used as the
weight. It totals 203,080,756, Brazil's whole 2022 population, which is the check that the
join is complete.

## 4. Three things in the data that would have gone wrong quietly

- **914 setores are delivered as SEVERAL ROWS EACH — and the population join must not see
  them.** IBGE ships a multi-part setor as one feature per disjoint part: river islands in
  Pará and Amazonas overwhelmingly, plus coastal fragments in Rio, São Paulo and Santa
  Catarina. `pop` is keyed on the setor CODE, so mapping it onto the parts as delivered gives
  a five-part setor **five times its population and five times its pull on the dots**. They
  are dissolved first: 868 setores that arrived as 5,405 rows. The tell was a plain
  duplicate-key check, and the arithmetic that confirms the fix is that the layer then has
  exactly the 468,099 distinct codes the population file has.
- **The two lagoon pseudo-municípios turn up again, in a third disguise.** `br_geo.py` drops
  Lagoa Mirim and Lagoa dos Patos from the municipal mesh by code. Here they are setores
  `430000100000000` and `430000200000000` — 2,884 km² and 10,202 km² of open water in Rio
  Grande do Sul — with **`CD_MUN` null and `CD_SIT` null**, so neither the water filter nor a
  código test catches them. Dropped on the general rule instead: a setor with no município
  cannot be placed in one.
- **`CD_SIT = 9` is "massas de água".** IBGE gives open water its own setor codes — 1,245 of
  them. They carry no population so a population weight already ignores them, but they are
  dropped outright so they cannot take a dot through the zero-population fallback either.
  This is IBGE having done part of §8.2c's inland-water job already.

## 5. What it cost and what it bought

```
                         before              after
  placement polygons     5,565 municípios    466,996 setores
  dots                   176,291             176,291        <- unchanged, by construction
  polygons holding a dot 5,565               150,088
  weighting              equal per município setor population
  fallbacks used         —                   0
```

**The dot count is identical**, which is §4.1's invariant: this is a placement change and may
not move a single count. All 26,591 (unit, node) rows were placed on real setor population
and **none** fell back to equal shares.

`water.py` then clips the sea out of 4,920 of the setores (0.02% of their area, 23 s, cached),
with one entirely-water setor left whole per §8.2c.

A side effect worth noting: Brazil's dot bbox pulled in from 28.8°W to 32.4°W, because Martim
Vaz and the other Atlantic islets belong to mainland municípios and no longer take dots —
their setores are uninhabited, so a population weight puts nothing there. The `view` override
in `countries.py` was already hiding this; now it is true of the data as well.

## 6. What it is not

**A population weight, not a religion one.** Nothing measures which setor a given church's
members live in, so a Catholic dot and an Assembleia de Deus dot are spread identically
inside a município. The map should be read as "religion by município, drawn where the people
are" — never as a neighbourhood measurement. `note_public` says so in as many words.

This is the same class of claim as the Philippines' barangay weighting
(`_PhBarangayWeighter`), and the two weighters are mechanically identical. They are kept
apart because a weighter is where a country's placement argument lives; **if a third country
needs one, they should become a single class.**

## 7. Gotchas for re-running

- The 27 downloads are ~1.5 GB and land in `data/raw/br_setores/`. `--fetch` skips what is
  already there, so a re-run is cheap.
- The build writes **one state at a time** with `mode="a"`. Concatenating 466,996 simplified
  polygons in memory first is several GB for no benefit.
- Geometry is simplified at `0.0002°` (~22 m), which cut coordinates to 25% on the test state
  with zero invalid or empty results. A dot only has to land in the right setor.
- pyogrio warns `MULTIPOLYGON inserted into layer of geometry type POLYGON` once per state
  after the dissolve. Benign — GDAL writes it correctly and reads it back — but it is why the
  layer is not a strictly conformant GeoPackage.
