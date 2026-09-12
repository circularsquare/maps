# Germany — Gemeinde boundaries

Built 2026-09-04. Rebuild with `python sources/de_geo.py --fetch`.

Writes `data/geo/de/de_gemeinden.gpkg`, layer `gemeinden`, 10,786 polygons, EPSG:4326.

---

## 1. Source

BKG **VG250**, Verwaltungsgebiete 1:250 000, *Gebietsstand 31.12.2022*, licence
**dl-de/by-2-0**. Free, no registration, no form.

```
https://daten.gdz.bkg.bund.de/produkte/vg/vg250_ebenen_1231/2022/
vg250_12-31.utm32s.shape.ebenen.zip
```

69,096,116 bytes. `de_geo.py` pulls out only `VG250_GEM.*` and reprojects from EPSG:25832.

**`GF == 4` is required.** GF is the *Geofaktor*; 4 is "mit Struktur Land". The 129 rows
with GF 2 are water bodies which carry the same ARS as their neighbours and would
duplicate keys.

## 2. The vintage was determined by trying them, and it is the whole reason this is clean

BKG publishes a **01.01** *and* a **31.12** edition of every year, and destatis nowhere
says which Gebietsstand the Sonderauswertung used. Measured, against `de.csv`:

| VG250 edition | land Gemeinden | census rows with no polygon |
|---|---|---|
| 01.01.2022 | 10,993 | 2 |
| 01.01.2023 | 10,981 | 10 |
| **31.12.2022** | **10,990** | **0** |

So the census was published on Gebietsstand 31.12.2022. Both near misses are worth
keeping, because both would have been survivable-but-wrong:

- **Against 01.01.2023** the ten leftovers (Bromskirchen, Ostrau, Zschaitz-Ottewig,
  Dünwald, Menteroda, Anrode …) are Gemeinden dissolved *during* 2023, after publication.
  A straightforward §8.1 vintage error.
- **Against 01.01.2022** the two leftovers are **Schwedt/Oder** and **Pinnow** in the
  Uckermark, and they are **not a merger**. The Gemeinde number is unchanged and only the
  *Verbandsschlüssel* moved — Schwedt from amtsfrei `0532` into Amt `5051`, Pinnow from
  Amt `5310` into `5051`:

  ```
  census  Pinnow   120 73 5051 440        VG250  Pinnow   120 73 5310 440
  census  Schwedt  120 73 5051 532        VG250  Schwedt  120 73 0532 532
  ```

  **This is the interesting one.** The obvious repair — join on the 8-digit AGS, which
  drops the Verband — makes both leftovers disappear and looks like a fix. It is not: it
  would then have left Passow, Berkholz-Meyenburg and Mark Landin as orphan polygons whose
  people are counted inside Schwedt and Pinnow but whose territory belongs to no census
  row, placing roughly 3,000 people in the wrong villages *with every count still
  reconciling*. **The longer key is the safer one here**, which is the exact opposite of
  Poland, where the LAU id had to be sliced down to six digits. There is no general rule
  about key length; there is only checking the join both ways and asking what the
  leftovers are.

## 3. The join

The census `geo_id` **is** the ARS, verbatim — no derived key, no slicing, no alias map.

```
census Gemeinden 10,786  |  polygons 10,990
census units with no polygon: 0
polygons with no census row: 204, by BEZ {'Gemeindefreies Gebiet': 204}
```

**All 204 leftovers are `Gemeindefreies Gebiet`** — unincorporated forest, lake and
military areas with no inhabitants, and therefore no religion row. `de_geo.py` asserts
that rather than assuming it, because "204 leftovers" and "204 leftovers that are all
uninhabited" are very different findings and only one of them is fine.

### The independent evidence

Names: **10,697 of 10,786 agree** after folding. The 89 that differ are all cosmetic, and
some are interesting — destatis carries official bilingual **Sorbian** names where BKG
carries the German alone:

```
120520000000  census 'Cottbus/Chóśebuz, Stadt'                 BKG 'Cottbus'
120610329329  census 'Märkische Heide/Markojska Góla'          BKG 'Märkische Heide'
120615113005  census 'Alt Zauche-Wußwerk/Stara Niwa-Wózwjerch' BKG 'Alt Zauche-Wußwerk'
034030000000  census 'Oldenburg (Oldenburg), Stadt'            BKG 'Oldenburg (Oldb)'
```

## 4. Placement, and Germany is the worst case on the map

Gemeinden are the count layer and the placement layer. **§8.2's trick does not apply at
all here**: Gemeinden are historical administrative units, not units engineered to a
population target, and they span six orders of magnitude.

| | |
|---|---|
| median Gemeinde | **1,797** people — finer than a Polish gmina |
| smallest | Dierfeld, **9** |
| largest | **Berlin, 3,596,999 — 4.3% of Germany in one polygon** |
| top 8 polygons | 12.7% of the country |
| 78 Gemeinden ≥100,000 | **31.6% of the country between them** |

So two thirds of Germany is drawn well and one third is drawn as city-sized blobs. Berlin
is the single worst polygon anywhere on this map — worse than Warsaw (4.7% but a fifth the
people), Prague or Bucharest — and it is exactly where the interesting thing is, because
the non-church half of Germany is disproportionately urban. Unlike Czechia's Prague and
Estonia's Tallinn, **there is no district-level religion table to swap in**: destatis
publishes Gemeinden and stops.

## 5. The fix — BUILT 2026-09-04, see `de_grid.md`

`Religion.zip` (`sources/de.md` §5) carries the **same three categories on a 100m INSPIRE
grid, 3,088,036 populated cells**. The cell id encodes the lower-left corner in
ETRS89-LAEA (EPSG:3035), so the geometry is computed, not downloaded — **no boundary file,
no join, no vintage question**.

That makes Germany the one country on this map where placement is **measured rather than
modelled**, and §8.2's approximation dropped outright rather than bounded. It is not merely a
finer placement layer: it carries the religion split per cell, so a Catholic dot is placed on
where the Catholics are, not on where the people are.

**Built.** `sources/de_grid.py` writes `de_grid_1km.gpkg`, which is now this country's `place`
layer — 209,154 cells, Berlin at **799** instead of 1, and 17,215 of 17,215 (unit, node) rows
placed on measured weights with no fallback. `de_gemeinden.gpkg` is still the file this script
writes and `de_grid.py` depends on it for the cell → Gemeinde assignment and the clip; it is
no longer the layer `countries.py` reads.

The 1km grid was chosen over the 100m one because the dot value binds first: 82,710 dots
against 3,088,036 cells would be 37 cells per dot. `de_grid.md` §3 has the numbers.

## 6. Reuse

VG250 covers only Germany, but the pattern generalises: **BKG's two editions per year mean
"the 2022 file" is ambiguous**, and any German dataset should be joined against both and
the leftovers read. Austria (Statistik Austria's Gemeinden) and Switzerland (swisstopo)
publish on the same annual-Gebietsstand model with the same hazard.
