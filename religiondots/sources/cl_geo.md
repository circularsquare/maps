# Chile — boundaries for the 346 census comunas

Wired 2026-09-04.

| | |
|---|---|
| source | OCHA **COD-AB `cod-ab-chl` v01**, `chl_admin3.shp` — 345 comunas |
| vintage | valid_on 2021-10-08, reviewed 2025-02-17, against a 2024 census |
| join | `adm3_pcode` = `CL` + the five-digit CUT. **345 of 346, both ways, no spares** |
| output | `data/geo/cl/cl_comunas.gpkg`, `cl_lookup.csv` |

The easy case, and worth writing down anyway for the two things that went wrong on the way
to it — neither of which was the join.

---

## 1. The geodatabase cannot be read, and it fails silently

COD ships four formats. The **geodatabase is 92 MB against the shapefile bundle's 205 MB**,
so it was tried first, and on this machine (GDAL 3.8.5, pyogrio 0.9.0, OpenFileGDB driver):

- `pyogrio.list_layers` returns all six layers with correct geometry types;
- `read_file(..., layer="chl_admin3")` **succeeds**, reports `crs=EPSG:4326`, and returns
  **zero features**;
- so does every other layer, including `chl_admin0`, which is one polygon.

No exception, no warning, no truncation error. A successful read of nothing.

**That is §5a's "HTTP 200 is not a download" in a new costume, and it generalises** — it is
in spec §12 now. Had `cl_geo.py` trusted the call, Chile would have failed later with an
empty join, and the symptom would have pointed at the join rather than at the driver. The
rule: **assert the feature count, not the absence of an exception.** `_read_layer()` does
exactly that and says so.

The shapefile reads fine. Chile's 205 MB is almost entirely the fjord coastline at full
resolution — `chl_admin3.shp` alone is 63 MB for 345 polygons.

## 2. The name check found a real error, in the boundary file

The join is by CUT — Chile's Código Único Territorial, a stable national standard — and it
matches 345 of 346 both ways with no spares. After Sri Lanka (`lk_geo.md`), a clean code
join is not something to accept on its own evidence, so names are compared as an independent
check. Three disagree:

| CUT | census | COD | |
|---|---|---|---|
| 06204 | Marchihue | Marchigüe | spelling |
| 16207 | Trehuaco | Treguaco | spelling |
| **01401** | **Pozo Almonte** | **Tocopilla** | **not spelling** |

Pozo Almonte and Tocopilla are different towns 400 km apart in different regions, so this
looked at first like Sri Lanka's disease — a code join quietly pairing the wrong units.

**It is not. COD's name is wrong and its geometry is right.** `CL01401` has
`adm2_name = Tamarugal`, `adm1_name = Región de Tarapacá` and `area_sqkm = 13,738`, which is
Pozo Almonte (13,766 km²) exactly; the real Tocopilla is `CL02301`, in Antofagasta, 4,101
km², present and correct. **COD carries the name "Tocopilla" twice and has no polygon called
Pozo Almonte at all.** So one `adm3_name` cell is mislabelled and nothing else is.

Two things follow, and both are in the script:

- **Names are written from INE, not from COD.** INE is authoritative for Chilean comuna
  names, and taking COD's would put "Tocopilla" in the tooltip of Pozo Almonte's polygon.
- **`KNOWN_NAME_DIFFS` lists all three with reasons, and a fourth would fail the build.** A
  name disagreement is not automatically a spelling variant; it has to be resolved, and the
  way to resolve it is what was done here — check the parent units and the area, which the
  code join does not determine.

## 3. The independent check the codes cannot fake

Names are only half-independent of a name-adjacent join, so `cl_geo.py` also compares the
census's **15+ count per comuna against that comuna's total population** from the D1
workbook. A correct join makes the ratio systematic; a scrambled one pairs a retirement
comuna with a young one and scatters it.

```
  min 0.745 (Cabo de Hornos)   median 0.822   max 0.941 (Río Verde)   on 345 comunas
```

A narrow band, which is what a correct join looks like and what a scrambled one cannot
produce. This is the check Sri Lanka's pcode join would have failed.

## 4. Antártica

**CUT 12202, 60 people aged 15+, has no polygon.** COD's admin3 stops at the continental and
island territory and does not carry the Chilean Antarctic claim, which is the right call for
a boundary set and would in any case put a dot near the South Pole. It is dropped, named in
the build output, named in `cl_lookup.csv` with `drawn=0`, and asserted in `countries.py` —
if a second comuna ever loses its polygon, `_cl_counts` fails rather than quietly shrinking
the country. At 1:1,000 it draws no dot either way.

## 5. Easter Island is drawn but not framed

Isla de Pascua is comuna 05201 and is in the data, so Chile's dot bbox runs to **109.4°W**
and fitting it would show the country as a sliver at the edge of an ocean. `countries.py`
sets `view=[-76.5, -56.0, -66.0, -17.3]`, continental Chile. Rapa Nui is still there for
anyone who pans to it — this is a framing default, not a filter.

## 6. Not done

- A 2024-vintage boundary set. The CUT has been stable and the join is complete, so there
  is nothing to gain; if INE publishes its own comuna geometry it would remove the reliance
  on COD's name field entirely.
- Reporting the `CL01401` mislabel upstream to OCHA. Worth doing and not done.
