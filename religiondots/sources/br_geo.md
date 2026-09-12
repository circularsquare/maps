# Brazil — boundaries for `br.csv` (município, malha 2010)

`data/geo/br/` · acquired 2026-09-03 · pairs with `sources/br.md`

**Join is clean for the year that matters: 5,565 of 5,565 2010 municípios matched, 0
missing.** The 2022 half of `br.csv` deliberately does not match — see §3.

---

## 1. What was downloaded

27 per-state zips, one per federative unit, plus the merged layer built from them.

| file in `data/geo/br/` | what |
|---|---|
| `<uf>_municipios.zip` × 27 | IBGE malha municipal 2010, one per state |
| `<uf>/<NN>MUE250GC_SIR.shp` | the shapefile inside each, 1:250,000, SIRGAS 2000 |
| `br_municipios_2010.gpkg` | **the one `countries.py` reads.** 5,565 municípios, merged |

Built by `python sources/br_geo.py`.

## 2. Re-fetch recipe

```
python sources/br_geo.py            # downloads what is missing, then merges
python sources/br_geo.py --merge    # merge only, from what is already on disk
```

Base URL, **http and not https** (§4):

```
http://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/
    malhas_municipais/municipio_2010/<uf>/<uf>_municipios.zip
```

## 3. Vintage — the whole reason this is the 2010 mesh (spec §8.1)

`br.csv` carries two censuses. The detailed half — 56 categories, the only municipal
denominational data Brazil has — is **2010, with 5,565 municípios**. The current mesh has
5,570.

The five created since are Pescaria Brava and Balneário Rincão (SC), Mojuí dos Campos (PA),
Paraíso das Águas (MS) and Pinto Bandeira (RS). **All five were split off existing
municípios**, and none of the old codes was retired. So joining 2010 data to a current mesh
would match all 5,565 codes, report no error, and quietly draw five parent municípios over
territory that no longer belongs to them.

That is §8.1's failure mode in its pure form: the codes match and the answer is wrong. The
Connecticut case fails loudly because the codes vanish; this one does not fail at all.

`br_geo.py` prints the join both ways, and the asymmetry is the evidence:

```
2010 data: 5,565 municipios, 0 with no polygon, 0 polygons with no data
2022 data: 5,570 municipios, 5 with no polygon, 0 polygons with no data
```

Those 5 are exactly the new municípios. **A 2022 build will need the 2022 mesh**, which is a
single national file rather than 27, and is a different download.

## 4. geoftp serves an incomplete TLS chain

`https://geoftp.ibge.gov.br/` fails to verify:

```
curl: (60) SSL certificate problem: unable to get local issuer certificate
Python: [SSL: CERTIFICATE_VERIFY_FAILED] unable to get local issuer certificate
```

**certifi does not fix it.** The missing piece is an intermediate certificate that the
server should be sending and does not; browsers paper over it by fetching the intermediate
themselves via the AIA extension, and command-line clients do not. So this is not a stale CA
bundle and not bot protection — it is a server misconfiguration, and it will look like a
local problem to anyone who hits it.

**Plain `http://` works and is what `br_geo.py` uses.** The files are public and unsigned
either way, so the TLS that is not working was not protecting anything. Worth revisiting if
IBGE fixes the chain.

### The malhas API is not an alternative

`https://servicodados.ibge.gov.br/api/v3/malhas/` has a valid certificate and serves
GeoJSON, but:

- `periodo=2010` returns **HTTP 500**. It serves only the current mesh.
- `qualidade` takes `minima | intermediaria | maxima`, not a number, and the error message
  is the only place that is written down.
- Its output is generalised even at `maxima`.

So it is the wrong tool for a vintage-specific job. Recorded because it is the obvious first
thing to reach for.

## 5. The shapefile name cannot be predicted from the URL

Inside `ac_municipios.zip` is `12MUE250GC_SIR.shp` — **numeric** state code 12, not `ac`. So
the member name does not follow the URL and `br_geo.py` globs for `*.shp` rather than
constructing it, and fails if a directory holds more or fewer than one.

Columns: `ID`, `CD_GEOCODM` (the 7-digit município code, the join key), `NM_MUNICIP`.

## 6. IBGE ships two lakes as municípios

The raw merge is **5,567**, not 5,565. The two extras are in Rio Grande do Sul:

| code | name |
|---|---|
| 4300001 | Lagoa Mirim |
| 4300002 | Lagoa dos Patos |

The two big coastal lagoons, carried in the municipal mesh as pseudo-municípios with codes of
their own. They hold no census rows, so nothing would have been placed in them either way,
but a polygon that is a lake does not belong in a layer of populated units and its presence
makes the count disagree with the expected 5,565 for no stated reason. `br_geo.py` drops them
by code and says so.

This is the Brazilian shape of the same problem the US solves with cartographic (`cb_`)
rather than TIGER boundaries — see sources.md §5, "pre-clipped to the coastline, so dots do
not scatter offshore".

## 7. Projection

**EPSG:4674, SIRGAS 2000** — a geographic CRS, and for practical purposes identical to
WGS 84 (the datums agree to within a few centimetres). All 27 states agree on it;
`br_geo.py` refuses to merge if they ever stop agreeing. `scatter.py` reprojects to 4326.

No Krovak-style surprise here, unlike Czechia — but the check is the same one sources.md §5c
asks for, and it is cheap.

## 8. What is not solved: placement inside a município

Czechia and Ireland need no placement layer because their count units are already tiny.
Brazil is the opposite end:

| | |
|---|---|
| municípios | 5,565 |
| median population | ~11,000 |
| largest | São Paulo, 11,253,503 |
| largest by area | Altamira (PA), ~159,000 km² |

Dots are currently spread uniformly across the município polygon, which in Amazonia means
spreading a town's population across an area the size of England. **This is the weakest
placement in the project** and it is a known gap rather than a decision.

The fix is IBGE's **setor censitário** — ~310,000 units for 2010, designed to a household
target, which is exactly the §8.2 shape. It is a much larger download (per-state, hundreds of
MB) and has not been taken yet. sources.md §5b's placement-layer table should gain a Brazil
row when it is.

## 9. Licence

IBGE geographic data is public. Attribution: "IBGE, Malha Municipal Digital 2010". Same
caveat as `sources/br.md` §10 — terms to be read properly before anything ships.
