# Romania — UAT boundaries

Built 2026-09-03 by `python sources/ro_geo.py --fetch`. Writes `data/geo/ro/ro_uat.gpkg`
(3,181 polygons) and `data/geo/ro/ro_uat_lookup.csv`, both of which `countries.py` reads.

## 1. Sources

Two files, both reused from Poland's ingest:

- **Eurostat GISCO, LAU 2021, 1:1M, EPSG:4326** — the same 98MB European zip
  `sources/pl_geo.md` describes. Romania has 3,181 features, and `LAU_ID` **is the SIRUTA
  code**.
- **Eurostat LAU 2021 – NUTS 2021 correspondence table**, 7MB:
  ```
  https://ec.europa.eu/eurostat/documents/345175/501971/EU-27-LAU-2021-NUTS-2021.xlsx
  ```
  Per-country sheets of `NUTS 3 CODE | LAU CODE | LAU NAME NATIONAL | LAU NAME LATIN |
  POPULATION | …`. This is the file that makes Romania joinable at all.

## 2. The problem: the census has no codes

`sources/ro.py` can only emit `geo_id` as the string `"JUDEŢ|NAME"`, because the INS
workbook identifies rows by name and nothing else. Romanian place names are **not unique
nationally** — `Călăraşi` is a county and three separate communes, `Păuleşti` is a commune
in both Prahova and Satu Mare — so the county half of the key is load-bearing.

The correspondence table bridges it because its `LAU NAME NATIONAL` column carries the
**type prefix in the same form the census uses** (`Municipiul Oradea`, `Oraş Aleşd`,
`Sânmartin`), and because Romanian NUTS3 regions *are* the judeţe, one for one.

So: `(judeţ, name)` → `(NUTS3, name)` → `LAU CODE` → GISCO polygon.

### The NUTS3 ↔ judeţ rename is derived, not typed

Rather than hand-writing 42 pairs, `ro_geo.py` matches each judeţ's set of UAT names
against each NUTS3's set of LAU names and takes the best overlap, then requires a perfect
bijection over all 42. That is a stronger check than a lookup table: a typed table cannot
notice that a boundary release has moved a commune between counties, and this does.

Result: 34 of 42 counties match on **every** name; the other 8 miss exactly one each,
which §4 resolves.

## 3. `ş` vs `ș` — an encoding trap that looks like a vintage problem

`ş` U+015F (s-**cedilla**) and `ș` U+0219 (s-**comma-below**) are different codepoints, and
so are `ţ` U+0163 and `ț` U+021B. Romanian orthography wants comma-below; a great deal of
software — including Eurostat here — emits cedilla. **INS writes the census with
comma-below and Eurostat writes the same names with cedilla.**

Without folding both to ASCII, roughly a third of Romanian place names fail to match, and
the failure reads exactly like a boundary-vintage mismatch. It is not one. `ro.py:fold()`
normalises cedilla→comma→ASCII and is shared by both scripts so the two sides cannot drift
apart.

## 4. The last eight names, and why elimination is legitimate here

After exact folded matching, 8 of 3,181 remained — one in each of 8 counties:

| judeţ | census | Eurostat | difference |
|---|---|---|---|
| Bacău | `ORAȘ SLĂNIC-MOLDOVA` | `Oraş Slănic Moldova` | hyphen |
| Neamţ | `MUNICIPIUL PIATRA-NEAMȚ` | `Municipiul Piatra Neamţ` | hyphen |
| Bistriţa-Năsăud | `CICEU-MIHĂIEȘTI` | `Ciceu - Mihaieşti` | spacing |
| Călăraşi | `ORAȘ LEHLIU- GARĂ` | `Oraş Lehliu Gară` | a stray space in the census |
| Cluj | `RÂȘCA` | `Rişca` | â/î |
| Galaţi | `SUHURLUI` | `Suhurului` | variant spelling |
| Covasna | `MUNICIPIUL SFÂNTU GHEORGHE` | `Municipiul Sfântul Gheorghe` | Sfântu/Sfântul |
| Ilfov | `DOBROEȘTI` | `Dobroieşti` | e/ie |

The first four are handled by `squash()`, which treats a hyphen as a space. The last four
are genuinely different spellings and no rule short of a hand-written alias table would
catch them.

They are resolved **by elimination**: after the two name passes, each of those counties had
**exactly one** census UAT and **exactly one** LAU code left unmatched, so they are the
same place and no spelling judgement is needed. `ro_geo.py` refuses to eliminate when two
or more remain on either side, because that would be a guess rather than a deduction.

```
resolved by: 3,173 exact name, 4 hyphen/space, 4 elimination
census UATs with no SIRUTA code: 0
census UATs with no polygon: 0
polygons with no census UAT: 0
```

## 5. Placement layer

UATs are the count layer and the placement layer — INS publishes religion at no finer
unit. Median UAT is about 3,000 people, which makes Romania the **finest count geography
on the map after Ireland's Small Areas and the UK's Output Areas**, and far finer than
Poland's gminy or Mexico's municipios.

## 6. Where it is not fine: Bucharest

| unit | population | area | share |
|---|---|---|---|
| Municipiul Bucureşti | 2,158,169 | 239.7 km² | **9.77%** |
| Municipiul Iaşi | 389,020 | 91.5 km² | 1.76% |
| Municipiul Cluj-Napoca | 327,927 | 174.9 km² | 1.48% |

(Populations are GISCO's `POP_2021`, an administrative figure; the census resident count
for Bucharest is 1,716,961. The share is computed within GISCO's own column so it is
internally consistent.)

**One tenth of Romania in a single 240 km² polygon** — worse than Warsaw's 4.69% and
approaching Prague's 12.4%. Bucharest's six sectors exist as administrative units and are
in some boundary sets, but **INS publishes no religion for them**, so subdividing would be
an allocation inventing structure the source does not have (spec §3.10). Left as one
polygon.

This is now the third country where the capital is the worst unit on the map — Prague,
Warsaw, Bucharest — and the pattern is that Central European capitals are single
municipalities. Only Czechia publishes below that level.

## 7. `ro_uat_lookup.csv`

Because the census carries no code, the resolved `(county|name) → SIRUTA` map is written to
disk and `countries.py` joins on it. If `sources/ro.py` is re-run and the names change,
`ro_geo.py` must be re-run too; `_ro_counts()` raises rather than silently dropping rows if
the lookup goes stale.
