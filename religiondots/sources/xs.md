# Israeli settlements in the West Bank (`xs`): CBS 2022 census, the units beyond the Green Line

**Drawn 2026-09-15** (session `d743fc47-settlers`), on Anita's ruling on ask 028. 267 CBS units,
723,899 people (Jews 695,648, the register's Others 28,251). 721 dots at 1:1,000, 69 at 1:10,000.
The first entry with `territory=False`: part of no country, no outline or wash, never Auto's pick.

- `sources/xs.py` -> `data/normalized/xs.csv` (from `data/normalized/il.csv` and
  `data/geo/il/dropped_units.json`; nothing is downloaded)
- `sources/xs_geo.py` -> `data/geo/xs/xs_units.gpkg`, `xs_places.gpkg` (reuses `ps_geo.py`)
- `taxonomy/xs2022.py` -> Israel's mapping restricted to Jews and Others; `countries/xs.py` -> the entry
- sources.md **§xs-2026-09-15** is the summary. Israel's record is `sources/il.md`, Palestine's
  `sources/ps.md`.

```
python sources/il.py; python sources/il_geo.py; python sources/ps_geo.py    # must be on disk first
python sources/xs.py
python sources/xs_geo.py
```

## 1. The ruling, and what the flag does

Anita, 2026-09-15, after ask 028 (Palestine drawn with the settlers on neither entry): *"i meant
like we draw them as part of neither country. but yeah like they dont actually have any territory
so it shouldnt be possible to auto-mode onto them. only way to see them should be in all countries
mode, or selecting them manually in the dropdown i guess. we can in the future apply this to some
other disputed places too maybe. northern cyprus?"*

Built as a general flag, `territory=False` on the entry (documented in the `countries.py` field
docstring, asserted to be a bool):

| where | what it does |
|---|---|
| `country_shapes.py` | leaves the entry out of `country_shapes.geojson`, so it has no wash and no outline, and the missing-polygon stop does not fire for it. Tested into the scratchpad: 170 codes, `il` and `ps` present, `xs` absent |
| `tiles.py` | writes `territory` (a bool, default true) into each counts.json entry, in the full build and in `--refresh-meta` |
| `index.html` | `autoMay(cc)` is false for it. Gated: the shapes branch of `considerCamera`, the small-country `framed` branch and the dot-tally `best`. An entry held by hand is let go when the reader switches Auto on. The title's "Religion in N countries" leaves it out, as it leaves out territories |

Unchanged on purpose: the picker lists it (`BUILT`), the all-countries view draws its dots, and
the coverage caption's "N of M countries record this" counts it, as that count already includes
territories. Hovering a settlement in the all-countries view gives Palestine's card, because the
shapes place the point in Palestine; that card does not include the settlers. `estimates.py`
needs nothing: only countries with a Pew or hand estimate go through its outline step.

The viewer edits were syntax-checked with `node --check`; they were not tried in a browser.

## 2. The code

`xs`, from ISO 3166-1's user-assigned range (AA, QM to QZ, XA to XZ, ZZ), which ISO never assigns;
Kosovo's `xk` is the precedent here. The loader needs two lowercase letters. Before this entry the
string `xs` appeared in no code, data or record file in the project as a code. Northern Cyprus
would take another code from the same range.

## 3. Who is in it

CBS 2022 units that `il_geo.py` dropped as beyond the Green Line (majority of area inside OCHA's
`cod-ab-pse` admin 0, East Jerusalem included), from `il.csv`:

| group | East Jerusalem (locality 3000) | elsewhere | total | here |
|---|---:|---:|---:|---|
| Jews | 221,457 | 474,190 | 695,648 | drawn |
| Others (no religion on the register) | 15,783 | 12,468 | 28,251 | drawn |
| Muslims | 359,478 | 67 | 359,545 | Palestine's entry |
| Christians | 13,651 | 61 | 13,712 | Palestine's entry |
| Druze | 0 | 0 | 0 | |

**Muslims and Christians are left to Palestine.** PCBS's Table 3 Jerusalem row includes J1, the
part Israel annexed (`sources/ps.md` §2), so East Jerusalem's Palestinians are already drawn.
Drawing CBS's Muslims and Christians here would draw them twice. By unit: 16 Arab-majority
Jerusalem units hold 352,737 Muslims, 7,438 Christians and 7,616 Jews and Others; 62 Jewish-majority
Jerusalem units hold 6,742 Muslims and 6,214 Christians; the 189 units outside Jerusalem hold 67
and 61.

**The register does not separate Arab from non-Arab Christians**, so the non-Arab Christians
living in these units (the ex-Soviet immigrant population Israel's entry describes) are on neither
entry. This source cannot size them; the 6,275 Christians in Jewish-majority units are a ceiling,
since some of those are Palestinians.

**Observance**, of the 695,648 Jews: ultra-religious 34.5%, religious 29.9%, secular 12.6%,
traditional 10.6%, the unsplit parent 7.9% (units under 85% Jewish, `sources/il.md` §5), the two
"other" rows 3.2%, mixed 1.3%. Of the 240,304 ultra-religious, Modi'in Illit has 80,160, East
Jerusalem 75,074 and Beitar Illit 60,560 (89.8% together).

Tiers are carried from `il.csv`: Jews 55,002 `measured` and 640,646 `derived` (the observance
rows), Others 28,251 `derived`. `sources/xs.py` pins all four group totals and stops if the Israel
build moves them.

## 4. The mapping (`taxonomy/xs2022.py`)

`il2022.MAP` and `REVIEW` restricted to the Jews and Others groups; every node is Israel's
(`judaism.haredi`, `.dati`, `.masorti`, `.hiloni`, `judaism`, `unrecorded`). Muslims, Christians
and Druze are EXCLUDED with the reason, which also keeps `islam` and `christianity` out of this
entry's coverage: select Islam and the entry is unlit, not lit and empty. No new nodes, so
`build_tree.py` was not run.

## 5. Placement (`sources/xs_geo.py`)

The same geometry `ps_geo.py` took the settlements out of Palestine's weights with:

- **Units**: `ps_geo.settlement_units()`. 148 on CBS's polygons; 119 placeholders under 0.05 km2
  (219,586 Jews and Others; Talmon, Shilo, Kiryat Arba, Beit El) replaced by discs at 4,000 people
  per km2, radius at least 300 m.
- **Hexes**: Kontur PS plus IL, de-duplicated on `h3`, kept where the centroid is in a COD-AB
  governorate: 5,885 hexes.
- **Pieces**: each unit cut by the hexes it overlaps, 1,066 pieces, weighted by Kontur population
  times the overlapping share (ps_geo's own formula).
- **Witness**: recomputing the hex-level removal from these pieces gives `ps_lookup.csv`'s
  `settlements_removed` per governorate to within 0.3 of a person (477,068 against 477,067).
- Every unit has populated pieces; the median unit has 100% of its shape under a populated hex
  (p10 99%). Nothing fell back to equal shares.

Scatter: 402 (unit, node) rows on these weights at 1:1,000; 721 dots on 360 pieces; 2,899 people
(0.40%) under one dot at the entry's level. 69 dots at 1:10,000.

**Why not Israel's placement.** Israel places on unit polygons with no grid, because §8.2e found
Kontur coarser than its statistical areas (`sources/il.md` §8). These are the same kind of units,
so inside a small one the weighting barely matters (one or two pieces); in a large one, a whole
jurisdiction drawn as one polygon, it puts the dots on the built-up part. Using the pieces also
keeps this entry and Palestine's agreeing on where these people live.

**Not checked.** `xs_places.gpkg` does not match `kontur_cap.py`'s layer pattern, so the density
cap check does not run on it, and `kontur_cap.csv` has no `ps` or `il` rows. Whether any hex under
Modi'in Illit or Beitar Illit sits at Kontur's 46,200/km2 cap was not looked at; it could only move
dots within a unit.

## 6. Israel and Palestine

- **Palestine**: `note_public`'s settlers sentence and `gap`'s last clause now point at this entry.
  Nothing else changed; `gap_share` is untouched. `sources/ps.md` §7 says so.
- **Israel**: unchanged. Its `note_public`, `gap` and `note` say what Israel's entry leaves out and
  never that the settlers are drawn nowhere.
- **No one is drawn twice**, and it is asserted: `_xs_counts` stops on any unit not in
  `dropped_units.json` (the units `_il_counts` excludes), and the groups are ones PCBS does not
  count.

## 7. Northern Cyprus

A later candidate for the same flag (Anita's suggestion), **not built** and no source looked for.
`cy`'s Natural Earth outline already leaves the north out, so the flag would apply as it stands.

## 8. Open

- `coverage.py` at step 9 failed on three `sn` Sufi nodes on 2026-09-15, during another session's
  Senegal node move; nothing of `xs`'s. It checks the last build, so `xs` passes vacuously until the
  build tail writes counts.json with it.

## 9. Review, 2026-09-15 (session `d743fc47-rev9`)

Full pass, including the `territory=False` change to shared code. `check_md.py` clean,
`built_countries.py --check` ok.

- **Figures.** Every `note_public`, `gap` and `grain` figure recomputes off `xs.csv`: 267 units;
  723,899 people (Jews 695,648, Others 28,251); 237,240 in locality 3000; ultra-religious 240,304,
  34.5% of Jews; religious 29.9, secular 12.6, traditional 10.6; 373,257 = 359,545 + 13,712; 2,711
  people per unit. The no-double-count check is rev8's (`sources/ps.md` §9) and was not redone.
- **The flag in code.** `countries.py` documents and asserts it; `country_shapes.py` leaves
  unshaped entries out of both the wanted set and the missing-polygon stop; `tiles.py` writes it
  in both paths; `index.html` gates the three camera routes and the title. `counts.json` has one
  `"territory": false`, and `country_shapes.geojson` has no `xs`.
- **Tried in a browser** (headless, `serve` on a private port, cache-busted), which §1 says had not
  been done:
  - Auto on, framed on the West Bank, then landed at zoom 12-14 on Modi'in Illit, Beitar Illit,
    Ma'ale Adumim, Ariel, Pisgat Ze'ev and the Hebron hills, each from no selection: never `xs`.
    Palestine everywhere, Israel at Pisgat Ze'ev. The screenshot shows Auto (Palestine) with
    Palestinian dots only.
  - **The same run with `autoMay` swapped for `!!META[cc]` never picked `xs` either.** At these
    spots the missing shape is what keeps Auto off it; `autoMay` guards the dot-tally and
    small-country routes, which this test did not reach. Not a defect, but the test does not show
    the gate itself doing anything.
  - Picker: "Israeli settlements in the West Bank, 721k" is listed. Picking it gives `xs` with Auto
    off: blue dots on the settlements and East Jerusalem, no outline, no wash. Switching Auto on
    afterwards lets it go, to Palestine.
  - All countries: `data/buffers/xs.bin` loads and the settlement dots draw blue among Palestine's
    green.
  - Title: "Religion in 157 countries"; 158 without the `territory` test.
- **`tools/review_dump.py xs` fails** with `ModuleNotFoundError: No module named 'il2022'`. The tool
  imports `taxonomy.xs2022` with only the project root on `sys.path`, and `xs2022.py` does a bare
  `from il2022 import`, which works only where `taxonomy/` itself is on the path, as in
  `countries.py`. So `xs`'s REVIEW entries are invisible to the precedent sweep. One line in either
  file fixes it; not made, since the tool is shared.
- **With inferred dots hidden**, `check_rollup.py xs` loses 59,170 people (8.2%): all 28,251
  `unrecorded`, and 30,920 on `judaism` from the Mixed, Other main lifestyle and Other rows, every
  one `derived` at a root. Israel's entry has the same pattern (13.1%) and `xs` carries its tiers;
  not re-decided here.
- **`gap` is slightly too clean.** It says the 373,257 Muslims and Christians are "left to
  Palestine's entry", while `note_public` and §3 say the non-Arab Christians among them (at most
  6,275) are on neither entry. Not edited; a wording point.
- **§8 is closed.** The build tail after this entry reported coverage ok for 171 countries.
