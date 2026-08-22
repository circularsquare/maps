# citybrowser data contract

The one document to agree on before touching anything else. Every script and
every module reads or writes one of these three files. If you are working on
this project in parallel with someone else, this is the interface between you.

## The three layers

```
data/base.json        GENERATED. Never hand-edited. Delete it and rebuild.
data/overrides.json   HAND-CURATED. Never generated. The irreplaceable file.
data/cities.json      EXPORT ONLY. A standalone merged copy for publishing.
```

**index.html loads `base.json` + `overrides.json` and merges in the browser** —
it does NOT load `cities.json`. This matters: an earlier version read
`cities.json`, so an edit was written to `overrides.json` but stayed invisible
until someone remembered to press Rebuild. That is indistinguishable from data
loss, and was reported as such. The edit loop must never depend on a build step.

`js/data.js::mergeOverrides` and `build.py::merge` implement the same merge and
must be kept in step.

`build.py` is the only thing that writes `cities.json`.
`serve.py` (edit mode) is the only thing that writes `overrides.json`.
The `fetch_*.py` scripts are the only things that write `base.json`.

## Key

The primary key is the **Wikidata QID** (`"Q1490"`). It is stable across
refetches, which is what lets curation survive a rebuild.

Cities that exist in neither source get a synthetic key `"x0001"`, assigned by
edit mode and never reused. Everything else about them is identical.

## base.json

```jsonc
{
  "Q1490": {
    "name":      "Tokyo",
    "lat":       35.6895,
    "lon":       139.6917,
    "pop":       13960000,          // Wikidata P1082, MAX over census years
    "elev":      40,                // P2044, ~16% coverage, null otherwise
    "admin":     "Q1489",           // P131
    "adminName": "Tokyo Metropolis",
    "country":   "Q17",             // P17 -- NOT reliable for dependencies,
    "types":     ["Q1137012"],      // raw P31; filter via settlement_types.json
    "ghs":       673,               // ID_UC_G0 if matched, else null
    "ghsConf":   "high",            // high | low | none    (see Matching)
    "ghsRole":   "centre",          // centre | member | near
    "ghsDistKm": 4.4                // to the centre's pop-weighted centroid
  }
}
```

The urban centre's OWN fields — name, population, area, member list, history —
are deliberately **not** here. They live in `ghs_centres.json` and are joined on
`ghs` in the browser.

Fields added later by their own stage, all nullable: `aliases`, `gdpPc`,
`gdpSrc`, `climate`.

## overrides.json

**Field-level patches, never whole records.** A whole-record copy means the next
refetch either silently loses your edits or silently clobbers them.

```jsonc
{
  "Q1490": {
    "name": { "value": "Tokyo", "was": "Tōkyō", "at": "2026-08-15" },
    "facts": { "value": ["...", "..."], "was": null, "at": "2026-08-15" },
    "_deleted": true                // tombstone; build.py drops the city
  },
  "x0001": {
    "_created": { "lat": 12.34, "lon": 56.78, "name": "Somewhere" }
  }
}
```

`was` records the base value the edit replaced. `build.py` compares it against
the *current* base value and, when they differ, marks the field **stale** — the
source has moved since you corrected it, so the correction may no longer apply.
That is the whole reason patches beat copies.

## cities.json

`base` merged with `overrides`, plus provenance so the UI can show what is
curated versus fetched.

```jsonc
{
  "Q1490": {
    "name": "Tokyo",
    "...": "all base fields, with overrides applied",
    "_edited": ["name", "facts"],   // which fields came from overrides
    "_stale":  ["pop"],             // override whose `was` no longer matches
    "_touched": true                // has ANY override -> renders non-grey
  }
}
```

## Matching (GHS enrichment)

GHS attaches to a city; it never defines one. Written by `match_ghs.py`.

Two independent fields, because "which centre" and "how sure" are two
questions. `ghsRole` says what the relationship IS:

- `centre` — the city **is** the urban centre; its name is the centre's main
  name. The centre's figures describe this city.
- `member` — the city is **inside** the centre; its name appears in the
  centre's `GC_UCN_LIS_2025` list. The figures describe the blob, not the city.
- `near` — neither, but the centre has no city claiming it and this one is the
  best candidate. A suggestion for review, nothing more.

`ghsConf` says how much to trust it:

- `high` — for `centre`, name agrees **and** population within 3x **and**
  within the distance cut. For `member`, being named in the centre's own list.
  Safe to show as fact.
- `low` — a candidate exists but fails one of those. **Show as a suggestion
  with a visible "not confidently matched" state.** Never as fact.
- `none` — no candidate.

Population is checked for `centre` only. A member being far smaller than its
centre is the normal case, not evidence against it.

A wrong confident match is worse than no match: grey reads as "not done yet",
whereas a wrong match reads as "done" and ships an error nobody will notice.

## ghs_centres.json

The urban centres themselves, keyed by centre id, written by `match_ghs.py`
and **joined in the browser** on `ghs` — exactly like `countries.json` is
joined on `country`, and for the same reason: the Guangzhou blob's 24-name
member list must exist once, not once per city that points at it.

```jsonc
{
  "10933": {
    "name": "Guangzhou", "pop": 42987704, "area": 6454,
    "members": ["Shenzhen", "Guangzhou", "Foshan", "..."],
    "nMembers": 24,                 // full count; `members` is capped at 14
    "hist": [4523000, ..., 44000000],   // 1975..2030 in 5-year steps, 12 values
    "eco": "...", "basin": "Si", "elev": 21, "capital": true,
    "koppen": "Cwa", "koppen2070": "Cwa",   // Köppen now / under SSP2-4.5
    "tempC": 23.0, "tempRange": 22.4, "precipMm": 1904   // annual
  }
}
```

`hist` is positional — see `GHS_YEARS` in `js/data.js`. The last two entries
are GHS **projections**, not observations, and must not be drawn as though they
were: `charts.js` dashes them.

Climate here is **annual only** — there is no monthly series in the GHS climate
theme. The card's monthly min–max band is a separate, still-unbuilt field.

## Review: `ghsConf` is authoritative, not `ghs`

Rejecting a match in the review queue sets **`ghsConf: "none"` and nothing
else**. `ghs` keeps pointing at the centre that was turned down, as the record
of what was rejected.

So anything reading GHS data must gate on `ghsConf !== "none"`, never on `ghs`
being present. `js/data.js` skips the `ghs_centres.json` join and `js/card.js`
skips both the climate and urban-centre blocks on that basis.

**Why not just null out `ghs`:** in the PATCH API, `value: null` means *clear
this override and revert to base*, not *set the field to null*. Sending it
deletes the rejection instead of recording it.

### `REVIEW_FIELDS` — edits that do not count as curation

```
ghs · ghsConf · ghsRole
```

An override on one of these appears in `_edited` but does **not** set
`_touched`. Accepting a GHS match is review, not curation; if it counted,
working the queue would render thousands of cities as "curated" on a map whose
whole progress signal is "faded = not yet curated".

Defined in **both** `js/data.js` and `build.py` and they must stay in step, the
same way `mergeOverrides` and `build.py::merge` must.

## Derived at load time (not in any file)

`js/data.js` adds these while indexing; no source writes them:

- `ghsCentre` — the joined `ghs_centres.json` record (a shared reference).
- `elevSrc: "ghs"` — `elev` was borrowed from the urban centre's average
  because Wikidata had none. The card labels it.
- `elevConflict` — Wikidata and GHS both have an elevation and they differ by
  more than `ELEV_CONFLICT_M` (500 m). One of them is wrong; usually Wikidata.
- `countryName`, `languageCandidates` — joined from `countries.json`.

## Module ownership

| file | owns |
|---|---|
| `js/colors.js` | the four ramps + language palette. Pure, no deps. |
| `js/data.js` | loading, indexing, projection |
| `js/map.js` | canvas draw loop, density thinning |
| `js/tiles.js` | raster basemap: tile cache, level choice, attribution |
| `js/card.js` | hover card rendering |
| `js/charts.js` | sparkline / pie / climate band. Pure renderers. |
| `js/koppen.js` | Köppen code → label. A lookup table, nothing else. |
| `js/edit.js` | edit mode UI, PATCH calls, the four tools |
| `js/review.js` | the GHS match queue and its accept/reject writes |
| `serve.py` | static serving + the PATCH endpoint |
| `build.py` | **the confluence — one owner only** |

`js/edit.js` and `serve.py`'s PATCH endpoint are two halves of one contract and
should have the same owner.
