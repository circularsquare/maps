// Loading and indexing. Owns the in-memory representation everything else reads.
//
// cities.json is an OBJECT keyed by QID (see SCHEMA.md) because the key is the
// contract with overrides. For drawing we need parallel typed arrays, so the
// conversion happens once here at load rather than per frame.

import { popColor, fade, rgb } from './colors.js';

export const state = {
  n: 0,
  keys: [],        // index -> "Q1490"
  index: new Map(),// "Q1490" -> index
  rec: [],         // index -> the full record (for the card / edit panel)
  nx: null, ny: null,   // normalised web-mercator, precomputed once
  r: null,              // base radius
  col: [],              // css colour, curation-aware
  touched: null,        // Uint8Array
  kind: null,           // Uint8Array, index into KINDS
  land: null,
  countries: {},
  ghs: {},          // GHS urban centres, keyed by centre id (see match_ghs.py)
};

// The 1975-2030 series ships as a bare array; these are its columns. 2030 (and
// 2025) are GHS projections, not observations — charts.js draws them dashed.
export const GHS_YEARS = [1975, 1980, 1985, 1990, 1995, 2000,
                          2005, 2010, 2015, 2020, 2025, 2030];
export const GHS_PROJ_FROM = 2025;

// See mergeOverrides. Must match build.py::REVIEW_FIELDS.
export const REVIEW_FIELDS = new Set(['ghs', 'ghsConf', 'ghsRole']);

// A Wikidata and a GHS elevation this far apart is not terrain variation, it is
// one of them being wrong — and spot-checking the tail says it is usually
// Wikidata (Facatativá at "3 m" when it sits on the Bogotá savanna at 2,600 m;
// Oyem at "3000 m" in a country whose highest point is 1,575 m). Surfaced as a
// curation flag rather than silently preferring either source.
export const ELEV_CONFLICT_M = 500;

// Kind codes, kept as small ints so the draw loop filters without string
// compares. Order matters only for the settings panel's display order.
export const KINDS = ['city', 'aggregate', 'admin'];
export const kindCode = k => Math.max(0, KINDS.indexOf(k));

export function mercY(lat) {
  const r = Math.max(-85.05, Math.min(85.05, lat)) * Math.PI / 180;
  return (1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2;
}

// Mirrors build.py's merge. The client reads base + overrides DIRECTLY rather
// than the built cities.json, because otherwise an edit is saved to
// overrides.json but invisible until someone remembers to hit Rebuild — which
// is indistinguishable from data loss, and was in fact reported as such.
//
// build.py still exists: it produces a distributable cities.json for publishing
// the map read-only. It is just no longer in the edit loop.
// Keep this in step with build.py::merge — see SCHEMA.md.
export function mergeOverrides(base, overrides) {
  for (const [key, ov] of Object.entries(overrides)) {
    if (ov._deleted) { delete base[key]; continue; }
    let rec = base[key];
    if (!rec) {
      if (!ov._created) continue;
      rec = base[key] = { ...ov._created, _created: true };
    }
    const edited = [], stale = [];
    for (const [field, patch] of Object.entries(ov)) {
      if (field.startsWith('_')) continue;
      // `was` is the base value at the time of the edit; if base has moved
      // since, the correction is suspect. Keep it, but flag it.
      if ('was' in patch && JSON.stringify(patch.was) !== JSON.stringify(rec[field] ?? null)) {
        stale.push(field);
      }
      rec[field] = patch.value;
      edited.push(field);
    }
    if (edited.length) {
      rec._edited = edited.sort();
      // Accepting or rejecting a GHS match is REVIEW, not curation. If it set
      // _touched, working the match queue would light up the map with "curated"
      // rings for thousands of cities nobody has written a fact about — and
      // "faded = not yet curated" is the entire progress signal.
      // Keep in step with REVIEW_FIELDS in build.py — see SCHEMA.md.
      if (edited.some(f => !REVIEW_FIELDS.has(f))) rec._touched = true;
    }
    if (stale.length) rec._stale = stale.sort();
  }
  return base;
}

export async function load() {
  const [land, base, overrides, countries, ghs] = await Promise.all([
    fetch('data/ne_50m_land.geojson').then(r => r.json()),
    fetch('data/base.json').then(r => r.json()),
    fetch('data/overrides.json').then(r => r.ok ? r.json() : {}).catch(() => ({})),
    fetch('data/countries.json').then(r => r.ok ? r.json() : {}).catch(() => ({})),
    // Optional: absent until match_ghs.py has run. Cities then simply have no
    // urban-centre section rather than the load failing.
    fetch('data/ghs_centres.json').then(r => r.ok ? r.json() : {}).catch(() => ({})),
  ]);
  state.land = land;
  state.countries = countries;
  state.ghs = ghs;
  const cities = mergeOverrides(base, overrides);

  // Sort population-descending once. The draw loop's density thinning relies on
  // this order: walking it means the LARGEST city in a screen cell wins.
  const entries = Object.entries(cities)
    .filter(([, c]) => Number.isFinite(c.lat) && Number.isFinite(c.lon))
    .sort((a, b) => (b[1].pop || 0) - (a[1].pop || 0));

  const n = entries.length;
  state.n = n;
  state.nx = new Float32Array(n);
  state.ny = new Float32Array(n);
  state.r = new Float32Array(n);
  state.touched = new Uint8Array(n);
  state.kind = new Uint8Array(n);
  state.keys = new Array(n);
  state.rec = new Array(n);
  state.col = new Array(n);

  for (let i = 0; i < n; i++) {
    const [key, c] = entries[i];
    state.keys[i] = key;
    state.rec[i] = c;
    state.index.set(key, i);
    state.nx[i] = (c.lon + 180) / 360;
    state.ny[i] = mercY(c.lat);
    // Sublinear so a 20M city does not swamp a 20k one; population clamped so a
    // stray non-settlement with a country-sized figure cannot paint a disc over
    // a continent.
    state.r[i] = 0.036 * Math.pow(Math.min(c.pop || 1e4, 4e7), 0.30);
    // Country name and language seeds are joined here rather than inlined in
    // base.json — identical for every city in a country, so inlining cost ~4 MB.
    const ci = countries[c.country];
    if (ci) {
      if (!c.countryName) c.countryName = ci.name;
      if (!c.languageCandidates && ci.langs && ci.langs.length) {
        c.languageCandidates = ci.langs;
      }
    }
    // Joined, not copied — this is a shared reference, so the Guangzhou blob's
    // 24-name member list exists once no matter how many of its cities point at
    // it. Attaching it here is what lets card.js stay a pure renderer.
    // `ghsConf` is authoritative, not `ghs`. A rejected match keeps its `ghs`
    // id as the record of what was turned down (see review.js), so joining on
    // the id alone would put the centre straight back on the card.
    if (c.ghs != null && c.ghsConf !== 'none' && ghs[c.ghs]) c.ghsCentre = ghs[c.ghs];

    // Elevation. Wikidata P2044 covers 52%; the urban centre's average covers
    // the rest of the matched cities. Checked against the 6,828 cities that
    // have both: median disagreement 13 m, 82% within 50 m — and blob AREA does
    // not predict the error (median 12 m for tiny blobs, 16 m for the largest),
    // so there is no size threshold worth gating on.
    //
    // FALLBACK ONLY. It never overwrites a Wikidata value, and the card labels
    // it, so a per-blob average is never mistaken for a surveyed figure.
    const ge = c.ghsCentre && c.ghsCentre.elev;
    if (ge != null) {
      if (c.elev == null) { c.elev = ge; c.elevSrc = 'ghs'; }
      else if (Math.abs(c.elev - ge) > ELEV_CONFLICT_M) c.elevConflict = ge;
    }
    state.touched[i] = c._touched ? 1 : 0;
    state.kind[i] = kindCode(c.kind);
    state.col[i] = colorFor(c);
  }
  return state;
}

export function colorFor(c) {
  const base = popColor(c.pop);
  return rgb(c._touched ? base : fade(base));
}

// Called after an edit so the dot recolours without a full reload.
export function refresh(key) {
  const i = state.index.get(key);
  if (i == null) return;
  state.touched[i] = state.rec[i]._touched ? 1 : 0;
  state.col[i] = colorFor(state.rec[i]);
}

export function add(key, rec) {
  const i = state.n++;
  // Typed arrays are fixed-length; hand-created cities are rare enough that
  // reallocating on each one is fine and keeps the hot path simple.
  const grow = (arr, Type) => { const a = new Type(state.n); a.set(arr); return a; };
  state.nx = grow(state.nx, Float32Array);
  state.ny = grow(state.ny, Float32Array);
  state.r = grow(state.r, Float32Array);
  state.touched = grow(state.touched, Uint8Array);
  state.kind = grow(state.kind, Uint8Array);
  state.keys[i] = key;
  state.rec[i] = rec;
  state.index.set(key, i);
  state.nx[i] = (rec.lon + 180) / 360;
  state.ny[i] = mercY(rec.lat);
  state.r[i] = 0.036 * Math.pow(Math.min(rec.pop || 1e4, 4e7), 0.30);
  state.touched[i] = 1;
  state.kind[i] = kindCode(rec.kind);
  state.col[i] = colorFor(rec);
  return i;
}
