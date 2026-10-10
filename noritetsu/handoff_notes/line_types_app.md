# Line types in the app: ready patch (APPLIED 2026-10-09)

Applied as written, re-anchored on the current index.html (every edit landed unchanged), with
the build_regions `write_search` diff; search.json rebuilt with the `types` column. Checked in
headless Chrome, both themes and modes: Tōkaidō Shinkansen "High-speed rail · JR Central ·
514.4 km", Hudson Line (MNR register) "Commuter rail · Metro-North · 109.0 km" with
"Intercity trains also run here.", Berlin S1 "Commuter rail · DB · 51.7 km", Ginza "Metro ·
Tokyo Metro · 14.1 km"; a search hit from search.json with Russia not loaded: "High-speed
rail · (no operator) · 641 km" (Sapsan). line_100_probe identical on ch, lu, jp, us.

For whoever holds `dist/index.html` (an app session was editing it on 2026-10-09, so this was
not applied). The data is shipped: `dist/data/<cc>/types.json` for every region, written by
`tools/line_types.py` (method and rules in `line_types.md`). Without the file a country simply
shows no type, so the patch is safe to land before or after a publish.

What it shows:

- **Line view**, the small line under the name, the type first:
  `Commuter rail · Metro-North Railroad · 109.0 km`. A register line that carries other kinds
  of train gets one muted sentence under it: `Intercity and high-speed trains also run here.`
- **Search results**, a line hit: `Intercity rail · Amtrak · 2224 km · named train`. From the
  loaded line where its country is in; from search.json once build_regions writes the `types`
  column (diff at the end). Until then, hits of unloaded countries show no type, as now.

Checked on a copy of today's index.html: every `before` below occurs exactly once, the patched
script passes `node --check`, and `node tools/lint_map_expressions.js` says "expression lint
OK (19 layers checked)". The wording on real lines:

    de 4000 Mannheim - Basel   Regional rail   | Intercity, high-speed and commuter trains also run here.
    de 1700 Hannover - Hamm    High-speed rail | Regional, intercity and commuter trains also run here.
    de 6020 Berlin Ring        Commuter rail
    us Hudson Line (MNR)       Commuter rail   | Intercity trains also run here.
    us Empire Builder          Night train
    us Northeast Corridor      Intercity rail  | High-speed and commuter trains also run here.
    jp 奥羽線                   Regional rail   | High-speed and intercity trains also run here.
    jp ゆりかもめ               People mover

## 1. loadRegion: fetch types.json with the country's other files

Before:

```js
      get('along.json').catch(() => null),
    ]).then(([l, s, f, a, , ne, o, al]) => {
      if (o) OPS_CC[cc] = o;       // before mergeRegion: its buildOps reads it
      mergeRegion(cc, l, s, f, a, ne);
```

After:

```js
      get('along.json').catch(() => null),
      // Line types written after the build (tools/line_types.py); none is fine.
      get('types.json').catch(() => null),
    ]).then(([l, s, f, a, , ne, o, al, ty]) => {
      if (o) OPS_CC[cc] = o;       // before mergeRegion: its buildOps reads it
      mergeRegion(cc, l, s, f, a, ne, ty);
```

## 2. mergeRegion: read them before LINE_FOLD, as the English names are

Before:

```js
function mergeRegion(cc, l, s, f, a, ne) {
  DATA_GEN++;
  fillEnglish(l.lines, s.stations, ne);
```

After:

```js
function mergeRegion(cc, l, s, f, a, ne, ty) {
  DATA_GEN++;
  fillEnglish(l.lines, s.stations, ne);
  fillTypes(l.lines, ty);
```

## 3. The helpers, after fillEnglish

Before:

```js
  const ln = ne.ln || {};
  for (const line of lines)
    if (!line.name_en && ln[line.id]) { line.name_en = ln[line.id]; line.enFill = 1; }
}
```

After:

```js
  const ln = ne.ln || {};
  for (const line of lines)
    if (!line.name_en && ln[line.id]) { line.name_en = ln[line.id]; line.enFill = 1; }
}

/* LINE TYPES (types.json, tools/line_types.py, line_types.md): what kind of railway a line is,
   in words a rider uses. `ltype` is the line's type; a register line (track) also has `lalso`,
   the other kinds of train over it, biggest share first. Keyed by the shipped ids, so read
   before LINE_FOLD, as the English names are. A country without the file shows no type. */
const TYPE_LABEL = {
  high_speed: 'High-speed rail', intercity: 'Intercity rail', night: 'Night train',
  regional: 'Regional rail', commuter: 'Commuter rail', metro: 'Metro', light_rail: 'Light rail',
  tram: 'Tram', monorail: 'Monorail', people_mover: 'People mover', maglev: 'Maglev',
  funicular: 'Funicular', tourist: 'Tourist railway',
};
// The same kinds as words before "trains": "Intercity and high-speed trains also run here."
const TYPE_WORD = {
  high_speed: 'high-speed', intercity: 'intercity', night: 'night', regional: 'regional',
  commuter: 'commuter', metro: 'metro', light_rail: 'light rail', tram: 'tram',
  monorail: 'monorail', people_mover: 'people mover', maglev: 'maglev', funicular: 'funicular',
  tourist: 'tourist',
};
function fillTypes(lines, ty) {
  if (!ty) return;
  for (const line of lines) {
    const t = ty[line.id];
    if (!t) continue;
    line.ltype = typeof t === 'string' ? t : t.type;
    line.lalso = (typeof t === 'string' ? [] : t.also || []).filter(x => TYPE_WORD[x]);
  }
}
function typeLabel(line) {
  return (line && TYPE_LABEL[line.ltype]) || '';
}
function typeAlso(line) {
  const ws = ((line && line.lalso) || []).map(t => TYPE_WORD[t]);
  if (!ws.length) return '';
  const list = ws.length === 1 ? ws[0] : `${ws.slice(0, -1).join(', ')} and ${ws[ws.length - 1]}`;
  return `${list[0].toUpperCase()}${list.slice(1)} trains also run here.`;
}
// A search hit's type, from the loaded line where its country is in, else from the index.
function hitType(l) {
  const t = typeLabel(LINE_BY_ID.get(l.id) || l);
  return t ? `${esc(t)} · ` : '';
}
```

## 4. combineParts: a line joined over a border keeps its type

combineParts builds a new object, so the fields have to be carried. Before:

```js
    kind: first('kind'), src: first('src'), service, dup: origs.some(p => p.dup),
```

After:

```js
    kind: first('kind'), src: first('src'), service, dup: origs.some(p => p.dup),
    ltype: first('ltype'), lalso: (named.find(p => p.ltype) || {}).lalso || [],
```

## 5. renderLine: the type under the name

Before:

```js
    + `<small>${opLinks(line) || 'operator not stated'}`
    + ` · ${line.km.toFixed(1)} km${closedNote}`
    + (line.parts && line.parts.length > 1 ? ` · in ${esc(line.regions.map(regionName).join(', '))}` : '')
    + `</small></div>`;
```

After:

```js
    + `<small>${typeLabel(line) ? `${esc(typeLabel(line))} · ` : ''}`
    + `${opLinks(line) || 'operator not stated'}`
    + ` · ${line.km.toFixed(1)} km${closedNote}`
    + (line.parts && line.parts.length > 1 ? ` · in ${esc(line.regions.map(regionName).join(', '))}` : '')
    + `</small></div>`
    + (typeAlso(line) ? `<p class="muted" style="margin:6px 0 0">${esc(typeAlso(line))}</p>` : '');
```

The `<h2 class="sec">` above it still says "Named train" or "Line"; the type sits beside the
operator rather than replacing that heading.

## 6. readSearchIndex: the type column, when search.json has it

Before:

```js
  for (const [cc, [ids, ns, es, refs, oi, ki, cols, kms, fl]] of Object.entries(d.lines)) {
    for (let i = 0; i < ids.length; i++) {
      const o = ops[oi[i]];
      const l = { id: ids[i], region: cc, name: ns[i],
                  name_en: es[i] || (ne[cc] && ne[cc].ln[ids[i]]) || '', ref: refs[i],
                  colour: cols[i], kind: d.kinds[ki[i]], km: kms[i],
                  service: !!(fl[i] & 1), dup: !!(fl[i] & 2), opShown: o.shown };
```

After:

```js
  for (const [cc, [ids, ns, es, refs, oi, ki, cols, kms, fl, ti]] of Object.entries(d.lines)) {
    for (let i = 0; i < ids.length; i++) {
      const o = ops[oi[i]];
      const l = { id: ids[i], region: cc, name: ns[i],
                  name_en: es[i] || (ne[cc] && ne[cc].ln[ids[i]]) || '', ref: refs[i],
                  colour: cols[i], kind: d.kinds[ki[i]], km: kms[i],
                  service: !!(fl[i] & 1), dup: !!(fl[i] & 2), opShown: o.shown,
                  // The type, where search.json has the `types` column (build_regions).
                  ltype: ti && ti[i] >= 0 ? (d.types || [])[ti[i]] : '' };
```

## 7. runSearch: the type in a line hit

Before:

```js
      + `<div class="s">${esc(h.l.opShown || opName(h.l))} · `
```

After:

```js
      + `<div class="s">${hitType(h.l)}${esc(h.l.opShown || opName(h.l))} · `
```

## build_regions.py: a `types` column in search.json (for the managing session)

Five edits, all in `write_search`. Tested by running the patched function on jp, de and us
into a scratch file: 4,476 lines, the column filled for 1,099 / 2,278 / 911 of them (the same
counts as types.json), one small integer per line. Old apps ignore the tenth column, so `v`
stays 2.

Docstring, before:

```
        {"v": 2, "ops": [[operator, operator_en, shown as], ...], "kinds": [kind, ...],
         "lines": {cc: [ids, names, names_en, refs, ops, kinds, colours, kms, flags]},
                                                 # flags: 1 named train, 2 as operated
```

After:

```
        {"v": 2, "ops": [[operator, operator_en, shown as], ...], "kinds": [kind, ...],
         "types": [type, ...],
         "lines": {cc: [ids, names, names_en, refs, ops, kinds, colours, kms, flags, types]},
                                                 # flags: 1 named train, 2 as operated
                                                 # types: index into "types", -1 for none
                                                 # (tools/line_types.py's types.json)
```

Before:

```python
    ops, opi, kinds, kindi = [], {}, [], {}
```

After:

```python
    # Line types (tools/line_types.py, types.json): a line's main type, by its own id.
    tyj = {}
    for cc in ccs:
        f = DIST / "data" / cc / "types.json"
        if f.exists():
            tyj[cc] = json.loads(f.read_text(encoding="utf-8"))
    ops, opi, kinds, kindi = [], {}, [], {}
    types, typei = [], {}
```

Before:

```python
        kind = intern(kinds, kindi, first("kind"), first("kind"))
```

After:

```python
        kind = intern(kinds, kindi, first("kind"), first("kind"))
        # The first piece with a type (the piece loaded for the hit comes first).
        t = next((tyj[cc][l["id"]] for _, cc, l in ps if l["id"] in tyj.get(cc, {})), None)
        t = t["type"] if isinstance(t, dict) else t
        ty = intern(types, typei, t, t) if t else -1
```

Before:

```python
        cols = out_lines.setdefault(ps[0][1], [[] for _ in range(9)])
        row = [fid, name, "" if name_en == name else name_en, first("ref"), op, kind,
               first("colour"), round(km, 1), flags]
```

After:

```python
        cols = out_lines.setdefault(ps[0][1], [[] for _ in range(10)])
        row = [fid, name, "" if name_en == name else name_en, first("ref"), op, kind,
               first("colour"), round(km, 1), flags, ty]
```

Before:

```python
    doc = {"v": 2, "ops": ops, "kinds": kinds, "lines": out_lines, "stations": out_st}
```

After:

```python
    doc = {"v": 2, "ops": ops, "kinds": kinds, "types": types, "lines": out_lines,
           "stations": out_st}
```

## Publishing and upkeep

- Upload `dist/data/<cc>/types.json` with each country's other files (27 bytes to 100 KB
  each, 0.5 MB in all).
- Rerun `python tools/line_types.py` after any country rebuild (a few minutes for all),
  then `tools/build_regions.py` for the search column.
- After landing: `node tools/lint_map_expressions.js dist/index.html`, and a look at a line
  view in each theme (the muted sentence uses the existing `.muted` style).
