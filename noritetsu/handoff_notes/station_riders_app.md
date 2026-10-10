# Station ridership in the station view: the app patch (APPLIED 2026-10-09)

Applied on the current index.html. One change: `roundRiders` tests `!(n >= 1)` instead of
`n < 1`, so an entry with neither `n` nor `b` reads "Fewer than one" rather than "About NaN".
Seen in headless Chrome, both themes and modes: 新宿 "About 2.7 million a day (2024)", Zürich
Hauptbahnhof "About 450,000 a day (2025)", Amsterdam Centraal "About 160,000 a weekday
(2024)", 서울 "About 340,000 a day (2023-25)" (tooltip naming its three sources), Châtelet -
Les Halles nothing.

Written 2026-10-09 by the station-riders data session, for whichever session holds
`dist/index.html` (another agent was editing it, so this was not applied). The data side is
done: `dist/data/<cc>/riders.json` and `dist/data/riders_sources.json`, made by
`tools/station_riders.py`; method and conventions in `station_riders.md`.

## What it does

One small grey line under the station's name in the station view (the `.big` block), in the
same size and colour as the name's `<small>` line under it:

    About 2.7 million a day (2024)        新宿
    About 450,000 a day (2025)            Zürich HB, and Zürich Hauptbahnhof pointing at it
    About 160,000 a weekday (2024)        Amsterdam Centraal (NS gives an average working day)
    About 340,000 a day (2023-25)         서울: four operators' counts of three different years
    0-4 a day (2025)                      a published or bounded band
    Fewer than one a day (2024)           a country halt with a few passengers a year

- No figure: nothing at all, no "no data" line.
- Rounded to two significant figures (730, 13,000, 450,000), millions with one decimal.
- "a weekday" when every source behind the figure counts an average working day
  (riders_sources.json `per`), otherwise "a day".
- The source on hover (riders_sources.json `label`, e.g. "SBB passenger counts"; several
  joined with "; "), and the year links to the source's page when there is one source.
- `{"at": id}` entries (another id of a split complex) show the figure they point at.
- riders.json is fetched the first time a station of that country is opened, not with the
  country's lines (most visits never open a station; jp's is 370 KB). Then the station view
  re-renders once. A missing file is an empty one. A station in two regions
  (`s.regions`) looks in each.

No new colour (uses `--g8a`, already in both themes), no layer, no paint expression, no
transition. New top-level names: `RIDERS`, `RIDER_SRC`, `RIDER_SRC_P`, `fetchRiders`,
`roundRiders`, `ridersLine` (none existed on 2026-10-09).

## Checked

- Applied to a copy of `dist/index.html` as of 2026-10-09 (the anchors are below; each
  matched exactly once) and `node tools/lint_map_expressions.js` on the copy: "expression
  lint OK (19 layers checked)".
- The snippet run under node against the real riders.json files (fetch read from disk): 新宿
  "About 2.7 million a day (2024)", 池袋's second id g003379 the same through `at`, Zürich
  Hauptbahnhof "About 450,000 a day (2025)", Amsterdam Centraal "About 160,000 a weekday
  (2024)", 서울 "About 340,000 a day (2023-25)", Châtelet - Les Halles nothing, Clapham
  Junction "About 67,000 a day (2024)", Bruxelles-Central, 臺北; one re-render after the
  fetch.
- Not looked at in a browser (the page is served only from dist/). Worth a screenshot of a
  station view in both themes after applying.

## The change

Three insertions. Anchors are the existing lines shown without `+`.

1. CSS, after the `.big small` rule:

```css
  .big small { font-size: 12px; font-weight: 400; color: var(--g8a); display: block; margin-top: 2px; }
+  .riders { font-size: 12px; color: var(--g8a); margin-top: 4px; }
+  .riders a { color: inherit; text-decoration: none; }
```

2. Script, just before `function renderStation(el) {`:

```js
/* RIDERSHIP PER STATION (tools/station_riders.py, station_riders.md): data/<cc>/riders.json is
   {id: {n, year, src, y0?}} (n people getting on + off on an average day; y0 when parts of a
   sum are older), {id: {b, year, src}} for a published band ("20-49"), or {id: {at: id}} for an
   id of a complex whose figure sits on another id. data/riders_sources.json says what each src
   counts. Shown as one small grey line under the station's name, and nothing at all when
   there is no figure. Fetched the first time a station of that country is opened, not with
   the country's lines: most visits never open a station. */
const RIDERS = {};          // cc -> its riders.json once fetched ({} if none); null while coming
let RIDER_SRC = {};         // src key -> {label, name, url, licence, counts, per}
let RIDER_SRC_P = null;
function fetchRiders(cc) {
  if (cc in RIDERS) return;
  RIDERS[cc] = null;
  const get = f => fetch(f).then(r => (r.ok ? r.json() : {})).catch(() => ({}));
  RIDER_SRC_P = RIDER_SRC_P || get('data/riders_sources.json').then(s => { RIDER_SRC = s || {}; });
  Promise.all([get(`data/${cc}/riders.json`), RIDER_SRC_P]).then(([r]) => {
    RIDERS[cc] = r || {};
    const s = VIEW.kind === 'station' && STATIONS[VIEW.id];
    if (s && (s.regions || [s.region]).includes(cc)) render();
  });
}
/* 2 significant figures, millions as "2.7 million": a station count is an estimate and the
   line is meant to be read in passing. */
function roundRiders(n) {
  if (n < 1) return null;
  if (n >= 1e6) return `${(Math.round(n / 1e5) / 10).toLocaleString('en-US')} million`;
  const p = Math.pow(10, Math.max(0, Math.floor(Math.log10(n)) - 1));
  return (Math.round(n / p) * p).toLocaleString('en-US');
}
function ridersLine(id) {
  const st = STATIONS[id];
  if (!st) return '';
  let r = null;
  for (const cc of st.regions || [st.region]) {
    fetchRiders(cc);
    const all = RIDERS[cc];
    if (!all || !all[id]) continue;
    r = all[id].at ? all[all[id].at] : all[id];
    if (r) break;
  }
  if (!r || !r.year) return '';
  const srcs = String(r.src || '').split('+').map(k => RIDER_SRC[k]).filter(Boolean);
  const per = srcs.length && srcs.every(s => s.per === 'weekday') ? 'a weekday' : 'a day';
  let what;
  if (r.b) what = `${r.b} ${per}`;
  else {
    const v = roundRiders(r.n);
    what = v ? `About ${v} ${per}` : `Fewer than one ${per}`;
  }
  const year = r.y0 && r.y0 !== r.year ? `${r.y0}-${String(r.year).slice(2)}` : `${r.year}`;
  const tip = srcs.map(s => s.label || s.name).join('; ');
  const url = srcs.length === 1 && srcs[0].url;
  return `<div class="riders" title="${esc(tip)}">${esc(what)} (`
    + (url ? `<a href="${esc(url)}" target="_blank" rel="noopener">${year}</a>` : year)
    + `)</div>`;
}
```

3. In `renderStation`, one line after the name block:

```js
    + `<h2 class="sec">Station</h2><div class="big" style="font-size:20px">${esc(stName(VIEW.id))}`
    + (stAlt(VIEW.id) ? `<small>${esc(stAlt(VIEW.id))}</small>` : '') + `</div>`
+    + ridersLine(VIEW.id)
    + `<h2 class="sec">Lines here (${lines.length})</h2>`;
```

## Publishing

`riders.json` per country and `riders_sources.json` go up with the other data files
(R2, beside each country's stations.json). Licences that ask for attribution (ORR's OGL,
TfL, SBB, NSW and Victoria's CC BY, Korail's 공공누리 type 1, Taiwan's OGDL) are met by the
hover label and the year's link; an about/credits line naming them would be the fuller way.
