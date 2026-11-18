// T-3556: the harness's reads, run inside the game's page. Each is a self-contained function of one
// JSON argument returning JSON (the driver sends its source to the page), so a later in-page API can
// carry the same list. Two kinds, kept apart:
//   SEE:   what a player can see on screen, as shown (strings verbatim, keyed by their visible label)
//   STATE: named reads of game state for asserts where the screen is not enough; never for outside agents
// The map's own projection (window.tw.map, MapLibre) converts lon/lat and page pixels, so both are exact.

/** @typedef {[number, number]} LngLat */

export const SEE = {
  /** the open panel (#inspector): title, subtitle, label/value rows, tables, alerts, button names, all text */
  panel: () => {
    const el = document.querySelector('#inspector');
    if (!el || !el.textContent.trim()) return null;
    const txt = (e) => (e?.innerText ?? '').replace(/[ \t]+/g, ' ').trim();
    const rows = {};
    for (const li of el.querySelectorAll('ul.rows > li')) {
      const label = txt(li.querySelector(':scope > .grow'));
      if (!label) continue;
      const rest = [...li.children].filter((c) => !c.classList.contains('grow')).map(txt).filter(Boolean).join(' ');
      rows[label] = rest;
    }
    const tables = [...el.querySelectorAll('table')].map((t) => {
      const all = [...t.rows].map((r) => [...r.cells].map(txt));
      return { cls: t.className, head: all[0] ?? [], rows: all.slice(1) };
    });
    // the name in the title row (a line's or a station's), else the row's first word longer than a badge's letter
    const head = el.querySelector('.title');
    const name = head?.querySelector('.name, input');
    return {
      title: name ? (name.value ?? txt(name)) : txt(head).split('\n').find((l) => l.length > 1) ?? '',
      subtitle: txt(el.querySelector('.sub')),
      rows,
      tables,
      alerts: [...el.querySelectorAll('.alerts li, .alert')].map(txt).filter(Boolean),
      buttons: [...el.querySelectorAll('button')].map((b) => (b.getAttribute('aria-label') || txt(b)).split('\n')[0]).filter(Boolean),
      text: txt(el),
    };
  },
  /** the selected line's panel: its schedule table as rows of { demand, trains, headway } and each stop's row */
  line: () => {
    const el = document.querySelector('#inspector');
    const t = el?.querySelector('table.sched');
    if (!t) return null;
    const txt = (e) => (e?.innerText ?? '').replace(/\s+/g, ' ').trim();
    const head = [...t.rows[0].cells].map((c) => txt(c).toLowerCase());
    const schedule = [...t.rows].slice(1).map((r) => Object.fromEntries([...r.cells].map((c, i) => [head[i] || String(i), txt(c)])));
    const stops = [...el.querySelectorAll('[data-stop]')].map((li) => ({ name: txt(li.querySelector('.stop-name')), boardings: txt(li.querySelector('.n.num')), time: txt(li.querySelector('.t')) }));
    return { schedule, stops, red: [...t.querySelectorAll('.bad')].map(txt).filter(Boolean) };
  },
  /** the top bar's money: the figure and its hover */
  money: () => {
    const c = document.querySelector('.cash');
    return c ? { cash: c.innerText.trim(), tip: c.getAttribute('data-tip') ?? '' } : null;
  },
  /** the tool notes at the map's top: the mode, hints and the last notice, line by line */
  notes: () => [...document.querySelectorAll('#tool-notes p')].map((p) => ({ text: p.innerText.replace(/\s+/g, ' ').trim(), bad: p.classList.contains('bad') })),
  /** the map's top row: the demand status tag and the busy tag */
  status: () => (document.querySelector('.sky-status')?.innerText ?? '').replace(/\s+/g, ' ').trim(),
  /** the build tool picked in the Build tab ('' when none is lit), and the track or station bar's switches by name */
  toolbar: () => {
    const on = document.querySelector('.tool-pick button.on .tool-label');
    const bar = document.querySelector('.tool-bar');
    const pressed = {};
    for (const b of bar ? bar.querySelectorAll('button[aria-pressed]') : []) pressed[b.getAttribute('aria-label')] = b.getAttribute('aria-pressed') === 'true';
    const seg = bar ? [...bar.querySelectorAll('.group:not(.levels):not(.tb-count) > button.on')].map((b) => b.innerText.trim()) : [];
    return { tool: on?.innerText.trim() ?? '', bar: bar?.getAttribute('aria-label') ?? '', pressed, type: seg[0] ?? '', price: bar?.querySelector('.price')?.innerText.replace(/\s+/g, ' ').trim() ?? '' };
  },
};

export const STATE = {
  /** the page is up: the world, the tools and the demand client exist */
  ready: () => !!(window.tw?.world?.value && window.tw.tools && window.twDemand?.client),
  /** the camera */
  view: () => { const m = window.tw.map, c = m.getCenter(); return { center: [c.lng, c.lat], zoom: m.getZoom(), bearing: m.getBearing(), pitch: m.getPitch() }; },
  /** page pixels of lon/lat points (the map's own projection plus its container's place), each with what is under it:
   * `onMap` when a press there reaches the map, else `cover` names the element in the way */
  toScreen: (/** @type {LngLat[]} */ pts) => {
    const m = window.tw.map, r = m.getContainer().getBoundingClientRect(), box = m.getCanvasContainer();
    return pts.map(([lng, lat]) => {
      const p = m.project([lng, lat]), x = r.left + p.x, y = r.top + p.y;
      const inView = p.x >= 0 && p.y >= 0 && p.x <= r.width && p.y <= r.height;
      const top = inView ? document.elementFromPoint(x, y) : null;
      const onMap = !!top && (top === m.getCanvas() || box.contains(top) || top.closest('.maplibregl-canvas-container, .tool-marks, .maplibregl-marker') !== null);
      const cover = onMap ? null : !inView ? 'outside the view' : top ? (top.id ? '#' + top.id : top.tagName.toLowerCase() + '.' + [...top.classList].join('.')) + ' ' + (top.closest('[aria-label], [id]')?.getAttribute('aria-label') ?? top.closest('[id]')?.id ?? '') : 'nothing';
      return { x, y, onMap, cover };
    });
  },
  /** lon/lat of page pixels */
  toLngLat: (/** @type {[number, number][]} */ xys) => {
    const m = window.tw.map, r = m.getContainer().getBoundingClientRect();
    return xys.map(([x, y]) => { const p = m.unproject([x - r.left, y - r.top]); return [p.lng, p.lat]; });
  },
  /** lon/lat box to a page clip rectangle, cut to the map; null when none of it is in view */
  clipOf: ({ box, pad = 0 }) => {
    const m = window.tw.map, r = m.getContainer().getBoundingClientRect();
    const [[w, s], [e, n]] = box, ps = [[w, s], [e, s], [e, n], [w, n]].map((q) => m.project(q));
    const xs = ps.map((p) => r.left + p.x), ys = ps.map((p) => r.top + p.y);
    const x0 = Math.max(r.left, Math.min(...xs) - pad), y0 = Math.max(r.top, Math.min(...ys) - pad);
    const x1 = Math.min(r.right, Math.max(...xs) + pad), y1 = Math.min(r.bottom, Math.max(...ys) + pad);
    return x1 > x0 && y1 > y0 ? { x: x0, y: y0, width: x1 - x0, height: y1 - y0 } : null;
  },
  /** what the map is doing: loaded, moving */
  mapIdle: () => window.tw.map.loaded() && !window.tw.map.isMoving(),
  /** how much is in the world */
  counts: () => {
    const w = window.tw.world.value;
    return { lines: w.save.lines.length, stations: w.save.stations.length, edges: w.edges.length, nodes: w.nodes.size, trackKm: Math.round(w.edges.reduce((a, e) => a + e.lengthM, 0) / 100) / 10 };
  },
  /** the edges from index `from` on: id, ends, tracks, kind, built or blueprint */
  edgesFrom: ({ from = 0 }) => window.tw.world.value.edges.slice(from).map((e) => ({ id: e.id, a: e.a, b: e.b, tracks: e.tracks, kind: e.kind, built: e.built ?? null, join: e.join ?? 0, way: e.autoWay ?? e.oneWay ?? 0 })),
  /** stations: id (num), name, lon/lat; `near` [lng, lat] sorts them by distance from it, `limit` caps the list */
  stations: ({ near = null, limit = 1e9 } = {}) => {
    const all = window.tw.world.value.save.stations.map((s) => ({ id: s.num, name: s.name, lng: s.lng, lat: s.lat }));
    if (near) { const k = Math.cos((near[1] * Math.PI) / 180); for (const s of all) s.m = Math.hypot((s.lng - near[0]) * k, s.lat - near[1]) * 111320; all.sort((a, b) => a.m - b.m); }
    return all.slice(0, limit);
  },
  /** lines: number, id, name, stop ids, status, trains needed by level */
  // `stops` the line's plan, as stop edits set it; `served` the part of it its trains run, where only part (T-3474)
  lines: () => { const w = window.tw.world.value; return w.save.lines.map((l) => ({ num: l.num, id: l.id, name: l.name, stops: (l.planStops ?? l.stops).map(Number), ...(l.planStops ? { served: l.stops.map(Number) } : {}), status: w.lineStats[l.id]?.status ?? null, trains: w.lineStats[l.id]?.trainsNeeded ?? null })); },
  /** the selection */
  selection: () => window.tw.selection.value ?? null,
  /** the track tool's state: a segment being placed, its start, whether the ghost asked for has landed, the street snap's phase */
  drawing: () => {
    const t = window.tw.tools;
    return { placing: !!t.placing, start: t.start ? [t.start.lng, t.start.lat] : null, settled: t.wantKey === t.previewKey, preview: t.lastPreview ? { ok: t.lastPreview.ok, lengthM: t.lastPreview.lengthM ?? null, cost: t.lastPreview.cost ?? null } : null, street: t.street?.phase ?? null, real: t.real?.spot?.phase ?? null, pointer: t.pointer ? [t.pointer.lng, t.pointer.lat] : null };
  },
  /** demand: the view's version and whether it is final, the newest solve's version, a solve waiting (see lib.mjs demandFinal) */
  demand: () => {
    const d = window.twDemand, v = d?.view?.(), s = d?.log?.findLast((x) => !d.client.fversions?.has(x.version));
    return { version: v?.version ?? null, final: !!v?.final, newest: s?.version ?? null, waiting: !!d?.client?.waiting, lines: v?.lines?.size ?? 0 };
  },
  /** the demand view's totals: trips and rail trips a day, its round, version and whether final, lines with riders;
   * `byLine` adds each line's riders a day by name */
  trips: ({ byLine = false } = {}) => {
    const v = window.twDemand?.view?.();
    if (!v) return null;
    const r = (x) => (typeof x === 'number' ? Math.round(x) : null);
    const out = { version: v.version ?? null, final: !!v.final, round: v.round ?? null, tripsPerDay: r(v.tripsPerDay), railTripsPerDay: r(v.railTripsPerDay), lines: v.lines?.size ?? 0 };
    if (byLine) {
      const names = new Map(window.tw.world.value.save.lines.map((l) => [String(l.id), l.name]));
      out.byLine = Object.fromEntries([...(v.lines ?? new Map())].map(([id, l]) => [names.get(String(id)) ?? String(id), r(l.ridersPerDay)]));
    }
    return out;
  },
  /** the clock: game time (s), running, held, speed step, rate (game s per real s) */
  clock: () => { const c = window.tw.clock; return { now: c.now(), running: c.running.peek(), held: c.held.peek(), speed: c.speed.peek(), rate: c.rate() }; },
  /** demand bubbles a click would select, scanned over the map every `step` px: one per bubble (its middle), the
   * selection kind and its page px and lon/lat; `kind` keeps one kind; at most `limit`. Needs ?debug=1 (twViews). */
  bubbles: ({ step = 8, kind = null, limit = 20 } = {}) => {
    const m = window.tw.map, c = m.getContainer(), r = c.getBoundingClientRect(), W = c.clientWidth, H = c.clientHeight;
    const groups = new Map();
    for (let y = 4; y < H - 4; y += step) for (let x = 4; x < W - 4; x += step) {
      const h = window.twViews?.click(x, y, false, true);
      if (!h || (kind && h.kind !== kind)) continue;
      // the screen must reach the map there too (a panel over it would take the press)
      const top = document.elementFromPoint(r.left + x, r.top + y);
      if (!top || !(top === m.getCanvas() || m.getCanvasContainer().contains(top) || top.closest('.maplibregl-canvas-container'))) continue;
      const key = JSON.stringify(h);
      const g = groups.get(key) ?? { kind: h.kind, n: 0, sx: 0, sy: 0 };
      g.n++; g.sx += x; g.sy += y;
      groups.set(key, g);
    }
    return [...groups.values()].sort((a, b) => b.n - a.n).slice(0, limit).map((g) => {
      const x = g.sx / g.n, y = g.sy / g.n, p = m.unproject([x, y]);
      return { kind: g.kind, lngLat: [Math.round(p.lng * 1e5) / 1e5, Math.round(p.lat * 1e5) / 1e5], px: [Math.round(r.left + x), Math.round(r.top + y)], hits: g.n };
    });
  },
  /** the game's money as numbers (US$ millions) and its mode */
  money: () => { const m = window.tw.moneyState.value; return m ? { mode: m.mode, cash: m.cash ?? null } : null; },
  /** the last load: from autosave, file or new */
  lastLoad: () => window.tw.client.lastLoad ?? null,
};
