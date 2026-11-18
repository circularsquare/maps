// T-3556: the harness's action layer: what a player does, as plain calls on lon/lat points and visible
// labels, through a small driver (lib/play.mjs is the Playwright one). Nothing here imports Playwright, so
// a later surface (an in-page API, an MCP-style wrapper for outside agents) can put its own driver under
// the same calls. Arguments and results are JSON: [lng, lat] points, labels and text as shown.
//
// The driver:
//   input                         'mouse' or 'touch'
//   see(name, arg) / state(name, arg)   a read from lib/reads.mjs, run in the page
//   move(x, y, { steps })         pointer to page px
//   press(x, y, { shift, ctrl, double })  a click (mouse) at page px
//   tap(x, y)                     a finger's tap at page px
//   control(target, how)          press a control: target { name, role = 'button', within, exact = true, nth, css };
//                                 how { file } answers the file picker it opens
//   fill(target, text)            type into a field (or a typed value's button) and press Enter
//   key(combo), text(s)           keyboard
//   camera({ center, zoom, bearing, pitch })   the view (a player's pan and zoom, done in one step)
//   sleep(ms)

/** @typedef {[number, number]} LngLat */

const PRESS_SETTLE_MS = 120;

/** poll `get` until `ok(value)`; the last value, or throws naming `what` after `timeout` ms */
async function until(d, get, ok, { timeout = 30000, every = 150, what = 'a condition' } = {}) {
  const t0 = Date.now();
  let v;
  for (;;) {
    v = await get();
    if (ok(v)) return v;
    if (Date.now() - t0 > timeout) throw new Error(`timed out after ${timeout} ms waiting for ${what}: ${JSON.stringify(v)?.slice(0, 300)}`);
    await d.sleep(every);
  }
}

const TRACK_TYPES = { regular: 'Regular', hsr: 'High speed', tram: 'Tram' };
const TRACK_COUNTS = { 1: 'Single track', 2: 'Double track', 4: 'Quad track' };
const SNAPS = { streets: 'Snap to streets', real: 'Snap to real world network', roads: 'Snap to roads', beside: 'Snap beside track' };
const TOOL_LABELS = { track: 'Track', station: 'Station', delete: 'Delete', select: 'Select' };
const TABS = { build: 'Build', lines: 'Lines', stations: 'Stations', money: 'Finances', cities: 'Cities', settings: 'Settings', saves: 'Saved games' };
const levelName = (l) => `Level ${l > 0 ? '+' + l : l < 0 ? '−' + -l : '0'}`;

/**
 * The calls on a game through driver `d`: `act` (player input), `see` (what the screen shows), `state` (named
 * game-state reads for asserts, never for outside agents), `toScreen`/`toLngLat` (exact, the map's projection).
 */
export function actions(d) {
  const see = (name, arg) => d.see(name, arg);
  const state = (name, arg) => d.state(name, arg);

  /** page px of a point, refusing one a press would not reach the map at */
  async function at(p, why = 'press') {
    const [s] = await state('toScreen', [p]);
    if (!s.onMap) throw new Error(`can't ${why} at [${p}]: ${s.cover} is there (page px ${Math.round(s.x)}, ${Math.round(s.y)}); move the camera or close the panel`);
    return s;
  }

  /** a press at a lon/lat: a click with the mouse, a tap with a finger */
  async function pressAt(p, { shift = false, ctrl = false } = {}) {
    const s = await at(p);
    if (d.input === 'touch') await d.tap(s.x, s.y);
    else {
      await d.move(s.x, s.y, { steps: 4 });
      await d.press(s.x, s.y, { shift, ctrl });
    }
    await d.sleep(PRESS_SETTLE_MS);
    return s;
  }

  async function mapIdle(timeout = 60000) {
    await until(d, () => state('mapIdle'), Boolean, { timeout, what: 'the map to stop moving' }).catch(() => {});
  }

  /** set a toolbar switch (by its name) to `on`, pressing it only when it differs */
  async function setSwitch(name, on) {
    const tb = await see('toolbar');
    if (!(name in tb.pressed)) throw new Error(`no switch "${name}" on the ${tb.bar || 'tool'} bar; it has ${Object.keys(tb.pressed).join(', ')}`);
    if (tb.pressed[name] !== on) await d.control({ name, within: '.tool-bar' });
    const now = await see('toolbar');
    if (now.pressed[name] !== on) throw new Error(`the "${name}" switch would not turn ${on ? 'on' : 'off'}`);
  }

  const act = {
    /** move the camera to `center` [lng, lat] at `zoom`, then wait for the map to settle */
    async camera(center, zoom, { bearing, pitch } = {}) {
      await d.camera({ center, zoom, bearing, pitch });
      await mapIdle();
    },

    /** open a dock tab by name: build, lines, stations, money, cities, settings (or its label) */
    async tab(name) {
      await d.control({ name: TABS[name] ?? name, within: '#dock nav.tabs' });
    },

    /**
     * Pick a build tool (track, station, delete, select) in the Build tab and set its bar's options, each
     * pressed only when it differs: track `type` (regular, hsr, tram), `tracks` (1, 2, 4), `level` (-3..3),
     * `snap` ({ streets, real, roads, beside }: true or false each), `keepPlacing`, `ramp`; station `oneTrack`.
     */
    async tool(name, opts = {}) {
      const label = TOOL_LABELS[name] ?? name;
      if ((await see('toolbar')).tool !== label) {
        await act.tab('build');
        await d.control({ css: `.tool-pick button:has(.tool-label:text-is("${label}"))` });
      }
      if (name === 'track') {
        if (opts.type) {
          const want = TRACK_TYPES[opts.type] ?? opts.type;
          if ((await see('toolbar')).type !== want) await d.control({ name: want, within: '.track-bar' });
        }
        if (opts.level !== undefined) await setSwitch(levelName(opts.level), true);
        if (opts.tracks) await setSwitch(TRACK_COUNTS[opts.tracks], true);
        for (const [k, on] of Object.entries(opts.snap ?? {})) await setSwitch(SNAPS[k] ?? k, on);
        if (opts.keepPlacing !== undefined) await setSwitch('Keep placing track', opts.keepPlacing);
        if (opts.ramp !== undefined) await setSwitch('Ramp over track you cross', opts.ramp);
      }
      if (name === 'station' && opts.oneTrack !== undefined) {
        await setSwitch(opts.oneTrack ? 'One way station' : 'Two way station', true);
      }
      return see('toolbar');
    },

    /**
     * Draw track through `points` [[lng, lat], ...] with the track tool as it is set (call `tool` first): a
     * press at the first point, then at each next one once its ghost has landed, Shift held to carry the run on.
     * Per segment: whether it was placed, the edges it added, the ghost's length and cost, and the notice shown.
     * `before(i)`, when given, runs with the pointer resting on point i just before its press (a check's own probe
     * of the hover or the ghost); `settleMs` is the rest after the ghost lands.
     */
    async drawTrack(points, { settleMs = 400, timeout = 30000, before: probe = null } = {}) {
      if ((await see('toolbar')).tool !== 'Track') throw new Error('drawTrack wants the track tool in hand: act.tool("track", ...) first');
      if (points.length < 2) throw new Error('drawTrack wants two points or more');
      const segments = [];
      const s0 = await at(points[0]);
      if (d.input === 'touch') await d.tap(s0.x, s0.y);
      else {
        await d.move(s0.x, s0.y, { steps: 3 });
        if (probe) { await d.sleep(500); await probe(0); }
        await d.press(s0.x, s0.y);
      }
      await d.sleep(PRESS_SETTLE_MS);
      for (let i = 1; i < points.length; i++) {
        const s = await at(points[i]);
        const before = await state('counts');
        if (d.input === 'touch') await d.tap(s.x, s.y);
        else await d.move(s.x, s.y, { steps: 8 });
        const ghost = await until(d, () => state('drawing'), (g) => g.settled && !['search', 'fit', 'preview'].includes(g.street) && !['rest', 'search', 'fit', 'preview'].includes(g.real), { timeout, what: 'the ghost' }).catch((e) => ({ error: e.message }));
        await d.sleep(settleMs);
        const g = await state('drawing');
        if (probe) await probe(i);
        const last = i === points.length - 1;
        if (d.input === 'touch') await d.tap(s.x, s.y); // the second tap on the same spot confirms
        else await d.press(s.x, s.y, { shift: !last });
        const done = await until(d, async () => ({ dr: await state('drawing'), c: await state('counts') }), (v) => !v.dr.placing && v.c.edges !== before.edges, { timeout, what: 'the segment to be placed' }).catch(() => null);
        segments.push({ ok: !!done, edges: await state('edgesFrom', { from: before.edges }), ghost: g.preview ?? ghost.preview ?? null, notes: await see('notes') });
        if (!done) break;
      }
      // a run left open ends: Enter with a keyboard, the touch bar's stop button (its cross) with a finger
      if ((await state('drawing')).start) await (d.input === 'touch' ? d.control({ css: '.touch-bar-host button.stop' }) : d.key('Enter'));
      return { ok: segments.length === points.length - 1 && segments.every((s) => s.ok), segments };
    },

    /** the Build tab's Construct blueprints; waits until no piece is left unbuilt */
    async construct({ timeout = 60000 } = {}) {
      await act.tab('build');
      await d.control({ name: 'Construct blueprints', exact: false, within: '#dock' });
      await until(d, () => state('edgesFrom', { from: 0 }), (es) => es.every((e) => e.built !== false), { timeout, what: 'construction' });
    },

    /** place a station with the station tool (as set) at `p`; the new station, or { ok: false, notes } */
    async placeStation(p, { timeout = 15000 } = {}) {
      if ((await see('toolbar')).tool !== 'Station') await act.tool('station');
      const before = await state('counts');
      await pressAt(p);
      if (d.input === 'touch') await pressAt(p);
      const ok = await until(d, () => state('counts'), (c) => c.stations > before.stations, { timeout, what: 'the station' }).then(() => true, () => false);
      if (!ok) return { ok: false, notes: await see('notes') };
      const [s] = await state('stations', { near: p, limit: 1 });
      return { ok: true, ...s };
    },

    /**
     * A new line from the Lines tab's New line button: named `name` in the field it opens (the default kept
     * when none), each of `stops` ([lng, lat] each, pressed at the nearest station's own point) added, then Done.
     * Zoomed out the map draws no stations to press: `zoom` moves the camera to each stop at that zoom first.
     * The line { num, id, name, stops } as the world holds it.
     */
    async newLine(stops, { name = null, zoom = null, timeout = 8000 } = {}) {
      const before = await state('lines');
      await act.tab('lines');
      await d.control({ name: 'New line', within: '#dock .pane' });
      const made = await until(d, () => state('lines'), (ls) => ls.length > before.length, { timeout, what: 'the new line' });
      const line = made.find((l) => !before.some((b) => b.id === l.id));
      // the name field opens focused; Enter keeps what is in it
      await d.sleep(300);
      if (name !== null) await d.fill({ focused: true }, name);
      else await d.key('Enter');
      for (const [k, p] of stops.entries()) {
        const [s] = await state('stations', { near: p, limit: 1 });
        if (!s) throw new Error('no station to add as a stop');
        if (zoom !== null) await act.camera([s.lng, s.lat], zoom);
        await pressAt([s.lng, s.lat]);
        await until(d, () => state('lines'), (ls) => ls.find((l) => l.id === line.id)?.stops.length > k, { timeout, what: `stop ${k + 1} (${s.name})` })
          .catch(async () => { throw new Error(`stop ${k + 1} (${s.name}) was not added; the tool says: ${(await see('notes')).map((n) => n.text).join(' / ')} (zoomed out, no station is drawn to press: pass { zoom: 15 })`); });
      }
      await d.control({ name: 'Done', within: '#tool-notes' });
      return (await state('lines')).find((l) => l.id === line.id);
    },

    /** select what is at `p` with the select tool (a station, a line, a train) and return the panel */
    async selectAt(p) {
      if ((await see('toolbar')).tool) await d.key('Escape');
      await pressAt(p);
      await d.sleep(400);
      return see('panel');
    },

    /** open a line's panel from the Lines tab by its name */
    async openLine(name) {
      await act.tab('lines');
      await d.control({ css: `#dock .pane li:has(.line-name:text-is("${name}"))` });
      await until(d, () => see('panel'), (p) => !!p?.text, { what: `the panel of ${name}` });
      return see('panel');
    },

    /** the open line panel's trains by demand level, typed into each row's value: [high, medium, low] (null leaves one) */
    async setTrains(levels) {
      for (const [i, n] of levels.entries()) if (n !== null && n !== undefined) await d.fill({ name: 'Type trains on the line', within: '#inspector', exact: false, nth: i }, String(n));
      await d.sleep(800);
      return see('line');
    },

    /** press a control by its visible name (or aria-label), within a part of the page (a CSS selector) */
    click: (name, { within, role = 'button', exact = true, nth } = {}) => d.control({ name, within, role, exact, nth }),
    /** type `text` into a field by its label (or a typed value's button) and press Enter */
    type: (name, text, { within, exact = false, nth } = {}) => d.fill({ name, within, exact, nth }, text),
    /** a key or chord, e.g. 'Enter', 'Escape', 'Control+Z' */
    key: (combo) => d.key(combo),
    /** a press at a lon/lat, a click or a tap; `shift` and `ctrl` for the mouse */
    press: pressAt,
  };

  return {
    act,
    see: {
      panel: () => see('panel'),
      line: () => see('line'),
      money: () => see('money'),
      notes: () => see('notes'),
      status: () => see('status'),
      toolbar: () => see('toolbar'),
    },
    state: {
      counts: () => state('counts'),
      lines: () => state('lines'),
      stations: (opts) => state('stations', opts ?? {}),
      edgesFrom: (from = 0) => state('edgesFrom', { from }),
      drawing: () => state('drawing'),
      demand: () => state('demand'),
      selection: () => state('selection'),
      view: () => state('view'),
      money: () => state('money'),
      lastLoad: () => state('lastLoad'),
      trips: (opts) => state('trips', opts ?? {}),
      clock: () => state('clock'),
      bubbles: (opts) => state('bubbles', opts ?? {}),
    },
    /** page px of lon/lat points, each with `onMap` (a press there reaches the map) or `cover` */
    toScreen: (pts) => state('toScreen', pts),
    /** lon/lat of page px points */
    toLngLat: (xys) => state('toLngLat', xys),
    mapIdle,
    until: (get, ok, opts) => until(d, get, ok, opts),
  };
}
