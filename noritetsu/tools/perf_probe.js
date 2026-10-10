/*
 * MAIN-THREAD PROFILE OF THE APP (2026-10-08, app_perf). Opens dist/index.html in a headless
 * Chrome over CDP, records the CPU profile, long tasks and long animation frames of a cold
 * open and of each action (select a line, clear, click on track), and prints the costliest
 * functions. Headless WebGL is swiftshader, so absolute times run ~2-4x a real machine's and
 * map drawing is slower still: compare runs with each other, and read the JS rows.
 *
 * Chrome first, on a private port with its own profile (never 9222; see HANDOFF "To see a
 * change"), and the maps server on 8800 (`python serve.py` at maps/):
 *
 *   "C:/Program Files/Google/Chrome/Application/chrome.exe" --headless=new \
 *     --enable-unsafe-swiftshader --remote-debugging-port=9361 \
 *     --remote-debugging-address=127.0.0.1 --user-data-dir=<temp dir> about:blank
 *
 *   CDP_PORT=9361 node tools/perf_probe.js rides              # once per profile: 100 rides
 *   CDP_PORT=9361 node tools/perf_probe.js world [page] [out.json]
 *
 * Scenarios: rides (makes 100 whole-line rides in jp, gb, ch, de, so a start loads those four
 * as a user's would; the others wait for them), world (no saved camera), europe (z5.5),
 * japan (z5.5, then loads Russia), clicks (real mouse clicks on the WCML and the Tokaido line
 * at z6.5-7 and z11), selects (click to drawn, many times over: see runSelects; env RUNS,
 * SEED, WAIT_MIN/WAIT_SPAN for the wait after each jump, W/H/DPR for the viewport, PROFILE_RUNS
 * for CPU profiles of chosen runs), frames (frame by frame, select / switch / clear: when the
 * dimming, the new line and the whole selection were first drawn; see runFrames; env LINES,
 * SEED). `page` is a file under dist/ ('' for index.html; a copy of an older
 * index.html beside it compares versions on the same data). Full results go to out.json.
 * Stop that Chrome by PID, re-queried by its --user-data-dir, when done.
 */
const fs = require('fs');
const [scenario = 'world', PAGE = '', OUT] = process.argv.slice(2);
const PORT = +(process.env.CDP_PORT || 9361);
const sleep = ms => new Promise(r => setTimeout(r, ms));

// Helpers put in the page before its own scripts run (they read its globals when called).
const HELP = `
window.__LT = []; window.__LOAF = [];
try { new PerformanceObserver(l => { for (const e of l.getEntries()) __LT.push([Math.round(e.startTime), Math.round(e.duration)]); })
  .observe({ type: 'longtask', buffered: true }); } catch (e) {}
try { new PerformanceObserver(l => { for (const e of l.getEntries()) __LOAF.push({ t: Math.round(e.startTime), d: Math.round(e.duration),
  b: Math.round(e.blockingDuration), s: (e.scripts || []).map(s => [s.invoker, s.sourceFunctionName, Math.round(s.duration), s.sourceCharPosition]) }); })
  .observe({ type: 'long-animation-frame', buffered: true }); } catch (e) {}
window.__idle = (ms = 15000) => new Promise(r => { if (!map._frameRequest && map.loaded()) return r();
  const t = setTimeout(r, ms); map.once('idle', () => { clearTimeout(t); r(); }); });
window.__find = (cc, re) => LINES.filter(l => l.region === cc || (l.regions || []).includes(cc))
  .filter(l => re.test(l.name || '') || re.test(l.name_en || '')).sort((a, b) => b.km - a.km)[0];
// Runs fn in a task of its own (as a click would), then waits for a frame and for the map.
window.__time = async fn => {
  let t0, t1, info;
  await new Promise(r => setTimeout(() => { t0 = performance.now(); info = fn(); t1 = performance.now(); r(); }, 0));
  await new Promise(r => requestAnimationFrame(() => setTimeout(r, 0))); const t2 = performance.now();
  await new Promise(r => setTimeout(r, 200)); await __idle(); const t3 = performance.now();
  return Object.assign(info || {}, { syncMs: Math.round(t1 - t0), firstFrameMs: Math.round(t2 - t0), idleMs: Math.round(t3 - t0) });
};
window.__sel = (cc, re) => __time(() => { const L = __find(cc, re); if (!L) return { missing: String(re) };
  showLine(L.id); return { name: L.name_en || L.name, km: Math.round(L.km), secs: L.sections.length }; });
window.__short = cc => __time(() => { const L = LINES.filter(l => l.region === cc && listed(l) && l.km > 1 && l.km < 6)[0];
  showLine(L.id); return { name: L.name_en || L.name, km: Math.round(L.km) }; });
window.__clear = () => __time(() => { goHome(); return {}; });
window.__clickPt = async (cc, re, zoom) => {
  const L = __find(cc, re); const g = await geomFor(L.id);
  const runs = Object.values(g).filter(p => p.length > 4).sort((a, b) => b.length - a.length);
  const run = runs[Math.floor(runs.length / 3)] || runs[0]; const pt = run[Math.floor(run.length / 2)];
  map.jumpTo({ center: pt, zoom }); await sleep(300); await __idle(30000); await sleep(1500); await __idle(30000);
  const p = map.project(pt); return { x: p.x, y: p.y, name: L.name_en || L.name };
  function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
};
/* CLICK TO DRAWN (the 'selects' scenario). Every setData on the page's GeoJSON sources, the
   worker's answer to it ('content'), each tile of them reloaded, each map frame (with whether
   the selection's sources were fully loaded when it was drawn), vector tiles arriving, fetches,
   clicks and the end of the click's task. A selection is drawn at the first frame drawn with
   'sel' and 'selst' loaded after their last setData. */
window.__selInstall = () => {
  if (window.__SI) return; window.__SI = true;
  window.__EV = [];
  const note = (k, i) => __EV.push([performance.now(), k, i]);
  window.__note = note;
  for (const id of ['sel', 'selst', 'stations', 'ridden', 'closed']) {
    const s = map.getSource(id), orig = s.setData.bind(s);
    s.setData = d => { note('set:' + id, (d && d.features || []).length); return orig(d); };
  }
  const SC = id => map.style.sourceCaches[id];
  const selLoaded = () => ['sel', 'selst'].every(id => map.getSource(id).loaded() && SC(id).loaded());
  map.on('sourcedata', e => {
    const id = e.sourceId || '';
    if (['sel', 'selst', 'stations'].includes(id)) {
      if (e.sourceDataType === 'content') note('content:' + id);
      else if (e.tile) note('tile:' + id);
    } else if (e.tile) note('vt', id);
  });
  map.on('render', () => note('render', selLoaded()));
  map.on('idle', () => note('idle'));
  try { new PerformanceObserver(l => { for (const e of l.getEntries())
    note('res', [e.name.replace(/^.*noritetsu\\//, '').split('?')[0], Math.round(e.startTime), Math.round(e.duration)]); })
    .observe({ type: 'resource' }); } catch (e) {}
  window.addEventListener('click', ev => {
    note('click', ev.timeStamp);
    const ch = new MessageChannel(); ch.port1.onmessage = () => note('taskEnd'); ch.port2.postMessage(0);
  }, true);
};
// A list click: showLine(id, null, true) in a task of its own, as a row's onclick runs.
window.__listSel = id => new Promise(r => setTimeout(() => {
  const t0 = performance.now(); __note('click', t0); showLine(id, null, true); __note('taskEnd'); r(t0); }, 0));
window.__state = () => ({ tilesLoaded: map.areTilesLoaded(), tileQueue: TILE_QUEUE.length,
  inflight: Object.values(map.style.sourceCaches).reduce((n, sc) => n + Object.values(sc._tiles)
    .filter(t => t.state === 'loading' || t.state === 'reloading').length, 0),
  loading: [...LOADING.keys()].filter(cc => !LOADED.has(cc)), z: +map.getZoom().toFixed(1) });
/* FRAME BY FRAME (the 'frames' scenario, 2026-10-09): what each drawn map frame showed. Every
   setData on 'sel'/'selst' is numbered and the number stamped on its features (__g), so when a
   tile of either is loaded the page can ask which setData that tile shows; each map frame
   ('render') notes whether the track was muted in it (track-mute-<__DIMCC>'s evaluated
   opacity), the newest selection setData drawn and how many features, and the rAF count. */
window.__frInstall = () => {
  if (window.__FI) return; window.__FI = true;
  window.__FR = []; window.__RAF = 0; window.__DIMCC = null;
  (function tick() { __RAF++; requestAnimationFrame(tick); })();
  let gen = 0;
  const SHOWN = { sel: [0, 0], selst: [0, 0] };
  window.__SHOWN = SHOWN;
  for (const id of ['sel', 'selst']) {
    const s = map.getSource(id), orig = s.setData.bind(s);
    s.setData = d => { const g = ++gen; const fs = (d && d.features) || [];
      for (const f of fs) (f.properties = f.properties || {}).__g = g;
      __FR.push([performance.now(), 'set', id, g, fs.length, __RAF]); return orig(d); };
  }
  const look = id => { const fs = map.querySourceFeatures(id); let g = 0;
    for (const f of fs) if (f.properties.__g > g) g = f.properties.__g; SHOWN[id] = [g, fs.length]; };
  map.on('sourcedata', e => { if ((e.sourceId === 'sel' || e.sourceId === 'selst') && (e.tile || e.sourceDataType === 'content')) look(e.sourceId); });
  const dimOn = () => { const cc = __DIMCC; const L = cc && map.style._layers['track-mute-' + cc];
    if (!L) return null; const p = L.paint.get('line-opacity'); const v = p && p.constantOr ? p.constantOr(-1) : p; return v > 0; };
  // The selection's layers hidden by opacity (selHidden, 2026-10-09): drawn as nothing.
  const hidden = () => { const L = map.style._layers['sel-line']; const p = L && L.paint.get('line-opacity');
    return !!p && (p.constantOr ? p.constantOr(1) : p) === 0; };
  map.on('render', () => __FR.push([performance.now(), 'r', dimOn(), SHOWN.sel[0], SHOWN.sel[1], SHOWN.selst[0], SHOWN.selst[1], __RAF, hidden()]));
  window.addEventListener('click', ev => __FR.push([ev.timeStamp, 'click', __RAF]), true);
};
// A list click or a clear in a task of its own, as a row's onclick or a click on empty map runs.
window.__task = fn => new Promise(r => setTimeout(() => { const t0 = performance.now();
  __FR.push([t0, 'click', __RAF]); fn(); __FR.push([performance.now(), 'taskEnd', __RAF]); r(t0); }, 0));
window.__afterClick = async () => {
  const t0 = window.__clickT0;
  while (!['line', 'pick'].includes(VIEW.kind) && performance.now() - t0 < 30000) await new Promise(r => setTimeout(r, 5));
  const t1 = performance.now(); await new Promise(r => setTimeout(r, 200)); await __idle(); const t2 = performance.now();
  return { view: VIEW.kind, openMs: Math.round(t1 - t0), idleMs: Math.round(t2 - t0) };
};
`;
const cam = c => (c ? `try { localStorage.setItem('noritetsu.camera', JSON.stringify(${JSON.stringify(c)})); } catch (e) {}`
  : `try { localStorage.removeItem('noritetsu.camera'); } catch (e) {}`);
const RIDE_CCS = ['jp', 'gb', 'ch', 'de'];
const READY = `typeof mapReady !== 'undefined' && mapReady && TILES.size > 0 && ${JSON.stringify(RIDE_CCS)}.every(cc => LOADED.has(cc))`;
const act = (name, js, wait = 5000) => ({ name, js, wait });
const click = (name, cc, re, z) => ({ name, click: `__clickPt('${cc}', ${re}, ${z})`, js: '__afterClick()', wait: 4000 });
const TOKAIDO = '/^東海道(本)?線|Tokaido Main/', WCML = '/West Coast Main Line/';
const SCEN = {
  rides: { cam: null, ready: "typeof mapReady !== 'undefined' && mapReady && Object.keys(REGIONS).length > 0", actions: [
    act('rides', `(async () => { if (RIDES.length) return { had: RIDES.length };
      for (const cc of ${JSON.stringify(RIDE_CCS)}) await loadRegion(cc);
      const rides = [];
      for (const cc of ${JSON.stringify(RIDE_CCS)})
        for (const l of LINES.filter(l => l.region === cc && listed(l) && l.km > 5 && l.km < 200).slice(0, 25)) rides.push(wholeRide(l, '2026-01-01'));
      addRides(rides, 'perf'); return { made: rides.length }; })()`, 1000)] },
  world: { cam: null, actions: [act('sel_wcml', `__sel('gb', ${WCML})`), act('clear', '__clear()'),
    act('sel_tokaido', `__sel('jp', ${TOKAIDO})`), act('clear2', '__clear()')] },
  europe: { cam: { c: [10, 50], z: 5.5, b: 0, f: 'de' }, actions: [act('sel_wcml', `__sel('gb', ${WCML})`),
    act('clear', '__clear()'), act('sel_short_ch', "__short('ch')"), act('clear2', '__clear()')] },
  japan: { cam: { c: [137.5, 36.5], z: 5.5, b: 0, f: 'jp' }, actions: [act('sel_tokaido', `__sel('jp', ${TOKAIDO})`),
    act('clear', '__clear()'), act('sel_short_jp', "__short('jp')"), act('clear2', '__clear()'),
    act('load_ru', "__time(() => { loadRegion('ru'); return {}; })", 15000),
    act('sel_ru_long', "__time(() => { const L = LINES.filter(l => l.region === 'ru' && listed(l) && !l.service).sort((a, b) => b.sections.length - a.sections.length)[0]; showLine(L.id); return { name: L.name_en || L.name, km: Math.round(L.km) }; })", 6000),
    act('clear3', '__clear()')] },
  clicks: { cam: { c: [10, 50], z: 5.5, b: 0, f: 'de' }, actions: [click('click_wcml_z6', 'gb', WCML, 6.5),
    act('clear', '__clear()'), click('click_wcml_z11', 'gb', WCML, 11), act('clear2', '__clear()'),
    click('click_tokaido_z7', 'jp', TOKAIDO, 7), act('clear3', '__clear()'),
    click('click_tokaido_z11', 'jp', TOKAIDO, 11), act('clear4', '__clear()')] },
  // Click to drawn, many times over (runSelects): N=<runs> in the environment, default 48.
  selects: { cam: { c: [10, 50], z: 5.5, b: 0, f: 'de' }, selects: true },
  // Frame by frame, select / switch / clear (runFrames): LINES=<n> lines, default 12.
  frames: { cam: { c: [10, 50], z: 5.5, b: 0, f: 'de' }, frames: true },
};

/* THE FRAMES SCENARIO (2026-10-09, Anita: "maybe we hide everything else and then change the
   draw style of the selected line. could we prioritize redrawing the selected line"). For each
   of LINES lines (3 a country in jp, gb, de, us, fixed seed): select it from nothing (a list
   click or a real mouse click on its track, after the map settled or 150 ms after a jump, in
   turn), switch straight to another line of the same country from the list, then clear. For
   each action, from the click: the first map frame drawn muted (or unmuted, for a clear), the
   first drawn with any of the new selection, the first with all of it (the last setData), and
   how many frames showed the map muted with nothing of the new line on it ('bad'; for a clear,
   unmuted with the old line still drawn). */
async function runFrames(ev, rpc, out) {
  const N = +(process.env.LINES || 12);
  await ev('__frInstall(); 0');
  await ev("Promise.all(['us'].map(loadRegion)).then(() => 0)");
  await sleep(4000);
  const lines = await ev(`(() => { let s = ${+(process.env.SEED || 11)};
    const r = () => ((s = (s * 1103515245 + 12345) % 2147483648) / 2147483648);
    const out = [];
    for (const cc of ['jp', 'gb', 'de', 'us']) {
      const ls = LINES.filter(l => l.region === cc && listed(l) && !l.service && l.km > 15 && l.km < 300)
        .sort((a, b) => a.id < b.id ? -1 : 1);
      for (let i = 0; i < ${Math.ceil(N / 4)}; i++) { const l = ls[Math.floor(r() * ls.length)];
        out.push({ id: l.id, cc, name: l.name_en || l.name, km: Math.round(l.km), secs: l.sections.length,
                   file: (partsOf(l)[0].files || [partsOf(l)[0].file || l.id])[0], part: partsOf(l)[0].region,
                   bbox: (REGIONS[cc].view || REGIONS[cc].bbox) }); } }
    return out; })()`);
  // Interleaved by country, so neighbours in the run are not the same country.
  const order = [];
  for (let k = 0; k < Math.ceil(N / 4); k++) for (let c = 0; c < 4; c++) { const L = lines[c * Math.ceil(N / 4) + k]; if (L) order.push(L); }
  out.lines = order.slice(0, N);
  const zooms = [8, 10, 12, 6.5], acts = [];
  const settle = async () => {
    const t0 = Date.now();
    while (Date.now() - t0 < 8000) {
      await sleep(200);
      const ok = await ev(`(() => { const s = __FR.filter(e => e[1] === 'set' && e[2] === 'sel'); const last = s.length ? s[s.length - 1] : null;
        const t = last ? last[0] : 0; if (performance.now() - t < 1200) return false;
        return !last || __FR.some(e => e[1] === 'r' && e[0] > t && e[3] >= last[3]) || performance.now() - t > 4000; })()`);
      if (ok) break;
    }
    return ev(`({ fr: __FR.slice(), lt: __LT.slice(), view: VIEW.kind, id: VIEW.id || null })`);
  };
  for (let i = 0; i < out.lines.length; i++) {
    const L = out.lines[i];
    const mode = Math.floor(i / 2) % 2 ? 'map' : 'list', when = i % 2 ? 'panned' : 'settled';
    const zoom = zooms[i % zooms.length];
    await ev(`__DIMCC = '${L.cc}'; 0`);
    let at = null;
    if (mode === 'map') {
      const p = await ev(`fetch('data/${L.part}/geom/${L.file}.json').then(r => r.json()).then(g => {
        const runs = Object.values(g).filter(p => p.length > 4).sort((a, b) => b.length - a.length);
        const run = runs[Math.floor(runs.length / 3)] || runs[0] || Object.values(g)[0]; return run[Math.floor(run.length / 2)]; })`);
      await ev(`map.jumpTo({ center: ${JSON.stringify(p)}, zoom: ${Math.max(zoom, 10)} }); 0`);
      if (when === 'settled') { await sleep(400); await ev('__idle(20000)'); await sleep(1500); await ev('__idle(20000)'); }
      // A click has to land on drawn track: 150 ms after a jump it often is not yet (us).
      else await sleep(500);
      at = await ev(`(() => { const q = map.project(${JSON.stringify(p)}); return { x: q.x, y: q.y }; })()`);
    } else {
      const [w, s, e, n] = L.bbox;
      const c = [w + (e - w) * 0.4, s + (n - s) * 0.6];
      await ev(`map.jumpTo({ center: ${JSON.stringify(c)}, zoom: ${zoom} }); 0`);
      if (when === 'settled') { await sleep(400); await ev('__idle(20000)'); await sleep(1500); await ev('__idle(20000)'); }
      else await sleep(150);
    }
    // 1. select from nothing
    await ev('__FR.length = 0; __LT.length = 0; 0');
    if (mode === 'map')
      for (const type of ['mouseMoved', 'mousePressed', 'mouseReleased'])
        await rpc('Input.dispatchMouseEvent', { type, x: at.x, y: at.y, button: 'left', clickCount: 1 });
    else await ev(`__task(() => showLine('${L.id}', null, true))`);
    let r = await settle();
    acts.push({ i, act: 'select', line: L.id, cc: L.cc, mode, when, zoom: mode === 'map' ? Math.max(zoom, 10) : zoom, ...r });
    if (r.view !== 'line') { await ev('goHome(); 0'); await sleep(800); continue; }
    // 2. straight to another line of the same country, from the list
    const L2 = out.lines.find(x => x.cc === L.cc && x.id !== L.id) || L;
    await ev('__FR.length = 0; __LT.length = 0; 0');
    await ev(`__task(() => showLine('${L2.id}', null, true))`);
    r = await settle();
    acts.push({ i, act: 'switch', line: L2.id, cc: L.cc, mode: 'list', when: 'settled', ...r });
    // 3. clear, as a click on empty map does
    await sleep(500);
    await ev('__FR.length = 0; __LT.length = 0; 0');
    await ev('__task(() => goHome())');
    r = await settle();
    acts.push({ i, act: 'clear', cc: L.cc, mode: 'task', when: 'settled', ...r });
    await sleep(500);
  }
  out.acts = acts.map(frameSummary);
  out.actsRaw = acts;
}

// One action's frames, in ms from the click and in map frames drawn after it.
function frameSummary(a) {
  const fr = a.fr, click = fr.find(e => e[1] === 'click');
  if (!click) return { i: a.i, act: a.act, missing: 'no click' };
  const t0 = click[0];
  const rs = fr.filter(e => e[1] === 'r' && e[0] > t0);
  const sets = fr.filter(e => e[1] === 'set' && e[0] >= t0 - 1);
  const selSets = sets.filter(e => e[2] === 'sel'), stSets = sets.filter(e => e[2] === 'selst');
  const firstG = selSets.length ? selSets[0][3] : null, lastG = selSets.length ? selSets[selSets.length - 1][3] : null;
  const lastStG = stSets.length ? stSets[stSets.length - 1][3] : null;
  const idx = f => { const k = rs.findIndex(f); return k < 0 ? null : k + 1; };
  const ms = k => (k == null ? null : Math.round(rs[k - 1][0] - t0));
  const clear = a.act === 'clear';
  // The new selection in a frame: a feature of this action's setData (or, clearing, none left).
  const lineIn = e => (clear ? e[4] === 0 || e[8] : !e[8] && firstG != null && e[3] >= firstG && e[4] > 0);
  const allIn = e => (clear ? (e[4] === 0 && e[6] === 0) || e[8]
    : !e[8] && (lastG == null || e[3] >= lastG) && (lastStG == null || e[5] >= lastStG));
  const dimWant = !clear;
  const kDim = idx(e => e[2] === dimWant), kLine = idx(lineIn), kAll = idx(allIn);
  // Muted with none of the new line drawn (clearing: unmuted with the old one still drawn).
  const bad = rs.filter((e, k) => (kLine == null || k + 1 < kLine) && (clear ? e[2] === false && e[4] > 0 && !e[8] : e[2] === true && a.act === 'select')).length;
  return { i: a.i, act: a.act, cc: a.cc, mode: a.mode, when: a.when, view: a.view,
    dim: [ms(kDim), kDim], line: [ms(kLine), kLine], all: [ms(kAll), kAll], bad,
    sets: selSets.map(e => [Math.round(e[0] - t0), e[4]]), raf: kLine ? rs[kLine - 1][7] - click[2] : null,
    lt: (a.lt || []).filter(([s, d]) => s + d >= t0 && s - t0 < 1000).map(([s, d]) => [Math.round(s - t0), d]) };
}

/* THE SELECTS SCENARIO (2026-10-08, Anita: "roughly 25% of the time it takes like ~600ms to
   draw ... feels random"). Twelve lines, three each in jp, gb, de and us (listed, 15-300 km,
   chosen by a fixed seed), selected N times in turn, alternating a real mouse click on the
   line's track (after a jump to it) and a list click (showLine with bring, after a jump to
   somewhere else in its country), at zooms 6.5 / 8 / 10 / 12, after waits of 0.1-2.5 s, so
   some clicks land while tiles are still coming. Each run's timeline (see __selInstall) goes
   to out.json; GC events come from a CDP trace. Prints the distribution of click -> drawn. */
async function runSelects(ev, rpc, onTrace, out) {
  const N = +(process.env.RUNS || 48);
  let seed = +(process.env.SEED || 7);
  const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
  await ev('__selInstall(); 0');
  await ev("Promise.all(['us'].map(loadRegion)).then(() => 0)");
  await sleep(4000);
  const lines = await ev(`(() => { let s = ${Math.floor(rnd() * 1e6)};
    const r = () => ((s = (s * 1103515245 + 12345) % 2147483648) / 2147483648);
    const out = [];
    for (const cc of ['jp', 'gb', 'de', 'us']) {
      const ls = LINES.filter(l => l.region === cc && listed(l) && !l.service && l.km > 15 && l.km < 300)
        .sort((a, b) => a.id < b.id ? -1 : 1);
      for (let i = 0; i < 3; i++) { const l = ls[Math.floor(r() * ls.length)];
        out.push({ id: l.id, cc, name: l.name_en || l.name, km: Math.round(l.km), secs: l.sections.length,
                   file: (partsOf(l)[0].files || [partsOf(l)[0].file || l.id])[0], part: partsOf(l)[0].region,
                   bbox: (REGIONS[cc].view || REGIONS[cc].bbox) }); } }
    return out; })()`);
  out.lines = lines;
  // GC from a trace: only GC events and user-timing marks are kept.
  const gcs = [];
  onTrace(e => { if ((e.name === 'MinorGC' || e.name === 'MajorGC') && e.ph === 'X' && e.dur > 500) gcs.push(e);
                 else if (e.name && e.name.startsWith('sel-run-')) gcs.push(e); });
  await rpc('Tracing.start', { categories: 'disabled-by-default-v8.gc,v8,blink.user_timing', transferMode: 'ReportEvents' });
  const zooms = [6.5, 8, 10, 12], runs = [];
  for (let i = 0; i < N; i++) {
    const L = lines[i % lines.length], mode = i % 2 ? 'list' : 'map';
    const zoom = zooms[Math.floor(i / 2) % zooms.length];
    const wait = +(process.env.WAIT_MIN || 100) + Math.floor(rnd() * +(process.env.WAIT_SPAN || 2400));
    let at = null;
    if (mode === 'map') {
      // A point on the line from its own file (not through GEOM: that would warm the app's cache).
      const p = await ev(`fetch('data/${L.part}/geom/${L.file}.json').then(r => r.json()).then(g => {
        const runs = Object.values(g).filter(p => p.length > 4).sort((a, b) => b.length - a.length);
        const run = runs[Math.floor(runs.length / 3)] || runs[0] || Object.values(g)[0]; return run[Math.floor(run.length / 2)]; })`);
      await ev(`map.jumpTo({ center: ${JSON.stringify(p)}, zoom: ${zoom} }); 0`);
      await sleep(wait);
      at = await ev(`(() => { const q = map.project(${JSON.stringify(p)}); return { x: q.x, y: q.y }; })()`);
    } else {
      const [w, s, e, n] = L.bbox;
      const c = [w + (e - w) * (0.25 + rnd() * 0.5), s + (n - s) * (0.25 + rnd() * 0.5)];
      await ev(`map.jumpTo({ center: ${JSON.stringify(c)}, zoom: ${zoom} }); 0`);
      await sleep(wait);
    }
    const state = await ev('__state()');
    const prof = process.env.PROFILE_RUNS && process.env.PROFILE_RUNS.split(',').includes(String(i));
    if (prof) await rpc('Profiler.start');
    const from = await ev(`performance.mark('sel-run-${i}'); __EV.length = 0; __LT.length = 0; performance.now()`);
    if (mode === 'map')
      for (const type of ['mouseMoved', 'mousePressed', 'mouseReleased'])
        await rpc('Input.dispatchMouseEvent', { type, x: at.x, y: at.y, button: 'left', clickCount: 1 });
    else await ev(`__listSel('${L.id}')`);
    // Settled: 1.5 s with no setData on 'sel' and a loaded frame after the last (or 10 s).
    const t0 = Date.now();
    while (Date.now() - t0 < 10000) {
      await sleep(250);
      const ok = await ev(`(() => { const s = __EV.filter(e => e[1] === 'set:sel'); if (!s.length) return performance.now() - ${from} > 3000;
        const last = s[s.length - 1][0]; return performance.now() - last > 1500 && __EV.some(e => e[1] === 'render' && e[2] && e[0] > last); })()`);
      if (ok) break;
    }
    const r = await ev(`({ ev: __EV.slice(), lt: __LT.slice(), view: VIEW.kind, id: VIEW.id || null })`);
    if (prof) { const p = aggregate((await rpc('Profiler.stop')).profile);
      console.log(`run ${i} profile:\n  ` + p.incl.slice(0, 25).join('\n  ')); }
    runs.push({ i, line: L.id, cc: L.cc, mode, zoom, wait, state, from, ...r });
    await ev('goHome(); 0');
    await sleep(800);
  }
  await rpc('Tracing.end');
  await sleep(3000);
  out.gcs = gcs;
  out.runs = runs.map(r => summarise(r, gcs));
  out.runsRaw = runs;
}

// One run's numbers, all in ms from the click.
function summarise(r, gcs) {
  const ev = r.ev, click = ev.find(e => e[1] === 'click');
  if (!click) return { i: r.i, mode: r.mode, missing: 'no click' };
  const t0 = r.mode === 'map' ? click[2] : click[0];   // a mouse click: the input's own time
  const rel = t => (t == null ? null : Math.round(t - t0));
  const sets = ev.filter(e => e[1] === 'set:sel'), renders = ev.filter(e => e[1] === 'render');
  const drawnAfter = t => { const f = renders.find(e => e[0] > t && e[2]); return f ? f[0] : null; };
  const firstSet = sets[0], lastSet = sets[sets.length - 1];
  const contents = ev.filter(e => e[1] === 'content:sel');
  const firstContent = firstSet && contents.find(e => e[0] > firstSet[0]);
  const res = ev.filter(e => e[1] === 'res' && e[2][1] >= t0 - 5);
  const lt = r.lt.filter(([s, d]) => s + d >= t0);
  const taskEnd = ev.find(e => e[1] === 'taskEnd');
  const firstRender = renders.find(e => e[0] > (taskEnd ? taskEnd[0] : t0));
  const until = lastSet ? drawnAfter(lastSet[0]) : null;
  const vt = ev.filter(e => e[1] === 'vt' && until && e[0] < until).length;
  const mark = gcs.find(e => e.name === `sel-run-${r.i}`);
  // GC pauses between the run's start and its drawn frame: [main thread, other (the worker)].
  let gcMs = null;
  if (mark && until) {
    const a = mark.ts, b = mark.ts + (until - r.from + 50) * 1000;
    const sum = f => Math.round(gcs.filter(e => e.ph === 'X' && e.ts + e.dur >= a && e.ts <= b && f(e))
      .reduce((n, e) => n + e.dur, 0) / 1000);
    gcMs = [sum(e => e.tid === mark.tid), sum(e => e.tid !== mark.tid)];
  }
  return { i: r.i, line: r.line, cc: r.cc, mode: r.mode, zoom: r.zoom, wait: r.wait, view: r.view, state: r.state,
    syncMs: taskEnd ? rel(taskEnd[0]) : null, firstFrameMs: firstRender ? rel(firstRender[0]) : null,
    sets: sets.map(e => rel(e[0])), firstDrawnMs: firstSet ? rel(drawnAfter(firstSet[0])) : null,
    drawnMs: rel(until), workerAckMs: firstSet && firstContent ? Math.round(firstContent[0] - firstSet[0]) : null,
    stationsSets: ev.filter(e => e[1] === 'set:stations' && (!until || e[0] < until)).length, vtBeforeDrawn: vt,
    fetches: res.map(([, , x]) => [x[0], x[1] - Math.round(t0), x[2]]),
    longTasks: lt.map(([s, d]) => [Math.round(s - t0), d]), gcMs };
}
function dist(xs) {
  const v = xs.filter(x => x != null).sort((a, b) => a - b), q = p => v[Math.min(v.length - 1, Math.floor(p * v.length))];
  return v.length ? { n: v.length, median: q(0.5), p75: q(0.75), p90: q(0.9), max: v[v.length - 1] } : { n: 0 };
}

// The CPU profile by function: self time, and inclusive time (each function once per sample).
function aggregate(profile) {
  const nodes = new Map(profile.nodes.map(n => [n.id, n]));
  const parent = new Map();
  for (const n of profile.nodes) for (const c of n.children || []) parent.set(c, n.id);
  const key = n => `${n.callFrame.functionName || '(anon)'} ${(n.callFrame.url || '').split('/').pop().split('?')[0]}:${n.callFrame.lineNumber + 1}`;
  const self = new Map(), incl = new Map(), stacks = new Map();
  let busy = 0;
  const stackOf = id => {
    if (!stacks.has(id)) { const s = new Set(); for (let x = id; x != null; x = parent.get(x)) s.add(key(nodes.get(x))); stacks.set(id, s); }
    return stacks.get(id);
  };
  for (let i = 0; i < profile.samples.length; i++) {
    const dt = (profile.timeDeltas[i + 1] || 0) / 1000, n = nodes.get(profile.samples[i]);
    if (n.callFrame.functionName === '(idle)') continue;
    busy += dt;
    self.set(key(n), (self.get(key(n)) || 0) + dt);
    for (const k of stackOf(profile.samples[i])) incl.set(k, (incl.get(k) || 0) + dt);
  }
  const top = (m, n) => [...m].sort((a, b) => b[1] - a[1]).slice(0, n).map(([k, v]) => `${v.toFixed(0).padStart(6)} ms  ${k}`);
  return { busyMs: Math.round(busy), self: top(self, 20), incl: top(incl, 30).slice(1) };
}
function tasks(lt, t0) {
  const t = lt.filter(([s]) => s >= t0);
  return { n: t.length, maxMs: Math.max(0, ...t.map(x => x[1])),
           tbtMs: t.reduce((n, [, d]) => n + Math.max(0, d - 50), 0),
           lastEndMs: Math.round(Math.max(t0, ...t.map(([s, d]) => s + d)) - t0) };
}

(async () => {
  const S = SCEN[scenario];
  if (!S) throw new Error(`no scenario ${scenario}: ${Object.keys(SCEN).join(', ')}`);
  const page = (await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json()).find(t => t.type === 'page');
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
  let id = 1, traceSink = null;
  const pending = new Map(), logs = [];
  ws.addEventListener('message', ev => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) { const [res, rej, what] = pending.get(m.id); pending.delete(m.id);
      m.error ? rej(new Error(`${what}: ${m.error.message}`)) : res(m.result); }
    if (m.method === 'Tracing.dataCollected' && traceSink) for (const e of m.params.value) traceSink(e);
    if (m.method === 'Runtime.exceptionThrown')
      logs.push('EXCEPTION ' + (m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text));
  });
  const rpc = (method, params = {}) => new Promise((res, rej) => {
    const i = id++; pending.set(i, [res, rej, method]); ws.send(JSON.stringify({ id: i, method, params }));
  });
  const ev = async js => {
    const r = await rpc('Runtime.evaluate', { expression: js, awaitPromise: true, returnByValue: true });
    return r.exceptionDetails ? { error: r.exceptionDetails.exception?.description || r.exceptionDetails.text } : r.result.value;
  };
  for (const m of ['Runtime.enable', 'Page.enable', 'Profiler.enable', 'Network.enable']) await rpc(m);
  // A rebuilt data file must not come from the cache, nor an edited page.
  await rpc('Network.setCacheDisabled', { cacheDisabled: true });
  await rpc('Emulation.setDeviceMetricsOverride', { width: +(process.env.W || 1400), height: +(process.env.H || 900),
    deviceScaleFactor: +(process.env.DPR || 1), mobile: false });
  await rpc('Page.navigate', { url: 'about:blank' });
  await sleep(300);
  const { identifier } = await rpc('Page.addScriptToEvaluateOnNewDocument', { source: cam(S.cam) + HELP });
  await rpc('Profiler.setSamplingInterval', { interval: 250 });
  await rpc('Profiler.start');
  await rpc('Page.navigate', { url: `http://localhost:8800/noritetsu/${PAGE}?v=${Date.now()}` });
  const t0 = Date.now();
  let readyMs = null;
  while (Date.now() - t0 < 90000) {
    await sleep(250);
    if ((await ev(S.ready || READY).catch(() => false)) === true) { readyMs = Date.now() - t0; break; }
  }
  await sleep(6000);
  const out = { scenario, page: PAGE || 'index.html', readyMs };
  out.open = { tasks: tasks(await ev('__LT'), 0), profile: aggregate((await rpc('Profiler.stop')).profile),
               frames: (await ev('__LOAF')).filter(f => f.b > 100) };
  await rpc('Page.removeScriptToEvaluateOnNewDocument', { identifier });
  if (S.frames) {
    await runFrames(ev, rpc, out);
    out.logs = logs;
    if (OUT) fs.writeFileSync(OUT, JSON.stringify(out, null, 1));
    const as = out.acts.filter(a => !a.missing && (a.act === 'clear' || a.view === 'line'));
    console.log(`frames on ${out.page}: ${out.acts.length} actions`);
    for (const act of ['select', 'switch', 'clear']) {
      const xs = as.filter(a => a.act === act);
      console.log(`${act.padEnd(7)} n ${xs.length}  dim ms ${JSON.stringify(dist(xs.map(a => a.dim[0])))}`);
      console.log(`${''.padEnd(7)} line ms ${JSON.stringify(dist(xs.map(a => a.line[0])))}  all ms ${JSON.stringify(dist(xs.map(a => a.all[0])))}`);
      console.log(`${''.padEnd(7)} line later than dim (frames): ${xs.map(a => (a.line[1] != null && a.dim[1] != null ? a.line[1] - a.dim[1] : 'na')).join(' ')}  bad frames: ${xs.map(a => a.bad).join(' ')}`);
    }
    for (const a of out.acts) console.log(JSON.stringify(a));
    if (logs.length) console.log(logs.slice(0, 10).join('\n'));
    ws.close();
    return;
  }
  if (S.selects) {
    await runSelects(ev, rpc, f => { traceSink = f; }, out);
    out.logs = logs;
    if (OUT) fs.writeFileSync(OUT, JSON.stringify(out, null, 1));
    // A map click that landed before the track was drawn hits nothing (view 'home'): not a selection.
    const rs = out.runs.filter(r => r.drawnMs != null && r.view === 'line');
    console.log(`selects on ${out.page}: ${out.runs.length} runs, ${rs.length} opened a line`);
    for (const k of ['syncMs', 'firstFrameMs', 'firstDrawnMs', 'drawnMs'])
      for (const mode of ['all', 'map', 'list'])
        console.log(`${k.padEnd(13)} ${mode.padEnd(4)} ${JSON.stringify(dist(rs.filter(r => mode === 'all' || r.mode === mode).map(r => r[k])))}`);
    for (const r of out.runs) console.log(JSON.stringify({ i: r.i, cc: r.cc, mode: r.mode, z: r.zoom, view: r.view,
      tl: r.state && r.state.tilesLoaded, inf: r.state && r.state.inflight, sync: r.syncMs, f1: r.firstFrameMs, d1: r.firstDrawnMs, d: r.drawnMs,
      ack: r.workerAckMs, sets: r.sets, st: r.stationsSets, vt: r.vtBeforeDrawn, gc: r.gcMs,
      lt: r.longTasks, fe: (r.fetches || []).filter(f => !/\.pbf|pmtiles|openfreemap/.test(f[0])).slice(0, 8) }));
    if (logs.length) console.log(logs.slice(0, 10).join('\n'));
    ws.close();
    return;
  }
  for (const a of S.actions) {
    await ev('__LT.length = 0; __LOAF.length = 0; 0');
    const at = await ev('performance.now()');
    await rpc('Profiler.start');
    let result;
    if (a.click) {
      const pt = await ev(a.click);
      await ev('window.__clickT0 = performance.now(); 0');
      for (const type of ['mouseMoved', 'mousePressed', 'mouseReleased'])
        await rpc('Input.dispatchMouseEvent', { type, x: pt.x, y: pt.y, button: 'left', clickCount: 1 });
      result = { at: pt, ...(await ev(a.js)) };
    } else result = await ev(a.js);
    await sleep(a.wait);
    out[a.name] = { result, tasks: tasks(await ev('__LT'), at), profile: aggregate((await rpc('Profiler.stop')).profile) };
  }
  out.logs = logs;
  if (OUT) fs.writeFileSync(OUT, JSON.stringify(out, null, 1));
  const line = (k, v) => console.log(`${k.padEnd(16)} longest ${String(v.tasks.maxMs).padStart(5)} ms  blocking ${String(v.tasks.tbtMs).padStart(5)} ms  ` +
    `last long task ends ${v.tasks.lastEndMs} ms  ${v.result ? JSON.stringify(v.result) : ''}`);
  console.log(`${scenario} on ${out.page}: ready ${readyMs} ms`);
  line('open', out.open);
  console.log('  ' + out.open.profile.incl.slice(0, 15).join('\n  '));
  for (const a of S.actions) { line(a.name, out[a.name]); console.log('  ' + out[a.name].profile.incl.slice(0, 6).join('\n  ')); }
  if (logs.length) console.log(logs.slice(0, 10).join('\n'));
  ws.close();
})().catch(e => { console.error(e.stack || e.message); process.exit(1); });
