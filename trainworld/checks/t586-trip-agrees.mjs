// T-586: a far-end bubble's hover and the trip explorer opened from it agree. In New York (the real
// save) and Mumbai (a ?new=1 game with T-292's Western line test), for bubble pairs (a selected
// bubble and two of its far-end bubbles, homes and destinations), under every purpose filter and on
// the weekday, the weekend and the average day: the hover's train, walk and drive shares (its mode
// bar's widths), the explorer's headline (its rows' model shares, and its bar's widths against the
// worker's rounded to whole trips, T-911) and the worker's own numbers
// (the selection's far end, `flows`, summed over the far bubble's cells here) agree within 1 point,
// and the two trip counts match. Writes artifacts/trainworld/t586-agree.json, and with SHOTS=1 the
// hover and the explorer side by side, light and dark, as artifacts/trainworld/t586-*.png.
//   node trainworld/checks/t586-trip-agrees.mjs [url]   (from the maps root; default localhost:6196)
// CITIES=nyc,mum picks the cities; SHOTS_ONLY=1 takes the screenshots alone. Prints PASS or FAIL
// lines and exits 1 on failure. About eight minutes with both cities.
import { readFile, writeFile } from 'node:fs/promises';
import { launchChrome, importSave, demandFinal, REAL_SAVE, ANGLE } from './lib.mjs';

const BASE = (process.argv[2] ?? 'http://localhost:6196/').replace(/\/?$/, '/');
const CITIES = (process.env.CITIES ?? 'nyc,mum').split(',');
const FILTERS = ['all', 'work', 'school', 'shopping', 'social', 'medical', 'airport'];
const DAYS = ['weekday', 'weekend', 'average'];
const TOL = 0.01;
const OUT = 'artifacts/trainworld/';
const MUMBAI = {
  centre: [72.8428, 19.0192], view: [72.835, 19.0], zoom: 13,
  line: [['Churchgate', 72.8272, 18.9355], ['Marine Lines', 72.8237, 18.9458], ['Charni Road', 72.8182, 18.9518],
    ['Grant Road', 72.8158, 18.9634], ['Mumbai Central', 72.8193, 18.9696], ['Mahalaxmi', 72.8242, 18.9825],
    ['Lower Parel', 72.8303, 18.9957], ['Prabhadevi', 72.8359, 19.0075], ['Dadar', 72.8428, 19.0192],
    ['Matunga Road', 72.8467, 19.0278], ['Mahim', 72.8466, 19.0407], ['Bandra', 72.8402, 19.0550]],
};

const { ctx, close } = await launchChrome({ prefix: 'tw-t586-', args: ANGLE });
const results = [], rows = [];
const report = (name, ok, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
const waitFinal = (page, after = -1) => demandFinal(page, { strict: false, after, timeout: 900000 });

async function open(page, city) {
  await page.goto(`${BASE}?debug=1&new=1&opened=all&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twViews && window.twDemand?.client, undefined, { timeout: 240000 });
  if (city === 'nyc') {
    const bytes = await readFile(REAL_SAVE);
    await importSave(page, bytes);
    await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
    await waitFinal(page);
    await page.evaluate(() => void window.tw.map.jumpTo({ center: [-73.99, 40.735], zoom: 13 }));
  } else {
    await page.evaluate((ll) => void window.tw.map.jumpTo({ center: ll, zoom: 11.2 }), MUMBAI.centre);
    await page.waitForFunction(() => window.twDemand.activeCity.value === 'mum', undefined, { timeout: 300000 });
    const before = await page.evaluate(() => window.twDemand.view()?.version ?? -1);
    await page.evaluate(async (stops) => {
      const t = window.tw, run = async (op) => { const r = await t.client.edit(op); if (!r.ok) throw Error(JSON.stringify({ op, r })); return r; };
      const pts = stops.map((s) => t.lngLatToPack(s[1], s[2]));
      const tail = (a, b) => { const dx = a.x - b.x, dy = a.y - b.y, l = Math.hypot(dx, dy); return { x: a.x + 300 * dx / l, y: a.y + 300 * dy / l }; };
      const path = [tail(pts[0], pts[1]), ...pts, tail(pts.at(-1), pts.at(-2))];
      const nodes = [];
      for (let i = 1; i < path.length; i++) {
        const from = i === 1 ? { kind: 'free', x: path[0].x, y: path[0].y, level: -2 } : { kind: 'node', node: nodes.at(-1) };
        await run({ op: 'route', from, to: { kind: 'free', x: path[i].x, y: path[i].y, level: -2 }, pis: [], single: false });
        const e = t.world.value.edges.at(-1);
        if (i === 1) nodes.push(e.a);
        nodes.push(e.b);
      }
      const at = nodes.slice(1, -1);
      for (let i = 0; i < stops.length; i++) await run({ op: 'addStation', at: { kind: 'node', node: at[i] }, platform: 160, name: stops[i][0] });
      const line = (await run({ op: 'addLine', stops: at, name: 'Western line test', colour: '#e4002b' })).line;
      await run({ op: 'setTrainCount', line, trains: [10, 10, 5] });
      await run({ op: 'constructAll' });
    }, MUMBAI.line);
    await waitFinal(page, before);
    await page.evaluate((m) => void window.tw.map.jumpTo({ center: m.view, zoom: m.zoom }), MUMBAI);
  }
  await page.evaluate(() => { const t = window.tw; t.display.value = { ...t.display.value, commuters: true, commuterEnd: 'home' }; });
  // the app's own modules, for the synchronous waits below
  await page.evaluate(async () => { window.__t586 = { trip: await import('/src/game/trip.ts'), cv: await import('/src/map/commuterView.ts') }; });
  await page.waitForTimeout(4000);
}

/** In the page: set the end, the purpose filter and the day. */
async function setView(page, end, filter, day) {
  await page.evaluate(async ([end, filter, day]) => {
    const ui = await import('/src/game/ui.ts');
    const t = window.tw;
    t.display.value = { ...t.display.value, commuters: true, commuterEnd: end, dayShown: day };
    ui.purposeFilter.value = filter;
  }, [end, filter, day]);
}

/** Bubbles on screen at the city's level: their screen centres (CSS px, map frame) and totals. */
const bubblesOnScreen = (page) => page.evaluate(() => {
  const t = window.tw, pr = window.twViews.stats.commuters.probe?.();
  if (!pr) return [];
  const z = t.map.getZoom(), lv = z >= 13.6 ? 0 : z >= 11.8 ? 1 : z >= 10 ? 2 : 3;
  const L = pr.levels[lv], W = t.map.getContainer().clientWidth, H = t.map.getContainer().clientHeight, out = [];
  for (let i = 0; i < L.group.length; i++) {
    const p = t.map.project(t.toLngLat(L.discs[i * 8], L.discs[i * 8 + 1], pr.anchor));
    if (p.x > 380 && p.x < W - 380 && p.y > 120 && p.y < H - 120) out.push({ x: p.x, y: p.y, r: L.discs[i * 8 + 2] });
  }
  return out;
});

/** The far bubbles drawn for the selection: group, screen centre, the hover's three numbers. */
const farOnScreen = (page) => page.evaluate(() => {
  const t = window.tw, f = window.twViews.stats.commuters.farProbe?.();
  if (!f) return [];
  const W = t.map.getContainer().clientWidth, H = t.map.getContainer().clientHeight, out = [];
  for (let i = 0; i < f.group.length; i++) {
    const p = t.map.project(t.toLngLat(f.discs[i * 8], f.discs[i * 8 + 1], f.anchor));
    if (p.x > 20 && p.x < W - 340 && p.y > 60 && p.y < H - 40) out.push({ group: f.group[i], x: p.x, y: p.y, n: f.train[i] + f.walk[i] + f.drive[i] });
  }
  return out;
});

const shares = (r, w, d) => { const n = r + w + d; return n > 0 ? [r / n, w / n, d / n] : [0, 0, 0]; };
const off = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])));
// T-911: the explorer's bar is in whole trips (game/trip.ts `wholePeople`, `peopleOf`): largest remainder
const whole = (s, n) => {
  const N = n > 0 ? Math.max(1, Math.round(n)) : 0, t = s.reduce((a, b) => a + b, 0);
  if (!N || !t) return s;
  const raw = s.map((x) => (x / t) * N), out = raw.map(Math.floor);
  const order = raw.map((_, i) => i).sort((a, b) => raw[b] - out[b] - (raw[a] - out[a]) || s[b] - s[a]);
  const left = N - out.reduce((a, b) => a + b, 0);
  for (let k = 0; k < left; k++) out[order[k]]++;
  return out.map((p) => p / N);
};

/** One far bubble under the setting: the hover, the click, the explorer and the worker's sums. */
async function pair(page, sel, far, filter) {
  const hover = await page.evaluate(([x, y]) => window.twViews.hover(x, y), [far.x, far.y]);
  if (!hover || !hover.some((l) => /Click to explore/.test(l))) return { skip: 'no far-bubble hover there' };
  // the bar's flex-grow by colour, in the scheme in force (map/modeBar.ts; walking is yellow under Train share, T-833)
  const cols = await page.evaluate(async () => { const c = (await import('/src/game/palette.ts')).modeColours(); return [c.train, c.walk, c.drive]; });
  const grow = cols.map((c) => { const m = hover[1].match(new RegExp(`flex-grow:([0-9.e+-]+);background:${c.replace(/[#()]/g, (s) => '\\' + s)}`)); return m ? Number(m[1]) : 0; });
  const hoverCount = hover[0].match(/^([0-9.,]+k?)/)?.[1];
  const box = await page.evaluate(() => window.tw.map.getContainer().getBoundingClientRect().toJSON());
  await page.mouse.click(box.left + far.x, box.top + far.y);
  await page.waitForFunction(() => { const s = window.tw.selection.value; return s?.kind === 'trip'; }, undefined, { timeout: 10000 });
  await page.waitForFunction(() => { const v = window.__t586.trip.tripView.value; return v && !v.asking && v.trip && document.querySelector('#inspector .trip-modes'); }, undefined, { timeout: 120000, polling: 250 });
  const ex = await page.evaluate(async ([sel, filter]) => {
    const m = await import('/src/game/trip.ts');
    // settled: answered, and the inspector drawn from that answer
    let v = null;
    for (let k = 0; k < 600; k++) {
      v = m.tripView.value;
      if (v && !v.asking && v.trip) { await new Promise((r) => requestAnimationFrame(() => setTimeout(r, 50))); if (m.tripView.value === v) break; }
      await new Promise((r) => setTimeout(r, 100));
    }
    if (!v?.trip) return null;
    const places = m.tripOpenedFrom.value?.places;
    const segs = [...document.querySelectorAll('#inspector .trip-bar .seg')].map((s) => ({ cls: s.className, w: parseFloat(s.style.width) / 100 }));
    const bar = ['seg-train', 'seg-walk', 'seg-drive'].map((c) => segs.find((s) => s.cls.includes(c))?.w ?? 0);
    const rowsEl = [...document.querySelectorAll('#inspector .trip-modes .rows li')].map((li) => ({ name: li.querySelector('.grow')?.textContent, tip: li.getAttribute('data-tip') }));
    const line = document.querySelector('#inspector .trip-modes')?.previousElementSibling?.textContent ?? '';
    // the worker's numbers: the selection's far end, grouped as the far bubbles draw it (T-909),
    // summed here over the far bubble's cells
    const raw = await window.twDemand.flows(sel.end, Uint32Array.from(sel.cells));
    const xy = (await (await import('/src/game/demand.ts')).cityCells())?.xy;
    const f = raw && xy ? (await import('/src/game/farChunks.ts')).dealFlows(raw, filter, xy) : raw;
    const inFar = new Set(places?.far ?? []);
    return { bar, rows: rowsEl, line, purpose: v.purpose, period: v.busiest, walkMin: v.trip.periods.find((p) => p)?.walkMin, f: f && {
      all: (() => { let r = 0, w = 0, d = 0; f.cells.forEach((c, i) => { if (inFar.has(c)) { r += f.farRail[i]; w += f.farWalk[i]; d += f.farDrive[i]; } }); return [r, w, d]; })(),
      by: Object.fromEntries([...(f.byPurpose ?? new Map())].map(([k, fe]) => { let r = 0, w = 0, d = 0; fe.cells.forEach((c, i) => { if (inFar.has(c)) { r += fe.farRail[i]; w += fe.farWalk[i]; d += fe.farDrive[i]; } }); return [k, [r, w, d]]; })),
    } };
  }, [sel, filter]);
  if (!ex) return { skip: 'the explorer did not settle' };
  const worker = !ex.f ? null : filter === 'all' ? ex.f.all : ex.f.by[filter] ?? [0, 0, 0];
  // T-911: the rows' hovers carry the model's exact share after "Model:"
  const rowShare = (n) => { const r = ex.rows.find((x) => x.name === n); const m = r?.tip?.match(/Model: ([0-9.]+)%/) ?? r?.tip?.match(/: ([0-9.]+)%/); return m ? Number(m[1]) / 100 : null; };
  return { hover: shares(...grow), hoverCount, explorer: ex.bar, rows: { train: rowShare('Train'), walk: rowShare('Walk'), drive: rowShare('Drive') }, line: ex.line, worker: worker && shares(...worker), workerN: worker && worker.reduce((a, b) => a + b, 0), purpose: ex.purpose, period: ex.period, walkMin: ex.walkMin };
}

try {
  for (const city of CITIES) {
    const page = await ctx.newPage();
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await open(page, city);
    // three selections: two home bubbles and one destination bubble, near the view's centre
    const picks = [];
    for (const end of ['home', 'home', 'work']) {
      await setView(page, end, 'all', 'weekday');
      await page.evaluate(() => { window.tw.selection.value = null; });
      await page.waitForTimeout(2500);
      const bs = (await bubblesOnScreen(page)).sort((a, b) => b.r - a.r);
      const taken = new Set(picks.map((p) => p.key));
      for (const b of bs) {
        const s = await page.evaluate(([x, y]) => window.twViews.click(x, y, false), [b.x, b.y]);
        if (s?.kind === 'commuters' && !taken.has(s.cells.join(','))) { picks.push({ end, sel: s, key: s.cells.join(',') }); break; }
      }
    }
    report(`${city}: three bubble selections`, picks.length === 3, `(${picks.map((p) => `${p.end} ${p.sel.cells.length} cells`).join(', ')})`);
    for (const [k, p] of process.env.SHOTS_ONLY === '1' ? [] : picks.entries()) {
      // two far bubbles: the busiest under all purposes on a weekday, and a mid-sized one
      await setView(page, p.end, 'all', 'weekday');
      await page.evaluate((s) => { window.tw.selection.value = s; }, p.sel);
      await page.waitForFunction(() => window.twViews.stats.commuters.farProbe?.(), undefined, { timeout: 120000, polling: 250 });
      await page.waitForTimeout(500);
      const fs = (await farOnScreen(page)).sort((a, b) => b.n - a.n);
      const groups = [fs[0], fs[Math.min(fs.length - 1, 8)]].filter(Boolean).map((f) => f.group);
      for (const group of new Set(groups)) {
        for (const day of DAYS) for (const filter of FILTERS) {
          await setView(page, p.end, filter, day);
          await page.evaluate((s) => { window.tw.selection.value = { ...s }; }, p.sel);
          const ok = await page.waitForFunction((filter) => { const c = window.__t586.cv.commuterPick.value; return c?.far && c.sel === window.tw.selection.value && c.purpose === filter && window.twViews.stats.commuters.farProbe?.(); }, filter, { timeout: 60000, polling: 250 }).then(() => true, () => false);
          // a purpose that does not reach that bubble (medical and airport trips are sparse): the
          // purpose's own busiest far bubble instead
          const on = ok ? await farOnScreen(page) : [];
          const far = on.find((f) => f.group === group) ?? on.sort((a, b) => b.n - a.n)[groups.indexOf(group)] ?? null;
          if (!far) { rows.push({ city, sel: k, group, day, filter, skip: ok ? 'no far bubble on screen' : 'no far end' }); continue; }
          const r = await pair(page, p.sel, far, filter);
          rows.push({ city, sel: k, end: p.end, group: far.group, own: far.group !== group, day, filter, ...r });
          await page.evaluate(() => { window.tw.selection.value = null; });
        }
      }
    }
    const mine = rows.filter((r) => r.city === city && !r.skip && r.worker);
    const bad = mine.filter((r) => off(r.hover, r.worker) > TOL || off(r.explorer, whole(r.worker, r.workerN)) > TOL || (r.rows.train !== null && Math.abs(r.rows.train - r.worker[0]) > TOL) || (r.rows.drive !== null && Math.abs(r.rows.drive - r.worker[2]) > TOL) || (r.rows.walk !== null && Math.abs(r.rows.walk - r.worker[1]) > TOL));
    const worst = Math.max(0, ...mine.map((r) => Math.max(off(r.hover, r.worker), off(r.explorer, whole(r.worker, r.workerN)))));
    report(`${city}: hover, explorer and worker shares within 1 point`, mine.length > 0 && bad.length === 0, `(${mine.length} pair settings, ${rows.filter((r) => r.city === city && r.skip).length} skipped, worst ${(100 * worst).toFixed(2)} points${bad.length ? `, ${bad.length} off: ${JSON.stringify(bad[0])}` : ''})`);
    // under half a trip the explorer says "Under 1" (T-911)
    const counts = mine.filter((r) => !r.line.startsWith(r.hoverCount + ' ') && !(r.hoverCount === '0' && r.line.startsWith('Under 1 ')));
    report(`${city}: the explorer's trip count is the hover's`, counts.length === 0, counts.length ? JSON.stringify(counts[0]) : `(${mine.length})`);
    const walkRows = mine.filter((r) => (r.rows.walk !== null) !== (r.walkMin < 30));
    report(`${city}: a walk row exactly when the walk is under 30 min`, walkRows.length === 0, walkRows.length ? JSON.stringify(walkRows[0]) : `(${mine.filter((r) => r.rows.walk !== null).length} with one)`);
    report(`${city}: no page errors`, errors.length === 0, errors.slice(0, 3).join(' | '));
    if (process.env.SHOTS === '1' && city === CITIES[0]) await shots(page, picks[0]);
    await page.close();
  }
} finally {
  if (process.env.SHOTS_ONLY !== '1') await writeFile(OUT + 't586-agree.json', JSON.stringify(rows, null, 1));
  await close();
}

/** The first selection's busiest far bubble, its hover and the explorer side by side, light and dark. */
async function shots(page, p) {
  for (const shade of ['light', 'dark']) {
    await page.evaluate((s) => { window.tw.shade.value = s; }, shade);
    await setView(page, p.end, 'all', 'average');
    await page.evaluate((s) => { window.tw.selection.value = { ...s }; }, p.sel);
    await page.waitForFunction(() => window.twViews.stats.commuters.farProbe?.(), undefined, { timeout: 60000 });
    await page.waitForTimeout(1500);
    const far = (await farOnScreen(page)).sort((a, b) => b.n - a.n)[0];
    const box = await page.evaluate(() => window.tw.map.getContainer().getBoundingClientRect().toJSON());
    const bx = box.left + far.x, by = box.top + far.y;
    await page.mouse.move(bx - 3, by - 3);
    await page.mouse.move(bx, by, { steps: 3 });
    await page.waitForTimeout(800);
    const tag = await page.evaluate(() => { const t = document.querySelector('.demand-tag'); return t && t.style.display !== 'none' ? t.getBoundingClientRect().toJSON() : null; });
    if (!tag) console.log('no hover tag in the screenshot');
    const hx = Math.max(0, bx - 140), hy = Math.max(0, by - 140);
    const hoverPng = await page.screenshot({ clip: { x: hx, y: hy, width: Math.max(420, (tag?.right ?? 0) - hx + 40), height: Math.max(320, (tag?.bottom ?? 0) - hy + 40) } });
    await page.mouse.click(box.left + far.x, box.top + far.y);
    await page.waitForFunction(() => { const v = window.__t586.trip.tripView.value; return v && !v.asking && v.trip && document.querySelector('#inspector .trip-modes'); }, undefined, { timeout: 120000 });
    await page.mouse.move(5, 5);
    await page.waitForTimeout(800);
    const panel = await page.evaluate(() => document.querySelector('#inspector').getBoundingClientRect().toJSON());
    const panelPng = await page.screenshot({ clip: { x: panel.left, y: panel.top, width: panel.width, height: Math.min(panel.height, 860) } });
    const comp = await ctx.newPage();
    await comp.setContent(`<body style="margin:0;background:${shade === 'dark' ? '#111' : '#fff'}"><div id="c" style="display:inline-flex;gap:12px;align-items:flex-start"><img src="data:image/png;base64,${hoverPng.toString('base64')}"><img src="data:image/png;base64,${panelPng.toString('base64')}"></div></body>`);
    await comp.waitForTimeout(300);
    await comp.locator('#c').screenshot({ path: `${OUT}t586-${shade}.png` });
    await comp.close();
    await page.evaluate(() => { window.tw.selection.value = null; });
  }
}
process.exit(results.length && results.every(Boolean) ? 0 : 1);
