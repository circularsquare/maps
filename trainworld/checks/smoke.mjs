// The standing smoke check (T-329): run before every commit. Prints one PASS or FAIL line per check
// and exits 1 if any failed. A few minutes.
//   node trainworld/checks/smoke.mjs [url]       (from the maps root; SMOKE_SHADE=dark for the dark theme)
// With no url it starts its own Vite (checks/dev-nowatch.mjs, port SMOKE_PORT or 6190) and stops it
// at the end; or pass a running one, e.g. http://localhost:8800/trainworld/ for the served build.
// Headless Chrome with a throwaway profile (deleted afterwards); Playwright from PLAYWRIGHT_MODULE,
// else `playwright`, else the npx cache.
//
// Browser: a plain load shows the start menu and its New game starts a game (T-677); a ?new=1 game
// (no menu) loads; in another ?new=1 game a line over dead space and one in Tokyo reach a final (T-845); the real save (T-007-real-nyc.save) imported into it has 109 lines,
// 1,756 stations and about 4,498.6 km of track; demand reaches a final result; Homes draws local
// bubbles in the bare new game and on the real save (T-994); a click on a station
// and on a train picks them; no console or page error, no GL error. Node: t133, t144, t145, t146, t1053 (every zone removed since HEAD has a successor).
import { spawn } from 'node:child_process';
import { readFile, readdir, mkdtemp, rm } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = fileURLToPath(new URL('.', import.meta.url));
const MAPS = fileURLToPath(new URL('../../', import.meta.url));
const SAVE = fileURLToPath(new URL('../T-007-real-nyc.save', import.meta.url));
const EXPECT = { lines: 109, stations: 1756, trackKm: 4498.6, kmTolerance: 1 };
const CHROME = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
// network noise that is not the game's: the basemap's tiles and fonts when offline, the favicon
const IGNORE = /favicon|openfreemap|AJAXError/;
// a glyph range glyphs.py leaves out on purpose: its 404 makes MapLibre draw the codepoints locally
const GLYPH_404 = /\/fonts\/[^/]+\/\d+-\d+\.pbf$/;

const results = [];
const report = (name, ok, detail = '') => {
  results.push(ok);
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`);
};
const t0 = Date.now();
const secs = () => ((Date.now() - t0) / 1000).toFixed(0) + ' s';

async function playwright() {
  if (process.env.PLAYWRIGHT_MODULE) return import(process.env.PLAYWRIGHT_MODULE);
  try { return await import('playwright'); } catch { /* not installed here */ }
  const npx = join(process.env.LOCALAPPDATA ?? '', 'npm-cache', '_npx');
  for (const d of existsSync(npx) ? await readdir(npx) : []) {
    const p = join(npx, d, 'node_modules', 'playwright', 'index.mjs');
    if (existsSync(p)) return import(pathToFileURL(p).href);
  }
  throw new Error('no Playwright: set PLAYWRIGHT_MODULE to a file URL of playwright/index.mjs');
}

/** this check's own Vite, or null when given a url */
async function startVite() {
  const port = Number(process.env.SMOKE_PORT ?? 6190);
  const child = spawn(process.execPath, [join(HERE, 'dev-nowatch.mjs'), String(port)], { stdio: ['ignore', 'pipe', 'pipe'] });
  let out = '';
  child.stdout.on('data', (d) => { out += d; });
  child.stderr.on('data', (d) => { out += d; });
  const deadline = Date.now() + 60000;
  while (!/localhost:/.test(out)) {
    if (child.exitCode !== null || Date.now() > deadline) throw new Error('Vite did not start: ' + out.slice(-500));
    await new Promise((r) => setTimeout(r, 200));
  }
  return { child, url: `http://localhost:${port}/` };
}

function runNode(file) {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, [join(HERE, file)], { cwd: MAPS, stdio: ['ignore', 'pipe', 'pipe'] });
    let out = '';
    child.stdout.on('data', (d) => { out += d; });
    child.stderr.on('data', (d) => { out += d; });
    child.on('close', (code) => resolve({ code, out }));
  });
}

/** the last meaningful line of a failed check's output (bundled code comes out as huge data: urls) */
const tail = (out) => out.split('\n').filter((l) => l.trim() && !l.startsWith('data:') && l.length < 400).slice(-3).join(' | ');

/** a built line of three stations 2 km apart, east to west across `[lng, lat]`, by the edit API; its number or null */
const smokeLine = (page, [lng, lat]) => page.evaluate(async ([lng, lat]) => {
  const tw = window.tw, c = tw.client;
  const dLng = 2 / (111.32 * Math.cos((lat * Math.PI) / 180));
  const xy = [-1, 0, 1].map((k) => { const p = tw.lngLatToPack(lng + k * dLng, lat); return [p.x, p.y]; });
  const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p[0], n.y - p[1]); if (d < bd) { bd = d; best = n.id; } } return best; };
  const wait = (ms) => new Promise((d) => setTimeout(d, ms));
  await c.edit({ op: 'route', from: { kind: 'free', x: xy[0][0], y: xy[0][1], level: -1 }, to: { kind: 'free', x: xy[1][0], y: xy[1][1], level: -1 }, pis: [], single: false });
  await wait(300);
  await c.edit({ op: 'route', from: { kind: 'node', node: near(xy[1]) }, to: { kind: 'free', x: xy[2][0], y: xy[2][1], level: -1 }, pis: [], single: false });
  await wait(300);
  for (const p of xy) { await c.edit({ op: 'addStation', at: { kind: 'node', node: near(p) }, platform: 200, name: '' }); await wait(150); }
  await c.edit({ op: 'constructAll' });
  await wait(300);
  const r = await c.edit({ op: 'addLine', stops: xy.map(near), name: 'Check', colour: '#0a7d3b' });
  if (r?.line === undefined) return null;
  await c.edit({ op: 'setTrainCount', line: r.line, trains: [6, 4, 3] });
  return r.line;
}, [lng, lat]);

const NYC = [-73.985, 40.758];
/** T-994: Homes turned on at city zoom draws the local bubbles; Homes off again after. The discs
 * drawn, or 0 when none came within 60 s. */
async function homesDrawn(page, at) {
  const before = await page.evaluate((c) => {
    window.tw.selection.value = null;
    window.tw.map.jumpTo({ center: c, zoom: 12 });
    window.tw.display.value = { ...window.tw.display.peek(), commuters: true, commuterEnd: 'home' };
    return window.twViews.overlay.draws;
  }, at);
  try {
    await page.waitForFunction((before) => {
      const o = window.twViews.overlay, p = window.twViews.stats.commuters.probe?.();
      return o.draws > before && o.discsDrawn > 0 && !!p && p.levels.some((l) => l.discs.length > 0);
    }, before, { timeout: 60000, polling: 250 });
    return await page.evaluate(() => window.twViews.overlay.discsDrawn);
  } catch {
    return 0;
  } finally {
    await page.evaluate(() => { window.tw.display.value = { ...window.tw.display.peek(), commuters: false }; });
  }
}

async function browserChecks(BASE) {
  const { chromium } = await playwright();
  const profile = await mkdtemp(join(tmpdir(), 'tw-smoke-'));
  const ctx = await chromium.launchPersistentContext(profile, { executablePath: CHROME, headless: true, viewport: { width: 1400, height: 900 }, args: ['--use-angle=d3d11', '--ignore-gpu-blocklist'] });
  const errors = [];
  try {
    const page = await ctx.newPage();
    // the demand final is a weekday's, comparable across runs since T-464 made the average day the default view
    // SMOKE_SHADE=dark runs it all in the dark theme (T-039)
    await page.addInitScript((shade) => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade, display: { dayShown: 'weekday' }, v: 2 })); } catch {} }, process.env.SMOKE_SHADE ?? 'light');
    page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
    page.on('console', (m) => { if (m.type() === 'error' && !IGNORE.test(m.text()) && !GLYPH_404.test(m.location()?.url ?? '')) errors.push(m.text()); });
    const sep = BASE.includes('?') ? '&' : '?';

    // 0. the start menu (T-677): a plain load waits on it with no worker started; New game starts one
    try {
      await page.goto(`${BASE}${sep}debug=1&v=${Date.now()}`);
      await page.waitForSelector('.start-menu .start-new', { timeout: 60000 });
      const before = await page.evaluate(() => performance.getEntriesByType('resource').filter((e) => /clock\.worker|demand\.worker/.test(e.name)).length);
      // T-728: New game opens the new game sheet; a normal game in New York, as before
      await page.click('.start-menu .start-new');
      await page.click('.start-menu .new-game-cities [data-city="nyc"]');
      await page.click('.start-menu .new-game .start-new');
      await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client && !document.querySelector('.start-menu'), undefined, { timeout: 180000 });
      const n = await page.evaluate(() => window.tw.world.value.save.stations.length);
      report('start menu', before === 0 && n === 0, `(${before} workers before the pick, ${n} stations after New game)`);
    } catch (e) {
      report('start menu', false, e.message.split('\n')[0]);
    }

    // 1. a new game
    try {
      await page.goto(`${BASE}${sep}debug=1&new=1&paused=1&v=${Date.now()}`);
      await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
      await page.waitForTimeout(2000);
      const n = await page.evaluate(() => ({ lines: window.tw.world.value.save.lines.length, stations: window.tw.world.value.save.stations.length }));
      report('new game loads', n.lines === 0 && n.stations === 0, `(${n.lines} lines, ${n.stations} stations, ${secs()})`);
    } catch (e) {
      report('new game loads', false, e.message.split('\n')[0]);
      return;
    }
    // 1a. T-994: Homes draws the local bubbles in a city no station reaches
    const bare = await homesDrawn(page, NYC);
    report('homes bubbles, bare city', bare > 0, `(${bare.toLocaleString('en-US')} discs)`);

    // 1b. T-845: new games away from New York reach a final: a line over dead space (no zone reaches
    // it, the view there too), then one in Tokyo; on a page of their own, so the real save's solve
    // below opens only what it did before
    const far = await ctx.newPage();
    far.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
    far.on('console', (m) => { if (m.type() === 'error' && !IGNORE.test(m.text()) && !GLYPH_404.test(m.location()?.url ?? '')) errors.push(m.text()); });
    try {
      await far.goto(`${BASE}${sep}debug=1&new=1&paused=1&v=${Date.now()}`);
      await far.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
      const got = [];
      for (const [name, at] of [['dead space', [-98.0, 38.5]], ['Tokyo', [139.70, 35.69]]]) {
        const t = Date.now();
        await far.evaluate((c) => void window.tw.map.jumpTo({ center: c, zoom: 12 }), at);
        if ((await smokeLine(far, at)) === null) throw new Error(`no line built in ${name}`);
        const want = got.length + 1;
        await far.waitForFunction((want) => {
          const d = window.twDemand, s = d.log.at(-1), v = d.view();
          return String(d.client.status).startsWith('failed') || (s && v && v.version === s.version && v.final && v.lines.size === want);
        }, want, { timeout: 90000, polling: 250 });
        const v = await far.evaluate(() => { const d = window.twDemand, v = d.view(); return { status: String(d.client.status), riders: [...v.lines.values()].map((l) => Math.round(l.ridersPerDay)) }; });
        if (v.status.startsWith('failed')) throw new Error(`${name}: ${v.status}`);
        got.push(`${name} ${((Date.now() - t) / 1000).toFixed(1)} s, riders ${v.riders.at(-1).toLocaleString('en-US')}`);
      }
      const riders = await far.evaluate(() => [...window.twDemand.view().lines.values()].map((l) => l.ridersPerDay));
      report('far new game final', riders.length === 2 && riders.some((r) => r === 0) && riders.some((r) => r > 0), `(${got.join('; ')})`);
    } catch (e) {
      report('far new game final', false, e.message.split('\n')[0]);
    } finally {
      await far.close();
    }

    // 2. the real save, imported into that game
    try {
      const bytes = await readFile(SAVE);
      await page.evaluate((data) => { const b = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes: b }, [b]); }, Array.from(bytes));
      await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
      await page.waitForTimeout(1000);
      const m = await page.evaluate(() => {
        const w = window.tw.world.value;
        return { lines: w.save.lines.length, stations: w.save.stations.length, trackKm: w.edges.reduce((s, e) => s + e.lengthM, 0) / 1000 };
      });
      const ok = m.lines === EXPECT.lines && m.stations === EXPECT.stations && Math.abs(m.trackKm - EXPECT.trackKm) < EXPECT.kmTolerance;
      report('real save counts', ok, `(${m.lines} lines, ${m.stations.toLocaleString('en')} stations, ${m.trackKm.toFixed(1)} km; want ${EXPECT.lines}, ${EXPECT.stations}, ${EXPECT.trackKm})`);
    } catch (e) {
      report('real save counts', false, e.message.split('\n')[0]);
      return;
    }

    // 3. demand reaches a final result for that network
    try {
      await page.waitForFunction(() => {
        const d = window.twDemand, s = d?.log?.at(-1), v = d?.view?.();
        return s && v && v.version === s.version && v.final && !d.client.waiting && v.lines.size > 100;
      }, undefined, { timeout: 300000, polling: 250 });
      const v = await page.evaluate(() => {
        const v = window.twDemand.view();
        // boardings by purpose (T-389): the commute's share of all boardings
        const by = {};
        for (const [k, p] of v.purposes ?? []) by[k] = [...p.lines.values()].reduce((s, l) => s + l.ridersPerDay, 0);
        const all = Object.values(by).reduce((s, x) => s + x, 0);
        return { lines: v.lines.size, rail: v.railTripsPerDay, round: v.round, work: all > 0 ? (by.work ?? 0) / all : NaN, longer: all > 0 ? (by.longer ?? 0) / all : NaN };
      });
      report('demand final', v.rail > 0, `(round ${v.round}, ${v.lines} lines, ${Math.round(v.rail).toLocaleString('en-US')} rail trips a day, ${(100 * v.work).toFixed(0)}% of boardings commutes, ${(100 * v.longer).toFixed(2)}% regional, the rest everyday, ${secs()})`);
    } catch (e) {
      report('demand final', false, e.message.split('\n')[0]);
    }
    // 3b. T-994: and in a city with lines
    const built = await homesDrawn(page, NYC);
    report('homes bubbles, real save', built > 0, `(${built.toLocaleString('en-US')} discs)`);

    // 4. a click on a station picks it: stations with no other station within 400 m, at zoom 16
    try {
      const cands = await page.evaluate(() => {
        const st = window.tw.world.value.save.stations, out = [];
        const m = (a, b) => Math.hypot((a.lng - b.lng) * 84000, (a.lat - b.lat) * 111000);
        for (let i = 0; i < st.length && out.length < 6; i += 37) {
          if (!st.some((o) => o !== st[i] && m(o, st[i]) < 400)) out.push({ id: st[i].id, lng: st[i].lng, lat: st[i].lat });
        }
        return out;
      });
      let got = null;
      for (const c of cands) {
        await page.evaluate((c) => { window.tw.selection.value = null; window.tw.map.jumpTo({ center: [c.lng, c.lat], zoom: 16 }); }, c);
        await page.waitForTimeout(800);
        const box = await page.evaluate((c) => { const r = window.tw.map.getContainer().getBoundingClientRect(), p = window.tw.map.project([c.lng, c.lat]); return [r.left + p.x, r.top + p.y]; }, c);
        await page.mouse.click(box[0], box[1]);
        await page.waitForTimeout(300);
        const sel = await page.evaluate(() => window.tw.selection.value);
        if (sel?.kind === 'station' && sel.station === c.id) { got = c.id; break; }
      }
      report('station click', !!got, got ? `(station ${got})` : `(none of ${cands.length} picked)`);
    } catch (e) {
      report('station click', false, e.message.split('\n')[0]);
    }

    // 5. a click on a train picks it: the clock paused at 08:30, trains from the GPU's own positions
    try {
      await page.evaluate(() => {
        const tw = window.tw;
        tw.clock.setRunning(false);
        tw.clock.set(Math.floor(tw.clock.now() / 86400) * 86400 + 8.5 * 3600);
        tw.selection.value = null;
        tw.overlay.invalidate();
      });
      await page.waitForFunction(() => window.tw.renderer.tripCount > 0, undefined, { timeout: 60000 });
      let got = null, tried = 0;
      for (let k = 0; k < 8 && !got; k++) {
        const target = await page.evaluate((k) => {
          const tw = window.tw, r = tw.renderer, gpu = r.probe(tw.clock.now()), a = r.net?.anchor ?? tw.anchor();
          const live = [];
          for (let i = 0; i < r.tripCount; i++) if (gpu[i * 2] < 1e8) live.push(i);
          const i = live[Math.floor((live.length * (k + 0.5)) / 8)];
          if (i === undefined) return null;
          const [lng, lat] = tw.toLngLat(gpu[i * 2], gpu[i * 2 + 1], a);
          return { lng, lat, line: Math.floor(r.trips[i * 2] / 6) };
        }, k);
        if (!target) break;
        tried++;
        await page.evaluate((t) => { window.tw.map.jumpTo({ center: [t.lng, t.lat], zoom: 16 }); window.tw.overlay.invalidate(); }, target);
        await page.waitForTimeout(800);
        // the train where it is now (the clock is paused, so it has not moved)
        const xy = await page.evaluate((t) => { const r = window.tw.map.getContainer().getBoundingClientRect(), p = window.tw.map.project([t.lng, t.lat]); return [r.left + p.x, r.top + p.y]; }, target);
        await page.mouse.click(xy[0], xy[1]);
        await page.waitForTimeout(300);
        const sel = await page.evaluate(() => window.tw.selection.value);
        if (sel?.kind === 'train') got = sel.line;
      }
      report('train click', !!got, got ? `(a train of line ${got})` : `(none of ${tried} picked)`);
    } catch (e) {
      report('train click', false, e.message.split('\n')[0]);
    }

    // 6. GL and console
    const gl = await page.evaluate(() => window.tw.renderer.gl?.getError?.() ?? -1).catch(() => -2);
    report('no GL error', gl === 0, gl === 0 ? '' : `(getError ${gl})`);
    report('no console errors', errors.length === 0, errors.length ? `(${errors.length}: ${errors.slice(0, 3).join(' | ').slice(0, 400)})` : '');
  } finally {
    await ctx.close();
    await rm(profile, { recursive: true, force: true }).catch(() => {});
  }
}

const given = process.argv[2];
let vite = null;
try {
  for (const f of ['t133-picking-check.mjs', 't144-check.mjs', 't145-check.mjs', 't146-check.mjs', 't1053-zone-ids.mjs']) {
    const r = await runNode(f);
    report(f.replace('.mjs', ''), r.code === 0, r.code === 0 ? '' : tail(r.out));
  }
  if (!given) vite = await startVite();
  const base = given ?? vite.url;
  console.log(`      browser checks on ${base}`);
  await browserChecks(base);
} catch (e) {
  report('smoke check ran', false, e.message.split('\n')[0]);
} finally {
  if (vite) vite.child.kill();
}
const failed = results.filter((ok) => !ok).length;
console.log(failed ? `SMOKE FAILED: ${failed} of ${results.length} checks, ${secs()}` : `SMOKE PASSED: ${results.length} checks, ${secs()}`);
process.exit(failed ? 1 : 0);
