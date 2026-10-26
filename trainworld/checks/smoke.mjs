// The standing smoke check (T-329): run before every commit. Prints one PASS or FAIL line per check
// and exits 1 if any failed. A few minutes.
//   node trainworld/checks/smoke.mjs [url]       (from the maps root; SMOKE_SHADE=dark for the dark theme)
// With no url it starts its own Vite (checks/dev-nowatch.mjs, port SMOKE_PORT or 6190) and stops it
// at the end; or pass a running one, e.g. http://localhost:8800/trainworld/ for the served build.
// Headless Chrome with a throwaway profile (deleted afterwards; a WARN line when over 20 tw-* folders
// older than a day sit in the temp folder); Playwright from PLAYWRIGHT_MODULE,
// else `playwright`, else the npx cache.
//
// Browser: a plain load shows the start menu and its New game starts a game (T-677); a ?new=1 game
// (no menu) loads; in another ?new=1 game a line over dead space and one in Tokyo reach a final (T-845); the real save (T-007-real-nyc.save) imported into it has 109 lines,
// 1,756 stations and about 4,498.6 km of track; demand reaches a final result; Homes draws local
// bubbles in the bare new game and on the real save (T-994); a click on a station
// and on a train picks them; no console or page error, no GL error. Node: t133, t144, t145, t146, t1053 (every zone removed since HEAD has a successor).
import { spawn } from 'node:child_process';
import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { HERE, MAPS, REAL_SAVE, NOISE, SMOKE_FIGURES, staleTempProfiles, launchChrome, collectErrors, importSave, demandFinal, acrossLine, startVite } from './lib.mjs';

const EXPECT = { lines: 109, stations: 1756, trackKm: 4498.6, kmTolerance: 1 };
// a glyph range glyphs.py leaves out on purpose: its 404 makes MapLibre draw the codepoints locally
const GLYPH_404 = /\/fonts\/[^/]+\/\d+-\d+\.pbf$/;

const results = [];
const report = (name, ok, detail = '') => {
  results.push(ok);
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`);
};
const t0 = Date.now();
const secs = () => ((Date.now() - t0) / 1000).toFixed(0) + ' s';

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

/** T-1135: figures of this run by url, for checks that compare against the data as it is now
 * (t883's default-fare rail trips) */
async function saveFigure(base, figures) {
  let all = {};
  try { all = JSON.parse(await readFile(SMOKE_FIGURES, 'utf8')); } catch { /* none yet */ }
  all[base.replace(/\/?$/, '/')] = { ...figures, at: new Date().toISOString() };
  await writeFile(SMOKE_FIGURES, JSON.stringify(all, null, 1) + '\n');
}

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
  const { ctx, close } = await launchChrome({ prefix: 'tw-smoke-' });
  const errors = [];
  try {
    const page = await ctx.newPage();
    // dayShown weekday: the demand final is a weekday's, so figures compare across runs (the default view is the average day)
    // SMOKE_SHADE=dark runs it all in the dark theme (T-039)
    await page.addInitScript((shade) => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade, display: { dayShown: 'weekday' }, v: 2 })); } catch {} }, process.env.SMOKE_SHADE ?? 'light');
    collectErrors(page, errors, NOISE, GLYPH_404);
    const sep = BASE.includes('?') ? '&' : '?';

    // 0. the start menu (T-677): a plain load waits on it with no worker started; New game starts one
    try {
      await page.goto(`${BASE}${sep}debug=1&v=${Date.now()}`);
      await page.waitForSelector('.start-menu .start-new', { timeout: 60000 });
      const before = await page.evaluate(() => performance.getEntriesByType('resource').filter((e) => /clock\.worker|demand\.worker/.test(e.name)).length);
      // T-728: New game opens the new game sheet; a normal game in New York
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
    collectErrors(far, errors, NOISE, GLYPH_404);
    try {
      await far.goto(`${BASE}${sep}debug=1&new=1&paused=1&v=${Date.now()}`);
      await far.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
      const got = [];
      for (const [name, at] of [['dead space', [-98.0, 38.5]], ['Tokyo', [139.70, 35.69]]]) {
        const t = Date.now();
        await far.evaluate((c) => void window.tw.map.jumpTo({ center: c, zoom: 12 }), at);
        if ((await acrossLine(far, at)) === null) throw new Error(`no line built in ${name}`);
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
      await importSave(page, await readFile(REAL_SAVE));
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
      await demandFinal(page, { polling: 250, lines: 100 });
      const v = await page.evaluate(() => {
        const v = window.twDemand.view();
        // boardings by purpose (T-389): the commute's share of all boardings
        const by = {};
        for (const [k, p] of v.purposes ?? []) by[k] = [...p.lines.values()].reduce((s, l) => s + l.ridersPerDay, 0);
        const all = Object.values(by).reduce((s, x) => s + x, 0);
        return { lines: v.lines.size, rail: v.railTripsPerDay, round: v.round, work: all > 0 ? (by.work ?? 0) / all : NaN, longer: all > 0 ? (by.longer ?? 0) / all : NaN };
      });
      report('demand final', v.rail > 0, `(round ${v.round}, ${v.lines} lines, ${Math.round(v.rail).toLocaleString('en-US')} rail trips a day, ${(100 * v.work).toFixed(0)}% of boardings commutes, ${(100 * v.longer).toFixed(2)}% regional, the rest everyday, ${secs()})`);
      await saveFigure(BASE, { realRail: Math.round(v.rail) }).catch((e) => console.log(`  smoke-figures.json not written: ${e.message}`));
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
    await close();
  }
}

const given = process.argv[2];
let vite = null;
try {
  // T-1213: Chrome profiles an older check left in the temp folder (23.9 GB once filled the disk)
  const stale = await staleTempProfiles();
  if (stale > 20) console.log(`WARN  ${stale} tw-* folders over a day old in ${tmpdir()}: old check profiles, safe to delete`);
  for (const f of ['t133-picking-check.mjs', 't144-check.mjs', 't145-check.mjs', 't146-check.mjs', 't1053-zone-ids.mjs']) {
    const r = await runNode(f);
    report(f.replace('.mjs', ''), r.code === 0, r.code === 0 ? '' : tail(r.out));
  }
  if (!given) vite = await startVite(Number(process.env.SMOKE_PORT ?? 6190));
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
