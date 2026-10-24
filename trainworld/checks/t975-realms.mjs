// T-975, T-1006: realms (game/realms.ts, map/realmOverlay.ts). First (T-1132, T-1133), in a profile of
// its own: the new game sheet's location button refused, in Lyon and in Sydney (a map start centred
// there); a map start shows every realm until its first station, in France, leaves the others dark.
// Then in a fresh profile with Homes on: the
// start menu's map draws no city bubbles; a new Tokyo game shows Japan only, the shade cut around
// Japan's zones and every other realm with a button; a click on France's button shows France; a
// reload with a station in Japan only hides France again; a station in France as well, and a reload
// shows both; (T-1049, T-1046) a click on a hidden realm shows it, a drag does not, and the world view has no
// circles or city bubbles. Measures the zoomed-out world (bubbles, frame time while panning) and what a pan across
// France and Germany at zoom 9 fetches. Screenshots and a JSON to artifacts/trainworld/t975-<TAG>-*.
//   BASE=http://localhost:6165/ [TAG=after] node trainworld/checks/t975-realms.mjs
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { launchChrome, collectErrors } from './lib.mjs';

const BASE = process.env.BASE ?? 'http://localhost:6165/';
const TAG = process.env.TAG ?? 'after';
const SHOTS = fileURLToPath(new URL('../../artifacts/trainworld/', import.meta.url));
await mkdir(SHOTS, { recursive: true });
const results = [];
const out = { tag: TAG };
const report = (name, ok, detail = '') => {
  results.push(ok);
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`);
};

const { ctx, close } = await launchChrome({ prefix: 'tw-t975-' });
const errors = [];
const fetched = [];
const watch = (page) => {
  collectErrors(page, errors, /favicon|openfreemap|AJAXError|ERR_FAILED/i);
  page.on('requestfinished', async (r) => {
    const u = r.url();
    if (!u.includes('/packs/')) return;
    let bytes = 0;
    try { bytes = (await r.sizes()).responseBodySize; } catch {}
    fetched.push({ t: Date.now(), file: u.split('/packs/')[1].split('?')[0], bytes });
  });
};
const settle = (page, ms) => page.waitForTimeout(ms);
/** the realms with a button, by name: those not shown */
const dark = (page) => page.evaluate(() => [...document.querySelectorAll('.realm-button button')].map((b) => b.getAttribute('aria-label').replace(/^Show /, '')));
/** the shade's triangles beyond the world's 2 (its own layer since T-1025): more as more zones are cut out */
const holes = (page) => page.evaluate(() => { const l = window.tw.map.getLayer('tw-realms-inactive')?.implementation; return l ? l.count / 3 - 2 : -1; });
/** a blueprint station on a short stretch of track at (lng, lat), by the edit API */
const station = (page, at) => page.evaluate(async ([lng, lat]) => {
  const tw = window.tw, c = tw.client;
  const d = 1 / (111.32 * Math.cos((lat * Math.PI) / 180));
  const a = tw.lngLatToPack(lng - d, lat), b = tw.lngLatToPack(lng + d, lat);
  await c.edit({ op: 'route', from: { kind: 'free', x: a.x, y: a.y, level: -1 }, to: { kind: 'free', x: b.x, y: b.y, level: -1 }, pis: [], single: false });
  await new Promise((ok) => setTimeout(ok, 400));
  let best = null, bd = 1e12;
  for (const n of tw.world.value.nodes.values()) { const k = Math.hypot(n.x - a.x, n.y - a.y); if (k < bd) { bd = k; best = n.id; } }
  await c.edit({ op: 'addStation', at: { kind: 'node', node: best }, platform: 200, name: '' });
  await new Promise((ok) => setTimeout(ok, 400));
  return tw.world.value.save.stations.length;
}, at);
const ready = (page) => page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client && !document.querySelector('.start-menu'), undefined, { timeout: 180000 });

/** frame cost zoomed out to the world: bubbles drawn, then 3 s of panning */
async function worldFrame(page, name) {
  await page.evaluate(() => { window.tw.display.value = { ...window.tw.display.value, commuters: true }; window.tw.map.jumpTo({ center: [60, 35], zoom: 1.8 }); });
  await settle(page, 4000);
  await page.screenshot({ path: join(SHOTS, `t975-${TAG}-${name}.png`) });
  return page.evaluate(async () => {
    const tw = window.tw, map = tw.map;
    const labels = document.querySelectorAll('.intercity-label').length;
    const long = [];
    const obs = new PerformanceObserver((l) => { for (const e of l.getEntries()) long.push(e.duration); });
    obs.observe({ entryTypes: ['longtask'] });
    tw.reset();
    const gaps = [];
    let last = performance.now();
    const t0 = last;
    await new Promise((done) => {
      const step = () => {
        const now = performance.now();
        gaps.push(now - last); last = now;
        map.panBy([6, 0], { animate: false });
        if (now - t0 < 3000) requestAnimationFrame(step); else done();
      };
      requestAnimationFrame(step);
    });
    obs.disconnect();
    gaps.sort((a, b) => a - b);
    const s = tw.stats();
    return { bubbles: labels, frames: gaps.length, frameMs: +(gaps.reduce((a, b) => a + b, 0) / gaps.length).toFixed(1), p95Ms: +gaps[Math.floor(gaps.length * 0.95)].toFixed(1), overlayDrawMs: +(s.drawMs ?? 0).toFixed(2), longTasks: long.length, longMs: Math.round(long.reduce((a, b) => a + b, 0)) };
  });
}

/** a pan across France and Germany at zoom 9: the packs fetched */
async function europePan(page) {
  const t0 = Date.now();
  const long = await page.evaluate(() => { window.__long = []; new PerformanceObserver((l) => { for (const e of l.getEntries()) window.__long.push(e.duration); }).observe({ entryTypes: ['longtask'] }); });
  for (const c of [[2.35, 48.86], [3.5, 48.5], [4.8, 48.9], [6.2, 49.1], [7.6, 49.4], [8.7, 50.1], [9.9, 50.6], [11.0, 51.0]]) {
    await page.evaluate((c) => { window.tw.map.jumpTo({ center: c, zoom: 9 }); }, c);
    await settle(page, 2500);
  }
  await settle(page, 4000);
  await page.screenshot({ path: join(SHOTS, `t975-${TAG}-europe-z9.png`) });
  const got = fetched.filter((f) => f.t >= t0);
  const kinds = {};
  for (const f of got) {
    const k = f.file.match(/\.(boundary\.geojson|market\.json|places\.json|json|bin)$/)?.[1] ?? f.file;
    kinds[k] = kinds[k] ?? { n: 0, kB: 0 };
    kinds[k].n++; kinds[k].kB += Math.round(f.bytes / 1024);
  }
  const lt = await page.evaluate(() => window.__long);
  return { files: got.length, kB: Math.round(got.reduce((a, f) => a + f.bytes, 0) / 1024), kinds, longTasks: lt.length, longMs: Math.round(lt.reduce((a, b) => a + b, 0)) };
}

// 0. T-1132, T-1133: a start on the map, and the location button, in a profile of their own
{
  const { ctx: ctx0, close: close0 } = await launchChrome({ prefix: 'tw-t975-map-' });
  try {
    const page = await ctx0.newPage();
    watch(page);
    await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday', commuters: true }, v: 2 })); } catch {} });
    const sheet = async () => {
      await page.goto(`${BASE}?debug=1&v=${Date.now()}`);
      await page.waitForSelector('.start-menu .start-new', { timeout: 60000 });
      await settle(page, 2000);
      await page.click('.start-menu .start-new');
      await page.waitForSelector('.start-menu .new-game');
    };
    const label = () => page.textContent('.new-game .start-new');
    const here = '.new-game-here';
    await sheet();
    const ors = await page.locator('.new-game-or').count();
    const mapLit = await page.locator('.new-game-map.on').count();
    // a refusal: the button says so, nothing else changes
    await ctx0.clearPermissions();
    await page.click(here);
    await page.waitForFunction(() => document.querySelector('.new-game-here')?.dataset.state === 'failed', undefined, { timeout: 20000 }).catch(() => {});
    const refused = { text: await page.textContent(here), label: await label(), mapLit: await page.locator('.new-game-map.on').count() };
    report('the sheet has its three ways joined by "or", and a refused location leaves it as it was', ors === 2 && mapLit === 1 && refused.text.trim() === 'Location unavailable' && refused.label === 'Start on the map' && refused.mapLit === 1, JSON.stringify({ ors, mapLit, refused }));
    // a location in a zone (Lyon), and one far from every zone (Sydney)
    await ctx0.grantPermissions(['geolocation']);
    await ctx0.setGeolocation({ longitude: 4.84, latitude: 45.76 });
    await page.click(here);
    await page.waitForFunction(() => document.querySelector('.new-game-here')?.dataset.state === 'located', undefined, { timeout: 20000 }).catch(() => {});
    const lyon = await label();
    await page.screenshot({ path: join(SHOTS, `t975-${TAG}-located.png`) });
    await page.click(here); // a second click unlights it
    const off = await label();
    await ctx0.setGeolocation({ longitude: 151.2, latitude: -33.87 });
    await page.click(here);
    await page.waitForFunction(() => document.querySelector('.new-game-here')?.dataset.state === 'located', undefined, { timeout: 20000 }).catch(() => {});
    const sydney = await label();
    await page.click('.new-game .start-new');
    await ready(page);
    await settle(page, 3000);
    const view = await page.evaluate(() => { const c = window.tw.map.getCenter(); return { lng: c.lng, lat: c.lat, zoom: window.tw.map.getZoom() }; });
    report('the location starts in its zone, or on the map centred there', /^Start in /.test(lyon) && off === 'Start on the map' && sydney === 'Start on the map' && Math.abs(view.lng - 151.2) < 1 && Math.abs(view.lat + 33.87) < 1 && view.zoom < 5,
      JSON.stringify({ lyon, off, sydney, view }));

    // a map start: every realm shown (no circles), the shade cut round every zone
    await sheet();
    const tPick = Date.now();
    await page.click('.new-game .start-new');
    await ready(page);
    const tReady = Date.now();
    await settle(page, 6000);
    await page.evaluate(() => { window.tw.map.jumpTo({ center: [8, 49], zoom: 4.5 }); });
    await settle(page, 3000);
    await page.screenshot({ path: join(SHOTS, `t975-${TAG}-map-start.png`) });
    const d = await dark(page), h = await holes(page);
    out.mapStart = { toWorldS: +((tReady - tPick) / 1000).toFixed(1), triangles: h };
    report('a map start shows every realm, the shade cut round every zone', d.length === 0 && h > 10000, `(${d.length} dark, ${h} triangles)`);
    // a station in France: the realms follow the usual rule
    await page.evaluate(() => { window.tw.map.jumpTo({ center: [4.84, 45.76], zoom: 12 }); });
    await settle(page, 1500);
    await station(page, [4.84, 45.76]);
    await page.evaluate(() => { window.tw.map.jumpTo({ center: [8, 49], zoom: 4.5 }); });
    await settle(page, 4000);
    const d1 = await dark(page);
    report('a map start\'s first station leaves its realm shown and the rest dark', d1.length > 10 && !d1.includes('France') && d1.includes('Germany'), `(${d1.length} dark)`);
    await page.screenshot({ path: join(SHOTS, `t975-${TAG}-map-start-built.png`) });
  } finally {
    await close0();
  }
}

try {
  const page = await ctx.newPage();
  watch(page);
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday', commuters: true }, v: 2 })); } catch {} });
  // 1. a new Tokyo game from the start menu, whose map draws no bubbles
  const t0 = Date.now();
  await page.goto(`${BASE}?debug=1&v=${Date.now()}`);
  await page.waitForSelector('.start-menu .start-new', { timeout: 60000 });
  await settle(page, 4000);
  const menuBubbles = await page.evaluate(() => document.querySelectorAll('.intercity-label').length);
  report('no city bubbles behind the start menu', menuBubbles === 0, `(${menuBubbles})`);
  const tPick = Date.now();
  await page.click('.start-menu .start-new');
  await page.click('.start-menu .new-game-cities [data-city="tyo"]');
  await page.click('.start-menu .new-game .start-new');
  await ready(page);
  const tReady = Date.now();
  // T-1051: while the loading card is up nothing is drawn over the basemap; when it goes, the end state is in
  {
    const sample = () => page.evaluate(() => {
      const map = window.tw.map, impl = map.getLayer('tw-realms-inactive')?.implementation;
      return {
        card: !!document.querySelector('.loading-card'),
        labels: [...document.querySelectorAll('.intercity-label')].filter((e) => e.style.visibility !== 'hidden').length,
        circles: [...document.querySelectorAll('.realm-button')].filter((e) => e.offsetParent && e.style.display !== 'none').length,
        shade: !!impl?.visible,
        outside: map.getLayer('tw-city-outside') ? map.getLayoutProperty('tw-city-outside', 'visibility') !== 'none' : false,
        outlines: map.getLayer('tw-city-boundaries') ? map.getLayoutProperty('tw-city-boundaries', 'visibility') !== 'none' : false,
      };
    });
    let n = 0, drawn = 0, last = await sample();
    for (const t0 = Date.now(); last.card && Date.now() - t0 < 120000; last = await sample()) {
      n++;
      if (last.labels || last.circles || last.shade || last.outside || last.outlines) drawn++;
      if (n === 1 || n % 20 === 0) await page.screenshot({ path: join(SHOTS, `t975-${TAG}-loading-${n}.png`) });
      await settle(page, 200);
    }
    report('nothing is drawn over the map while the game loads', n > 0 && drawn === 0, `(${n} samples with the card up, ${drawn} with something drawn)`);
    await settle(page, 1500);
    last = await sample();
    report('the end state arrives when the card goes', !last.card && last.shade && last.outside, JSON.stringify(last));
  }
  await settle(page, 8000);
  const start = fetched.filter((f) => f.t >= tPick);
  out.startup = { menuS: +((tPick - t0) / 1000).toFixed(1), toWorldS: +((tReady - tPick) / 1000).toFixed(1), files: start.length, kB: Math.round(start.reduce((a, f) => a + f.bytes, 0) / 1024) };
  console.log('startup', JSON.stringify(out.startup));
  const d0 = await dark(page);
  report('a Tokyo start shows Japan only', d0.length > 10 && !d0.includes('Japan') && d0.includes('France'), `(${d0.length} realms dark)`);
  const h0 = await holes(page);
  report('the shade is cut around the shown zones', h0 > 0 && !start.some((f) => f.file === 'realms.geojson'), `(${h0} triangles)`);
  out.world = await worldFrame(page, 'world');
  console.log('world', JSON.stringify(out.world));

  out.pan = await europePan(page);
  console.log('europe pan', JSON.stringify(out.pan));

  // 2. France's button
  {
    await page.evaluate(() => { window.tw.map.jumpTo({ center: [5, 48], zoom: 4.5 }); });
    await settle(page, 5000);
    await page.screenshot({ path: join(SHOTS, `t975-${TAG}-europe-dark.png`) });
    await page.click('.realm-button button[aria-label="Show France"]');
    await settle(page, 5000);
    await page.screenshot({ path: join(SHOTS, `t975-${TAG}-france-shown.png`) });
    const d1 = await dark(page);
    const h1 = await holes(page);
    report('a click shows France', !d1.includes('France') && d1.length === d0.length - 1 && h1 > h0, `(${d1.length} dark, ${h1} triangles)`);
  }

  {
    // 3. a station in Japan, then a reload: France is dark again
    await page.evaluate(() => { window.tw.map.jumpTo({ center: [139.70, 35.69], zoom: 12 }); });
    await settle(page, 1500);
    await station(page, [139.70, 35.69]);
    await page.evaluate(() => window.tw.client.post({ kind: 'flush' }));
    await settle(page, 3000);
    await page.goto(`${BASE}?debug=1&continue=1&v=${Date.now()}`);
    await ready(page);
    await settle(page, 5000);
    const d2 = await dark(page);
    report('a reload hides France again', d2.includes('France') && !d2.includes('Japan'), `(${d2.length} dark)`);

    // 4. a station in France too: both shown after a reload
    await page.evaluate(() => { window.tw.map.jumpTo({ center: [4.84, 45.76], zoom: 12 }); });
    await settle(page, 1500);
    await station(page, [4.84, 45.76]);
    await settle(page, 1500);
    const d3 = await dark(page);
    report('a station in France shows it', !d3.includes('France'), `(${d3.length} dark)`);
    await page.evaluate(() => window.tw.client.post({ kind: 'flush' }));
    await settle(page, 3000);
    await page.goto(`${BASE}?debug=1&continue=1&v=${Date.now()}`);
    await ready(page);
    await settle(page, 5000);
    const d4 = await dark(page);
    const n = await page.evaluate(() => window.tw.world.value.save.stations.length);
    report('a save with stations in Japan and France shows both', !d4.includes('France') && !d4.includes('Japan') && d4.length === d0.length - 1, `(${n} stations, ${d4.length} dark)`);
    await page.evaluate(() => { window.tw.map.jumpTo({ center: [60, 35], zoom: 1.8 }); });
    await settle(page, 3000);
    await page.screenshot({ path: join(SHOTS, `t975-${TAG}-two-realms.png`) });
  }
  // 5. T-1049, T-1046: a click on a hidden realm's land shows it, a drag does not; none at the world view
  {
    const at = (lng, lat) => page.evaluate(([lng, lat]) => { const q = window.tw.map.project([lng, lat]), r = document.querySelector('#map').getBoundingClientRect(); return [q.x + r.left, q.y + r.top]; }, [lng, lat]);
    const view = async (center, zoom) => { await page.evaluate(([center, zoom]) => { window.tw.map.jumpTo({ center, zoom }); }, [center, zoom]); await settle(page, 2500); };
    const before = await dark(page);
    const tip = await page.evaluate(() => document.querySelector('.realm-button button[aria-label="Show Germany"]')?.getAttribute('data-tip') ?? '');
    report('a realm circle has the game\'s own tag naming the realm', /^Germany\n/.test(tip) && await page.evaluate(() => !document.querySelector('.realm-button button[title]')), JSON.stringify(tip));
    await view([10.5, 51], 5);
    let [x, y] = await at(10.5, 51);
    await page.mouse.move(x, y);
    await page.mouse.down();
    for (let i = 1; i <= 8; i++) await page.mouse.move(x + 15 * i, y + 5 * i);
    await page.mouse.up();
    await settle(page, 1500);
    const afterDrag = await dark(page);
    report('a drag across a hidden realm does not show it', afterDrag.includes('Germany') && afterDrag.length === before.length, `(${afterDrag.length} dark)`);
    await view([5, 48], 1.8);
    [x, y] = await at(10.5, 51);
    await page.mouse.click(x, y);
    await settle(page, 1500);
    const world = await page.evaluate(() => ({ circles: [...document.querySelectorAll('.realm-button')].filter((e) => e.offsetParent).length, labels: document.querySelectorAll('.intercity-label').length }));
    report('at the world view there are no circles, no city labels and a click shows nothing', world.circles === 0 && world.labels === 0 && (await dark(page)).includes('Germany'), JSON.stringify(world));
    await view([10.5, 51], 5);
    [x, y] = await at(10.5, 51);
    await page.mouse.click(x, y);
    await settle(page, 3000);
    const afterClick = await dark(page);
    report('a click on a hidden realm shows it', !afterClick.includes('Germany') && afterClick.length === before.length - 1, `(${afterClick.length} dark)`);
    await page.screenshot({ path: join(SHOTS, `t975-${TAG}-germany-clicked.png`) });
    await view([5, 48], 4.5);
  }

  report('no console errors', errors.length === 0, errors.slice(0, 3).join(' | '));
} finally {
  await close();
}
await writeFile(join(SHOTS, `t975-${TAG}.json`), JSON.stringify(out, null, 1));
process.exit(results.every(Boolean) ? 0 : 1);
