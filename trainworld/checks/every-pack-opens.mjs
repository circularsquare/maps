// T-484: every registered zone opens and solves without a demand error. For each zone in
// registry.json, a fresh ?new=1 game flies to the zone's centre (its pack opens for bubbles), then
// a three-station line is built across the centre and demand must reach a final result. One PASS or
// FAIL line per zone, with the worker's error; exits 1 if any failed. About 30 s a zone.
//   node trainworld/checks/every-pack-opens.mjs [url] [zone ...]      (from the maps root)
// T-492: with --broken it instead serves one zone's city pack, then its market pack, broken (a
// zone that downloads whole and does not open) and checks that zone alone is flagged and left out
// while another zone's line solves. About 2 minutes; name `city` or `market` to run one of the two.
// With no url it starts its own Vite (checks/dev-nowatch.mjs, port PACKS_PORT or 6191). A pack being
// rewritten while the check reads it can fail spuriously: rerun that zone alone.
import { launchChrome, collectErrors, acrossLine, startVite, registry } from './lib.mjs';

async function checkZone(ctx, base, city) {
  const page = await ctx.newPage();
  const errors = [];
  collectErrors(page, errors);
  const failed = () => page.evaluate(() => (String(window.twDemand?.client?.status ?? '').startsWith('failed') ? window.twDemand.client.status : null));
  try {
    const sep = base.includes('?') ? '&' : '?';
    await page.goto(`${base}${sep}debug=1&new=1&opened=all&paused=1&v=${Date.now()}`);
    await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 120000 });
    await page.evaluate(([c, z]) => void window.tw.map.jumpTo({ center: c, zoom: z }), [city.center, Math.max(city.zoom, 11)]);
    await page.waitForFunction((id) => window.twDemand.activeCity.peek() === id, city.id, { timeout: 60000, polling: 100 });
    // its pack opened for the view (bubbles): give it time to download and open
    await page.waitForTimeout(6000);
    let f = await failed();
    if (f) return { ok: false, step: 'view', detail: f };
    const line = await acrossLine(page, city.center);
    if (line === null) return { ok: false, step: 'build', detail: 'no line built' };
    await page.waitForFunction(() => {
      const d = window.twDemand, s = d?.log?.at(-1), v = d?.view?.();
      if (String(d.client.status).startsWith('failed')) return true;
      return s && v && v.version === s.version && v.final && !d.client.waiting && v.lines.size > 0;
    }, undefined, { timeout: 240000, polling: 250 });
    f = await failed();
    if (f) return { ok: false, step: 'solve', detail: f };
    const v = await page.evaluate(() => { const d = window.twDemand, s = d.log.at(-1), v = d.view(); return { rail: Math.round(v.railTripsPerDay), cities: s.cities.join('+') }; });
    const errs = errors.filter((e) => !/^demand:/.test(e));
    if (errs.length) return { ok: false, step: 'console', detail: errs.slice(0, 2).join(' | ').slice(0, 300) };
    return { ok: true, detail: `${v.rail.toLocaleString('en-US')} rail trips a day, zones ${v.cities}` };
  } catch (e) {
    const f = await failed().catch(() => null);
    return { ok: false, step: 'timeout', detail: f ?? e.message.split('\n')[0] };
  } finally {
    await page.close();
  }
}

/**
 * T-492: a pack that downloads whole and does not open leaves only its zone out. `bad`'s city pack
 * (`kind` city: its .bin cut in half) or market pack (`kind` market: one zone dropped, so it
 * disagrees with the registry's row) is served broken; the zone must be flagged on the status tag,
 * tried again once and then left alone, while a line in `good` solves to a final result.
 */
async function checkBroken(ctx, base, kind, bad, good) {
  const page = await ctx.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  const item = kind === 'city' ? bad.id : `market:${bad.id}`;
  let refused = 0;
  page.on('console', (m) => { if (/would not open/.test(m.text())) refused++; });
  if (process.env.PACKS_LOG) page.on('console', (m) => console.log(`      [${kind}] ${m.type()}: ${m.text().slice(0, 300)}`));
  if (kind === 'city') {
    await page.route(`**/packs/${bad.id}.bin*`, async (route) => {
      const r = await route.fetch(), b = await r.body();
      await route.fulfill({ response: r, body: b.subarray(0, b.length >> 1) });
    });
  } else {
    await page.route(`**/packs/${bad.id}.market.json*`, async (route) => {
      const r = await route.fetch(), j = await r.json();
      j.zones.pop();
      await route.fulfill({ response: r, body: JSON.stringify(j) });
    });
  }
  const jump = async (c) => {
    await page.evaluate(([center, z]) => void window.tw.map.jumpTo({ center, zoom: z }), [c.center, Math.max(c.zoom, 11)]);
    await page.waitForFunction((id) => window.twDemand.activeCity.peek() === id, c.id, { timeout: 60000, polling: 100 });
  };
  const tag = () => page.evaluate(() => { const t = document.querySelector('.status-tag'); return t ? { text: t.textContent, tip: t.getAttribute('data-tip') } : null; });
  const state = () => page.evaluate((item) => { const c = window.twDemand.client; return { fails: c.openFails.get(item) ?? 0, out: c.broken.has(item), status: String(c.status) }; }, item);
  try {
    await page.goto(`${base}${base.includes('?') ? '&' : '?'}debug=1&new=1&opened=all&paused=1&v=${Date.now()}`);
    await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 120000 });
    await jump(bad);
    await page.waitForFunction((item) => window.twDemand.client.openFails.get(item) >= 1, item, { timeout: 90000, polling: 250 });
    let t = await tag();
    if (!t?.text.includes('could not load')) return { ok: false, step: 'flag', detail: `status tag ${JSON.stringify(t)}` };
    // the rest of the world carries on while the bad zone waits for its one retry
    await jump(good);
    const line = await acrossLine(page, good.center);
    if (line === null) return { ok: false, step: 'build', detail: 'no line built' };
    await page.waitForFunction(() => {
      const d = window.twDemand, s = d?.log?.at(-1), v = d?.view?.();
      return String(d.client.status).startsWith('failed') || (s && v && v.version === s.version && v.final && !d.client.waiting && v.lines.size > 0);
    }, undefined, { timeout: 240000, polling: 250 });
    let s = await state();
    if (s.status.startsWith('failed')) return { ok: false, step: 'solve', detail: s.status };
    // back in view the zone is tried once more, fails the same way, and is not tried again
    await jump(bad);
    await page.waitForFunction((item) => window.twDemand.client.openFails.get(item) >= 2, item, { timeout: 60000, polling: 250 });
    await page.waitForTimeout(3000);
    s = await state();
    t = await tag();
    if (s.status.startsWith('failed') || !s.out || s.fails !== 2) return { ok: false, step: 'retry', detail: JSON.stringify(s) };
    if (!t?.tip?.includes('stays out')) return { ok: false, step: 'flag', detail: `status tag ${JSON.stringify(t)}` };
    await page.waitForTimeout(2000);
    if (refused !== 2) return { ok: false, step: 'retry', detail: `refused ${refused} times, wanted 2` };
    if (errors.length) return { ok: false, step: 'console', detail: errors.slice(0, 2).join(' | ').slice(0, 300) };
    return { ok: true, detail: `${bad.id} ${kind} pack flagged and tried twice, ${good.id} solved` };
  } catch (e) {
    return { ok: false, step: 'timeout', detail: e.message.split('\n')[0] };
  } finally {
    await page.close();
  }
}

const args = process.argv.slice(2);
const brokenMode = args.includes('--broken');
const given = args.find((a) => /^https?:/.test(a));
const only = args.filter((a) => a !== given && a !== '--broken');
const zones = registry().cities;
const cities = zones.filter((c) => !only.length || only.includes(c.id));
let vite = null, failures = 0, chrome = null;
const t0 = Date.now();
try {
  if (!given) vite = await startVite(Number(process.env.PACKS_PORT ?? 6191));
  const base = given ?? vite.url;
  console.log(`      ${brokenMode ? 'broken packs' : cities.length + ' zones'} on ${base}`);
  chrome = await launchChrome({ prefix: 'tw-packs-', size: [1200, 800] });
  const ctx = chrome.ctx;
  if (brokenMode) {
    const byId = (id) => zones.find((c) => c.id === id);
    for (const kind of only.length ? only : ['city', 'market']) {
      const r = await checkBroken(ctx, base, kind, byId('nhv'), byId('hfd'));
      if (!r.ok) failures++;
      console.log(`${r.ok ? 'PASS' : 'FAIL'}  broken ${kind} pack  ${r.ok ? '' : r.step + ': '}${r.detail}`);
    }
  } else for (const city of cities) {
    const r = await checkZone(ctx, base, city);
    if (!r.ok) failures++;
    console.log(`${r.ok ? 'PASS' : 'FAIL'}  ${city.id} ${city.name}  ${r.ok ? '' : r.step + ': '}${r.detail}`);
  }
} finally {
  await chrome?.close();
  if (vite) vite.child.kill();
}
const secs = ((Date.now() - t0) / 1000).toFixed(0);
const what = brokenMode ? `${only.length || 2} broken packs` : `${cities.length} zones`;
console.log(failures ? `PACKS FAILED: ${failures} of ${what}, ${secs} s` : `PACKS PASSED: ${what}, ${secs} s`);
process.exit(failures ? 1 : 0);
