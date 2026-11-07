// T-2588: the key of a save's network inputs (demandCache.ts `saveNetKey`) is the same on every load of
// one save, so the demand cache's read ahead of the network (T-1539) hits; and a small edit gives
// another key. A big save is opened as a file (the player's path), then RUNS reloads into its slot, the
// clock running between them; each load logs the clock worker's key, the read ahead's result, and the
// key the slot's stored bytes give now, with and without the save's eras tail (T-1071's trains changing
// over, which the snapshot never reads). Then one line's trains +1, the slot flushed: its key must differ.
//   node trainworld/checks/t2588-netkey.mjs    (from the maps root; TW_CORES=3 or a ledger claim)
//   env: BASE (http://localhost:6178/), SAVE (Anita's North America save), RUNS (3), OUT (artifacts/trainworld/t2588)
// PASS lines: one key on every load; the read ahead hits on every reload after the first; the edit's key differs.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { launchChrome, collectErrors, demandFinal, ARTIFACTS } from './lib.mjs';

const BASE = process.env.BASE ?? 'http://localhost:6178/';
const SAVE = process.env.SAVE ?? join(ARTIFACTS, 'anita-2026-11-05/north-america-day-361.save');
const RUNS = Number(process.env.RUNS ?? 3);
const OUT = process.env.OUT ?? join(ARTIFACTS, 't2588');
await mkdir(OUT, { recursive: true });
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);

const { ctx, close } = await launchChrome({ profile: join(OUT, 'profile') });
const page = ctx.pages()[0] ?? (await ctx.newPage());
const errors = [];
collectErrors(page, errors);
// DIAG=1: what the read ahead compared, the zones the stored result covered against the zones the load needed
if (process.env.DIAG) await ctx.addInitScript(() => {
  const w = (window.__diag = []);
  const iv = setInterval(() => {
    const c = window.twDemand?.client;
    if (!c) return;
    clearInterval(iv);
    const tc = c.tryCache.bind(c), sn = c.solveNeed.bind(c);
    let inAhead = false;
    c.tryCache = async (svc, ahead) => {
      if (ahead) { inAhead = true; w.push({ t: performance.now(), stored: Object.keys(ahead.packs ?? {}).sort() }); }
      try { return await tc(svc, ahead); } finally { if (ahead) inAhead = false; }
    };
    c.solveNeed = (svc) => {
      const r = sn(svc);
      if (inAhead) w.push({ t: performance.now(), need: r && [...r].sort(), fetching: [...c.fetching].sort(), prefetched: [...c.prefetched].sort(), opened: [...c.opened].sort() });
      return r;
    };
  }, 5);
});

/** the key the slot's stored bytes give now (as the clock worker works it out on a load), and the same
 * with the eras tail left on (the key before T-2588), and the tail's length */
const slotKeys = () => page.evaluate(async () => {
  const [sf, dc, cc] = await Promise.all([import('/src/workers/saveFile.ts'), import('/src/workers/demandCache.ts'), import('/src/workers/clockClient.ts')]);
  const slot = cc.saveSlotsState.peek().active;
  const st = await sf.readStored(slot);
  const { header, track } = await sf.decodeSave(st.bytes);
  const h = sf.currentZones(header);
  const tail = new TextDecoder().decode(track.subarray(track.length - 4));
  const eras = tail.startsWith('ERA') ? new DataView(track.buffer, track.byteOffset).getUint32(track.length - 8, true) + 8 : 0;
  const head = new TextEncoder().encode(JSON.stringify([h.city ?? '', h.fares ?? null, h.fleet ?? 0, h.fleetFast ?? 0, h.fleetBus ?? 0, h.mode ?? 'normal']));
  const all = new Uint8Array(track.length + head.length); all.set(track); all.set(head, track.length);
  const whole = [...new Uint8Array(await crypto.subtle.digest('SHA-256', all))].map((b) => b.toString(16).padStart(2, '0')).join('');
  return { slot, at: st.at, day: st.day, key: await dc.saveNetKey(track, h), withEras: whole, eras, trackBytes: track.length };
});
const short = (k) => k?.slice(0, 8) ?? '-';
const ready = () => page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client && window.twViews, undefined, { timeout: 300000, polling: 100 });
const flush = async () => {
  const before = await page.evaluate(() => window.tw.lastSaved.peek());
  await page.evaluate(() => window.tw.client.post({ kind: 'flush' }));
  // (a flush with nothing new to write may write nothing)
  await page.waitForFunction((b) => window.tw.lastSaved.peek() > b, before, { timeout: 60000 }).catch(() => {});
};
/** after a load: the key the clock worker sent, and the read ahead's result ("hit" or its miss) */
const loadState = () => page.evaluate(() => {
  const c = window.twDemand.client, ahead = c.cacheLogs.filter((l) => l.ahead);
  return { diag: window.__diag ?? null, key: c.loadKey?.key ?? null, ahead: ahead.map((l) => l.miss ?? 'hit'), atNetwork: c.cacheLogs.filter((l) => !l.ahead).map((l) => l.miss ?? 'hit') };
});

const out = { save: SAVE, base: BASE, loads: [], errors };
let ok = true;
try {
  await page.goto(`${BASE}?debug=1&paused=1&v=${Date.now()}`);
  await ready();
  const bytes = await readFile(SAVE);
  await page.evaluate(async (data) => { const m = await import('/src/workers/clockClient.ts'); await m.importGame(new File([Uint8Array.from(data)], 'game.save')); }, Array.from(bytes));
  await page.waitForURL(/load=file/, { timeout: 120000 });
  await ready();
  await page.waitForFunction(() => window.tw.world.value.save.lines.length > 0, undefined, { timeout: 300000, polling: 100 });
  await demandFinal(page, { timeout: 1200000 });
  await page.waitForFunction(() => window.twDemand.client.cacheLog?.wrote, undefined, { timeout: 120000, polling: 200 }).catch(() => null);
  await flush();
  const s0 = await loadState(), k0 = await slotKeys();
  out.loads.push({ kind: 'import', ...s0, slotNow: k0 });
  log(`import: worker key ${short(s0.key)}; slot now ${short(k0.key)} (with eras ${short(k0.withEras)}, eras ${k0.eras} of ${k0.trackBytes} bytes, day ${k0.day})`);
  for (let i = 0; i < RUNS; i++) {
    const hash = await page.evaluate(() => location.hash);
    await page.goto(`${BASE}?debug=1&slot=${encodeURIComponent(k0.slot)}&v=${Date.now()}${hash}`);
    await ready();
    await demandFinal(page, { timeout: 1200000 });
    await page.waitForFunction(() => window.twDemand.client.cacheLog?.wrote, undefined, { timeout: 120000, polling: 200 }).catch(() => null);
    await page.waitForTimeout(5000); // the clock runs on, as in play
    await flush();
    const s = await loadState(), k = await slotKeys();
    out.loads.push({ kind: 'reload', i, ...s, slotNow: k });
    log(`reload ${i}: worker key ${short(s.key)}, read ahead ${s.ahead.join(',') || '-'}, at the network ${s.atNetwork.join(',') || '-'}; slot now ${short(k.key)} (with eras ${short(k.withEras)}, eras ${k.eras} bytes, day ${k.day})`);
  }
  const reloads = out.loads.filter((l) => l.kind === 'reload');
  const keys = new Set(reloads.map((l) => l.key));
  const sameKey = keys.size === 1 && !keys.has(null) && reloads.every((l) => l.slotNow.key === reloads[0].key);
  ok &&= sameKey;
  console.log(`${sameKey ? 'PASS' : 'FAIL'}  one network key on every reload and in the slot after it (${reloads.map((l) => `${short(l.key)}/${short(l.slotNow.key)}`).join(', ')}; with the eras tail ${reloads.map((l) => short(l.slotNow.withEras)).join(', ')})`);
  const hits = reloads.slice(1).every((l) => l.ahead.length > 0 && l.ahead.every((a) => a === 'hit'));
  ok &&= hits;
  console.log(`${hits ? 'PASS' : 'FAIL'}  the read ahead of the network hits on reloads 2-${RUNS} (${reloads.map((l, i) => `${i + 1}: ${l.ahead.join(',') || '-'}`).join('; ')})`);
  // a small edit: one line's trains +1
  const pick = await page.evaluate(() => {
    const w = window.tw.world.value;
    const l = w.save.lines.find((x) => w.lineStats?.[x.id]?.trainsNeeded?.[0] > 0);
    return { num: l.num, trains: w.lineStats[l.id].trainsNeeded.slice(0, 3) };
  });
  await page.evaluate((p) => window.tw.client.edit({ op: 'setTrainCount', line: p.num, trains: p.trains.map((t) => t + 1) }), pick);
  await page.waitForTimeout(1500);
  await flush();
  const ke = await slotKeys();
  out.edit = { pick, slotNow: ke };
  const differs = !!ke.key && ke.key !== reloads[0].key;
  ok &&= differs;
  console.log(`${differs ? 'PASS' : 'FAIL'}  line ${pick.num}'s trains +1 gives another key (${short(ke.key)} against ${short(reloads[0].key)})`);
} catch (e) {
  ok = false;
  console.log('FAIL  ' + (e?.stack ?? e));
}
const errs = errors.filter((e) => !/favicon|openfreemap|AJAXError/.test(e));
if (errs.length) { ok = false; console.log('FAIL  console errors: ' + errs.slice(0, 3).join(' | ')); }
await writeFile(join(OUT, 't2588-netkey.json'), JSON.stringify(out, null, 1));
await close();
process.exit(ok ? 0 : 1);
