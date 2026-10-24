// T-1007: forced panics in the clock worker (dev server only), recovery and state: a panicking edit
// is undone, an idle or read panic keeps the state, the fourth in a minute leaves the model down.
//   BASE=http://localhost:6132/ node trainworld/checks/t1007-panic.mjs   (about 2 minutes)
import { readFile } from 'node:fs/promises';
import { launchChrome, importSave, demandFinal, REAL_SAVE } from './lib.mjs';
const BASE = process.env.BASE ?? 'http://localhost:6132/';
const { ctx, close } = await launchChrome({ prefix: 'tw-t1007-' });
const page = await ctx.newPage();
const consoleErrors = [], pageErrors = [];
page.on('pageerror', (e) => pageErrors.push(e.message));
page.on('console', (m) => { if (m.type() === 'error' && !/openfreemap|AJAXError|favicon|Failed to load resource/.test(m.text())) consoleErrors.push(m.text()); });
let fails = 0;
const check = (name, ok, detail = '') => { if (!ok) fails++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`); };
const noticeText = () => page.evaluate(async () => (await import('/src/game/world.ts')).notice.peek()?.text ?? null);
const fp = () => page.evaluate(() => {
  const w = window.tw.world.value;
  return JSON.stringify({ edges: w.edges.length, nodes: w.nodes.size, stations: w.save.stations.length, lines: w.save.lines.map((l) => `${l.name}:${l.stops.join('.')}`).join('|'), km: Math.round(w.edges.reduce((s, e) => s + e.lengthM, 0)), cash: Math.round(w.save.cash / 1000) });
});
const edit = (op) => page.evaluate((op) => window.tw.client.edit(op), op);
const settle = (ms = 2500) => page.waitForTimeout(ms);
try {
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await settle(1500);
  // ---- phase A: a new game, track placed by edits
  const route = (lng, lat, lng2, lat2) => page.evaluate(([a, b, c, d]) => {
    const p = window.tw.lngLatToPack(a, b), q = window.tw.lngLatToPack(c, d);
    return window.tw.client.edit({ op: 'route', from: { kind: 'free', x: p.x, y: p.y, level: 0 }, to: { kind: 'free', x: q.x, y: q.y, level: 0 }, pis: [], single: false });
  }, [lng, lat, lng2, lat2]);
  const f0 = await fp();
  const a1 = await route(-74.30, 40.65, -74.24, 40.65);
  check('A1 track placed before any panic', a1.ok, JSON.stringify(a1.issues));
  const f1 = await fp();
  check('A1 changed the state', f1 !== f0);
  // a panic during an edit, after it was applied in the model: the edit is undone
  await page.evaluate(() => window.tw.client.debugPanic('afterEdit'));
  const a2 = await route(-74.22, 40.62, -74.16, 40.62);
  check('A2 panicking edit answers not ok', !a2.ok && a2.issues.includes('Internal'), JSON.stringify(a2));
  await settle();
  check('A2 notice says the edit was undone', (await noticeText()) === 'That edit failed and was undone.', String(await noticeText()));
  check('A2 state equals the state before the edit', (await fp()) === f1, await fp());
  check('A2 panic logged to the console', consoleErrors.some((t) => /Rust panic in edit/.test(t)), consoleErrors.at(-1)?.slice(0, 200));
  // track can be placed again
  const a3 = await route(-74.22, 40.62, -74.16, 40.62);
  check('A3 track placed after the panic', a3.ok, JSON.stringify(a3.issues));
  const f3 = await fp();
  check('A3 changed the state', f3 !== f1);
  const pv = await page.evaluate(() => new Promise((res) => {
    const p = window.tw.lngLatToPack(-74.40, 40.60), q = window.tw.lngLatToPack(-74.35, 40.60);
    window.tw.client.preview({ from: { kind: 'free', x: p.x, y: p.y, level: 0 }, to: { kind: 'free', x: q.x, y: q.y, level: 0 }, pis: [], single: false }, (a) => res({ ok: a.ok, pts: a.pts.length, issues: a.issues }));
  }));
  check('A4 ghost preview works', pv.ok && pv.pts > 0, JSON.stringify(pv));
  // a panic with nothing in flight (the model's borrow held): the state stays
  await page.evaluate(() => window.tw.client.debugPanic('now'));
  await settle();
  check('A5 idle panic keeps the state', (await fp()) === f3, await fp());
  check('A5 notice about the restore', /restored to your last edit/.test(String(await noticeText())), String(await noticeText()));
  // a panic in a read-only call: the live network is used
  await page.evaluate(() => window.tw.client.debugPanic('read'));
  await settle();
  check('A6 read panic keeps the state', (await fp()) === f3);
  // a query still answers
  const q = await page.evaluate(() => window.tw.client.query({ what: 'lineMarks', line: 0 }).then((d) => d.length >= 0));
  check('A7 queries answer', q === true);
  // ---- the fourth panic in a minute leaves the model down, with a reload notice
  await page.evaluate(() => window.tw.client.debugPanic('now'));
  await settle();
  check('B1 down notice', /Reload the page/.test(String(await noticeText())), String(await noticeText()));
  const a4 = await route(-74.0, 40.58, -73.95, 40.58);
  check('B2 edits are refused cleanly when down', !a4.ok, JSON.stringify(a4));
  // ---- phase C: the real save, after a reload (own page state)
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await settle(1500);
  const bytes = await readFile(REAL_SAVE);
  await importSave(page, bytes);
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await settle(3000);
  const c0 = await fp();
  await page.evaluate(() => window.tw.client.debugPanic('now'));
  await settle(4000);
  const c1 = await fp();
  check('C1 real save: state after an idle panic equals before', c1 === c0, c0 === c1 ? '' : c0 + '\n' + c1);
  const pick = await page.evaluate(() => {
    const tw = window.tw, w = tw.world.value, net = tw.renderer.net;
    const e = w.edges.find((e) => e.lines.length === 1 && e.lengthM > 800 && net.edgePts.get(e.id)?.length >= 6);
    const p = net.edgePts.get(e.id), n = p.length / 3;
    const [lng, lat] = tw.toLngLat((p[0] + p[(n - 1) * 3]) / 2, (p[1] + p[(n - 1) * 3 + 1]) / 2, net.anchor ?? tw.anchor());
    return { edge: e.id, lng, lat };
  });
  const st = await page.evaluate(async (p) => { const q = window.tw.lngLatToPack(p.lng, p.lat); const a = await window.tw.client.edit({ op: 'addStation', at: { kind: 'edge', edge: p.edge, x: q.x, y: q.y }, platform: 320, name: '' }); return a; }, pick);
  check('C2 station added on the real save', st.ok, JSON.stringify(st.issues));
  const c2 = await fp();
  await page.evaluate(() => window.tw.client.debugPanic('afterEdit'));
  const st2 = await page.evaluate(async (p) => { const q = window.tw.lngLatToPack(p.lng, p.lat); return window.tw.client.edit({ op: 'constructAll' }); }, pick);
  check('C3 panicking constructAll answers not ok', !st2.ok);
  await settle(4000);
  check('C3 real save: state equals the one before the panicking edit', (await fp()) === c2);
  const st3 = await page.evaluate(() => window.tw.client.edit({ op: 'constructAll' }));
  check('C4 constructAll works after the panic', st3.ok, JSON.stringify(st3.issues));
  check('C4 cash fell', JSON.parse(await fp()).cash < JSON.parse(c2).cash);
  const und = await page.evaluate(() => window.tw.client.edit({ op: 'undo' }));
  check('C5 undo after a rebuild is a refusal or fine (no crash)', typeof und.ok === 'boolean');
  // the clock runs on after a rebuild
  await demandFinal(page, { timeout: 240000 });
  const t0 = await page.evaluate(() => { window.tw.clock.setRunning(true); return window.tw.clock.now(); });
  await settle(4000);
  check('C6 demand final again and the game clock advances', (await page.evaluate(() => window.tw.clock.now())) > t0);
  check('no page errors', pageErrors.length === 0, pageErrors.slice(0, 3).join(' | '));
  const unexpected = consoleErrors.filter((t) => !/Rust panic in/.test(t));
  check('no other console errors', unexpected.length === 0, unexpected.slice(0, 3).join(' | '));
} finally {
  await close();
}
console.log(fails ? `${fails} FAILED` : 'ALL PASS');
process.exit(fails ? 1 : 0);
