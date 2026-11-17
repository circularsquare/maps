// T-3561: with train track picked, the pointer over a tram track end (no train snap in reach) shows the red tag
// "Train track cannot be connected to tram track"; a train end within snap reach wins and shows no tag; tram to tram
// shows none. The mirror: tram picked over a train end with no tram end in reach.
// Layout (screen px at zoom 15, y 450): train track 200..390, tram track 400..600 (loose ends at 390 and 400 are 10 px
// apart), so 600 is a tram end far from any train end and 200 a train end far from any tram end.
// Picture: artifacts/trainworld/t3561.png (train picked over the far tram end).
// From the maps root, on a Vite of your own (node trainworld/checks/dev-nowatch.mjs 6141):
//   node trainworld/checks/t3561-tram-tag.mjs [dev url]
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6141/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const AT = [-66.9146, 10.5061], Y = 450;

const chrome = await launchChrome({ prefix: 'tw-t3561-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 7, build: { level: 0, keep: false } })); } catch {} });
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&city=caa&mode=sandbox&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 180000 });
  await page.evaluate(([c]) => window.tw.map.jumpTo({ center: c, zoom: 15 }), [AT]);
  await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }).catch(() => {});
  await wait(2000);
  const made = await page.evaluate(async ([y]) => {
    const tw = window.tw, ll = (x) => tw.map.unproject([x, y]);
    const P = (x) => { const p = ll(x), q = tw.lngLatToPack(p.lng, p.lat); return { kind: 'free', x: q.x, y: q.y, level: 0 }; };
    const ok = [];
    ok.push((await tw.client.edit({ op: 'route', from: P(200), to: P(390), pis: [], single: false, tram: false })).ok);
    await new Promise((r) => setTimeout(r, 600));
    ok.push((await tw.client.edit({ op: 'route', from: P(400), to: P(600), pis: [], single: false, tram: true })).ok);
    await new Promise((r) => setTimeout(r, 800));
    const w = tw.world.value;
    return { ok, tram: w.edges.filter((e) => e.kind).length, rail: w.edges.filter((e) => !e.kind).length };
  }, [Y]);
  check(made.ok.every(Boolean) && made.tram > 0 && made.rail > 0, 'a train piece and a tram piece laid', JSON.stringify(made));
  const pick = (t) => page.evaluate(async (t) => {
    const ui = await import('/src/game/ui.ts');
    ui.tool.value = 'track'; ui.setTrackType(t); ui.snapStreets.value = false; ui.snapRealTram.value = false; ui.snapReal.value = false; ui.keepPlacing.value = false;
  }, t);
  const box = await page.evaluate(() => { const r = window.tw.map.getContainer().getBoundingClientRect(); return [r.left, r.top]; });
  const tagAt = async (x) => {
    await page.mouse.move(box[0] + x - 40, box[1] + Y + 60);
    await page.mouse.move(box[0] + x, box[1] + Y, { steps: 6 });
    await wait(500);
    return page.evaluate(() => { const t = document.querySelector('.tool-tag.build-tag'); return t && getComputedStyle(t).display !== 'none' ? { text: t.textContent, bad: t.classList.contains('bad') } : null; });
  };
  const TRAIN = 'Train track cannot be connected to tram track', TRAM = 'Tram track cannot be connected to train track';
  await pick(2);
  let r = await tagAt(600);
  check(r?.bad && r.text === TRAIN, 'train picked, far tram end: red refusal', JSON.stringify(r));
  await page.screenshot({ path: join(ARTIFACTS, 't3561.png') });
  r = await tagAt(400);
  check(!r || r.text !== TRAIN, 'train picked, tram end with a train end 10 px away: no refusal', JSON.stringify(r));
  r = await tagAt(200);
  check(!r || r.text !== TRAIN, 'train picked, over a train end: no refusal', JSON.stringify(r));
  await pick('tram');
  r = await tagAt(600);
  check(!r || (r.text !== TRAIN && r.text !== TRAM), 'tram picked, tram end: no refusal', JSON.stringify(r));
  r = await tagAt(200);
  check(r?.bad && r.text === TRAM, 'tram picked, far train end: red refusal', JSON.stringify(r));
  r = await tagAt(400);
  check(!r || r.text !== TRAM, 'tram picked, train end with a tram end 10 px away: no refusal', JSON.stringify(r));
  check(!errors.length, 'no page errors', errors.slice(0, 3).join(' | '));
} finally {
  await chrome.close?.();
}
console.log(fail ? `${fail} FAILED` : 'all passed');
process.exit(fail ? 1 : 0);
