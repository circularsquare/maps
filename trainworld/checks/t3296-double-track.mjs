// T-3296, T-3321: single track rebuilt as double, and double as single, with the track tool's 1-line and 2-line
// icons in the Select tool's bar. On a new game in New York: a built single-track run of three pieces with a
// station on each joint and a line over it. The whole run picked: the 1-line icon is on; the 2-line icon chosen,
// the Rebuild hover shows the price before anything is spent, Rebuild asks, then every piece is double, the cash
// taken is the price shown and the line still runs. Then back: the 2-line icon is on, the 1-line icon chosen, the
// hover says Free, the question says it's free, nothing is taken, every piece is single and the line runs. Then
// blueprint: a single run made double at once, free, and Ctrl+Z brings it back; made double and then single again,
// free; and blueprint high-speed track refused with its reason. The bar is shot in both themes.
// Screenshots artifacts/trainworld/t3296/. From the maps root, on a Vite of your own:
//   node trainworld/checks/t3296-double-track.mjs [dev url]
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6157/';
const OUT = join(ARTIFACTS, 't3296');
mkdirSync(OUT, { recursive: true });
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const C = [-73.93, 40.668];
const lngAt = (dx) => C[0] + dx / (111320 * Math.cos((C[1] * Math.PI) / 180));
const latAt = (dy) => C[1] + dy / 110540;

const chrome = await launchChrome({ prefix: 'tw-t3296-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
collectErrors(page, errors);

const snap = () => page.evaluate(() => {
  const w = window.tw.world.value;
  return {
    cash: w.save.cash,
    edges: w.edges.map((e) => ({ id: Number(e.id), built: e.built, tracks: e.tracks, hsr: !!e.hsr, y: w.nodes.get(e.a)?.y })),
    lines: w.save.lines.map((l) => ({ num: Number(l.num), stops: l.stops.map(Number), status: w.lineStats[l.id]?.status })),
  };
});
const pick = (edges, built) => page.evaluate(async ([edges, built]) => { (await import('/src/game/blueprintPick.ts')).bpPicked.value = { edges, stations: [], built }; }, [edges, built]);
const jump = (lng, lat, zoom) => page.evaluate(([lng, lat, zoom]) => window.tw.map.jumpTo({ center: [lng, lat], zoom }), [lng, lat, zoom]);
const notice = () => page.evaluate(async () => (await import('/src/game/world.ts')).notice.value?.text ?? '');

/** A run through `xs` (m east of C, at `dy` m north) with a station on every joint, single track unless `hsr` (then
 * high-speed double); built with a line when `construct`. */
const build = (xs, dy, construct, hsr = false) => page.evaluate(async ([c, xs, dy, construct, hsr]) => {
  const tw = window.tw, cl = tw.client, o = tw.lngLatToPack(c[0], c[1]);
  const pause = (ms) => new Promise((d) => setTimeout(d, ms));
  const k = 1 / Math.cos((c[1] * Math.PI) / 180);
  const P = (dx) => ({ x: o.x + dx * k, y: o.y + dy * k });
  const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p.x, n.y - p.y); if (d < bd) { bd = d; best = n.id; } } return best; };
  const res = [];
  let from = { kind: 'free', ...P(xs[0]), level: 0 };
  for (let i = 1; i < xs.length; i++) {
    res.push(await cl.edit({ op: 'route', from, to: { kind: 'free', ...P(xs[i]), level: 0 }, pis: [], single: !hsr, hsr })); await pause(250);
    from = { kind: 'node', node: near(P(xs[i])) };
  }
  const st = hsr ? [] : xs.map((x) => near(P(x)));
  for (const n of st) { res.push(await cl.edit({ op: 'addStation', at: { kind: 'node', node: n }, platform: 160, name: '' })); await pause(200); }
  let line = { ok: true, line: -1 };
  if (construct) {
    res.push(await cl.edit({ op: 'constructAll' })); await pause(500);
    line = await cl.edit({ op: 'addLine', stops: st, name: 'Single line', colour: '#c0392b' });
    await pause(800);
  }
  return { st, line: line.line, ok: line.ok, refused: res.filter((r) => !r.ok).map((r) => JSON.stringify(r.issues ?? r).slice(0, 160)) };
}, [C, xs, dy, construct, hsr]);

const bar = page.locator('.blueprint-bar');
const one = bar.locator('button[aria-label="Single track"]');
const two = bar.locator('button[aria-label="Double track"]');
const rb = bar.locator('button', { hasText: 'Rebuild' });
const pressed = async () => [await one.getAttribute('aria-pressed'), await two.getAttribute('aria-pressed')].join(' ');
/** The bar and a little round it. */
const barShot = async (name) => {
  const b = await bar.boundingBox();
  await page.screenshot({ path: join(OUT, name), clip: { x: Math.max(0, b.x - 16), y: Math.max(0, b.y - 12), width: b.width + 32, height: b.height + 24 } });
};
const price = (tip) => { const m = /\$([\d.,]+)\s*([MB])/.exec(tip ?? ''); return m ? Number(m[1].replace(/,/g, '')) * (m[2] === 'B' ? 1000 : 1) : NaN; };

try {
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 180000 });
  await page.evaluate(async () => { (await import('/src/game/ui.ts')).tool.value = 'blueprint'; });

  const run = await build([-5000, -2500, 2500, 5000], 0, true);
  check(run.ok && !run.refused.length, 'the single run, its stations and its line are built', run.refused.join(' | '));
  if (!run.ok) throw new Error('no line');
  let s = await snap();
  const ids = s.edges.map((e) => e.id);
  check(ids.length === 3 && s.edges.every((e) => e.tracks === 1 && e.built), 'three built single-track pieces', JSON.stringify(s.edges.map((e) => e.tracks)));
  await jump(lngAt(-2500), latAt(0), 17.5);
  await page.waitForTimeout(1500);
  await page.screenshot({ path: join(OUT, 'before-station.png') });

  // single to double
  await jump(C[0], C[1], 11.8);
  await pick(ids, true);
  await page.waitForTimeout(600);
  check((await one.count()) === 1 && (await two.count()) === 1, 'the bar has the 1-line and 2-line icons');
  check(await pressed() === 'true false', 'single track picked: the 1-line icon is on', await pressed());
  await barShot('bar-single.png');
  await page.screenshot({ path: join(OUT, 'before-pick.png') });
  await two.click();
  await page.waitForTimeout(1500);
  check(await pressed() === 'false true', 'the 2-line icon chosen', await pressed());
  let tip = await rb.getAttribute('data-tip');
  console.log('hover:', JSON.stringify(tip));
  check(/double track/.test(tip ?? '') && /\$/.test(tip ?? ''), 'the Rebuild hover names double track and shows the price before the click', tip);
  await barShot('bar-to-double.png');
  await page.screenshot({ path: join(OUT, 'pick-double.png') });
  let before = await snap();
  await rb.click();
  await page.waitForSelector('dialog.confirmation[open]', { timeout: 5000 });
  let ask = await page.locator('dialog.confirmation #confirmation-text').innerText();
  console.log('confirm:', ask);
  check(/double track/.test(ask) && /4 built stations/.test(ask), 'the question names double track and the 4 stations', ask);
  await page.screenshot({ path: join(OUT, 'confirm.png') });
  await page.locator('dialog.confirmation .btn.danger').click();
  await page.waitForTimeout(2500);
  s = await snap();
  console.log('notice:', await notice());
  const p2 = price(tip);
  check(s.edges.every((e) => e.tracks === 2), 'every piece is double', JSON.stringify(s.edges.map((e) => e.tracks)));
  check(Math.abs((before.cash - s.cash) / 1e6 - p2) / p2 < 0.01, 'the cash taken is the price shown', `taken ${(before.cash - s.cash) / 1e6}M, shown ${p2}M`);
  let ln = s.lines.find((l) => l.num === run.line);
  check(ln && ln.status !== 'broken', 'the line still runs', ln?.status);
  await pick([], false);
  await jump(C[0], C[1], 11.8);
  await page.waitForTimeout(1500);
  await page.screenshot({ path: join(OUT, 'after.png') });
  await jump(lngAt(-2500), latAt(0), 17.5);
  await page.waitForTimeout(1500);
  await page.screenshot({ path: join(OUT, 'after-station.png') });

  // and back to single: free
  await jump(C[0], C[1], 11.8);
  await pick(ids, true);
  await page.waitForTimeout(600);
  check(await pressed() === 'false true', 'double track picked: the 2-line icon is on', await pressed());
  await barShot('bar-double.png');
  await one.click();
  await page.waitForTimeout(1500);
  check(await pressed() === 'true false', 'the 1-line icon chosen', await pressed());
  tip = await rb.getAttribute('data-tip');
  console.log('hover:', JSON.stringify(tip));
  check(/single track/.test(tip ?? '') && /\nFree/.test(tip ?? '') && !/\$/.test(tip ?? ''), 'the Rebuild hover names single track and says Free', tip);
  await barShot('bar-to-single.png');
  before = await snap();
  await rb.click();
  await page.waitForSelector('dialog.confirmation[open]', { timeout: 5000 });
  ask = await page.locator('dialog.confirmation #confirmation-text').innerText();
  console.log('confirm:', ask);
  check(/single track/.test(ask) && /It's free/.test(ask) && !/\$/.test(ask), 'the question names single track and says it is free', ask);
  await page.screenshot({ path: join(OUT, 'confirm-single.png') });
  await page.locator('dialog.confirmation .btn.danger').click();
  await page.waitForTimeout(2500);
  s = await snap();
  console.log('notice:', await notice());
  check(s.edges.every((e) => e.tracks === 1), 'every piece is single again', JSON.stringify(s.edges.map((e) => e.tracks)));
  check(s.cash === before.cash, 'nothing taken, nothing back', `${before.cash} -> ${s.cash}`);
  ln = s.lines.find((l) => l.num === run.line);
  check(ln && ln.status !== 'broken', 'the line still runs on single track', ln?.status);
  await pick([], false);

  // blueprint: at once, free, one undo
  const bp = await build([-5000, -2500, 2500, 5000], 3000, false);
  check(bp.ok && !bp.refused.length, 'a blueprint single run is drawn', bp.refused.join(' | '));
  s = await snap();
  const bids = s.edges.filter((e) => !e.built).map((e) => e.id);
  const bpTracks = (s) => s.edges.filter((e) => !e.built).map((e) => e.tracks);
  check(bids.length === 3 && bpTracks(s).every((t) => t === 1), 'three blueprint single pieces');
  const cash0 = s.cash;
  await jump(C[0], latAt(3000), 11.8);
  await pick(bids, false);
  await page.waitForTimeout(600);
  await two.click();
  await page.waitForTimeout(1500);
  s = await snap();
  check(bpTracks(s).every((t) => t === 2) && s.cash === cash0, 'blueprint to double at once and free', JSON.stringify(bpTracks(s)));
  check(await pressed() === 'false true', 'and the 2-line icon is on', await pressed());
  await page.screenshot({ path: join(OUT, 'blueprint-double.png') });
  await page.keyboard.press('Control+z');
  await page.waitForTimeout(1500);
  s = await snap();
  check(bpTracks(s).every((t) => t === 1), 'Ctrl+Z brings the single track back', JSON.stringify(bpTracks(s)));
  await two.click();
  await page.waitForTimeout(1200);
  await one.click();
  await page.waitForTimeout(1500);
  s = await snap();
  console.log('notice:', await notice());
  check(bpTracks(s).every((t) => t === 1) && s.cash === cash0, 'blueprint double to single at once and free', JSON.stringify(bpTracks(s)));
  await page.keyboard.press('Control+z');
  await page.waitForTimeout(1500);
  s = await snap();
  check(bpTracks(s).every((t) => t === 2), 'Ctrl+Z brings the double track back', JSON.stringify(bpTracks(s)));
  await pick([], false);

  // blueprint high-speed double goes single and keeps its type (T-3352)
  const hs = await build([-5000, 5000], 6000, false, true);
  check(hs.ok && !hs.refused.length, 'a blueprint high-speed run is drawn', hs.refused.join(' | '));
  s = await snap();
  const hids = s.edges.filter((e) => !e.built && !bids.includes(e.id)).map((e) => e.id);
  await jump(C[0], latAt(6000), 11.8);
  await pick(hids, false);
  await page.waitForTimeout(600);
  await one.click();
  await page.waitForTimeout(1500);
  const said = await notice();
  console.log('notice:', said);
  s = await snap();
  check(s.edges.filter((e) => hids.includes(e.id)).every((e) => e.tracks === 1 && e.hsr), 'high-speed track goes single and stays high speed', said);
  await pick([], false);

  // the bar in the dark theme, a built double pick with single chosen
  await page.evaluate(async () => { (await import('/src/game/ui.ts')).shade.value = 'dark'; });
  await jump(C[0], C[1], 11.8);
  await pick(ids, true);
  await page.waitForTimeout(800);
  await barShot('bar-single-dark.png');
  await two.click();
  await page.waitForTimeout(1200);
  await barShot('bar-to-double-dark.png');
} catch (e) {
  fail++;
  console.log('FAIL exception', e?.stack ?? e);
}
check(!errors.length, 'no console or page errors', errors.slice(0, 3).join(' | '));
await chrome.close();
console.log(fail ? `FAIL: ${fail} failed` : 'PASS: all checks');
process.exit(fail ? 1 : 0);
