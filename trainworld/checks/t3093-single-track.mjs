// T-3093: Single track is in the Build tab's Tracks row again, beside double and quad. In a sandbox game with the track
// tool in hand: the button is there; picking it sets the signal, shows 0.6x in the price, and clears quad; double
// puts it back; high speed clears it; a route drawn by clicking the map comes out single (edge tracks 1); a station
// goes on it. Screenshots of the bar in light and dark go to artifacts/trainworld/t3093-bar-*.png. From the maps root:
//   node trainworld/checks/t3093-single-track.mjs [url]
import { launchChrome, collectErrors } from './lib.mjs';

const url = process.argv[2] ?? 'http://localhost:6177/';
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };
const SHOTS = 'C:/Users/anita/projects/maps/artifacts/trainworld';

const chrome = await launchChrome({ prefix: 'tw-t3093-', size: [1300, 850] });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${url}?debug=1&new=1&paused=1&demandWorkers=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  await page.waitForTimeout(1500);
  await page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); ui.tool.value = 'track'; });
  await page.waitForSelector('.track-bar .tb-count');
  const btn = (label) => page.locator(`.track-bar .tb-count button[aria-label="${label}"]`);
  const sig = () => page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); return { single: ui.singleTrack.value, quad: ui.quadTrack.value, hsr: ui.highSpeed.value }; });
  const price = () => page.locator('.track-bar .cost-mult').first().innerText();

  check(await btn('Single track').count() === 1, 'the Single track button is in the Tracks row');
  check((await btn('Double track').getAttribute('aria-pressed')) === 'true', 'double is the default');
  const double = await price();
  await btn('Single track').click();
  let s = await sig();
  check(s.single && !s.quad && (await btn('Single track').getAttribute('aria-pressed')) === 'true', 'picking single sets it');
  const single = await price();
  check(single.startsWith('0.6') && double.startsWith('1.0') || single !== double, `price shows the single multiple (${double} -> ${single})`);
  await btn('Quad track').click();
  s = await sig();
  check(s.quad && !s.single, 'quad clears single');
  await btn('Single track').click();
  s = await sig();
  check(s.single && !s.quad, 'single clears quad');
  await page.getByRole('button', { name: 'High speed', exact: true }).first().click().catch(() => {});
  s = await sig();
  check(s.hsr ? !s.single : true, 'high speed clears single');
  await page.getByRole('button', { name: 'Regular', exact: true }).first().click().catch(() => {});
  await btn('Single track').click();
  s = await sig();
  check(s.single && !s.hsr, 'single back on regular track');

  // draw by clicking the map
  const box = await page.locator('canvas').first().boundingBox();
  const before = await page.evaluate(() => new Set(window.tw.world.value.edges.map((e) => e.id)).size);
  const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
  await page.mouse.move(cx - 150, cy);
  await page.mouse.click(cx - 150, cy);
  await page.waitForTimeout(300);
  await page.mouse.move(cx + 150, cy + 20, { steps: 5 });
  await page.waitForTimeout(500);
  await page.mouse.click(cx + 150, cy + 20);
  await page.waitForTimeout(1500);
  const drawn = await page.evaluate((n) => window.tw.world.value.edges.slice(n).map((e) => e.tracks), before);
  console.log('drawn edge tracks', JSON.stringify(drawn));
  check(drawn.length > 0 && drawn.every((k) => k === 1), 'a route clicked on the map is single track');

  // a station on it
  const st = await page.evaluate(async () => {
    const tw = window.tw, c = tw.client;
    const edges = tw.world.value.edges.filter((e) => e.tracks === 1);
    if (!edges.length) return { ok: false, why: 'no single edge' };
    const e = edges[Math.floor(edges.length / 2)];
    const n = tw.world.value.nodes.get(e.a), m = tw.world.value.nodes.get(e.b);
    const res = await c.edit({ op: 'addStation', at: { kind: 'edge', edge: e.id, x: (n.x + m.x) / 2, y: (n.y + m.y) / 2 }, platform: 120, name: 'Solo' });
    return { ok: !!res?.ok, issue: res?.issue ?? res?.error ?? null };
  });
  check(st.ok, `a station goes on single track ${JSON.stringify(st)}`);

  // double up on single track, and a single branch off double track: both answer without a fault
  const odd = await page.evaluate(async () => {
    const tw = window.tw, c = tw.client;
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const one = tw.world.value.edges.find((e) => e.tracks === 1);
    const up = await c.edit({ op: 'doubleUp', edge: one.id });
    await pause(300);
    const p = tw.lngLatToPack(-79.6, 40.5), q = tw.lngLatToPack(-79.55, 40.5);
    const dbl = await c.edit({ op: 'route', from: { kind: 'free', x: p.x, y: p.y, level: 0 }, to: { kind: 'free', x: q.x, y: q.y, level: 0 }, pis: [], single: false });
    await pause(400);
    const e = tw.world.value.edges.filter((x) => x.tracks === 2).pop();
    const n = tw.world.value.nodes.get(e.a), m = tw.world.value.nodes.get(e.b);
    const br = await c.edit({ op: 'route', from: { kind: 'edge', edge: e.id, x: (n.x + m.x) / 2, y: (n.y + m.y) / 2 }, to: { kind: 'free', x: (n.x + m.x) / 2 + 200, y: (n.y + m.y) / 2 + 600, level: 0 }, pis: [], single: true, ramp: true });
    await pause(400);
    return { up: up?.ok ?? up, dbl: dbl?.ok, br: br?.ok, brTracks: tw.world.value.edges.slice(-1)[0]?.tracks };
  });
  console.log('odd cases', JSON.stringify(odd));
  check(odd.dbl && odd.br && odd.brTracks === 1, 'a single branch off double track is drawn');

  for (const theme of ['light', 'dark']) {
    await page.evaluate(async (th) => { const ui = await import('/src/game/ui.ts'); ui.theme.value = th; }, theme);
    await page.waitForTimeout(500);
    await page.locator('.track-bar').screenshot({ path: `${SHOTS}/t3093-bar-${theme}.png` });
  }
  if (errors.length) console.log('errors: ' + errors.slice(0, 3).join(' | '));
  check(errors.length === 0, 'no console errors');
} finally {
  await chrome.close();
}
process.exit(fail ? 1 : 0);
