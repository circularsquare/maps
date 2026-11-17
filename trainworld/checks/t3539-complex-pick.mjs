// T-3539: with a station complex selected, a click on one of its stations selects that station alone; a click on
// a station of another complex selects that complex; the same station again goes back to its complex.
// Real New York save, Times Sq-42 St. Screenshots to artifacts/trainworld/t3539_*.png. Uses the Vite on the url:
//   node trainworld/checks/t3539-complex-pick.mjs [url]
import { ARTIFACTS, REAL_SAVE, collectErrors, demandFinal, importSave, launchChrome } from './lib.mjs';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';

const BASE = process.argv[2] ?? 'http://localhost:6125/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const chrome = await launchChrome({ prefix: 'tw-t3539-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.singleStation, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await demandFinal(page, { strict: false, timeout: 300000, polling: 1000 }).catch(() => console.log('  (demand did not finish)'));
  await page.waitForTimeout(1500);
  const box = await (await page.$('#map')).boundingBox();
  const settle = async (ms = 800) => { await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }); await page.waitForTimeout(ms); };
  const px = (s) => page.evaluate((s) => { const q = window.tw.map.project([s.lng, s.lat]); return [q.x, q.y]; }, s);
  const state = () => page.evaluate(() => ({ sel: window.tw.selection.value, one: window.tw.singleStation.value }));
  const clickAt = async ([x, y]) => { await page.mouse.click(box.x + x, box.y + y); await page.waitForTimeout(500); };

  const data = await page.evaluate(() => {
    const st = window.tw.world.value.save.stations;
    const t = st.find((x) => x.name.startsWith('Times Sq') && x.built);
    const members = window.tw.complexOf(t.id).map((id) => st.find((x) => x.id === id));
    return { members: members.map((m) => ({ id: m.id, name: m.name, lng: m.lng, lat: m.lat })) };
  });
  check(data.members.length >= 2, 'Times Sq-42 St is a complex of several stations', `${data.members.length} stations`);
  const cx = data.members.reduce((a, m) => a + m.lng, 0) / data.members.length, cy = data.members.reduce((a, m) => a + m.lat, 0) / data.members.length;
  await page.evaluate(([x, y]) => window.tw.map.jumpTo({ center: [x, y], zoom: 16.5 }), [cx, cy]);
  await settle(1500);
  // two members that are far apart on screen, so a click lands on one only
  let best = null;
  for (const a of data.members) for (const b of data.members) {
    if (a.id >= b.id) continue;
    const [pa, pb] = [await px(a), await px(b)];
    const d = Math.hypot(pa[0] - pb[0], pa[1] - pb[1]);
    if (!best || d > best.d) best = { a, b, pa, pb, d };
  }
  check(best.d > 30, 'two members are drawn apart', `${Math.round(best.d)} px`);
  const { a, b } = best;
  const shot = (n) => page.screenshot({ path: join(ARTIFACTS, n) });

  await page.evaluate(() => { window.tw.selection.value = null; });
  await clickAt(best.pa);
  let s = await state();
  check(s.sel?.kind === 'station' && s.one === null && data.members.some((m) => m.id === s.sel.station), 'a click on a station of an unselected complex selects the complex', `${s.sel?.station}, alone ${s.one}`);
  await page.waitForTimeout(500);
  // before: what the old code showed after a click on the second station, the complex again
  await page.evaluate((id) => { window.tw.selection.value = { kind: 'station', station: id }; }, b.id);
  await page.waitForTimeout(900);
  await shot('t3539_before.png');
  await page.evaluate((id) => { window.tw.selection.value = { kind: 'station', station: id }; }, a.id);
  await page.waitForTimeout(500);

  await clickAt(best.pb);
  await page.waitForTimeout(500);
  s = await state();
  check(s.sel?.station === b.id && s.one === b.id, 'with the complex selected, a click on one of its stations selects it alone', `${b.name} ${s.sel?.station}, alone ${s.one}`);
  const sel = await page.evaluate(() => window.tw.renderer.selectedAlso?.size ?? -1);
  check(sel === 0, 'the map draws only that station selected', `${sel} others`);
  await shot('t3539_after.png');

  await clickAt(best.pa);
  s = await state();
  check(s.sel?.station === a.id && s.one === a.id, 'with one station alone, a click on another of the complex selects that one alone', `alone ${s.one}`);

  await clickAt(best.pa);
  s = await state();
  check(s.one === null && s.sel?.station === a.id, 'a click on the station already alone goes back to the complex', `alone ${s.one}`);

  // a station of another complex
  const other = await page.evaluate((ids) => {
    const st = window.tw.world.value.save.stations;
    const c = window.tw.map.getCenter();
    let best = null;
    for (const x of st) {
      if (!x.built || ids.includes(x.id)) continue;
      const m = window.tw.complexOf(x.id);
      if (m.some((i) => ids.includes(i))) continue;
      const d = Math.hypot(x.lng - c.lng, x.lat - c.lat);
      if (!best || d < best.d) best = { id: x.id, lng: x.lng, lat: x.lat, n: m.length, d };
    }
    return best;
  }, data.members.map((m) => m.id));
  await page.evaluate((o) => window.tw.map.jumpTo({ center: [o.lng, o.lat], zoom: 16.5 }), other);
  await settle(1500);
  await clickAt(await px(other));
  s = await state();
  check(s.sel?.kind === 'station' && s.sel.station !== a.id && s.one === null, 'a click on a station of another complex selects that complex', `${s.sel?.station}, ${other.n} stations`);
  check(errors.length === 0, 'no page errors', errors.slice(0, 3).join(' | '));
} finally {
  await chrome.close();
}
process.exit(fail ? 1 : 0);
