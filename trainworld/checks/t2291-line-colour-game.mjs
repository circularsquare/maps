// T-2291 in the game: a new line's automatic colour is re-picked when its stops put it beside a line of a
// near-identical colour, and a colour the player set stays.
//   node trainworld/checks/t2291-line-colour-game.mjs [url]    (from the maps root; default http://localhost:6182/)
// Needs ../artifacts/trainworld/t2290/line8.save (`node trainworld/checks/t2290-line8-game.mjs` makes it): five
// lines in New York, line 3 through Homestead Park and Rye Neck. A new line is made by the N key; line 3 is
// recoloured to a near copy of its colour, and clicking Homestead Park and Rye Neck puts the new line beside it.
import { readFile } from 'node:fs/promises';
import { ARTIFACTS, NOISE, launchChrome, collectErrors, importSave } from './lib.mjs';
import { loadGame } from './t2291-load.mjs';

await loadGame('palette');
const { besideColours, deltaE, APART } = await loadGame('colourPick');

const BASE = process.argv[2] ?? 'http://localhost:6182/';
const results = [];
const report = (name, ok, detail = '') => {
  results.push(ok);
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`);
};

const { ctx, close } = await launchChrome({ prefix: 'tw-t2291-' });
const errors = [];
try {
  const page = await ctx.newPage();
  collectErrors(page, errors, NOISE);
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(ARTIFACTS + 't2290/line8.save'));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.stations.length > 0, undefined, { timeout: 120000 });
  await page.waitForTimeout(1500);

  const S = await page.evaluate(() => Object.fromEntries(window.tw.world.value.save.stations.map((s) => [s.name, { num: s.num, lng: s.lng, lat: s.lat }])));
  const lineOf = async (num) => {
    const w = await page.evaluate(() => { const v = window.tw.world.value; return { lines: v.save.lines.map((l) => ({ id: l.id, num: l.num, colour: l.colour, stops: l.stops })), edges: v.edges.map((e) => e.lines), at: Object.fromEntries(v.save.stations.map((s) => [s.id, [s.lng, s.lat]])) }; });
    const l = w.lines.find((x) => x.num === num);
    return l && { colour: l.colour.toLowerCase(), stops: l.stops.length, beside: besideColours(l, w.lines, w.edges, (s) => w.at[s] ?? null).map((c) => c.toLowerCase()) };
  };
  const view = (lng, lat, zoom) => page.evaluate(([lng, lat, zoom]) => window.tw.map.jumpTo({ center: [lng, lat], zoom }), [lng, lat, zoom]);
  const newLine = async (random = 0) => {
    await page.evaluate(() => { document.activeElement?.blur?.(); window.tw.tools.exit(); window.tw.selection.value = null; });
    const before = await page.evaluate(() => Math.max(-1, ...window.tw.world.value.save.lines.map((l) => l.num)));
    await page.evaluate((r) => { window.__random = Math.random; Math.random = () => r; }, random);
    await page.keyboard.press('n');
    await page.waitForFunction((b) => window.tw.world.value.save.lines.some((l) => l.num > b), before, { timeout: 30000 });
    await page.evaluate(() => { Math.random = window.__random; document.activeElement?.blur?.(); });
    await page.waitForTimeout(800);
    return page.evaluate((b) => Math.max(...window.tw.world.value.save.lines.filter((l) => l.num > b).map((l) => l.num)), before);
  };
  const click = async (num, name) => {
    const before = (await lineOf(num)).stops;
    const xy = await page.evaluate((p) => { const r = window.tw.map.getContainer().getBoundingClientRect(), q = window.tw.map.project([p.lng, p.lat]); return [r.left + q.x, r.top + q.y]; }, S[name]);
    await page.mouse.move(xy[0], xy[1]);
    await page.mouse.click(xy[0], xy[1]);
    for (let k = 0; k < 40; k++) {
      await page.waitForTimeout(250);
      if ((await lineOf(num)).stops !== before) break;
    }
    await page.waitForTimeout(1200); // the recolour is a second edit
  };
  const mid = { lng: (S['Homestead Park'].lng + S['Rye Neck'].lng) / 2, lat: (S['Homestead Park'].lat + S['Rye Neck'].lat) / 2 };
  const recolour = (n, c) => page.evaluate(([n, c]) => window.tw.client.edit({ op: 'lineLook', line: n, colour: c }), [n, c]);
  const LINE3 = await page.evaluate(() => window.tw.world.value.save.lines.find((l) => l.name === 'Line 3').num);
  const nudge = (h) => '#' + [1, 3, 5].map((i) => Math.min(255, parseInt(h.slice(i, i + 2), 16) + 6).toString(16).padStart(2, '0')).join('');

  // 1. the new line starts with a colour away from the five lines; line 3 then takes a near copy of it, so the new line's stops put it beside a near-duplicate
  await view(mid.lng, mid.lat, 11.5);
  const a = await newLine();
  const a0 = (await lineOf(a)).colour;
  const BLUEISH = nudge(a0);
  await recolour(LINE3, BLUEISH);
  await page.waitForTimeout(500);
  await click(a, 'Homestead Park');
  await click(a, 'Rye Neck');
  const A = await lineOf(a);
  const gaps = A.beside.map((c) => deltaE(c, A.colour));
  report('a new line beside a near-identical line is re-picked, at least APART from every line beside it', A.stops >= 2 && A.colour !== a0 && A.beside.includes(BLUEISH) && Math.min(...gaps) >= APART, `${a0} -> ${A.colour}, ${A.stops} stops, beside ${A.beside.join(' ')}, smallest gap ${Math.min(...gaps).toFixed(1)}`);

  // 2. a colour the player set stays
  await page.evaluate(() => window.tw.tools.exit());
  await view(mid.lng, mid.lat, 11.5);
  const b = await newLine();
  await recolour(b, nudge(BLUEISH));
  await page.waitForTimeout(500);
  await click(b, 'Homestead Park');
  await click(b, 'Rye Neck');
  const B = await lineOf(b);
  report('a colour the player set is kept beside a near-identical line', B.stops >= 2 && B.colour === nudge(BLUEISH), `${B.colour}, ${B.stops} stops, beside ${B.beside.join(' ')}`);

  report('no console or page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
} finally {
  await close();
}
const bad = results.filter((r) => !r).length;
console.log(bad ? `FAIL  ${bad} check(s) failed` : 'PASS  all');
process.exit(bad ? 1 : 0);
