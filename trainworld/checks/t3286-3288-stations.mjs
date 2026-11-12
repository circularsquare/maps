// T-3286: a platform lengthens through a node that is only a clicked corner of a drawn run.
// T-3288: the stations of a complex that nobody named show one name; a renamed one keeps its own.
// Screenshots go to artifacts/trainworld/t3286/.
// From the worktree root:   node trainworld/checks/t3286-3288-stations.mjs [url]
import { launchChrome, collectErrors, ARTIFACTS } from './lib.mjs';
import { mkdirSync } from 'node:fs';

const url = process.argv[2] ?? 'http://localhost:6155/';
const OUT = ARTIFACTS + 't3286/';
mkdirSync(OUT, { recursive: true });
const LAT = 40.652, LNG0 = -73.985;
const dLat = (m) => m / 111320, dLng = (m) => m / (111320 * Math.cos((LAT * Math.PI) / 180));
const at = (xm, ym) => [LNG0 + dLng(xm), LAT + dLat(ym)];

const chrome = await launchChrome({ prefix: 'tw-t3286-', size: [1200, 800] });
const errors = [];
let failed = 0;
const ok = (cond, what) => { console.log(cond ? 'PASS' : 'FAIL', what); if (!cond) failed++; };
const helpers = `
  window.H = (() => { const tw = window.tw, c = tw.client, pause = (ms) => new Promise((d) => setTimeout(d, ms));
  const xy = (p) => { const q = tw.lngLatToPack(p[0], p[1]); return [q.x, q.y]; };
  const free = (p) => { const [x, y] = xy(p); return { kind: 'free', x, y, level: -1 }; };
  const near = (p) => { const [x, y] = xy(p); let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - x, n.y - y); if (d < bd) { bd = d; best = n.id; } } return best; };
  const stations = () => tw.world.value.save.stations;
  const edgeAt = (p) => { const [x, y] = xy(p); const nodes = tw.world.value.nodes; return tw.world.value.edges.find((e) => { const a = nodes.get(e.a), b = nodes.get(e.b); return a && b && Math.min(a.x, b.x) < x && Math.max(a.x, b.x) > x && Math.abs(a.y - y) < 30; }); };
  return { tw, c, pause, xy, free, near, stations, edgeAt }; })(); 0;
`;
const shot = async (page, name, center, zoom) => {
  await page.evaluate(([c, z]) => window.tw.map.jumpTo({ center: c, zoom: z }), [center, zoom]);
  await page.waitForTimeout(1800);
  await page.screenshot({ path: OUT + name });
};
try {
  const page = await chrome.ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${url}?debug=1&new=1&demandWorkers=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });

  await page.evaluate(helpers);
  // ---- T-3286: a run drawn as four pieces, joints at 500, 1000 and 1500 m; a 60 m platform at 700 m
  const P = [at(0, 0), at(500, 0), at(1000, 0), at(1500, 0), at(2000, 0)];
  const r1 = await page.evaluate(async ([P]) => {
    const { tw, c, pause, xy, free, near, stations, edgeAt } = window.H;
    const issues = [];
    let a = await c.edit({ op: 'route', from: free(P[0]), to: free(P[1]), pis: [], single: false }); issues.push(...(a.issues ?? [])); await pause(300);
    for (let k = 2; k < P.length; k++) { a = await c.edit({ op: 'route', from: { kind: 'node', node: near(P[k - 1]) }, to: free(P[k]), pis: [], single: false }); issues.push(...(a.issues ?? [])); await pause(300); }
    const pieces = tw.world.value.edges.length;
    const e = edgeAt([P[0][0] + (P[1][0] - P[0][0]) * 1.4, P[0][1]]);
    const [x, y] = xy([P[0][0] + (P[1][0] - P[0][0]) * 1.4, P[0][1]]);
    const s = await c.edit({ op: 'addStation', at: { kind: 'edge', edge: e.id, x, y }, platform: 60, name: 'Clicked' });
    await pause(400);
    const st = stations().find((q) => q.name === 'Clicked');
    return { issues, pieces, added: s.ok, addIssues: s.issues, num: st?.num, piecesAfterAdd: tw.world.value.edges.length };
  }, [P]);
  console.log(JSON.stringify(r1));
  ok(r1.added, 'a 60 m station is placed at 700 m ' + JSON.stringify(r1.addIssues));
  const centre = at(700, 0);
  await shot(page, 't3286-before-60m.png', centre, 16.6);
  const r2 = await page.evaluate(async ([num]) => {
    const { tw, c, pause, xy, free, near, stations, edgeAt } = window.H;
    const before = tw.world.value.edges.length;
    const a = await c.edit({ op: 'stationProps', node: num, platform: 400, name: 'Clicked' });
    await pause(500);
    const s = stations().find((q) => q.num === num);
    return { ok: a.ok, issues: a.issues, length: s?.length, before, after: tw.world.value.edges.length };
  }, [r1.num]);
  console.log(JSON.stringify(r2));
  ok(r2.ok && r2.length === 400, 'the platform lengthens to 400 m past the clicked node at 500 m ' + JSON.stringify(r2.issues));
  ok(r2.after < r2.before, 'the pieces under it were joined');
  await shot(page, 't3286-after-400m.png', centre, 16.6);
  const r3 = await page.evaluate(async ([num]) => {
    const { tw, c, pause, xy, free, near, stations, edgeAt } = window.H;
    const a = await c.edit({ op: 'stationProps', node: num, platform: 400, name: 'Clicked' });
    const s = stations().find((q) => q.num === num);
    return { named: !!s.named, name: s.name };
  }, [r1.num]);
  ok(!r3.named, 'a platform change flags no typed name');

  // ---- T-3288: three parallel tracks 40 m apart, a station at the west end of each
  const base = 1500;
  const row = (k) => [at(0, base + 40 * k), at(600, base + 40 * k)];
  const r4 = await page.evaluate(async ([R]) => {
    const { tw, c, pause, xy, free, near, stations, edgeAt } = window.H;
    const add = async (p, name) => { const a = await c.edit({ op: 'addStation', at: { kind: 'node', node: near(p) }, platform: 100, name }); await pause(250); return a; };
    for (const row of R) { const a = await c.edit({ op: 'route', from: free(row[0]), to: free(row[1]), pis: [], single: false }); await pause(300); if (!a.ok) return { err: 'route' }; }
    const s = [await add(R[0][0], 'Alpha'), await add(R[1][0], 'Beta'), await add(R[2][0], 'Gamma')];
    await pause(500);
    return { ok: s.map((x) => x.ok), names: stations().map((x) => [x.name, !!x.named]) };
  }, [[row(0), row(1), row(2)]]);
  console.log(JSON.stringify(r4));
  const sameName = r4.names?.filter(([n]) => n === 'Alpha').length === 3;
  ok(sameName, 'three stations of one complex all show the first one\'s name: ' + JSON.stringify(r4.names));
  await page.evaluate(() => { const s = window.H.stations().find((q) => q.name === 'Alpha'); window.tw.select({ kind: 'station', station: s.id }); });
  await shot(page, 't3288-one-name.png', at(60, base + 40), 17.4);
  const r5 = await page.evaluate(async () => {
    const { tw, c, pause, xy, free, near, stations, edgeAt } = window.H;
    const beta = stations().filter((s) => s.name === 'Alpha')[1];
    const a = await c.edit({ op: 'stationProps', node: beta.num, platform: beta.length, name: 'Hand Name' });
    await pause(500);
    return { ok: a.ok, names: stations().map((x) => [x.name, !!x.named]) };
  });
  console.log(JSON.stringify(r5));
  ok(r5.names.filter(([n]) => n === 'Hand Name').length === 1 && r5.names.filter(([n]) => n === 'Alpha').length === 2, 'renaming one station names that one only');
  await shot(page, 't3288-one-renamed.png', at(60, base + 40), 17.4);
  const r6 = await page.evaluate(async () => {
    const { tw, c, pause, xy, free, near, stations, edgeAt } = window.H;
    const head = stations().filter((s) => s.name === 'Alpha').sort((a, b) => a.num - b.num)[0];
    const a = await c.edit({ op: 'stationName', node: head.num, name: 'Central' });
    await pause(500);
    return { ok: a.ok, names: stations().map((x) => [x.name, !!x.named]) };
  });
  console.log(JSON.stringify(r6));
  ok(r6.names.filter(([n]) => n === 'Central').length === 2 && r6.names.filter(([n]) => n === 'Hand Name').length === 1, 'the complex rename takes every untyped station, not the typed one');
  ok(errors.length === 0, 'no console errors ' + errors.slice(0, 2).join(' | '));
} finally {
  await chrome.close();
}
process.exit(failed ? 1 : 0);
