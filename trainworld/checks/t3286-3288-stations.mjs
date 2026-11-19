// T-3286: a platform lengthens through a node that is only a clicked corner of a drawn run.
// T-3288, T-3750, T-3759: the stations of a complex bound to its name show the complex's name (the oldest typed
// name, else the oldest station's), and so does a station building nobody named, at every zoom and after a load;
// an unbound station keeps its own name, beside the building's too.
// Screenshots go to artifacts/trainworld/t3286/.
// From the worktree root:   node trainworld/checks/t3286-3288-stations.mjs [url]
import { launchChrome, collectErrors, importSave, ARTIFACTS } from './lib.mjs';
import { mkdirSync } from 'node:fs';
import { readFile } from 'node:fs/promises';

const url = process.argv[2] ?? 'http://localhost:6155/';
const OUT = ARTIFACTS + 't3286/';
mkdirSync(OUT, { recursive: true });
const LAT = 40.652, LNG0 = -73.985;
const dLat = (m) => m / 111320, dLng = (m) => m / (111320 * Math.cos((LAT * Math.PI) / 180));
const at = (xm, ym) => [LNG0 + dLng(xm), LAT + dLat(ym)];

const chrome = await launchChrome({ prefix: 'tw-t3286-', size: [1200, 800], acceptDownloads: true });
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
  // T-3750: a typed name is the complex's: the stations nobody typed a name for follow it
  const r5 = await page.evaluate(async () => {
    const { tw, c, pause, xy, free, near, stations, edgeAt } = window.H;
    const beta = stations().filter((s) => s.name === 'Alpha')[1];
    const a = await c.edit({ op: 'stationProps', node: beta.num, platform: beta.length, name: 'Hand Name' });
    await pause(500);
    return { ok: a.ok, names: stations().map((x) => [x.name, !!x.named]) };
  });
  console.log(JSON.stringify(r5));
  ok(r5.names.filter(([n]) => n === 'Hand Name').length === 3 && r5.names.filter(([, t]) => t).length === 1, 'renaming one station of a complex names the complex (T-3750)');
  await shot(page, 't3288-one-renamed.png', at(60, base + 40), 17.4);
  // T-3759: a station unbound from its complex's name keeps a name of its own; the complex's rename leaves it
  const r6 = await page.evaluate(async () => {
    const { tw, c, pause, xy, free, near, stations, edgeAt } = window.H;
    const names = () => stations().filter((s) => s.name !== 'Clicked').sort((a, b) => a.num - b.num).map((x) => x.name);
    const gamma = stations().filter((s) => !s.named).sort((a, b) => b.num - a.num)[0];
    const bound = await c.edit({ op: 'stationProps', node: gamma.num, platform: gamma.length, name: 'Other' }); await pause(500);
    const afterBound = names();
    await c.edit({ op: 'undo' }); await pause(400);
    const u = await c.edit({ op: 'bindStation', node: gamma.num, bound: false }); await pause(500);
    const unbound = names();
    const o = await c.edit({ op: 'renameStation', node: gamma.num, name: 'Other' }); await pause(500);
    const own = names();
    const cx = await c.edit({ op: 'renameComplex', node: gamma.num, name: 'Hall' }); await pause(500);
    const hall = names(), hub = stations().find((s) => s.num === gamma.num)?.hub;
    return { ok: [bound.ok, u.ok, o.ok, cx.ok], afterBound, unbound, own, hall, hub };
  });
  console.log(JSON.stringify(r6));
  ok(r6.afterBound.every((n) => n === 'Other'), 'renaming a bound station names its complex ' + JSON.stringify(r6.afterBound));
  ok(r6.ok.every(Boolean) && r6.unbound.join() === 'Hand Name,Hand Name,Gamma', 'unbinding a station shows its own stored name ' + JSON.stringify(r6.unbound));
  ok(r6.own.join() === 'Hand Name,Hand Name,Other', 'an unbound station is renamed alone ' + JSON.stringify(r6.own));
  ok(r6.hall.join() === 'Hall,Hall,Other' && r6.hub === 'Hall', 'the complex\'s rename leaves the unbound station its name ' + JSON.stringify(r6.hall));
  const r6u = await page.evaluate(async () => {
    const { c, pause, stations } = window.H;
    for (let k = 0; k < 3; k++) { await c.edit({ op: 'undo' }); await pause(400); }
    return stations().filter((s) => s.name !== 'Clicked').map((x) => [x.name, !!x.unbound]);
  });
  ok(r6u.length === 3 && r6u.every(([n, u]) => n === 'Hand Name' && !u), 'undo binds it again with the complex\'s name ' + JSON.stringify(r6u));

  // ---- T-3750: a station building nobody named shows the complex's name, whatever renames it, at every
  // zoom (its name from 13 in, the stations' below) and after a save and load
  await page.evaluate(() => { const d = window.tw.display; if (!d.value.stationNames) d.value = { ...d.value, stationNames: true }; });
  const corners = [at(-60, base - 30), at(160, base - 30), at(160, base + 110), at(-60, base + 110)];
  const r7 = await page.evaluate(async ([P]) => {
    const { tw, c, pause, xy } = window.H;
    await c.edit({ op: 'constructAll' }); await pause(500);
    const a = await c.edit({ op: 'addBuilding', corners: P.flatMap(xy), name: '' }); await pause(500);
    const beta = window.H.stations().find((s) => s.named);
    const b = await c.edit({ op: 'stationProps', node: beta.num, platform: beta.length, name: 'Union' }); await pause(500);
    return { ok: a.ok && b.ok, issues: a.issues, buildings: tw.world.value.save.buildings.map((x) => x.name), names: window.H.stations().filter((x) => x.name !== 'Clicked').map((x) => x.name) };
  }, [corners]);
  console.log(JSON.stringify(r7));
  const labels = async (z) => {
    await page.evaluate(([cc, zz]) => window.tw.map.jumpTo({ center: cc, zoom: zz }), [at(50, base + 40), z]);
    await page.waitForTimeout(1500);
    return page.evaluate(() => [...new Set([...document.querySelectorAll('.stn-label')].filter((e) => e.style.visibility !== 'hidden' && e.closest('.stn-labels')?.style.display !== 'none').map((e) => e.textContent))].filter((n) => n !== 'Clicked'));
  };
  const shown = async () => ({ z17: await labels(17), z14: await labels(14), z12: await labels(12.5) });
  const l7 = await shown();
  console.log(JSON.stringify(l7));
  ok(r7.ok && r7.buildings.join() === 'Union' && r7.names.every((n) => n === 'Union'), 'the building and every station show the renamed complex');
  ok([l7.z17, l7.z14, l7.z12].every((l) => l.join() === 'Union'), 'one name at street, middle and far zooms: ' + JSON.stringify(l7));
  await shot(page, 't3750-building-renamed.png', at(50, base + 40), 16);
  const [dl] = await Promise.all([page.waitForEvent('download'), page.evaluate(() => window.tw.client.post({ kind: 'export' }))]);
  const bytes = await readFile(await dl.path());
  await importSave(page, bytes);
  await page.waitForFunction(() => window.tw.world.value?.save.buildings?.length === 1 && window.tw.world.value.save.stations.length === 4, undefined, { timeout: 60000 });
  await page.waitForTimeout(1500);
  const r8 = await page.evaluate(() => ({ buildings: window.tw.world.value.save.buildings.map((x) => x.name), names: window.H.stations().filter((x) => x.name !== 'Clicked').map((x) => x.name) }));
  const l8 = await shown();
  console.log(JSON.stringify(r8), JSON.stringify(l8));
  ok(r8.buildings.join() === 'Union' && r8.names.every((n) => n === 'Union') && [l8.z17, l8.z14, l8.z12].every((l) => l.join() === 'Union'), 'the same after a save and load');
  // T-3759: a station unbound from the complex's name keeps its own label beside the building's, at every zoom
  const r9 = await page.evaluate(async () => {
    const { c, pause, stations } = window.H;
    const g = stations().filter((s) => s.name !== 'Clicked').sort((a, b) => b.num - a.num)[0];
    const a = await c.edit({ op: 'bindStation', node: g.num, bound: false }); await pause(600);
    return { ok: a.ok, buildings: window.tw.world.value.save.buildings.map((x) => x.name), names: stations().filter((x) => x.name !== 'Clicked').sort((a, b) => a.num - b.num).map((x) => x.name) };
  });
  const l9 = await shown();
  console.log(JSON.stringify(r9), JSON.stringify(l9));
  ok(r9.ok && r9.buildings.join() === 'Union' && r9.names.join() === 'Union,Union,Gamma', 'an unbound station shows its own name, the building the complex\'s');
  // (40 m apart, the two names collide at zoom 14: one is drawn); far out the complex goes by its own name
  ok(l9.z17.slice().sort().join() === 'Gamma,Union' && l9.z14.length > 0 && l9.z14.every((n) => n === 'Union' || n === 'Gamma'), 'both names close in: ' + JSON.stringify(l9));
  ok(l9.z12.join() === 'Union', 'far out, the complex\'s name only: ' + JSON.stringify(l9.z12));
  // with no building, and every station unbound, far out still shows the complex's name
  const r10 = await page.evaluate(async () => {
    const { c, pause, stations } = window.H;
    await c.edit({ op: 'removeBuilding', id: window.tw.world.value.save.buildings[0].id }); await pause(400);
    for (const s of stations().filter((x) => x.name !== 'Clicked' && !x.unbound)) { await c.edit({ op: 'bindStation', node: s.num, bound: false }); await pause(300); }
    const g = stations().filter((s) => s.name !== 'Clicked').sort((a, b) => a.num - b.num)[0];
    await c.edit({ op: 'renameStation', node: g.num, name: 'First' }); await pause(400);
    await c.edit({ op: 'renameComplex', node: g.num, name: 'Whole' }); await pause(500);
    return { buildings: (window.tw.world.value.save.buildings ?? []).length, names: stations().filter((x) => x.name !== 'Clicked').sort((a, b) => a.num - b.num).map((x) => [x.name, !!x.unbound, x.hub ?? null]) };
  });
  const l10 = await shown();
  console.log(JSON.stringify(r10), JSON.stringify(l10));
  ok(r10.buildings === 0 && r10.names.every(([, u, h]) => u && h === 'Whole') && l10.z17.includes('First') && !l10.z17.includes('Whole') && l10.z12.join() === 'Whole', 'all unbound: close in their own names, far out the complex\'s: ' + JSON.stringify(l10));
  await shot(page, 't3759-unbound-building.png', at(50, base + 40), 17);
  ok(errors.length === 0, 'no console errors ' + errors.slice(0, 2).join(' | '));
} finally {
  await chrome.close();
}
process.exit(failed ? 1 : 0);
