// T-2122 to-the-bit A/B: a new game in a zone, its first line built, the final; then the corridor
// table (key and a hash of every column), the final view hashed (hash.mjs's way), and the longer
// trips of bubble selections (both ends, three cell sets) hashed. env: BASE, ZONE, TAG, WORKERS (1)
import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { launchChrome, demandFinal } from 'file:///C:/Users/anita/projects/maps/trainworld/checks/lib.mjs';

const BASE = process.env.BASE ?? 'http://localhost:6160';
const ZONE = process.env.ZONE ?? 'jak';
const TAG = process.env.TAG ?? 'ab';
const SCR = process.env.SCR ?? 'C:/Users/anita/AppData/Local/Temp/t2122';
const Z = JSON.parse(await readFile(new URL('./lines.json', import.meta.url), 'utf8'))[ZONE];
const { ctx, close } = await launchChrome({ prefix: join(SCR, `prof-ab-${TAG}-`), args: ['--enable-unsafe-swiftshader'] });
const page = await ctx.newPage();
page.on('pageerror', (e) => console.log('pageerror', e.message.slice(0, 200)));
await page.goto(`${BASE}/?debug=1&paused=1&mode=sandbox&opened=all&new=1&demandWorkers=${process.env.WORKERS ?? 1}&v=${Date.now()}`);
await page.waitForFunction(() => window.tw?.world.value && window.twDemand?.client && window.twDemand.view()?.round >= 0, undefined, { timeout: 240000 });
await page.evaluate((ll) => { window.tw.map.jumpTo({ center: ll, zoom: 11.2 }); }, Z.centre);
await page.waitForFunction((id) => window.twDemand.client.start.loads.find((l) => l.item === id)?.openedAt, ZONE, { timeout: 300000 });
await new Promise((r) => setTimeout(r, 6000));
const v = await page.evaluate(() => window.twDemand.view()?.version ?? 0);
await page.evaluate(async (L) => {
  const t = window.tw, run = async (op) => { const r = await t.client.edit(op); if (!r.ok) throw Error(JSON.stringify({ op, r })); return r; };
  const pts = L.stops.map((s) => t.lngLatToPack(s[1], s[2]));
  const tail = (a, b) => { const dx = a.x - b.x, dy = a.y - b.y, l = Math.hypot(dx, dy); return { x: a.x + 300 * dx / l, y: a.y + 300 * dy / l }; };
  const path = [tail(pts[0], pts[1]), ...pts, tail(pts.at(-1), pts.at(-2))];
  const nodes = [];
  for (let i = 1; i < path.length; i++) {
    const from = i === 1 ? { kind: 'free', x: path[0].x, y: path[0].y, level: L.level } : { kind: 'node', node: nodes.at(-1) };
    await run({ op: 'route', from, to: { kind: 'free', x: path[i].x, y: path[i].y, level: L.level }, pis: [], single: false });
    const e = t.world.value.edges.at(-1);
    if (i === 1) nodes.push(e.a);
    nodes.push(e.b);
  }
  const at = nodes.slice(1, -1);
  for (let i = 0; i < L.stops.length; i++) await run({ op: 'addStation', at: { kind: 'node', node: at[i] }, platform: 160, name: L.stops[i][0] });
  const line = (await run({ op: 'addLine', stops: at, name: 'ab', colour: '#2a7ab0' })).line;
  await run({ op: 'setTrainCount', line, trains: [10, 10, 5] });
  await run({ op: 'constructAll' });
}, Z.lines[0]);
await demandFinal(page, { after: v, timeout: 600000 });
await new Promise((r) => setTimeout(r, 12000));
const out = await page.evaluate(async () => {
  const h8 = async (x) => { const b = new TextEncoder().encode(typeof x === 'string' ? x : JSON.stringify(x)); const h = new Uint8Array(await crypto.subtle.digest('SHA-256', b)); return Array.from(h.slice(0, 8), (v) => v.toString(16).padStart(2, '0')).join(''); };
  const ser = (x) => {
    if (x === null || x === undefined) return null;
    if (x instanceof Map) return [...x.entries()].sort((a, b) => (String(a[0]) < String(b[0]) ? -1 : 1)).map(([k, v]) => [k, ser(v)]);
    if (ArrayBuffer.isView(x)) return Array.from(x);
    if (Array.isArray(x)) return x.map(ser);
    if (typeof x === 'object') return Object.keys(x).sort().filter((k) => k !== 'version').map((k) => [k, ser(x[k])]);
    return x;
  };
  const c = window.twDemand.client;
  const tables = {};
  for (const [k, t] of c.corridorTables) {
    const cols = {};
    for (const f of ['from', 'to', 'longer', 'commutes', 'air', 'car']) cols[f] = await h8(new Uint8Array(t[f].buffer, t[f].byteOffset, t[f].byteLength).join(','));
    tables[k] = { n: t.n, cols };
  }
  const dv = window.twDemand.view();
  const sel = {};
  const n = window.twDemand.cities()?.get?.('jak')?.cells ?? 40000;
  for (const [name, step, off] of [['a', 13, 0], ['b', 97, 5], ['c', 401, 2]]) {
    const cells = Uint32Array.from({ length: 300 }, (_, i) => off + i * step);
    for (const end of ['home', 'work']) {
      const r = await c.intercitySelection(end, cells);
      sel[`${name}.${end}`] = r ? await h8(JSON.stringify(ser(r))) : null;
      sel[`${name}.${end}.air`] = r ? [Math.round(r.starts?.air ?? r.from?.air ?? 0), Object.keys(r).join(',')] : null;
    }
  }
  return { tables, view: await h8(JSON.stringify(ser(dv))), cities: await h8(JSON.stringify(ser(window.twDemand.cities()))), rail: Math.round(dv.railTripsPerDay), sel };
});
console.log(JSON.stringify(out, null, 1));
await writeFile(`C:/Users/anita/projects/maps/artifacts/trainworld/t2122/ab-${TAG}.json`, JSON.stringify(out, null, 1));
await Promise.race([close(), new Promise((r) => setTimeout(r, 30000))]);
process.exit(0);
