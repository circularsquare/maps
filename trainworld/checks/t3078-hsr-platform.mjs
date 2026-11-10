// T-3078: a long station (10 cars, 400 m) goes on high-speed track. Tram release v0.2 let HSR edges carry the
// raw TrackKind 1 into the app's `edge.kind`, which every "is it tram" test read as truthy. In a sandbox game:
// high-speed, regular and tram track are built; no rail edge carries a kind; the station tool's length at a
// spot is the rail one on high-speed and regular track and the tram one on tram track; a 400 m station is
// placed on high-speed and regular track; on tram track a long stop is refused (the sim keeps 20-80 m) and a
// 40 m one placed. From the maps root:
//   node trainworld/checks/t3078-hsr-platform.mjs [url]
import { launchChrome, collectErrors } from './lib.mjs';

const url = process.argv[2] ?? 'http://localhost:6196/';
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };
const LAT = 40.3, LNG0 = -79.6, KM = 1 / (111.32 * Math.cos((LAT * Math.PI) / 180)), KMLAT = 1 / 110.57;
const row = (k) => [[LNG0, LAT + k * KMLAT], [LNG0 + 3 * KM, LAT + k * KMLAT]];

const chrome = await launchChrome({ prefix: 'tw-t3078-', size: [1200, 800] });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${url}?debug=1&new=1&paused=1&demandWorkers=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  await page.waitForTimeout(1000);
  const r = await page.evaluate(async ([rows]) => {
    const tw = window.tw, c = tw.client, ui = await import('/src/game/ui.ts');
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const xy = (p) => { const q = tw.lngLatToPack(p[0], p[1]); return [q.x, q.y]; };
    const out = {};
    const build = async (name, [a, b], extra) => {
      const [ax, ay] = xy(a), [bx, by] = xy(b);
      const before = new Set(tw.world.value.edges.map((e) => e.id));
      const res = await c.edit({ op: 'route', from: { kind: 'free', x: ax, y: ay, level: 0 }, to: { kind: 'free', x: bx, y: by, level: 0 }, pis: [], single: false, ...extra });
      await pause(400);
      await c.edit({ op: 'constructAll' });
      await pause(600);
      const edges = tw.world.value.edges.filter((e) => !before.has(e.id));
      out[name] = { ok: res?.ok, edges: edges.map((e) => ({ id: e.id, kind: e.kind ?? null, hsr: !!e.hsr })) };
    };
    await build('hsr', rows[0], { hsr: true });
    await build('rail', rows[1], {});
    await build('tram', rows[2], { tram: true });
    const t = tw.tools;
    const mid = (name) => out[name].edges[Math.floor(out[name].edges.length / 2)].id;
    ui.buildPlatform.value = 400;
    ui.buildTramPlatform.value = 40;
    await pause(100);
    for (const name of ['hsr', 'rail', 'tram']) out[name].len = t.platFor({ kind: 'edge', edge: mid(name), lx: 0, ly: 0 });
    // the sim's own answer for a station of a given length on each kind
    const place = async (name, platform) => {
      const e = tw.world.value.edges.find((x) => x.id === mid(name));
      const n = tw.world.value.nodes.get(e.a), m = tw.world.value.nodes.get(e.b);
      const res = await c.edit({ op: 'addStation', at: { kind: 'edge', edge: e.id, x: (n.x + m.x) / 2, y: (n.y + m.y) / 2 }, platform, name: `${name}${platform}` });
      await pause(300);
      return { ok: !!res?.ok, issue: res?.issue ?? res?.error ?? null };
    };
    out.p = { hsr400: await place('hsr', 400), rail400: await place('rail', 400), tram400: await place('tram', 400), tram40: await place('tram', 40) };
    return out;
  }, [[row(0), row(1), row(2)]]);
  console.log(JSON.stringify(r));
  check(r.hsr.ok && r.rail.ok && r.tram.ok, 'three kinds of track were built');
  check(r.hsr.edges.length > 0 && r.hsr.edges.every((e) => e.hsr && e.kind === null), 'high-speed edges carry no tram kind');
  check(r.rail.edges.every((e) => !e.hsr && e.kind === null), 'regular edges carry no tram kind');
  check(r.tram.edges.length > 0 && r.tram.edges.every((e) => e.kind >= 2), 'tram edges carry kind 2 or 3');
  check(r.hsr.len === 400 && r.rail.len === 400, 'the tool length (400 m) is used on high-speed and regular track');
  check(r.tram.len === 40, 'the tram length is used on tram track');
  check(r.p.hsr400.ok, `a 400 m station on high-speed track is accepted ${JSON.stringify(r.p.hsr400)}`);
  check(r.p.rail400.ok, `a 400 m station on regular track is accepted ${JSON.stringify(r.p.rail400)}`);
  check(!r.p.tram400.ok, `a 400 m stop on tram track is refused ${JSON.stringify(r.p.tram400)}`);
  check(r.p.tram40.ok, `a 40 m stop on tram track is accepted ${JSON.stringify(r.p.tram40)}`);
  if (errors.length) console.log('errors: ' + errors.slice(0, 3).join(' | '));
} finally {
  await chrome.close();
}
process.exit(fail ? 1 : 0);
