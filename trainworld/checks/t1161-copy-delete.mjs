// T-1161 in a new game through the edit API: a copied line starts with no trains while its source
// keeps its own; removing the piece past a station on the track's last stretch leaves the station
// where it was, the track ending just past its platform, and between two stations both stay put;
// with real clicks, auto elevation draws on from +2 track at +2 with 0 picked, and off, at 0.
// About a minute.
// Usage: node trainworld/checks/t1161-copy-delete.mjs [url] (default http://localhost:6125/)

import { buildLine, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6125/';
const sep = BASE.includes('?') ? '&' : '?';
let failed = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) failed++; };

const { ctx, close } = await launchChrome({ prefix: 'tw-t1161-' });
try {
  const page = await ctx.newPage();
  const errors = [];
  collectErrors(page, errors);
  await page.goto(`${BASE}${sep}debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });

  // 1. a copied line has no trains
  const r = await buildLine(page, [[-73.99, 40.75], [-73.975, 40.755], [-73.96, 40.76]], { trains: [6, 4, 3] });
  check(r?.ok && r.line !== undefined, 'a line built with trains');
  const copy = await page.evaluate(async (line) => {
    const tw = window.tw;
    const a = await tw.client.edit({ op: 'copyLine', line });
    await new Promise((d) => setTimeout(d, 300));
    const w = tw.world.value, st = (id) => w.lineStats[id];
    const ln = (num) => w.save.lines.find((l) => l.num === num);
    return { ok: a.ok, copy: a.line, src: st(line)?.trainsNeeded, dst: st(a.line)?.trainsNeeded, srcStops: ln(line)?.stops, dstStops: ln(a.line)?.stops, dstTph: ln(a.line)?.tph };
  }, r.line);
  check(copy.ok && copy.copy !== undefined, 'the line copied');
  check(copy.srcStops?.length === 3 && JSON.stringify(copy.dstStops) === JSON.stringify(copy.srcStops), `the copy has the source's stops (${copy.dstStops?.length})`);
  check(copy.dst?.every((n) => n === 0), `the copy has no trains (${copy.dst})`);
  check(copy.src?.some((n) => n > 0), `the source keeps its trains (${copy.src})`);

  // 3. removing past a station: blueprint track in Brooklyn, a station at 1 km and 3 km of 4 km
  const del = await page.evaluate(async () => {
    const tw = window.tw, c = tw.client;
    const a = tw.lngLatToPack(-73.97, 40.64), b = { x: a.x + 4000, y: a.y };
    await c.edit({ op: 'route', from: { kind: 'free', x: a.x, y: a.y, level: 0 }, to: { kind: 'free', x: b.x, y: b.y, level: 0 }, pis: [], single: false });
    const w0 = tw.world.value;
    const near = (x, y) => [...w0.nodes.values()].reduce((m, n) => (Math.hypot(n.x - x, n.y - y) < Math.hypot(m.x - x, m.y - y) ? n : m));
    const n0 = near(a.x, a.y).id, n1 = near(b.x, b.y).id;
    // pack metres to ground metres along this track (the frame's scale here)
    const k = w0.edges.find((e) => e.a === n0 || e.b === n0).lengthM / 4000;
    const here = (n) => Math.abs(n.y - a.y) < 50;
    const edgeOn = (x) => tw.world.value.edges.find((e) => { const w = tw.world.value, na = w.nodes.get(e.a), nb = w.nodes.get(e.b); return here(na) && here(nb) && Math.min(na.x, nb.x) < x && Math.max(na.x, nb.x) > x; });
    for (const dx of [1000, 3000]) {
      const x = a.x + dx;
      const s = await c.edit({ op: 'addStation', at: { kind: 'edge', edge: edgeOn(x).id, x, y: a.y }, platform: 200, name: '' });
      if (!s.ok) return { error: `station at ${dx} m refused` };
    }
    const w = tw.world.value;
    const stations = [...w.nodes.values()].filter((n) => n.platform > 0 && here(n)).sort((p, q) => p.x - q.x);
    const [s1, s3] = stations.map((n) => ({ id: n.id, x: n.x, y: n.y }));
    const between = (p, q) => tw.world.value.edges.find((e) => (e.a === p && e.b === q) || (e.a === q && e.b === p));
    // loose ends of this track, ground metres from its start
    const loose = () => [...tw.world.value.nodes.values()].filter((n) => n.ports === 1 && here(n)).map((n) => (n.x - a.x) * k).sort((p, q) => p - q);
    const m = (s) => (s.x - a.x) * k;
    const last = await c.edit({ op: 'removeEdge', edge: between(s3.id, n1).id });
    const after1 = tw.world.value.nodes.get(s3.id);
    const out = { at1: m(s1), at3: m(s3), last: last.ok, moved1: Math.hypot(after1.x - s3.x, after1.y - s3.y) * k, ports1: after1.ports, loose1: loose() };
    const mid = await c.edit({ op: 'removeEdge', edge: between(s1.id, s3.id).id });
    const p1 = tw.world.value.nodes.get(s1.id), p3 = tw.world.value.nodes.get(s3.id);
    return { ...out, mid: mid.ok, moved2: Math.max(Math.hypot(p1.x - s1.x, p1.y - s1.y), Math.hypot(p3.x - s3.x, p3.y - s3.y)) * k, ports2: [p1.ports, p3.ports], loose2: loose() };
  });
  if (del.error) check(false, del.error);
  else {
    const f = (v) => v.map((x) => x.toFixed(1)).join(', ');
    // track past a platform's far end: more than the 1 m a platform needs, at most the 4 m kept
    const pastOk = (p) => p > 1 && p <= 4.1;
    check(del.last && del.moved1 < 0.01 && del.ports1 === 2, `the last stretch removed, the station unmoved (${del.moved1.toFixed(3)} m)`);
    const past = del.loose1.at(-1) - del.at3 - 100;
    check(del.loose1.length === 2 && pastOk(past), `the track ends ${past.toFixed(1)} m past the platform (loose ends at ${f(del.loose1)} m, station at ${del.at3.toFixed(1)})`);
    check(del.mid && del.moved2 < 0.01 && del.ports2.every((p) => p === 2), `the stretch between the stations removed, both unmoved (${del.moved2.toFixed(3)} m)`);
    const [, e1, e3] = del.loose2;
    check(del.loose2.length === 4 && pastOk(e1 - del.at1 - 100) && pastOk(del.at3 - 100 - e3), `each keeps track to its platform end (loose ends at ${f(del.loose2)} m)`);
  }
  // 2. auto elevation, with real clicks: from the loose end of +2 track the next segment ends at +2
  // with 0 picked; with the switch off, at 0
  const end = await page.evaluate(async () => {
    const tw = window.tw, c = tw.client;
    const a = tw.lngLatToPack(-73.97, 40.62);
    await c.edit({ op: 'route', from: { kind: 'free', x: a.x, y: a.y, level: 2 }, to: { kind: 'free', x: a.x + 1500, y: a.y, level: 2 }, pis: [], single: false });
    const n = [...tw.world.value.nodes.values()].find((m) => Math.hypot(m.x - a.x - 1500, m.y - a.y) < 1);
    return { id: n.id, level: n.level, lng: n.lng, lat: n.lat };
  });
  check(end.level === 2, `the first piece at +2 (${end.level})`);
  await page.keyboard.press('1');
  await page.waitForTimeout(400);
  const auto = page.locator('button[aria-label="Elevation of the track you extend"]');
  await page.locator('.build-options .levels button', { hasText: /^0$/ }).click();
  if ((await auto.getAttribute('aria-pressed')) !== 'true') await auto.click();
  const drawFrom = async (lng, lat, dLng) => {
    const pt = async (x, y) => page.evaluate(([x, y]) => { const r = window.tw.map.getCanvas().getBoundingClientRect(), p = window.tw.map.project([x, y]); return [r.left + p.x, r.top + p.y]; }, [x, y]);
    // the track tool in hand (Esc after a run can leave it)
    if (await page.evaluate(() => !document.querySelector('.tool-pick button.on')?.textContent?.includes('1'))) await page.keyboard.press('1');
    await page.evaluate(([lng, lat]) => void window.tw.map.jumpTo({ center: [lng, lat], zoom: 14.5 }), [lng, lat]);
    await page.waitForTimeout(300);
    const v0 = await page.evaluate(() => window.tw.world.value.version);
    const [x0, y0] = await pt(lng, lat), [x1, y1] = await pt(lng + dLng, lat);
    await page.mouse.click(x0, y0);
    await page.waitForTimeout(500);
    await page.mouse.move(x1, y1, { steps: 4 });
    await page.waitForTimeout(800);
    await page.mouse.click(x1, y1);
    await page.waitForFunction((v) => window.tw.world.value.version > v, v0, { timeout: 15000 }).catch(() => {});
    const note = await page.evaluate(() => document.querySelector('#tool-notes')?.textContent ?? '');
    await page.keyboard.press('Escape');
    await page.waitForTimeout(300);
    return page.evaluate(([lng, lat, note]) => {
      const w = window.tw.world.value;
      const n = [...w.nodes.values()].filter((m) => m.ports === 1).sort((p, q) => Math.hypot(p.lng - lng, p.lat - lat) - Math.hypot(q.lng - lng, q.lat - lat))[0];
      return n ? { level: n.level, lng: n.lng, lat: n.lat, edges: w.edges.length, note } : null;
    }, [lng + dLng, lat, note]);
  };
  const autoEnd = await drawFrom(end.lng, end.lat, 0.006);
  check(autoEnd?.level === 2, `with auto on, a segment from +2 track ends at +2 with 0 picked (${autoEnd?.level}; ${autoEnd?.note})`);
  await auto.click();
  const plainEnd = autoEnd && await drawFrom(autoEnd.lng, autoEnd.lat, 0.012);
  check(plainEnd?.level === 0, `with auto off, at the elevation picked (${plainEnd?.level}; ${plainEnd?.note})`);

  check(errors.length === 0, `no page errors${errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''}`);
} finally {
  await close();
}
process.exit(failed ? 1 : 0);
