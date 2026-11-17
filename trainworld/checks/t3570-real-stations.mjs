// T-3570, T-3571: real stations on the real world network map, and new stations named after them. In new sandbox
// games in New York, Tokyo and Seoul, on the playtest harness (T-3556):
//   - Settings > Real world network > Show stations is off at first; pressed, the region's stations are drawn, and
//     hovering the mark on a real station shows its name (Latin letters in Tokyo and Seoul);
//   - with the marks off again, a station placed with the station tool on the real station's point is named after
//     it, and one placed 1 km or more away, where no real station is within 250 m, is not.
// Screenshots, off and on at zoom 12 and 15, New York and Tokyo: artifacts/trainworld/t3570/. From the maps root:
//   node trainworld/checks/t3570-real-stations.mjs [url]
// The real stations' positions and names come from noritetsu's stations.json, through the dev server's /noritetsu proxy.
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { ARTIFACTS } from './lib.mjs';
import { launchGames } from './lib/play.mjs';

const url = process.argv[2] ?? 'http://localhost:6149/';
const OUT = join(ARTIFACTS, 't3570');
mkdirSync(OUT, { recursive: true });
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };

const metres = (a, b) => Math.hypot((a[0] - b[0]) * 111320 * Math.cos((a[1] * Math.PI) / 180), (a[1] - b[1]) * 110540);
const along = (p, m, deg) => {
  const r = (deg * Math.PI) / 180;
  return [p[0] + (m * Math.sin(r)) / (111320 * Math.cos((p[1] * Math.PI) / 180)), p[1] + (m * Math.cos(r)) / 110540];
};

const CASES = [
  { city: 'nyc', cc: 'us', local: 'Times Square–42nd Street', want: 'Times Square–42nd Street', shots: true },
  { city: 'tyo', cc: 'jp', local: '新宿', want: 'Shinjuku', shots: true },
  { city: 'seo', cc: 'kr', local: '서울', want: 'Seoul Station' },
];

const games = await launchGames({ size: [1400, 900], prefix: 'tw-t3570-' });
try {
  for (const c of CASES) {
    const raw = (await (await fetch(new URL(`/noritetsu/data/${c.cc}/stations.json`, url))).json()).stations;
    const real = Object.values(raw).filter((s) => !s.j).map((s) => ({ n: s.n, e: s.e, p: [s.x, s.y] }));
    const target = real.filter((s) => s.n === c.local).sort((a, b) => b.n.length - a.n.length)[0];
    if (!target) { check(false, `${c.city}: ${c.local} is in stations.json`); continue; }
    const P = target.p;
    const nearestTo = (q) => real.reduce((b, s) => { const d = metres(q, s.p); return d < b.d ? { s, d } : b; }, { s: null, d: Infinity });
    // a spot 1-1.5 km away with no real station within 250 m
    let far = null;
    for (let m = 1000; !far && m <= 1500; m += 100)
      for (let deg = 0; deg < 360 && !far; deg += 15) { const q = along(P, m, deg); if (nearestTo(q).d > 250) far = { q, deg, m }; }
    if (!far) { check(false, `${c.city}: a spot 1-1.5 km away with no real station within 250 m`); continue; }

    const g = await games.open({ url, city: c.city, mode: 'sandbox' });
    try {
      await g.act.camera(P, 15);
      // marks off at first
      const off = await g.page.evaluate(() => window.twRealStations?.());
      check(off && !off.on, `${c.city}: the marks are off by default`);
      if (c.shots) {
        await g.act.camera(P, 12); await g.driver.sleep(1500);
        await g.shot(join(OUT, `${c.city}_z12_off.png`));
        await g.act.camera(P, 15); await g.driver.sleep(1500);
        await g.shot(join(OUT, `${c.city}_z15_off.png`));
      }
      await g.act.tab('settings');
      await g.driver.control({ name: 'Show stations', role: 'checkbox' });
      await g.page.waitForFunction((cc) => window.twRealStations?.().drawn.some((r) => r.cc === cc), c.cc, { timeout: 60000 });
      await g.page.waitForFunction(() => window.tw.map.loaded() && window.tw.map.areTilesLoaded(), undefined, { timeout: 30000 });
      await g.driver.sleep(1500);
      if (c.shots) {
        await g.shot(join(OUT, `${c.city}_z15_on.png`));
        await g.act.camera(P, 12); await g.driver.sleep(2000);
        await g.shot(join(OUT, `${c.city}_z12_on.png`));
        await g.act.camera(P, 15); await g.driver.sleep(1500);
      }
      // the hover on the real station's mark
      const [s] = await g.toScreen([P]);
      await g.driver.move(s.x + 30, s.y + 30);
      await g.driver.move(s.x, s.y, { steps: 3 });
      await g.driver.sleep(500);
      const tag = await g.page.evaluate(() => [...document.querySelectorAll('.demand-tag')].find((t) => t.style.display !== 'none' && t.textContent)?.textContent ?? null);
      if (c.shots) await g.shot(join(OUT, `${c.city}_hover.png`), { clip: { x: s.x - 200, y: s.y - 120, width: 400, height: 240 } });
      check(tag === c.want, `${c.city}: hovering the mark on ${c.local} shows "${c.want}" (got ${JSON.stringify(tag)})`);
      // marks off again: naming does not need them
      await g.driver.control({ name: 'Show stations', role: 'checkbox' });
      const after = await g.page.evaluate(() => window.twRealStations?.());
      check(after && !after.on, `${c.city}: the marks go off again`);

      // track through the real station's point, and through the far spot
      await g.act.tool('track', { type: 'regular', tracks: 2, snap: { real: false } });
      await g.act.drawTrack([along(P, 300, 270), P, along(P, 300, 90)]);
      await g.act.camera(far.q, 15);
      await g.act.drawTrack([along(far.q, 300, 270), far.q, along(far.q, 300, 90)]);
      await g.act.camera(P, 15);
      await g.act.tool('station');
      // the click must not wait on the download, so the check waits for it: the tool asks for the view's region
      await g.page.waitForFunction((cc) => window.twRealStations?.().loaded.includes(cc), c.cc, { timeout: 60000 });
      const on = await g.act.placeStation(P);
      check(on.ok && on.name === c.want, `${c.city}: a station on ${c.local} is named "${c.want}" (got ${JSON.stringify(on.ok ? on.name : on.notes)})`);
      await g.act.camera(far.q, 15);
      const off2 = await g.act.placeStation(far.q);
      const n = nearestTo(far.q);
      check(off2.ok && off2.name !== c.want && off2.name !== n.s.n && off2.name !== n.s.e,
        `${c.city}: a station ${far.m} m away (nearest real station ${n.s.e || n.s.n} at ${Math.round(n.d)} m) is not named after a real one (got ${JSON.stringify(off2.ok ? off2.name : off2.notes)})`);
      const errs = g.pageErrors();
      check(!errs.length, `${c.city}: no page errors${errs.length ? ': ' + errs.slice(0, 3).join(' | ') : ''}`);
    } finally {
      await g.close();
    }
  }
} finally {
  await games.close();
}
console.log(fail ? `${fail} failed` : 'all passed');
process.exit(fail ? 1 : 0);
