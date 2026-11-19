// T-3495: the track tool snaps to each kind of drawn rail. For each place, the Real world network at All
// with snap on: finds a drawn feature of each class (used rail, unused rail "n", metro, light rail), clicks
// a point on it, rests the pointer on a point further along the same feature and reports whether a route is
// found (spot.phase done, a real line of length > 0). GRAPHS=<dir> serves that folder's .rgr files in place of
// the data pack's (to test a rebuild before it is shipped). T-3069: MODE=tram draws tram track with its own
// real snap (classes tram, light rail and rail), and every run says how many tram features are drawn (none
// while train track snaps). Each class prints the route lengths, to compare graphs. From the maps root:
// SETTING=passenger runs under the Passenger setting instead of All; ACROSS=1 ends each route on another feature of
// the class 150 to 600 px away, so routes pass junctions; PLACE=1 places the first route found and prints the kinds
// of the edges it added. Each class also prints how many previews the track model accepted.
//   [GRAPHS=dir] [MODE=tram] [SETTING=passenger] [ACROSS=1] [PLACE=1] [ONLY=london,chicago] node trainworld/checks/t3495-snap-route.mjs [url]
import { readFile, readdir } from 'node:fs/promises';
import { launchChrome, collectErrors } from './lib.mjs';

const url = process.argv[2] ?? 'http://localhost:6189/';
const PLACES = { london: [-0.12, 51.5], chicago: [-87.65, 41.88], tokyo: [139.75, 35.68], berlin: [13.4, 52.5], sydney: [151.2, -33.87],
  melbourne: [144.96, -37.81], amsterdam: [4.9, 52.37], zurich: [8.54, 47.38], kaohsiung: [120.3, 22.62],
  basel: [7.59, 47.56], ruhr: [6.86, 51.5], manchester: [-2.35, 53.47], tokyomono: [139.749, 35.596], zurichfun: [8.545, 47.3765] };
const TRAM = process.env.MODE === 'tram';
const CLASSES = TRAM ? ['tram', 'light_rail', 'rail', 'monorail', 'freight', 'abandoned'] : ['rail', 'rail-unused', 'subway', 'light_rail', 'narrow_gauge', 'monorail', 'funicular', 'heritage', 'tram', 'freight', 'abandoned'];
const SETTING = process.env.SETTING ?? 'all', ACROSS = !!process.env.ACROSS;
let placed = !process.env.PLACE;
const only = process.env.ONLY?.split(',');
const chrome = await launchChrome({ prefix: 'tw-t3495r-', size: [1400, 900] });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  collectErrors(page, errors);
  if (process.env.GRAPHS) {
    for (const f of await readdir(process.env.GRAPHS)) if (f.endsWith('.rgr') || f === 'index.json') {
      const body = await readFile(process.env.GRAPHS + '/' + f);
      await page.route('**/realgraph/' + f, (r) => r.fulfill({ status: 200, body, contentType: f.endsWith('.json') ? 'application/json' : 'application/octet-stream' }));
    }
  }
  await page.goto(`${url}?debug=1&new=1&demandWorkers=1&z=${process.env.ZOOM ?? 13}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });
  for (const [name, c] of Object.entries(PLACES)) {
    if (only && !only.includes(name)) continue;
    await page.evaluate(async ([c, tram, setting]) => {
      const ui = await import('/src/game/ui.ts');
      ui.display.value = { ...ui.display.value, realNetwork: setting };
      ui.keepPlacing.value = true; ui.tool.value = 'track';
      ui.setTrackType(tram ? 'tram' : 2);
      if (tram) { ui.snapRealTram.value = true; ui.snapStreets.value = false; } else ui.snapReal.value = true;
      window.tw.map.jumpTo({ center: c, zoom: Number(new URLSearchParams(location.search).get('z') ?? 13) });
    }, [c, TRAM, SETTING]);
    await page.waitForTimeout(10000);
    // candidate (start, end) screen points per class: a feature's points with 60 to 300 px between them
    const cands = await page.evaluate((across) => {
      const map = window.tw.map, r = map.getContainer().getBoundingClientRect();
      const layers = map.getStyle().layers.filter((l) => l.id.startsWith('tw-real')).map((l) => l.id);
      const out = {}, feats = {};
      for (const f of map.queryRenderedFeatures(undefined, { layers })) {
        const p = f.properties, cls = (p.k === 'rail' && p.n && !p.x ? 'rail-unused' : p.k) + (p.r === 1 ? '' : '');
        if (f.geometry.type !== 'LineString') continue;
        const pts = f.geometry.coordinates.map((q) => map.project(q)).filter((q) => q.x > 40 && q.y > 100 && q.x < r.width - 40 && q.y < r.height - 40);
        if (pts.length < 3) continue;
        const a = pts[0], b = pts[pts.length - 1], d = Math.hypot(a.x - b.x, a.y - b.y);
        if (across) {
          if (window.tw.tools.real.ring(a.x, a.y)) (feats[cls] ??= []).push([r.left + a.x, r.top + a.y]);
          continue;
        }
        if (d < 60 || d > 300 || !window.tw.tools.real.ring(a.x, a.y) || !window.tw.tools.real.ring(b.x, b.y)) continue;
        (out[cls] ??= []).push([[r.left + a.x, r.top + a.y], [r.left + b.x, r.top + b.y]]);
      }
      // across features: each start paired with the first other feature's point 150 to 600 px away
      for (const [cls, list] of Object.entries(feats))
        for (let i = 0; i < list.length; i++) {
          const b = list.find((q, j) => j !== i && Math.hypot(q[0] - list[i][0], q[1] - list[i][1]) >= 150 && Math.hypot(q[0] - list[i][0], q[1] - list[i][1]) <= 600);
          if (b) (out[cls] ??= []).push([list[i], b]);
        }
      return out;
    }, ACROSS);
    const drawn = await page.evaluate(() => {
      const map = window.tw.map, layers = map.getStyle().layers.filter((l) => l.id.startsWith('tw-real')).map((l) => l.id);
      const n = {};
      for (const f of map.queryRenderedFeatures(undefined, { layers })) n[f.properties.k] = (n[f.properties.k] ?? 0) + 1;
      return n;
    });
    console.log(name.padEnd(9), 'drawn while snapping:', JSON.stringify(drawn));
    for (const cls of CLASSES) {
      const list = cands[cls] ?? [];
      let ok = 0, tried = 0, last = null, accepted = 0;
      const lens = [];
      for (const [a, b] of list.slice(0, 6)) {
        tried++;
        await page.mouse.move(...a); await page.waitForTimeout(250);
        await page.mouse.down(); await page.mouse.up(); await page.waitForTimeout(400);
        const began = await page.evaluate(() => !!window.tw.tools.real.from);
        for (let i = 1; i <= 8; i++) { await page.mouse.move(a[0] + ((b[0] - a[0]) * i) / 8, a[1] + ((b[1] - a[1]) * i) / 8); await page.waitForTimeout(30); }
        await page.waitForTimeout(1500);
        const s = await page.evaluate(() => { const r = window.tw.tools.real; return { phase: r.spot?.phase, len: r.last?.lengthM, ring: !!r.ringAt, ok: r.spot?.preview?.ok, issues: r.spot?.preview?.issues?.slice(0, 2), worst: r.last?.worst, corners: r.last?.corners }; });
        if (s.phase === 'done' && s.ok) accepted++;
        if (!placed && s.phase === 'done' && s.ok) {
          placed = true;
          const kinds0 = await page.evaluate(() => window.tw.world.value.edges.length);
          await page.mouse.down(); await page.mouse.up(); await page.waitForTimeout(2500);
          const kinds = await page.evaluate((n0) => window.tw.world.value.edges.slice(n0).map((e) => e.kind ?? 0), kinds0);
          console.log(name.padEnd(9), cls.padEnd(12), 'placed: edges added', kinds.length, 'kinds', JSON.stringify([...new Set(kinds)]));
        }
        if (began && s.phase === 'done' && s.len > 0) ok++;
        lens.push(s.phase === 'done' ? Math.round(s.len) : s.phase);
        // SHOT=dir: the first routed ghost of each class, for a look at the fit
        if (process.env.SHOT && s.phase === 'done' && ok === 1) await page.screenshot({ path: `${process.env.SHOT}/t3495r_${name}_${cls}${TRAM ? '_tram' : ''}.png` });
        last = { began, ...s };
        await page.keyboard.press('Escape'); await page.waitForTimeout(300);
        await page.evaluate(async (tram) => { const ui = await import('/src/game/ui.ts'); if (tram) { ui.snapRealTram.value = true; ui.snapStreets.value = false; } else ui.snapReal.value = true; ui.tool.value = 'track'; }, TRAM);
        await page.waitForTimeout(300);
      }
      console.log(name.padEnd(9), cls.padEnd(12), tried ? ok + '/' + tried + ' routed, ' + accepted + ' accepted' : 'no candidate in view', lens.length ? 'm ' + lens.join(' ') : '', last ? JSON.stringify(last) : '');
    }
  }
  if (errors.length) console.log('errors: ' + errors.slice(0, 3).join(' | '));
} finally {
  await chrome.close();
}
