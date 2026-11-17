// T-3495: what the Real world network layer draws vs what the track tool can snap to. For each place:
// All setting (snap off): every drawn feature by kind; Passenger with snap on: the drawn features, and how
// many of their sample points the snap ring finds (tools.real.ring). Writes the samples to a json for
// the graph comparison (checks/t3495-graph-compare.py). From the maps root:
//   node trainworld/checks/t3495-snap-survey.mjs [url] [out.json]
import { writeFile } from 'node:fs/promises';
import { launchChrome, collectErrors } from './lib.mjs';

const url = process.argv[2] ?? 'http://localhost:6189/';
const OUT = process.argv[3] ?? 'artifacts/trainworld/t3495-survey.json';
const PLACES = {
  london: [-0.12, 51.5], tokyo: [139.75, 35.68], chicago: [-87.65, 41.88], berlin: [13.4, 52.5], kaohsiung: [120.3, 22.63],
  amman: [35.93, 31.95], sydney: [151.2, -33.87], mumbai: [72.87, 19.07], basel: [7.6, 47.56], istanbul: [29.0, 41.0],
};
const only = process.env.ONLY?.split(',');
const chrome = await launchChrome({ prefix: 'tw-t3495-', size: [1400, 900] });
const errors = [];
const out = {};
try {
  const page = await chrome.ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${url}?debug=1&new=1&demandWorkers=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });
  const set = (all, snap, center, zoom) => page.evaluate(async ([all, snap, c, z]) => {
    const ui = await import('/src/game/ui.ts');
    ui.display.value = { ...ui.display.value, realNetwork: all ? 'all' : 'passenger' };
    ui.snapReal.value = snap;
    ui.tool.value = snap ? 'track' : 'select';
    window.tw.map.jumpTo({ center: c, zoom: z });
  }, [all, snap, center, zoom]);
  const read = (ring) => page.evaluate((ring) => {
    const tw = window.tw, map = tw.map;
    const layers = map.getStyle().layers.filter((l) => l.id.startsWith('tw-real')).map((l) => l.id);
    const fs = map.queryRenderedFeatures(undefined, { layers });
    const byKind = {}, seen = new Set(), feats = [], allFeats = [];
    let sampled = 0, ringed = 0;
    const r = map.getContainer().getBoundingClientRect();
    for (const f of fs) {
      const p = f.properties, key = (p.k ?? '?') + (p.n ? ' n' : '') + (p.x ? ' x' : '') + (p.r === 1 ? ' r1' : '');
      const g = f.geometry, lines = g.type === 'LineString' ? [g.coordinates] : g.type === 'MultiLineString' ? g.coordinates : [];
      const pts = [];
      for (const l of lines) for (const c of l) {
        const id = c[0].toFixed(5) + ',' + c[1].toFixed(5);
        if (seen.has(id)) continue; seen.add(id);
        const q = map.project(c);
        if (q.x < 20 || q.y < 20 || q.x > r.width - 20 || q.y > r.height - 20) continue;
        pts.push(c);
        if (ring) { sampled++; if (tw.tools.real.ring(q.x, q.y)) ringed++; }
      }
      byKind[key] = (byKind[key] ?? 0) + pts.length;
      if (pts.length) (ring ? feats : allFeats).push({ key, pts });
    }
    return { byKind, sampled, ringed, feats, allFeats };
  }, ring);
  for (const [name, c] of Object.entries(PLACES)) {
    if (only && !only.includes(name)) continue;
    for (const z of [11]) {
      await set(true, false, c, z);
      await page.waitForTimeout(6000);
      const all = await read(false);
      await set(false, true, c, z);
      await page.waitForTimeout(6000);
      const pax = await read(true);
      out[name + '@' + z] = { all: all.byKind, pax: pax.byKind, sampled: pax.sampled, ringed: pax.ringed, feats: pax.feats, allFeats: all.allFeats };
      console.log(name, z, 'ALL', JSON.stringify(all.byKind), '| SNAP drawn', JSON.stringify(pax.byKind), 'ring', pax.ringed + '/' + pax.sampled);
    }
  }
  await writeFile(OUT, JSON.stringify(out));
  if (errors.length) console.log('errors: ' + errors.slice(0, 3).join(' | '));
} finally {
  await chrome.close();
}
