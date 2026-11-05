// T-2255: the network diagram's graphs as the game itself builds them (the clock worker, from the network
// as drawn on the map), for the diagram checks. The game's graph is not the one diagram-saves.mjs builds
// from the same save (other places by float rounding, a complex fewer in New York), and the layout grows
// any difference into a different drawing, so the checks also lay out the game's own graphs. Run as a
// script it imports each save into a new game in headless Chrome, opens the diagram, and keeps the graph
// the page sends its layout worker, the names' measured widths, and what the worker sent back (its times,
// and a hash of its places and pills, which the checks compare with their own layout of the same graph:
// the layout uses no engine-dependent maths, diagram/fmath.ts, so the two agree bit for bit).
//   node trainworld/checks/diagram-game.mjs [url] [nyc,tyo,l8]      (about 30 s a city; writes
//   artifacts/trainworld/diagram-game/<city>.json)
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { ARTIFACTS, REAL_SAVE, NOISE, launchChrome, collectErrors, importSave } from './lib.mjs';

const OUT = `${ARTIFACTS}diagram-game/`;
const FILES = { nyc: REAL_SAVE, tyo: fileURLToPath(new URL('../data/packs/real/tyo.save', import.meta.url)), l8: fileURLToPath(new URL('../../artifacts/trainworld/t2290/line8.save', import.meta.url)), xp: fileURLToPath(new URL('../../artifacts/trainworld/t2405/express.save', import.meta.url)) };
/** the code a layout, drawing and names depend on: a capture made before it changed has no page figures to compare */
const LAYOUT_SRC = ['layout.ts', 'blocks.ts', 'lsqr.ts', 'fmath.ts', 'draw.ts', 'names.ts', 'waterReal.ts'];
const GRAPH_SRC = ['graph.ts'];
const srcHash = (files) => {
  const h = createHash('sha1');
  for (const f of files) h.update(readFileSync(new URL(`../app/src/diagram/${f}`, import.meta.url)));
  return h.digest('hex').slice(0, 12);
};

/** a hash of numbers (a layout's places, a drawing's pills), to the bit */
export function bitsHash(...arrays) {
  const h = createHash('sha1');
  for (const v of arrays) h.update(Buffer.from(Float64Array.from(v).buffer));
  return h.digest('hex').slice(0, 12);
}
export const pillsFlat = (pills) => pills.flatMap((p) => Array.from(p));

/** the game's graph for `city` from its capture, or null: `g` the graph, `text`, `width`, `width2` and
 * `fontPx` the names as the page measured them, `page` what its worker sent back, `fresh` whether the
 * layout code is as it was at the capture (else `page` is of older code), `graphFresh` the same for graph.ts; `frame`
 * and `zones` the graph's frame and the zones whose water masks the page read (T-2541; none in older captures) */
export function gameGraph(city) {
  const file = `${OUT}${city}.json`;
  if (!existsSync(file)) return null;
  const c = JSON.parse(readFileSync(file, 'utf8')), r = c.graph;
  const g = { ...r, key: Int32Array.from(r.key), x: Float64Array.from(r.x), y: Float64Array.from(r.y), ea: Int32Array.from(r.ea), eb: Int32Array.from(r.eb), first: Int32Array.from(r.first), on: Int32Array.from(r.on), lines: r.lines.map((l) => ({ ...l, chain: Int32Array.from(l.chain), stop: Uint8Array.from(l.stop) })) };
  // a capture from before T-2405 has no strokes: one per colour on each edge, as the layout then drew them
  g.stroke = r.stroke ? Int32Array.from(r.stroke) : Int32Array.from(r.on, (l, k) => { let e = 0; while (r.first[e + 1] <= k) e++; for (let j = r.first[e]; j <= k; j++) if (r.lines[r.on[j]].colour === r.lines[l].colour) return r.on[j]; return l; });
  return { g, frame: c.frame, zones: c.zones, text: c.names.text.map((t) => t ?? ''), width: Float32Array.from(c.names.width), width2: Float32Array.from(c.names.width2), fontPx: c.names.fontPx, page: c.page, captured: c.captured, fresh: c.layoutSrc === srcHash(LAYOUT_SRC), graphFresh: c.graphSrc === srcHash(GRAPH_SRC) };
}

/** the city names a check was asked for, each `<city>` (the save's graph, diagram-saves.mjs) or `<city>-game` */
export const splitCity = (name) => (name.endsWith('-game') ? [name.slice(0, -5), true] : [name, false]);

async function capture(base, city) {
  const errors = [];
  const { ctx, close } = await launchChrome({ prefix: `tw-diagram-game-${city}-`, size: [1400, 900] });
  try {
    const page = await ctx.newPage();
    await page.addInitScript(() => {
      try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 2 })); } catch {}
      const plain = (o) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, ArrayBuffer.isView(v) ? Array.from(v) : Array.isArray(v) && k === 'lines' ? v.map(plain) : v]));
      window.__dg = { sent: [], back: [] };
      const pm = Worker.prototype.postMessage;
      Worker.prototype.postMessage = function (m, t) {
        if (m && m.kind === 'graph' && m.graph) {
          window.__dg.sent.push({ seq: m.seq, frame: m.frame, zones: m.water?.zones, graph: plain(m.graph), names: { text: m.names.text, width: Array.from(m.names.width), width2: Array.from(m.names.width2), fontPx: m.names.fontPx } });
          if (!this.__dg) {
            this.__dg = true;
            this.addEventListener('message', (e) => {
              const d = e.data;
              if (d?.kind !== 'layout') return;
              window.__dg.back.push({ seq: d.seq, x: Array.from(d.layout.x), y: Array.from(d.layout.y), pills: d.drawing.pills.map((p) => Array.from(p)), hooks: Array.from(d.drawing.hooks ?? []), ms: { ...d.layout.ms, ...d.ms }, pieces: d.layout.pieces.length });
            });
          }
        }
        return pm.call(this, m, t);
      };
    });
    collectErrors(page, errors, NOISE);
    const sep = base.includes('?') ? '&' : '?';
    await page.goto(`${base}${sep}debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
    await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
    await importSave(page, await readFile(FILES[city]));
    await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 180000 });
    await page.waitForTimeout(1500);
    // the camera stays where the save left it: the graph's places are measured in the map's frame there
    await page.keyboard.press('m');
    await page.waitForFunction(() => window.__dg.back.length > 0 && window.__dg.back.length === window.__dg.sent.length, undefined, { timeout: 300000, polling: 500 });
    // any graph sent after the first (a network state arriving late) is laid out too
    await page.waitForTimeout(3000);
    await page.waitForFunction(() => window.__dg.back.length === window.__dg.sent.length, undefined, { timeout: 300000, polling: 500 });
    const { sent, back } = await page.evaluate(() => window.__dg);
    if (errors.length) console.log(`${city}: page errors: ${errors.slice(0, 3).join(' | ')}`);
    return { sent, back };
  } finally {
    await close();
  }
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const base = process.argv[2] ?? 'http://localhost:6194/';
  const cities = (process.argv[3] ?? 'nyc,tyo,l8').split(',');
  mkdirSync(OUT, { recursive: true });
  for (const city of cities) {
    const t0 = Date.now();
    const { sent, back } = await capture(base, city);
    // the first graph, laid out from nothing; later ones are rebuilds from it, which a check does not repeat
    const s = sent[0], b = back.find((x) => x.seq === s.seq);
    const page = { seq: b.seq, ms: b.ms, pieces: b.pieces, hooks: b.hooks, places: bitsHash(b.x, b.y), pills: bitsHash(pillsFlat(b.pills)), graphsSent: sent.length };
    writeFileSync(`${OUT}${city}.json`, JSON.stringify({ city, captured: new Date().toISOString(), url: base, layoutSrc: srcHash(LAYOUT_SRC), graphSrc: srcHash(GRAPH_SRC), frame: s.frame, zones: s.zones, graph: s.graph, names: s.names, page }));
    console.log(`${city}: ${s.graph.key.length} complexes, ${sent.length} graph${sent.length === 1 ? '' : 's'} sent; the page laid it out in ${(b.ms.total / 1000).toFixed(2)} s (B ${(b.ms.b / 1000).toFixed(2)}), drawing ${(b.ms.drawing / 1000).toFixed(2)} s, names ${(b.ms.names / 1000).toFixed(2)} s; ${((Date.now() - t0) / 1000).toFixed(0)} s in all`);
  }
}
