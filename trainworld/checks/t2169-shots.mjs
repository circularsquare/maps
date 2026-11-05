// T-2169: pictures of t2169-water.mjs' render files (run it with --png first), painted by the diagram's own draw.ts in a
// static page, one per view and theme, into artifacts/trainworld/t2169/.
//   node trainworld/checks/t2169-shots.mjs <views.json>
//   views: [{ city, name, at: "Jamaica" | [x, y] | ["Jamaica", dx, dy] (pitches from it) | "fit", px, w?, h?, dark?, debug? }]
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { ARTIFACTS, HERE, launchChrome } from './lib.mjs';

const views = JSON.parse(readFileSync(process.argv[2]));
const OUT = process.env.WATER_OUT ?? `${ARTIFACTS}t2169/`;
const { build } = await import(pathToFileURL(join(HERE, '../app/node_modules/vite/dist/node/index.js')).href);
const app = fileURLToPath(new URL('../app', import.meta.url)).replace(/\\/g, '/');
const entry = `export { paint, WATER_SHARE } from '${app}/src/diagram/draw.ts'; export { SHADES } from '${app}/src/game/shadeTable.ts'; export { paintNames } from '${app}/src/diagram/names.ts'; export { mixOklab } from '${app}/src/game/palette.ts';`;
const bundle = await build({
  configFile: false, logLevel: 'silent',
  plugins: [{ name: 'e', resolveId(id) { if (id.endsWith('shots-entry')) return '\0shots-entry'; }, load(id) { if (id === '\0shots-entry') return entry; } }],
  build: { write: false, minify: false, lib: { entry: 'shots-entry', formats: ['iife'], name: 'TW' } },
});
writeFileSync(`${OUT}paint.js`, (Array.isArray(bundle) ? bundle[0] : bundle).output.find((o) => o.type === 'chunk').code);
const cities = [...new Set(views.map((v) => v.city))];
writeFileSync(`${OUT}shots.html`, `<!doctype html><meta charset="utf-8"><body style="margin:0"><canvas id="c"></canvas>
<script src="paint.js"></script>${cities.map((c) => `<script src="render-${c}.js"></script>`).join('')}
<script>
window.render = (city, at, px, w, h, dark, dbg) => {
  const D = window.TW_DATA[city], c = document.getElementById('c');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d');
  const d = { ...D, lines: D.lines.map((l) => ({ ...l, cmds: Float64Array.from(l.cmds), path: undefined })), small: Uint8Array.from(D.small),
    water: { rivers: D.water.rivers.map((r) => Float64Array.from(r)), riverEnds: (D.water.riverEnds ?? []).map((r) => Float64Array.from(r)), sea: Float64Array.from(D.water.sea), bands: D.water.bands } };
  let cx, cy;
  // "fit": the whole drawing in the picture (px then the largest that fits, at most the view's)
  if (at === 'fit') { const b = D.box; cx = (b[0] + b[2]) / 2; cy = (b[1] + b[3]) / 2; px = Math.min(px, (0.96 * w * D.pitch) / (b[2] - b[0]), (0.96 * h * D.pitch) / (b[3] - b[1])); }
  else if (typeof at === 'string') { const i = D.text.indexOf(at); if (i < 0) return 'no ' + at; cx = D.x[i]; cy = D.y[i]; }
  else if (typeof at[0] === 'string') { const i = D.text.indexOf(at[0]); if (i < 0) return 'no ' + at[0]; cx = D.x[i] + at[1] * D.pitch; cy = D.y[i] + at[2] * D.pitch; }
  else [cx, cy] = at;
  const scale = px / d.pitch;
  const theme = dark
    ? { bg: '#262120', ink: '#2b2b2b', pill: '#ffffff', selected: '#d9788e', border: '#73798b', edge: '#d8d8d8', water: TW.mixOklab('#262120', TW.SHADES.dark.water, TW.WATER_SHARE) }
    : { bg: '#fbf7f0', ink: '#2b2b2b', pill: '#ffffff', selected: '#d9788e', border: '#73798b', water: TW.mixOklab('#fbf7f0', TW.SHADES.light.water, TW.WATER_SHARE) };
  TW.paint(ctx, d, w, h, cx, cy, scale, theme);
  if (dbg) { ctx.save(); ctx.setTransform(scale, 0, 0, -scale, w / 2 - cx * scale, h / 2 + cy * scale); ctx.lineWidth = 1.5 / scale; for (const r of D.runs) { ctx.strokeStyle = r.river === undefined ? '#000' : 'hsl(' + ((r.river * 137) % 360) + ' 80% 30%)'; ctx.lineWidth = 3 / scale; ctx.beginPath(); for (let k = 0; k < r.p.length; k += 2) k ? ctx.lineTo(r.p[k], r.p[k + 1]) : ctx.moveTo(r.p[k], r.p[k + 1]); ctx.stroke(); ctx.strokeStyle = '#f00'; ctx.lineWidth = 3 / scale; for (let k = 0; k < r.marks.length; k += 4) { ctx.beginPath(); ctx.moveTo(r.marks[k], r.marks[k + 1]); ctx.lineTo(r.marks[k + 2], r.marks[k + 3]); ctx.stroke(); ctx.fillStyle = '#f00'; ctx.fillRect(r.marks[k + 2] - 3 / scale, r.marks[k + 3] - 3 / scale, 6 / scale, 6 / scale); } ctx.lineWidth = 1.5 / scale; } for (let k = 0; k < D.failed.length; k += 4) { ctx.fillStyle = '#00f'; ctx.fillRect(D.failed[k] - 4 / scale, D.failed[k + 1] - 4 / scale, 8 / scale, 8 / scale); ctx.fillStyle = '#0a0'; ctx.fillRect(D.failed[k + 2] - 4 / scale, D.failed[k + 3] - 4 / scale, 8 / scale, 8 / scale); ctx.strokeStyle = '#00f'; ctx.beginPath(); ctx.moveTo(D.failed[k], D.failed[k + 1]); ctx.lineTo(D.failed[k + 2], D.failed[k + 3]); ctx.stroke(); } ctx.strokeStyle = '#f0f'; ctx.lineWidth = 3 / scale; for (let k = 0; k < (D.falsePts ?? []).length; k += 2) { ctx.beginPath(); ctx.arc(D.falsePts[k], D.falsePts[k + 1], 7 / scale, 0, 7); ctx.stroke(); } ctx.strokeStyle = '#e00'; ctx.lineWidth = 1 / scale; for (const v of D.seaRings ?? []) { ctx.beginPath(); for (let k = 0; k < v.length; k += 2) k ? ctx.lineTo(v[k], v[k + 1]) : ctx.moveTo(v[k], v[k + 1]); ctx.closePath(); ctx.stroke(); } ctx.restore(); }
  TW.paintNames(ctx, D.plan, d, w, h, cx, cy, scale, 1, { ink: dark ? '#e8e2da' : '#2b2b2b', halo: theme.bg, font: 'system-ui, sans-serif' });
  return 'ok';
};
</script>`);
const { ctx, close } = await launchChrome({ prefix: 'tw-diagwater-', size: [1400, 900], args: [] });
const page = ctx.pages()[0] ?? (await ctx.newPage());
await page.goto(pathToFileURL(`${OUT}shots.html`).href);
for (const v of views) {
  const w = v.w ?? 1200, h = v.h ?? 800;
  const r = await page.evaluate(([c, at, px, w, h, dark, dbg]) => window.render(c, at, px, w, h, dark, dbg), [v.city, v.at, v.px, w, h, !!v.dark, !!v.debug]);
  if (r !== 'ok') { console.log(v.city, v.name, r); continue; }
  const file = `${OUT}${v.name}.png`;
  await page.locator('#c').screenshot({ path: file });
  console.log(file);
}
await close();
