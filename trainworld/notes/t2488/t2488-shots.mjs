// T-2488 spike: pictures of t2488-borders.mjs' render files, painted by the diagram's own draw.ts and names.ts in a
// static page (light theme), one per view and border variant, into artifacts/trainworld/diaglines/.
//   node trainworld/checks/t2488-shots.mjs <views.json>     views: [{ city, name, at: "Jamaica" | [x, y], px, w?, h? }]
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { ARTIFACTS, HERE, launchChrome } from './lib.mjs';

const views = JSON.parse(readFileSync(process.argv[2]));
const variants = (process.argv[3] ?? 'now,wide,lines').split(',');
const OUT = `${ARTIFACTS}diaglines/`;
const { build } = await import(pathToFileURL(join(HERE, '../app/node_modules/vite/dist/node/index.js')).href);
const app = fileURLToPath(new URL('../app', import.meta.url)).replace(/\\/g, '/');
const entry = `export { paint } from '${app}/src/diagram/draw.ts'; export { paintNames } from '${app}/src/diagram/names.ts';`;
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
const fixArr = (o) => o;
window.render = (city, variant, at, px, w, h) => {
  const D = window.TW_DATA[city], c = document.getElementById('c');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d');
  const d = { ...D, lines: D.lines.map((l) => ({ ...l, cmds: Float64Array.from(l.cmds), path: undefined })), small: Uint8Array.from(D.small), borders: variant === 'none' ? undefined : { cmds: Float64Array.from(D.borders[variant]) } };
  let cx, cy;
  if (typeof at === 'string') { const i = D.text.indexOf(at); if (i < 0) return 'no ' + at; cx = D.x[i]; cy = D.y[i]; } else [cx, cy] = at;
  const scale = px / d.pitch;
  const theme = { bg: '#fbf5ef', ink: '#2b2b2b', pill: '#ffffff', selected: '#d9788e', border: '#73798b' };
  TW.paint(ctx, d, w, h, cx, cy, scale, theme);
  const plan = D.names;
  TW.paintNames(ctx, plan, d, w, h, cx, cy, scale, 1, { ink: '#2b2b2b', halo: '#fbf5ef', font: 'system-ui, sans-serif' });
  return 'ok';
};
</script>`);
const { ctx, close } = await launchChrome({ prefix: 'tw-diaglines-', size: [1400, 900], args: [] });
const page = ctx.pages()[0] ?? (await ctx.newPage());
await page.goto(pathToFileURL(`${OUT}shots.html`).href);
for (const v of views) for (const variant of variants) {
  const w = v.w ?? 1200, h = v.h ?? 800;
  const r = await page.evaluate(([c, va, at, px, w, h]) => window.render(c, va, at, px, w, h), [v.city, variant, v.at, v.px, w, h]);
  if (r !== 'ok') { console.log(v.city, v.name, r); continue; }
  const file = `${OUT}${v.name}-${variant}.png`;
  await page.locator('#c').screenshot({ path: file });
  console.log(file);
}
await close();
