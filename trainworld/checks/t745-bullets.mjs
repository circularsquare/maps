// T-745: line bullets' symbols centred in their circle, measured on the rendered pixels. On the
// real save (T-007-real-nyc.save), for light and dark, 100% and 90% UI, at devicePixelRatio 1 and 2:
// the Lines tab's bullets, the Stations tab's small ones, the line inspector's large one, and the
// map tags' bullets (ui/bullet.ts through map/trackTagRows.ts's lineBullet, in a track tag on the
// map, normal and small, plus symbols the save lacks). For each bullet, the ink (pixels toward the
// symbol's colour, with their partial coverage at the edges) is boxed and the box's centre compared
// with the circle's as drawn (its edges found the same way). Fails when any offset is over LIMIT
// device pixels. Writes artifacts/trainworld/t745-bullets-<TAG>.json and crops
// t745-<TAG>-<config>-{tag,dock,lg}.png.
//   node trainworld/checks/t745-bullets.mjs [url]     (from the maps root; default localhost:6160)
// Env: LIMIT (device px, default 0.6), TAG (file name tag, default "after"). About two minutes.
import { mkdtemp, rm, readdir, readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const BASE = process.argv[2] ?? 'http://localhost:6160/';
const SAVE = fileURLToPath(new URL('../T-007-real-nyc.save', import.meta.url));
// half a device pixel, plus what the rasteriser itself moves a glyph at 1x (Nunito's "S" 0.55)
const LIMIT = Number(process.env.LIMIT ?? 0.6);
const TAG = process.env.TAG ?? 'after';
const OUT = 'artifacts/trainworld';

async function playwright() {
  if (process.env.PLAYWRIGHT_MODULE) return import(process.env.PLAYWRIGHT_MODULE);
  try { return await import('playwright'); } catch { /* not installed here */ }
  const npx = join(process.env.LOCALAPPDATA ?? '', 'npm-cache', '_npx');
  for (const d of existsSync(npx) ? await readdir(npx) : []) {
    const p = join(npx, d, 'node_modules', 'playwright', 'index.mjs');
    if (existsSync(p)) return import(pathToFileURL(p).href);
  }
  throw new Error('no Playwright');
}

/** In the page: put every kind of bullet on screen, each marked `data-t745="<where>"`. */
async function stage(page, inspector) {
  await page.evaluate(async (inspector) => {
    const ui = await import('/src/game/ui.ts');
    const { lineBullet } = await import('/src/map/trackTagRows.ts');
    document.querySelectorAll('[data-t745]').forEach((e) => e.removeAttribute('data-t745'));
    document.getElementById('t745-tag')?.remove();
    const lines = window.tw.world.value.save.lines;
    ui.tab.value = 'lines';
    if (inspector) {
      // a line inspector: its large bullet and the small ones at its transfer stops
      const busy = lines.reduce((a, b) => (b.stops.length > a.stops.length ? b : a));
      ui.selection.value = { kind: 'line', line: String(busy.id) };
      await new Promise((r) => setTimeout(r, 600));
      document.querySelectorAll('#inspector .chip.lg').forEach((e) => e.setAttribute('data-t745', 'inspector-lg'));
      document.querySelectorAll('#inspector .chip.sm').forEach((e) => e.setAttribute('data-t745', 'inspector-sm'));
      return;
    }
    // the Lines tab (no selection): every line's bullet
    ui.selection.value = null;
    await new Promise((r) => setTimeout(r, 600));
    document.querySelectorAll('#dock .chip:not(.sm):not(.lg)').forEach((e) => e.setAttribute('data-t745', 'lines-tab'));
    // map tags: one track tag holding every line's bullet, normal and small
    const tag = document.createElement('div');
    tag.id = 't745-tag';
    tag.className = 'tool-tag track-tag';
    Object.assign(tag.style, { position: 'absolute', left: '8px', bottom: '8px', zIndex: 50, maxWidth: '520px', display: 'flex', flexDirection: 'column', gap: '6px', padding: '8px', background: 'var(--map-land)' });
    // 6 px apart, not the tag's 2, so the circles' edges measure clear of their neighbours
    const row = (small) => { const d = document.createElement('div'); d.className = 'tag-lines'; d.style.gap = '6px'; for (const l of lines) { const b = lineBullet(l, small); b.setAttribute('data-t745', small ? 'tag-sm' : 'tag'); d.append(b); } return d; };
    tag.append(row(false), row(true));
    // symbols the save lacks: digits, two characters, a descender, kana
    const extra = document.createElement('div');
    extra.className = 'tag-lines'; extra.style.gap = '6px';
    for (const [i, t] of ['1', '4', '7', 'A', 'W', 'Q', 'g', 'GS', 'A1', '42', '7X', 'あ', '山'].entries()) for (const small of [false, true]) {
      const b = lineBullet({ ...lines[i % lines.length], letter: t }, small);
      b.setAttribute('data-t745', /[　-鿿]/.test(t) ? 'tag-kana' : small ? 'tag-sm-extra' : 'tag-extra');
      extra.append(b);
    }
    tag.append(extra);
    window.tw.map.getContainer().append(tag);
    await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  }, inspector);
}

/** In the page: measure every marked bullet in the screenshot `b64` (device pixels). */
function measure(b64) {
  return (async () => {
    const dpr = devicePixelRatio;
    const img = await createImageBitmap(await (await fetch('data:image/png;base64,' + b64)).blob());
    const cv = new OffscreenCanvas(img.width, img.height), g = cv.getContext('2d');
    g.drawImage(img, 0, 0);
    const px = g.getImageData(0, 0, img.width, img.height).data, W = img.width;
    const rgb = (s) => (s.match(/[\d.]+/g) ?? []).slice(0, 3).map(Number);
    const out = [], norm = new OffscreenCanvas(1, 1).getContext('2d');
    for (const el of document.querySelectorAll('[data-t745]')) {
      const r = el.getBoundingClientRect();
      if (r.width < 4 || r.top < 0 || r.bottom > innerHeight || r.left < 0 || r.right > innerWidth) continue;
      // whole and uncovered (not cut by a scrolling list's edge, not under a hover tag)
      const corners = [[r.left + r.height / 2, r.top + 1], [r.left + r.height / 2, r.bottom - 1], [r.right - r.height / 2, r.top + 1], [r.right - r.height / 2, r.bottom - 1]];
      if (!corners.every(([x, y]) => { const h = document.elementFromPoint(x, y); return h && (h === el || el.contains(h)); })) continue;
      // the circle's colour is the chip's --c (whether the span or an SVG shape paints it)
      const cs = getComputedStyle(el), T = rgb(cs.color);
      norm.fillStyle = '#000'; norm.fillStyle = cs.getPropertyValue('--c').trim() || cs.backgroundColor;
      const hex = String(norm.fillStyle), B = hex.startsWith('#') ? [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)) : rgb(hex);
      const d = T.map((t, i) => t - B[i]), dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
      if (dd < 900) continue;
      const L = r.left * dpr, R = r.right * dpr, Tp = r.top * dpr, Bt = r.bottom * dpr;
      const cx = (L + R) / 2, cy = (Tp + Bt) / 2, rad = (Bt - Tp) / 2;
      const x0 = Math.floor(L), x1 = Math.ceil(R), y0 = Math.floor(Tp), y1 = Math.ceil(Bt);
      const colMax = new Float64Array(x1 - x0), rowMax = new Float64Array(y1 - y0);
      for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
        // inside the pill, 1.5 device px clear of its anti-aliased rim
        const sx = Math.min(Math.max(x + 0.5, L + rad), R - rad), dist = Math.hypot(x + 0.5 - sx, y + 0.5 - cy);
        if (dist > rad - 1.5 * dpr) continue;
        const o = (y * W + x) * 4;
        const a = Math.min(1, Math.max(0, ((px[o] - B[0]) * d[0] + (px[o + 1] - B[1]) * d[1] + (px[o + 2] - B[2]) * d[2]) / dd));
        if (a > colMax[x - x0]) colMax[x - x0] = a;
        if (a > rowMax[y - y0]) rowMax[y - y0] = a;
      }
      const edges = (m, o) => {
        const TH = 0.1;
        let i = 0, j = m.length - 1;
        while (i < m.length && m[i] < TH) i++;
        while (j >= 0 && m[j] < TH) j--;
        if (i > j) return null;
        return [o + i + 1 - m[i], o + j + m[j]];
      };
      const h = edges(colMax, x0), v = edges(rowMax, y0);
      if (!h || !v) continue;
      // the circle as drawn (the browser may snap a box's background to whole pixels): its edges
      // found from outside along the middle row and a column clear of the ink, with coverage
      const at = (x, y) => { const o = (y * W + x) * 4; return [px[o], px[o + 1], px[o + 2]]; };
      const cover = (P, O) => { const e = B.map((b, i) => b - O[i]), ee = e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; return ee < 900 ? null : Math.min(1, Math.max(0, ((P[0] - O[0]) * e[0] + (P[1] - O[1]) * e[1] + (P[2] - O[2]) * e[2]) / ee)); };
      const scan = (x, y, sx, sy, n) => {
        const O = at(x, y);
        for (let i = 1; i < n; i++) {
          const a = cover(at(x + sx * i, y + sy * i), O);
          if (a === null) return null;
          if (a > 0.1) return { i, a };
        }
        return null;
      };
      const out1 = Math.ceil(1.5 * dpr), my = Math.floor(cy), mx = Math.floor(cx);
      const sl = scan(x0 - out1, my, 1, 0, 3 * out1 + 2), sr = scan(x1 - 1 + out1, my, -1, 0, 3 * out1 + 2);
      const st = scan(mx, y0 - out1, 0, 1, 3 * out1 + 2), sb = scan(mx, y1 - 1 + out1, 0, -1, 3 * out1 + 2);
      // a circle not found (something else beside it) is not measured
      if (!(sl && sr && st && sb)) continue;
      const left = x0 - out1 + sl.i + 1 - sl.a, right = x1 - 1 + out1 - sr.i + sr.a;
      const top = y0 - out1 + st.i + 1 - st.a, bottom = y1 - 1 + out1 - sb.i + sb.a;
      const ccx = (left + right) / 2, ccy = (top + bottom) / 2;
      if (Math.abs(bottom - top - (Bt - Tp)) > 1.5 || Math.abs(right - left - (R - L)) > 1.5) continue;
      const text = el.textContent;
      out.push({ where: el.getAttribute('data-t745'), text, dx: (h[0] + h[1]) / 2 - ccx, dy: (v[0] + v[1]) / 2 - ccy, snap: [ccx - cx, ccy - cy], ink: [h[1] - h[0], v[1] - v[0]], rect: [r.left, r.top, r.width, r.height] });
    }
    return out;
  })();
}

const { chromium } = await playwright();
const results = [], all = {};
const report = (name, ok, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
const bytes = Array.from(await readFile(SAVE));

for (const dpr of [1, 2]) {
  const profile = await mkdtemp(join(tmpdir(), 'tw-t745-'));
  const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, deviceScaleFactor: dpr, args: ['--use-angle=d3d11'] });
  try {
    const page = await ctx.newPage();
    page.on('pageerror', (e) => console.log('pageerror', e.message));
    await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
    await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
    await page.evaluate((data) => { const b = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes: b }, [b]); }, bytes);
    await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
    await page.waitForTimeout(2000);
    for (const shade of ['light', 'dark']) for (const scale of [1, 0.9]) {
      const name = `${shade}-ui${Math.round(scale * 100)}-dpr${dpr}`;
      await page.evaluate(async ([s, k]) => { const ui = await import('/src/game/ui.ts'); ui.shade.value = s; ui.setDisplay('uiScale', k); }, [shade, scale]);
      const m = [];
      for (const inspector of [true, false]) {
        await stage(page, inspector);
        // a symbol whose face was still loading is drawn in the fallback's measure until the face
        // loads; a hover tag is built again on every move, so build it again once fonts are in
        await page.evaluate(() => document.fonts.ready);
        await stage(page, inspector);
        await page.waitForTimeout(300);
        const png = await page.screenshot({ type: 'png' });
        m.push(...await page.evaluate(measure, png.toString('base64')));
        if (inspector) {
          const lg = await page.evaluate(() => document.querySelector('[data-t745="inspector-lg"]')?.getBoundingClientRect().toJSON());
          if (lg) await page.screenshot({ path: `${OUT}/t745-${TAG}-${name}-lg.png`, clip: { x: lg.x - 4, y: lg.y - 4, width: lg.width + 8, height: lg.height + 8 } });
        }
      }
      all[name] = m;
      const by = {};
      for (const b of m) (by[b.where] ??= []).push(b);
      for (const [where, bs] of Object.entries(by)) {
        const worst = bs.reduce((w, b) => (Math.max(Math.abs(b.dx), Math.abs(b.dy)) > Math.max(Math.abs(w.dx), Math.abs(w.dy)) ? b : w));
        const mean = (f) => (bs.reduce((s, b) => s + f(b), 0) / bs.length).toFixed(2);
        const detail = `(${bs.length} bullets, mean dx ${mean((b) => b.dx)} dy ${mean((b) => b.dy)}, worst "${worst.text}" dx ${worst.dx.toFixed(2)} dy ${worst.dy.toFixed(2)} device px)`;
        report(`${name} ${where}`, Math.max(Math.abs(worst.dx), Math.abs(worst.dy)) <= LIMIT, detail);
      }
      // crops: the tag's first bullets and the Lines tab
      const tag = await page.evaluate(() => document.getElementById('t745-tag')?.getBoundingClientRect().toJSON());
      if (tag) await page.screenshot({ path: `${OUT}/t745-${TAG}-${name}-tag.png`, clip: { x: tag.x, y: tag.y, width: Math.min(260, tag.width), height: tag.height } });
      const dock = await page.evaluate(() => document.getElementById('dock')?.getBoundingClientRect().toJSON());
      if (dock) await page.screenshot({ path: `${OUT}/t745-${TAG}-${name}-dock.png`, clip: { x: dock.x, y: dock.y, width: dock.width, height: Math.min(500, dock.height) } });
    }
  } finally {
    await ctx.close();
    await rm(profile, { recursive: true, force: true }).catch(() => {});
  }
}
await writeFile(`${OUT}/t745-bullets-${TAG}.json`, JSON.stringify(all, null, 1));
process.exit(results.length && results.every(Boolean) ? 0 : 1);
