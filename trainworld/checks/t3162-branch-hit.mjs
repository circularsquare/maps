// T-3162: with single track picked, each track of double track drawn as two tracks is a branch point across its
// whole drawn width, at any zoom: a pointer anywhere over the outer track (and over the inner one) starts a branch
// from that track (`Snap.join`), the pointer just past the drawn pair does not. T-3163: the branch dot is 15 px.
// From the maps root, on a Vite of your own (node trainworld/checks/dev-nowatch.mjs 6112):
//   node trainworld/checks/t3162-branch-hit.mjs [dev url]
// Pictures in artifacts/trainworld/t3162-*.png and t3163-*.png.
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6112/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const RC = [-73.93, 40.675];

const chrome = await launchChrome({ prefix: 'tw-t3162-', size: [1400, 900], deviceScaleFactor: 1 });
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 7, build: { level: 0, keep: false } })); } catch {} });
const errors = [];
collectErrors(page, errors);
const shot = (name) => page.screenshot({ path: join(ARTIFACTS, `${name}.png`) });
const settle = async (ms = 600) => {
  await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }).catch(() => {});
  await wait(ms);
};

try {
  await page.goto(`${BASE}?debug=1&new=1&mode=sandbox&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 180000 });
  await page.evaluate(([c]) => {
    const tw = window.tw, o = tw.lngLatToPack(c[0], c[1]);
    const P = (dx, dy) => ({ x: o.x + dx, y: o.y + 3000 + dy });
    const ll = (p) => { const q = tw.packToLngLat(p.x, p.y); return Array.isArray(q) ? q : [q.lng, q.lat]; };
    window.__j = { P, ll };
  }, [RC]);
  await page.evaluate(async () => {
    const { P } = window.__j, cl = window.tw.client;
    await cl.edit({ op: 'route', from: { kind: 'free', ...P(-3000, 0), level: 0 }, to: { kind: 'free', ...P(3000, 0), level: 0 }, pis: [], single: false });
  });
  await wait(800);
  await page.evaluate(async () => {
    const ui = await import('/src/game/ui.ts');
    ui.tool.value = 'track'; ui.buildLevel.value = 0; ui.keepPlacing.value = false; ui.quadTrack.value = false;
    ui.singleTrack.value = true; ui.tramTrack.value = false; ui.highSpeed.value = false; ui.designSpeed.value = 60;
  });
  const box = await (await page.$('#map')).boundingBox();
  const lanes = () => page.evaluate(async () => (({ gap, t, share }) => ({ gap, t, share }))((await import('/src/map/renderer.ts')).laneState(window.tw.map.getZoom())));
  const hoverAt = async (x, y) => {
    await page.mouse.move(x - 25, y, { steps: 2 });
    await page.mouse.move(x, y, { steps: 2 });
    await wait(250);
    return page.evaluate(() => {
      const h = window.tw.tools.hover;
      if (!h) return null;
      const p = h.join ? window.tw.map.project(window.tw.toLngLat(h.join[0], h.join[1])) : null;
      const c = window.tw.map.project(window.tw.toLngLat(h.lx, h.ly));
      return { kind: h.kind, side: p ? (p.y < c.y ? 'up' : 'down') : null };
    });
  };
  for (const zoom of [17.5, 19, 20.5]) {
    await page.evaluate(([z]) => window.tw.map.jumpTo({ center: window.__j.ll(window.__j.P(-900, 0)), zoom: z }), [zoom]);
    await settle();
    const L = await lanes();
    const [cx, cy] = await page.evaluate(() => { const q = window.tw.map.project(window.__j.ll(window.__j.P(-900, 0))); return [q.x, q.y]; }).then(([x, y]) => [box.x + x, box.y + y]);
    const centre = (L.gap * L.t) / 2, half = (L.share * L.gap) / 2, outer = centre + half;
    console.log(`zoom ${zoom}: gap ${L.gap.toFixed(1)} px, t ${L.t.toFixed(2)}, track centre ${centre.toFixed(1)} px off the line, half width ${half.toFixed(1)}`);
    if (!(L.t > 0.9)) { console.log('  (tracks not fully apart, skipped)'); continue; }
    for (const [side, sg] of [['up', -1], ['down', 1]]) {
      // the outer part of the drawn track (2 px inside its outer edge), its middle, then 8 px past the pair
      for (const [what, off, want] of [['outer part', outer - 2, true], ['middle', centre, true], ['inner part', Math.max(1, centre - half + 2), true], ['past the pair', outer + 8, false]]) {
        const h = await hoverAt(cx, cy + sg * off);
        const ok = want ? h?.kind === 'edge' && h.side === side : h?.kind !== 'edge';
        check(ok, `zoom ${zoom}: ${what} of the ${side === 'up' ? 'upper' : 'lower'} track ${want ? 'branches from it' : 'starts no branch'}`, JSON.stringify(h));
      }
    }
    if (zoom === 20.5) {
      await hoverAt(cx, cy - (outer - 2));
      await shot('t3162-outer-upper');
      await hoverAt(cx, cy + (outer - 2));
      await shot('t3162-outer-lower');
    }
  }
  // the dot, T-3163: branch 15 px, join 15 px
  await page.evaluate(() => window.tw.map.jumpTo({ center: window.__j.ll(window.__j.P(-900, 0)), zoom: 20.5 }));
  await settle();
  const L = await lanes();
  const [cx, cy] = await page.evaluate(() => { const q = window.tw.map.project(window.__j.ll(window.__j.P(-900, 0))); return [q.x, q.y]; }).then(([x, y]) => [box.x + x, box.y + y]);
  await hoverAt(cx, cy - ((L.gap * L.t) / 2));
  const dot = await page.evaluate(() => { const r = document.querySelector('.snap-ring.branch'); if (!r) return null; const b = r.getBoundingClientRect(); return { w: b.width, h: b.height, shown: getComputedStyle(r).display !== 'none' }; });
  check(dot && dot.shown && Math.abs(dot.w - 15) < 0.6, 'the branch dot is 15 px wide (10 px before)', JSON.stringify(dot));
  await shot('t3163-branch-dot');
  const bad = errors.filter((e) => !/favicon|404/.test(String(e)));
  check(bad.length === 0, 'no console errors', bad.slice(0, 3).join(' | '));
} catch (e) {
  await shot('t3162-died').catch(() => {});
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `T-3162 FAILED: ${fail}` : 'T-3162 PASSED');
process.exit(fail ? 1 : 0);
