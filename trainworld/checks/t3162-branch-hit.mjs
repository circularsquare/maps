// T-3162: with single track picked, each track of double track drawn as two tracks is a branch point across its
// whole drawn width, at any zoom: a pointer anywhere over the outer track (and over the inner one) starts a branch
// from that track (`Snap.join`), the pointer just past the drawn pair does not. T-3163, T-3204: the branch dot is 19 px.
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
  check(dot && dot.shown && Math.abs(dot.w - 19) < 0.6, 'the branch dot is 19 px wide (10, then 15 px before)', JSON.stringify(dot));
  await shot('t3163-branch-dot');
  // ---- T-3190: the double main's loose end (x 3000): zoomed in, a pointer over one track near the end carries on that
  // track (the node, with a join on that side), zoomed out it joins neither; built, the join counts and the way follows
  const endAt = (z) => page.evaluate(([z]) => { window.tw.map.jumpTo({ center: window.__j.ll(window.__j.P(3000, 0)), zoom: z }); return null; }, [z]).then(() => settle());
  const endHover = async (off) => {
    const [ex, ey] = await page.evaluate(() => { const q = window.tw.map.project(window.__j.ll(window.__j.P(3000, 0))); return [q.x, q.y]; }).then(([x, y]) => [box.x + x, box.y + y]);
    // along: 5 px short of the end node, on the main
    await page.mouse.move(ex - 60, ey + off, { steps: 2 });
    await page.mouse.move(ex - 5, ey + off, { steps: 2 });
    await wait(300);
    return page.evaluate(() => {
      const h = window.tw.tools.hover;
      if (!h) return null;
      const c = window.tw.map.project(window.tw.toLngLat(h.lx, h.ly));
      const p = h.join ? window.tw.map.project(window.tw.toLngLat(h.join[0], h.join[1])) : null;
      return { kind: h.kind, side: p ? (p.y < c.y ? 'up' : 'down') : null };
    }).then((h) => ({ h, ex, ey }));
  };
  await endAt(15);
  {
    const { h } = await endHover(0);
    check(h?.kind === 'node' && h.side === null, 'T-3190 zoomed out: the loose end is the node, joining neither track', JSON.stringify(h));
  }
  // T-3204: the dot shows which of the two things a double's loose end will do, at any zoom: merge both tracks into the
  // single one (orange, centred) or carry on one track (yellow, offset to its side by at least 6 px)
  const dotOf = (off) => endHover(off).then(({ h }) => page.evaluate(() => {
    const t = window.tw.tools.hover, r = document.querySelector('.snap-ring');
    return { kind: t?.kind, merge: !!t?.merge, dot: t?.dot ?? null, cls: r?.className ?? '', bg: r ? getComputedStyle(r).backgroundColor : '' };
  }));
  const colours = {};
  for (const zoom of [15, 17, 20.5]) {
    await endAt(zoom);
    const L = await lanes();
    const m = await dotOf(0), up = await dotOf(-Math.max(5, (L.gap * L.t) / 2)), down = await dotOf(Math.max(5, (L.gap * L.t) / 2));
    check(m.kind === 'node' && m.merge && !m.dot && /merge/.test(m.cls), `T-3204 zoom ${zoom}: the pointer on the centre line merges both tracks (orange dot, centred)`, JSON.stringify(m));
    check(up.kind === 'node' && !up.merge && up.dot && up.dot[1] <= -5.9, `T-3204 zoom ${zoom}: above the centre carries on the upper track (dot offset up)`, JSON.stringify(up));
    check(down.kind === 'node' && !down.merge && down.dot && down.dot[1] >= 5.9, `T-3204 zoom ${zoom}: below the centre carries on the lower track (dot offset down)`, JSON.stringify(down));
    colours[zoom] = [m.bg, up.bg];
    check(colours[zoom][0] !== colours[zoom][1], `T-3204 zoom ${zoom}: merging both and carrying on one track are different colours`, JSON.stringify(colours[zoom]));
    if (zoom === 17) { await dotOf(-Math.max(5, (L.gap * L.t) / 2)); await shot('t3204-one-track-z17'); await dotOf(0); await shot('t3204-merge-z17'); }
    if (zoom === 15) { await dotOf(-6); await shot('t3204-one-track-z15'); await dotOf(0); await shot('t3204-merge-z15'); }
  }
  for (const zoom of [19, 20.5]) {
    await endAt(zoom);
    const L = await lanes();
    const centre = (L.gap * L.t) / 2, half = (L.share * L.gap) / 2;
    for (const [side, sg] of [['up', -1], ['down', 1]]) {
      for (const [what, off] of [['middle', centre], ['outer part', centre + half - 2]]) {
        const { h } = await endHover(sg * off);
        check(h?.kind === 'node' && h.side === side, `T-3190 zoom ${zoom}: the ${what} of the ${side === 'up' ? 'upper' : 'lower'} track at the loose end carries on that track`, JSON.stringify(h));
      }
    }
  }
  // build a branch from each track of the end, one after the other, and read what the sim made of it
  const singles = () => page.evaluate(() => window.tw.world.value.edges.filter((e) => e.tracks === 1).map((e) => e.id));
  const ways = [];
  for (const [side, sg, far] of [['up', -1, [3250, -150]], ['down', 1, [3250, 150]]]) {
    const before = await singles();
    await endAt(20.5);
    const L = await lanes();
    const { h, ex, ey } = await endHover(sg * (L.gap * L.t) / 2);
    check(h?.kind === 'node' && h.side === side, `T-3190: ${side} track picked to carry on`, JSON.stringify(h));
    await page.mouse.click(ex - 5, ey + sg * (L.gap * L.t) / 2);
    await wait(400);
    await page.evaluate(([f]) => window.tw.map.jumpTo({ center: window.__j.ll(window.__j.P(3300, 0)), zoom: 16 }), [far]);
    await settle();
    const [fx, fy] = await page.evaluate(([f]) => { const q = window.tw.map.project(window.__j.ll(window.__j.P(f[0], f[1]))); return [q.x, q.y]; }, [far]).then(([x, y]) => [box.x + x, box.y + y]);
    await page.mouse.move(fx - 20, fy, { steps: 2 });
    await page.mouse.move(fx, fy, { steps: 2 });
    await wait(1200);
    console.log('      ghost', JSON.stringify(await page.evaluate(() => { const a = window.tw.tools.lastPreview; return a ? { ok: a.ok, joins: a.joins ? [...a.joins] : null, issues: a.issues } : null; })));
    await shot(`t3190-far-${side}`);
    await page.mouse.click(fx, fy);
    await wait(300);
    await page.keyboard.press("Enter");
    await wait(1500);
    await shot(`t3190-after-${side}`);
    console.log('      edges', JSON.stringify(await page.evaluate(() => window.tw.world.value.edges.map((e) => [e.id, e.tracks, e.join, e.oneWay, e.a, e.b]))));
    const b = await page.evaluate(async ([before]) => {
      const t0 = Date.now();
      for (;;) {
        const w = window.tw.world.value, e = w.edges.find((x) => x.tracks === 1 && !before.includes(x.id));
        if (e) return { id: e.id, join: e.join, way: e.oneWay ?? 0 };
        if (Date.now() - t0 > 15000) return null;
        await new Promise((r) => setTimeout(r, 150));
      }
    }, [before]);
    check(b && b.join !== 0 && b.way !== 0, `T-3190: built from the ${side} track: its join counts and it runs one way`, JSON.stringify(b));
    ways.push(b?.way);
    await endAt(20.5);
    await shot(`t3190-built-${side}`);
    // delete it again so the next one starts from a loose end
    if (b) await page.evaluate(async (id) => { await window.tw.client.edit({ op: 'removeEdge', edge: id }); }, b?.id).catch(() => {});
    await wait(600);
  }
  check(ways[0] && ways[1] && ways[0] !== ways[1], 'T-3190: the two tracks give opposite ways', JSON.stringify(ways));
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
