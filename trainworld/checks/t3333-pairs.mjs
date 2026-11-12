// T-3333: a line on quad track runs its trains on both pairs. A 6 km quad stretch, a line of three stations on the
// drawn pair: no switch until every stop has a station beside it on the other pair; then the switch, ticked, makes
// the hidden twin (one line in the Lines tab, the panel's trains the two pairs' together); a station beside a stop
// taken away leaves the switch on, the line on one pair and a notice naming the stop; undo brings the twin back.
// Pictures in artifacts/trainworld/t3333-*.png. From the maps root, on a Vite of your own
// (node trainworld/checks/dev-nowatch.mjs 6177):
//   node trainworld/checks/t3333-pairs.mjs [dev url]
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6177/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const RC = [-73.93, 40.675];
const SWITCH = 'Run trains on two pairs of tracks';

const chrome = await launchChrome({ prefix: 'tw-t3333-', size: [1400, 900], deviceScaleFactor: 1 });
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 7, build: { level: 0, keep: false } })); } catch {} });
const errors = [];
collectErrors(page, errors);
const shot = (name) => page.screenshot({ path: join(ARTIFACTS, `t3333-${name}.png`) });
const edit = (op) => page.evaluate(async (op) => { const r = await window.tw.client.edit(op); await new Promise((d) => setTimeout(d, 500)); return r; }, op);

try {
  await page.goto(`${BASE}?debug=1&new=1&mode=sandbox&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 180000 });
  await page.evaluate(([c]) => {
    const tw = window.tw, o = tw.lngLatToPack(c[0], c[1]);
    window.__q = {
      P: (dx, dy) => ({ x: o.x + dx, y: o.y + 3000 + dy }),
      // the edge whose line passes nearest x, y, and how far off it is
      near: (x, y, skip = () => false) => {
        const w = tw.world.value;
        let best = null, bd = 1e9;
        for (const e of w.edges) {
          if (skip(e)) continue;
          const a = w.nodes.get(e.a), b = w.nodes.get(e.b);
          if (x < Math.min(a.x, b.x) - 1 || x > Math.max(a.x, b.x) + 1) continue;
          const t = (x - a.x) / ((b.x - a.x) || 1), yy = a.y + t * (b.y - a.y), d = Math.abs(yy - y);
          if (d < bd) { bd = d; best = e; }
        }
        return best;
      },
    };
  }, [RC]);
  const made = await page.evaluate(async () => {
    const tw = window.tw, cl = tw.client, { P, near } = window.__q;
    const r = await cl.edit({ op: 'route', from: { kind: 'free', ...P(-3000, 0), level: 0 }, to: { kind: 'free', ...P(3000, 0), level: 0 }, pis: [], single: false, quad: true });
    await new Promise((d) => setTimeout(d, 800));
    const y0 = P(0, 0).y, ids = [];
    for (const dx of [-2000, 0, 2000]) {
      const p = P(dx, 0), e = near(p.x, p.y);
      await cl.edit({ op: 'addStation', at: { kind: 'edge', edge: e.id, x: p.x, y: p.y }, platform: 200, name: { '-2000': 'West', '0': 'Mid', '2000': 'East' }[dx] });
      await new Promise((d) => setTimeout(d, 300));
      const w = tw.world.value;
      ids.push([...w.nodes.values()].filter((n) => n.platform > 0).sort((a, b) => Math.hypot(a.x - p.x, a.y - p.y) - Math.hypot(b.x - p.x, b.y - p.y))[0].id);
    }
    const line = await cl.edit({ op: 'addLine', stops: ids, name: 'Q', colour: '#d6336c' });
    await new Promise((d) => setTimeout(d, 600));
    await cl.edit({ op: 'setTrainCount', line: line.line, trains: [7, 4, 2] });
    await new Promise((d) => setTimeout(d, 600));
    // the other pair: the parallel track 8 m off the drawn one
    const other = [...tw.world.value.edges].filter((e) => { const a = tw.world.value.nodes.get(e.a), b = tw.world.value.nodes.get(e.b); const d = Math.abs((a.y + b.y) / 2 - y0); return d > 5 && d < 16; });
    const side = other.length ? Math.sign((tw.world.value.nodes.get(other[0].a).y + tw.world.value.nodes.get(other[0].b).y) / 2 - y0) : 0;
    window.__q.side = side;
    const mid = (e) => (tw.world.value.nodes.get(e.a).y + tw.world.value.nodes.get(e.b).y) / 2 - y0;
    window.__q.off = other.length ? Math.max(...other.map((e) => Math.abs(mid(e)))) : 8;
    return { route: r.ok, line: line?.line, stops: ids, side };
  });
  check(made.route && made.line !== undefined && made.side !== 0, 'quad drawn and a line on its drawn pair', JSON.stringify(made));
  const L = String(made.line);
  await page.evaluate(async (l) => { (await import('/src/game/ui.ts')).selection.value = { kind: 'line', line: l }; }, L);
  await page.waitForFunction(() => !document.body.innerText.includes('Loading city data'), undefined, { timeout: 120000 }).catch(() => {});
  await wait(800);
  const hasSwitch = () => page.evaluate((s) => [...document.querySelectorAll('label.check')].some((x) => x.textContent.includes(s)), SWITCH);
  check(!(await hasSwitch()), 'no switch while the other pair has no stations');

  // stations beside each stop on the other pair
  const twins = await page.evaluate(async () => {
    const tw = window.tw, cl = tw.client, { P, near, side } = window.__q, ids = [];
    for (const dx of [-2000, 0, 2000]) {
      const p = P(dx, window.__q.off * side), e = near(p.x, p.y);
      const had = new Set([...tw.world.value.nodes.values()].filter((n) => n.platform > 0).map((n) => n.id));
      await cl.edit({ op: 'addStation', at: { kind: 'edge', edge: e.id, x: p.x, y: p.y }, platform: 200, name: { '-2000': 'West', '0': 'Mid', '2000': 'East' }[dx] });
      await new Promise((d) => setTimeout(d, 300));
      ids.push([...tw.world.value.nodes.values()].find((n) => n.platform > 0 && !had.has(n.id))?.id);
    }
    return ids;
  });
  await wait(800);
  check(await hasSwitch(), 'the switch shows once every stop has a station beside it');
  const before = await page.evaluate((l) => window.tw.world.value.lineStats[l].trainsNeeded, L);
  await page.locator('label.check', { hasText: SWITCH }).click();
  await wait(1200);
  const on = await page.evaluate((l) => {
    const w = window.tw.world.value, st = w.lineStats[l], twin = st.twin, ts = twin && w.lineStats[twin];
    const rows = [...document.querySelectorAll('.line-row, .lines-tab li, [data-line]')].length;
    return { pairs: st.pairs, twin, twinOf: ts?.twinOf, own: st.trainsNeeded, other: ts?.trainsNeeded, listed: w.save.lines.filter((x) => !x.twinOf).length, all: w.save.lines.length, twinStops: twin ? w.save.lines.find((x) => x.id === twin)?.stops : null, rows };
  }, L);
  check(on.pairs === 1 && on.twin && on.twinOf === L, 'ticked: the line runs on both pairs through a hidden twin', JSON.stringify(on));
  check(JSON.stringify(on.twinStops) === JSON.stringify(twins.map(String)), 'the twin stops at the stations beside the stops', `${JSON.stringify(on.twinStops)} vs ${JSON.stringify(twins)}`);
  check(on.listed === 1 && on.all === 2, 'one line listed, two in the world', JSON.stringify(on));
  check(on.own.every((x, k) => x === Math.ceil(before[k] / 2) && on.other[k] === x), 'each pair runs half the trains', `${before} -> ${on.own} + ${on.other}`);
  const high = await page.evaluate(() => [...document.querySelectorAll('tr')].find((r) => r.firstElementChild?.textContent.trim() === 'High')?.textContent.replace(/\s+/g, ' ').trim() ?? '');
  const want = on.own[0] + on.other[0];
  check(high.startsWith(`High${want}`), 'the panel shows the two pairs\' trains together', high);
  await shot('1-on');
  // built and running: trains on both pairs, half a headway apart
  await edit({ op: 'constructAll' });
  await page.evaluate(async () => {
    const tw = window.tw, { P } = window.__q;
    tw.clock.setRunning(false);
    tw.clock.set(Math.floor(tw.clock.now() / 86400) * 86400 + 9.2 * 3600);
    const q = tw.packToLngLat(P(-2150, 0).x, P(-2150, 0).y), [lng, lat] = Array.isArray(q) ? q : [q.lng, q.lat];
    tw.map.jumpTo({ center: [lng, lat], zoom: 17.3 });
  });
  await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }).catch(() => {});
  await wait(2500);
  await shot('3-trains');

  // a stop's station beside it taken away: switch on, one pair, a notice naming the stop
  const r = await edit({ op: 'removeStation', node: twins[1] });
  await wait(1000);
  const off = await page.evaluate((l) => { const st = window.tw.world.value.lineStats[l]; return { pairs: st.pairs, twin: st.twin ?? null, trains: st.trainsNeeded }; }, L);
  const notice = await page.evaluate(() => [...document.querySelectorAll('.alerts')].map((x) => x.textContent).join(' | '));
  check(r?.ok !== false && off.pairs === 2 && off.twin === null, 'a missing station beside a stop: on, one pair, no twin', JSON.stringify(off));
  check(/Trains run on one pair of tracks\. .+ has no platform on the other pair\./.test(notice), 'the notice names the stop', notice);
  check(await page.evaluate((s) => [...document.querySelectorAll('label.check')].find((x) => x.textContent.includes(s))?.querySelector('input')?.checked === true, SWITCH), 'the switch stays ticked');
  await shot('2-one-pair');
  await edit({ op: 'undo' });
  await wait(1000);
  const back = await page.evaluate((l) => { const st = window.tw.world.value.lineStats[l]; return { pairs: st.pairs, twin: st.twin ?? null }; }, L);
  check(back.pairs === 1 && back.twin === on.twin, 'undo brings the twin back', JSON.stringify(back));

  // a click on the twin selects the line
  await page.evaluate(async (t) => { (await import('/src/game/ui.ts')).selection.value = { kind: 'line', line: t }; }, on.twin);
  await wait(300);
  const sel = await page.evaluate(async () => (await import('/src/game/ui.ts')).selection.value);
  check(sel?.kind === 'line' && sel.line === L, 'selecting the twin shows the line', JSON.stringify(sel));

  // off: one pair, the whole line's trains
  await page.locator('label.check', { hasText: SWITCH }).click();
  await wait(1000);
  const unset = await page.evaluate((l) => { const w = window.tw.world.value, st = w.lineStats[l]; return { pairs: st.pairs ?? 0, twin: st.twin ?? null, trains: st.trainsNeeded, lines: w.save.lines.length }; }, L);
  check(unset.pairs === 0 && unset.twin === null && unset.lines === 1, 'unticked: one line, no twin', JSON.stringify(unset));

  const bad = errors.filter((e) => !/favicon|404/.test(String(e)));
  check(bad.length === 0, 'no console errors', bad.slice(0, 3).join(' | '));
} catch (e) {
  await shot('died').catch(() => {});
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `T-3333 FAILED: ${fail}` : 'T-3333 PASSED');
process.exit(fail ? 1 : 0);
