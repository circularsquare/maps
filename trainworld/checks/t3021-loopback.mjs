// T-3021/T-3022: turning a line's end round a loop of track. On a new game in New York, a layout like
// Anita's Pittsburgh line 5 built with the edit API: a stem A - B - M, a junction J, and a loop through
// L and K back into J; past A a second, smaller loop with no station. A line A, B, M, L, K: in stop
// editing M's marker says "Click to loop back here"; a click on it turns the line (stops out and back
// over the stem, one way); undo takes it back; the marker then opens it again; and the far end turns
// too. T-3050: a line with a station-bearing loop at each end (entered stop by stop) is one row per station
// with a bend above and below, in the plain and the related-lines view. Times the worker's stop tiers. Screenshots artifacts/trainworld/t3021-*.png.
// From the maps root, on a Vite of your own:
//   node trainworld/checks/t3021-loopback.mjs [dev url]
// About a minute.
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';
import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6153/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const C = [-73.93, 40.668];

const chrome = await launchChrome({ prefix: 'tw-t3021-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
collectErrors(page, errors);
const edit = (op) => page.evaluate((op) => window.tw.client.edit(op), op);
const out = { when: new Date().toISOString() };

try {
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 180000 });
  const built = await page.evaluate(async (c) => {
    const tw = window.tw, cl = tw.client, o = tw.lngLatToPack(c[0], c[1]);
    const P = (dx, dy) => ({ x: o.x + dx, y: o.y + dy });
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p.x, n.y - p.y); if (d < bd) { bd = d; best = n.id; } } return best; };
    const lv = -1, pis = (...ps) => ps.map((p) => ({ x: p.x, y: p.y, level: lv }));
    const free = (p) => ({ kind: 'free', x: p.x, y: p.y, level: lv });
    const at = (n) => ({ kind: 'node', node: n });
    const res = [];
    const go = async (from, to, ps) => { res.push(await cl.edit({ op: 'route', from, to, pis: ps, single: false })); await pause(300); };
    const A = P(-4000, 0), B = P(-2000, 0), M = P(0, 0), J = P(1000, 0), L = P(2000, 750), K = P(3150, 375);
    await go(free(A), free(B), []);
    const nA = near(A), nB = near(B);
    await go(at(nB), free(M), []);
    const nM = near(M);
    await go(at(nM), free(J), []);
    const nJ = near(J);
    await go(at(nJ), free(L), pis(P(1200, 0)));
    const nL = near(L);
    await go(at(nL), free(K), pis(P(2800, 1500)));
    const nK = near(K);
    await go(at(nK), at(nJ), pis(P(3500, -750), P(1600, -150)));
    // the small loop past A
    const Jw = P(-5000, 0), X = P(-5800, 600);
    await go(at(nA), free(Jw), []);
    const nJw = near(Jw);
    await go(at(nJw), free(X), pis(P(-5200, 0)));
    const nX = near(X);
    await go(at(nX), at(nJw), pis(P(-6400, 1050), P(-6900, -500), P(-5600, -150)));
    for (const n of [nA, nB, nM, nL, nK]) { res.push(await cl.edit({ op: 'addStation', at: at(n), platform: 160, name: '' })); await pause(200); }
    res.push(await cl.edit({ op: 'constructAll' }));
    await pause(500);
    const line = await cl.edit({ op: 'addLine', stops: [nA, nB, nM, nL, nK], name: 'Loop check', colour: '#7b3fb3' });
    await pause(800);
    return { n: { A: nA, B: nB, M: nM, J: nJ, L: nL, K: nK, Jw: nJw, X: nX }, line: line.line, ok: line.ok, refused: res.filter((r) => !r.ok).map((r) => JSON.stringify(r.issues ?? r).slice(0, 140)) };
  }, C);
  check(built.ok && !built.refused.length, 'the layout is built with the edit API', built.refused.join(' | '));
  const N = built.n, id = built.line;
  if (!built.ok) throw new Error('no line');
  const names = ['on', 'route', 'reroute', 'cut', 'reverse', 'close', 'open', 'turn', 'unturn'];
  const tiers = () => page.evaluate(async ([line, nodes, names]) => {
    const { queryClock } = await import('/src/workers/clockClient.ts');
    const t = performance.now();
    const d = await queryClock({ what: 'stopTiers', line });
    const ms = performance.now() - t;
    const out = { ms };
    for (const [k, n] of Object.entries(nodes)) for (let i = 0; i + 2 < d.length; i += 3) if (d[i] === n) out[k] = names[d[i + 1]];
    return out;
  }, [id, N, names]);
  const info = () => page.evaluate((num) => {
    const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === num), st = w.lineStats[l.id];
    return { stops: l.stops.map(Number), loop: st.loop, oneWay: st.oneWay, status: st.status, round: Math.round(st.roundTripByLevel[0]) };
  }, id);
  const s0 = await info();
  console.log('open line', JSON.stringify(s0));

  // stop editing, as Edit stops does
  await page.evaluate(async ([num, c]) => {
    const tw = window.tw, ui = await import('/src/game/ui.ts');
    const l = tw.world.value.save.lines.find((x) => x.num === num);
    tw.selection.value = { kind: 'line', line: l.id };
    ui.tool.value = 'line';
    ui.lineDraft.value = { line: num, stops: l.stops.map(Number) };
    tw.map.jumpTo({ center: [c[0] + 0.005, c[1] + 0.002], zoom: 13.3 });
  }, [id, C]);
  await page.waitForTimeout(2500);
  const t1 = await tiers();
  check(t1.M === 'turn', 'M, the stop before the loop, offers the turn', JSON.stringify(t1));
  check(t1.A !== 'close', 'the first stop offers no balloon back past the stem', t1.A);
  check(t1.A === 'turn', 'A, before the small loop past it, offers the turn too', t1.A);
  out.tiersMs = [t1.ms];

  /** the screen point of station node n's marker: the marker nearest where the node projects */
  const markerAt = (n) => page.evaluate((n) => {
    const tw = window.tw, node = tw.world.value.nodes.get(String(n)) ?? [...tw.world.value.nodes.values()].find((x) => Number(x.id) === n);
    const [lng, lat] = tw.packToLngLat(node.x, node.y);
    const p = tw.map.project([lng, lat]);
    const rect = tw.map.getContainer().getBoundingClientRect();
    let best = null, bd = 1e9;
    for (const el of document.querySelectorAll('.stop-marks > div')) {
      if (el.style.display === 'none') continue;
      const m = /translate\((-?[\d.]+)px, (-?[\d.]+)px\)/.exec(el.style.transform);
      if (!m) continue;
      const d = Math.hypot(+m[1] - p.x, +m[2] - p.y);
      if (d < bd) { bd = d; best = { x: +m[1] + rect.left, y: +m[2] + rect.top, d }; }
    }
    return best;
  }, n);
  let mk = await markerAt(N.M);
  check(mk && mk.d < 30, 'M has a marker on screen', JSON.stringify(mk));
  await page.mouse.move(mk.x, mk.y);
  await page.waitForTimeout(500);
  const tag = await page.evaluate(() => document.querySelector('.stop-tag')?.innerText ?? '');
  check(/Click to loop back here/.test(tag), 'its hover says what the click does', tag.replace(/\n/g, ' / '));
  await page.screenshot({ path: join(ARTIFACTS, 't3021-marker.png') });

  await page.mouse.click(mk.x, mk.y);
  await page.waitForTimeout(1500);
  const s1 = await info();
  const want = [N.A, N.B, N.M, N.L, N.K, N.M, N.B, N.A];
  console.log('turned', JSON.stringify(s1));
  const sameish = JSON.stringify(s1.stops) === JSON.stringify(want);
  check(s1.loop !== null && s1.oneWay, 'the click turns the line, one way', JSON.stringify(s1));
  check(sameish || s1.loop === 'circle', 'its stops run out, round the loop and back over the stem', `${s1.stops} want ${want} (or a circle round both loops)`);
  out.turned = s1;
  out.roundBefore = s0.round;
  await page.waitForTimeout(1500);
  await page.screenshot({ path: join(ARTIFACTS, 't3021-turned.png') });
  const t2 = await tiers();
  console.log('tiers turned', JSON.stringify(t2));

  // undo
  await edit({ op: 'undo' });
  await page.waitForTimeout(1200);
  const s2 = await info();
  check(JSON.stringify(s2.stops) === JSON.stringify(s0.stops) && !s2.oneWay && s2.loop === null, 'undo puts the line back', JSON.stringify(s2));

  // the far end first, then the near one: both ends
  await edit({ op: 'turnEnd', line: id, station: N.A });
  await page.waitForTimeout(1200);
  const s3 = await info();
  console.log('west turned', JSON.stringify(s3));
  const t3 = await tiers();
  console.log('tiers west turned', JSON.stringify(t3));
  if (s3.loop === 'balloon') {
    const k = Object.entries(t3).find(([k, v]) => v === 'turn')?.[0];
    if (k) await edit({ op: 'turnEnd', line: id, station: N[k] });
    await page.waitForTimeout(1200);
  }
  const s4 = await info();
  console.log('both', JSON.stringify(s4));
  check(s4.loop === 'circle' && s4.oneWay, 'both ends turned: a one-way circle round both loops', JSON.stringify(s4));
  await page.evaluate(([c]) => window.tw.map.jumpTo({ center: [c[0] - 0.012, c[1] + 0.002], zoom: 12.4 }), [C]);
  await page.waitForTimeout(2000);
  await page.screenshot({ path: join(ARTIFACTS, 't3021-both-ends.png') });

  // opening: back to the turned east end only, then the marker opens it
  await edit({ op: 'setStops', line: id, stops: s0.stops });
  await edit({ op: 'turnEnd', line: id, station: N.M });
  await page.waitForTimeout(1500);
  const t5 = await tiers();
  const s5 = await info();
  console.log('east turned again', JSON.stringify(s5), JSON.stringify(t5));
  if (s5.loop === 'balloon') {
    check(t5.M === 'unturn', 'on the turned line M offers to open it', JSON.stringify(t5));
    await page.evaluate(([c]) => window.tw.map.jumpTo({ center: [c[0] + 0.005, c[1] + 0.002], zoom: 13.3 }), [C]);
    await page.waitForTimeout(1500);
    mk = await markerAt(N.M);
    await page.mouse.move(mk.x, mk.y);
    await page.waitForTimeout(500);
    const tag2 = await page.evaluate(() => document.querySelector('.stop-tag')?.innerText ?? '');
    check(/Click to stop looping back here/.test(tag2), 'its hover says so', tag2.replace(/\n/g, ' / '));
    await page.mouse.click(mk.x, mk.y);
    await page.waitForTimeout(1500);
    const s6 = await info();
    check(JSON.stringify(s6.stops) === JSON.stringify(s0.stops), 'the click opens it again', JSON.stringify(s6));
  }

  // T-3050: a stem with a loop past each end, each loop with stations of its own, stops entered in order
  // (not turned with the click): the panel shows the stem once, a bend above and a bend below
  const C2 = [-73.86, 40.74];
  const two = await page.evaluate(async (c) => {
    const tw = window.tw, cl = tw.client, o = tw.lngLatToPack(c[0], c[1]);
    const P = (dx, dy) => ({ x: o.x + dx, y: o.y + dy });
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p.x, n.y - p.y); if (d < bd) { bd = d; best = n.id; } } return best; };
    const lv = -1, pis = (...ps) => ps.map((p) => ({ x: p.x, y: p.y, level: lv }));
    const free = (p) => ({ kind: 'free', x: p.x, y: p.y, level: lv });
    const at = (n) => ({ kind: 'node', node: n });
    const res = [];
    const go = async (from, to, ps) => { res.push(await cl.edit({ op: 'route', from, to, pis: ps, single: false })); await pause(300); };
    const A = P(-4000, 0), B = P(-2000, 0), M = P(0, 0), J = P(1000, 0), L = P(2000, 750), K = P(3150, 375);
    await go(free(A), free(B), []);
    const nA = near(A), nB = near(B);
    await go(at(nB), free(M), []);
    const nM = near(M);
    await go(at(nM), free(J), []);
    const nJ = near(J);
    await go(at(nJ), free(L), pis(P(1200, 0)));
    const nL = near(L);
    await go(at(nL), free(K), pis(P(2800, 1500)));
    const nK = near(K);
    await go(at(nK), at(nJ), pis(P(3500, -750), P(1600, -150)));
    const Jw = P(-5000, 0), X1 = P(-5800, 600), X2 = P(-6700, -100);
    await go(at(nA), free(Jw), []);
    const nJw = near(Jw);
    await go(at(nJw), free(X1), pis(P(-5200, 0)));
    const nX1 = near(X1);
    await go(at(nX1), free(X2), pis(P(-6400, 1050)));
    const nX2 = near(X2);
    await go(at(nX2), at(nJw), pis(P(-6200, -800), P(-5600, -150)));
    const nm = ['Ash', 'Birch', 'Cedar', 'Larch', 'Kauri', 'Xylia', 'Yew'];
    for (const [k, n] of [nA, nB, nM, nL, nK, nX1, nX2].entries()) { res.push(await cl.edit({ op: 'addStation', at: at(n), platform: 160, name: nm[k] })); await pause(200); }
    res.push(await cl.edit({ op: 'constructAll' }));
    await pause(500);
    const line = await cl.edit({ op: 'addLine', stops: [nA, nB, nM, nL, nK], name: 'Two loops', colour: '#1d5fb8' });
    await pause(800);
    const stops = [nA, nB, nM, nL, nK, nM, nB, nA, nX1, nX2, nA];
    const set = await cl.edit({ op: 'setStops', line: line.line, stops });
    await pause(500);
    const one = await cl.edit({ op: 'lineOneWay', line: line.line, oneWay: true });
    await pause(1000);
    return { n: { A: nA, B: nB, M: nM, L: nL, K: nK, X1: nX1, X2: nX2 }, line: line.line, stops, ok: line.ok && set.ok && one.ok, refused: res.filter((r) => !r.ok).map((r) => JSON.stringify(r.issues ?? r).slice(0, 140)) };
  }, C2);
  check(two.ok && !two.refused.length, 'the two-loop layout and its stops are built', two.refused.join(' | '));
  const T = two.n;
  const panel = () => page.evaluate(() => {
    const ul = document.querySelector('.stops'), rows = [...(ul?.querySelectorAll(':scope > li:not(.loop-end)') ?? [])];
    return { turned: !!ul?.classList.contains('turned'), rows: rows.map((li) => ({ name: li.querySelector('.stop-name')?.textContent ?? '', arc: (li.querySelector('.arc')?.className ?? '').replace('arc', '').trim(), once: li.classList.contains('oneway') })) };
  });
  const showLine = (num, center, zoom) => page.evaluate(async ([num, c, z]) => {
    const tw = window.tw, ui = await import('/src/game/ui.ts');
    ui.tool.value = null;
    const l = tw.world.value.save.lines.find((x) => x.num === num);
    tw.selection.value = { kind: 'line', line: l.id };
    tw.map.jumpTo({ center: c, zoom: z });
  }, [num, center, zoom]);
  await page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); ui.relatedLines.value = false; });
  await showLine(two.line, [C2[0] - 0.012, C2[1] + 0.002], 12.4);
  await page.waitForTimeout(2500);
  const p2 = await panel();
  console.log('two loops panel', JSON.stringify(p2));
  const rowNames = p2.rows.map((r) => r.name);
  check(p2.turned && rowNames.length === 7 && new Set(rowNames).size === 7, 'the two-loop line is one row per station (7 rows)', rowNames.join(', '));
  check(p2.rows.filter((r) => r.arc === 'start').length === 2 && p2.rows.filter((r) => r.arc === 'end').length === 2, 'a bend above and a bend below', p2.rows.map((r) => r.arc || '-').join(' '));
  check(p2.rows.filter((r) => r.once).length === 4, 'each loop station is half-filled (one way), the stem stations full', String(p2.rows.filter((r) => r.once).length));
  await page.evaluate(() => document.querySelector('.stops')?.scrollIntoView({ block: 'center' }));
  await page.waitForTimeout(300);
  await page.screenshot({ path: join(ARTIFACTS, 't3050-two-loops.png') });
  // the related-lines view draws it too, with lines that meet it on the stem and on a loop
  await edit({ op: 'addLine', stops: [T.B, T.M], name: 'Stem cross', colour: '#d9531e' });
  await edit({ op: 'addLine', stops: [T.X2, T.A], name: 'Top cross', colour: '#2a9d5c' });
  await page.waitForTimeout(1200);
  // the related-lines view draws it too
  await page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); ui.relatedLines.value = true; });
  await page.waitForTimeout(1500);
  const rel = await page.evaluate(() => document.querySelectorAll('.stops.related-rows > li[data-r]').length);
  check(rel === 7, 'the related-lines view has the same seven rows', String(rel));
  await page.evaluate(() => document.querySelector('.stops')?.scrollIntoView({ block: 'center' }));
  await page.waitForTimeout(300);
  await page.screenshot({ path: join(ARTIFACTS, 't3050-two-loops-related.png') });
  await page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); ui.relatedLines.value = false; });
  // a stem station comes out of both passes, a loop station once (the closing stop moves with its first)
  const infoOf = (num) => page.evaluate((num) => { const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === num); return l.stops.map(Number); }, num);
  const lenBefore = (await infoOf(two.line)).length;
  await edit({ op: 'setStops', line: two.line, stops: two.stops.filter((s) => s !== T.B).concat([]) });
  const afterStem = await infoOf(two.line);
  check(afterStem.length === lenBefore - 2 && !afterStem.includes(T.B), 'taking the stem station out of the stops takes both passes', afterStem.join(','));
  await edit({ op: 'undo' });
  await page.waitForTimeout(800);
  // Anita's own order: the line starts on a loop station (Amsterdam's first stop sat past the stem's end)
  await edit({ op: 'setStops', line: two.line, stops: [T.X1, T.X2, T.A, T.B, T.M, T.L, T.K, T.M, T.B, T.A, T.X1] });
  await page.waitForTimeout(1500);
  const p3 = await panel();
  check(p3.turned && p3.rows.length === 7 && p3.rows.filter((r) => r.once).length === 4, 'the same shape read from a line that starts on a loop station', p3.rows.map((r) => r.name).join(', '));
  await page.getByRole('button', { name: /Edit stops/ }).click();
  await page.waitForTimeout(800);
  await page.locator('.stops > li:not(.loop-end)').first().locator('button.danger').click();
  await page.waitForTimeout(1500);
  const s7 = await infoOf(two.line);
  check(s7[0] === s7[s7.length - 1] && !s7.includes(T.X1) && s7.length === 10, 'a loop station comes out once and the line closes on its new first stop', s7.join(','));
  await page.getByRole('button', { name: /Stop editing/ }).click().catch(() => {});
  await edit({ op: 'undo' });
  await page.waitForTimeout(800);
  for (let k = 0; k < 4; k++) out.tiersMs.push((await tiers()).ms);
  console.log('stop tiers ms', out.tiersMs.map(Math.round).join(' '));
  check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} catch (e) {
  await page.screenshot({ path: join(ARTIFACTS, 't3021-died.png') }).catch(() => {});
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
await writeFile(join(ARTIFACTS, 't3021-loopback.json'), JSON.stringify(out, null, 1)).catch(() => {});
console.log(fail ? `T-3021 FAILED: ${fail}` : 'T-3021 PASSED');
process.exit(fail ? 1 : 0);
