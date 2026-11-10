// T-3094: one-way single track. A ring of single track in Brooklyn (four stations, built with the edit API) and two
// lines that both run over the piece between the first two stations. Making that piece one way through the track
// inspector's switch asks first and sends the line whose way back ran against it round the long way; Reverse
// direction turns it round and the other line's path moves; undo restores each step. Screenshots of the arrows in
// light and dark, and of the inspector, in artifacts/trainworld/t3094-*.png.
// From the maps root, on a Vite of your own (node trainworld/checks/dev-nowatch.mjs 6178):
//   node trainworld/checks/t3094-oneway.mjs [dev url]
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6178/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

const RC = [-73.945, 40.672], H = 1200;
const chrome = await launchChrome({ prefix: 'tw-t3094-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
collectErrors(page, errors);
const edit = (op) => page.evaluate((op) => window.tw.client.edit(op), op);
const settle = async (ms = 800) => { await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }); await page.waitForTimeout(ms); };
const line = (num) => page.evaluate((num) => { const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === num); if (!l) return null; const st = w.lineStats[l.id]; return { id: l.id, path: st.path, back: st.stopS[1].at(-1), name: l.name }; }, num);
const flag = (id) => page.evaluate((id) => window.tw.world.value.edges.find((e) => e.id === id)?.oneWay ?? 0, id);

try {
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 180000 });

  const ring = await page.evaluate(async ([c, h]) => {
    const tw = window.tw, cl = tw.client, o = tw.lngLatToPack(c[0], c[1]);
    const P = (dx, dy) => ({ x: o.x + dx, y: o.y + dy });
    const mids = [P(h, 0), P(0, h), P(-h, 0), P(0, -h)], corners = [P(h, h), P(-h, h), P(-h, -h), P(h, -h)];
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p.x, n.y - p.y); if (d < bd) { bd = d; best = n.id; } } return best; };
    const pi = (p) => [{ x: p.x, y: p.y, level: -1 }];
    const res = [], nodes = [];
    res.push(await cl.edit({ op: 'route', from: { kind: 'free', x: mids[0].x, y: mids[0].y, level: -1 }, to: { kind: 'free', x: mids[1].x, y: mids[1].y, level: -1 }, pis: pi(corners[0]), single: true }));
    await pause(300);
    nodes.push(near(mids[0]), near(mids[1]));
    for (const k of [2, 3]) {
      res.push(await cl.edit({ op: 'route', from: { kind: 'node', node: nodes[k - 1] }, to: { kind: 'free', x: mids[k].x, y: mids[k].y, level: -1 }, pis: pi(corners[k - 1]), single: true }));
      await pause(300);
      nodes.push(near(mids[k]));
    }
    res.push(await cl.edit({ op: 'route', from: { kind: 'node', node: nodes[3] }, to: { kind: 'node', node: nodes[0] }, pis: pi(corners[3]), single: true }));
    await pause(500);
    for (const n of nodes) { res.push(await cl.edit({ op: 'addStation', at: { kind: 'node', node: n }, platform: 200, name: '' })); await pause(200); }
    res.push(await cl.edit({ op: 'constructAll' }));
    await pause(500);
    const a = await cl.edit({ op: 'addLine', stops: [nodes[0], nodes[1]], name: 'Ring A', colour: '#0a7d3b' });
    await pause(400);
    const b = await cl.edit({ op: 'addLine', stops: [nodes[1], nodes[0]], name: 'Ring B', colour: '#b5179e' });
    await pause(500);
    const e01 = tw.world.value.edges.find((e) => (e.a === nodes[0] && e.b === nodes[1]) || (e.a === nodes[1] && e.b === nodes[0]));
    return { a: a.line, b: b.line, ok: a.ok && b.ok, e01: e01?.id, single: e01?.tracks, refused: res.filter((r) => !r.ok).map((r) => JSON.stringify(r.issues ?? r).slice(0, 80)) };
  }, [RC, H]);
  check(ring.ok && !ring.refused.length && ring.single === 1, 'a ring of single track and two lines over one piece', ring.refused.join(' | '));
  const all = await page.evaluate(() => { const w = window.tw.world.value; return w.save.lines.map((l) => ({ id: l.id, num: l.num, name: l.name, path: w.lineStats[l.id].path })); });
  const A = all.find((l) => l.name === 'Ring A'), B = all.find((l) => l.name === 'Ring B');
  check(A.path.length === 1 && B.path.length === 1, 'both lines run the one piece', `A ${A.path} B ${B.path}`);

  // select the piece, make it one way through the inspector
  await page.evaluate((e) => { window.tw.selection.value = { kind: 'track', edge: e }; }, ring.e01);
  await page.waitForTimeout(1200);
  const sw = page.getByText('Run trains one way');
  check(await sw.count() === 1, 'the inspector has the switch on single track');
  await page.evaluate(() => { window.tw.map.jumpTo({ center: window.tw.map.getCenter(), zoom: 14.2 }); });
  await sw.click();
  const dialog = page.locator('dialog.confirmation');
  await dialog.waitFor({ timeout: 20000 });
  const rows = await dialog.locator('li').allTextContents();
  console.log('confirmation:', await dialog.locator('p').first().textContent(), '|', rows.join(' | '));
  check(rows.some((r) => /Ring A takes another way/.test(r)) && rows.some((r) => /Ring B takes another way/.test(r)), 'the confirmation lists the lines whose way changes');
  await dialog.getByRole('button', { name: 'Run one way' }).click();
  await page.waitForTimeout(1500);
  const f1 = await flag(ring.e01);
  check(f1 === 1 || f1 === 2, 'the piece runs one way', `flag ${f1}`);
  const A1 = await line(A.num), B1 = await line(B.num);
  const startsAtA = A.path[0] & 1;
  // the flag is a to b (1): whichever line ran a to b keeps its path; the other goes round the long way
  const moved = [A1, B1].filter((l, k) => l.path.length === 3);
  check(moved.length === 1, 'the line that ran against it takes the long way', `A ${A1.path} B ${B1.path}`);
  const alive = await page.evaluate(() => window.tw.world.value.save.lines.every((l) => window.tw.world.value.lineStats[l.id].status !== 'broken'));
  check(alive, 'no line is broken');

  // arrows: street zoom, light then dark
  await page.evaluate(() => { window.tw.selection.value = null; });
  await page.locator('.loading, .load-card, dialog').first().waitFor({ state: 'hidden', timeout: 5000 }).catch(() => {});
  await page.getByText(/Loading city data|Calculating demand/).first().waitFor({ state: 'hidden', timeout: 240000 }).catch(() => {});
  const shot = async (name) => {
    await page.evaluate((e) => { const w = window.tw.world.value; const ed = w.edges.find((x) => x.id === e); const a = w.nodes.get(ed.a), b = w.nodes.get(ed.b); window.tw.map.jumpTo({ center: [(a.lng + b.lng) / 2, (a.lat + b.lat) / 2], zoom: 14.6 }); window.tw.overlay.invalidate(); }, ring.e01);
    await settle(1800);
    await page.screenshot({ path: join(ARTIFACTS, name) });
  };
  await shot('t3094-arrows-light.png');
  const n1 = await page.evaluate(() => window.tw.renderer.oneWayTrack?.size ?? -1);
  check(n1 > 0, 'chevrons are laid on the one-way piece', `${n1}`);
  await page.evaluate(() => { window.tw.shade.value = 'dark'; });
  await page.waitForTimeout(1000);
  await shot('t3094-arrows-dark.png');
  await page.evaluate(() => { window.tw.shade.value = 'light'; });

  // the inspector with the switch on and the button
  await page.evaluate((e) => { window.tw.selection.value = { kind: 'track', edge: e }; }, ring.e01);
  await page.waitForTimeout(1200);
  const insp = await page.$('#inspector, .inspector');
  await (insp ?? page).screenshot({ path: join(ARTIFACTS, 't3094-inspector.png') });

  // reverse
  await page.getByRole('button', { name: 'Reverse direction' }).click();
  await dialog.waitFor({ timeout: 20000 });
  await dialog.getByRole('button', { name: 'Reverse', exact: true }).click();
  await page.waitForTimeout(1500);
  const f2 = await flag(ring.e01);
  check(f2 === 3 - f1, 'reversing flips the way', `${f1} -> ${f2}`);
  const A2 = await line(A.num), B2 = await line(B.num);
  // against the new way A must leave the piece; B is on the long way already and stays there
  check(A2.path.length === 3 && B2.path.length === 3, 'the line that now ran against it takes the long way', `A ${A2.path} B ${B2.path}`);

  // one undo each
  await edit({ op: 'undo' });
  await page.waitForTimeout(1000);
  check((await flag(ring.e01)) === f1, 'undo puts the direction back');
  await edit({ op: 'undo' });
  await page.waitForTimeout(1000);
  check((await flag(ring.e01)) === 0, 'a second undo puts both ways back');
  const A3 = await line(A.num), B3 = await line(B.num);
  check(A3.path.join() === A.path.join() && B3.path.join() === B.path.join(), 'and both lines their paths', `A ${A3.path} B ${B3.path}`);
  const bad = errors.filter((e) => !/favicon|404/.test(String(e)));
  check(bad.length === 0, 'no console errors', bad.slice(0, 3).join(' | '));
} catch (e) {
  console.log('FAIL', e?.stack ?? e);
  fail++;
} finally {
  await chrome.close?.();
}
console.log(fail ? `FAIL: ${fail}` : 'PASS: all');
process.exit(fail ? 1 : 0);
