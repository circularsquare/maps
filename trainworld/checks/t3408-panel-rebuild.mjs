// T-3408: rebuilding one built piece from the track panel, against the Select tool's bar on the same piece. On a
// new game in New York: a straight 8 km run of double track with a station at each end and one in the middle,
// built, with a line over it. The game is exported, then the M-B piece goes through four rebuilds from its panel
// (high speed, back to regular, single track, back to double), each by its buttons, the Rebuild hover and the
// confirm; then the export is loaded again and the same four go through the bar with that one piece picked.
// Checks each step's hover, quote and cash taken are the same both ways and the cash taken is the quote, the
// piece ends as asked, the line keeps running, undo is cleared as for the bar (the rebuild is final), the panel has
// the hint to the Select tool, and a blueprint piece's panel has no rebuild. Screenshots of the panel, light and
// dark: artifacts/trainworld/t3408_*.png.
// From the maps root, on a Vite of your own:
//   node trainworld/checks/t3408-panel-rebuild.mjs [dev url]
// About two minutes.
import { ARTIFACTS, collectErrors, importSave, launchChrome } from './lib.mjs';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6107/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const C = [-73.93, 40.668];

const chrome = await launchChrome({ prefix: 'tw-t3408-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
collectErrors(page, errors);

const piece = (id) => page.evaluate((id) => {
  const w = window.tw.world.value, e = w.edges.find((x) => x.id === id);
  return e && { tracks: e.tracks, hsr: !!e.hsr, cost: Math.round(e.cost), built: e.built, cash: w.save.cash, canUndo: !!w.canUndo, status: Object.values(w.lineStats)[0]?.status };
}, id);
const quote = (edge, kind, tracks) => page.evaluate(async ([edge, kind, tracks]) => {
  const { queryClock } = await import('/src/workers/clockClient.ts');
  const d = await queryClock({ what: 'rebuildQuote', edges: [edge], stations: [], level: null, kind, tracks });
  return { price: d[1], changed: d[2], refused: d[4] };
}, [edge, kind, tracks]);
const waitFor = async (fn, what) => {
  for (let k = 0; k < 60; k++) {
    if (await fn()) return true;
    await page.waitForTimeout(400);
  }
  check(false, `waited for ${what}`);
  return false;
};
const selectEdge = async (edge) => { await page.evaluate((edge) => { window.tw.selection.value = { kind: 'track', edge }; }, edge); await page.waitForTimeout(800); };
const panel = () => page.locator('#inspector:has(.piece-rebuild), .inspector:has(.piece-rebuild)').first();
const shot = async (file) => {
  const p = await page.$('#inspector, .inspector');
  await page.waitForTimeout(600);
  await (p ?? page).screenshot({ path: join(ARTIFACTS, file) });
};

// the four steps: the button to press in the panel or the bar, and the quote's kind and tracks for it
const STEPS = [
  { name: 'high speed', button: (root) => root.locator('button', { hasText: /^High speed$/ }), kind: 2, tracks: 0, want: { tracks: 2, hsr: true } },
  { name: 'regular', button: (root) => root.locator('button', { hasText: /^Regular$/ }), kind: 1, tracks: 0, want: { tracks: 2, hsr: false } },
  { name: 'single track', button: (root) => root.locator('button[aria-label="Single track"]'), kind: 0, tracks: 1, want: { tracks: 1, hsr: false } },
  { name: 'double track', button: (root) => root.locator('button[aria-label="Double track"]'), kind: 0, tracks: 2, want: { tracks: 2, hsr: false } },
];

/** One step through `root` (the panel's actions or the bar): press the choice, read the Rebuild hover, confirm. */
async function step(root, s, edge, label) {
  const before = await piece(edge);
  const q = await quote(edge, s.kind, s.tracks);
  await s.button(root()).click();
  const rebuildBtn = () => root().locator('button', { hasText: /Rebuild/ });
  await waitFor(async () => !(await rebuildBtn().isDisabled()), `${label} ${s.name}: Rebuild to be offered`);
  const tip = await rebuildBtn().getAttribute('data-tip');
  if (label === 'panel' && s.name === 'high speed') await shot('t3408_chosen_light.png');
  await rebuildBtn().click();
  await page.waitForSelector('dialog.confirmation[open]', { timeout: 5000 });
  const ask = await page.locator('dialog.confirmation #confirmation-text').innerText();
  await page.locator('dialog.confirmation .btn.danger').click();
  await waitFor(async () => { const p = await piece(edge); return p.tracks === s.want.tracks && p.hsr === s.want.hsr; }, `${label} ${s.name}: the piece to change`);
  await page.waitForTimeout(600);
  const after = await piece(edge);
  if (label === 'panel' && s.name === 'single track') await shot('t3408_single_light.png');
  const taken = before.cash - after.cash;
  check(after.tracks === s.want.tracks && after.hsr === s.want.hsr, `${label}: ${s.name} rebuilt`, JSON.stringify(after));
  check(Math.abs(taken - q.price * 1e6) < 2, `${label}: ${s.name} takes the quote`, `taken ${(taken / 1e6).toFixed(3)}M, quoted ${q.price.toFixed(3)}M`);
  check(!after.canUndo, `${label}: ${s.name} clears undo (final, as the bar's)`);
  check(after.status && after.status !== 'broken', `${label}: the line still runs after ${s.name}`, after.status);
  return { tip, ask, taken: Math.round(taken), price: q.price, cost: after.cost, changed: q.changed, refused: q.refused };
}

try {
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 180000 });
  const built = await page.evaluate(async (c) => {
    const tw = window.tw, cl = tw.client, o = tw.lngLatToPack(c[0], c[1]);
    const P = (dx, dy) => ({ x: o.x + dx, y: o.y + dy });
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p.x, n.y - p.y); if (d < bd) { bd = d; best = n.id; } } return best; };
    const free = (p) => ({ kind: 'free', x: p.x, y: p.y, level: 0 });
    const at = (n) => ({ kind: 'node', node: n });
    const res = [];
    const A = P(-4000, 0), M = P(0, 0), B = P(4000, 0);
    res.push(await cl.edit({ op: 'route', from: free(A), to: free(M), pis: [], single: false })); await pause(300);
    const nA = near(A), nM = near(M);
    res.push(await cl.edit({ op: 'route', from: at(nM), to: free(B), pis: [], single: false })); await pause(300);
    const nB = near(B);
    for (const n of [nA, nM, nB]) { res.push(await cl.edit({ op: 'addStation', at: at(n), platform: 160, name: '' })); await pause(200); }
    // a blueprint stub beside the run, for the blueprint panel's check
    res.push(await cl.edit({ op: 'route', from: free(P(-4000, 3000)), to: free(P(0, 3000)), pis: [], single: false })); await pause(300);
    const stub = tw.world.value.edges.find((e) => !e.built && Math.abs(tw.world.value.nodes.get(e.a).y - (o.y + 3000)) < 50)?.id;
    res.push(await cl.edit({ op: 'construct', edges: tw.world.value.edges.filter((e) => e.id !== stub).map((e) => e.id), stations: [nA, nM, nB] })); await pause(500);
    const line = await cl.edit({ op: 'addLine', stops: [nA, nM, nB], name: 'Panel rebuild', colour: '#7b3fb3' });
    await pause(800);
    const w = tw.world.value;
    const mb = w.edges.find((e) => (e.a === nM && e.b === nB) || (e.b === nM && e.a === nB));
    return { ok: line.ok, mb: mb?.id, stub, refused: res.filter((r) => !r.ok).map((r) => JSON.stringify(r.issues ?? r).slice(0, 140)) };
  }, C);
  check(built.ok && built.mb !== undefined && built.stub !== undefined && !built.refused.length, 'the run, its stations and its line are built, and a blueprint stub', JSON.stringify(built));
  if (!built.ok || built.mb === undefined) throw new Error('no line');
  await page.evaluate((c) => { window.tw.map.jumpTo({ center: c, zoom: 12.4 }); }, C);
  const p0 = await piece(built.mb);
  check(p0.built && p0.tracks === 2 && !p0.hsr, 'the M-B piece is built double regular track', JSON.stringify(p0));

  // the blueprint stub's panel has no rebuild
  await selectEdge(built.stub);
  await page.locator('.top-speed').first().waitFor({ timeout: 20000 });
  check((await page.locator('.piece-rebuild').count()) === 0, 'a blueprint piece\'s panel has no rebuild');

  // the save as it is now, for the bar's run
  const dl = page.waitForEvent('download', { timeout: 60000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'export' }));
  const bytes = await readFile(await (await dl).path());

  // the panel
  await selectEdge(built.mb);
  await panel().waitFor({ timeout: 20000 });
  const text = (await page.locator('#inspector, .inspector').first().innerText()).replace(/\s+/g, ' ');
  check(/Use the Select tool \(4\) to rebuild longer stretches/.test(text), 'the panel has the hint to the Select tool (4)', text.slice(0, 300));
  check((await panel().locator('button', { hasText: /Rebuild/ }).isDisabled()), 'Rebuild is greyed until a choice is made');
  await shot('t3408_panel_light.png');
  const viaPanel = [];
  for (const s of STEPS) viaPanel.push(await step(panel, s, built.mb, 'panel'));
  check(await panel().locator('button[aria-label="Double track"].on').count() === 1 && await panel().locator('button.on', { hasText: /^Regular$/ }).count() === 1, 'after the steps the panel shows the piece as double regular track again');
  await page.evaluate(async () => { (await import('/src/game/ui.ts')).shade.value = 'dark'; });
  await panel().locator('button', { hasText: /^High speed$/ }).click();
  await waitFor(async () => !(await panel().locator('button', { hasText: /Rebuild/ }).isDisabled()), 'the dark shot\'s quote');
  await shot('t3408_chosen_dark.png');
  await panel().locator('button', { hasText: /^Regular$/ }).click(); // what the piece has: drops the choice
  await page.waitForTimeout(400);
  check(await panel().locator('button', { hasText: /Rebuild/ }).isDisabled(), 'picking what the piece has drops the choice');
  await shot('t3408_panel_dark.png');
  await page.evaluate(async () => { (await import('/src/game/ui.ts')).shade.value = 'light'; });

  // the bar, from the same save
  await importSave(page, bytes);
  await waitFor(async () => { const p = await piece(built.mb); return p && p.tracks === 2 && !p.hsr && Math.abs(p.cash - p0.cash) < 1; }, 'the save to load again');
  await page.waitForTimeout(2500);
  await page.evaluate(async (edge) => {
    const ui = await import('/src/game/ui.ts'), pick = await import('/src/game/blueprintPick.ts');
    window.tw.selection.value = null;
    ui.tool.value = 'blueprint';
    pick.bpPicked.value = { edges: [edge], stations: [], built: true };
  }, built.mb);
  const bar = () => page.locator('.blueprint-bar').first();
  await bar().waitFor({ timeout: 10000 });
  const viaBar = [];
  for (const s of STEPS) viaBar.push(await step(bar, s, built.mb, 'bar'));

  for (let k = 0; k < STEPS.length; k++) {
    const a = viaPanel[k], b = viaBar[k];
    console.log(`${STEPS[k].name}: panel ${a.taken / 1e6}M (quote ${a.price.toFixed(3)}M, piece cost ${a.cost / 1e6}M) | bar ${b.taken / 1e6}M (quote ${b.price.toFixed(3)}M, piece cost ${b.cost / 1e6}M)`);
    console.log(`   hover ${JSON.stringify(a.tip)}\n   ask   ${JSON.stringify(a.ask)}`);
    check(a.tip === b.tip && a.ask === b.ask, `${STEPS[k].name}: the same hover and question from the panel and the bar`, a.tip === b.tip ? '' : `${JSON.stringify(a.tip)} vs ${JSON.stringify(b.tip)}`);
    check(a.taken === b.taken && a.cost === b.cost && a.changed === b.changed && a.refused === b.refused, `${STEPS[k].name}: the same charge and result from the panel and the bar`, `${a.taken} vs ${b.taken}`);
  }
  check(viaPanel[0].taken > 0 && viaPanel[2].taken === 0 && /Free/.test(viaPanel[2].tip ?? ''), 'high speed costs, single track is free');
  check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} catch (e) {
  await page.screenshot({ path: join(ARTIFACTS, 't3408_died.png') }).catch(() => {});
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `T-3408 FAILED: ${fail}` : 'T-3408 PASSED');
process.exit(fail ? 1 : 0);
