// T-2983: the clock worker's cost of a bus line's Auto/Manual stop switch, on the real save (about 2 minutes).
//   node trainworld/checks/t2983-bus-switch.mjs [dev url]     (from the maps root)
import { readFile } from 'node:fs/promises';
import { REAL_SAVE, collectErrors, importSave, launchChrome } from './lib.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6145/';
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const chrome = await launchChrome({ prefix: 'tw-t2983-', size: [1400, 900], deviceScaleFactor: 1 });
const page = await chrome.ctx.newPage();
const errors = [];
collectErrors(page, errors);
try {
  await page.goto(`${BASE}?debug=1&new=1&opened=all&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client && window.tw.tools, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 50, undefined, { timeout: 300000 });
  await wait(Number(process.env.WAIT_MS ?? 70000)); // demand's first solve of the real save
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  const made = await page.evaluate(async () => {
    const w = window.tw.world.value;
    const rail = w.save.lines.find((l) => l.stops.length >= 6 && !w.lineStats[l.id].bus);
    const pts = rail.stops.slice(0, 5).map((s) => w.nodes.get(s) ?? w.nodes.get(Number(s))).filter(Boolean).flatMap((n) => [n.x, n.y, 1]);
    const a = await window.tw.client.edit({ op: 'addBusLine', route: pts, name: 'Check bus', colour: '#c0392b' });
    const b = a.ok ? await window.tw.client.edit({ op: 'constructLine', line: a.line }) : a;
    return { ok: a.ok && b.ok, line: a.line };
  });
  console.log('bus line', JSON.stringify(made));
  await wait(5000);
  const ops = [{ op: 'busSpacing', spacing: 600 }, { op: 'busSpacing', spacing: 0 }, { op: 'busHand' }, { op: 'busSpacing', spacing: 0 }, { op: 'busSpacing', spacing: 600 }, { op: 'busSpacing', spacing: 0 }];
  for (const o of ops) {
    const r = await page.evaluate((o) => window.tw.client.edit(o), { ...o, line: made.line });
    await wait(3500);
    const e = await page.evaluate(() => { const l = window.tw.client.editLog; return l[l.length - 1]; });
    const w = e?.worker ?? {};
    console.log(JSON.stringify(o), 'ok', r.ok, 'worker', Object.entries(w).map(([k, v]) => k + '=' + (typeof v === 'number' ? Math.round(v * 10) / 10 : v)).join(' '));
  }
  console.log('errors', errors.slice(0, 3).join(' | '));
} finally { await chrome.close(); }
