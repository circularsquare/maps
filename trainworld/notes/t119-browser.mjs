// T-119: how long the demand solve and the commuter bubbles take on the real NYC save, then with
// one Philadelphia line added. Needs `python serve.py` at the maps root (localhost:8800/trainworld/)
// and playwright. Run from the maps root:
//   LABEL=after WORKERS=3 node trainworld/notes/t119-browser.mjs
// Isolated browser profile; writes trainworld/notes/T-119-browser-<LABEL>.json.
import { readFile, writeFile, mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const label = process.env.LABEL ?? 'run';
const workers = process.env.WORKERS ? `&demandWorkers=${process.env.WORKERS}` : '';
const profile = await mkdtemp(join(tmpdir(), 't119-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1300, height: 900 } });
try {
  const page = await ctx.newPage(), errors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto(`http://localhost:8800/trainworld/?debug=1&new=1&paused=1${workers}`);
  await page.waitForFunction(() => window.tw?.world.value && window.twDemand?.client && window.twViews);
  await page.waitForFunction(() => window.twDemand.view()?.round === 3, undefined, { timeout: 120000 });
  // bubbles on before anything is imported, so the chain is as a player's would be
  await page.evaluate(() => {
    window.tw.display.value = { ...window.tw.display.value, commuters: true, commuterEnd: 'home' };
    window.__tl = [];
    window.__long = [];
    try { new PerformanceObserver(l => { for (const e of l.getEntries()) window.__long.push({ t: e.startTime, ms: e.duration }); }).observe({ entryTypes: ['longtask'] }); } catch {}
    let last = '';
    setInterval(() => {
      const v = window.twDemand.view(), b = window.twViews.stats.commuters.builds, k = `${v?.version}:${v?.round}:${b}`;
      if (k !== last) { last = k; window.__tl.push({ t: performance.now(), version: v?.version, round: v?.round, builds: b }); }
    }, 5);
  });
  await page.waitForTimeout(3000);
  const bytes = await readFile('trainworld/T-007-real-nyc.save');
  const settle = (minLines, afterVersion) => page.waitForFunction(([m, v]) => {
    const d = window.twDemand.view();
    return d && d.round === 3 && d.lines.size >= m && d.version > v;
  }, [minLines, afterVersion], { timeout: 240000 });
  const measure = (name) => page.evaluate((name) => {
    const t = window.twDemand, s = t.log.at(-1), tl = window.__tl.filter(e => e.t >= s.sentAt);
    const at = (r) => { const e = tl.find(e => e.version === s.version && e.round === r); return e ? e.t - s.sentAt : null; };
    const firstBuild = tl.find(e => e.builds > (window.__tl.filter(x => x.t < s.sentAt).at(-1)?.builds ?? 0));
    const lastBuild = tl.at(-1);
    return {
      name, version: s.version, workers: t.start.workers.length,
      arrivedToSentMs: s.sentAt - s.arrivedAt,
      publishMs: [0, 1, 2, 3].map(at),
      firstEstimateMs: s.firstAt - s.sentAt, refinedMs: s.refinedAt - s.sentAt,
      firstBubblesMs: firstBuild ? firstBuild.t - s.sentAt : null,
      lastBubblesMs: lastBuild.builds > 0 ? lastBuild.t - s.sentAt : null,
      bubbleTimeline: tl.map(e => ({ ms: Math.round(e.t - s.sentAt), v: e.version, r: e.round, b: e.builds })),
      setupMs: s.setupMs, periodMs: s.periodMs, memMB: s.memBytes.map(b => Math.round(b / 1e6)),
      lines: t.view().lines.size,
      longTasks: window.__long.filter(e => e.t >= s.sentAt).map(e => [Math.round(e.t - s.sentAt), Math.round(e.ms)]),
    };
  }, name);
  const v0 = await page.evaluate(() => window.twDemand.view().version);
  await page.evaluate(data => { const bytes = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes }, [bytes]); }, Array.from(bytes));
  await settle(100, v0);
  await page.waitForTimeout(2500);
  const legacy = await measure('real NYC network');
  const v1 = await page.evaluate(() => window.twDemand.view().version);
  const fixture = await page.evaluate(async () => {
    const t = window.tw, run = async op => { const a = await t.client.edit(op); if (!a.ok) throw Error(JSON.stringify({ op, a })); return a; };
    const R = 6371008.8, rad = Math.PI / 180, x = R * (-75.1636 + 73.985) * rad * Math.cos(40.758 * rad), y = R * (39.9526 - 40.758) * rad;
    await run({ op: 'route', from: { kind: 'free', x, y: y - 2000, level: -2 }, to: { kind: 'free', x, y: y + 2000, level: -2 }, pis: [], single: false });
    const edge = t.world.value.edges.at(-1);
    await run({ op: 'addStation', at: { kind: 'node', node: edge.a }, platform: 200, name: 'Philadelphia south' });
    await run({ op: 'addStation', at: { kind: 'node', node: edge.b }, platform: 200, name: 'Philadelphia north' });
    const line = (await run({ op: 'addLine', stops: [edge.a, edge.b], name: 'Philadelphia check', colour: '#17a488' })).line;
    await run({ op: 'setTrainCount', line, trains: [4, 4, 2] }); await run({ op: 'constructAll' }); return { line };
  });
  await settle(110, v1);
  await page.waitForTimeout(2500);
  const combined = await measure('real NYC + Philadelphia line');
  // the figures a result comparison needs: rail trips by period and line loads
  const summary = await page.evaluate(() => {
    const d = window.twDemand.view(), c = window.twDemand.cities();
    const lines = [...d.lines].map(([id, v]) => [id, Math.round(v.ridersPerDay), v.fullestPct]);
    return { split: d.split, tripsPerDay: d.tripsPerDay, rail: d.railTripsPerDay, byPeriod: d.trainShareByPeriod, cities: [...c].map(([id, v]) => [id, v.railTripsPerDay, v.split]), intercity: window.twDemand.intercity(), lines };
  });
  const start = await page.evaluate(() => JSON.parse(JSON.stringify(window.twDemand.start)));
  const result = { label, workers: combined.workers, start, legacy, combined, summary, errors };
  console.log(JSON.stringify({ label, legacy: { ...legacy, bubbleTimeline: undefined, periodMs: undefined }, combined: { ...combined, bubbleTimeline: undefined, periodMs: undefined } }, null, 1));
  await writeFile(`trainworld/notes/T-119-browser-${label}.json`, JSON.stringify(result, null, 1));
  if (errors.length) console.log('page errors', errors);
} finally { await ctx.close(); }
