// T-2848: after a recovered panic on an unchanged network, fares, rides and points carry on (the rebuild reposts the
// figures; no new solve comes). T-2849: failed demand no longer marks later edits as being worked out, mid-game
// or after "Play without demand", and the 4x cap lets go. Dev server only (`debugPanic`).
//   BASE=http://localhost:6188/ node trainworld/checks/t2848-panic-figures.mjs   (about 5 minutes)
import { readFile } from 'node:fs/promises';
import { launchChrome, importSave, demandFinal, REAL_SAVE } from './lib.mjs';
const BASE = process.env.BASE ?? 'http://localhost:6188/';
const { ctx, close } = await launchChrome({ prefix: 'tw-t2848-' });
const page = await ctx.newPage();
const pageErrors = [];
page.on('pageerror', (e) => pageErrors.push(e.message));
let fails = 0;
const check = (name, ok, detail = '') => { if (!ok) fails++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`); };
const URL_ = () => `${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`;
const appUp = () => page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 240000 });

/** fares, running, rides and points so far, today's as they move */
const totals = () => page.evaluate(async () => {
  const { moneyState, liveMoney } = await import('/src/game/money.ts');
  const today = liveMoney.peek().today, days = moneyState.peek()?.ledger ?? [];
  const all = [...days.filter((d) => d.day !== today?.day), ...(today ? [today] : [])];
  const sum = (k) => all.reduce((s, d) => s + (d[k] ?? 0), 0);
  return { fares: sum('fares'), running: sum('running'), rides: sum('rides'), points: sum('points') };
});
const window8s = async () => {
  const a = await totals();
  await page.waitForTimeout(8000);
  const b = await totals();
  return Object.fromEntries(Object.keys(a).map((k) => [k, b[k] - a[k]]));
};
const pendingState = () => page.evaluate(async () => {
  const lf = await import('/src/game/lineFigures.ts'), d = await import('/src/game/demand.ts'), l = await import('/src/game/loading.ts');
  return { pending: [...lf.linesPending.peek().keys()], updating: d.demandUpdating.peek(), capped: lf.demandCapped.peek(), stopped: l.demandStopped.peek() };
});
/** the busiest kind of edit a player makes: a running line's headway 40% longer */
const editHeadway = () => page.evaluate(async () => {
  const w = window.tw.world.value;
  const l = w.save.lines.find((x) => w.lineStats[x.id]?.status === 'running');
  const secs = [l.tph.high, l.tph.medium, l.tph.low].map((n) => (n > 0 ? Math.round((3600 / n) * 1.4) : 0));
  const a = await window.tw.client.edit({ op: 'setHeadway', line: Number(l.num), seconds: secs });
  return { ok: a?.ok, issues: a?.issues, id: l.id };
});
const loadingModule = () => page.evaluate(async () => { window.__loading = await import('/src/game/loading.ts'); });
const openRealSave = async () => {
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 180000 });
};

try {
  // ---- T-2848
  await page.goto(URL_());
  await appUp();
  await page.waitForTimeout(1500);
  await openRealSave();
  await demandFinal(page, { timeout: 420000, held: true });
  await page.evaluate(() => window.tw.clock.setRunning(true));
  await page.waitForTimeout(4000); // past the first hour's partial settling
  const before = await window8s();
  check('real save earns before the panic', before.fares > 0, JSON.stringify(before));
  const solves = () => page.evaluate(() => ({ log: window.twDemand.log.length, version: window.twDemand.view()?.version }));
  const s0 = await solves();
  const game0 = await page.evaluate(() => {
    const c = window.tw.client, post = c.post.bind(c), g = c.game;
    window.__posted = [];
    c.post = (m, t) => { window.__posted.push(m.kind); return post(m, t); };
    c.debugPanic('now');
    return g;
  });
  // the rebuild of a big network takes seconds; `game` moves when its "loaded" has been handled
  await page.waitForFunction((g) => window.tw.client.game > g, game0, { timeout: 120000, polling: 100 });
  await page.waitForFunction(() => !window.tw.clock.held.peek(), undefined, { timeout: 120000 });
  await page.waitForTimeout(3000);
  const posted = await page.evaluate(() => window.__posted);
  console.log('      posted after the panic:', [...new Set(posted)].join(' '));
  for (const k of ['income', 'points', 'riders', 'transfers']) check(`rebuild reposts ${k}`, posted.includes(k));
  const after = await window8s();
  const s1 = await solves();
  check('no new solve came (the network is unchanged)', s0.log === s1.log && s0.version === s1.version, JSON.stringify([s0, s1]));
  check('fares carry on after the panic', after.fares > before.fares * 0.4, JSON.stringify(after));
  check('rides carry on after the panic', before.rides === 0 || after.rides > before.rides * 0.4, `${before.rides} -> ${after.rides}`);
  check('running costs carry on', after.running > 0, JSON.stringify(after));

  // ---- T-2849, demand stopped mid-game
  await loadingModule();
  await page.evaluate(() => window.tw.failDemand());
  await page.waitForFunction(() => window.__loading.demandStopped.peek() === 'error', undefined, { timeout: 20000, polling: 200 });
  await page.evaluate(() => window.tw.clock.setRunning(true));
  const e1 = await editHeadway();
  check('A edit accepted', e1.ok, JSON.stringify(e1));
  await page.waitForTimeout(12000);
  const a = await pendingState();
  check('A the edited line is not pending', a.pending.length === 0, JSON.stringify(a));
  check('A demand is not marked as updating', a.updating === false, JSON.stringify(a));

  // ---- T-2849, failed startup and "Play without demand"
  await page.goto(URL_());
  // the demand client must exist, or failDemand marks the startup failed and the pool carries on solving
  await page.waitForFunction(() => window.tw?.failDemand && window.twDemand?.client, undefined, { timeout: 120000, polling: 50 });
  await loadingModule();
  // once its workers have reported in: a failure before that is overwritten by their "ready" and the pool solves on
  const atFail = await page.evaluate(() => new Promise((res) => {
    const t = setInterval(() => {
      if (window.twDemand.client.status !== 'ready') return;
      clearInterval(t);
      const s = window.__loading.startup.peek();
      window.tw.failDemand();
      res(s);
    }, 10);
  }));
  check('B failed while starting up', atFail === 'loading', atFail);
  await page.waitForFunction(() => window.__loading.startup.peek() === 'failed', undefined, { timeout: 60000, polling: 200 });
  await page.waitForSelector('.loading-actions .btn', { timeout: 60000 });
  await page.locator('.loading-actions .btn').last().click();
  await page.waitForFunction(() => window.__loading.startup.peek() === 'ready', undefined, { timeout: 60000, polling: 200 });
  check('B the demand pool is failed', await page.evaluate(() => window.twDemand.client.status.startsWith('failed')));
  await page.waitForFunction(() => window.tw?.world?.value, undefined, { timeout: 180000 });
  await openRealSave();
  await page.waitForTimeout(4000);
  await page.evaluate(() => window.tw.clock.setRunning(true));
  const e2 = await editHeadway();
  check('B edit accepted', e2.ok, JSON.stringify(e2));
  await page.waitForTimeout(12000);
  const b = await pendingState();
  check('B the edited line is not pending', b.pending.length === 0, JSON.stringify(b));
  check('B demand is not marked as updating', b.updating === false, JSON.stringify(b));
  check('B the clock is not capped', b.capped === false, JSON.stringify(b));
  check('no page errors', pageErrors.length === 0, pageErrors.slice(0, 3).join(' | '));
} catch (e) {
  check('check ran', false, String(e?.stack ?? e));
} finally {
  await close();
}
console.log(fails ? `${fails} FAILED` : 'ALL PASS');
process.exit(fails ? 1 : 0);
