// T-2411, T-2420: the Profit track display on the real save. On the weekday, each rail line's money placed on
// its edges (game/stretchMoney.ts) sums to its fares as the worker is paid them (`incomeByDay`) and to its
// upkeep share (`line_upkeep`); the track stretches sum to the network's; the map draws the view and the
// track's hover gives the stretch's profit, per km, fares, running trains and upkeep.
//   node trainworld/checks/t2411-profit-map.mjs [url]     (from the maps root; default localhost:6195)
// Prints PASS or FAIL lines and exits 1 on failure; pictures in artifacts/trainworld/profit/. A few minutes
// (the real save's demand solve, DEMAND_WORKERS workers, 2 by default).
import { mkdirSync, readFileSync } from 'node:fs';
import { ARTIFACTS, REAL_SAVE, collectErrors, demandFinal, importSave, launchChrome } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6195/';
const WORKERS = process.env.DEMAND_WORKERS ?? 2;
const OUT = `${ARTIFACTS}profit/`;
mkdirSync(OUT, { recursive: true });
const { ctx, close } = await launchChrome({ prefix: 'tw-t2411-', size: [1400, 900] });
const results = [];
const report = (name, ok, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
const errors = [];

try {
  const page = await ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=${WORKERS}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.moneyState?.value, undefined, { timeout: 180000 });
  await importSave(page, readFileSync(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 180000 });
  await demandFinal(page, { timeout: 900000 });
  const sums = await page.evaluate(async () => {
    const { setDisplay } = await import('/src/game/ui.ts');
    const { stretchMoney } = await import('/src/game/stretchMoney.ts');
    const { incomeByDay } = await import('/src/game/money.ts');
    const { PERIODS } = await import('/src/game/periods.ts');
    setDisplay('dayShown', 'weekday');
    setDisplay('trackColour', 'profit');
    const sm = stretchMoney.value;
    if (!sm) return null;
    const w = window.tw.world.value, inc = incomeByDay.value[0];
    const placed = new Map();
    for (const e of sm.edges.values()) for (const [id, p] of e.lines) {
      const q = placed.get(id) ?? { fares: 0, running: 0, upkeep: 0 };
      q.fares += p.fares; q.running += p.running; q.upkeep += p.upkeep;
      placed.set(id, q);
    }
    let lines = 0, worstFare = 0, worstUpkeep = 0, lineProfit = 0;
    for (const l of w.save.lines) {
      const st = w.lineStats[l.id];
      if (st?.status !== 'running' || st.bus) continue;
      const li = inc.lines.get(l.id), p = placed.get(l.id) ?? { fares: 0, running: 0, upkeep: 0 };
      const fares = li ? li.perHour.reduce((a, v, k) => a + v * (PERIODS[k].to - PERIODS[k].from), 0) * 1e6 : 0;
      worstFare = Math.max(worstFare, Math.abs(p.fares - fares) / Math.max(1, fares));
      worstUpkeep = Math.max(worstUpkeep, Math.abs(p.upkeep - st.upkeep));
      lineProfit += p.fares - p.running - p.upkeep;
      lines++;
    }
    let stretchProfit = 0, bare = 0, losing = 0, n = 0;
    for (const s of new Set(sm.stretchOf.values())) {
      stretchProfit += s.fares - s.running - s.upkeep;
      if (s.bare) bare += s.upkeep;
      if (s.fares - s.running - s.upkeep < 0) losing++;
      n++;
    }
    return { lines, worstFare, worstUpkeep, lineProfit, stretchProfit, bare, losing, stretches: n };
  });
  report('stretch money worked out on the weekday', !!sums, JSON.stringify(sums));
  if (sums) {
    report(`each line's edges sum to its weekday fares (${sums.lines} lines)`, sums.worstFare < 1e-9, `worst ${sums.worstFare.toExponential(2)} of its fares`);
    report("each line's edges sum to its upkeep share", sums.worstUpkeep < 1, `worst $${sums.worstUpkeep.toFixed(4)}`);
    report('the stretches sum to the network (lines less bare track)', Math.abs(sums.stretchProfit - (sums.lineProfit - sums.bare)) < 1, `stretches $${(sums.stretchProfit / 1e6).toFixed(1)}M, lines $${(sums.lineProfit / 1e6).toFixed(1)}M, bare track $${(sums.bare / 1e6).toFixed(2)}M; ${sums.losing} of ${sums.stretches} stretches losing`);
  }
  // the view drawn, the region and Manhattan
  await page.evaluate(() => window.tw.map.jumpTo({ center: [-73.5, 40.95], zoom: 7.6 }));
  await page.waitForTimeout(2500);
  await page.screenshot({ path: `${OUT}profit-region.png` });
  // the stretch on 14 St, L between 6 Av and Union Sq: hover it
  const at = await page.evaluate(async () => {
    const w = window.tw.world.value;
    const st = (name) => w.save.stations.find((s) => s.name === name);
    const a = st('14 St/6 Av'), b = st('14 St-Union Sq');
    if (!a || !b) return null;
    const lng = (a.lng + b.lng) / 2, lat = (a.lat + b.lat) / 2;
    window.tw.map.jumpTo({ center: [lng, lat], zoom: 12 });
    return { lng, lat };
  });
  await page.waitForTimeout(2500);
  await page.screenshot({ path: `${OUT}profit-manhattan.png` });
  // hover a few running stretches 1-3 km long, close in, at points along them, until one gives the tag
  let tag = '';
  if (at) {
    await page.evaluate(async () => (await import('/src/game/ui.ts')).setDisplay('trains', false));
    const spots = await page.evaluate(() => {
      const w = window.tw.world.value;
      return w.edges.filter((e) => e.built && e.lines.length && e.lengthM > 1000 && e.lengthM < 3000).slice(0, 6).map((e) => {
        const a = w.nodes.get(e.a), b = w.nodes.get(e.b);
        return [0.4, 0.5, 0.6].map((f) => [a.lng + (b.lng - a.lng) * f, a.lat + (b.lat - a.lat) * f]);
      }).flat();
    });
    for (const [lng, lat] of spots) {
      await page.mouse.move(5, 300);
      await page.evaluate(([lng, lat]) => window.tw.map.jumpTo({ center: [lng, lat], zoom: 14 }), [lng, lat]);
      await page.waitForTimeout(700);
      // the tag's own function (map/trackHover.ts) round the point, then the pointer where it answers
      const hit = await page.evaluate(async ([lng, lat]) => {
        const { trackTip } = await import('/src/map/trackHover.ts');
        const ov = window.tw.overlay, p = window.tw.map.project([lng, lat]);
        for (let r = 0; r <= 12; r += 2) for (const [dx, dy] of [[0, 0], [r, 0], [-r, 0], [0, r], [0, -r]]) {
          const rows = trackTip(ov, p.x + dx, p.y + dy, 14);
          const text = (rows ?? []).map((x) => (typeof x === 'string' ? x : x.textContent)).join('\n');
          if (/^Profit/.test(text)) {
            const box = window.tw.map.getContainer().getBoundingClientRect();
            return { text, x: box.left + p.x + dx, y: box.top + p.y + dy };
          }
        }
        return null;
      }, [lng, lat]);
      if (!hit) continue;
      tag = hit.text;
      await page.mouse.move(hit.x, hit.y);
      await page.waitForTimeout(500);
      break;
    }
    await page.screenshot({ path: `${OUT}profit-hover.png` });
  }
  report('the track hover gives profit, per km, fares, running trains and upkeep', /Profit −?\$[\d.]+[kMB]? per weekday/.test(tag) && /per km/.test(tag) && /Fares \$/.test(tag) && /Running trains −\$/.test(tag) && /Upkeep −\$/.test(tag), JSON.stringify(tag));
  const pressed = await page.$$eval('.demand-selector button[aria-pressed="true"]', (bs) => bs.map((b) => b.textContent));
  report('the Profit button is on', pressed.includes('Profit'), JSON.stringify(pressed));
} catch (e) {
  report('ran to the end', false, e.message);
} finally {
  report('no console errors', errors.length === 0, errors.slice(0, 3).join(' | '));
  await close();
}
console.log(results.every(Boolean) ? 'T-2411 PASSED' : 'T-2411 FAILED');
process.exit(results.length && results.every(Boolean) ? 0 : 1);
