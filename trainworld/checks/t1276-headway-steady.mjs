// T-1276: stepping a line's headway keeps its schedule steady. The real save, its busiest line on
// headways, the clock running; the Medium headway stepped up four times, then down four, as fast as a
// player clicks, then a headway typed. Every DOM change in the inspector is sampled: the schedule's
// rows keep their order (High, Medium, Low), no row, cell or the panel's title ever blanks, and a
// level's train count moves only one way while its headway is stepped one way (no back and forth).
//   node trainworld/checks/t1276-headway-steady.mjs [url]     (default http://localhost:6183/)
// PLAYWRIGHT_MODULE as smoke.mjs. Output: ../artifacts/trainworld/t1276/check.json.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { launchChrome, collectErrors, importSave, demandFinal, REAL_SAVE, ARTIFACTS } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6183/';
const OUT = join(ARTIFACTS, 't1276');
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);

await mkdir(OUT, { recursive: true });
const { ctx, close } = await launchChrome({ prefix: 'tw-t1276-', size: [1400, 900] });
const page = await ctx.newPage();
const errors = [];
collectErrors(page, errors);
const out = { base: BASE, errors, problems: [] };
let failed = false;
try {
  await page.goto(`${BASE}?debug=1&new=1&opened=all&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.waitForTimeout(1000);
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await demandFinal(page);
  log('demand final');
  const pick = await page.evaluate(() => {
    const tw = window.tw, w = tw.world.value;
    const lines = w.save.lines.filter((l) => { const s = w.lineStats[l.id]; return s && s.status === 'running' && !s.loop && l.stops.length > 2; });
    lines.sort((a, b) => w.lineStats[b.id].trainsNeeded[1] - w.lineStats[a.id].trainsNeeded[1]);
    const l = lines[0];
    tw.select({ kind: 'line', line: l.id });
    tw.clock.setSpeed(1);
    tw.clock.setRunning(true);
    return { id: l.id, num: l.num, name: l.name, trains: w.lineStats[l.id].trainsNeeded, auto: !!w.lineStats[l.id].autoS };
  });
  out.line = pick;
  log('line', JSON.stringify(pick));
  await page.waitForSelector('#inspector table.sched');
  if (!pick.auto) {
    await page.click('#inspector .h .group button:has-text("Headway")');
    await page.waitForSelector('#inspector table.sched [aria-label="More headway"]');
  }
  await page.waitForTimeout(1500);
  // every change to the inspector, and every frame, read back as the schedule's rows
  await page.evaluate(() => {
    const ins = document.querySelector('#inspector');
    const read = (why) => {
      const rows = [...ins.querySelectorAll('table.sched tr')].slice(1).map((tr) => {
        const td = tr.querySelectorAll('td');
        // the headway cell: the stepper's value (a typing box reads its own value), and any muted
        // running headway left of it
        const v = td[2]?.querySelector('.v');
        return [td[0]?.textContent ?? '', td[1]?.textContent ?? '', (v?.tagName === 'INPUT' ? 'typing ' + v.value : v?.textContent ?? td[2]?.textContent ?? ''), td[2]?.querySelector('.hw-real')?.textContent ?? ''];
      });
      return { t: Math.round(performance.now()), why, title: ins.querySelector('.title')?.textContent ?? '', kids: ins.childElementCount, rows };
    };
    window.__samples = [read('start')];
    const obs = new MutationObserver(() => { if (window.__samples) window.__samples.push(read('dom')); else obs.disconnect(); });
    obs.observe(ins, { subtree: true, childList: true, characterData: true, attributes: true });
    const frame = () => { if (window.__samples) { window.__samples.push(read('frame')); requestAnimationFrame(frame); } };
    requestAnimationFrame(frame);
  });
  // no edit in flight for a second
  const settle = async () => {
    await page.waitForTimeout(300);
    await page.evaluate(() => { window.__busyAt = performance.now(); });
    await page.waitForFunction(() => {
      if (window.tw.client.pending.size) window.__busyAt = performance.now();
      return performance.now() - window.__busyAt > 1000;
    }, undefined, { timeout: 120000, polling: 100 });
  };
  const step = async (dir, n, gap) => {
    for (let i = 0; i < n; i++) {
      await page.evaluate((mark) => window.__samples.push({ mark }), `${dir} ${i}`);
      await page.click(`#inspector table.sched tr:nth-child(3) [aria-label="${dir} headway"]`);
      await page.waitForTimeout(gap);
    }
    await settle();
  };
  await step('More', 4, 120);
  await step('Less', 4, 120);
  await step('More', 3, 600);
  await step('Less', 3, 600);
  // a typed headway
  await page.evaluate(() => window.__samples.push({ mark: 'typed' }));
  await page.click('#inspector table.sched tr:nth-child(3) [aria-label="Type headway"]');
  await page.keyboard.type('7');
  await page.keyboard.press('Enter');
  await settle();
  await page.screenshot({ path: join(OUT, 'after.png') });
  const samples = await page.evaluate(() => { const s = window.__samples; window.__samples = null; return s; });
  out.end = await page.evaluate((id) => { const st = window.tw.world.value.lineStats[id]; return { autoS: st.autoS, trains: st.trainsNeeded, editMs: window.tw.client.editLog?.slice(-6).map((e) => Math.round(e.worker?.total ?? -1)) }; }, pick.id);
  log('end', JSON.stringify(out.end));
  out.samples = samples.length;
  // the checks
  const order = ['High', 'Medium', 'Low'];
  let dir = 0, run = null;
  const seen = new Map(), byHeadway = new Map();
  for (const s of samples) {
    if (s.mark) { dir = s.mark.startsWith('More') ? -1 : s.mark.startsWith('Less') ? 1 : 0; if (/ 0$|typed/.test(s.mark)) run = null; continue; }
    const where = `${s.why} at ${s.t} ms`;
    if (!s.title.trim()) out.problems.push(`panel title blank (${where})`);
    if (s.rows.length !== 3) { out.problems.push(`${s.rows.length} schedule rows (${where})`); continue; }
    s.rows.forEach((r, i) => {
      if (!r[0].includes(order[i])) out.problems.push(`row ${i} reads "${r[0]}" (${where})`);
      if (!r[1].trim() || !r[2].trim()) out.problems.push(`row ${order[i]} has a blank cell: ${JSON.stringify(r)} (${where})`);
    });
    // the Medium count while the headway steps one way: never back
    const n = Number(s.rows[1][1].replace(/[^0-9]/g, ''));
    if (dir !== 0 && Number.isFinite(n)) {
      if (run && run.dir === dir && Math.sign(n - run.n) === -dir && n !== run.n) out.problems.push(`Medium count went back ${run.n} -> ${n} while stepping ${dir < 0 ? 'up' : 'down'} (${where})`);
      run = { dir, n };
    }
    // one headway shows one count and running headway throughout, whatever is in flight; the count
    // may settle by one train when the edit lands (the round trip's waiting at busy track moves
    // with the trains)
    const m = s.rows[1], hw = m[2].trim();
    if (hw && !/typing/.test(hw)) {
      const was = byHeadway.get(hw), n = Number(m[1]);
      if (was && (Math.abs(was.n - n) > 1 || was.real !== m[3]) && !out.problems.some((p) => p.startsWith(`Medium at ${hw}`))) out.problems.push(`Medium at ${hw} showed "${was.n} ${was.real}" and later "${n} ${m[3]}" (${where})`);
      byHeadway.set(hw, { n, real: m[3] });
    }
    const key = s.rows.map((r) => r.join('|')).join(' / ');
    seen.set(key, (seen.get(key) ?? 0) + 1);
  }
  out.states = [...seen.keys()].slice(0, 80);
  failed = out.problems.length > 0;
  log('samples', samples.length, 'distinct schedules', seen.size, 'problems', out.problems.length);
  for (const p of out.problems.slice(0, 15)) log(' ', p);
} catch (e) {
  failed = true;
  out.error = String(e?.stack ?? e);
  log('error', out.error);
} finally {
  await close();
}
await writeFile(join(OUT, 'check.json'), JSON.stringify(out, null, 1));
if (errors.length) { failed = true; log('page errors', errors.slice(0, 5)); }
console.log(`${failed ? 'FAIL' : 'PASS'}  t1276 stepping a headway keeps the schedule steady`);
process.exit(failed ? 1 : 0);
