// T-3599: the calculating trips hover's time left against the real wait. Opens a save (the real one by default),
// waits for its final, runs the clock, then makes each edit in turn: from the edit to the edit's demand final, every
// 250 ms, the status tag's text and hover. Writes each edit's samples and the workers' `work` messages to
// artifacts/trainworld/t3599/<name>.json, and prints, at each tenth of the wait, the time left the hover said against
// the time it really had left. PASS when every hover said after its first 20% of the wait is within a third of
// the real time left, or within 20 s when under a minute was left (the hover then says "Less than a minute left").
//   node trainworld/checks/t3599-time-left.mjs [url] [save] [edit ...]
// An edit is `lng,lat[,half|off]` (the line with a stop nearest the point: its trains halved, or set to none) or
// `undo`. Default: the real save, a line near Times Square halved, then undo. From the maps root, on your own Vite
// (port 6198 by default); TW_CORES_WANT=4 gives the page 3 solving workers, as a player on a 12-thread machine
// gets. NAME=<label> names the output. About 3 minutes on the real save; 30 or more on a big Asia save.
import { mkdirSync, writeFileSync } from 'node:fs';
import { join, basename } from 'node:path';
import { ARTIFACTS, REAL_SAVE } from './lib.mjs';
import { openGame } from './lib/play.mjs';

const URL_ = process.argv[2] ?? 'http://localhost:6198/';
const SAVE = process.argv[3] && process.argv[3] !== '-' ? process.argv[3] : REAL_SAVE;
const EDITS = process.argv.slice(4).length ? process.argv.slice(4) : ['-73.986,40.758,half', 'undo'];
const NAME = process.env.NAME ?? basename(SAVE).replace(/\.save$/, '');
const OUT = join(ARTIFACTS, 't3599');
mkdirSync(OUT, { recursive: true });

const g = await openGame({ url: URL_, save: SAVE, paused: true, prefix: 'tw-t3599-' });
const page = g.page;
let failed = 0;
const runs = [];
// the workers' plans as they log them (dev server): which populations have a pass to time their units by
const plans = [];
page.on('console', (m) => { if (m.text().startsWith('T3599')) plans.push(m.text()); });
try {
  const t0 = Date.now();
  await g.waitDemand({ timeout: 3600000, polling: 1000 });
  console.log(`loaded and solved in ${Math.round((Date.now() - t0) / 1000)} s; workers ${await page.evaluate(() => window.twDemand.client.computeN)}`);
  // the clock runs fast, as in play: an edit's whole solve then caps it and keeps the tag up to the final
  await page.evaluate(() => { window.tw.clock.setRunning(true); window.tw.clock.setMultiple(10); });
  await page.waitForTimeout(5000);
  for (const spec of EDITS) {
    // the player looks at what they edit: the zone in view is solved first
    if (spec !== 'undo') {
      const [lng, lat] = spec.split(',').map(Number);
      await page.evaluate(([lng, lat]) => window.tw.map.jumpTo({ center: [lng, lat], zoom: 12 }), [lng, lat]);
      await page.waitForTimeout(4000);
    }
    const before = await page.evaluate(() => window.twDemand.log.at(-1)?.version ?? 0);
    // the line: the one with a stop nearest the point, its panel opened
    const pick = spec === 'undo' ? null : await page.evaluate((spec) => {
      const tw = window.tw, w = tw.world.value;
      const [lng, lat, how = 'half'] = spec.split(',');
      const k = Math.cos((+lat * Math.PI) / 180);
      const st = new Map(w.save.stations.map((s) => [s.id, s]));
      let best = null, bd = Infinity;
      for (const l of w.save.lines) {
        if (l.twinOf) continue;
        for (const id of l.stops) { const s = st.get(id); if (!s) continue; const d = Math.hypot((s.lng - lng) * k, s.lat - lat); if (d < bd) { bd = d; best = l; } }
      }
      const had = w.lineStats[best.id]?.trainsNeeded ?? [0, 0, 0];
      const trains = how === 'off' ? [0, 0, 0] : had.map((n) => Math.max(1, Math.floor(n / 2)));
      tw.selection.value = { kind: 'line', line: best.id };
      return { had, trains, what: `${best.name} (${best.stops.length} stops, ${Math.round(bd * 111.32 * 10) / 10} km away) trains ${had.join('/')} -> ${trains.join('/')}` };
    }, spec);
    if (pick) await page.waitForTimeout(1500);
    await page.evaluate(() => {
      const T = (window.__t3599 = { samples: [], start: performance.now() });
      T.timer = setInterval(() => {
        const el = document.querySelector('.status-tag');
        T.samples.push({ t: performance.now() - T.start, text: el?.querySelector('.demand-busy')?.textContent ?? null, tip: el?.getAttribute('data-tip') ?? null });
      }, 250);
    });
    // the player's way, so the edit counts as theirs (the clock's cap, the lines' labels): the panel's
    // trains typed in, or Ctrl+Z
    if (pick) { await g.act.setTrains(pick.trains.map((n, i) => (n === pick.had[i] ? null : n))); await page.evaluate(() => { window.tw.selection.value = null; }); }
    else await g.act.key('Control+z');
    const what = pick ? pick.what : 'undo';
    // SHOT=1: the pointer on the tag 15 s in, the hover as shown, light and dark (the first edit only)
    if (process.env.SHOT && spec === EDITS[0]) {
      await page.waitForTimeout(15000);
      const box = await page.locator('.status-tag').first().boundingBox().catch(() => null);
      if (box) {
        await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
        for (const theme of ['light', 'dark']) {
          await page.evaluate((th) => { document.documentElement.dataset.shade = th; }, theme);
          await page.waitForTimeout(1200);
          await page.screenshot({ path: join(ARTIFACTS, 't3599', `${NAME}-hover-${theme}.png`), clip: { x: Math.max(0, box.x - 40), y: Math.max(0, box.y - 20), width: 520, height: 160 } });
        }
        await page.evaluate(() => { document.documentElement.dataset.shade = 'light'; });
        await page.mouse.move(5, 5);
      }
    }
    // the edit's solve: the first sent after it; its final (or a newer one's) ends the wait
    await page.waitForFunction((b) => (window.twDemand.log.at(-1)?.version ?? 0) > b, before, { timeout: 60000, polling: 50 });
    const ver = await page.evaluate(() => window.twDemand.log.at(-1).version);
    await page.waitForFunction((ver) => { const v = window.twDemand.view(); return v?.final && v.version >= ver && !window.twDemand.client.waiting; }, ver, { timeout: 3600000, polling: 100 });
    const endMs = await page.evaluate(() => performance.now() - window.__t3599.start);
    await page.waitForTimeout(2000);
    const run = await page.evaluate((ver) => {
      const T = window.__t3599;
      clearInterval(T.timer);
      const logs = window.twDemand.log.filter((s) => s.version >= ver);
      return {
        samples: T.samples,
        solves: logs.map((s) => ({ version: s.version, sentAt: s.sentAt - T.start, barrier: !!s.barrier, clusters: s.clusters ?? null, periods: s.periods ?? null, localShownAt: s.localShownAt != null ? s.localShownAt - T.start : null, work: (s.workLog ?? []).map((x) => ({ ...x, at: x.at - T.start })) })),
      };
    }, ver);
    // the wait as the player sees it: the edit to the last sample with the tag up, or to the final
    const busy = run.samples.filter((x) => x.text && x.t <= endMs + 2000);
    const tagEnd = busy.length ? busy.at(-1).t + 250 : endMs;
    const real = Math.max(endMs, tagEnd);
    const out = { spec, what, version: ver, finalMs: endMs, tagEndMs: tagEnd, plans: plans.splice(0).filter((x) => x.startsWith(`T3599 v${ver} `)), ...run };
    for (const p of out.plans) console.log(`  ${p}`);
    runs.push(out);
    console.log(`\n${spec}: ${what}\n  final at ${(endMs / 1000).toFixed(1)} s, tag gone at ${(tagEnd / 1000).toFixed(1)} s; solves ${run.solves.map((s) => `v${s.version}${s.barrier ? ' barrier' : ''} clusters ${JSON.stringify(s.clusters)}`).join(', ')}`);
    // the hover at each tenth of the wait against the time really left
    const parse = (tip) => {
      if (!tip) return null;
      if (/Less than a minute left/.test(tip)) return { lo: 0, hi: 60000, text: 'Less than a minute left' };
      const m = tip.match(/About (\d+) min left/);
      if (m) return { mid: +m[1] * 60000, text: m[0] };
      if (/Time left not known yet/.test(tip)) return { none: true, text: 'Time left not known yet' };
      return null;
    };
    let bad = 0, said = 0;
    // (the tag "being recomputed" tells of the zone in view's figures: its time left is to that tag's end)
    const REC = /being recomputed/;
    const recEnd = (run.samples.filter((x) => REC.test(x.text ?? '')).at(-1)?.t ?? -250) + 250;
    for (let f = 0.1; f < 0.95; f += 0.1) {
      const at = f * real, s = run.samples.findLast((x) => x.t <= at);
      const left = (REC.test(s?.text ?? '') ? recEnd : real) - at, p = parse(s?.tip);
      let ok = true;
      if (p?.mid !== undefined) ok = Math.abs(p.mid - left) <= Math.max(left / 3, 20000);
      else if (p?.hi !== undefined) ok = left <= 60000 + 20000;
      if (p && !p.none) said++;
      if (f > 0.2 && p && !ok) bad++;
      console.log(`  ${Math.round(f * 100)}% (${(at / 1000).toFixed(0)} s): ${(left / 1000).toFixed(0)} s left, tag "${s?.text ?? ''}", hover "${(s?.tip ?? '').replace(/\n/g, ' | ')}"${p && !ok ? '  <- off' : ''}`);
    }
    if (bad) failed++;
    console.log(`  ${bad ? 'FAIL' : 'PASS'} ${said} of 9 samples with a time left, ${bad} off by more than a third`);
  }
} catch (e) {
  console.log('FAILED', e.message);
  failed++;
} finally {
  writeFileSync(join(OUT, `${NAME}.json`), JSON.stringify({ save: SAVE, runs }, null, 1));
  // (a big save's tile and pack fetches hit Chrome's request cap; those hid a worker's crash once)
  const errs = g.pageErrors().filter((e) => !/ERR_INSUFFICIENT_RESOURCES/.test(e));
  if (errs.length) console.log('page errors:', errs.slice(0, 3).join(' | '));
  await g.close();
  console.log(failed ? 't3599-time-left FAIL' : 't3599-time-left PASS');
  process.exit(failed ? 1 : 0);
}
