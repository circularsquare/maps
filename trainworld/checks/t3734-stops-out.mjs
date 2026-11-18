// T-3734 stage 1: stop clicks still being sent when the player saves or leaves (T-3731). On the real New York save,
// the longest rail line's stop editing gets five quick stop clicks and is ended at once, so the clicks are still
// waiting at the exit (120 line reads are queued on the clock worker as in T-3731's probe, a stand-in for a slow
// sim; on the code before T-3731 every click was lost without them too). Then:
//   1. a manual save: every click is sent, and the save only after the last of them;
//   2. Save and quit to main menu (the leave message recorded, not sent, so the page stays): the same for the leave.
// Each line's stops afterwards are its stops before minus the five clicked.
//   node trainworld/checks/t3734-stops-out.mjs [url]     (from the maps root; default localhost:6177)
// Prints PASS / FAIL lines and exits 1 on a FAIL. About a minute (most of it the save's load).
import { REAL_SAVE } from './lib.mjs';
import { openGame } from './lib/play.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6177/';
let failed = false;
const report = (name, ok, detail) => { failed ||= !ok; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };

const g = await openGame({ url: BASE, save: REAL_SAVE, saveVia: 'post', prefix: 'tw-t3734-stops-' });
try {
  const page = g.page;
  // every edit, save and leave the page sends the clock worker, in order; a leave is kept here
  await page.evaluate(() => {
    const c = window.tw.client, post = c.post.bind(c);
    window.__posts = [];
    c.post = (m, t) => {
      if (m.kind === 'edit') window.__posts.push(m.op.op);
      else if (m.kind === 'saveSlot' || m.kind === 'leave') window.__posts.push(m.kind);
      if (m.kind === 'leave') return;
      return post(m, t);
    };
  });
  /** five stop clicks on the `nth` longest rail line, stop editing ended at once, then `then` ('save' or 'leave') */
  const run = (nth, then) => page.evaluate(async ([nth, then]) => {
    const tw = window.tw, ui = await import('/src/game/ui.ts'), cc = await import('/src/workers/clockClient.ts');
    const w = tw.world.value;
    const rail = w.save.lines.filter((l) => !w.lineStats[l.id]?.bus && !w.lineStats[l.id]?.loop && !l.planStops && l.stops.length >= 12).sort((a, b) => b.stops.length - a.stops.length);
    const L = rail[nth], s = L.stops.map(Number), clicked = s.slice(2, 7);
    ui.tool.value = 'select';
    tw.selection.value = { kind: 'line', line: L.id };
    ui.tool.value = 'line';
    ui.lineDraft.value = { line: L.num, stops: s };
    await new Promise((r) => setTimeout(r, 500));
    const from = window.__posts.length;
    for (let k = 0; k < 120; k++) void tw.client.query({ what: 'stopTiers', line: w.save.lines[k % w.save.lines.length].num });
    for (const st of clicked) tw.tools.toggleStop(st);
    tw.tools.exit();
    const atExit = window.__posts.slice(from).filter((p) => p === 'setStops').length;
    if (then === 'save') void Promise.resolve(cc.manageSaveSlot('save')).catch(() => {});
    else cc.leaveToMenu(() => {});
    const t0 = performance.now();
    while (!window.__posts.slice(from).includes(then === 'save' ? 'saveSlot' : 'leave') && performance.now() - t0 < 60000) await new Promise((r) => setTimeout(r, 100));
    // the stops once the last edit sent is answered
    const want = s.filter((x) => !clicked.includes(x)).join();
    while (tw.world.value.save.lines.find((x) => x.num === L.num).stops.map(Number).join() !== want && performance.now() - t0 < 60000) await new Promise((r) => setTimeout(r, 100));
    const posts = window.__posts.slice(from);
    return { line: L.num, clicks: clicked.length, atExit, posts, stops: tw.world.value.save.lines.find((x) => x.num === L.num).stops.length, want: s.length - clicked.length, ms: Math.round(performance.now() - t0) };
  }, [nth, then]);

  for (const [nth, then] of [[0, 'save'], [1, 'leave']]) {
    const r = await run(nth, then);
    const k = r.posts.indexOf(then === 'save' ? 'saveSlot' : 'leave');
    const before = r.posts.slice(0, k < 0 ? r.posts.length : k).filter((p) => p === 'setStops').length;
    report(`${then}: every stop click is sent, and the ${then} after the last`, r.atExit < r.clicks && k >= 0 && before === r.clicks && r.stops === r.want,
      `(line ${r.line}: ${r.clicks} clicks, ${r.atExit} sent at the exit; sent in order ${r.posts.join(' ')}; stops ${r.stops}, want ${r.want}; ${r.ms} ms)`);
  }
  const errs = g.pageErrors().filter((e) => !/WebGL|GPU/i.test(e));
  report('no page errors', errs.length === 0, errs.slice(0, 3).join(' | '));
} finally {
  await g.close();
}
process.exit(failed ? 1 : 0);
