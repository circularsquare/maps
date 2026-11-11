// T-2766: stop clicks on a big line, on the real New York save. The busiest line (most stops) has five of its stops
// clicked off and then back on, first slowly (each click waits for its world), then the same clicks quickly, 150 ms
// apart. The quick clicks all land, in order, with the same stops as the slow ones and one network built for them;
// each click's marker and stop list row change soon after it (timed, with the click's edit sent and answered); a click
// the cash cannot pay for is refused at that click, and the cash shown matches the worker's after paid clicks.
// From the maps root, on a Vite of your own:   node trainworld/checks/t2766-stopwait.mjs [dev url]
//   GAP=150 (ms between quick clicks)   MEASURE=1 (timings only, no pass or fail on them)
// About four minutes. Writes artifacts/trainworld/t2766-stopwait.json.
import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, demandFinal, importSave, launchChrome, REAL_SAVE } from './lib.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6205/';
const GAP = Number(process.env.GAP ?? 150);
const MEASURE = process.env.MEASURE === '1';
/** TL=1: print every message both ways in the quick bursts */
const TL = process.env.TL === '1';
/** a quick click's marker and list row change within this of the click, ms: typically 60-200 (the main thread's own work on
 * the new world, and now and then a slow row build in the worker), a click waiting for the last one's whole state took
 * 0.3-4 s; a guard against that coming back, the network's timing is checked exactly below */
const SHOWN_MS = 600;
/** T-3159: the clicked marker is drawn at the click, a frame or two on three cores (the stop list's row waits for the world) */
const MARKER_MS = 50;
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

const chrome = await launchChrome({ prefix: 'tw-t2766-', size: [1400, 900] });
const errors = [];
const out = { when: new Date().toISOString(), gap: GAP };
try {
  const page = await chrome.ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=2&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.tools, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) >= 100, undefined, { timeout: 300000, polling: 250 });
  await demandFinal(page, { timeout: 600000 }).catch(() => console.log('      no demand final'));
  await page.waitForTimeout(2000);

  // the busiest rail line: the most stops; five stops spread along it, never an end
  const pick = await page.evaluate(async () => {
    const tw = window.tw, w = tw.world.value, ui = await import('/src/game/ui.ts');
    const rail = w.save.lines.filter((l) => !w.lineStats[l.id]?.bus && w.lineStats[l.id]?.status === 'running' && !w.lineStats[l.id]?.loop);
    const l = rail.sort((a, b) => b.stops.length - a.stops.length)[0];
    const n = l.stops.length, at = [];
    for (let k = 1; k <= 5; k++) at.push(l.stops[Math.round((k * (n - 1)) / 6)]);
    const ids = [...new Set(at)].filter((s) => l.stops.filter((x) => x === s).length === 1);
    tw.selection.value = { kind: 'line', line: l.id };
    ui.tool.value = 'line';
    ui.lineDraft.value = { line: l.num, stops: l.stops.map(Number) };
    const nm = new Map(w.save.stations.map((s) => [s.id, s.name]));
    // the clicked stations all on screen (a marker off it is not kept up to date)
    const ll = w.save.stations.filter((s) => ids.includes(s.id)).map((s) => [s.lng, s.lat]);
    const lo = [Math.min(...ll.map((p) => p[0])), Math.min(...ll.map((p) => p[1]))], hi = [Math.max(...ll.map((p) => p[0])), Math.max(...ll.map((p) => p[1]))];
    tw.map.fitBounds([lo, hi], { padding: { top: 120, bottom: 120, left: 480, right: 120 }, animate: false });
    return { num: l.num, id: l.id, name: l.name, stops: l.stops, ids, names: ids.map((s) => nm.get(s)) };
  });
  out.line = pick;
  // its trains kept to a headway, so stops put back need trains again (bought, or refused when the cash is short)
  pick.auto = await page.evaluate(async (id) => {
    const tw = window.tw, wm = await import('/src/game/world.ts');
    if (!tw.world.value.lineStats[id].autoS) wm.setAutoTrains(id, true);
    const until = performance.now() + 30000;
    while (performance.now() < until && !tw.world.value.lineStats[id].autoS) await new Promise((r) => setTimeout(r, 50));
    return tw.world.value.lineStats[id].autoS;
  }, pick.id);
  console.log(`      line ${pick.name}: ${pick.stops.length} stops, headway ${JSON.stringify(pick.auto)}; clicking ${pick.names.join(', ')}`);
  await page.waitForTimeout(2500);

  // in the page: the clicks of `seq` (station ids), `gap` ms apart (0: each waits for its world), and per click the
  // ms from it to its edit sent, answered, its marker's and list row's change, and the world holding it
  const burst = (seq, gap) => page.evaluate(async ([seq, gap, num, TL]) => {
    const tw = window.tw, c = tw.client, wm = await import('/src/game/world.ts'), money = await import('/src/game/money.ts');
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    // demand's solve after the last edits done first: taking its result is the main thread's for a moment
    const solving = () => { const d = window.twDemand, v = d?.view?.(); return !(v?.final && !d.client.waiting); };
    for (const until = performance.now() + 120000; performance.now() < until && solving();) await sleep(100);
    await sleep(1000);
    const line = () => tw.world.value.save.lines.find((l) => l.num === num);
    const nm = new Map(tw.world.value.save.stations.map((s) => [s.id, s.name]));
    const rows = (id) => [...document.querySelectorAll('ul.stops > li[data-stop] .stop-name')].filter((e) => e.textContent.trim() === nm.get(id)).length;
    const mark = (id) => tw.tools.stops.marks.get(id)?.cls ?? '';
    const isOn = (cls) => / on( |$)/.test(cls);
    const clicks = [], posts = [], answers = [], states = [];
    const t0 = performance.now(), now = () => performance.now() - t0;
    const post = c.post, receive = c.receive;
    const tl = [];
    c.post = function (msg, tr) {
      if (msg.kind === 'edit') posts.push({ t: now(), seq: msg.seq, op: msg.op.op });
      if (TL) tl.push(`${Math.round(now())} > ${msg.kind}${msg.op ? ':' + msg.op.op : ''}${msg.q ? ':' + msg.q.what : ''}`);
      return post.call(this, msg, tr);
    };
    c.receive = function (msg) {
      if (TL) tl.push(`${Math.round(now())} < ${msg.kind}${msg.kind === 'edited' ? (msg.ok ? ' ok' : ' refused') : ''}${msg.kind === 'state' ? (msg.state.net ? ' net' : ' rows') : ''}`);
      if (msg.kind === 'edited') answers.push({ t: now(), seq: msg.seq, ok: msg.ok, early: !!msg.early, charge: msg.charge, issues: msg.issues, ms: msg.ms });
      if (msg.kind === 'state') states.push({ t: Math.round(now()), net: !!msg.state.net });
      return receive.call(this, msg);
    };
    let bad = null;
    const noticeOff = wm.notice.subscribe((n) => { if (n?.bad && n.at >= t0 && !bad) bad = { t: now(), text: n.text }; });
    const cashOf = () => money.liveMoney.peek().cash;
    // watch every task for the changes, so a frame that paints them is caught
    let watching = true;
    const watch = () => {
      for (const k of clicks) {
        if (k.mark === undefined && isOn(mark(k.id)) === k.add) k.mark = now() - k.t;
        if (k.row === undefined && (k.add ? rows(k.id) > k.rows0 : rows(k.id) < k.rows0)) k.row = now() - k.t;
        if (k.cashAt === undefined && cashOf() !== k.cash0) k.cashAt = now() - k.t;
        if (k.world === undefined && line().stops.includes(k.id) === k.add) k.world = now() - k.t;
      }
    };
    const obs = new MutationObserver(watch);
    obs.observe(document.body, { subtree: true, childList: true, attributes: true, characterData: true });
    const frame = () => { if (!watching) return; watch(); requestAnimationFrame(frame); };
    requestAnimationFrame(frame);
    for (const id of seq) {
      const add = !line().stops.includes(id);
      const k = { id, add, t: now(), rows0: rows(id), cash0: cashOf() };
      clicks.push(k);
      tw.tools.stops.activate(id, 0, 0);
      if (gap > 0) await sleep(gap);
      else {
        const until = performance.now() + 30000;
        while (performance.now() < until && (k.world === undefined || c.pending.size)) await sleep(5);
        await sleep(1200);
      }
    }
    // everything answered, the last state in, and a moment for its trips
    const until = performance.now() + 60000;
    while (performance.now() < until && (c.pending.size || clicks.some((k) => k.world === undefined))) await sleep(20);
    await sleep(3000);
    watching = false;
    obs.disconnect();
    noticeOff();
    c.post = post;
    c.receive = receive;
    // each click's edit is the k-th edit sent after the burst began
    const r = clicks.map((k, i) => {
      const p = posts[i], a = p && answers.find((x) => x.seq === p.seq);
      return { id: k.id, add: k.add, at: Math.round(k.t), sent: p ? Math.round(p.t - k.t) : null, op: p?.op, answered: a ? Math.round(a.t - k.t) : null, ok: a?.ok, early: a?.early, charge: a?.charge, issues: a?.issues, ms: a?.ms && Object.fromEntries(Object.entries(a.ms).map(([k, v]) => [k, Math.round(v)])), mark: k.mark === undefined ? null : Math.round(k.mark), row: k.row === undefined ? null : Math.round(k.row), world: k.world === undefined ? null : Math.round(k.world), cash: k.cashAt === undefined ? null : Math.round(k.cashAt) };
    });
    return { clicks: r, markers: Object.fromEntries(seq.map((id) => [id, isOn(mark(id))])), edits: posts.length, states, bad, stops: line().stops, cash: cashOf(), stateCash: money.moneyState.peek()?.cash, ...(TL ? { tl } : {}) };
  }, [seq, gap, pick.num, TL]);
  const undoAll = (n) => page.evaluate(async (n) => { for (let i = 0; i < n; i++) await window.tw.client.edit({ op: 'undo' }); await new Promise((r) => setTimeout(r, 2500)); }, n);
  const stopsNow = () => page.evaluate((num) => window.tw.world.value.save.lines.find((l) => l.num === num).stops, pick.num);
  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  const show = (label, b) => {
    console.log(`      ${label}: ${b.edits} edits sent, ${b.states.length} states, ${b.states.filter((s) => s.net).length} with a network`);
    for (const k of b.clicks) console.log(`        ${k.add ? 'add   ' : 'remove'} ${k.id}  sent ${k.sent}  answered ${k.answered}${k.early ? ' early' : ''}  marker ${k.mark}  row ${k.row}  world ${k.world}  ${k.ok ? '' : 'REFUSED ' + JSON.stringify(k.issues)}${k.charge ? ' charge ' + k.charge.toFixed(2) : ''}  cash ${k.cash}  worker ${JSON.stringify(k.ms)}`);
    if (b.tl) console.log('        ' + b.tl.filter((x) => !/ < (cash|trips)$/.test(x)).join('\n        '));
  };

  // cash to spare, and only the cars in use owned, so a stop that needs another train pays the same each time
  const flush = () => page.evaluate(() => window.tw.client.debugCash(5000, true)).then(() => page.waitForTimeout(400));

  // ---- slowly, then the same quickly
  const slowOff = await burst(pick.ids, 0);
  show('slow, off', slowOff);
  await flush();
  const slowOn = await burst(pick.ids, 0);
  show('slow, back on', slowOn);
  out.slow = { off: slowOff, on: slowOn };
  await undoAll(pick.ids.length * 2);
  check(same(await stopsNow(), pick.stops), 'undone, the line has its stops back');

  const quickOff = await burst(pick.ids, GAP);
  show(`quick (${GAP} ms), off`, quickOff);
  await flush();
  const quickOn = await burst(pick.ids, GAP);
  show(`quick (${GAP} ms), back on`, quickOn);
  out.quick = { off: quickOff, on: quickOn };
  for (const [label, q, s] of [['off', quickOff, slowOff], ['back on', quickOn, slowOn]]) {
    check(q.edits === pick.ids.length && q.clicks.every((k) => k.ok), `quick clicks ${label}: each sent one edit and was taken`, `${q.edits} edits`);
    check(q.clicks.every((k, i) => i === 0 || k.at + k.sent >= q.clicks[i - 1].at + q.clicks[i - 1].sent), `quick clicks ${label}: sent in the order clicked`);
    check(same(q.stops, s.stops), `quick clicks ${label}: the same stops as the slow clicks`, `${q.stops.length} and ${s.stops.length} stops`);
    const lastAnswer = Math.max(...q.clicks.map((k) => k.at + (k.answered ?? 0))), nets = q.states.filter((s) => s.net);
    check(nets.length === 1 && nets[0].t > lastAnswer, `quick clicks ${label}: one network built for the run, after the last click was answered`, `${nets.map((s) => s.t).join(" ")} ms; last answer ${lastAnswer} ms; ${q.states.length} states`);
    const late = q.clicks.filter((k) => k.mark === null || k.row === null || k.mark > SHOWN_MS || k.row > SHOWN_MS);
    if (!MEASURE) check(late.length === 0, `quick clicks ${label}: every marker and list row changed within ${SHOWN_MS} ms`, late.map((k) => `${k.mark}/${k.row}`).join(' '));
    const slowMarks = q.clicks.filter((k) => k.mark === null || k.mark > MARKER_MS);
    if (!MEASURE) check(slowMarks.length === 0, `quick clicks ${label}: every marker changed within ${MARKER_MS} ms of its click`, q.clicks.map((k) => k.mark).join(' '));
    else console.log(`      markers ${q.clicks.map((k) => k.mark).join(' ')}, rows ${q.clicks.map((k) => k.row).join(' ')}`);
    check(q.clicks.every((k) => q.markers[k.id] === k.add), `quick clicks ${label}: each marker ends as its stop is`, JSON.stringify(q.markers));
  }
  const paid = [...slowOn.clicks].filter((k) => k.charge > 0);
  out.paid = paid.map((k) => ({ id: k.id, charge: k.charge }));
  const sum = (b) => b.clicks.reduce((s, k) => s + (k.charge ?? 0), 0);
  check(Math.abs(sum(quickOn) - sum(slowOn)) < 1e-6, 'quick clicks pay what slow ones did', `${sum(quickOn).toFixed(3)} and ${sum(slowOn).toFixed(3)} US$M`);
  check(Math.abs(quickOn.cash - quickOn.stateCash) < 1e-6 && Math.abs(slowOn.cash - quickOn.cash) < 1e-6, 'the cash shown after the quick clicks is the worker\'s, and the slow clicks\'', `${quickOn.cash} shown, ${quickOn.stateCash} in the money state, ${slowOn.cash} slow`);
  const paidQuick = quickOn.clicks.filter((k) => k.charge > 0);
  if (!MEASURE && paidQuick.length) check(paidQuick.every((k) => k.cash !== null && k.cash < SHOWN_MS), 'a paid quick click\'s cash shown soon after it', paidQuick.map((k) => k.cash).join(' '));

  // ---- the stops put back quickly again with the cash short: the first click that buys trains is refused at that
  // click, the free ones before it taken
  const dearAt = slowOn.clicks.findIndex((k) => k.charge > 0);
  if (dearAt < 1) check(false, 'a click back on that buys trains, behind a free one, for the refusal', `at ${dearAt}`);
  else {
    await undoAll(pick.ids.length);
    await page.evaluate(() => window.tw.client.debugCash(0.001, true));
    await page.waitForTimeout(500);
    const b = await burst(pick.ids.slice(0, dearAt + 1), GAP);
    show('cash short: free clicks, then a dear one', b);
    out.refusal = b;
    const k = b.clicks[dearAt];
    check(b.clicks.slice(0, dearAt).every((x) => x.ok) && k.ok === false && k.issues?.some((x) => x.startsWith('Money:')), 'the dear click is refused for money, the free ones taken', JSON.stringify(k.issues));
    check(b.markers[k.id] === false && b.clicks.slice(0, dearAt).every((x) => b.markers[x.id]), 'the refused marker is back to a non-stop, the free ones are stops', JSON.stringify(b.markers));
    if (!MEASURE) check(b.bad && b.bad.t - k.at < 300, 'the refusal is told within 300 ms of its click', b.bad ? `${Math.round(b.bad.t - k.at)} ms` : 'no notice');
    else console.log(`      refusal told ${b.bad ? Math.round(b.bad.t - k.at) : null} ms after its click`);
  }
} catch (e) {
  console.log('FAIL', e.stack ?? e.message);
  fail++;
}
check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
out.errors = errors;
await writeFile(join(ARTIFACTS, 't2766-stopwait.json'), JSON.stringify(out, null, 1));
await chrome.close();
console.log(fail ? `T-2766 FAILED (${fail})` : 'T-2766 PASSED');
process.exit(fail ? 1 : 0);
