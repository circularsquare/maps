// Reads a "Download diagnostics" file (?diagnostics=1, notes/T-4206.md) and prints each edit's wait to trips calculated,
// then the demand passes, restarts, zones and the calls it counted instead of logging.
//   node trainworld/tools/diag_report.mjs <file.json> [--session N] [--events] [--extract-save out.save]
// --session N reads the Nth previous session (0 is the newest) instead of the current one; --events adds every event.
import { readFileSync, writeFileSync } from 'node:fs';

const args = process.argv.slice(2);
const flag = (name) => (args.includes(name) ? args[args.indexOf(name) + 1] : undefined);
const file = args.find((a, i) => !a.startsWith('--') && !['--extract-save', '--session'].includes(args[i - 1]));
if (!file) { console.error('usage: node trainworld/tools/diag_report.mjs <file.json> [--session N] [--events] [--extract-save out.save]'); process.exit(2); }
const report = JSON.parse(readFileSync(file, 'utf8'));
if (flag('--extract-save')) {
  if (!report.save) { console.error('no save in this file'); process.exit(1); }
  writeFileSync(flag('--extract-save'), Buffer.from(report.save.base64, 'base64'));
  console.log(`save written: ${flag('--extract-save')} (${report.save.bytes} bytes)`);
}
const sessionAt = flag('--session');
const session = sessionAt === undefined ? report.current : report.previous?.[Number(sessionAt)];
if (!session) { console.error('no such session'); process.exit(1); }
const events = [...session.events].sort((a, b) => a.at - b.at);
const t0 = session.started;
const rel = (at) => ((at - t0) / 1000).toFixed(1).padStart(7) + 's';
const sec = (ms) => (ms / 1000).toFixed(1) + ' s';
const find = (re, from = 0, pred = () => true) => events.find((e) => e.at >= from && re.test(e.event) && pred(e));
const m = session.metadata ?? {};
console.log(`session ${session.id.slice(0, 8)}  build ${m.build}  cores ${m.cores}  memory ${m.memoryGB} GB  recorded ${sec(session.updated - session.started)} (${events.length} events)`);
console.log(`save in file: ${report.save ? Math.round(report.save.bytes / 1024) + ' KB (--extract-save)' : 'none'}   exported ${new Date(report.exported).toISOString()}`);
const counts = Object.entries(session.counts ?? {});
if (counts.length) console.log('counted, not logged: ' + counts.map(([k, v]) => `${k} x${v.n} (slowest ${v.maxMs} ms)`).join(', '));

// passes
const passes = events.filter((e) => e.event === 'dm-pass').map((e) => {
  const d = e.detail, v = d.version;
  const at = (name) => find(new RegExp('^' + name + '$'), 0, (x) => x.detail.version === v)?.at;
  const stop = find(/^dm-stopped$/, 0, (x) => x.detail.version === v);
  return { ...d, estimate: at('dm-estimate'), early: at('dm-early-lines'), localEnd: at('dm-local-end'), localShown: at('dm-local-shown'), wholeStart: at('dm-whole-start'),
    held: at('dm-held'), final: at('dm-final'), stopped: stop?.at, ranMs: stop?.detail.ranMs };
});
const finals = events.filter((e) => e.event === 'dm-final');
const heldBy = (p) => {
  const q = passes.find((x) => x.version < p.version && x.held !== undefined && x.held <= p.sent && (x.final === undefined || x.final >= p.sent - 2000 || x.stopped >= p.sent - 2000));
  return q && p.sent - p.arrived > 2000 ? `; waited ${sec(p.sent - p.arrived)} behind pass ${q.version}'s ${q.final !== undefined ? 'final' : 'stop'}` : '';
};
const finalFor = (p) => finals.find((f) => f.detail.version >= p.version && f.at >= p.sent);

console.log('\nEDITS (seconds from the click; percent of the whole wait, click to calculating trips done)');
const clicks = events.filter((e) => e.source === 'page' && e.event === 'edit-click');
if (!clicks.length) console.log('  none recorded');
for (const [i, click] of clicks.entries()) {
  const op = click.detail.edit?.op, line = click.detail.edit?.line;
  const post = find(/^clock-post$/, click.at - 50);
  const answer = post && find(/^clock-answer$/, post.at, (x) => x.detail.seq === post.detail.seq);
  const applied = post && find(/^apply-end$/, post.at, (x) => x.detail.op === op);
  const next = clicks[i + 1]?.at ?? Infinity;
  const net = find(/^dm-network$/, click.at - 50);
  const pass = net?.detail.changed === false || (net && net.at - click.at > 20000) ? undefined : passes.find((p) => p.arrived >= (net?.at ?? click.at) - 50);
  const final = pass && finalFor(pass);
  const head = `#${i + 1} ${op}${line !== undefined ? ' line ' + line : ''}  at ${rel(click.at)} into the session`;
  if (!pass) {
    const why = net && net.at - click.at > 20000 ? 'no network reached demand within 20 s (a change demand does not read, or its network came with a later edit)' : net?.detail.changed === false ? `its network reached demand ${sec(net.at - click.at)} later and changed nothing demand reads (no calculating)` : 'no demand pass followed (still waiting, or the recording ends first)';
    console.log(head + '\n  ' + why);
    continue;
  }
  const end = final?.at;
  const total = end ? end - click.at : undefined;
  const pct = (a, b) => (total && a !== undefined && b !== undefined ? ` ${Math.round(((b - a) / total) * 100)}%` : '');
  const seg = (label, a, b) => (a !== undefined && b !== undefined ? `  ${label.padEnd(44)} ${((b - click.at) / 1000).toFixed(1).padStart(7)} s  (+${sec(b - a)}${pct(a, b)})` : `  ${label.padEnd(44)}       -`);
  console.log(head + (next < Infinity && next < (end ?? Infinity) ? `   (edit #${i + 2} came ${sec(next - click.at)} later, before this one finished)` : ''));
  if (post) console.log(`  clock: edit applied ${applied?.detail.ms !== undefined ? Math.round(applied.detail.ms) + ' ms' : '?'}, answered ${answer ? sec(answer.at - post.at) + ' after sent' : 'never seen'}`);
  let prev = click.at;
  const steps = [['network reached demand (clock, state, page)', pass.arrived], ['pass sent to the workers (queue / packs wait)', pass.sent],
    ['local stage ended', pass.localEnd], ['first lines shown', pass.localShown ?? pass.early], ['whole solve started', pass.wholeStart],
    pass.stopped ? ['pass stopped by a newer edit', pass.stopped] : null, ['calculating trips done' + (final && final.detail.version !== pass.version ? ` (pass ${final.detail.version})` : ''), end]].filter(Boolean);
  for (const [label, at] of steps) { console.log(seg(label, prev, at)); if (at !== undefined) prev = at; }
  console.log(`  pass ${pass.version}: ${pass.why}${pass.kind === 'edit' ? '' : ' (' + pass.kind + ')'}, ${pass.zones} zones, ${pass.markets} market packs${heldBy(pass)}`);
  const stop = end ?? Math.max(pass.sent + 1, ...[pass.localShown, pass.wholeStart, pass.stopped].filter(Boolean));
  const notable = events.filter((e) => e.at >= click.at && e.at <= stop && /^(dm-zone|dm-level|dm-worker-restart|dm-stopped|dm-held|dm-waiting|dm-market-packs)$/.test(e.event));
  for (const e of notable.slice(0, 40)) console.log('    ' + rel(e.at) + '  ' + e.event.slice(3) + ' ' + JSON.stringify(e.detail).slice(0, 150));
  if (notable.length > 40) console.log(`    ... ${notable.length - 40} more (--events)`);
  const settle = events.filter((e) => e.at >= click.at && e.at <= (end ?? stop) && /^settle/.test(e.event));
  for (const e of settle) console.log('    ' + rel(e.at) + '  ' + e.event + ' ' + JSON.stringify(e.detail));
}

console.log('\nPASSES  (arrived -> sent: waiting for packs, a final or a restart; local, shown, whole, final: seconds after sent)');
for (const p of passes) {
  const s = (v) => (v === undefined ? '    -' : ((v - p.sent) / 1000).toFixed(1).padStart(5));
  console.log(`  v${String(p.version).padEnd(4)} ${rel(p.sent)}  wait ${sec(p.sent - p.arrived).padStart(8)}  local ${s(p.localEnd)}  shown ${s(p.localShown)}  whole ${s(p.wholeStart)}  final ${s(p.final)}${p.stopped ? '  STOPPED after ' + sec(p.ranMs) : ''}  ${p.kind}, ${p.zones} zones: ${p.why}`);
}

console.log('\nCALCULATING TRIPS  (on / off)');
let on;
for (const e of events.filter((x) => x.event === 'dm-state')) {
  if (e.detail.calculating && on === undefined) on = e.at;
  else if (!e.detail.calculating && on !== undefined) { console.log(`  ${rel(on)} to ${rel(e.at)}  ${sec(e.at - on)}`); on = undefined; }
}
if (on !== undefined) console.log(`  ${rel(on)} to the end of the recording (still calculating)`);

const restarts = events.filter((e) => e.event === 'dm-worker-restart');
console.log('\nWORKER RESTARTS' + (restarts.length ? '' : '  none'));
for (const e of restarts) console.log(`  ${rel(e.at)}  worker ${e.detail.worker} ${e.detail.planned ? 'planned (memory)' : e.detail.download ? 'code download' : 'panic'}: ${e.detail.why}`);

const zones = events.filter((e) => e.event === 'dm-zone' && e.detail.ev === 'open');
console.log(`\nZONES  ${zones.length} opened (${zones.filter((e) => e.detail.bare).length} bare), ${events.filter((e) => e.event === 'dm-level').length} level changes, ${events.filter((e) => e.event === 'dm-zone' && e.detail.ev === 'closed').length} closed`);
const gaps = events.filter((e) => /heartbeat-gap|slow-phase|slow-receive|slow-query|page-error|clock-error|error/.test(e.event));
if (gaps.length) { console.log('\nSTALLS AND ERRORS'); for (const e of gaps.slice(-20)) console.log(`  ${rel(e.at)}  ${e.source} ${e.event} ${JSON.stringify(e.detail).slice(0, 160)}`); }
// T-4359: the load in hand (waitsFor: the solve it waits on, against the newest; packsIn: its packs all open) and what the
// status tag, the loading card and the zones' bars said, bars in tens of percent
const shownEv = events.filter((e) => e.event === 'dm-load' || e.event === 'dm-tag');
if (shownEv.length) {
  console.log('\nLOAD IN HAND AND WHAT THE TAG SAID' + (shownEv.length > 40 ? `  (last 40 of ${shownEv.length})` : ''));
  for (const e of shownEv.slice(-40)) {
    const { t: _t, ...d } = e.detail ?? {};
    console.log(`  ${rel(e.at)}  ${e.event === 'dm-load' ? 'load' : 'tag '}  ${JSON.stringify(d).slice(0, 220)}`);
  }
}
if (report.demandNow) {
  console.log(`\nAT DOWNLOAD  workers (WASM MB): ${(report.demandNow.workers ?? []).map((w) => w.wasm + '/' + w.need).join(' ')}  zones open ${report.demandNow.zonesOpen}, market packs ${report.demandNow.marketPacksOpen}`);
  if ('load' in report.demandNow) console.log(`  load in hand: ${JSON.stringify(report.demandNow.load)}\n  shown: ${JSON.stringify(report.demandNow.shown)}`);
}
if (args.includes('--events')) { console.log('\nALL EVENTS'); for (const e of events) console.log(`  ${rel(e.at)}  ${e.source} ${e.event} ${JSON.stringify(e.detail ?? '').slice(0, 200)}`); }