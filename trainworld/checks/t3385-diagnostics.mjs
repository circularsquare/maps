// Test the recorder independently of map/demand downloads. Deliberately block both event loops,
// then reload: the previous session must still identify the action and which heartbeat stopped.
// node trainworld/checks/t3385-diagnostics.mjs http://localhost:6117/
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { ARTIFACTS, launchChrome } from './lib.mjs';

const base = process.argv[2] ?? 'http://localhost:6117/';
const chrome = await launchChrome({ prefix: 'tw-t3385-' });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.route('**/diagnostic-check*', (route) => route.fulfill({ contentType: 'text/html', body: '<!doctype html><title>Diagnostics check</title><body></body>' }));
  const open = async (query) => {
    await page.goto(`${base}diagnostic-check${query}`);
    await page.evaluate(() => import('/src/game/freezeTrace.ts'));
  };
  const report = async () => {
    const downloading = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Download diagnostics' }).click();
    const file = await downloading;
    return JSON.parse(await readFile(await file.path(), 'utf8'));
  };
  await open('');
  assert.equal(await page.getByRole('button', { name: 'Download diagnostics' }).count(), 0, 'disabled by default');
  await open('?diagnostics=1');
  await page.getByText('Diagnostics recording locally', { exact: true }).waitFor();
  await page.evaluate(async () => {
    const mod = await import('/src/game/freezeTrace.ts');
    const url = URL.createObjectURL(new Blob([`
      let port;
      onmessage = (e) => {
        if (e.data.kind === 'freezeTrace') {
          port = e.data.port;
          port.postMessage({ event: 'pulse' });
          setInterval(() => port.postMessage({ event: 'pulse' }), 1000);
        } else {
          port.postMessage({ event: 'apply-start', detail: { op: 'dropStop' } });
          const until = performance.now() + 7000;
          while (performance.now() < until) {}
          port.postMessage({ event: 'apply-end' });
        }
      };
    `], { type: 'text/javascript' }));
    window.testClock = new Worker(url);
    mod.attachClockTrace(window.testClock);
    URL.revokeObjectURL(url);
  });
  await page.waitForTimeout(1200);
  await page.evaluate(async () => {
    const mod = await import('/src/game/freezeTrace.ts');
    mod.trace('edit-click', { edit: mod.editTrace({ op: 'dropStop', line: 42, i: 3 }) });
    window.testClock.postMessage('block');
  });
  await page.waitForTimeout(8200);
  let r = await report();
  assert(r.current.events.some((e) => e.source === 'clock' && e.event === 'heartbeat-gap'), 'clock stall recorded');
  assert(!r.current.events.some((e) => e.source === 'page' && e.event === 'heartbeat-gap'), 'page remained responsive');
  await page.evaluate(async () => {
    const mod = await import('/src/game/freezeTrace.ts');
    mod.trace('page-test-start');
    mod.tracePhase('page-block-test', () => {
      const until = performance.now() + 7000;
      while (performance.now() < until) {}
    });
  });
  await page.waitForTimeout(1200);
  r = await report();
  assert(r.current.events.some((e) => e.source === 'page' && e.event === 'heartbeat-gap'), 'page stall recorded independently');
  assert(r.current.events.some((e) => e.event === 'slow-phase' && e.detail.name === 'page-block-test' && e.detail.ms >= 7000), 'slow phase identified');
  assert.deepEqual(r.current.activePhases, {}, 'completed phases cleared');
  const sessionId = r.current.id;
  await open('');
  await page.getByText('Diagnostics recording locally', { exact: true }).waitFor();
  r = await report();
  const before = r.previous.find((s) => s.id === sessionId);
  assert(before?.events.some((e) => e.event === 'edit-click' && e.detail.edit.line === 42), 'action survived reload');
  assert(before.events.some((e) => e.source === 'page' && e.event === 'heartbeat-gap'), 'stall survived reload');
  await page.evaluate(async () => {
    const mod = await import('/src/game/freezeTrace.ts');
    mod.trace('edit-click', { edit: { op: 'bounded', line: 7 } });
    for (let i = 0; i < 5000; i++) mod.trace('bounded-test', { i });
    for (let i = 0; i < 5000; i++) mod.trace('dm-bounded-test', { i, t: Date.now() - 5000 + i });
    for (let i = 0; i < 300; i++) { mod.trace('receive-start', { kind: 'trips' }); mod.trace('receive-end', { kind: 'trips', ms: 1 }); }
  });
  r = await report();
  const ofKind = (re) => r.current.events.filter((e) => re.test(e.event)).length;
  assert.equal(ofKind(/^bounded-test$/), 600, 'bounded log (other events)');
  assert.equal(ofKind(/^dm-bounded-test$/), 4000, 'bounded log (demand events, kept by their own time)');
  assert(r.current.events.some((e) => e.event === 'edit-click'), 'busy kinds do not push out edits');
  assert.equal(r.current.counts['receive:trips'].n, 300, 'frequent calls are counted');
  assert.equal(ofKind(/^receive-/), 0, 'and not logged one by one');
  assert(r.current.metadata.browser && r.current.metadata.build, 'metadata survives log rotation');
  assert.equal(typeof r.current.pageHealth.visible, 'boolean', 'last page health retained');
  await page.screenshot({ path: `${ARTIFACTS}/t3385-diagnostics.png` });
  await page.getByRole('button', { name: 'Stop recording' }).click();
  await open('');
  assert.equal(await page.getByRole('button', { name: 'Download diagnostics' }).count(), 0, 'stop persists');
  if (process.argv.includes('--game')) {
    await page.goto(`${base}?new=1&debug=1&diagnostics=1&paused=1&demandWorkers=1`);
    await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client?.running, undefined, { timeout: 180000 });
    await page.evaluate(async () => {
      const client = window.tw.client;
      const a = await client.edit({ op: 'addLine', stops: [], name: 'Diagnostics check', colour: '#4488aa' });
      if (!a.ok || a.line === undefined) throw new Error('test line refused');
      await client.edit({ op: 'setStops', line: a.line, stops: [] }, 'rows');
    });
    const game = await report();
    assert(game.current.events.some((e) => e.source === 'clock' && e.event === 'apply-start' && e.detail.op === 'setStops'), 'real clock checkpoint');
    assert(game.current.events.some((e) => e.source === 'clock' && e.event === 'edited'), 'real clock answer checkpoint');
    assert(game.current.events.some((e) => e.source === 'page' && e.event === 'clock-answer'), 'real page answer checkpoint');
    assert(game.current.clockHealth.wasmMB > 0, 'real track WASM memory captured');
    await writeFile(`${ARTIFACTS}/t3385-diagnostics-game.json`, JSON.stringify(game, null, 2));
    console.log('PASS real game clock/page checkpoints, including rows stop edit');
  }
  assert.deepEqual(errors, []);
  await writeFile(`${ARTIFACTS}/t3385-diagnostics.json`, JSON.stringify(r, null, 2));
  console.log('PASS disabled by default, clock stall, page stall, reload recovery, bounded log, persistent stop');
} finally { await chrome.close(); }
