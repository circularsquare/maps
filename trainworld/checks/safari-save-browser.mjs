// Browser/clock-worker integration with unusable native compression streams.
// node trainworld/checks/safari-save-browser.mjs http://localhost:6186/
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { launchChrome, collectErrors, importSave, REAL_SAVE } from './lib.mjs';

const base = process.argv[2] ?? 'http://localhost:6186/';
const { ctx, close } = await launchChrome({ prefix: 'tw-safari-save-', acceptDownloads: true });
const errors = [];
try {
  await ctx.addInitScript(() => {
    // Every module worker, including the clock, gets the same broken browser APIs as the page.
    const NativeWorker = Worker;
    const broken = `globalThis.CompressionStream = globalThis.DecompressionStream = class { constructor() { throw new Error('faulty native codec invoked'); } };`;
    globalThis.CompressionStream = globalThis.DecompressionStream = class { constructor() { throw new Error('faulty native codec invoked'); } };
    globalThis.Worker = class extends NativeWorker {
      constructor(url, options) {
        if (options?.type === 'module') {
          const absolute = new URL(url, location.href).href;
          const start = `\nconst queued = []; onmessage = e => queued.push(e); await import(${JSON.stringify(absolute)}); for (const event of queued) onmessage(event);`;
          url = URL.createObjectURL(new Blob([broken, start], { type: 'text/javascript' }));
        }
        super(url, options);
      }
    };
  });
  const page = await ctx.newPage();
  const poll = async (fn, arg) => {
    for (let i = 0; i < 120; i++) {
      const answer = await page.evaluate(fn, arg);
      if (answer) return answer;
      await page.waitForTimeout(250);
    }
    throw new Error('Timed out waiting for persisted save state');
  };
  collectErrors(page, errors, /favicon|openfreemap|AJAXError|ERR_NETWORK_ACCESS_DENIED/);
  await page.goto(`${base}?debug=1&paused=1&backupMs=50`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });
  const beforeSlot = await page.evaluate(async () => (await import('/src/workers/clockClient.ts')).saveSlotsState.value.active);
  await importSave(page, readFileSync(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file', undefined, { timeout: 120000 });
  await page.waitForFunction(() => window.tw.world.value.save.lines.length > 0, undefined, { timeout: 60000 });
  const loaded = await page.evaluate(() => ({ ...window.tw.client.lastLoad, lines: window.tw.world.value.save.lines.length }));
  assert.ok(!loaded.error, JSON.stringify(loaded));
  assert.ok(loaded.lines > 0, 'original NYC lines loaded');
  const slot = await poll(async (before) => {
    const s = (await import('/src/workers/clockClient.ts')).saveSlotsState.value;
    return s.active && s.active !== before && s.writable && await (await import('/src/workers/saveFile.ts')).readStored(s.active) ? s.active : null;
  }, beforeSlot);
  await page.evaluate(async () => (await import('/src/workers/clockClient.ts')).manageSaveSlot('save'));
  await poll(async (key) => (await (await import('/src/workers/saveFile.ts')).listBackups(key)).length > 0, slot);
  const stored = await page.evaluate(async (key) => {
    const sf = await import('/src/workers/saveFile.ts');
    const backups = await sf.listBackups(key);
    const records = [{ key, record: await sf.readStored(key) }];
    for (const b of backups) records.push({ key: b.key, record: await sf.readStored(b.key) });
    for (const { key, record } of records) {
      if (!record) throw new Error(`Missing test record ${key}; backups ${JSON.stringify(backups)}`);
      await sf.decodeSave(record.bytes);
    }
    return records.length;
  }, slot);
  assert.ok(stored >= 2, 'saved slot and backup independently decode');
  const exported = page.waitForEvent('download', { timeout: 60000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'export' }));
  const file = readFileSync(await (await exported).path());
  assert.equal(gunzipSync(file).subarray(0, 4).toString(), 'TWG1', 'export independently valid');
  const checked = await page.evaluate(async (data) => {
    const sf = await import('/src/workers/saveFile.ts');
    const { header, track } = await sf.decodeSave(Uint8Array.from(data));
    const shared = await import('/src/game/sharedSave.ts');
    const forShare = await shared.forShare(Uint8Array.from(data));
    const copy = forShare && await shared.copyFromShare(forShare, 'safari-regression');
    const copied = copy && await sf.decodeSave(copy);
    const codec = await import('/src/workers/gzip.ts');
    const raw = new TextEncoder().encode('UI stays responsive while gzip runs.\n'.repeat(50000));
    let timerRan = false;
    const timer = setTimeout(() => { timerRan = true; }, 0);
    const gz = await codec.compressGzip(raw);
    const offMain = timerRan; clearTimeout(timer);
    let capRejected = false, corruptRejected = false;
    try { await codec.decompressGzip(gz, 10); } catch { capRejected = true; }
    try { await sf.decodeSave(Uint8Array.from(data).slice(0, -8)); } catch { corruptRejected = true; }
    const afterError = await codec.decompressGzip(gz);
    const { browserOf } = await import('/src/ui/FeedbackPanel.tsx');
    return { offMain, capRejected, corruptRejected, afterError: afterError.length === raw.length && raw.byteLength > 0,
      clock: header.clock, track: track.length, copy: copied?.header.copy, sameTrack: copied?.track.every((b, i) => b === track[i]) && copied?.track.length === track.length,
      safari: browserOf('Mozilla/5.0 (Macintosh) AppleWebKit/605.1.15 Version/16.4.1 Safari/605.1.15') };
  }, [...file]);
  assert.ok(checked.track > 20);
  assert.equal(checked.copy, 'safari-regression');
  assert.ok(checked.sameTrack, 'community copy keeps the full network');
  assert.equal(checked.safari, 'Safari 16.4.1');
  assert.ok(checked.offMain, 'large gzip work yields the UI thread');
  assert.ok(checked.capRejected && checked.corruptRejected && checked.afterError, 'codec worker rejects bad input and remains usable');
  await page.reload();
  await page.waitForFunction(() => window.tw?.world?.value?.save?.lines?.length > 0, undefined, { timeout: 120000 });
  assert.equal(await page.evaluate(() => window.tw.world.value.save.lines.length), loaded.lines, 'persisted save reopens with every line');
  assert.deepEqual(errors, [], 'no browser or worker errors');
  console.log(`PASS browser + clock worker: ${loaded.lines} NYC lines imported, slot + backup validated, page reloaded with every line, exported ${file.length} bytes, community save copied, Safari point version retained; native codecs disabled`);
} finally { await close(); }
