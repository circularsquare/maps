// T-3734 stage 1: two tabs on one save slot (T-3611, T-3698). One Chrome profile, tabs A and B on the same slot.
//   1. build: A's save is held while it is being built; B changes the game and saves it manually (taking the
//      slot); A is let go. B's save must still be in the slot (T-3611, the lost-lease message arriving in time).
//   2. write: the same with the roles swapped and the hold just before the slot write, after the lease was
//      checked (T-3611, the message too late; the write itself must refuse).
//   3. T-3698 (open, expected to fail): the other tab writes the slot while this one still holds its lease (a
//      late lock message, stood in for by a write from B's page without the lease); then A saves manually. It
//      should take the slot back or say the game was saved in another tab, not "could not be saved". Printed as
//      XFAIL while it fails and XPASS once fixed (then make it a plain line); neither changes the exit code.
// The holds are stand-ins for a slow worker: saveFile.ts (the storage layer, which the refactor keeps) is patched
// in each tab's clock worker so `encodeSave` or `writeSlot` waits on a BroadcastChannel message naming that tab.
//   node trainworld/checks/t3734-two-tabs.mjs [url]     (from the maps root; default localhost:6177)
// Prints PASS / FAIL / XFAIL lines and exits 1 on a FAIL. About 40 s on three cores.
import { createHash } from 'node:crypto';
import { launchChrome, collectErrors, ANGLE, acrossLine } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6177/';
let failed = false;
const report = (name, ok, detail) => { failed ||= !ok; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
const pause = (ms) => new Promise((r) => setTimeout(r, ms));

/** the gate put at the top of saveFile.ts in a worker: `await __tw3734?.("build" | "write")` holds once when armed */
const gate = (tab) => `const __tw3734 = (() => { if (typeof WorkerGlobalScope === "undefined" || !/clock.worker/.test(self.location.href)) return null;
  const c = new BroadcastChannel("t3734"); let armed = null, open = null;
  c.onmessage = (e) => { const d = e.data; if (d?.tab !== ${JSON.stringify(tab)}) return;
    if (d.cmd === "hold") { armed = d.at; c.postMessage({ tab: ${JSON.stringify(tab)}, armed: d.at }); } else if (d.cmd === "release") { armed = null; open?.(); open = null; } };
  return async (at) => { if (armed !== at) return; armed = null; c.postMessage({ tab: ${JSON.stringify(tab)}, held: at }); await new Promise((r) => { open = r; }); };
})();\n`;
const patched = { A: 0, B: 0 };

const { ctx, close } = await launchChrome({ prefix: 'tw-t3734-tabs-', size: [1300, 900], args: ANGLE });
const errors = [];
try {
  const open = async (tab) => {
    const page = await ctx.newPage();
    collectErrors(page, errors);
    await page.route(/\/src\/workers\/saveFile\.ts/, async (route) => {
      const res = await route.fetch();
      let t = await res.text();
      const n = t.length;
      t = t.replace(/(export async function encodeSave\([^)]*\)\s*\{)/, '$1 await __tw3734?.("build");')
        .replace(/(export async function writeSlot\([^)]*\)\s*\{)/, '$1 await __tw3734?.("write");');
      if (t.length - n === 2 * ' await __tw3734?.("build");'.length) patched[tab]++;
      await route.fulfill({ response: res, body: gate(tab) + t });
    });
    // what each manual save answered
    await page.addInitScript(() => {
      const W = window.Worker;
      window.__byHand = [];
      window.Worker = class extends W {
        constructor(u, o) { super(u, o); this.addEventListener('message', (e) => { if (e.data?.kind === 'savedByHand') window.__byHand.push(e.data.error ?? 'ok'); }); }
      };
    });
    await page.goto(`${BASE}?debug=1&paused=1&lang=en&v=${Date.now()}`);
    await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
    await page.waitForTimeout(1500);
    return page;
  };
  const record = (page, key) => page.evaluate(async (key) => {
    const r = await (await import('/src/workers/saveFile.ts')).readStored(key);
    if (!r) return null;
    const d = await crypto.subtle.digest('SHA-256', r.bytes);
    return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, '0')).join('').slice(0, 12) + ' edit ' + (r.edit ?? r.at);
  }, key);
  const slots = (page) => page.evaluate(async () => {
    const s = (await import('/src/workers/clockClient.ts')).saveSlotsState.value;
    return { active: s.active, writable: s.writable, reason: s.reason ?? '', error: s.error ?? '' };
  });
  // a manual save, as Ctrl+S; its answer from the worker
  const saveByHand = async (page) => {
    const n = await page.evaluate(() => window.__byHand.length);
    await page.evaluate(async () => { void (await import('/src/workers/clockClient.ts')).manageSaveSlot('save'); });
    await page.waitForFunction((n) => window.__byHand.length > n, n, { timeout: 60000 });
    return page.evaluate(() => window.__byHand.at(-1));
  };
  let seq = 0;
  const edit = async (page) => {
    seq++;
    await page.evaluate((s) => window.tw.client.post({ kind: 'edit', seq: 973400 + s, op: { op: 'fares', base: 2 + s * 0.1, perKm: 0.1 } }), seq);
    await page.waitForTimeout(1200);
  };
  /** arms the hold in `tab`'s clock worker; resolves once the worker says it is armed */
  const hold = (page, tab, at) => page.evaluate(([tab, at]) => new Promise((armed, no) => {
    const c = new BroadcastChannel('t3734');
    window.__held = new Promise((r) => { c.onmessage = (e) => { if (e.data?.tab !== tab) return; if (e.data.armed === at) armed(); if (e.data.held === at) r(true); }; setTimeout(() => r(false), 30000); });
    setTimeout(() => no(new Error('the clock worker never armed its hold')), 10000);
    new BroadcastChannel('t3734').postMessage({ tab, cmd: 'hold', at });
  }), [tab, at]);
  const release = (page, tab) => page.evaluate((tab) => new BroadcastChannel('t3734').postMessage({ tab, cmd: 'release' }), tab);

  const A = await open('A');
  const line = await acrossLine(A, [-73.985, 40.758], 2);
  await saveByHand(A);
  const key = (await slots(A)).active;
  const B = await open('B');
  const b0 = await slots(B);
  if (!patched.A || !patched.B) throw new Error(`saveFile.ts was not patched in both tabs (${JSON.stringify(patched)}): the stand-in no longer fits`);
  if (line === null || b0.active !== key || b0.writable) throw new Error(`set-up: A's line ${line}; B on ${b0.active}, writable ${b0.writable} (want ${key}, read-only)`);

  /** `writer` holds the slot; its save is held at `at` while `taker` changes the game and saves it manually */
  const race = async (name, writer, wTab, taker, at) => {
    await edit(writer);
    await hold(writer, wTab, at);
    const wSave = writer.evaluate(async () => { const n = window.__byHand.length; void (await import('/src/workers/clockClient.ts')).manageSaveSlot('save'); for (let i = 0; i < 600 && window.__byHand.length === n; i++) await new Promise((r) => setTimeout(r, 100)); return window.__byHand.at(-1) ?? 'none'; });
    const held = await writer.evaluate(() => window.__held);
    if (!held) { await release(writer, wTab); throw new Error(`${name}: the writer's save never reached the "${at}" hold`); }
    await edit(taker);
    const tSave = await saveByHand(taker);
    await pause(800);
    const afterTake = await record(taker, key);
    await release(writer, wTab);
    const wAnswer = await wSave;
    await pause(1500);
    const final = await record(taker, key);
    report(`${name}: the manual save of the tab that took the slot is kept`, tSave === 'ok' && final === afterTake,
      `(taker's save ${tSave}; slot after it ${afterTake}, at the end ${final}; held writer's save ${wAnswer}; writer now ${JSON.stringify(await slots(writer))})`);
  };
  await race('build', A, 'A', B, 'build');
  await race('write', B, 'B', A, 'write');

  // ---- 3. T-3698: A holds the lease; the slot is written under it; A saves manually
  const a3 = await slots(A);
  await B.evaluate(async (key) => {
    const sf = await import('/src/workers/saveFile.ts');
    const r = await sf.readStored(key);
    await sf.writeSlot(key, { ...r, at: Date.now(), edit: Date.now() + 5000 });
  }, key);
  const forged = await record(A, key);
  await edit(A);
  const answer = await saveByHand(A);
  const after = await record(A, key);
  const s3 = await slots(A);
  // the answer is a message key (clockWorker.otherTabSaved) or text; `reason` already says "another tab" today
  const ok = answer === 'ok' && after !== forged || /otherTab|another tab/i.test(answer) || /another tab/i.test(s3.error);
  console.log(`${ok ? 'XPASS (T-3698 looks fixed: make this a plain PASS line)' : 'XFAIL (T-3698, open)'}  a manual save after the slot was written under this tab's lease takes it back or says another tab saved it  (A before ${JSON.stringify(a3)}; answer "${answer}"; slot ${forged} -> ${after}; A after ${JSON.stringify(s3)})`);

  const real = errors.filter((e) => !/WebGL|GPU/i.test(e));
  report('no page errors', real.length === 0, real.slice(0, 3).join(' | '));
} finally {
  await close();
}
process.exit(failed ? 1 : 0);
