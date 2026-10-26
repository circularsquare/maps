// T-1413: each building action plays its one sound, and the Sound switch silences them. Counts the
// audio nodes the page makes: "ke-cha" is 2 noise bursts and 2 oscillators, "bung" 1 buffer and 1
// oscillator, "tsh" 1 noise burst.
//   node trainworld/checks/t1413-triggers.mjs [url]     (from the maps root; a running dev server)
import { GL, NOISE, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6143/';
const { ctx, close } = await launchChrome({ prefix: 'tw-t1413-', size: [1200, 800], args: [...GL, '--autoplay-policy=no-user-gesture-required'] });
const errors = [];
let fails = 0;
const check = (name, ok, detail = '') => { if (!ok) fails++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`); };
try {
  const page = await ctx.newPage();
  collectErrors(page, errors, NOISE, /\/fonts\/[^/]+\/\d+-\d+\.pbf$/);
  await page.addInitScript(() => {
    window.__snd = { buf: 0, osc: 0 };
    const P = BaseAudioContext.prototype;
    const cb = P.createBufferSource, co = P.createOscillator;
    P.createBufferSource = function () { window.__snd.buf++; return cb.call(this); };
    P.createOscillator = function () { window.__snd.osc++; return co.call(this); };
  });
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  await page.mouse.click(600, 400);
  await page.waitForTimeout(2500);
  const sound = (op) => page.evaluate(async (op) => {
    const w = await import('/src/game/world.ts');
    const b = { ...window.__snd };
    const a = await w.edit(op, true);
    return { ok: a.ok, node: a.node, buf: window.__snd.buf - b.buf, osc: window.__snd.osc - b.osc };
  }, op);
  const wait = () => page.waitForTimeout(400);
  const xy = (lng, lat) => page.evaluate(([a, b]) => window.tw.lngLatToPack(a, b), [lng, lat]);
  const p = await xy(-74.30, 40.65), q = await xy(-74.24, 40.65), r = await xy(-74.18, 40.65);

  const t1 = await sound({ op: 'route', from: { kind: 'free', x: p.x, y: p.y, level: 0 }, to: { kind: 'free', x: q.x, y: q.y, level: 0 }, pis: [], single: false });
  check('track placed: one tsh', t1.ok && t1.buf === 1 && t1.osc === 0, JSON.stringify(t1));
  await wait();
  const t2 = await sound({ op: 'route', from: { kind: 'node', node: t1.node }, to: { kind: 'free', x: r.x, y: r.y, level: 0 }, pis: [], single: false });
  check('second track piece: one tsh', t2.ok && t2.buf === 1 && t2.osc === 0, JSON.stringify(t2));
  await wait();
  const s1 = await sound({ op: 'addStation', at: { kind: 'node', node: t1.node }, platform: 200, name: '' });
  check('station placed: one bung', s1.ok && s1.buf === 1 && s1.osc === 1, JSON.stringify(s1));
  await wait();
  const bad = await sound({ op: 'addStation', at: { kind: 'node', node: 999999 }, platform: 200, name: '' });
  check('a refused edit is silent', !bad.ok && bad.buf === 0 && bad.osc === 0, JSON.stringify(bad));
  const c = await sound({ op: 'constructAll' });
  check('constructAll (many pieces): one ke-cha', c.ok && c.buf === 2 && c.osc === 2, JSON.stringify(c));
  await wait();
  const num = await page.evaluate(() => window.tw.world.value.save.stations.find((s) => s.built)?.num);
  const u = await sound({ op: 'stationUpgrade', nodes: [num], bit: 4, on: true });
  check('upgrade built: one bung', u.ok && u.buf === 1 && u.osc === 1, JSON.stringify(u));
  await wait();
  const u2 = await sound({ op: 'stationUpgrade', nodes: [num], bit: 4, on: false });
  check('upgrade taken out: silent', u2.ok && u2.buf === 0 && u2.osc === 0, JSON.stringify(u2));
  // the Sound switch
  await page.evaluate(async () => { const m = await import('/src/game/ui.ts'); m.display.value = { ...m.display.value, sound: false }; });
  const off = await sound({ op: 'route', from: { kind: 'free', x: p.x, y: p.y + 4000, level: 0 }, to: { kind: 'free', x: q.x, y: q.y + 4000, level: 0 }, pis: [], single: false });
  check('Sound off: track placed is silent', off.ok && off.buf === 0 && off.osc === 0, JSON.stringify(off));
  check('no console errors', errors.length === 0, errors.slice(0, 3).join(' | '));
} finally {
  await close();
}
process.exit(fails ? 1 : 0);
