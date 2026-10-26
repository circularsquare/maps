// T-1413, T-1474: each building action plays its one sound at the click, and the Sound switch silences
// them. Counts the audio nodes the page makes and the length of each recording started: "ke-cha" is 2
// noise bursts and 2 oscillators, "bung" 1 buffer and 1 oscillator, the recordings 1 buffer each (the
// fft 0.163 s for track, the snap 0.62 s for a station, the pop 0.42 s for a panel edit).
//   node trainworld/checks/t1413-triggers.mjs [url]     (from the maps root; a running dev server)
import { GL, NOISE, acrossLine, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6143/';
const { ctx, close } = await launchChrome({ prefix: 'tw-t1413-', size: [1400, 900], args: [...GL, '--autoplay-policy=no-user-gesture-required'] });
const errors = [];
let fails = 0;
const check = (name, ok, detail = '') => { if (!ok) fails++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`); };
const TRACK = 0.163, SNAP = 0.62, POP = 0.42;
try {
  const page = await ctx.newPage();
  collectErrors(page, errors, NOISE, /\/fonts\/[^/]+\/\d+-\d+\.pbf$/);
  await page.addInitScript(() => {
    window.__snd = { buf: 0, osc: 0, dur: [] };
    const P = BaseAudioContext.prototype;
    const cb = P.createBufferSource, co = P.createOscillator;
    P.createBufferSource = function () { window.__snd.buf++; return cb.call(this); };
    P.createOscillator = function () { window.__snd.osc++; return co.call(this); };
    // the recordings by length: the synthesised buffers are 0.25 s of noise and 0.4 s of string
    const start = AudioBufferSourceNode.prototype.start;
    AudioBufferSourceNode.prototype.start = function (...a) { window.__snd.dur.push(+(this.buffer?.duration ?? 0).toFixed(3)); return start.apply(this, a); };
  });
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  await page.mouse.click(600, 400);
  // the context opens when the page is next idle, and the files decode
  await page.waitForTimeout(4500);
  const snap = () => page.evaluate(() => ({ buf: window.__snd.buf, osc: window.__snd.osc, n: window.__snd.dur.length }));
  /** the nodes made and the buffers started by `f` and the 600 ms after it */
  const around = async (f) => {
    const b = await snap();
    const r = await f();
    await page.waitForTimeout(600);
    const a = await page.evaluate((n) => ({ buf: window.__snd.buf, osc: window.__snd.osc, dur: window.__snd.dur.slice(n) }), b.n);
    return { ...(r ?? {}), buf: a.buf - b.buf, osc: a.osc - b.osc, dur: a.dur };
  };
  const sound = (op) => around(() => page.evaluate(async (op) => {
    const w = await import('/src/game/world.ts');
    const a = await w.edit(op, true);
    return { ok: a.ok, node: a.node };
  }, op));
  const near = (d, x) => d.length === 1 && Math.abs(d[0] - x) < 0.01;
  const jump = (lng, lat, zoom) => page.evaluate(([a, b, z]) => { window.tw.map.jumpTo({ center: [a, b], zoom: z }); }, [lng, lat, zoom]);
  const screen = (lng, lat) => page.evaluate(([a, b]) => { const p = window.tw.map.project([a, b]), r = window.tw.map.getContainer().getBoundingClientRect(); return [r.left + p.x, r.top + p.y]; }, [lng, lat]);
  const setTool = (t) => page.evaluate(async (t) => { const ui = await import('/src/game/ui.ts'); ui.keepPlacing.value = false; ui.tool.value = t; }, t);
  const click = async ([x, y]) => { await page.mouse.move(x, y, { steps: 6 }); await page.waitForTimeout(500); await page.mouse.down(); await page.mouse.up(); };

  // track and a station: real clicks, since the tools play their own sounds in the click handler
  const A = [-74.46, 40.56], B = [-74.44, 40.565];
  await jump(-74.45, 40.562, 14);
  await page.waitForTimeout(1500);
  await setTool('track');
  const t0 = await around(async () => { await click(await screen(...A)); });
  check('first click of a piece: silent', t0.buf === 0 && t0.osc === 0, JSON.stringify(t0));
  const t1 = await around(async () => { await click(await screen(...B)); });
  const placed = await page.evaluate(() => window.tw.world.value.edges.length);
  check('track placed: one fft', placed === 1 && t1.buf === 1 && t1.osc === 0 && near(t1.dur, TRACK), JSON.stringify(t1));
  await setTool('station');
  const s1 = await around(async () => { await click(await screen((A[0] + B[0]) / 2, (A[1] + B[1]) / 2)); });
  const built = await page.evaluate(() => window.tw.world.value.save.stations.length);
  check('station placed: one snap', built === 1 && s1.buf === 1 && s1.osc === 0 && near(s1.dur, SNAP), JSON.stringify(s1));
  await setTool('select');

  // edits sent from panels and buttons play when sent
  const c = await sound({ op: 'constructAll' });
  check('constructAll (many pieces): one ke-cha', c.ok && c.buf === 2 && c.osc === 2, JSON.stringify(c));
  const num = await page.evaluate(() => window.tw.world.value.save.stations.find((s) => s.built)?.num);
  const u = await sound({ op: 'stationUpgrade', nodes: [num], bit: 4, on: true });
  check('upgrade built: one bung', u.ok && u.buf === 1 && u.osc === 1 && !near(u.dur, POP), JSON.stringify(u));
  const u2 = await sound({ op: 'stationUpgrade', nodes: [num], bit: 4, on: false });
  check('upgrade taken out: one pop', u2.ok && u2.buf === 1 && u2.osc === 0 && near(u2.dur, POP), JSON.stringify(u2));
  const bad = await sound({ op: 'stationUpgrade', nodes: [999999], bit: 4, on: false });
  check('a refused panel edit has sounded (the sound is played at the click)', !bad.ok && bad.buf === 1 && near(bad.dur, POP), JSON.stringify(bad));

  // a line: its length, train counts and headways
  await jump(-74.40, 40.64, 14);
  const line = await acrossLine(page, [-74.40, 40.64], 2);
  await page.waitForTimeout(800);
  check('a line built', line !== null);
  if (line !== null) {
    const id = String(line);
    const cars = await sound({ op: 'lineCars', line, cars: 2 });
    check('train length: one pop', cars.buf === 1 && cars.osc === 0 && near(cars.dur, POP), JSON.stringify(cars));
    const step = await around(() => page.evaluate(async (id) => {
      const w = await import('/src/game/world.ts');
      w.stepTrainCount(id, 'high', 1);
      w.stepTrainCount(id, 'high', 1); // a second click while the first edit is in flight
    }, id));
    check('two train-count clicks together: one pop, not one per edit', step.buf === 1 && step.osc === 0 && near(step.dur, POP), JSON.stringify(step));
    const auto = await around(() => page.evaluate(async (id) => { const w = await import('/src/game/world.ts'); w.setAutoTrains(id, true); }, id));
    check('switch to headways: one pop', auto.buf === 1 && near(auto.dur, POP), JSON.stringify(auto));
    await page.waitForTimeout(1500);
    const hw = await around(() => page.evaluate(async (id) => { const w = await import('/src/game/world.ts'); w.stepHeadway(id, 'high', 1); }, id));
    check('a headway step: one pop', hw.buf === 1 && near(hw.dur, POP), JSON.stringify(hw));
    const paste = await around(() => page.evaluate(async (id) => {
      const w = await import('/src/game/world.ts');
      const sc = await import('/src/game/scheduleCopy.ts');
      sc.scheduleCopy.value = { from: 'x', fast: false, cars: 3, headway: null, trains: [4, 3, 2] };
      const l = w.world.value.save.lines.find((x) => x.id === id);
      await sc.pasteSchedule(l, w.world.value.lineStats[id]);
    }, id));
    check('a pasted schedule (up to four edits): one pop', paste.buf === 1 && near(paste.dur, POP), JSON.stringify(paste));
  }

  // T-1490: the fare graph pops once when a dragged node is let go, and only if the fare changed
  await page.click('[aria-label="Money"]');
  await page.waitForTimeout(1000);
  const knobEl = await page.$('.fg-node .fg-knob');
  await knobEl.scrollIntoViewIfNeeded();
  const knob = await knobEl.boundingBox();
  const kx = knob.x + knob.width / 2, ky = knob.y + knob.height / 2;
  await page.mouse.move(kx, ky);
  await page.mouse.down();
  const still = await around(async () => { await page.mouse.up(); });
  check('a fare node pressed and let go unmoved: silent', still.buf === 0 && still.osc === 0, JSON.stringify(still));
  await page.mouse.move(kx, ky);
  await page.mouse.down();
  const dragging = await around(async () => { await page.mouse.move(kx, ky - 14, { steps: 7 }); });
  check('dragging a fare node: silent', dragging.buf === 0 && dragging.osc === 0, JSON.stringify(dragging));
  const dropped = await around(async () => { await page.mouse.up(); });
  check('a fare node let go after a change: one pop', dropped.buf === 1 && dropped.osc === 0 && near(dropped.dur, POP), JSON.stringify(dropped));

  // the Sound switch
  await page.evaluate(async () => { const m = await import('/src/game/ui.ts'); m.display.value = { ...m.display.value, sound: false }; });
  await setTool('track');
  const off = await around(async () => { await click(await screen(A[0], A[1] + 0.02)); await click(await screen(B[0], B[1] + 0.02)); });
  const placed2 = await page.evaluate(() => window.tw.world.value.edges.length);
  check('Sound off: track placed is silent', placed2 > placed && off.buf === 0 && off.osc === 0, JSON.stringify(off));
  check('no console errors', errors.length === 0, errors.slice(0, 3).join(' | '));
} finally {
  await close();
}
process.exit(fails ? 1 : 0);
