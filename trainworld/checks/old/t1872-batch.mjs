// T-1872: four small items. (1) the delete tool's red hover goes when the tool changes, by key, by
// the Build tab's button, and with the pointer left on the map or taken off it; with the clear
// switched off (the old behaviour) it stays. (2) the headway steppers go to whole minutes, and minus
// from no service gives 5 minutes. (3) a copied line's name and symbol. (4) a game the player opens
// starts on the city with the most built stations. Uses the Vite already on `port`
// (dev-nowatch.mjs). From the maps root:
//   node trainworld/checks/t1872-batch.mjs [port]
import { readFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, REAL_SAVE, launchChrome, collectErrors, importSave, acrossLine } from './lib.mjs';

const port = Number(process.argv[2] ?? 6172);
const OUT = join(ARTIFACTS, 't1872');
await mkdir(OUT, { recursive: true });
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const chrome = await launchChrome({ prefix: 'tw-t1872-', size: [1400, 900] });
const errors = [];
const open = (p) => p.goto(`http://localhost:${port}/?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`).then(() => p.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 }));
try {
  const page = await chrome.ctx.newPage();
  collectErrors(page, errors);
  await open(page);

  // ---- 2. headway steps (pure)
  const steps = await page.evaluate(async () => {
    const { nextHeadway: n } = await import('/src/game/world.ts');
    return { up537: n(322.2, 1), down537: n(322.2, -1), up5: n(300, 1), down5: n(300, -1), none: n(0, -1), noneUp: n(0, 1), min: n(60, -1), min2: n(120, -1), max: n(7200, 1), nearMax: n(7140, 1), up30: n(330, 1), down30: n(330, -1), up10: n(600, 1), up60: n(3600, 1) };
  });
  console.log(JSON.stringify(steps));
  check(steps.up537 === 360 && steps.down537 === 300, '5.37 min: plus 6, minus 5');
  check(steps.up5 === 360 && steps.down5 === 240, '5 min: plus 6, minus 4');
  check(steps.none === 300 && steps.noneUp === 900, 'no service: minus 5 min, plus 15 min');
  check(steps.min === 0 && steps.min2 === 60 && steps.max === 7200 && steps.nearMax === 7200, 'limits kept (1 min minus is none, 120 min is the top)');
  check(steps.up30 === 360 && steps.down30 === 300 && steps.up10 === 660 && steps.up60 === 3660, 'whole minutes above 5 and 15 min too');

  // ---- a built line for the rest
  await page.evaluate(() => window.tw.map.jumpTo({ center: [-73.9, 40.75], zoom: 12 }));
  const line = await acrossLine(page, [-73.9, 40.75], 2);
  check(line !== null, 'a test line was built', String(line));
  await page.waitForTimeout(1500);

  // headway through the real stepper function
  const hw = await page.evaluate(async (num) => {
    const w = await import('/src/game/world.ts');
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const id = String(num);
    await window.tw.client.edit({ op: 'setHeadway', line: num, seconds: [322, 322, 322] });
    await pause(800);
    const a = w.headwayShown(id)?.[0];
    w.stepHeadway(id, 'high', -1);
    await pause(800);
    const b = w.headwayShown(id)?.[0];
    w.stepHeadway(id, 'high', 1);
    await pause(800);
    const c = w.headwayShown(id)?.[0];
    await window.tw.client.edit({ op: 'setHeadway', line: num, seconds: [0, 0, 0] });
    await pause(800);
    const d0 = w.headwayShown(id);
    return { a, b, c, d0 };
  }, line);
  console.log(JSON.stringify(hw));
  check(hw.b === 300 && hw.c === 360, 'the stepper goes 322 -> 300 -> 360', JSON.stringify(hw));

  // ---- 3. clone names
  const look = (num, name, letter) => page.evaluate(([n, nm, l]) => window.tw.client.edit({ op: 'lineLook', line: n, name: nm, letter: l }), [num, name, letter]);
  const copyOf = async (num) => {
    const a = await page.evaluate((n) => window.tw.client.edit({ op: 'copyLine', line: n }), num);
    await page.waitForTimeout(500);
    return page.evaluate((id) => { const l = window.tw.world.value.save.lines.find((x) => x.num === id); return l && { name: l.name, symbol: l.symbol ?? '', letter: l.letter }; }, a.line);
  };
  await look(line, 'Line 1', '');
  await page.waitForTimeout(500);
  const c1 = await copyOf(line);
  check(c1.name === 'Line 1a' && c1.letter === '1a' && c1.symbol === '1a', 'copy of "Line 1"', JSON.stringify(c1));
  const c2 = await copyOf(line);
  check(c2.name === 'Line 1b' && c2.letter === '1b', 'second copy of "Line 1" is 1b', JSON.stringify(c2));
  const nums = await page.evaluate(() => window.tw.world.value.save.lines.map((l) => [l.num, l.name]));
  const oneA = nums.find((x) => x[1] === 'Line 1a')[0];
  const c3 = await copyOf(oneA);
  check(c3.name === 'Line 1c' && c3.letter === '1c', 'copy of "Line 1a" skips the taken 1b', JSON.stringify(c3));
  await look(line, 'Tokaido mainline', 't');
  await page.waitForTimeout(500);
  const t1 = await copyOf(line);
  check(t1.name === 'Tokaido mainline clone 1' && t1.symbol === 't1', 'copy of "Tokaido mainline" / t', JSON.stringify(t1));
  const t2 = await copyOf(line);
  check(t2.name === 'Tokaido mainline clone 2' && t2.symbol === 't2', 'second copy: clone 2 / t2', JSON.stringify(t2));
  await look(line, 'Tokaido mainline', 'tm');
  await page.waitForTimeout(500);
  const t3 = await copyOf(line);
  check(t3.name === 'Tokaido mainline clone 3' && t3.symbol === 't3', 'symbol "tm" gives t3 (t1, t2 taken)', JSON.stringify(t3));
  await look(line, 'Tokaido mainline', '');
  await page.waitForTimeout(500);
  const t4 = await copyOf(line);
  check(t4.symbol === 'T4' || t4.symbol === 'T1', 'no symbol: first letter of the name shown', JSON.stringify(t4));
  await page.evaluate(() => window.tw.client.edit({ op: 'lineLook', line: window.tw.world.value.save.lines[0].num, letter: '' }));

  // ---- 1. the delete hover
  await look(line, 'Check line', '');
  await page.evaluate(async () => {
    const { tool } = await import('/src/game/ui.ts');
    tool.value = 'select';
    window.tw.map.jumpTo({ center: [-73.9, 40.75], zoom: 13 });
  });
  await page.waitForFunction(() => window.tw.map.loaded());
  await page.keyboard.press('b');
  await page.waitForTimeout(800);
  const box = await (await page.$('#map')).boundingBox();
  const onTrack = () => page.evaluate(() => {
    const n = window.tw.renderer.net;
    let best = null;
    for (const [edge, pts] of n.edgePts) {
      if (pts.length < 6) continue;
      const k = (pts.length / 3 >> 1) * 3;
      const [x, y] = window.tw.overlay.toScreen(pts[k], pts[k + 1]);
      if (x > 100 && y > 100 && x < 1200 && y < 800 && (!best || Math.abs(x - 600) < Math.abs(best[0] - 600))) best = [x, y];
    }
    return best;
  });
  const state = () => page.evaluate(() => ({ tool: window.tw.tools.tool?.peek?.() ?? null, preview: !!window.tw.renderer.previewSrc?.count, del: !!window.tw.tools.delHover }));
  const hover = async () => {
    await page.keyboard.press('3');
    await page.waitForTimeout(300);
    const at = await onTrack();
    await page.mouse.move(box.x + at[0] - 30, box.y + at[1] - 30);
    await page.mouse.move(box.x + at[0], box.y + at[1], { steps: 4 });
    await page.waitForTimeout(400);
    return at;
  };
  const at = await hover();
  const s0 = await state();
  check(s0.preview && s0.del, 'hovering a segment in the delete tool shows the red stretch', JSON.stringify(s0));
  await page.screenshot({ path: join(OUT, 'delete-hover.png') });

  // a. by key, pointer still on the map
  await page.keyboard.press('2');
  await page.waitForTimeout(400);
  const sa = await state();
  check(!sa.preview && !sa.del, 'key 2 (station) clears it', JSON.stringify(sa));
  await page.screenshot({ path: join(OUT, 'after-key.png') });

  // b. by the Build tab's button (the pointer leaves the map for it)
  await hover();
  await page.click('button:has(.tool-label:text-is("Station"))');
  await page.waitForTimeout(400);
  const sb = await state();
  check(!sb.preview && !sb.del, 'the Station button clears it', JSON.stringify(sb));

  // c. the pointer taken off the map, then the key
  await hover();
  await page.mouse.move(box.x + box.width + 40, box.y + 300, { steps: 3 });
  await page.waitForTimeout(300);
  const sc0 = await state();
  check(!sc0.preview && !sc0.del, 'the pointer leaving the map clears it', JSON.stringify(sc0));
  await page.keyboard.press('2');
  await page.waitForTimeout(300);
  check(!(await state()).preview, 'and the key after it');

  // d. the Select tool and the track tool too
  for (const [key, name] of [['4', 'Select'], ['1', 'Track']]) {
    await hover();
    await page.keyboard.press(key);
    await page.waitForTimeout(300);
    const s = await state();
    check(!s.preview && !s.del, `key ${key} (${name}) clears it`, JSON.stringify(s));
  }

  // e. the late reply: a delete click's edit finishes after the tool changed (the reported bug). An
  // unbuilt piece goes without a question, so the click's edit is awaited; it is held back 1.5 s here.
  await page.evaluate(async () => {
    const tw = window.tw, p = tw.lngLatToPack(-73.93, 40.79), q = tw.lngLatToPack(-73.90, 40.79);
    await tw.client.edit({ op: 'route', from: { kind: 'free', x: p.x, y: p.y, level: -1 }, to: { kind: 'free', x: q.x, y: q.y, level: -1 }, pis: [], single: false });
  });
  await page.waitForTimeout(1200);
  const spot = await page.evaluate(() => {
    const w = window.tw.world.value, n = window.tw.renderer.net;
    for (const e of w.edges) {
      if (e.built) continue;
      const pts = n.edgePts.get(e.id);
      if (!pts || pts.length < 6) continue;
      const k = (pts.length / 3 >> 1) * 3;
      return window.tw.overlay.toScreen(pts[k], pts[k + 1]);
    }
    return null;
  });
  check(!!spot, 'an unbuilt piece of track to delete', JSON.stringify(spot));
  await page.evaluate(() => window.tw.map.jumpTo({ center: [-73.915, 40.79], zoom: 13 }));
  await page.waitForTimeout(800);
  const spot2 = await page.evaluate(() => {
    const w = window.tw.world.value, n = window.tw.renderer.net;
    for (const e of w.edges) {
      if (e.built) continue;
      const pts = n.edgePts.get(e.id);
      if (!pts || pts.length < 6) continue;
      const k = (pts.length / 3 >> 1) * 3;
      return window.tw.overlay.toScreen(pts[k], pts[k + 1]);
    }
    return null;
  });
  await page.evaluate(() => { const c = window.tw.client, o = c.edit.bind(c); c.edit = async (...a) => { await new Promise((r) => setTimeout(r, 1500)); return o(...a); }; });
  await page.keyboard.press('3');
  await page.mouse.move(box.x + spot2[0] - 20, box.y + spot2[1] - 20);
  await page.mouse.move(box.x + spot2[0], box.y + spot2[1], { steps: 4 });
  await page.waitForTimeout(300);
  check((await state()).preview, 'the unbuilt piece shows red under the delete tool');
  // the mechanism: a hover computed while another tool is up paints red, and only a tool change clears it
  await page.keyboard.press('2');
  await page.evaluate((s) => { window.tw.tools.deleteHover(s[0], s[1]); }, spot2);
  check((await state()).preview, 'control: a delete hover run under the station tool does paint red (what the late reply used to do)');
  await page.screenshot({ path: join(OUT, 'bug-control.png') });
  await page.keyboard.press('3');
  await page.mouse.move(box.x + spot2[0] - 20, box.y + spot2[1] - 20);
  await page.mouse.move(box.x + spot2[0], box.y + spot2[1], { steps: 4 });
  await page.waitForTimeout(300);
  await page.mouse.click(box.x + spot2[0], box.y + spot2[1]);
  await page.keyboard.press('2');
  await page.waitForTimeout(3500);
  const sd = await state();
  check(!sd.preview && !sd.del, 'switching to the station tool before the delete finishes leaves no red', JSON.stringify(sd));
  await page.screenshot({ path: join(OUT, 'after-late-reply.png') });
  // ---- 4. the start view
  const g = await chrome.ctx.newPage();
  collectErrors(g, errors);
  await open(g);
  await g.evaluate(() => window.tw.map.jumpTo({ center: [10, 50], zoom: 3 }));
  await g.evaluate(() => window.tw.client.fitNextGame());
  await importSave(g, await readFile(REAL_SAVE));
  await g.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await g.waitForTimeout(5000);
  const v = await g.evaluate(async () => {
    const { builtCities } = await import('/src/game/worldTotals.ts');
    const { CITIES } = await import('/src/game/cities.ts');
    const b = [...builtCities()].sort((a, c) => c[1] - a[1]);
    const c = CITIES.find((x) => x.id === b[0][0]);
    const m = window.tw.map, ctr = m.getCenter();
    return { top: b[0], second: b[1] ?? null, want: c && { center: c.center, zoom: c.zoom }, got: { center: [ctr.lng, ctr.lat], zoom: m.getZoom() } };
  });
  console.log(JSON.stringify(v));
  check(v.want && Math.abs(v.got.zoom - v.want.zoom) < 0.01 && Math.abs(v.got.center[0] - v.want.center[0]) < 1e-3 && Math.abs(v.got.center[1] - v.want.center[1]) < 1e-3, 'the opened save is on the city with the most stations', JSON.stringify(v));
  await g.screenshot({ path: join(OUT, 'start-view.png') });
  check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} finally {
  await chrome.close();
}
process.exit(fail ? 1 : 0);
