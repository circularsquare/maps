// T-2163, T-2235: the network diagram's clicks, hovers, camera, build keys, picture and loading bar, on the
// real New York save in headless Chrome. The first layout shows "Calculating diagram" with a loading bar that
// rises; the diagram opens on the complex nearest the map's centre; a hover on Times Sq shows its tag and a
// click opens its inspector; M back goes to the map centred on it, and M again opens the diagram centred on it;
// a click on a line selects it, the rest faded (screenshots light and dark); a build key goes back to the map
// with the tool on; "Save picture" downloads a PNG whose pitch is at a names level.
//   node trainworld/checks/t2163-diagram-clicks.mjs [url]      (about a minute; screenshots and the PNG in
//   artifacts/trainworld/t2163/)
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { ARTIFACTS, REAL_SAVE, NOISE, launchChrome, collectErrors, importSave } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6194/';
const OUT = `${ARTIFACTS}t2163/`;
mkdirSync(OUT, { recursive: true });
const CENTRE = 'Times Sq-42 St';
let failed = 0;
const line = (ok, name, info) => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${info ? ': ' + info : ''}`); };
const errors = [];
const { ctx, close } = await launchChrome({ prefix: 'tw-t2163-', size: [1400, 900], acceptDownloads: true });
try {
  const page = await ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 2 })); } catch {} });
  collectErrors(page, errors, NOISE);
  const sep = BASE.includes('?') ? '&' : '?';
  await page.goto(`${BASE}${sep}debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await page.waitForTimeout(1500);

  // the map on Times Sq, so the diagram opens there
  const ts = await page.evaluate((name) => {
    const st = window.tw.world.value.save.stations.filter((s) => s.name.startsWith(name.split('-')[0]));
    const s = st.find((x) => x.name === name) ?? st[0];
    window.tw.map.jumpTo({ center: [s.lng, s.lat], zoom: 14 });
    return { id: s.id, lng: s.lng, lat: s.lat, name: s.name };
  }, CENTRE);

  // 1. the first layout: the loading bar, its share rising
  await page.evaluate(() => {
    const seen = (window.twBarSeen = { text: false, values: [] });
    new MutationObserver(() => {
      for (const t of document.querySelectorAll('.status-tag')) if (t.textContent.includes('Calculating diagram')) seen.text = true;
      const v = document.querySelector('.status-tag .load-bar')?.getAttribute('aria-valuenow');
      if (v !== null && v !== undefined && seen.values.at(-1) !== +v) seen.values.push(+v);
    }).observe(document.body, { subtree: true, childList: true, attributes: true, attributeFilter: ['aria-valuenow', 'style'] });
  });
  const t0 = Date.now();
  await page.keyboard.press('m');
  await page.waitForFunction(() => window.twDiagram?.layouts >= 1, undefined, { timeout: 90000 });
  const firstS = (Date.now() - t0) / 1000;
  await page.waitForTimeout(300);
  const bar = await page.evaluate(() => ({ ...window.twBarSeen, after: !!document.querySelector('.status-tag .load-bar') }));
  const rising = bar.values.every((v, i) => i === 0 || v >= bar.values[i - 1]);
  line(bar.text && bar.values.length >= 4 && rising && Math.max(...bar.values) >= 80 && !bar.after, 'loading bar with "Calculating diagram" rises through the first layout and goes',
    `${firstS.toFixed(1)} s; values ${bar.values.join(' ')}`);
  const opened = await page.evaluate(() => window.twDiagram.middle());
  line(opened === CENTRE, 'the diagram opens on the complex nearest the map\'s centre', opened);

  // 2. hover and click on Times Sq, once the loading card in the middle has gone
  await page.waitForFunction(() => !document.querySelector('.loading-card'), undefined, { timeout: 240000, polling: 1000 });
  await page.evaluate((n) => window.twDiagram.centreOn(n, 14), CENTRE);
  await page.waitForTimeout(200);
  const at = await page.evaluate((n) => window.twDiagram.where(n), CENTRE);
  await page.mouse.move(at[0], at[1]);
  await page.waitForTimeout(150);
  const tag = await page.evaluate(() => { const t = document.querySelector('.track-tag'); return t && t.style.display !== 'none' ? t.innerText : null; });
  line(!!tag && tag.includes('Times Sq'), 'a hover on Times Sq shows its tag', tag?.replace(/\n/g, ' | '));
  await page.mouse.click(at[0], at[1]);
  await page.waitForTimeout(400);
  const sel = await page.evaluate(() => { const s = window.tw.selection.value; const st = s?.kind === 'station' ? window.tw.world.value.save.stations.find((x) => x.id === s.station) : null; return { kind: s?.kind, name: st?.name, inspector: !!document.querySelector('.complex-stations'), text: document.body.innerText.includes('Times Sq') }; });
  line(sel.kind === 'station' && /Times Sq|42 St/.test(sel.name ?? '') && sel.inspector, 'a click on Times Sq opens its inspector', JSON.stringify(sel));
  await page.screenshot({ path: `${OUT}station-selected-light.png` });

  // 3. the camera: pan away, M back to the map centred on Times Sq, M again to the diagram centred on it
  await page.mouse.move(700, 450);
  await page.mouse.down();
  await page.mouse.move(400, 300, { steps: 6 });
  await page.mouse.up();
  await page.evaluate(() => window.tw.map.jumpTo({ center: [-73.95, 40.65] }));
  await page.keyboard.press('m');
  await page.waitForTimeout(300);
  const back = await page.evaluate((t) => { const c = window.tw.map.getCenter(); const m = 111320; return { shown: document.querySelector('.diagram-canvas').style.display, off: Math.hypot((c.lng - t.lng) * m * Math.cos(t.lat * Math.PI / 180), (c.lat - t.lat) * m) }; }, ts);
  line(back.shown === 'none' && back.off < 400, 'M returns to the map centred on Times Sq', `${back.off.toFixed(0)} m off`);
  await page.evaluate(() => window.tw.map.jumpTo({ center: [-73.95, 40.65] }));
  await page.keyboard.press('m');
  await page.waitForTimeout(300);
  const again = await page.evaluate(() => window.twDiagram.middle());
  line(again === CENTRE, 'M again opens the diagram on the selected station', again);

  // 4. a line click: the line full, the rest faded
  const lineNum = await page.evaluate((id) => window.tw.world.value.save.lines.find((l) => l.stops.includes(id) && l.name.length <= 3)?.num ?? window.tw.world.value.save.lines.find((l) => l.stops.includes(id))?.num, ts.id);
  await page.evaluate((n) => window.twDiagram.centreOn(n, 9), CENTRE);
  await page.waitForTimeout(200);
  const lp = await page.evaluate((n) => window.twDiagram.linePoint(n), lineNum);
  if (lp) {
    await page.mouse.move(lp[0], lp[1]);
    await page.waitForTimeout(150);
    const ltag = await page.evaluate(() => { const t = document.querySelector('.track-tag'); return t && t.style.display !== 'none' ? t.innerText : null; });
    const lname = await page.evaluate((n) => window.tw.world.value.save.lines.find((l) => l.num === n).name, lineNum);
    line(!!ltag && ltag.includes(lname), 'a hover on a line shows its tag', ltag?.replace(/\n/g, ' | '));
    await page.mouse.click(lp[0], lp[1]);
    await page.waitForTimeout(400);
  }
  const lsel = await page.evaluate(() => window.tw.selection.value);
  line(lsel?.kind === 'line' && Number(lsel.line) === lineNum, 'a click on a line selects it', JSON.stringify(lsel));
  await page.mouse.move(1390, 890);
  await page.waitForTimeout(200);
  await page.screenshot({ path: `${OUT}line-selected-light.png` });
  await page.evaluate(() => { window.tw.shade.value = 'dark'; });
  await page.waitForTimeout(400);
  await page.screenshot({ path: `${OUT}line-selected-dark.png` });
  await page.evaluate(() => { window.tw.shade.value = 'light'; });
  await page.waitForTimeout(200);

  // 5. a build key goes back to the map with its tool
  await page.keyboard.press('1');
  await page.waitForTimeout(300);
  const built = await page.evaluate(() => ({ shown: document.querySelector('.diagram-canvas').style.display, on: [...document.querySelectorAll('[aria-pressed="true"]')].map((b) => b.getAttribute('data-tip') ?? b.textContent).join(', ') }));
  line(built.shown === 'none' && built.on.includes('Build'), 'key 1 goes back to the map, the Build tab open', `${built.shown}; pressed: ${built.on.slice(0, 80).replace(/\n/g, ' ')}`);
  await page.keyboard.press('Escape');
  await page.waitForTimeout(200);

  // 6. the picture
  await page.evaluate(() => { window.tw.selection.value = null; });
  await page.keyboard.press('m');
  await page.waitForSelector('.diagram-save', { timeout: 30000 });
  const [dl] = await Promise.all([page.waitForEvent('download', { timeout: 60000 }), page.click('.diagram-save')]);
  const file = `${OUT}nyc-diagram.png`;
  await dl.saveAs(file);
  const png = readFileSync(file), w = png.readUInt32BE(16), h = png.readUInt32BE(20);
  const info = await page.evaluate(() => window.twDiagramPicture);
  line(dl.suggestedFilename().endsWith(' diagram.png') && w === info.w && h === info.h && info.pitchPx >= info.levels[0], 'Save picture downloads the whole diagram with names',
    `${dl.suggestedFilename()}, ${w} x ${h}, pitch ${info.pitchPx.toFixed(2)} px (names from ${info.levels[0].toFixed(2)}), ${(png.length / 1e6).toFixed(1)} MB`);
  writeFileSync(`${OUT}result.json`, JSON.stringify({ firstS, bar, opened, tag, sel, back, again, lineNum, lsel, picture: { w, h, ...info, name: dl.suggestedFilename() } }, null, 1));
} catch (e) {
  line(false, 'check ran', e.message.split('\n')[0]);
} finally {
  line(!errors.length, 'no page or console errors', errors.slice(0, 3).join(' / '));
  await close();
}
process.exit(failed ? 1 : 0);
