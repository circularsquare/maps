// T-1835: five small UI fixes. The group pill's text centred (measured: the ink box of the text
// against the pill's inside, at 2x), Settings > Game "Getting started" Start again in a normal game,
// the keep placing hover, one colour for hotkey hints, and the demand bubble icon. Screenshots light
// and dark in artifacts/trainworld/t1835/. From the maps root:
//   node trainworld/checks/t1835-ui.mjs [url]   (about a minute)
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, REAL_SAVE, GL, launchChrome, collectErrors, importSave } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6193/';
const OUT = join(ARTIFACTS, 't1835');
await mkdir(OUT, { recursive: true });
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };
const errors = [];
const chrome = await launchChrome({ prefix: 'tw-t1835-', size: [1400, 900], args: GL, deviceScaleFactor: 2 });
try {
  const page = await chrome.ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&opened=all&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.evaluate(() => { window.tw.client.lastLoad = null; });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await page.waitForTimeout(3000);
  const settle = (ms = 800) => page.waitForTimeout(ms);
  const ui = (fn, arg) => page.evaluate(async ([src, arg]) => { const m = await import('/src/game/ui.ts'); return (0, eval)(src)(m, arg); }, [fn.toString(), arg]);
  const edit = (op) => page.evaluate(async (op) => (await import('/src/game/world.ts')).edit(op), op);

  // ---- group pill centring
  const ls = await page.evaluate(() => window.tw.world.value.save.lines.slice(0, 6).map((l) => ({ id: l.id, num: l.num, name: l.name })));
  console.log('lines', JSON.stringify(ls));
  await edit({ op: 'lineGroup', lines: [ls[0].id, ls[1].id], group: 'new' });
  await settle();
  const group = await page.evaluate((id) => window.tw.world.value.save.lines.find((l) => l.id === id)?.group, ls[0].id);
  check(!!group, `a group was made (${group})`);
  await ui((m) => { m.tab.value = 'lines'; m.selection.value = null; });
  await page.waitForSelector(`.lines-box li[data-drop="g:${group}"] .group-pill`);
  await settle(1200);
  /** the text's ink box against the pill's inside, px, + below the middle */
  const centre = (sel) => page.$$eval(sel, (pills) => pills.map((p) => {
    const t = p.querySelector('.group-pill-text'), cs = getComputedStyle(t);
    const probe = document.createElement('span');
    probe.style.cssText = 'display:inline-block;width:0;height:0;vertical-align:baseline';
    t.appendChild(probe);
    const base = probe.getBoundingClientRect().bottom;
    probe.remove();
    const c = document.createElement('canvas').getContext('2d');
    c.font = `${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
    c.textBaseline = 'alphabetic';
    const m = c.measureText(t.textContent);
    const r = p.getBoundingClientRect(), s = r.height / p.offsetHeight; // the dock is zoomed: metrics in screen px
    const inkTop = base - m.actualBoundingBoxAscent * s, inkBot = base + m.actualBoundingBoxDescent * s;
    const mid = (r.top + r.bottom) / 2;
    // the cap height: what the pill centres (the letters' bodies, descenders hang below)
    const cap = c.measureText('H').actualBoundingBoxAscent * s;
    return { text: t.textContent, ink: +((inkTop + inkBot) / 2 - mid).toFixed(2), caps: +(base - cap / 2 - mid).toFixed(2), font: cs.fontSize + ' ' + cs.fontFamily.slice(0, 40), pill: [r.top, r.height], textBox: [t.getBoundingClientRect().top, t.getBoundingClientRect().height], base, lh: getComputedStyle(p).lineHeight, spanLh: cs.lineHeight, spanDisplay: cs.display, pillDisplay: getComputedStyle(p).display, parent: p.parentElement.className, top: cs.top, zoom: [...(function* () { for (let e = p; e; e = e.parentElement) { const z = getComputedStyle(e).zoom, tr = getComputedStyle(e).transform; if (z !== '1' || tr !== 'none') yield e.id + '.' + e.className + ' zoom ' + z + ' ' + tr; } })()] };
  }));
  const before = await centre('.group-pill');
  console.log('pill offsets lines tab (+ = low)', JSON.stringify(before));
  await edit({ op: 'groupName', group, name: 'Tokaido Main Line' });
  await settle();
  const shot = async (name, sel) => {
    const el = await page.$(sel); await el.scrollIntoViewIfNeeded(); await settle(300); await el.screenshot({ path: join(OUT, name) });
    // the same without the pill's text: the pixel measure (ink rows against the pill's inside) is made from the pair
    if (!name.includes('pill')) return;
    await page.addStyleTag({ content: '.group-pill-text { visibility: hidden !important; }' }).then(async (h) => { await el.screenshot({ path: join(OUT, 'notext-' + name) }); await h.evaluate((s) => s.remove()); });
  };
  const row = `.lines-box li[data-drop="g:${group}"]`;
  await shot('light-pill-lines-2x.png', row);
  const lines1 = await centre('.lines-box .group-pill');
  console.log('pill offsets (named)', JSON.stringify(lines1));
  await ui((m, g) => { m.selection.value = { kind: 'group', group: g }; }, group);
  await page.waitForSelector('#inspector .title .group-pill');
  await settle(1000);
  const big = await centre('#inspector .title .group-pill');
  console.log('pill offsets group view', JSON.stringify(big));
  await shot('light-pill-group-view-2x.png', '#inspector .title .group-pill');
  await page.screenshot({ path: join(OUT, 'light-group-view.png'), clip: { x: 0, y: 40, width: 440, height: 360 } });
  await edit({ op: 'groupLook', group, symbol: 'Keiyo', colour: null });
  await settle();
  const desc = await centre('#inspector .title .group-pill');
  console.log('pill offsets with descenders', JSON.stringify(desc));
  await shot('light-pill-descender-2x.png', '#inspector .title .group-pill');
  await ui((m, id) => { m.selection.value = { kind: 'line', line: id }; }, ls[0].id);
  await page.waitForSelector('#inspector .line-group .group-pill');
  await settle(800);
  await shot('light-pill-line-panel-2x.png', '#inspector .line-group');
  const panel = await centre('#inspector .line-group .group-pill');
  console.log('pill offsets line panel', JSON.stringify(panel));
  const all = [...before, ...lines1, ...big, ...desc, ...panel];
  // the offsets above use canvas metrics and are only a guide (the panels are zoomed); the pill's real
  // pixels are measured from the saved pairs (light-pill-*-2x.png and notext-*) by a script that diffs them
  console.log('offsets', all.map((o) => o.ink).join(', '));
  check(all.every((o) => Math.abs(o.caps) <= 0.5), `every pill's capital height is centred within 0.5 screen px (${all.map((o) => o.caps).join(', ')})`);
  await edit({ op: 'groupLook', group, symbol: '', colour: null });

  // ---- Settings: Getting started
  await ui((m) => { m.selection.value = null; m.tab.value = 'settings'; });
  await settle(600);
  const btn = '[data-guide-restart]';
  await page.waitForSelector(btn);
  check((await page.$eval(btn, (b) => b.textContent)) === 'Start again', 'Settings > Game has a Start again button in a normal game');
  await shot('light-settings-getting-started.png', '.game-mode:has([data-guide-restart])');
  const gstate = () => page.evaluate(async () => { const g = (await import('/src/game/money.ts')).moneyState.value?.guide; return g ? { done: g.done, hidden: !!g.hidden, closed: !!g.closed } : null; });
  console.log('guide before', JSON.stringify(await gstate()));
  await page.evaluate(async () => { (await import('/src/workers/clockClient.ts')).saveGuide({ done: 63, closed: true }); });
  await settle(800);
  check(!(await page.$('.guide')), 'with the guide closed there is no card');
  await page.click(btn);
  await settle(1200);
  const g2 = await gstate();
  console.log('guide after', JSON.stringify(g2));
  check(!!(await page.$('.guide')) && !g2.closed && !g2.hidden, 'Start again shows the card (closed and hidden cleared)');
  await page.screenshot({ path: join(OUT, 'light-guide-restarted.png') });

  // ---- hotkey colour, hovers
  await ui((m) => { m.tab.value = 'build'; });
  await settle(600);
  const keyCols = await page.evaluate(() => [...document.querySelectorAll('.key')].map((e) => getComputedStyle(e).color));
  check(keyCols.length > 0 && new Set(keyCols).size === 1, `all ${keyCols.length} key letters on the Build tab share one colour (${keyCols[0]})`);
  const hover = async (sel, name) => {
    const b = await (await page.$(sel)).boundingBox();
    await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2);
    await page.mouse.move(b.x + b.width / 2 + 2, b.y + b.height / 2 + 1);
    await settle(300);
    const tag = await page.$('.ui-tag');
    const text = await tag.textContent();
    const keyCol = await page.evaluate(() => { const k = document.querySelector('.ui-tag .tip-key'); return k && getComputedStyle(k).color; });
    await page.screenshot({ path: join(OUT, name), clip: { x: Math.max(0, b.x - 150), y: Math.max(0, b.y - 40), width: 520, height: 160 } });
    return { text, keyCol };
  };
  const kp = await hover('button[aria-label="Keep placing track"]', 'light-hover-keep-placing.png');
  console.log('keep placing hover', JSON.stringify(kp));
  check(kp.text.startsWith('Keep placing track after a click (Caps Lock)') && !!kp.keyCol, 'keep placing hover reads "Keep placing track after a click (Caps Lock)" with the key lettered');
  const tabHover = await hover('.tab[data-guide-tab]', 'light-hover-tab.png');
  console.log('tab hover', JSON.stringify(tabHover));
  check(tabHover.keyCol === kp.keyCol, 'tab hover key shares the colour');
  await page.mouse.move(700, 450);
  await page.screenshot({ path: join(OUT, 'light-build-tab.png'), clip: { x: 0, y: 40, width: 440, height: 360 } });

  // ---- demand bubble icon
  await page.screenshot({ path: join(OUT, 'light-bubble-row.png') });
  const hb = await page.$('.demand-selector .hint-cell');
  const hbb = await hb.boundingBox();
  await page.screenshot({ path: join(OUT, 'light-bubble-icon-2x.png'), clip: { x: hbb.x - 20, y: hbb.y - 10, width: 280, height: hbb.height + 20 } });
  check((await page.$eval('.hint-cell .bubble', (c) => getComputedStyle(c).stroke)) === 'none', 'the bubble icon has no outline');

  // ---- help panel
  await ui((m) => { m.tab.value = 'settings'; });
  await settle(400);
  await page.evaluate(async () => { (await import('/src/ui/HelpPanel.tsx')).helpOpen.value = true; });
  await settle(500);
  await page.screenshot({ path: join(OUT, 'light-help.png') });
  await page.evaluate(async () => { (await import('/src/ui/HelpPanel.tsx')).helpOpen.value = false; });

  // ---- dark
  await ui((m) => { m.shade.value = 'dark'; m.tab.value = 'build'; });
  await settle(1000);
  await page.screenshot({ path: join(OUT, 'dark-build-tab.png'), clip: { x: 0, y: 40, width: 440, height: 360 } });
  const dk = await hover('button[aria-label="Keep placing track"]', 'dark-hover-keep-placing.png');
  console.log('dark hover', JSON.stringify(dk));
  await page.mouse.move(700, 450);
  const hb2 = await (await page.$('.demand-selector .hint-cell')).boundingBox();
  await page.screenshot({ path: join(OUT, 'dark-bubble-icon-2x.png'), clip: { x: hb2.x - 20, y: hb2.y - 10, width: 280, height: hb2.height + 20 } });
  await page.evaluate(async () => { (await import('/src/ui/HelpPanel.tsx')).helpOpen.value = true; });
  await ui((m) => { m.tab.value = 'settings'; });
  await settle(600);
  await page.screenshot({ path: join(OUT, 'dark-help.png') });
  await page.evaluate(async () => { (await import('/src/ui/HelpPanel.tsx')).helpOpen.value = false; });
  await ui((m) => { m.tab.value = 'lines'; m.selection.value = null; });
  await edit({ op: 'groupLook', group, symbol: 'Keiyo', colour: null });
  await settle(1000);
  await shot('dark-pill-lines-2x.png', row);
  check(errors.length === 0, 'no console errors' + (errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''));
} finally {
  await chrome.close();
}
process.exitCode = fail ? 1 : 0;
