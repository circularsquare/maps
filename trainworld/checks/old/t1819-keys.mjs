// T-1819, T-1834: the key map. Presses each key in a new game and reads which dock tab, tool and elevation
// button is on: Shift+1 to Shift+7 elevation, 1 2 3 4 tools, B L A F Y tabs, S edit stops, G add passed, U bus, nothing
// while typing in a text box, a bare digit not an elevation, hovers showing the keys. From the maps root:
//   node trainworld/checks/old/t1819-keys.mjs [url]   (about 30 seconds)
import { launchChrome, collectErrors, GL } from '../lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6193/';
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };
const errors = [];
const chrome = await launchChrome({ prefix: 'tw-t1819-', size: [1400, 900], args: GL });
try {
  const page = await chrome.ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value, undefined, { timeout: 180000 });
  await page.waitForTimeout(1500);

  const tabOn = () => page.evaluate(() => document.querySelector('#dock .tabs .tab.on')?.getAttribute('aria-label') ?? null);
  const toolOn = () => page.evaluate(() => {
    const b = document.querySelector('.tool-pick button.on');
    return b ? (b.querySelector('.tool-label')?.textContent ?? '') : null;
  });
  // T-2282: the level buttons are on the map's track bar only while the track tool is on, so the level is read from its signal
  const levelOn = () => page.evaluate(async () => { const [ui, t] = await Promise.all([import('/src/game/ui.ts'), import('/src/game/types.ts')]); return t.LEVELS.indexOf(ui.buildLevel.value); });
  const press = async (k) => { await page.keyboard.press(k); await page.waitForTimeout(150); };

  // tabs, from any tab
  for (const [k, name] of [['b', 'Build'], ['l', 'Lines'], ['a', 'Stations'], ['f', 'Finances'], ['y', 'Cities'], ['b', 'Build']]) {
    await press(k);
    const got = await tabOn();
    check(got === name, `key ${k.toUpperCase()} opens ${name} (tab on: ${got})`);
  }
  await press('L');
  check(await tabOn() === 'Lines', 'a capital L (Caps Lock) opens Lines too');

  // the tab hovers carry the keys
  const tips = await page.evaluate(() => [...document.querySelectorAll('#dock .tabs .tab')].map((b) => b.getAttribute('data-tip')));
  console.log('tab hovers', JSON.stringify(tips));
  check(['Build (B)', 'Lines (L)', 'Stations (A)', 'Finances (F)', 'Cities (Y)'].every((t, i) => tips[i] === t), 'tab hovers show the keys and Finances replaces Money');

  // elevation, from the Build tab and from another tab
  await press('b');
  for (let n = 1; n <= 7; n++) {
    await press(`Shift+${n}`);
    const i = await levelOn();
    check(i === n - 1, `key Shift+${n} sets elevation button ${n} (on: ${i + 1})`);
  }
  await press('3');
  check(await levelOn() === 6, 'a bare 3 leaves the elevation alone');
  await press('Escape');
  await press('l');
  await press('Shift+2');
  await press('b');
  check(await levelOn() === 1, 'Shift+digit sets the elevation from the Lines tab');
  await press('1');
  const lvTips = await page.evaluate(() => [...document.querySelectorAll('.tool-bar .levels button')].map((b) => b.getAttribute('data-tip')));
  await press('Escape');
  console.log('elevation hovers', JSON.stringify(lvTips));
  check(lvTips.length === 7 && lvTips.every((t, i) => t.split('\n')[0].endsWith(`(Shift+${i + 1})`)), 'the track bar\'s level hovers name Shift+number');

  // tools, from another tab; the Build tab opens
  for (const [k, name] of [['1', 'Track'], ['2', 'Station'], ['3', 'Delete'], ['4', 'Select']]) {
    await press('l');
    await press(k);
    const t = await tabOn(), got = await toolOn();
    check(t === 'Build' && got === name, `key ${k} picks ${name} and opens Build (tab ${t}, tool ${got})`);
    await press('Escape');
  }
  const toolTips = await page.evaluate(() => [...document.querySelectorAll('.tool-pick button')].map((b) => [b.getAttribute('data-tip'), b.querySelector('.key')?.textContent]));
  console.log('tool hovers', JSON.stringify(toolTips));
  check(['1', '2', '3', '4', 'U'].every((k, i) => toolTips[i][1] === k && toolTips[i][0].includes(`(${k})`)), 'tool buttons show their keys in label and hover');

  // the bus tool
  await press('Escape');
  await press('u');
  const bus = await page.evaluate(() => document.querySelector('.tool-bus')?.classList.contains('on'));
  check(bus === true, 'key U starts the bus tool');
  await press('Escape');
  await press('Escape');
  check(await page.evaluate(() => !document.querySelector('.tool-bus')?.classList.contains('on')), 'Esc leaves the bus tool');

  // the T-1819 tool letters are gone
  for (const k of ['r', 'p', 'x', 'm']) await press(k);
  check(await toolOn() === null, 'R P X M no longer pick a tool');

  // S and G are line actions, not tabs: with no line selected they change nothing
  await press('b');
  await press('s');
  await press('g');
  check(await tabOn() === 'Build', 'S and G (edit stops, add passed) leave the tab alone with no line selected');

  // nothing fires while typing in a text box
  await press('b');
  await page.evaluate(() => {
    const i = document.createElement('input');
    i.type = 'text';
    i.id = 't1819-box';
    i.style.cssText = 'position:fixed;left:10px;top:10px;z-index:99999';
    document.body.appendChild(i);
    i.focus();
  });
  const before = { tab: await tabOn(), lv: await levelOn() };
  await page.keyboard.type('lafy5r1', { delay: 30 });
  const val = await page.evaluate(() => document.getElementById('t1819-box').value);
  const after = { tab: await tabOn(), lv: await levelOn(), tool: await toolOn() };
  check(val === 'lafy5r1' && after.tab === before.tab && after.lv === before.lv && after.tool === null, `typing in a text box changes nothing (typed ${val}, ${JSON.stringify(after)})`);

  check(errors.length === 0, 'no console errors' + (errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''));
} finally {
  await chrome.close();
}
process.exitCode = fail ? 1 : 0;
