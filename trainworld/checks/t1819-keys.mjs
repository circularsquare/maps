// T-1819: the key map. Presses each key in a new game and reads which dock tab, tool and elevation
// button is on: 1 to 7 elevation, B L A F Y tabs, S edit stops, G add passed, R P X M U tools, nothing while typing in a text
// box, Shift+digit not an elevation, hovers showing the keys. From the maps root:
//   node trainworld/checks/t1819-keys.mjs [url]   (about 30 seconds)
import { launchChrome, collectErrors, GL } from './lib.mjs';

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
  const levelOn = () => page.evaluate(() => [...document.querySelectorAll('.levels button')].findIndex((b) => b.classList.contains('on')));
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
    await press(String(n));
    const i = await levelOn();
    check(i === n - 1, `key ${n} sets elevation button ${n} (on: ${i + 1})`);
  }
  await press('Shift+3');
  check(await levelOn() === 6, 'Shift+3 leaves the elevation alone');
  await press('l');
  await press('2');
  await press('b');
  check(await levelOn() === 1, 'a digit sets the elevation from the Lines tab');
  const lvTips = await page.evaluate(() => [...document.querySelectorAll('.levels button')].map((b) => b.getAttribute('data-tip')));
  console.log('elevation hovers', JSON.stringify(lvTips));
  check(lvTips.every((t, i) => t.endsWith(`(${i + 1})`)), 'elevation hovers end in the number key');

  // tools, from another tab; the Build tab opens
  for (const [k, name] of [['r', 'Track'], ['p', 'Station'], ['x', 'Delete'], ['m', 'Select']]) {
    await press('l');
    await press(k);
    const t = await tabOn(), got = await toolOn();
    check(t === 'Build' && got === name, `key ${k.toUpperCase()} picks ${name} and opens Build (tab ${t}, tool ${got})`);
    await press('Escape');
  }
  const toolTips = await page.evaluate(() => [...document.querySelectorAll('.tool-pick button')].map((b) => [b.getAttribute('data-tip'), b.querySelector('.key')?.textContent]));
  console.log('tool hovers', JSON.stringify(toolTips));
  check(['R', 'P', 'X', 'M', 'U'].every((k, i) => toolTips[i][1] === k && toolTips[i][0].includes(`(${k})`)), 'tool buttons show their letters in label and hover');

  // the bus tool
  await press('Escape');
  await press('u');
  const bus = await page.evaluate(() => document.querySelector('.tool-bus')?.classList.contains('on'));
  check(bus === true, 'key U starts the bus tool');
  await press('Escape');
  await press('Escape');
  check(await page.evaluate(() => !document.querySelector('.tool-bus')?.classList.contains('on')), 'Esc leaves the bus tool');

  // the old number tool keys and the old letters
  await press('3');
  check(await toolOn() === null, 'key 3 no longer picks a tool');

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
  await page.keyboard.type('lafy5r', { delay: 30 });
  const val = await page.evaluate(() => document.getElementById('t1819-box').value);
  const after = { tab: await tabOn(), lv: await levelOn(), tool: await toolOn() };
  check(val === 'lafy5r' && after.tab === before.tab && after.lv === before.lv && after.tool === null, `typing in a text box changes nothing (typed ${val}, ${JSON.stringify(after)})`);

  check(errors.length === 0, 'no console errors' + (errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''));
} finally {
  await chrome.close();
}
process.exitCode = fail ? 1 : 0;
