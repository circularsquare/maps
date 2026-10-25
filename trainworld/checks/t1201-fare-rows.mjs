// T-1201, T-1234: the Money tab's fare rows as rates, at the dock's narrowest width (280 px). On a
// new game (default nodes 0, 20 and 100 km on $1.50 + $0.10 a km): the rows read "0-[20]km +$0.10 per
// km" and "Past 20 km" (the end distance a typed box with no steps, the last row's none); no row
// wraps; no "10 km ride" line; a rate's step goes to the next whole cent ($0.138 to $0.14 or $0.13);
// stepping or typing a stretch's rate moves its node's fare and every later node's by the same
// amount; a rate never goes below 0; the base fare's step moves every node; a typed stretch end keeps
// both rates; a drag on the graph still sets a fare and the rows follow. Screenshots of the graph and
// rows, light and dark, go to artifacts/trainworld/t1201/.
//   node trainworld/checks/t1201-fare-rows.mjs [url]   (from the maps root; default localhost:6149)
// Prints PASS or FAIL lines and exits 1 on failure. About a minute.
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { launchChrome, ARTIFACTS } from './lib.mjs';

const BASE = (process.argv[2] ?? 'http://localhost:6149/').replace(/\/?$/, '/');
const OUT = join(ARTIFACTS, 't1201');
const DEFAULT = '[[0,1.5],[20,3.5],[100,11.5]]';

let failed = false;
const report = (name, ok, detail) => { if (!ok) failed = true; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };

await mkdir(OUT, { recursive: true });
const { ctx, close } = await launchChrome({ prefix: 'tw-t1201-' });
try {
  const page = await ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { usUnits: false }, dock: { dockW: 280, paneH: 760 }, v: 2 })); } catch {} });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.moneyState?.value, undefined, { timeout: 180000 });

  const nodesNow = () => page.evaluate(() => JSON.stringify(window.tw.moneyState.value.fares.nodes));
  const waitNodes = (want) => page.waitForFunction((want) => JSON.stringify(window.tw.moneyState.value.fares.nodes) === want, want, { timeout: 30000 }).catch(() => {});
  const waitChange = (before) => page.waitForFunction((b) => JSON.stringify(window.tw.moneyState.value.fares.nodes) !== b, before, { timeout: 30000 }).catch(() => {});
  const setNodes = async (nodes) => { await page.evaluate((nodes) => window.tw.client.edit({ op: 'fares', nodes }), nodes); await waitNodes(JSON.stringify(nodes)); await page.waitForTimeout(200); };
  // each row's text with its spaces dropped ("0-20km+$0.10perkm")
  const rowsText = () => page.$$eval('ul.fare-rows > li', (ls) => ls.map((l) => l.textContent.replace(/\s+/g, '')));
  const click = async (sel) => { const before = await nodesNow(); await page.click(sel); await waitChange(before); await page.waitForTimeout(150); return JSON.parse(await nodesNow()); };
  const type = async (label, text) => {
    const before = await nodesNow();
    await page.click(`button[aria-label="Type ${label}"]`);
    await page.fill(`input[aria-label="${label}"]`, text);
    await page.keyboard.press('Enter');
    await waitChange(before);
    await page.waitForTimeout(150);
    return JSON.parse(await nodesNow());
  };
  const near = (a, b) => Math.abs(a - b) < 1e-6;

  await page.click('button.tab[aria-label="Money"]');
  await waitNodes(DEFAULT);
  const rows = await rowsText();
  report('rows read as rates', rows[1] === '0-20km+$0.10perkm' && rows[2] === 'Past20km+$0.10perkm'
    && !(await page.$('button[aria-label="More end of stretch 1"]')) && !!(await page.$('button[aria-label="Type end of stretch 1"]'))
    && !(await page.$('button[aria-label="Type end of stretch 2"]')), `(${JSON.stringify(rows)})`);
  // no row wraps: every cell one line high, the dock at its narrowest
  const fit = await page.evaluate(() => {
    const ul = document.querySelector('ul.fare-rows');
    const cells = [...ul.querySelectorAll(':scope > li > *')].map((c) => Math.round(c.getBoundingClientRect().height));
    const pane = document.querySelector('.pane').getBoundingClientRect().width;
    return { pane: Math.round(pane), cells, over: ul.scrollWidth > ul.clientWidth };
  });
  report('no row wraps at the narrowest dock', fit.cells.every((hh) => hh <= 26) && !fit.over, `(pane ${fit.pane} px, cell heights ${fit.cells.join(',')}, overflow ${fit.over})`);
  const ride = await page.evaluate(() => /\d+ ?(km|mi) ride/.test(document.querySelector('.pane').textContent));
  report('no 10 km ride line', !ride, '');

  const shade = async (s) => { await page.evaluate((s) => { document.documentElement.dataset.shade = s; window.tw.shade.value = s; }, s); await page.waitForTimeout(300); };
  const shot = async (name) => {
    const g = await page.$('svg.fare-graph');
    await g.evaluate((el) => el.scrollIntoView({ block: 'start' }));
    await page.waitForTimeout(200);
    const box = await g.boundingBox();
    const list = await (await page.$('ul.fare-rows')).boundingBox();
    await page.screenshot({ path: join(OUT, `${name}.png`), clip: { x: box.x - 14, y: box.y - 30, width: box.width + 28, height: list.y + list.height - box.y + 38 } });
  };

  // whole cents: $0.138 steps to $0.14 and to $0.13
  await setNodes([[0, 1.5], [20, 4.26], [100, 11.5]]);
  const odd = (await rowsText())[1];
  await both();
  const upOdd = await click('button[aria-label="More rate of stretch 1"]');
  await setNodes([[0, 1.5], [20, 4.26], [100, 11.5]]);
  const downOdd = await click('button[aria-label="Less rate of stretch 1"]');
  report('a rate steps to the next whole cent', odd === '0-20km+$0.138perkm' && near(upOdd[1][1], 4.3) && near(downOdd[1][1], 4.1), `(${odd}; up ${JSON.stringify(upOdd)}, down ${JSON.stringify(downOdd)})`);

  // the first stretch's rate up a step: the 20 km node up $0.20, the 100 km node too
  await setNodes(JSON.parse(DEFAULT));
  const up = await click('button[aria-label="More rate of stretch 1"]');
  report('a rate step moves its node and every later one the same', near(up[1][1], 3.7) && near(up[2][1], 11.7) && near(up[0][1], 1.5), `(${JSON.stringify(up)})`);
  // typed: the second stretch at $0.02 a km
  const typed = await type('rate of stretch 2', '0.02');
  report('a typed rate sets the stretch\'s slope', near(typed[2][1], 3.7 + 0.02 * 80) && near(typed[1][1], 3.7), `(${JSON.stringify(typed)})`);
  // typed below 0: held at 0
  const zero = await type('rate of stretch 2', '-1');
  report('a rate never goes below 0', near(zero[2][1], zero[1][1]), `(${JSON.stringify(zero)})`);
  await setNodes([[0, 1.5], [20, 2.5], [100, 4.1]]);
  // the base fare's step keeps both rates
  const based = await click('button[aria-label="More base fare"]');
  report('the base fare\'s step moves every node', near(based[0][1], 1.75) && near(based[1][1], 2.75) && near(based[2][1], 4.35), `(${JSON.stringify(based)})`);
  // the first stretch's end typed out to 25 km: rates kept, so the 25 km node is at 1.75 + 25 x 0.05
  const moved = await type('end of stretch 1', '25');
  const pastRow = (await rowsText())[2];
  report('a stretch\'s end keeps both rates, the last row following it', moved[1][0] === 25 && near(moved[1][1], 3) && near(moved[2][1], 3 + 75 * 0.02) && pastRow?.startsWith('Past25km'), `(${JSON.stringify(moved)}, ${JSON.stringify(pastRow)})`);
  // a rate already 0 does not step lower
  await setNodes([[0, 1.5], [20, 3.5], [100, 3.5]]);
  const before = await nodesNow();
  await page.click('button[aria-label="Less rate of stretch 2"]');
  await page.waitForTimeout(800);
  report('a rate of 0 steps no lower', (await nodesNow()) === before, `(${await nodesNow()})`);

  // a drag on the graph still sets a fare: the third node up, and its row's rate follows
  await setNodes(JSON.parse(DEFAULT));
  const k = (await page.$$('svg.fare-graph .fg-knob'))[2];
  const bb = await k.boundingBox();
  const x = bb.x + bb.width / 2, y = bb.y + bb.height / 2;
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.move(x, y - 4, { steps: 2 });
  await page.mouse.move(x, y - 30, { steps: 6 });
  await page.mouse.up();
  await waitChange(DEFAULT);
  await page.waitForTimeout(200);
  const dragged = JSON.parse(await nodesNow());
  const r2 = (dragged[2][1] - dragged[1][1]) / (dragged[2][0] - dragged[1][0]);
  const r2Text = Math.abs(r2 * 100 - Math.round(r2 * 100)) < 1e-6 ? r2.toFixed(2) : r2.toFixed(3);
  const rows2 = await rowsText();
  report('a drag sets a fare and the row\'s rate follows', dragged[2][0] === 100 && dragged[2][1] > 11.5 && near(dragged[1][1], 3.5) && rows2[2]?.includes(`+$${r2Text}perkm`), `(${JSON.stringify(dragged)}, ${JSON.stringify(rows2[2])})`);
  report('no page errors', errors.length === 0, errors.slice(0, 3).join(' | '));

  async function both() {
    // parked over the map, away from any hover
    await page.mouse.move(900, 800);
    for (const s of ['light', 'dark']) { await shade(s); await shot(`fares-${s}`); }
    await shade('light');
    await page.hover('button[aria-label="Type rate of stretch 1"]');
    await page.waitForTimeout(600);
    await shot('fares-rate-hover');
    await page.mouse.move(900, 800);
  }
} finally {
  await close();
}
process.exit(failed ? 1 : 0);
