// T-1168 stage 3: screenshots of the line panel's related-lines view, light and dark, at a desktop
// size and a 390 px phone, into artifacts/trainworld/t1168s1/: the real save's 2, 6X and 4, a
// column's hover, and a new game's line beside its copy made into an express. Also checks the
// toggle, that a column click selects its line, and no console errors. From the maps root:
//   node trainworld/checks/t1168-shots.mjs <port>
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, REAL_SAVE, launchChrome, collectErrors, importSave, buildLine, startVite } from './lib.mjs';

const OUT = join(ARTIFACTS, 't1168s1');
await mkdir(OUT, { recursive: true });
const port = Number(process.argv[2] ?? 6168);
const vite = await startVite(port);
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };

async function session(size, name, run) {
  const chrome = await launchChrome({ prefix: 'tw-t1168s-', size });
  const errors = [];
  try {
    const page = await chrome.ctx.newPage();
    await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { usUnits: false }, v: 2 })); } catch {} });
    collectErrors(page, errors);
    await page.goto(`${vite.url}?debug=1&new=1&paused=1`);
    await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
    await run(page);
    check(errors.length === 0, `${name}: no console errors${errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''}`);
  } finally {
    await chrome.close();
  }
}

const shade = async (page, s) => { await page.evaluate((s) => { document.documentElement.dataset.shade = s; window.tw.shade.value = s; }, s); await page.waitForTimeout(300); };
const select = async (page, lineName) => {
  await page.evaluate((n) => { const l = window.tw.world.value.save.lines.find((x) => x.name === n); window.tw.selection.value = { kind: 'line', line: l.id }; }, lineName);
  await page.waitForSelector('button.related-toggle', { timeout: 30000 });
  if (!(await page.$('button.related-toggle.on'))) await page.click('button.related-toggle');
  await page.waitForSelector('.related .pill', { timeout: 30000 });
  await page.waitForTimeout(600);
};
/** the Stops heading down to the end of the drawing, the panel's width */
const shot = async (page, file) => {
  const h = await page.$('button.related-toggle');
  await h.evaluate((el) => el.closest('.h').scrollIntoView({ block: 'start' }));
  await page.waitForTimeout(250);
  const hb = await (await h.evaluateHandle((el) => el.closest('.h'))).boundingBox();
  const rb = await (await page.$('.related')).boundingBox();
  const vp = page.viewportSize();
  const bottom = Math.min(vp.height, rb.y + rb.height + 10);
  await page.screenshot({ path: join(OUT, file), clip: { x: Math.max(0, hb.x - 12), y: Math.max(0, hb.y - 8), width: Math.min(vp.width, hb.width + 24), height: bottom - Math.max(0, hb.y - 8) } });
};
const both = async (page, base) => { for (const s of ['light', 'dark']) { await shade(page, s); await page.mouse.move(2, 2); await shot(page, `${base}-${s}.png`); } await shade(page, 'light'); };

for (const [label, size] of [['desktop', [1400, 2000]], ['phone', [390, 2400]]]) {
  await session(size, `real save, ${label}`, async (page) => {
    await importSave(page, await readFile(REAL_SAVE));
    await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
    await page.waitForTimeout(1500);
    for (const n of label === 'desktop' ? ['subway 2', 'subway 6X', 'subway 4'] : ['subway 2', 'subway 6X']) {
      await select(page, n);
      await both(page, `${label}-${n.replace('subway ', '').toLowerCase()}`);
    }
    if (label !== 'desktop') return;
    // the 1's column beside the 2: its hover, then a click selects the 1
    await select(page, 'subway 2');
    const one = await page.evaluate(() => { const l = window.tw.world.value.save.lines.find((x) => x.name === 'subway 1'); return l.colour; });
    // a point on the column itself, between two stop pills
    const pt = await page.evaluate((c) => {
      const p = document.querySelector(`.related-draw g.link[data-tip^="subway 1 runs beside"] path[stroke="${c}"]:not([data-tip])`);
      p.scrollIntoView({ block: 'center' });
      const r = p.getBoundingClientRect();
      for (let y = r.y + r.height / 2; y < r.y + r.height; y += 2) for (let x = r.x - 3; x <= r.x + r.width + 3; x += 0.5) if (document.elementFromPoint(x, y) === p) return { x, y };
      return null;
    }, one);
    const box = { x: pt.x - 3, y: pt.y - 3, width: 6, height: 6 };
    await page.mouse.move(pt.x - 20, pt.y);
    await page.mouse.move(pt.x, pt.y, { steps: 4 });
    await page.waitForTimeout(700);
    const tag = await page.evaluate(() => { const t = document.querySelector('.tool-tag.ui-tag'); return t && t.style.display !== 'none' ? t.innerText : ''; });
    check(tag.includes('runs beside'), `the 1's column has its hover (${tag.replace(/\n/g, ' | ')})`);
    await page.screenshot({ path: join(OUT, 'desktop-2-hover.png'), clip: { x: Math.max(0, box.x - 120), y: Math.max(0, box.y - 120), width: 560, height: 260 } });
    await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
    await page.waitForTimeout(500);
    const now = await page.evaluate(() => { const s = window.tw.selection.value; return s?.kind === 'line' ? window.tw.world.value.save.lines.find((x) => x.id === s.line)?.name : null; });
    check(now === 'subway 1', `a click on the 1's column selects it (${now})`);
    check(!!(await page.$('button.related-toggle.on')), 'the toggle stays on for the next line');
    await page.click('button.related-toggle');
    await page.waitForTimeout(300);
    check(!(await page.$('.related')) && !!(await page.$('ul.stops')), 'toggled off: the plain list again');
  });
  await session(size, `copied express, ${label}`, async (page) => {
    const local = await buildLine(page, [[-73.99, 40.70], [-73.985, 40.705], [-73.98, 40.71], [-73.975, 40.715], [-73.97, 40.72], [-73.965, 40.725], [-73.96, 40.73]], { name: 'Local', colour: '#0a7d3b' });
    await page.evaluate(async (line) => {
      const tw = window.tw, c = tw.client;
      const stops = tw.world.value.save.lines.find((l) => l.num === line).stops;
      const names = ['Harbour', 'Mill Lane', 'Church St', 'Market', 'Park Av', 'Hill Rd', 'North End'];
      for (let k = 0; k < stops.length; k++) await c.edit({ op: 'stationProps', node: Number(stops[k]), platform: 200, name: names[k] });
      const copy = await c.edit({ op: 'copyLine', line });
      await c.edit({ op: 'setStops', line: copy.line, stops: [stops[0], stops[3], stops[6]] });
      await c.edit({ op: 'lineLook', line: copy.line, name: 'Express', colour: '#d9467a' });
    }, local.line);
    await page.waitForTimeout(800);
    await select(page, 'Local');
    await both(page, `${label}-copied-local`);
    await select(page, 'Express');
    await both(page, `${label}-copied-express`);
  });
}
vite.child.kill();
process.exit(fail ? 1 : 0);
