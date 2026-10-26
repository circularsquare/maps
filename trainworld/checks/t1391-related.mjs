// T-1391: the related-lines diagram on Anita's edge cases, in a new game: her Shanghai case (6
// beside the orange line from Dongfang Rd to N Nanquan Rd, ending there), a red line interlined
// for two stops, a green line beside the orange one past its terminus and ending mid-way, a line
// coming back to a station it already stopped at, a loop with a line beside it across its seam
// and one crossing it, and a local with its copy made an express. Then the real save's 2 and 6X.
// Screenshots light and dark, desktop and a 390 px phone, into artifacts/trainworld/t1391/ with a
// prefix (before, after). From the maps root:
//   node trainworld/checks/t1391-related.mjs <port> [prefix]
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, REAL_SAVE, launchChrome, collectErrors, importSave, startVite } from './lib.mjs';

const OUT = join(ARTIFACTS, 't1391');
await mkdir(OUT, { recursive: true });
const port = Number(process.argv[2] ?? 6139);
const prefix = process.argv[3] ?? 'after';
const only = process.argv[4] ?? '';
// T1391_URL: a server already running (an older build, for the before shots)
const vite = process.env.T1391_URL ? { url: process.env.T1391_URL, child: { kill() {} } } : await startVite(port);
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };

// ---- the networks, in metres east and north of an origin
const at = ([lng0, lat0]) => (dx, dy) => [lng0 + dx / (111320 * Math.cos((lat0 * Math.PI) / 180)), lat0 + dy / 110574];
const K = 675; // the orange line's station spacing, m
const sh = at([-73.975, 40.685]);
const hp = at([-73.94, 40.665]);
const lp = at([-73.90, 40.70]);
const ex = at([-73.98, 40.65]);
const polar = (f, r, deg) => f(r * Math.cos((deg * Math.PI) / 180), r * Math.sin((deg * Math.PI) / 180));
const ORANGE = ['Shiji Park', 'Century Av', 'Dongfang Rd', 'N Nanquan Rd', 'Lancun Rd', 'Tangqiao', 'Nanpu Bridge'];

/** Each spec: a line through points ({ p: [lng, lat], name } a new station, { p } a bare vertex,
 * { at: 'Line/stop' } an existing station of a line built before, { node: 'Line/stop' } routing
 * through it without making track); `stops` names its stops in order (default every station). */
const SPECS = [
  { name: '4', letter: '4', colour: '#f08a24', pts: ORANGE.map((n, k) => ({ p: sh(k * K, 0), name: n })) },
  { name: '6', letter: '6', colour: '#8e44ad', pts: [{ p: sh(0, 2300), name: 'Wulian Rd' }, { p: sh(K, 1300), name: 'Jufeng Rd' }, { p: sh(2 * K, 80), name: 'Dongfang Rd' }, { p: sh(3 * K, 80), name: 'N Nanquan Rd' }] },
  { name: '2', letter: '2', colour: '#d0312d', pts: [{ p: sh(2.2 * K, -560), name: 'Longyang Rd' }, { at: '4/4' }, { at: '4/5' }, { p: sh(6.8 * K, 560), name: 'Yuyuan Garden' }] },
  { name: '8', letter: '8', colour: '#2e9e4f', pts: [{ p: sh(4 * K, -80), name: 'Lancun Rd' }, { p: sh(5 * K, -80), name: 'Tangqiao' }, { p: sh(6 * K, -80), name: 'Nanpu Bridge' }, { p: sh(7 * K, -80), name: 'Expo Av' }, { p: sh(8 * K, -60), name: 'Houtan' }] },
  { name: 'Hook', letter: 'H', colour: '#1f6fb2', pts: [{ p: hp(-1200, 0), name: 'West Gate' }, { p: hp(0, 0), name: 'Union Sq' }, { p: hp(600, 0), name: 'Mill St' }, { p: hp(1000, 300) }, { p: hp(1000, 900), name: 'Upper Rd' }, { p: hp(600, 1200) }, { p: hp(0, 1200), name: 'Hilltop' }, { p: hp(-300, 900) }, { p: hp(-300, 300) }, { p: hp(-60, 60), name: 'Union Sq' }] },
  // the loop: 12 vertices, a station at every other one, closed back to the first
  { name: 'Ring', letter: 'R', colour: '#5b6770', loop: true, pts: Array.from({ length: 12 }, (_, k) => (k % 2 ? { p: polar(lp, 700, k * 30) } : { p: polar(lp, 700, k * 30), name: ['East End', 'North Gate', 'Hill Park', 'West End', 'South Gate', 'Lake Rd'][k / 2] })) },
  { name: 'Arc', letter: 'A', colour: '#c2185b', pts: [{ p: polar(lp, 1150, -110), name: 'Far South' }, { p: polar(lp, 780, -60), name: 'Lake Rd' }, { p: polar(lp, 780, -30) }, { p: polar(lp, 780, 0), name: 'East End' }, { p: polar(lp, 780, 30) }, { p: polar(lp, 780, 60), name: 'North Gate' }, { p: polar(lp, 1150, 110), name: 'Far North' }] },
  { name: 'Cross', letter: 'C', colour: '#e0a800', pts: [{ p: lp(-760, -900), name: 'Low Rd' }, { p: lp(-760, 0), name: 'West End' }, { p: lp(-760, 900), name: 'High Rd' }] },
  { name: 'Local', letter: 'L', colour: '#0a7d3b', pts: ['Harbour', 'Mill Lane', 'Church St', 'Market', 'Park Av', 'Hill Rd', 'North End'].map((n, k) => ({ p: ex(k * 600, k * 300), name: n })) },
];

async function buildAll(page) {
  return page.evaluate(async (SPECS) => {
    const tw = window.tw, c = tw.client;
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const pack = ([lng, lat]) => { const p = tw.lngLatToPack(lng, lat); return [p.x, p.y]; };
    const near = (xy) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - xy[0], n.y - xy[1]); if (d < bd) { bd = d; best = n.id; } } return best; };
    const lineStops = (name) => tw.world.value.save.lines.find((l) => l.name === name).stops;
    const log = [];
    for (const s of SPECS) try {
      const nodes = [];
      const ref = (q) => { const [ln, k] = q.split('/'); return Number(lineStops(ln)[Number(k)]); };
      const end = (q) => (q.at ? { kind: 'node', node: ref(q.at) } : { kind: 'free', x: pack(q.p)[0], y: pack(q.p)[1], level: -1 });
      for (let i = 1; i < s.pts.length; i++) {
        const q = s.pts[i];
        const from = i === 1 ? end(s.pts[0]) : { kind: 'node', node: nodes[i - 1] };
        // two stations of a line built before: on its track, nothing to build
        if (!(q.at && s.pts[i - 1].at)) {
          const r = await c.edit({ op: 'route', from, to: end(q), pis: [], single: false });
          if (r && r.ok === false) log.push(`${s.name}: route ${i} refused ${r.reason ?? r.error ?? ''}`);
          await pause(250);
        }
        if (i === 1) nodes.push(s.pts[0].at ? ref(s.pts[0].at) : near(pack(s.pts[0].p)));
        nodes.push(q.at ? ref(q.at) : near(pack(q.p)));
      }
      if (s.loop) {
        await c.edit({ op: 'route', from: { kind: 'node', node: nodes[nodes.length - 1] }, to: { kind: 'node', node: nodes[0] }, pis: [], single: false });
        await pause(250);
      }
      await pause(300);
      const stops = [];
      for (let i = 0; i < s.pts.length; i++) {
        const q = s.pts[i];
        if (q.at) { stops.push(nodes[i]); continue; }
        if (!q.name) continue;
        const r = await c.edit({ op: 'addStation', at: { kind: 'node', node: nodes[i] }, platform: 200, name: q.name });
        if (r && r.ok === false) log.push(`${s.name}: station ${q.name} at ${nodes[i]} refused ${JSON.stringify(r).slice(0, 200)}`);
        stops.push(nodes[i]);
        await pause(150);
      }
      await c.edit({ op: 'constructAll' });
      await pause(300);
      if (s.loop) stops.push(stops[0]);
      const r = await c.edit({ op: 'addLine', stops, name: s.name, colour: s.colour });
      if (!r?.ok) { log.push(`${s.name}: line refused ${JSON.stringify(r).slice(0, 300)} stops ${JSON.stringify(stops)}`); continue; }
      await c.edit({ op: 'lineLook', line: r.line, name: s.name, colour: s.colour, letter: s.letter });
      await pause(300);
      const l = tw.world.value.save.lines.find((x) => x.num === r.line);
      log.push(`${s.name}: ${l.stops.length} stops, ${tw.world.value.lineStats[l.id]?.status}${tw.world.value.lineStats[l.id]?.loop ? ' ' + tw.world.value.lineStats[l.id].loop : ''}`);
    } catch (e) { log.push(`${s.name}: ${e.message}`); }
    if (!tw.world.value.save.lines.some((l) => l.name === 'Local')) return log;
    // the express: the local's copy stopping at three of its seven
    const local = tw.world.value.save.lines.find((l) => l.name === 'Local');
    const copy = await c.edit({ op: 'copyLine', line: local.num });
    await c.edit({ op: 'setStops', line: copy.line, stops: [local.stops[0], local.stops[3], local.stops[6]].map(Number) });
    await c.edit({ op: 'lineLook', line: copy.line, name: 'Express', colour: '#d9467a', letter: 'X' });
    await pause(500);
    return log;
  }, SPECS);
}

async function session(size, name, run) {
  const chrome = await launchChrome({ prefix: 'tw-t1391-', size, deviceScaleFactor: 2 });
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

const shade = async (page, s) => { await page.evaluate((s) => { document.documentElement.dataset.shade = s; window.tw.shade.value = s; }, s); await page.waitForTimeout(250); };
const select = async (page, lineName, related = true) => {
  await page.evaluate((n) => { const l = window.tw.world.value.save.lines.find((x) => x.name === n); window.tw.selection.value = { kind: 'line', line: l.id }; }, lineName);
  await page.waitForSelector('button.related-toggle', { timeout: 30000 });
  const on = !!(await page.$('button.related-toggle.on'));
  if (on !== related) await page.click('button.related-toggle');
  if (related) await page.waitForSelector('.related .pill', { timeout: 30000 });
  await page.waitForTimeout(600);
};
/** the Stops heading down to the end of the stops, the panel's width */
const shot = async (page, file) => {
  const h = await page.$('button.related-toggle');
  await h.evaluate((el) => el.closest('.h').scrollIntoView({ block: 'start' }));
  await page.waitForTimeout(200);
  const hb = await (await h.evaluateHandle((el) => el.closest('.h'))).boundingBox();
  const rb = await (await page.$('.related') ?? await page.$('ul.stops')).boundingBox();
  const vp = page.viewportSize();
  const bottom = Math.min(vp.height, rb.y + rb.height + 10);
  await page.screenshot({ path: join(OUT, file), clip: { x: Math.max(0, hb.x - 12), y: Math.max(0, hb.y - 8), width: Math.min(vp.width, hb.width + 24), height: bottom - Math.max(0, hb.y - 8) } });
};
const both = async (page, base) => { for (const s of ['light', 'dark']) { await shade(page, s); await page.mouse.move(2, 2); await shot(page, `${prefix}-${base}-${s}.png`); } await shade(page, 'light'); };

/** what the panel draws now: columns and bars by colour, row names, bullets by row */
const drawn = (page) => page.evaluate(() => {
  const cols = [...document.querySelectorAll('.related-draw g.link path:last-child')].map((p) => p.getAttribute('stroke'));
  const bars = [...document.querySelectorAll('.related-draw > g > line')].map((p) => p.getAttribute('stroke'));
  const rows = [...document.querySelectorAll('ul.stops > li')].filter((li) => li.querySelector('.stop-name')).map((li) => ({
    name: li.querySelector('.stop-name').textContent.trim(),
    chips: [...li.querySelectorAll(':scope > .chip')].map((c) => c.getAttribute('data-tip')),
    time: !!li.querySelector('.t') && getComputedStyle(li.querySelector('.t')).display !== 'none',
  }));
  return { cols, bars, rows };
});
const colourOf = (page, n) => page.evaluate((n) => window.tw.world.value.save.lines.find((x) => x.name === n).colour, n);

for (const [label, size] of [['desktop', [1400, 2000]], ['phone', [390, 2400]]]) {
  if (only && only !== label) continue;
  await session(size, `edge cases, ${label}`, async (page) => {
    const log = await buildAll(page);
    for (const x of log) console.log('  ' + x);
    if (log.some((x) => /refused|Cannot/.test(x))) throw new Error('the networks did not build');
    await page.waitForTimeout(800);
    for (const n of ['4', '6', '2', '8', 'Hook', 'Ring', 'Arc', 'Local', 'Express']) {
      await select(page, n);
      await both(page, `${label}-${n.toLowerCase()}`);
      const d = await drawn(page);
      if (label !== 'desktop') continue;
      const has = async (m) => d.cols.includes(await colourOf(page, m));
      const bar = async (m) => d.bars.includes(await colourOf(page, m));
      if (n === '4') {
        check(await has('6') && !(await bar('6')), 'the 4: the 6 beside it, no crossing bar (points 1, 2)');
        check(await has('2') && !(await bar('2')), 'the 4: the 2 interlined for two stops beside it');
        check(await has('8'), 'the 4: the 8 beside it');
        const r = (nm) => d.rows.find((x) => x.name === nm);
        check(r('Dongfang Rd')?.chips.includes('6') && r('N Nanquan Rd')?.chips.includes('6'), `the 4: the 6's bullet at both ends of the shared stretch (${JSON.stringify(d.rows.map((x) => [x.name, x.chips]))})`);
        // the 2's turn where it leaves has its own hover
        const red = await colourOf(page, '2');
        const pt = await page.evaluate((c) => {
          const ends = [...document.querySelectorAll(`.related-draw g.link path[stroke="${c}"][data-tip]`)];
          const p = ends.at(-1), q = p.getPointAtLength(3), m = p.getScreenCTM();
          return { x: m.a * q.x + m.e, y: m.d * q.y + m.f, n: ends.length };
        }, red);
        await page.mouse.move(pt.x + 10, pt.y + 10);
        await page.mouse.move(pt.x, pt.y, { steps: 3 });
        await page.waitForTimeout(600);
        const tag = await page.evaluate(() => { const t = document.querySelector('.tool-tag.ui-tag'); return t && t.style.display !== 'none' ? t.innerText : ''; });
        check(pt.n === 2 && tag.includes('2 goes on to Yuyuan Garden'), `the 4: the 2 turns off at both ends, the lower turn's hover (${tag.replace(/\n/g, ' | ')})`);
        await page.mouse.move(2, 2);
      }
      if (n === '6') check(await has('4') && !(await bar('4')), 'the 6: the 4 beside it to N Nanquan Rd');
      if (n === 'Ring') check(d.rows.filter((x) => x.name === 'East End').length === 1, `the loop: every station once (${d.rows.map((x) => x.name).join(', ')})`);
      if (n === 'Hook') check(d.rows.find((x) => x.name === 'Union Sq')?.chips.includes('Hook'), 'the hook: itself a transfer at Union Sq');
    }
    // related lines stay while the express's stops are edited, with the stations it passes between
    await select(page, 'Express');
    await page.click('.line-stop-actions button:first-child');
    await page.waitForSelector('.related li.passed', { timeout: 30000 });
    await page.waitForTimeout(600);
    await both(page, `${label}-express-editing`);
    if (label === 'desktop') {
      const al = await page.evaluate(() => {
        const ul = document.querySelector('.related ul.stops');
        const pills = [...document.querySelectorAll('.related-draw .pill')].map((p) => p.getBoundingClientRect());
        const rows = [...ul.querySelectorAll(':scope > li[data-r]')].map((li) => li.getBoundingClientRect());
        // every row's centre has a pill on it
        return rows.every((r) => pills.some((p) => Math.abs(p.y + p.height / 2 - (r.y + r.height / 2)) < 1.5));
      });
      check(al, 'editing stops: the related lines stay, each row on its pill with the passed rows between');
    }
    await page.click('.line-stop-actions button:first-child');
    await page.waitForTimeout(400);
    // with related lines off: the loop once round, the hook its own transfer
    for (const n of ['Hook', 'Ring']) {
      await select(page, n, false);
      await both(page, `${label}-${n.toLowerCase()}-plain`);
      const d = await drawn(page);
      if (label !== 'desktop') continue;
      if (n === 'Hook') check(d.rows.filter((x) => x.name === 'Union Sq').every((x) => x.chips.includes('Hook')), 'related off: the hook its own transfer at Union Sq (point 7)');
      if (n === 'Ring') check(d.rows.filter((x) => x.name === 'East End').length === 1, 'related off: the loop once round');
    }
  });
  await session(size, `real save, ${label}`, async (page) => {
    await importSave(page, await readFile(REAL_SAVE));
    await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
    await page.waitForTimeout(1500);
    for (const n of label === 'desktop' ? ['subway 2', 'subway 6X', 'subway 4'] : ['subway 2', 'subway 6X']) {
      await select(page, n);
      await both(page, `${label}-real-${n.replace('subway ', '').toLowerCase()}`);
    }
  });
}
vite.child.kill();
process.exit(fail ? 1 : 0);
