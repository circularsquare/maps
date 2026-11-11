// T-3127: the ferry tool's port tag and snap. Clicks a little way inland and out on the water of New York's harbour
// find the shore at wide and close zooms (the reach is a fixed number of screen pixels, 200 m at least), the tag
// says "Place port" there and "Too far from water" past the reach, away from the city's pack (Lisbon, Hobart) the
// same, and the tag still comes when the style had no tile address yet at the tool's first use (the water worker
// once waited for ever for it). From the maps root, on a Vite of your own:
//   node trainworld/checks/t3127-ferryport.mjs [dev url]
// About 2 minutes.
import { collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6186/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

const WHITEHALL = [-74.0130, 40.7012];
const M = 1 / 111000;
const off = (p, east, north) => [p[0] + (east * M) / Math.cos((p[1] * Math.PI) / 180), p[1] + north * M];

const chrome = await launchChrome({ prefix: 'tw-t3127-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const page = await chrome.ctx.newPage();
collectErrors(page, errors);

const box = () => page.$('#map').then((m) => m.boundingBox());
const at = async (p) => { const b = await box(); const [x, y] = await page.evaluate(([a, c]) => { const q = window.tw.map.project([a, c]); return [q.x, q.y]; }, p); return { x: Math.round(b.x + x), y: Math.round(b.y + y) }; };
const settle = async () => { await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }); await page.waitForTimeout(500); };
const jump = async (center, zoom) => { await page.evaluate(([c, z]) => window.tw.map.jumpTo({ center: c, zoom: z }), [center, zoom]); await settle(); };
const tags = () => page.evaluate(() => [...document.querySelectorAll('.tool-tag')].filter((e) => e.style.display !== 'none').map((e) => e.innerText.replace(/\n/g, ' / ')).join(' // '));
const ports = () => page.evaluate(() => window.tw.tools.ferry.ports.length);

/** hover a point and wait for the tag to say something */
async function hover(p, want, timeout = 30000) {
  const q = await at(p);
  await page.mouse.move(q.x - 5, q.y - 5);
  await page.mouse.move(q.x, q.y, { steps: 3 });
  const t0 = Date.now();
  let tag = '';
  while (Date.now() - t0 < timeout) {
    tag = await tags();
    if (want.test(tag)) break;
    await page.waitForTimeout(250);
  }
  console.log(`      hover ${JSON.stringify(p)} -> "${tag}" after ${Date.now() - t0} ms`);
  return { tag, q };
}

try {
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=2&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools?.ferry, undefined, { timeout: 180000 });

  // the style gives no tile address for the first 4 s of the tool's life
  await page.evaluate(() => {
    const m = window.tw.map, real = m.getStyle.bind(m), t0 = Date.now();
    m.getStyle = () => (Date.now() - t0 < 4000 && new Error().stack.includes('roadTileTemplate') ? undefined : real());
  });
  await jump(WHITEHALL, 12);
  await page.click('button.tab[aria-label="Build"]');
  await page.click('button.tool-ferry');
  await page.waitForTimeout(300);
  let r = await hover(off(WHITEHALL, 0, -400), /Place port/, 40000);
  check(/Place port/.test(r.tag), 'the tag comes although the style had no tile address when the tool began', r.tag);

  // zoom 12 (about 30 m a pixel, reach 1200 m): out on the water, and inland
  r = await hover(off(WHITEHALL, 0, -400), /Place port/);
  check(/Place port/.test(r.tag), 'z12, 400 m out on the water: Place port', r.tag);
  r = await hover(off(WHITEHALL, 0, 400), /Place port/);
  check(/Place port/.test(r.tag), 'z12, 400 m inland: Place port', r.tag);
  await page.mouse.click(r.q.x, r.q.y);
  await page.waitForFunction(() => window.tw.tools.ferry.ports.length >= 1, undefined, { timeout: 30000 }).catch(() => {});
  check((await ports()) === 1, 'a click 400 m inland places a port', `${await ports()}`);
  const land = await page.evaluate(() => { const p = window.tw.tools.ferry.ports[0]; return p ? [p.landing.lng, p.landing.lat, p.berth.lng, p.berth.lat] : null; });
  const dMoved = land ? Math.hypot((land[0] - off(WHITEHALL, 0, 400)[0]) * 84000, (land[1] - off(WHITEHALL, 0, 400)[1]) * 111000) : -1;
  check(dMoved > 150 && dMoved < 1300, 'its landing is on the shore, a way from the click', `${Math.round(dMoved)} m`);
  await page.keyboard.press('Backspace');
  await page.waitForTimeout(300);

  // zoom 15 (about 3.6 m a pixel, reach 200 m)
  await jump(WHITEHALL, 15);
  r = await hover(off(WHITEHALL, 0, -150), /Place port/);
  check(/Place port/.test(r.tag), 'z15, 150 m out on the water: Place port', r.tag);
  r = await hover(off(WHITEHALL, 0, 150), /Place port/);
  check(/Place port/.test(r.tag), 'z15, 150 m inland: Place port', r.tag);
  r = await hover(off(WHITEHALL, 150, 700), /Too far/, 15000);
  check(/Too far from water/.test(r.tag), 'z15, past the reach inland: Too far from water', r.tag);
  await jump([-74.02, 40.50], 15); // Raritan Bay, over 4 km from either shore
  r = await hover([-74.02, 40.50], /Place ports on land/, 15000);
  check(/Place ports on land by the water/.test(r.tag), 'z15, open water far from land: Place ports on land by the water', r.tag);

  // away from any pack: the basemap's own water
  for (const [name, c] of [['Lisbon', [-9.1393, 38.7075]], ['Hobart', [147.3294, -42.8821]]]) {
    await jump(c, 13);
    r = await hover(off(c, 0, 0), /Place port|Too far|Place ports/, 40000);
    check(/Place port|Too far|Place ports/.test(r.tag), `${name}, far from the New York pack: the tag comes`, r.tag);
  }
  await jump([-9.1393, 38.7075], 13);
  const lis = await page.evaluate(async () => { const f = window.tw.tools.ferry; return await f.water.port([-9.1393, 38.7075 + 0.003], 800); });
  check(lis?.ok === true, 'Lisbon: a point 330 m off the Tagus snaps with a reach of 800 m', JSON.stringify(lis?.ok ?? lis));

  // touch: no hover tag (a tap places the port at once)
  check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} finally {
  await chrome.close();
}
console.log(fail ? `FAIL ${fail}` : 'PASS all');
process.exit(fail ? 1 : 0);
