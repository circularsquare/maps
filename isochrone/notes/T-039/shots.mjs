// 100 m windows on hilly towns, before and after T-039, 1400 x 900, with each L0.1 solve's time:
//   node isochrone/notes/T-039/shots.mjs <tag> [query]      e.g. before "fine=pre_t039"  /  after ""
// -> maps/artifacts/isochrone/t039_<place>_z<zoom>_<tag>.png
import { mkdtemp, rm, readdir } from 'node:fs/promises';
import { existsSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { pathToFileURL } from 'node:url';

const tag = process.argv[2] || 'test';
const query = process.argv[3] || '';
const OUT = 'C:/Users/anita/projects/maps/artifacts/isochrone/';
const SCR = tmpdir().split('\\').join('/') + '/iso-t039/';   // a private Chrome profile per run
// [name, start, view centre, zooms]: the start in the valley, the view on the slopes beside it
const PLACES = [['innsbruck', [11.393, 47.268], [11.40, 47.29], [9.5, 11.5]], ['cluj', [23.60, 46.77], [23.58, 46.76], [9.5, 11.5]],
    ['genoa', [8.93, 44.41], [8.95, 44.43], [11.5]]];
async function playwright() {
    const npx = join(process.env.LOCALAPPDATA ?? '', 'npm-cache', '_npx');
    for (const d of existsSync(npx) ? await readdir(npx) : []) {
        const p = join(npx, d, 'node_modules', 'playwright', 'index.mjs');
        if (existsSync(p)) return import(pathToFileURL(p).href);
    }
    throw new Error('no playwright');
}
mkdirSync(SCR + 'chrome', { recursive: true });
const profile = await mkdtemp(SCR + 'chrome/p-');
const { chromium } = await playwright();
const ctx = await chromium.launchPersistentContext(profile, {
    executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true,
    viewport: { width: 1400, height: 900 }, args: ['--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
try {
    const page = await ctx.newPage();
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await page.goto(`http://localhost:8800/isochrone/?${query ? query + '&' : ''}v=${Date.now()}`);
    await page.waitForFunction(() => window.solveLog?.some((e) => e.level === 10), undefined, { timeout: 90000, polling: 200 });
    const entry = async (level, from) => {
        await page.waitForFunction(([l, f]) => window.solveLog.slice(f).some((e) => e.level === l), [level, from], { timeout: level === 0.1 ? 20000 : 90000, polling: 200 });
        return page.evaluate(([l, f]) => window.solveLog.slice(f).find((e) => e.level === l), [level, from]);
    };
    for (const [name, at, centre, zooms] of PLACES) {
        let n = await page.evaluate(() => window.solveLog.length);
        await page.evaluate((c) => map.jumpTo({ center: c, zoom: 9 }), at);
        await page.waitForTimeout(500);
        const xy = await page.evaluate((c) => { const p = map.project(c); return [p.x, p.y]; }, at);
        await page.mouse.click(xy[0], xy[1]);
        await entry(10, n);
        for (const z of zooms) {
            n = await page.evaluate(() => window.solveLog.length);
            await page.evaluate(([c, zz]) => map.jumpTo({ center: c, zoom: zz }), [centre, z]);
            // the last L0.1 window is kept when it holds the new view: then no new entry
            const e = await entry(0.1, n).catch(() => null);
            await page.waitForTimeout(2500);
            await page.screenshot({ path: OUT + `t039_${name}_z${z}_${tag}.png` });
            if (e) console.log(`${name} z${z}: L0.1 ${e.cols} x ${e.rows} (${((e.cols * e.rows) / 1e6).toFixed(2)} M), fetch ${e.fetchMs} ms, solve ${e.solveMs} ms`);
            else console.log(`${name} z${z}: the L0.1 window kept`);
        }
    }
    // solve times at the cap: views 0.4 degree apart round Cluj at zoom 9.5, each a new window
    const times = [];
    for (let k = 0; k < 6; k++) {
        const n = await page.evaluate(() => window.solveLog.length);
        await page.evaluate((c) => map.jumpTo({ center: c, zoom: 9.5 }), [23.2 + 0.4 * (k % 3), 46.6 + 0.3 * Math.floor(k / 3)]);
        const e = await entry(0.1, n).catch(() => null);
        if (e) times.push(`${e.solveMs} ms (${((e.cols * e.rows) / 1e6).toFixed(2)} M)`);
    }
    console.log(`Cluj z9.5 solves: ${times.join(', ')}`);
    console.log(`errors: ${errors.length ? errors.join(' | ') : 'none'}`);
} finally {
    await ctx.close();
    await rm(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 }).catch(() => {});
}
process.exit(0);
