// Mountain places that get 100 m only with the mountain tiles, 1400 x 900, zoom 11.5:
//   node isochrone/notes/T-057/shots.mjs <tag> [query]      e.g. before ""  /  after "fine=t057"
// -> maps/artifacts/isochrone/t057_<place>_z11.5_<tag>.png, and the finest level shown
import { mkdtemp, rm, readdir } from 'node:fs/promises';
import { existsSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { pathToFileURL } from 'node:url';

const tag = process.argv[2] || 'test';
const query = process.argv[3] || '';
const OUT = 'C:/Users/anita/projects/maps/artifacts/isochrone/';
const SCR = tmpdir().split('\\').join('/') + '/iso-t057/';   // a private Chrome profile per run
// [name, start, view centre]
const PLACES = [['chamonix', [6.87, 45.92], [6.88, 45.90]], ['zermatt', [7.75, 46.02], [7.74, 46.00]]];
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
    const entry = async (level, from, timeout) => {
        await page.waitForFunction(([l, f]) => window.solveLog.slice(f).some((e) => e.level === l), [level, from], { timeout, polling: 200 });
        return page.evaluate(([l, f]) => window.solveLog.slice(f).find((e) => e.level === l), [level, from]);
    };
    for (const [name, at, centre] of PLACES) {
        let n = await page.evaluate(() => window.solveLog.length);
        await page.evaluate((c) => map.jumpTo({ center: c, zoom: 9 }), at);
        await page.waitForTimeout(500);
        const xy = await page.evaluate((c) => { const p = map.project(c); return [p.x, p.y]; }, at);
        await page.mouse.click(xy[0], xy[1]);
        await entry(10, n, 90000);
        n = await page.evaluate(() => window.solveLog.length);
        await page.evaluate((c) => map.jumpTo({ center: c, zoom: 11.5 }), centre);
        const e = await entry(0.1, n, 20000).catch(() => null);
        await page.waitForTimeout(4000);
        await page.screenshot({ path: OUT + `t057_${name}_z11.5_${tag}.png` });
        console.log(`${name}: ${e ? `L0.1 ${e.cols} x ${e.rows}, fetch ${e.fetchMs} ms, solve ${e.solveMs} ms` : 'no L0.1 window'}`);
    }
    console.log(`errors: ${errors.length ? errors.join(' | ') : 'none'}`);
} finally {
    await ctx.close();
    await rm(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 }).catch(() => {});
}
process.exit(0);
