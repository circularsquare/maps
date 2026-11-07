// Walking starts in mountains, today's tiles against the world T-011 build, 1400 x 900:
//   node isochrone/notes/T-011/shots.mjs <tag> [query]      e.g. before ""  /  after "tiles=t011"
// -> maps/artifacts/isochrone/t011w_<place>_z<zoom>_<tag>.png
import { mkdtemp, rm, readdir } from 'node:fs/promises';
import { existsSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { pathToFileURL } from 'node:url';

const tag = process.argv[2] || 'test';
const query = process.argv[3] || '';
const OUT = 'C:/Users/anita/projects/maps/artifacts/isochrone/';
const SCR = tmpdir().split('\\').join('/') + '/iso-t011w/';   // a private Chrome profile per run
const PLACES = [['brig', [7.99, 46.32], [6, 8]], ['cusco', [-71.97, -13.53], [6, 8]], ['kathmandu', [85.32, 27.71], [6, 8]]];
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
        await page.waitForFunction(([l, f]) => window.solveLog.slice(f).some((e) => e.level === l), [level, from], { timeout: 90000, polling: 200 });
        return page.evaluate(([l, f]) => window.solveLog.slice(f).find((e) => e.level === l), [level, from]);
    };
    for (const [name, at, zooms] of PLACES) {
        let n = await page.evaluate(() => window.solveLog.length);
        await page.evaluate((c) => map.jumpTo({ center: c, zoom: 5 }), at);
        await page.waitForTimeout(500);
        const xy = await page.evaluate((c) => { const p = map.project(c); return [p.x, p.y]; }, at);
        await page.mouse.click(xy[0], xy[1]);
        await entry(10, n);
        for (const z of zooms) {
            n = await page.evaluate(() => window.solveLog.length);
            await page.evaluate(([c, zz]) => map.jumpTo({ center: c, zoom: zz }), [at, z]);
            await entry(z >= 7 ? 1 : 3, n);
            await page.waitForTimeout(3000);
            await page.screenshot({ path: OUT + `t011w_${name}_z${z}_${tag}.png` });
        }
    }
    console.log(`errors: ${errors.length ? errors.join(' | ') : 'none'}`);
} finally {
    await ctx.close();
    await rm(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 }).catch(() => {});
}
process.exit(0);
