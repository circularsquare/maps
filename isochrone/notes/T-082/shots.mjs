// Canoe start at Coari, zoom 4 and 7, 1400 x 900: node shots.mjs <tag>
// -> maps/artifacts/isochrone/t082_coari_z4_<tag>.png, t082_coari_z7_<tag>.png
import { mkdtemp, rm, readdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const tag = process.argv[2] || 'test';
const OUT = 'C:/Users/anita/projects/maps/artifacts/isochrone/';
const COARI = [-63.14, -4.08];
async function playwright() {
    const npx = join(process.env.LOCALAPPDATA ?? '', 'npm-cache', '_npx');
    for (const d of existsSync(npx) ? await readdir(npx) : []) {
        const p = join(npx, d, 'node_modules', 'playwright', 'index.mjs');
        if (existsSync(p)) return import(pathToFileURL(p).href);
    }
    throw new Error('no playwright');
}
const profile = await mkdtemp(join(tmpdir(), 'iso-t082-'));
const { chromium } = await playwright();
const ctx = await chromium.launchPersistentContext(profile, {
    executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true,
    viewport: { width: 1400, height: 900 }, args: ['--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
try {
    const page = await ctx.newPage();
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await page.goto(`http://localhost:8800/isochrone/?mode=canoe&v=${Date.now()}`);
    await page.waitForFunction(() => window.solveLog?.some((e) => e.level === 10), undefined, { timeout: 90000, polling: 200 });
    const entry = async (level, from) => {
        await page.waitForFunction(([l, f]) => window.solveLog.slice(f).some((e) => e.level === l), [level, from], { timeout: 90000, polling: 200 });
        return page.evaluate(([l, f]) => window.solveLog.slice(f).find((e) => e.level === l), [level, from]);
    };
    let n = await page.evaluate(() => window.solveLog.length);
    await page.evaluate((c) => map.jumpTo({ center: c, zoom: 4 }), COARI);
    await page.waitForTimeout(500);
    const xy = await page.evaluate((c) => { const p = map.project(c); return [p.x, p.y]; }, COARI);
    await page.mouse.click(xy[0], xy[1]);
    const e10 = await entry(10, n);
    await page.waitForTimeout(2500);
    await page.screenshot({ path: OUT + `t082_coari_z4_${tag}.png` });
    n = await page.evaluate(() => window.solveLog.length);
    await page.evaluate((c) => map.jumpTo({ center: c, zoom: 7 }), COARI);
    const e1 = await entry(1, n);
    await page.waitForTimeout(3000);
    await page.screenshot({ path: OUT + `t082_coari_z7_${tag}.png` });
    console.log(`L10 solve ${e10.solveMs} ms, L1 window ${e1.cols} x ${e1.rows} solve ${e1.solveMs} ms; errors: ${errors.length ? errors.join(' | ') : 'none'}`);
} finally {
    await ctx.close();
    await rm(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 }).catch(() => {});
}
process.exit(0);
