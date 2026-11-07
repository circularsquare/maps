// T-093: solve times on the page in headless Chrome, 1400 x 900, the heap search (served in place of
// js/search.js through a route) against the bucket queue, sessions alternating.
//   node isochrone/notes/T-093/chrome.mjs [rounds=2]   (search_heap_src.js as in bench.mjs)
// Each session: walking start at Cluj (L10), zoom 8 there (L1), six L0.1 windows at the cap round
// Cluj at zoom 9.5 (as notes/T-039/shots.mjs), then a canoe start at Constanta (L10) and zoom 8 (L1).
import { mkdtemp, rm, readdir } from 'node:fs/promises';
import { existsSync, mkdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { pathToFileURL, fileURLToPath } from 'node:url';

const SCR = fileURLToPath(new URL('./', import.meta.url)).replace(/\\/g, '/');
const ROUNDS = +(process.argv[2] || 2);
const HEAP = readFileSync(SCR + 'search_heap_src.js', 'utf8');
async function playwright() {
    const npx = join(process.env.LOCALAPPDATA ?? '', 'npm-cache', '_npx');
    for (const d of existsSync(npx) ? await readdir(npx) : []) {
        const p = join(npx, d, 'node_modules', 'playwright', 'index.mjs');
        if (existsSync(p)) return import(pathToFileURL(p).href);
    }
    throw new Error('no playwright');
}
const { chromium } = await playwright();
const PROF = tmpdir().split('\\').join('/') + '/iso-t093/';   // a private Chrome profile per session
mkdirSync(PROF, { recursive: true });
const URL_ = 'http://localhost:8800/isochrone/';
const CLUJ = [23.60, 46.77], CONSTANTA = [28.63, 44.17];

async function session(variant) {
    const profile = await mkdtemp(PROF + 'p-');
    const ctx = await chromium.launchPersistentContext(profile, {
        executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true,
        viewport: { width: 1400, height: 900 }, args: ['--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
    const out = { variant, l10: [], l1: [], l01: [], c10: [], c1: [], served: 0 };
    try {
        await ctx.route('**/isochrone/js/search.js*', (route) => {
            out.served++;
            if (variant === 'heap') return route.fulfill({ status: 200, contentType: 'text/javascript', body: HEAP });
            return route.continue();
        });
        const page = await ctx.newPage();
        const errors = [];
        page.on('pageerror', (e) => errors.push(e.message));
        const entry = async (level, from, timeout = 90000) => {
            await page.waitForFunction(([l, f]) => window.solveLog.slice(f).some((e) => e.level === l), [level, from], { timeout, polling: 100 });
            return page.evaluate(([l, f]) => window.solveLog.slice(f).find((e) => e.level === l), [level, from]);
        };
        const len = () => page.evaluate(() => window.solveLog.length);
        const click = async (at) => {
            const xy = await page.evaluate((c) => { const p = map.project(c); return [p.x, p.y]; }, at);
            await page.mouse.click(xy[0], xy[1]);
        };
        await page.goto(`${URL_}?v=${Date.now()}`);
        await entry(10, 0);
        let n = await len();
        await page.evaluate((c) => map.jumpTo({ center: c, zoom: 4 }), CLUJ);
        await page.waitForTimeout(300);
        await click(CLUJ);
        out.l10.push((await entry(10, n)).solveMs);
        n = await len();
        await page.evaluate((c) => map.jumpTo({ center: c, zoom: 8 }), CLUJ);
        let e = await entry(1, n);
        out.l1.push(`${e.solveMs} (${(e.cols * e.rows / 1e6).toFixed(2)} M)`);
        for (let k = 0; k < 6; k++) {
            n = await len();
            await page.evaluate((c) => map.jumpTo({ center: c, zoom: 9.5 }), [23.2 + 0.4 * (k % 3), 46.6 + 0.3 * Math.floor(k / 3)]);
            e = await entry(0.1, n, 30000).catch(() => null);
            if (e) out.l01.push(`${e.solveMs} (${(e.cols * e.rows / 1e6).toFixed(2)} M)`);
            await page.waitForTimeout(300);
        }
        await page.goto(`${URL_}?mode=canoe&v=${Date.now()}`);
        await entry(10, 0);
        n = await len();
        await page.evaluate((c) => map.jumpTo({ center: c, zoom: 4 }), CONSTANTA);
        await page.waitForTimeout(300);
        await click(CONSTANTA);
        e = await entry(10, n);
        out.c10.push(`${e.solveMs} ${e.mode}`);
        n = await len();
        await page.evaluate((c) => map.jumpTo({ center: c, zoom: 8 }), CONSTANTA);
        e = await entry(1, n);
        out.c1.push(`${e.solveMs} (${(e.cols * e.rows / 1e6).toFixed(2)} M)`);
        out.errors = errors;
    } finally {
        await ctx.close();
        await rm(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 }).catch(() => {});
    }
    return out;
}

for (let r = 0; r < ROUNDS; r++) {
    for (const v of r % 2 ? ['bucket', 'heap'] : ['heap', 'bucket']) {
        const o = await session(v);
        console.log(`${v.padEnd(6)} served ${o.served}: walk L10 ${o.l10.join(' ')} | L1 z8 ${o.l1.join(' ')} | L0.1 z9.5 ${o.l01.join(', ')} | canoe L10 ${o.c10.join(' ')} | canoe L1 z8 ${o.c1.join(' ')}${o.errors.length ? ' | errors: ' + o.errors.join(' | ') : ''}`);
    }
}
process.exit(0);
