// T-3192: the game in Firefox with fingerprinting resistance on (LibreWolf's and Mullvad's default), which caps
// WebGL2 at its minimums and below (8 varyings, 2048 px textures and renderbuffers). Prints the limits the page
// sees, any startup failure and WebGL warnings; with the real save, pictures at three zooms.
//   node trainworld/checks/old/t3192-firefox-rfp.mjs <dev url> [rfp|min|plain] [--real]
// Needs Playwright's own Firefox: `npx playwright install firefox` (about 120 MB; delete it after).
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { ARTIFACTS, REAL_SAVE, playwright, tempProfile } from '../lib.mjs';

const [base = 'http://localhost:6118/', mode = 'rfp'] = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const real = process.argv.includes('--real');
// Playwright's Firefox asserts on Vite's HMR socket opening; harmless here
process.on('uncaughtException', (e) => { if (!/Assertion error/.test(String(e))) { console.error(e); process.exit(1); } });
const prefs = { 'webgl.force-enabled': true };
if (mode === 'rfp') prefs['privacy.resistFingerprinting'] = true;
if (mode === 'min') prefs['webgl.min_capability_mode'] = true;
const { firefox } = await playwright();
const profile = await tempProfile('tw-t3192-ff-');
const ctx = await firefox.launchPersistentContext(profile.dir, { headless: true, viewport: { width: 1400, height: 900 }, firefoxUserPrefs: prefs });
const page = ctx.pages()[0] ?? await ctx.newPage();
const logs = [];
page.on('console', (m) => { if (m.type() === 'error' || /WebGL|not drawn/.test(m.text())) logs.push(m.type() + ': ' + m.text().slice(0, 300)); });
page.on('pageerror', (e) => logs.push('pageerror: ' + String(e).slice(0, 300)));
await page.goto('about:blank');
console.log('limits', JSON.stringify(await page.evaluate(() => {
  const gl = document.createElement('canvas').getContext('webgl2');
  if (!gl) return 'no webgl2';
  return Object.fromEntries(['MAX_VARYING_VECTORS', 'MAX_VERTEX_TEXTURE_IMAGE_UNITS', 'MAX_COMBINED_TEXTURE_IMAGE_UNITS', 'MAX_TEXTURE_SIZE', 'MAX_RENDERBUFFER_SIZE', 'MAX_VERTEX_ATTRIBS', 'MAX_VERTEX_UNIFORM_VECTORS', 'MAX_FRAGMENT_UNIFORM_VECTORS'].map((k) => [k, gl.getParameter(gl[k])]));
})));
await page.goto(`${base}?debug=1&new=1&paused=1`);
await page.waitForFunction(() => window.tw?.world?.value || /stopped while starting|beim Start/.test(document.body.innerText), undefined, { timeout: 120000 });
const failed = await page.evaluate(() => document.body.innerText.match(/[^\n]*(stopped while starting|beim Start)[^\n]*/)?.[0] ?? null);
console.log(failed ? `FAIL  startup: ${failed}` : 'PASS  startup');
await page.screenshot({ path: join(ARTIFACTS, `t3192-firefox-${mode}.png`) });
if (real && !failed) {
  const bytes = Array.from(readFileSync(REAL_SAVE));
  await page.evaluate((data) => { const b = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes: b }, [b]); }, bytes);
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  for (const z of [12.5, 15.5, 17.6]) {
    await page.evaluate((z) => window.tw.map.jumpTo({ center: [-73.9873, 40.7553], zoom: z }), z);
    await page.waitForTimeout(6000);
    await page.screenshot({ path: join(ARTIFACTS, `t3192-firefox-${mode}-real-z${z}.png`) });
  }
}
for (const l of logs.slice(0, 30)) console.log(l);
await ctx.close();
