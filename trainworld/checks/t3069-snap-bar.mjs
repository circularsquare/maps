// T-3069: the track bar's snap switches, rail and tram, at desktop and phone width: screenshots of the bar,
// and the switches' exclusions (tram's real snap and Snap to streets turn each other off). From the maps root:
//   node trainworld/checks/t3069-snap-bar.mjs [url] [out dir]
import { launchChrome, collectErrors, ARTIFACTS } from './lib.mjs';

const url = process.argv[2] ?? 'http://localhost:6145/';
const out = process.argv[3] ?? ARTIFACTS;
const errors = [];
let fails = 0;
const ok = (cond, what) => { console.log((cond ? 'ok   ' : 'FAIL ') + what); if (!cond) fails++; };
for (const [w, h, tag] of [[1400, 900, 'desktop'], [390, 844, 'phone']]) {
  const chrome = await launchChrome({ prefix: 'tw-t3069-', size: [w, h], ...(tag === 'phone' ? { isMobile: true, hasTouch: true, deviceScaleFactor: 2 } : { deviceScaleFactor: 2 }) });
  try {
    const page = await chrome.ctx.newPage();
    collectErrors(page, errors);
    await page.goto(`${url}?debug=1&new=1&demandWorkers=1`);
    await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });
    for (const type of [2, 'tram']) {
      await page.evaluate(async (type) => {
        const ui = await import('/src/game/ui.ts');
        ui.tool.value = 'track';
        ui.setTrackType(type);
        window.tw.map.jumpTo({ center: [4.9, 52.37], zoom: 14 });
      }, type);
      await page.waitForTimeout(1500);
      const bar = page.locator('.track-bar');
      const box = await bar.boundingBox();
      const pad = 12;
      const name = `t3069_bar_${tag}_${type === 'tram' ? 'tram' : 'rail'}.png`;
      await page.screenshot({ path: out + name, clip: { x: Math.max(0, box.x - pad), y: Math.max(0, box.y - pad), width: Math.min(w - Math.max(0, box.x - pad), box.width + 2 * pad), height: box.height + 2 * pad } });
      console.log('     ', name, JSON.stringify(box));
    }
    // exclusions, with Tram picked
    const r = await page.evaluate(async () => {
      const ui = await import('/src/game/ui.ts');
      const click = (label) => [...document.querySelectorAll('.track-bar button')].find((b) => b.getAttribute('aria-label') === label)?.click();
      const real = document.querySelector('.track-bar .build-toggles button:nth-child(2)')?.getAttribute('aria-label');
      const before = { streets: ui.snapStreets.value, real: ui.snapRealTram.value, rail: ui.snapReal.value };
      click(real);
      await new Promise((r) => setTimeout(r, 50));
      const afterReal = { streets: ui.snapStreets.value, real: ui.snapRealTram.value, mode: ui.snapMode.value, rail: ui.snapReal.value };
      const streetsLabel = document.querySelector('.track-bar .tb-streets')?.getAttribute('aria-label');
      click(streetsLabel);
      await new Promise((r) => setTimeout(r, 50));
      const afterStreets = { streets: ui.snapStreets.value, real: ui.snapRealTram.value, mode: ui.snapMode.value };
      return { before, afterReal, afterStreets };
    });
    console.log('     ', JSON.stringify(r));
    // switch sequences by real clicks: after each, no two snaps on at once for the type shown, and switching type keeps that
    const seq = await page.evaluate(async () => {
      const ui = await import('/src/game/ui.ts');
      const wait = () => new Promise((r) => setTimeout(r, 120));
      const lbl = { real: 'Snap to real world network', streets: 'Snap to streets', roads: 'Snap to roads', beside: 'Snap beside track' };
      const click = async (k) => { const b = [...document.querySelectorAll('.track-bar button')].find((x) => x.getAttribute('aria-label')?.startsWith(lbl[k])); if (!b) return false; b.click(); await wait(); return true; };
      const state = () => ({ type: ui.trackType.value, rail: ui.snapReal.value, tram: ui.snapRealTram.value, roads: ui.snapRoads.value, streets: ui.snapStreets.value, beside: ui.besideOn.value, mode: ui.snapMode.value });
      const type = async (t) => { ui.setTrackType(t); await wait(); };
      const out = [];
      const pressed = async (k, want) => { const b = [...document.querySelectorAll('.track-bar button')].find((x) => x.getAttribute('aria-label')?.startsWith(lbl[k])); if (b && (b.getAttribute('aria-pressed') === 'true') !== want) await click(k); };
      // the checker's sequence: rail real on, tram: streets off, beside on, back to Regular
      await type(2); await pressed('real', true); await type('tram'); await pressed('streets', false); await click('beside'); await type(2);
      out.push(['rail real, tram beside, back to rail', state()]);
      // rail: real on, then beside; then roads
      await pressed('beside', false); await pressed('real', true); await click('beside'); out.push(['rail real then beside', state()]);
      await click('roads'); out.push(['rail roads after beside', state()]);
      // tram: real on, then streets, then real again
      await type('tram'); await pressed('real', true); out.push(['tram real', state()]);
      await click('streets'); out.push(['tram streets after real', state()]);
      await click('real'); out.push(['tram real after streets', state()]);
      await type(2); out.push(['back to rail', state()]);
      return out;
    });
    for (const [what, s] of seq) {
      const tram = s.type === 'tram';
      const on = tram ? [s.tram, s.streets, s.beside] : [s.rail, s.roads, s.beside];
      ok(on.filter(Boolean).length <= 1, `${tag}: ${what}: at most one snap on ${JSON.stringify(s)}`);
    }
    const first = seq[0][1];
    ok(!first.rail && first.beside && first.mode === null, `${tag}: rail real, tram beside, back to rail: beside on, real snap off`);
    // both tram snaps forced on (an old browser's state, or set by hand): the real snap wins, no street snap or street price
    const both = await page.evaluate(async () => {
      const ui = await import('/src/game/ui.ts');
      ui.setTrackType('tram'); ui.snapRealTram.value = true; ui.snapStreets.value = true;
      await new Promise((r) => setTimeout(r, 300));
      return { street: window.tw.tools.street.isOn, mode: ui.snapMode.value, price: document.querySelector('.track-bar .price')?.textContent };
    });
    console.log('     ', JSON.stringify(both));
    ok(!both.street && both.mode === 'tram' && /0\.5x/.test(both.price ?? ''), `${tag}: both on, the real snap wins (no street snap, off-street price)`);
    // the tram row with its real snap on: four switches, beside back
    await page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); ui.snapRealTram.value = true; ui.snapStreets.value = false; });
    await page.waitForTimeout(500);
    {
      const box = await page.locator('.track-bar').boundingBox(), pad = 12;
      await page.screenshot({ path: out + `t3069_bar_${tag}_tram_real.png`, clip: { x: Math.max(0, box.x - pad), y: Math.max(0, box.y - pad), width: Math.min(w - Math.max(0, box.x - pad), box.width + 2 * pad), height: box.height + 2 * pad } });
      const fits = await page.evaluate(() => { const b = document.querySelector('.track-bar').getBoundingClientRect(); return b.right <= innerWidth && b.left >= 0; });
      ok(fits, `${tag}: tram row with the real snap on fits the width`);
    }
    ok(r.afterReal.real && !r.afterReal.streets && r.afterReal.mode === 'tram' && r.afterReal.rail === r.before.rail, `${tag}: tram real snap on turns streets off, rail snap untouched`);
    ok(r.afterStreets.streets && !r.afterStreets.real && r.afterStreets.mode === null, `${tag}: streets on turns tram real snap off`);
  } finally {
    await chrome.close();
  }
}
if (errors.length) console.log('errors: ' + errors.slice(0, 3).join(' | '));
process.exit(fails ? 1 : 0);
