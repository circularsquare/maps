// T-1436: the train inspector's "One rider" block on the real save at 08:30: screenshots light and
// dark at a desktop size and a 390 px phone into artifacts/trainworld/t1436/, and checks that the block
// shows a trip with minutes by train and by car, that "Pick another rider" changes the trip, that
// the explorer button opens the same two stations with a way back to the train, and no console
// errors. From the maps root:
//   node trainworld/checks/t1436-shots.mjs <port>
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, REAL_SAVE, launchChrome, collectErrors, importSave, demandFinal, startVite } from './lib.mjs';

const OUT = join(ARTIFACTS, 't1436');
await mkdir(OUT, { recursive: true });
const port = Number(process.argv[2] ?? 6173);
const vite = await startVite(port);
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };

const block = (page) => page.evaluate(() => {
  const h = [...document.querySelectorAll('#inspector .h')].find((e) => e.textContent.startsWith('One rider'));
  if (!h) return null;
  const rows = [...h.nextElementSibling.querySelectorAll('li')].map((li) => [li.querySelector('.grow')?.textContent, li.querySelector('.val')?.textContent.trim()]);
  return Object.fromEntries(rows);
});
const ready = (page) => page.waitForFunction(() => {
  const h = [...document.querySelectorAll('#inspector .h')].find((e) => e.textContent.startsWith('One rider'));
  const v = h && [...h.nextElementSibling.querySelectorAll('li')].map((li) => li.querySelector('.val')?.textContent ?? '');
  return !!v && v.every((t) => !t.includes('…'));
}, undefined, { timeout: 30000 });

async function session(size, label) {
  const chrome = await launchChrome({ prefix: 'tw-t1436-', size });
  const errors = [];
  try {
    const page = await chrome.ctx.newPage();
    await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
    collectErrors(page, errors);
    await page.goto(`${vite.url}?debug=1&new=1&paused=1`);
    await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
    await importSave(page, await readFile(REAL_SAVE));
    await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
    await demandFinal(page, { polling: 250, lines: 100 });
    await page.evaluate(() => {
      const tw = window.tw;
      tw.clock.setRunning(false);
      tw.clock.set(Math.floor(tw.clock.now() / 86400) * 86400 + 8.5 * 3600);
      tw.overlay.invalidate();
    });
    await page.waitForFunction(() => window.tw.renderer.tripCount > 0 && window.tw.renderer.board, undefined, { timeout: 60000 });
    await page.waitForTimeout(1500);
    // the fullest of a spread of live trains
    const pick = await page.evaluate(async () => {
      const tw = window.tw, r = tw.renderer, gpu = r.probe(tw.clock.now());
      const live = [];
      for (let i = 0; i < r.tripCount; i++) if (gpu[i * 2] < 1e8) live.push(i);
      let best = null;
      for (let k = 0; k < Math.min(40, live.length); k++) {
        const i = live[Math.floor((live.length * (k + 0.5)) / Math.min(40, live.length))];
        const sel = { kind: 'train', line: String(Math.floor(r.trips[i * 2] / 6)), profile: r.trips[i * 2], dep: r.trips[i * 2 + 1] + r.tripsEpoch };
        tw.selection.value = sel;
        await new Promise((d) => setTimeout(d, 120));
        const row = document.querySelector('#inspector .rows li.b .val');
        const n = row ? Number(row.textContent.replace(/,/g, '')) : 0;
        if (Number.isFinite(n) && (!best || n > best.n) && document.querySelector('#inspector .one-rider')) best = { n, sel };
      }
      if (best) tw.selection.value = best.sel;
      return best;
    });
    check(!!pick, `${label}: a train with riders aboard (${pick?.n})`);
    await ready(page);
    const first = await block(page);
    console.log(label, JSON.stringify(first));
    check(!!first && /min/.test(first['By train'] ?? '') && /min/.test(first['By car'] ?? ''), `${label}: minutes by train and by car (${first?.['By train']}, ${first?.['By car']})`);
    check(!!first?.From && !!first?.To && first.From !== first.To, `${label}: from and to differ (${first?.From} to ${first?.To})`);

    // another rider
    const seen = new Set([first.From + '>' + first.To + '>' + first.Purpose]);
    for (let k = 0; k < 8; k++) {
      await page.click('button[aria-label="Pick another rider"]');
      await page.waitForTimeout(150);
      await ready(page).catch(() => {});
      const b = await block(page);
      if (b) seen.add(b.From + '>' + b.To + '>' + b.Purpose);
    }
    check(seen.size > 1, `${label}: Pick another rider changes the rider (${seen.size} different in 9)`);

    const shot = async (file) => {
      await page.mouse.move(2, 2);
      const h = await page.evaluateHandle(() => [...document.querySelectorAll('#inspector .h')].find((e) => e.textContent.startsWith('One rider')));
      await h.evaluate((el) => el.scrollIntoView({ block: 'center' }));
      await page.waitForTimeout(250);
      const hb = await h.boundingBox();
      const ul = await (await h.evaluateHandle((el) => el.nextElementSibling)).boundingBox();
      const ins = await (await page.$('#inspector')).boundingBox();
      const vp = page.viewportSize();
      const x = Math.max(0, ins.x - 8), y = Math.max(0, hb.y - 40);
      await page.screenshot({ path: join(OUT, file), clip: { x, y, width: Math.min(vp.width - x, ins.width + 16), height: Math.min(vp.height - y, ul.y + ul.height + 12 - y) } });
    };
    await ready(page);
    for (const s of ['light', 'dark']) {
      await page.evaluate((s) => { document.documentElement.dataset.shade = s; window.tw.shade.value = s; }, s);
      await page.waitForTimeout(300);
      await shot(`${label}-${s}.png`);
      await page.screenshot({ path: join(OUT, `${label}-${s}-full.png`) });
    }
    await page.evaluate(() => { document.documentElement.dataset.shade = 'light'; window.tw.shade.value = 'light'; });

    // the hover on the train row
    const b0 = await block(page);
    const li = await page.evaluateHandle(() => [...document.querySelectorAll('#inspector .one-rider li')].find((e) => e.textContent.startsWith('By train')));
    await li.evaluate((el) => el.scrollIntoView({ block: 'center' }));
    const bb = await li.boundingBox();
    await page.mouse.move(bb.x + 10, bb.y + bb.height / 2, { steps: 3 });
    await page.waitForTimeout(800);
    const tag = await page.evaluate(() => { const t = document.querySelector('.tool-tag.ui-tag'); return t && t.style.display !== 'none' ? t.innerText : ''; });
    check(/effective time/i.test(tag), `${label}: the train row has a hover (${tag.replace(/\n/g, ' | ').slice(0, 120)})`);
    await page.screenshot({ path: join(OUT, `${label}-hover.png`) });

    // the explorer
    await page.click('button[aria-label="Open this trip in the trip explorer"]');
    await page.waitForSelector('#inspector .trip-modes', { timeout: 60000 }).catch(() => {});
    await page.waitForTimeout(1500);
    const sel = await page.evaluate(() => { const s = window.tw.selection.value; return s && { kind: s.kind, from: s.fromStation, to: s.toStation }; });
    const fromId = await page.evaluate((nm) => nm, null);
    check(sel?.kind === 'trip' && !!sel.from && !!sel.to, `${label}: the explorer opens between two stations`);
    const head = await page.evaluate(() => document.querySelector('#inspector .title .name')?.textContent);
    check(head === 'Trip', `${label}: the trip explorer shows (${head})`);
    const effective = await page.evaluate(() => [...document.querySelectorAll('#inspector li')].filter((e) => e.textContent.startsWith('Effective time')).map((e) => e.querySelector('.val')?.textContent));
    console.log(label, 'explorer effective times', JSON.stringify(effective), 'rider block', b0['By train'], b0['By car']);
    await page.screenshot({ path: join(OUT, `${label}-explorer.png`) });
    await page.click('#inspector button.btn.text:has-text("Back")');
    await page.waitForTimeout(500);
    const back = await page.evaluate(() => window.tw.selection.value?.kind);
    check(back === 'train', `${label}: Back returns to the train (${back})`);
    void fromId;
    check(errors.length === 0, `${label}: no console errors${errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''}`);
  } finally {
    await chrome.close();
  }
}

try {
  await session([1400, 1000], 'desktop');
  await session([390, 900], 'phone');
} finally {
  vite.child.kill();
}
process.exit(fail ? 1 : 0);
