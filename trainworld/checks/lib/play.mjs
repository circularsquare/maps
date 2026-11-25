// T-3556: the agent playtest harness. Open the game in a headless Chrome of its own and play it the way a
// player does: tools and options picked on the real toolbar, track drawn by pointer presses at lon/lat points
// (exact: the map's own projection), panels' controls pressed by their visible names, and the figures a player
// sees read back as shown. Use it instead of writing a Playwright driver from scratch.
//
//   import { openGame } from './lib/play.mjs';
//   const g = await openGame({ url: 'http://localhost:6139/', city: 'nyc', mode: 'sandbox' });   // or { save: REAL_SAVE }
//   try {
//     await g.act.camera([-73.99, 40.75], 14);
//     await g.act.tool('track', { type: 'regular', tracks: 1, snap: { real: false } });
//     const run = await g.act.drawTrack([[-74.0, 40.75], [-73.99, 40.752], [-73.98, 40.755]]);   // run.segments[i].edges
//     await g.act.placeStation([-74.0, 40.75]); await g.act.placeStation([-73.98, 40.755]);
//     await g.act.construct();
//     const line = await g.act.newLine([[-74.0, 40.75], [-73.98, 40.755]], { name: 'Check', zoom: 15 }); // zoomed out no station is drawn to press
//     await g.act.setTrains([3, 2, 2]);                          // the open line panel's steppers (the panel caps them)
//     await g.waitDemand();                                       // the demand final (lib.mjs demandFinal)
//     console.log((await g.see.line()).schedule, (await g.see.money()).cash);
//     await g.shot('t9999_line.png', { element: '#inspector' }); // or { box: [[w, s], [e, n]], pad: 20 }
//   } finally { await g.close(); }
//
// Three kinds of call, kept apart so a later public surface can take `act` and `see` as they are:
//   g.act.*    player input only: camera, tab, tool, drawTrack, construct, placeStation, newLine, selectAt, openLine,
//              setTrains, click(name), type(name, text), key, press(lngLat)
//   g.see.*    what the screen shows, verbatim: panel (title, rows by label, tables, alerts), line (schedule, stops),
//              money, notes (the tool notes and notice), status, toolbar
//   g.state.*  named game-state reads for asserts the screen can't answer (counts, lines, stations, edgesFrom,
//              drawing, demand, selection, view); a check's own probes stay in the check, through `g.page`
//   g.toScreen([[lng, lat]]) / g.toLngLat([[x, y]]) page px both ways; a press where a panel covers the map throws.
// Options of openGame: url; city (a new game there, `?new=1&city=`) or save (a .save file, opened through the Saved
// games panel's Open a file; `saveVia: 'post'` posts it as smoke does, faster); mode ('sandbox'); paused (true);
// ui (the trainworld-ui localStorage object, verbatim); query (more URL parameters); size ([1400, 900]); touch (a
// finger instead of the mouse); prefix (the throwaway profile's name). English is pinned (labels are English).
// Cores: launchChrome claims from the ledger (checks/cores.mjs) and caps the page's demand workers to the claim;
// the Chrome and its profile go at close() or when the process exits. Several games in one Chrome: launchGames().
import { readFile } from 'node:fs/promises';
import { isAbsolute, join } from 'node:path';
import { ARTIFACTS, collectErrors, demandFinal, importSave, launchChrome } from '../lib.mjs';
import { actions } from './actions.mjs';
import { SEE, STATE } from './reads.mjs';

/** the Playwright driver under the action layer (lib/actions.mjs lists what a driver has) */
function playwrightDriver(page, input) {
  const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  // an exact name, else the same name followed by a button's key hint ("New line" is named "New lineN")
  const locate = async (t) => {
    if (t.css) return page.locator(t.css);
    const scope = t.within ? page.locator(t.within).first() : page;
    const role = t.role ?? 'button';
    const exact = scope.getByRole(role, { name: t.name, exact: t.exact ?? true });
    if (t.exact === false || typeof t.name !== 'string' || await exact.count()) return exact;
    return scope.getByRole(role, { name: new RegExp(`^${esc(t.name)}\\s*\\S{1,6}$`) });
  };
  const one = async (t) => {
    const all = await locate(t);
    if (t.nth !== undefined) return all.nth(t.nth);
    const n = await all.count();
    for (let i = 0; i < n; i++) if (await all.nth(i).isVisible()) return all.nth(i);
    return all.first();
  };
  const describe = (t) => t.css ?? `${t.role ?? 'button'} "${t.name}"${t.within ? ' in ' + t.within : ''}${t.nth !== undefined ? ' #' + t.nth : ''}`;
  return {
    input,
    see: (name, arg) => page.evaluate(SEE[name], arg ?? null),
    state: (name, arg) => page.evaluate(STATE[name], arg ?? null),
    // whole pixels: Chrome floors an event's fractional px, so a rounded one is within half a pixel of the point
    move: (x, y, { steps = 1 } = {}) => page.mouse.move(Math.round(x), Math.round(y), { steps }),
    async press(x, y, { shift = false, ctrl = false, double = false } = {}) {
      if (shift) await page.keyboard.down('Shift');
      if (ctrl) await page.keyboard.down('Control');
      try { await page.mouse.click(Math.round(x), Math.round(y), { clickCount: double ? 2 : 1 }); } finally {
        if (ctrl) await page.keyboard.up('Control');
        if (shift) await page.keyboard.up('Shift');
      }
    },
    tap: (x, y) => page.touchscreen.tap(Math.round(x), Math.round(y)),
    async control(t, how = {}) {
      const el = await one(t);
      try {
        if (how.file) {
          const [chooser] = await Promise.all([page.waitForEvent('filechooser', { timeout: 10000 }), el.click({ timeout: 10000 })]);
          await chooser.setFiles(how.file);
        } else if (input === 'touch') await el.tap({ timeout: 10000 });
        else await el.click({ timeout: 10000 });
      } catch (e) { throw new Error(`can't press ${describe(t)}: ${e.message.split('\n')[0]}`); }
    },
    async fill(t, text) {
      if (!t.focused) {
        const el = await one(t);
        try { await el.click({ timeout: 10000 }); } catch (e) { throw new Error(`can't reach ${describe(t)}: ${e.message.split('\n')[0]}`); }
      }
      await page.keyboard.press('Control+A');
      await page.keyboard.type(text);
      await page.keyboard.press('Enter');
    },
    key: (combo) => page.keyboard.press(combo),
    text: (s) => page.keyboard.type(s),
    camera: (v) => page.evaluate((v) => void window.tw.map.jumpTo(Object.fromEntries(Object.entries(v).filter(([, x]) => x !== undefined && x !== null))), v),
    sleep: (ms) => page.waitForTimeout(ms),
  };
}

/** ms a game may take to open; TW_READY_MS for a save known to open slowly */
const READY_MS = Number(process.env.TW_READY_MS) || 300000;

/** a game in a new tab of `chrome` (launchChrome's); see openGame for the options */
async function openIn(chrome, { url, city = null, save = null, saveVia = 'panel', mode = null, paused = true, ui = null, query = '', touch = false, lang = 'en', errorsIgnore } = {}) {
  if (!url) throw new Error('openGame wants a url (your own Vite, e.g. http://localhost:6139/)');
  const page = await chrome.ctx.newPage();
  const errors = [];
  collectErrors(page, errors, errorsIgnore);
  if (ui) await page.addInitScript((ui) => { try { if (!sessionStorage.getItem('tw-play-ui')) { sessionStorage.setItem('tw-play-ui', '1'); localStorage.setItem('trainworld-ui', JSON.stringify(ui)); } } catch {} }, ui);
  const q = new URLSearchParams(typeof query === 'string' ? query : Object.entries(query).map(([k, v]) => [k, String(v)]));
  q.set('debug', '1');
  if (lang) q.set('lang', lang);
  if (paused) q.set('paused', '1');
  if (!save || saveVia === 'panel' || saveVia === 'post') { q.set('new', '1'); if (city) q.set('city', city); }
  if (mode) q.set('mode', mode);
  q.set('v', String(Date.now()));
  await page.goto(`${url}${url.includes('?') ? '&' : '?'}${q}`);
  await page.waitForFunction(STATE.ready, undefined, { timeout: READY_MS });
  const d = playwrightDriver(page, touch ? 'touch' : 'mouse');
  const g = actions(d);
  if (save) {
    const bytes = await readFile(save);
    if (saveVia === 'post') {
      await importSave(page, bytes);
      await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: READY_MS });
    } else {
      // the player's way: Saved games, Open a file; the page reloads into the file's game
      await g.act.tab('saves');
      await Promise.all([page.waitForNavigation({ timeout: READY_MS }), d.control({ name: 'Open a file' }, { file: save })]);
      await page.waitForFunction(() => window.tw?.client?.lastLoad?.from === 'file' && window.tw.world.value?.save.lines.length > 0 && window.tw.tools, undefined, { timeout: READY_MS });
    }
  }
  return {
    page, errors, driver: d, ...g,
    /** wait for demand's final result (lib.mjs demandFinal: strict, `after`, `held`, `lines`, `timeout`) */
    waitDemand: (opts) => demandFinal(page, opts),
    /** a screenshot to `path` (a bare name goes in artifacts/trainworld/): the page, `{ element: css }`, a lon/lat
     * `{ box: [[w, s], [e, n]], pad }`, or a page `{ clip }` */
    async shot(path, { element, box, pad = 0, clip } = {}) {
      const file = isAbsolute(path) ? path : join(ARTIFACTS, path);
      if (element) await page.locator(element).first().screenshot({ path: file });
      else {
        const c = box ? await d.state('clipOf', { box, pad }) : clip;
        if (box && !c) throw new Error(`the box ${JSON.stringify(box)} is not in view: move the camera first`);
        await page.screenshot({ path: file, clip: c });
      }
      return file;
    },
    /** page and console errors so far (the basemap's offline noise left out) */
    pageErrors: () => errors.slice(),
    close: () => page.close(),
  };
}

/**
 * One headless Chrome for several games (each `open` a new tab), closed with `close()`. `size` [w, h],
 * `touch` (a touch screen), `prefix` (the profile's name) and launchChrome's other options.
 */
export async function launchGames({ size = [1400, 900], touch = false, prefix = 'tw-play-', ...options } = {}) {
  const chrome = await launchChrome({ prefix, size, deviceScaleFactor: 1, ...(touch ? { hasTouch: true } : {}), ...options });
  return {
    chrome,
    open: (opts) => openIn(chrome, { touch, ...opts }),
    close: () => chrome.close(),
  };
}

/** A game in a Chrome of its own (see the top of this file); `close()` ends both. */
export async function openGame({ size, prefix, chromeOptions, ...opts } = {}) {
  const games = await launchGames({ size, touch: opts.touch, prefix, ...chromeOptions });
  try {
    const g = await openIn(games.chrome, { touch: opts.touch, ...opts });
    return { ...g, chrome: games.chrome, close: () => games.close() };
  } catch (e) {
    await games.close();
    throw e;
  }
}
