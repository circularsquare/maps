// T-3768: no two things in a side panel touch, and nothing sticks out of it or is squeezed to nothing. In a new New York game in normal mode with one small line, at
// 1400 px and at 390 px (a phone: touch, mobile), it opens Settings (in a game, and as it is before a game is
// loaded), the Build, Lines, Stations, Finances and Cities tabs, the line's and a station's panels, the Advanced
// fares panel and the start menu's settings, collects what is drawn of the visible controls (buttons, inputs,
// selects, links, check boxes: the border box, or a bare icon button's icon and text) and of each line of text
// (its font size tall, the leading left out; cut where a box clips it), and fails when two closer than MIN px on
// both axes, or overlapping, are not meant to share a box, naming both. Meant to share a box: the buttons of one
// `.group` (Segmented, joined buttons), a stepper's parts, a list's sort headers, the line shapes' grid, and text
// in one flow (one block). Charts' svg labels count as text, each label its own block. It also fails a control or
// line of text that reaches past the panel's left or right edge (the panel scrolled to its left), and a line of text
// cut sideways to under SQUEEZE px and under half its width (a label squeezed to nothing), where it is in view up and
// down. KNOWN lists findings left for later; they print every run and do not fail.
// From the tree root, on a Vite of your own (node trainworld/checks/dev-nowatch.mjs 6172):
//   node trainworld/checks/t3768-panel-gaps.mjs [dev url] [--shot <tag>] [--gaps] [--only desktop|phone]
// (about 5 minutes). --shot <tag> saves the settings tab as artifacts/trainworld/t3768_settings[_phone]_<tag>.png;
// --gaps prints the gap above each row of the settings tab, to compare a layout change against; --panels <dir> saves
// each panel and a crop around each pair found.
import { join } from 'node:path';
import { buildLine } from './lib.mjs';
import { openGame } from './lib/play.mjs';

const args = process.argv.slice(2);
const BASE = args.find((a) => /^https?:/.test(a)) ?? 'http://localhost:6172/';
const opt = (k) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : null; };
const SHOT = opt('--shot');
const GAPS = args.includes('--gaps');
const ONLY = opt('--only');
const PANELS = opt('--panels');
/** px: two boxes closer than this on both axes touch */
const MIN = 3;
/** px: a line of text cut narrower than this, and to under half its width, is cut to nothing */
const SQUEEZE = 12;
/** findings left for later: [panel (substring), first name (substring), second name (substring), why] */
const KNOWN = [
  ['station panel', 'station length', 'way station', 'rows holding a control are 28 px (T-1802) and the station type buttons fill theirs, 2 px under the length stepper; a row gap is a design call'],
];

let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

/** In the page: the boxes in `root` that touch, each pair named. Self-contained (sent to the page as source). */
function touching({ root: sel, min, squeeze: SQUEEZE }) {
  const root = document.querySelector(sel);
  if (!root) return null;
  const CONTROL = 'button, input:not([type=hidden]), select, textarea, a[href], label.check, [role=button], [role=slider]';
  // parts meant to share a box: joined buttons, a stepper, sort headers, the line shapes' grid
  const SHARED = '.group, .stepper, .sort-head, .shape-grid';
  const shown = (el) => el.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true, opacityProperty: true, visibilityProperty: true });
  /** `r` cut to every scrolling or clipping box from `from` up to the root (the root's own scroll is not a cut) */
  const clip = (from, r) => {
    let { left, top, right, bottom } = r;
    for (let a = from; a && a !== root; a = a.parentElement) {
      const cs = getComputedStyle(a);
      if (cs.overflowX === 'visible' && cs.overflowY === 'visible') continue;
      const b = a.getBoundingClientRect();
      if (cs.overflowX !== 'visible') { left = Math.max(left, b.left); right = Math.min(right, b.right); }
      if (cs.overflowY !== 'visible') { top = Math.max(top, b.top); bottom = Math.min(bottom, b.bottom); }
    }
    return right - left > 0.5 && bottom - top > 0.5 ? { left, top, right, bottom } : null;
  };
  const short = (s) => { s = s.replace(/\s+/g, ' ').trim(); return s.length > 36 ? s.slice(0, 35) + '…' : s; };
  const nameOf = (el) => {
    const n = el.getAttribute('aria-label') || el.textContent || el.getAttribute('placeholder') || el.value || '';
    return `${el.tagName.toLowerCase()}${el.type && el.tagName === 'INPUT' ? '[' + el.type + ']' : ''} "${short(n)}"`;
  };
  /** the box text flows in: the nearest ancestor that is not inline */
  const blockOf = (el) => { if (el.closest('svg')) return el.closest('text') ?? el; while (el && el !== root && /^(inline|contents)$/.test(getComputedStyle(el).display)) el = el.parentElement; return el; };
  /** whether `a` and `b` are parts of one box meant to be shared (any such box around `a` holding `b`) */
  const sharedBox = (a, b) => { for (let s = a.closest(SHARED); s && root.contains(s); s = s.parentElement?.closest(SHARED)) if (s.contains(b)) return true; return false; };
  /** a line of text's box `rr` by its font size: the leading above and below the letters left out */
  const lettersOf = (p, rr) => {
    const inset = Math.max(0, (rr.height - parseFloat(getComputedStyle(p).fontSize) * (p.currentCSSZoom ?? 1)) / 2);
    return { left: rr.left, right: rr.right, top: rr.top + inset, bottom: rr.bottom - inset, width: rr.width, height: rr.height - 2 * inset };
  };
  const ink = (c) => !/^(transparent|rgba\(\d+, \d+, \d+, 0\))$/.test(c);
  const painted = (el) => {
    const cs = getComputedStyle(el);
    return ink(cs.backgroundColor) || cs.boxShadow !== 'none'
      || ['Top', 'Right', 'Bottom', 'Left'].some((s) => parseFloat(cs[`border${s}Width`]) > 0 && cs[`border${s}Style`] !== 'none' && ink(cs[`border${s}Color`]));
  };
  /** what a control shows: its box when it has a border or a fill, else its icons and text (a bare icon button) */
  const seen = (el) => {
    if (el.matches('input, select, textarea') || painted(el)) return el.getBoundingClientRect();
    let u = null;
    const add = (r) => {
      if (r.width <= 0 || r.height <= 0) return;
      u = u ? { left: Math.min(u.left, r.left), top: Math.min(u.top, r.top), right: Math.max(u.right, r.right), bottom: Math.max(u.bottom, r.bottom) } : { left: r.left, top: r.top, right: r.right, bottom: r.bottom };
    };
    for (const s of el.querySelectorAll('svg, img, canvas, input')) if (!s.parentElement.closest('svg')) add(s.getBoundingClientRect());
    const w = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
    for (let n; (n = w.nextNode());) {
      if (!n.nodeValue.trim() || n.parentElement.closest('svg')) continue;
      const g = document.createRange();
      g.selectNodeContents(n);
      for (const rr of g.getClientRects()) add(lettersOf(n.parentElement, rr));
    }
    return u ?? el.getBoundingClientRect();
  };
  const out = [];
  // each finding's elements carry its number, for --panels' crops
  for (const el of document.querySelectorAll('[data-t3768]')) el.removeAttribute('data-t3768');
  const mark = (el, k) => el.setAttribute('data-t3768', `${el.getAttribute('data-t3768') ?? ''} ${k}`.trim());
  // the panel's own edges, sideways (it scrolls down by design, never sideways without something sticking out)
  root.scrollLeft = 0;
  const rb = root.getBoundingClientRect(), z = root.currentCSSZoom ?? 1;
  const edgeL = rb.left + root.clientLeft * z, edgeR = edgeL + root.clientWidth * z;
  const pastEdge = (el, r, name) => {
    const over = Math.max(edgeL - r.left, r.right - edgeR);
    if (over <= 0.5) return;
    mark(el, out.length);
    out.push({ a: name, b: 'the panel edge', text: `${name} is ${Math.round(over)} px past the panel's edge` });
  };
  const items = [];
  for (const el of root.querySelectorAll(CONTROL)) {
    if (!shown(el)) continue;
    const outer = el.parentElement?.closest(CONTROL);
    if (outer && root.contains(outer)) continue; // part of a bigger control (a check's box)
    const r = clip(el.parentElement, seen(el));
    if (!r) continue;
    items.push({ el, r, name: nameOf(el) });
    pastEdge(el, r, nameOf(el));
  }
  const walk = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  for (let n; (n = walk.nextNode());) {
    if (!n.nodeValue.trim()) continue;
    const p = n.parentElement;
    if (!p || !shown(p)) continue;
    const c = p.closest(CONTROL);
    if (c && root.contains(c)) continue; // a control's own text
    const range = document.createRange();
    range.selectNodeContents(n);
    const block = blockOf(p);
    const name = `text "${short(n.nodeValue)}"`;
    for (const rr of range.getClientRects()) {
      // the text's own box clips it too (an ellipsis)
      const full = lettersOf(p, rr);
      // a label cut sideways to a letter or two (where it is in view up and down, not scrolled away)
      const across = clip(p, { ...full, top: -1e9, bottom: 1e9 }), upDown = clip(p, { ...full, left: -1e9, right: 1e9 });
      const wide = across ? across.right - across.left : 0;
      if (upDown && wide < SQUEEZE && wide < (full.right - full.left) / 2) {
        mark(p, out.length);
        out.push({ a: name, b: 'cut', text: `${name} is cut to ${Math.round(wide)} px wide` });
      }
      const r = clip(p, full);
      if (!r) continue;
      items.push({ el: p, r, block, name });
      pastEdge(p, r, name);
    }
  }
  for (let i = 0; i < items.length; i++) {
    for (let j = i + 1; j < items.length; j++) {
      const a = items[i], b = items[j];
      const dx = Math.max(b.r.left - a.r.right, a.r.left - b.r.right);
      const dy = Math.max(b.r.top - a.r.bottom, a.r.top - b.r.bottom);
      if (dx >= min || dy >= min) continue;
      if (a.block && b.block && a.block === b.block) continue; // one flow of text
      if (sharedBox(a.el, b.el)) continue;
      mark(a.el, out.length); mark(b.el, out.length);
      out.push({ a: a.name, b: b.name, gap: Math.round(Math.max(dx, dy) * 10) / 10 });
    }
  }
  return { items: items.length, out };
}

/** In the page: the gap above each shown row of the settings tab (`.settings-list`'s children, groups opened) */
function settingsGaps() {
  const list = document.querySelector('.settings-list');
  if (!list) return [];
  const rows = [];
  const add = (el) => {
    if (!el.checkVisibility()) return;
    if (el.matches('.checks, .stack, [data-search-group]')) { for (const c of el.children) add(c); return; }
    const r = el.getBoundingClientRect();
    if (r.height > 0) rows.push({ el, top: r.top, bottom: r.bottom });
  };
  for (const c of list.children) add(c);
  return rows.map((r, i) => {
    const name = (r.el.getAttribute('aria-label') || r.el.textContent || r.el.className).replace(/\s+/g, ' ').trim().slice(0, 28);
    return `${String(i ? Math.round(r.top - rows[i - 1].bottom) : 0).padStart(4)} ${String(Math.round(r.bottom - r.top)).padStart(4)}  ${r.el.className || r.el.tagName.toLowerCase()}: ${name}`;
  });
}

// Brooklyn, east of Prospect Park
const C = [-73.93, 40.66];
const P = (dx, dy) => [C[0] + dx / (111.32 * Math.cos((C[1] * Math.PI) / 180)), C[1] + dy / 111.32];
const UI = { shade: 'light', display: { bubbleColours: 'ramp', sound: true, dayShown: 'weekday' }, v: 15 };

const SIZES = [
  { tag: 'desktop', size: [1400, 900], touch: false, chromeOptions: { deviceScaleFactor: 1 } },
  { tag: 'phone', size: [390, 844], touch: true, chromeOptions: { isMobile: true, deviceScaleFactor: 2 } },
].filter((s) => !ONLY || s.tag === ONLY);

const said = (f) => f.text ?? `${f.a} and ${f.b}, ${f.gap} px`;
const known = (panel, f) => KNOWN.find(([p, a, b]) => panel.includes(p) && ((f.a.includes(a) && f.b.includes(b)) || (f.a.includes(b) && f.b.includes(a))));

for (const s of SIZES) {
  const g = await openGame({ url: BASE, city: 'nyc', ui: UI, size: s.size, touch: s.touch, chromeOptions: s.chromeOptions, prefix: `tw-t3768-${s.tag}-` });
  const page = g.page;
  try {
    const built = await buildLine(page, [P(-1.5, 0), P(0, 0.3), P(1.5, 0)], { name: 'Gaps' });
    check(!!built?.ok, `${s.tag}: a small line built for the panels`, JSON.stringify(built?.issues ?? []));
    await g.waitDemand({ timeout: 180000 }).catch(() => console.log('      (demand final not reached in 3 minutes; measuring as it is)'));
    const ids = await page.evaluate(() => { const w = window.tw.world.value; return { line: w.save.lines[0]?.id, station: w.save.stations[0]?.id }; });
    const select = (sel) => page.evaluate(async (sel) => { const ui = await import('/src/game/ui.ts'); ui.selection.value = sel; }, sel);

    const measure = async (panel, root) => {
      await page.waitForTimeout(400);
      const r = await page.evaluate(touching, { root, min: MIN, squeeze: SQUEEZE });
      if (!r) { check(false, `${s.tag} ${panel}: ${root} is shown`); return; }
      const bad = [];
      for (const f of r.out) {
        const k = known(`${s.tag} ${panel}`, f);
        if (k) console.log(`known ${s.tag} ${panel}: ${said(f)} (${k[3]})`);
        else bad.push(f);
      }
      check(!bad.length, `${s.tag} ${panel}: nothing touches (${r.items} boxes)`, bad.map(said).join('; '));
      if (PANELS) {
        const base = join(PANELS, `${s.tag}_${panel.replace(/\W+/g, '_')}`);
        await page.locator(root).first().screenshot({ path: base + '.png' });
        // a crop around each pair found (known ones too), scrolled into view
        for (let k = 0; k < Math.min(r.out.length, 12); k++) {
          const c = await page.evaluate((k) => {
            const els = [...document.querySelectorAll(`[data-t3768~="${k}"]`)];
            if (!els.length) return null;
            els[0].scrollIntoView({ block: 'center' });
            const rs = els.map((e) => e.getBoundingClientRect());
            const x = Math.max(0, Math.min(...rs.map((q) => q.left)) - 16), y = Math.max(0, Math.min(...rs.map((q) => q.top)) - 16);
            return { x, y, width: Math.min(innerWidth - x, Math.max(...rs.map((q) => q.right)) + 16 - x), height: Math.min(innerHeight - y, Math.max(...rs.map((q) => q.bottom)) + 16 - y) };
          }, k);
          if (c && c.width > 0 && c.height > 0) await page.screenshot({ path: `${base}_${k}.png`, clip: c });
        }
      }
    };

    await g.act.tab('settings');
    // a normal game: the sandbox switch is there (Anita's touching pair)
    check(await page.locator('#dock .pane [data-mode-switch]').count() === 1, `${s.tag} settings: the Switch to sandbox mode button is shown`);
    await measure('settings', '#dock .pane');
    if (GAPS) console.log(`gaps ${s.tag} settings (above, height, row):\n` + (await page.evaluate(settingsGaps)).join('\n'));
    if (SHOT) {
      const name = `t3768_settings${s.tag === 'phone' ? '_phone' : ''}_${SHOT}.png`;
      if (s.tag === 'desktop') {
        // the whole tab: the window made tall enough for the pane to show all of it
        const h = await page.evaluate(() => document.querySelector('#dock .pane').scrollHeight);
        await page.setViewportSize({ width: s.size[0], height: h + 420 });
        await page.evaluate((h) => { const p = document.querySelector('#dock .pane'); p.style.height = h + 'px'; p.style.maxHeight = 'none'; }, h);
        await page.waitForTimeout(500);
        await g.shot(name, { element: '#dock .pane' });
        await page.evaluate(() => { const p = document.querySelector('#dock .pane'); p.style.height = ''; p.style.maxHeight = ''; });
        await page.setViewportSize({ width: s.size[0], height: s.size[1] });
      } else {
        // the Game section, where the running side and the sandbox button are
        await page.evaluate(() => [...document.querySelectorAll('.settings-list > .h')].find((h) => h.textContent.trim() === 'Game')?.scrollIntoView({ block: 'start' }));
        await page.waitForTimeout(300);
        await g.shot(name, { element: '#dock .pane' });
      }
      console.log(`      shot ${name}`);
    }
    // before a game is loaded the Game section has only the player's own checks
    const held = await page.evaluate(async () => { const m = await import('/src/game/money.ts'); window.__t3768 = m.moneyState.peek(); m.moneyState.value = null; return true; });
    await measure('settings before a game', '#dock .pane');
    check(await page.locator('#dock .pane [data-mode-switch]').count() === 0, `${s.tag} settings before a game: no game mode rows (still measured as before a game)`);
    if (held) await page.evaluate(async () => { const m = await import('/src/game/money.ts'); m.moneyState.value = window.__t3768; });

    for (const [tab, panel] of [['build', 'Build tab'], ['lines', 'Lines tab'], ['stations', 'Stations tab'], ['money', 'Finances tab'], ['cities', 'Cities tab']]) {
      await g.act.tab(tab);
      await measure(panel, '#dock .pane');
    }
    // Advanced fares, from the Finances tab
    await g.act.tab('money');
    await page.locator('#dock .pane button.fare-advanced').first().click();
    await page.waitForSelector('dialog.fares-panel[open]', { timeout: 10000 });
    await measure('Advanced fares', 'dialog.fares-panel');
    await page.keyboard.press('Escape');
    await page.waitForTimeout(300);

    await select({ kind: 'line', line: ids.line });
    await measure('line panel', '#inspector');
    await select({ kind: 'station', station: ids.station });
    await measure('station panel', '#inspector');
    await select(null);

    // the start menu's settings, before any game
    const menu = await g.chrome.ctx.newPage();
    await menu.goto(`${BASE}?debug=1&lang=en&v=${Date.now()}`);
    await menu.locator('.start-gear').first().click({ timeout: 120000 });
    await menu.waitForSelector('.start-settings', { timeout: 10000 });
    await menu.waitForTimeout(300);
    const r = await menu.evaluate(touching, { root: '.start-settings', min: MIN, squeeze: SQUEEZE });
    check(!!r && !r.out.length, `${s.tag} start menu settings: nothing touches (${r?.items} boxes)`, (r?.out ?? []).map(said).join('; '));
    await menu.close();
    const errs = g.pageErrors();
    if (errs.length) console.log(`      page errors: ${errs.slice(0, 3).join(' | ')}`);
  } finally {
    await g.close();
  }
}
console.log(fail ? `${fail} failing` : 'all pass');
process.exit(fail ? 1 : 0);
