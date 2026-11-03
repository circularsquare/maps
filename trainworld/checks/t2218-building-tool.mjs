// T-2218: the station building tool on the real New York save, with the mouse. Times Sq-42 St's upgrade
// row starts the tool on the rectangle round its complex; Escape drops it and three corners clicked a
// few degrees off square come out square (the snap), Enter takes the fourth and closes it; Construct pays
// the quoted US$2,000 a m². Edit outline: a corner dragged out pays for the floor added, still square.
// Renamed (free), then removed (nothing refunded). A tiny outline is refused with its reason, the floor's
// hover names the building, and a click on the floor selects the complex. Screenshots light and dark to
// artifacts/trainworld/t2218/. Uses the Vite already on the url (dev-nowatch.mjs). From the maps root:
//   node trainworld/checks/t2218-building-tool.mjs [url]
import { ARTIFACTS, REAL_SAVE, collectErrors, demandFinal, importSave, launchChrome } from './lib.mjs';
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';

const OUT = join(ARTIFACTS, 't2218');
await mkdir(OUT, { recursive: true });
const BASE = process.argv[2] ?? 'http://localhost:6189/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

const chrome = await launchChrome({ prefix: 'tw-t2218-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools?.building, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  // demand's first answer, so its loading card is off the map for the screenshots
  await demandFinal(page, { strict: false, timeout: 300000, polling: 1000 }).catch(() => console.log('  (demand did not finish in 5 minutes)'));
  await page.waitForTimeout(1500);
  const box = await (await page.$('#map')).boundingBox();
  const settle = async (ms = 800) => {
    await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 });
    await page.waitForTimeout(ms);
  };
  /** a pack point on the page, px */
  const onPage = async (p) => { const [x, y] = await page.evaluate(([x, y]) => { const q = window.tw.map.project(window.tw.packToLngLat(x, y)); return [q.x, q.y]; }, [p.x, p.y]); return { x: box.x + x, y: box.y + y }; };
  const tool = () => page.evaluate(() => {
    const b = window.tw.tools.building, q = b.quote;
    return { active: b.active, closed: b.closed, id: b.id, pts: b.pts.map((p) => ({ ...p })), quote: q && { m2: q.m2, price: q.price, members: q.members, issues: q.issues }, tag: [...document.querySelectorAll('.tool-tag')].filter((e) => e.style.display !== 'none').map((e) => e.textContent).join(' / ') };
  });
  /** the worst corner's distance from 90 degrees, degrees */
  const offSquare = (pts) => Math.max(...pts.map((b, i) => {
    const a = pts[(i + pts.length - 1) % pts.length], c = pts[(i + 1) % pts.length];
    const u = Math.atan2(a.y - b.y, a.x - b.x), v = Math.atan2(c.y - b.y, c.x - b.x);
    return Math.abs(((Math.abs(u - v) % Math.PI) - Math.PI / 2) * 180 / Math.PI);
  }));
  const cash = () => page.evaluate(() => window.tw.world.value.save.cash / 1e6);
  const notice = () => page.evaluate(() => document.querySelector('#tool-notes')?.textContent ?? '');
  const quoteFresh = () => page.waitForFunction(() => { const b = window.tw.tools.building, c = b.quoted(); return b.quote && c && b.quote.key === c.join(); }, undefined, { timeout: 30000 });

  // ---- Times Sq-42 St, its complex selected
  const st = await page.evaluate(() => {
    const s = window.tw.world.value.save.stations.find((x) => x.name.startsWith('Times Sq') && x.built);
    return s && { id: s.id, name: s.name, lng: s.lng, lat: s.lat };
  });
  check(!!st, 'Times Sq-42 St is in the save', st?.name);
  await page.evaluate((s) => { window.tw.map.jumpTo({ center: [s.lng, s.lat], zoom: 15.6 }); window.tw.selection.value = { kind: 'station', station: s.id }; }, st);
  await settle(1200);
  const cash0 = await cash();

  // ---- the upgrade row's icon: the rectangle round the complex
  await page.click('button[aria-label="Station building"]');
  await page.waitForTimeout(500);
  let t = await tool();
  check(t.active && t.closed && t.pts.length === 4 && t.id === null, 'the upgrade row starts the tool on a closed rectangle', `${t.pts.length} corners`);
  check(offSquare(t.pts) < 0.01, 'the offered rectangle is square', `${offSquare(t.pts).toFixed(4)} degrees off`);
  const offer = t.pts;
  await page.mouse.move(box.x + 30, box.y + 200);
  await quoteFresh();
  t = await tool();
  check(t.quote && !t.quote.issues.length && t.quote.members >= 2, 'the offer can be built', `${Math.round(t.quote?.m2)} m², US$${t.quote?.price.toFixed(1)}M, ${t.quote?.members} stations; tag "${t.tag}"`);
  await page.screenshot({ path: join(OUT, 'offer-light.png'), clip: box });

  // ---- Escape drops it; three corners clicked a little off square come out square
  await page.keyboard.press('Escape');
  await page.waitForTimeout(200);
  t = await tool();
  check(t.active && t.pts.length === 0, 'Escape drops the offered outline and keeps the tool', `${t.pts.length} corners`);
  const c0 = await onPage(offer[0]), c1 = await onPage(offer[1]), c2 = await onPage(offer[2]);
  // the second corner 5 px off its line, the third 6 px off
  const nx = -(c1.y - c0.y), ny = c1.x - c0.x, nl = Math.hypot(nx, ny);
  const off = (p, d) => ({ x: p.x + nx / nl * d, y: p.y + ny / nl * d });
  for (const [p, d] of [[c0, 0], [c1, 5], [c2, -6]]) {
    const q = off(p, d);
    await page.mouse.move(q.x - 3, q.y - 3);
    await page.mouse.move(q.x, q.y, { steps: 3 });
    await page.waitForTimeout(150);
    await page.mouse.click(q.x, q.y);
    await page.waitForTimeout(300);
  }
  t = await tool();
  check(t.pts.length === 3, 'three corners placed', `${t.pts.length}`);
  await page.keyboard.press('Enter');
  await page.waitForTimeout(300);
  t = await tool();
  const sq = offSquare(t.pts);
  check(t.closed && t.pts.length === 4 && sq < 1, 'Enter takes the fourth corner; the outline is within 1 degree of square', `${sq.toFixed(3)} degrees off`);
  await page.mouse.move(box.x + 30, box.y + 220);
  await quoteFresh();
  t = await tool();
  const quoted = t.quote;
  check(quoted && !quoted.issues.length && /m²/.test(t.tag) && /\$\d/.test(t.tag) && /stations inside/.test(t.tag), 'the tag gives the floor, the price and the stations inside', t.tag);
  await page.screenshot({ path: join(OUT, 'drawn-light.png'), clip: box });

  // ---- Construct
  await page.click('#tool-notes button:has-text("Construct")');
  await page.waitForFunction(() => !window.tw.tools.building.active && (window.tw.world.value.save.buildings?.length ?? 0) === 1, undefined, { timeout: 30000 });
  await page.waitForTimeout(800);
  const cash1 = await cash();
  const b1 = await page.evaluate(() => window.tw.world.value.save.buildings[0]);
  check(Math.abs(cash0 - cash1 - quoted.price) < 0.01 && Math.abs(quoted.price - 0.002 * quoted.m2) < 0.01, 'Construct pays US$2,000 a m²', `paid US$${(cash0 - cash1).toFixed(2)}M for ${Math.round(quoted.m2)} m²`);
  const n1 = await notice();
  check(/^Station building built at .+ for \$\d/.test(n1), 'the notice says it was built and for how much', n1);
  const panel1 = await page.evaluate(() => document.querySelector('#inspector, .inspector')?.textContent ?? '');
  check(panel1.includes('Station building') && panel1.includes('Edit outline') && panel1.includes('Remove building') && panel1.includes(b1.name), 'the complex view has the building\'s section', b1.name);
  // the ledger books it as construction today
  const ledger = await page.evaluate(() => { const m = window.tw.moneyState?.value; return m ? (m.ledger.at(-1)?.build ?? null) : null; });
  check(ledger === null || ledger >= quoted.price - 0.01, 'the day\'s ledger counts it as construction', `build US$${ledger?.toFixed?.(1)}M today`);

  // ---- screenshots, light and dark, the complex selected, once demand has the joined complex
  await demandFinal(page, { timeout: 300000, polling: 1000 }).catch(() => console.log('  (demand did not finish in 5 minutes)'));
  await page.mouse.move(box.x + 30, box.y + 240);
  await settle(1000);
  await page.screenshot({ path: join(OUT, 'built-light.png'), clip: box });
  await page.evaluate(() => document.querySelector('li.building-name')?.scrollIntoView({ block: 'center' }));
  await page.waitForTimeout(300);
  await page.screenshot({ path: join(OUT, 'built-light-panel.png') });
  await page.evaluate(() => window.tw.shade.value = 'dark');
  await page.waitForTimeout(1000);
  await page.screenshot({ path: join(OUT, 'built-dark.png'), clip: box });
  await page.screenshot({ path: join(OUT, 'built-dark-panel.png') });
  await page.evaluate(() => window.tw.map.jumpTo({ zoom: 14 }));
  await settle(1000);
  await page.screenshot({ path: join(OUT, 'built-dark-z14.png'), clip: box });
  await page.evaluate(() => window.tw.shade.value = 'light');
  await page.evaluate(() => window.tw.map.jumpTo({ zoom: 13.5 }));
  await settle(1000);
  await page.screenshot({ path: join(OUT, 'built-light-z13.5.png'), clip: box });
  await page.evaluate(() => window.tw.map.jumpTo({ zoom: 15.6 }));
  await settle(800);

  // ---- the floor's hover, and a click on it selects the complex
  const inside = await page.evaluate(() => {
    // a point on the floor away from every box: the outline's first corner, nudged inside
    const b = window.tw.world.value.save.buildings[0], c = b.corners, n = c.length / 2;
    let mx = 0, my = 0;
    for (let i = 0; i < n; i++) mx += c[2 * i] / n, my += c[2 * i + 1] / n;
    return { x: c[0] + (mx - c[0]) * 0.06, y: c[1] + (my - c[1]) * 0.06 };
  });
  const ip = await onPage(inside);
  await page.evaluate(() => { window.tw.selection.value = null; });
  await page.mouse.move(ip.x - 5, ip.y);
  await page.mouse.move(ip.x, ip.y, { steps: 3 });
  await page.waitForTimeout(600);
  const hover = await page.evaluate(() => [...document.querySelectorAll('.track-tag')].filter((e) => e.style.display !== 'none').map((e) => e.textContent).join(' / '));
  check(/Station building/.test(hover) && hover.includes(b1.name), 'the floor\'s hover names the building', hover);
  await page.mouse.click(ip.x, ip.y);
  await page.waitForTimeout(500);
  const picked = await page.evaluate(() => window.tw.selection.value);
  check(picked?.kind === 'station', 'a click on the floor selects the complex', JSON.stringify(picked));
  await page.evaluate((s) => { window.tw.selection.value = { kind: 'station', station: s.id }; }, st);
  await page.waitForTimeout(500);

  // ---- Edit outline: two corners dragged out a little off line, the outline still square, the
  // added floor paid
  await page.click('button:has-text("Edit outline")');
  await page.waitForTimeout(400);
  t = await tool();
  check(t.active && t.closed && t.id === b1.id && t.pts.length === 4, 'Edit outline opens the tool on the building', `${t.pts.length} corners`);
  const before = t.pts;
  const axis = Math.atan2(before[1].y - before[0].y, before[1].x - before[0].x);
  const along = (a, b) => { const d = Math.atan2(b.y - a.y, b.x - a.x) - axis; const r = ((d % (Math.PI / 2)) + Math.PI / 2) % (Math.PI / 2); return Math.min(r, Math.PI / 2 - r) * 180 / Math.PI; };
  /** drag corner `i` 40 px on along the edge from corner `from`, and `aside` px across it */
  const dragOut = async (i, from, aside) => {
    const now = (await tool()).pts;
    const p = await onPage(now[i]), f = await onPage(now[from]);
    const ex = p.x - f.x, ey = p.y - f.y, el = Math.hypot(ex, ey);
    const to = { x: p.x + ex / el * 40 - ey / el * aside, y: p.y + ey / el * 40 + ex / el * aside };
    await page.mouse.move(p.x, p.y);
    await page.waitForTimeout(150);
    await page.mouse.down();
    await page.mouse.move(to.x, to.y, { steps: 8 });
    await page.mouse.up();
    await page.waitForTimeout(300);
  };
  await dragOut(2, 1, 3);
  t = await tool();
  const moved = Math.hypot(t.pts[2].x - before[2].x, t.pts[2].y - before[2].y);
  check(moved > 0 && t.pts.length === 4 && along(t.pts[1], t.pts[2]) < 0.01, 'a dragged corner moves, its edge 3 px off line snapped onto it', `moved ${moved.toFixed(1)} pack units, edge ${along(t.pts[1], t.pts[2]).toFixed(4)} degrees off`);
  await dragOut(3, 0, -3);
  t = await tool();
  check(t.pts.length === 4 && offSquare(t.pts) < 1, 'with the other corner dragged out too, the outline is within 1 degree of square', `${offSquare(t.pts).toFixed(4)} degrees off`);
  await page.mouse.move(box.x + 30, box.y + 260);
  await quoteFresh();
  t = await tool();
  const added = t.quote;
  check(added && !added.issues.length && added.price > 0 && /added floor/.test(t.tag), 'the tag prices the floor added', t.tag);
  await page.screenshot({ path: join(OUT, 'edit-light.png'), clip: box });
  const cash2a = await cash();
  await page.click('#tool-notes button:has-text("Save outline")');
  await page.waitForFunction(() => !window.tw.tools.building.active, undefined, { timeout: 30000 });
  await page.waitForTimeout(800);
  const cash2 = await cash();
  check(Math.abs(cash2a - cash2 - added.price) < 0.01, 'Save outline pays for the floor added', `paid US$${(cash2a - cash2).toFixed(2)}M, quoted US$${added.price.toFixed(2)}M`);
  const n2 = await notice();
  check(/station building changed for \$\d/.test(n2), 'the notice says what the change cost', n2);

  // ---- rename: free
  await page.click('li.building-name button[aria-label="Rename"]');
  await page.waitForFunction(() => document.activeElement?.classList.contains('name-edit'), undefined, { timeout: 5000 });
  await page.keyboard.press('Control+A');
  await page.keyboard.type('Crossroads Hall');
  await page.keyboard.press('Enter');
  await page.waitForFunction(() => window.tw.world.value.save.buildings?.[0]?.name === 'Crossroads Hall', undefined, { timeout: 15000 }).catch(() => {});
  const renamed = await page.evaluate(() => window.tw.world.value.save.buildings?.[0]?.name);
  const cash3 = await cash();
  check(renamed === 'Crossroads Hall' && Math.abs(cash3 - cash2) < 1e-6, 'renaming is free', `${renamed}`);

  // ---- a tiny outline is refused with its reason
  await page.evaluate((s) => {
    const t = window.tw, p = t.lngLatToPack(s.lng, s.lat), k = Math.cosh(p.y / 6378137), d = 10 * k;
    t.tools.building.begin([p.x - d, p.y - d, p.x + d, p.y - d, p.x + d, p.y + d, p.x - d, p.y + d], null, null);
  }, { lng: st.lng + 0.01, lat: st.lat });
  await page.mouse.move(box.x + 40, box.y + 300);
  await quoteFresh();
  t = await tool();
  check(t.quote?.issues.includes('BuildingTooSmall') && /needs at least 1,000 m²/.test(t.tag), 'a tiny outline is refused with its reason', t.tag);
  await page.keyboard.press('Escape');
  await page.keyboard.press('Escape');
  await page.waitForTimeout(200);

  // ---- remove: asks first, nothing refunded
  await page.evaluate((s) => { window.tw.selection.value = { kind: 'station', station: s.id }; }, st);
  await page.waitForTimeout(400);
  await page.click('button:has-text("Remove building")');
  await page.waitForTimeout(300);
  const q = await page.evaluate(() => document.querySelector('dialog.confirmation')?.textContent ?? '');
  check(/Remove the Crossroads Hall station building\? Nothing is refunded\./.test(q), 'Remove building asks first', q);
  await page.click('dialog.confirmation button.danger');
  await page.waitForFunction(() => !(window.tw.world.value.save.buildings?.length), undefined, { timeout: 15000 });
  await page.waitForTimeout(500);
  const cash4 = await cash();
  check(Math.abs(cash4 - cash3) < 1e-6, 'removing refunds nothing', `cash US$${cash3.toFixed(2)}M -> US$${cash4.toFixed(2)}M`);
  const n4 = await notice();
  check(/Removed the Crossroads Hall station building\./.test(n4), 'the notice says it was removed', n4);
  check(Math.abs(cash0 - cash4 - quoted.price - added.price) < 0.02, 'in all: the floor and the floor added, nothing back', `US$${(cash0 - cash4).toFixed(2)}M`);

  check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} catch (e) {
  check(false, 'ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `T-2218 FAILED: ${fail}` : 'T-2218 PASSED');
process.exit(fail ? 1 : 0);
