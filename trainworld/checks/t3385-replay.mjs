// Replay the user's captured stop edits with the real Asia save and optimistic stop rows.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { ARTIFACTS, importSave, launchChrome } from './lib.mjs';
const base = process.argv[2] ?? 'http://localhost:6118/';
const report = JSON.parse(await readFile(`${ARTIFACTS}/t3385-user-freeze.json`, 'utf8'));
const session = report.previous.find((s) => s.id === '07629e74-f342-46bc-8705-8684733051f9');
const clicks = session.events.filter((e) => e.event === 'edit-click').map((e) => e.detail);
const chrome = await launchChrome({ prefix: 'tw-t3385-replay-' });
try {
  const page = await chrome.ctx.newPage();
  await page.goto(`${base}?new=1&debug=1&paused=1&diagnostics=1&demandWorkers=1`);
  await page.waitForFunction(() => window.tw?.world?.value, undefined, { timeout: 180000 });
  await importSave(page, await readFile(`${ARTIFACTS}/reports/t3385-asia.save`));
  await page.waitForFunction(() => window.tw.world.value?.save.lines.length === 186, undefined, { timeout: 180000 });
  await page.evaluate(async (before) => {
    const tw = window.tw, ui = await import('/src/game/ui.ts');
    const a = await tw.client.edit({ op: 'setStops', line: 185, stops: before.map(Number), autoFast: false });
    if (!a.ok) throw new Error('initial stop order refused');
    const l = tw.world.value.save.lines.find((l) => l.num === 185);
    tw.selection.value = { kind: 'line', line: l.id };
    ui.relatedLines.value = true;
    ui.lineDraft.value = { line: 185, stops: l.stops.map(Number) }; ui.tool.value = 'line';
    const s = tw.world.value.save.stations.find((s) => s.id === '7717');
    if (s) tw.map.jumpTo({ center: [s.lng, s.lat], zoom: 13 });
  }, clicks[0].lineStops);
  await page.waitForTimeout(500);
  for (const click of clicks) {
    const operation = page.evaluate(async (op) => {
      const tw = window.tw, ui = await import('/src/game/ui.ts');
      const { edit } = await import('/src/game/world.ts');
      const { guessRow, settleRow } = await import('/src/ui/inspector/stopRowGuess.ts');
      const old = tw.world.value.save.lines.find((l) => l.num === op.line);
      const removed = op.op === 'setStops' ? old.stops.filter((s) => !op.stops.includes(Number(s))) : [];
      for (const s of removed) guessRow(s, true, op.line);
      const a = await edit(op, false, false, 'rows');
      const l = tw.world.value.save.lines.find((l) => l.num === op.line);
      ui.lineDraft.value = { line: op.line, stops: l.stops.map(Number) };
      for (const s of removed) settleRow(s, l.stops.includes(s));
      if (!a.ok) throw new Error('replay edit refused');
    }, click.edit);
    let timeout;
    await Promise.race([operation, new Promise((_, reject) => { timeout = setTimeout(() => reject(new Error('page stalled on replay edit')), 15000); })]).finally(() => clearTimeout(timeout));
    await page.waitForTimeout(450);
  }
  const result = await page.evaluate(() => ({ stops: window.tw.world.value.save.lines.find((l) => l.num === 185).stops.map(Number),
    diagram: !!document.querySelector('.related'),
    padding: parseFloat(document.querySelector('.related')?.style.paddingTop ?? '0') }));
  assert(result.diagram, 'related diagram was mounted during the replay');
  assert.deepEqual(result.stops, clicks.at(-1).edit.stops);
  assert(result.padding < 1000, 'diagram padding bounded');
  await page.screenshot({ path: `${ARTIFACTS}/t3385-fixed-replay.png` });
  console.log(`PASS ${clicks.length} captured edits replayed; final stops match, padding ${result.padding}px, page responsive`);
} finally { await chrome.close(); }
