// A guessed-out stop remains in the related diagram until its edit lands. Its zero layout box
// must not feed back into padding and keep the Preact render queue running forever.
import assert from 'node:assert/strict';
import { launchChrome } from './lib.mjs';
const base = process.argv[2] ?? 'http://localhost:6118/';
const chrome = await launchChrome({ prefix: 'tw-t3385-row-' });
try {
  const page = await chrome.ctx.newPage();
  // Isolate the real diagram/layout from the rest of the game's startup and worker pool.
  await page.route('**/src/game/world.ts', (r) => r.fulfill({ contentType: 'text/javascript', body: `export const world={value:null};export const linesByNum=w=>new Map(w.save.lines.map(l=>[l.num,l]));export const stationNames=w=>new Map(w.save.stations.map(s=>[s.id,s.name]));` }));
  await page.route('**/src/game/ui.ts', (r) => r.fulfill({ contentType: 'text/javascript', body: 'export const selection={value:null};' }));
  await page.route('**/src/ui/useClockQuery.ts', (r) => r.fulfill({ contentType: 'text/javascript', body: 'export const useClockQuery=()=>null;' }));
  await page.route('**/hidden-row-check', (r) => r.fulfill({ contentType: 'text/html', body: `<!doctype html><style>
    .related{position:relative}.stops{margin:0;padding:0}li{height:25px;list-style:none}.guess-out{display:none}
    </style><div id="root"></div>` }));
  await page.goto(`${base}hidden-row-check`);
  await page.evaluate(async () => {
    const source = await (await fetch('/src/main.ts')).text();
    const preact = /["']([^"']*\/preact\.js[^"']*)["']/.exec(source)[1];
    const { h, render, options } = await import(preact);
    await import('/src/game/ui.ts');
    const { RelatedStops } = await import('/src/ui/inspector/RelatedStops.tsx');
    const { world } = await import('/src/game/world.ts');
    const line = { id: '0', num: 0, name: 'Test', colour: '#f80', stops: ['1','2','3'] };
    world.value = { version: 1, save: { lines: [line], stations: line.stops.map((id) => ({ id, name: id })) }, lineStats: { '0': {} } };
    window.renders = 0;
    const before = options.__r;
    options.__r = (v) => {
      before?.(v);
      if (v.type === RelatedStops && ++window.renders > 60) throw new Error('layout render loop');
    };
    window.renderHidden = (hidden) => {
      window.renders = 0;
      render(h(RelatedStops, { l: line, stopS: [0,100,200], colour: '#f80', loop: false, flip: false,
        transfers: () => [], lRow: (i) => h('li', { 'data-r': true, class: i === hidden ? 'guess-out' : '' }, `Stop ${i}`) }), document.getElementById('root'));
    };
    window.renderHidden(0);
  });
  await page.waitForTimeout(1000);
  const result = await page.evaluate(() => ({ renders: window.renders, padding: document.querySelector('.related').style.paddingTop }));
  console.log(JSON.stringify(result));
  assert(result.renders < 10, 'hidden stop must settle without a layout render loop');
  assert(parseFloat(result.padding) < 50, 'padding stays bounded');
  for (const hidden of [1, 2, -1, 0, -1]) {
    await page.evaluate((i) => window.renderHidden(i), hidden);
    await page.waitForTimeout(100);
    const next = await page.evaluate(() => ({ renders: window.renders, padding: parseFloat(document.querySelector('.related').style.paddingTop) }));
    assert(next.renders < 10 && next.padding < 50, `row ${hidden} settles after hiding/restoring`);
  }
  console.log('PASS hidden stop layout settles');
} finally { await chrome.close(); }
