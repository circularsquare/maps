// Run from maps: node trainworld/checks/t1053-load-check.mjs [port]      (default 6153; about a minute)
// Saves naming removed zones load (T-1053), in a browser: its own Vite (port), its registry without
// Windham, New London and Litchfield (wnd, nlc, lit) and a successor table sending all three to
// Hartford (hfd), as T-1113's merge would. A copy of the real save and a new game's empty save, their
// headers naming the three and a made-up id, each load; the game they export names hfd where they
// named the three, has their ledgers and earliest star under hfd, and names none of the four.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { gzipSync, gunzipSync } from 'node:zlib';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { launchChrome, importSave, REAL_SAVE } from './lib.mjs';

const APP = fileURLToPath(new URL('../app', import.meta.url)).replace(/\\/g, '/');
const port = Number(process.argv[2] ?? 6153);
const GONE = ['wnd', 'nlc', 'lit'], MADE_UP = 't1053-made-up', TO = 'hfd';

const realReg = JSON.parse(readFileSync(APP + '/src/game/registry.json', 'utf8'));
const reg = { ...realReg, cities: realReg.cities.filter((c) => !GONE.includes(c.id)) };
const table = Object.fromEntries(GONE.map((z) => [z, TO]));

function decode(bytes) {
  const raw = gunzipSync(bytes), n = raw.readUInt32LE(4);
  return { header: JSON.parse(raw.subarray(8, 8 + n).toString('utf8')), track: raw.subarray(8 + n) };
}
function encode(header, track) {
  const json = Buffer.from(JSON.stringify(header), 'utf8'), len = Buffer.alloc(4);
  len.writeUInt32LE(json.length);
  return gzipSync(Buffer.concat([Buffer.from('TWG1'), len, json, track]));
}
const day = (d, fares) => ({ day: d, fares, running: 0.5, trains: 0, build: 0, rides: 100 });
/** a header as the game wrote it before the three were merged, plus a zone this build never had */
const named = (h) => ({
  ...h, region: realReg.world, clock: 3 * 86400 + 7 * 3600,
  cities: [...realReg.cities.map((c) => c.id), MADE_UP], start: 'nyc', opened: ['nyc', 'wnd', MADE_UP, 'nlc', TO],
  regions: { [TO]: { ledger: [day(1, 2)], hours: [] }, wnd: { ledger: [day(1, 3), day(2, 4)], hours: [] }, nlc: { ledger: [day(1, 5)], hours: [] }, [MADE_UP]: { ledger: [day(1, 7)], hours: [] } },
  stars: { wnd: 250000, nlc: 200000, [TO]: 240000, [MADE_UP]: 1000 },
  notes: { 900: [3, 'lit'], 901: [5, MADE_UP] },
});

const require = createRequire(APP + '/package.json');
const { createServer } = await import(pathToFileURL(require.resolve('vite')).href);
const server = await createServer({
  root: APP, configFile: APP + '/vite.config.ts', logLevel: 'warn', server: { port, strictPort: true, watch: null, hmr: false },
  plugins: [{ name: 't1053-scratch-registry', enforce: 'pre', load(id) {
    const f = id.split('?')[0].replace(/\\/g, '/');
    if (f.endsWith('/src/game/registry.json')) return JSON.stringify(reg);
    if (f.endsWith('/src/game/zone-successors.json')) return JSON.stringify(table);
  } }],
});
await server.listen();
const { ctx, close } = await launchChrome({ prefix: 'tw-t1053-', size: [1200, 800], acceptDownloads: true });
const errors = [];
let failed = false;
try {
  const page = await ctx.newPage();
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(`http://localhost:${port}/?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });
  const exported = async () => {
    const dl = page.waitForEvent('download', { timeout: 60000 });
    await page.evaluate(() => window.tw.client.post({ kind: 'export' }));
    return decode(await readFile(await (await dl).path()));
  };
  const load = async (bytes, lines) => {
    await page.evaluate(() => { window.tw.client.lastLoad = null; });
    await importSave(page, bytes);
    await page.waitForFunction((n) => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length === n, lines, { timeout: 120000 });
    await page.waitForTimeout(1500);
    return (await exported()).header;
  };
  const check = (what, h) => {
    const ids = JSON.stringify([h.cities, h.opened, h.start, Object.keys(h.regions ?? {}), Object.keys(h.stars ?? {})]);
    for (const z of [...GONE, MADE_UP]) assert.ok(!ids.includes(`"${z}"`), `${what}: still names ${z}`);
    assert.equal(h.opened[0], 'nyc', `${what}: opened ${h.opened}`);
    assert.ok(h.opened.includes(TO), `${what}: opened ${h.opened}`);
    const d1 = (z) => (h.regions?.[z]?.ledger ?? []).filter((d) => d.day === 1).reduce((s, d) => s + d.fares, 0);
    assert.equal(d1(TO), 2 + 3 + 5, `${what}: hfd's day 1 fares`);
    assert.equal((h.regions?.[TO]?.ledger ?? []).find((d) => d.day === 2)?.fares, 4, `${what}: hfd's day 2 fares`);
    assert.ok(d1('') >= 7, `${what}: the made-up zone's fares outside every zone`);
    assert.equal(h.stars?.[TO], 200000, `${what}: hfd's star`);
    assert.deepEqual(h.notes?.[900], [3, TO], `${what}: a line note's city`);
    console.log(`PASS  ${what}  (opened ${h.opened.join(' ')}; hfd day 1 fares ${d1(TO)}, star ${h.stars[TO]}; ${h.cities.length} cities)`);
  };

  const empty = await exported();
  check('new game save naming removed zones', await load(encode(named(empty.header), empty.track), 0));
  const real = decode(await readFile(REAL_SAVE));
  check('real save naming removed zones', await load(encode(named(real.header), real.track), 109));
  const plain = await load(await readFile(REAL_SAVE), 109);
  assert.ok(plain.cities.every((z) => z !== 'wnd'), 'the real save, unchanged, loads');
  console.log(`PASS  real save as it is  (${plain.cities.length} cities)`);
  assert.deepEqual(errors, [], 'page errors');
} catch (e) {
  failed = true;
  console.log('FAIL  ' + e.message.split('\n')[0]);
} finally {
  await close();
  await server.close();
}
console.log(failed ? 't1053 load: FAILED' : 't1053 load: PASSED');
process.exit(failed ? 1 : 0);
