// T-854: run.mjs for each city and mode, one at a time; skips outputs that exist.
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const REAL = 'C:/Users/anita/projects/maps/trainworld/data/packs/real';
const SAVES = { nyc: 'C:/Users/anita/projects/maps/trainworld/T-007-real-nyc.save', lon: `${REAL}/lon.save`, par: `${REAL}/par.save`, tyo: `${REAL}/tyo.save` };
const cities = (process.argv[2] ?? 'nyc,lon,par,tyo').split(',');
const modes = (process.argv[3] ?? 'real,zero,flat,dens').split(',');
for (const c of cities) for (const m of modes) {
  const out = join(HERE, `${c}_${m.replace(':', '')}.json`);
  if (existsSync(out)) continue;
  console.log('==', c, m);
  const r = spawnSync(process.execPath, [join(HERE, 'run.mjs'), c, SAVES[c], m, out], { stdio: 'inherit' });
  console.log('==', c, m, 'exit', r.status);
}
