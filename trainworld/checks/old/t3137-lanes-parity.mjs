// T-3137: the lanes model decides what the old rank arithmetic did on networks without single track
// joins. Builds `sim/src/bin/lanes_parity.rs`, fingerprints the real save, the Northeast save and two
// small games of its own (right- and left-hand running: flat and flying junctions, a flyover, single
// track two-way and one way, quad with a crossover, a circle line, a flat crossing), and compares each section (paths, fresh routes, badges, junction
// inspectors, the train guard's pairs, capacity resources and figures, the lines' figures) with a
// baseline.
//
//   node trainworld/checks/t3137-lanes-parity.mjs            compare with the baseline
//   node trainworld/checks/t3137-lanes-parity.mjs --write    make this build the baseline
//
// The baseline's full text is in artifacts/trainworld/t3137-parity/base-*.txt (made from master
// a5d6a81a, before the lanes); `t3137-parity-digest.json` beside this file holds each section's
// hash, so a compare still works without the artifacts (it then cannot show the first difference).
// A stage that changes what a network decides on purpose (C2's joins in the small games) reviews the
// differences and writes a new baseline. Timings of a load, the searches and two edits go to the log.
import { execFileSync, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { gunzipSync } from 'node:zlib';
import { HERE, ARTIFACTS, REAL_SAVE } from './lib.mjs';

const write = process.argv.includes('--write');
const SIM = join(HERE, '..', 'sim');
const OUT = join(ARTIFACTS, 't3137-parity');
const DIGEST = join(HERE, 't3137-parity-digest.json');
mkdirSync(OUT, { recursive: true });

const env = { ...process.env, CARGO_BUILD_JOBS: process.env.CARGO_BUILD_JOBS ?? '3', PATH: `${join(homedir(), '.cargo', 'bin')};${process.env.PATH}` };
execFileSync('cargo', ['build', '--release', '--bin', 'lanes_parity'], { cwd: SIM, env, stdio: 'inherit' });
const bin = join(SIM, 'target', 'release', process.platform === 'win32' ? 'lanes_parity.exe' : 'lanes_parity');

// the save's network part: after the gzip, the 4-byte magic and the header's length and JSON
const SAVES = { nyc: REAL_SAVE, northeast: join(HERE, '..', 'T-125-northeast.save') };
const tracks = {};
for (const [name, file] of Object.entries(SAVES)) {
  const raw = gunzipSync(readFileSync(file));
  tracks[name] = join(OUT, `${name}.track`);
  writeFileSync(tracks[name], raw.subarray(8 + raw.readUInt32LE(4)));
}

const sections = (text) => {
  const out = {};
  let name = null;
  for (const line of text.split('\n')) {
    if (line.startsWith('## ')) out[name = line.slice(3)] = [];
    else if (name && line) out[name].push(line);
  }
  return out;
};
const hash = (lines) => createHash('sha256').update(lines.join('\n')).digest('hex').slice(0, 16);

const digest = existsSync(DIGEST) ? JSON.parse(readFileSync(DIGEST, 'utf8')) : {};
let fails = 0;
for (const src of ['nyc', 'northeast', 'game', 'game-left']) {
  const now = join(OUT, `now-${src}.txt`);
  const run = spawnSync(bin, [tracks[src] ?? src, now], { env, encoding: 'utf8' });
  if (run.status !== 0) {
    console.log(`FAIL ${src}: lanes_parity exited ${run.status}\n${run.stderr}`);
    fails++;
    continue;
  }
  const timing = run.stderr;
  writeFileSync(join(OUT, `timing-${src}.log`), timing);
  console.log(`${src} timings:\n  ${timing.trim().split('\n').join('\n  ')}`);
  const cur = sections(readFileSync(now, 'utf8'));
  const basePath = join(OUT, `base-${src}.txt`);
  if (write) {
    writeFileSync(basePath, readFileSync(now));
    digest[src] = Object.fromEntries(Object.entries(cur).map(([k, v]) => [k, `${v.length} ${hash(v)}`]));
    continue;
  }
  const base = existsSync(basePath) ? sections(readFileSync(basePath, 'utf8')) : null;
  for (const [name, lines] of Object.entries(cur)) {
    const want = digest[src]?.[name];
    const got = `${lines.length} ${hash(lines)}`;
    let ok = want ? want === got : false;
    let detail = `${lines.length} rows`;
    if (base?.[name]) {
      const b = base[name];
      ok = b.length === lines.length && b.every((x, i) => x === lines[i]);
      if (!ok) {
        const i = lines.findIndex((x, k) => x !== b[k]);
        detail = `${lines.length} rows against ${b.length}; first difference at row ${i}:\n    was ${(b[i] ?? '(none)').slice(0, 300)}\n    now ${(lines[i] ?? '(none)').slice(0, 300)}`;
      }
    } else if (!want) {
      detail = 'no baseline';
    }
    if (!ok) fails++;
    console.log(`${ok ? 'PASS' : 'FAIL'} ${src} ${name} (${detail})`);
  }
}
if (write) {
  writeFileSync(DIGEST, JSON.stringify(digest, null, 1) + '\n');
  console.log(`wrote the baseline: ${OUT}\\base-*.txt and ${DIGEST}`);
} else {
  console.log(fails ? `FAIL ${fails} sections differ` : 'PASS every section as the baseline');
  process.exitCode = fails ? 1 : 0;
}
