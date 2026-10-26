// T-1413: renders the building sounds (map/buildVoices.ts) offline in headless Chrome, through the same
// compressor the game uses, to WAV files in artifacts/trainworld/t1413/, and prints each one's peak,
// level and where it sounds (so lengths and envelopes can be checked against the design).
//   node trainworld/checks/t1413-render.mjs [url]     (from the maps root; a running dev server)
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, launchChrome } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6143/';
const OUT = join(ARTIFACTS, 't1413');
const SR = 48000;
await mkdir(OUT, { recursive: true });

const { ctx, close } = await launchChrome({ prefix: 'tw-t1413-', size: [600, 400] });
try {
  const page = await ctx.newPage();
  await page.route('**/t1413-blank.html', (r) => r.fulfill({ contentType: 'text/html', body: '<!doctype html><title>t1413</title>' }));
  await page.goto(`${BASE}t1413-blank.html`);
  const rendered = await page.evaluate(async ({ SR }) => {
    const v = await import(`/src/map/buildVoices.ts?t=${Date.now()}`);
    const LEVEL = 0.5; // MASTER at volume 1
    // the compressor's first second of a fresh context plays louder than a running game's, so every
    // sound starts a second in, and that second is cut off
    const render = async (secs, play) => {
      const off = new OfflineAudioContext(1, Math.ceil(SR * (secs + 0.95)), SR);
      const comp = off.createDynamicsCompressor();
      comp.connect(off.destination);
      play(off, comp);
      return Array.from((await off.startRendering()).getChannelData(0)).slice(Math.round(SR * 0.95));
    };
    const rnd = () => 0.5;
    return {
      'build-kecha': await render(0.6, (c, d) => v.keCha(c, d, 1, LEVEL, rnd)),
      'station-bung': await render(0.7, (c, d) => v.bung(c, d, 1, LEVEL)),
      'track-tsh': await render(0.4, (c, d) => v.tsh(c, d, 1, LEVEL)),
      'all-in-a-row': await render(2.4, (c, d) => { v.keCha(c, d, 1.05, LEVEL, rnd); v.bung(c, d, 1.85, LEVEL); v.tsh(c, d, 2.65, LEVEL); }),
    };
  }, { SR });

  const wav = (x) => {
    const b = Buffer.alloc(44 + x.length * 2);
    b.write('RIFF', 0); b.writeUInt32LE(36 + x.length * 2, 4); b.write('WAVEfmt ', 8);
    b.writeUInt32LE(16, 16); b.writeUInt16LE(1, 20); b.writeUInt16LE(1, 22); b.writeUInt32LE(SR, 24);
    b.writeUInt32LE(SR * 2, 28); b.writeUInt16LE(2, 32); b.writeUInt16LE(16, 34); b.write('data', 36); b.writeUInt32LE(x.length * 2, 40);
    for (let i = 0; i < x.length; i++) b.writeInt16LE(Math.round(Math.max(-1, Math.min(1, x[i])) * 32767), 44 + i * 2);
    return b;
  };
  for (const [name, x] of Object.entries(rendered)) {
    await writeFile(join(OUT, `${name}.wav`), wav(x));
    // 2 ms RMS envelope in dB re the file's peak; sounding = within 40 dB of it
    const w = Math.round(SR * 0.002), env = [];
    let peak = 0, sum = 0;
    for (const s of x) { peak = Math.max(peak, Math.abs(s)); sum += s * s; }
    for (let i = 0; i + w <= x.length; i += w) { let e = 0; for (let j = 0; j < w; j++) e += x[i + j] ** 2; env.push(20 * Math.log10(Math.sqrt(e / w) / peak + 1e-9)); }
    const runs = [];
    let from = -1, quiet = 0;
    env.forEach((d, i) => {
      if (d > -40) { if (from < 0) from = i; quiet = 0; }
      else if (from >= 0 && ++quiet >= 8) { runs.push([from, i - quiet]); from = -1; }
    });
    if (from >= 0) runs.push([from, env.length - 1]);
    const ms = (i) => Math.round(i * w / SR * 1000);
    const runPeak = ([a, b]) => { let p = 0; for (let i = a * w; i < Math.min(x.length, (b + 1) * w); i++) p = Math.max(p, Math.abs(x[i])); return p.toFixed(3); };
    console.log(`${name}.wav: peak ${peak.toFixed(3)} (${(20 * Math.log10(peak)).toFixed(1)} dBFS), rms ${(20 * Math.log10(Math.sqrt(sum / x.length))).toFixed(1)} dBFS over the file; within 40 dB of the peak at ms ${runs.map((r) => `${ms(r[0])}-${ms(r[1] + 1)} (peak ${runPeak(r)})`).join(', ')}`);
  }
  console.log(`files in ${OUT}`);
} finally {
  await close();
}
