// Save regression for WebKit 254021: native compression streams can silently truncate output.
// Run from maps: node trainworld/checks/safari-save.mjs
// Synthetic files only; Node's independent zlib validates new gzip output.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { gzipSync, gunzipSync, deflateSync } from 'node:zlib';
import { loadApp } from './lib.mjs';

const codec = await loadApp('src/workers/gzipCore.ts');
const saves = await loadApp('src/workers/saveFile.ts');
const real = new Uint8Array(await readFile('trainworld/T-007-real-nyc.save'));
const original = await saves.decodeSave(real);
const native = { CompressionStream, DecompressionStream };
let nativeCalls = 0;
class BrokenStream {
  constructor() {
    nativeCalls++;
    // Model the old engine silently emitting only its header + one 16 KiB flush buffer.
    return new TransformStream({ transform(bytes, controller) { controller.enqueue(bytes.subarray(0, 16394)); } });
  }
}
globalThis.CompressionStream = globalThis.DecompressionStream = BrokenStream;
try {
  for (const length of [0, 35578, 35579, 48891, 65536, 2 << 20]) {
    let seed = 937;
    const raw = Uint8Array.from({ length }, (_, i) => {
      seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5;
      return i % 3 ? i % 251 : seed & 255;
    });
    const t = performance.now();
    const gz = codec.compressGzip(raw);
    assert.deepEqual(new Uint8Array(gunzipSync(gz)), raw, 'independent gzip round-trip');
    assert.deepEqual(codec.decompressGzip(gzipSync(raw)), raw, 'independent gzip input');
    assert.deepEqual(codec.compressGzip(raw), gz, 'deterministic bytes');
    if (length) assert.throws(() => codec.decompressGzip(gz, length - 1), /too large/, 'output cap');
    for (const cut of [1, 8, Math.floor(gz.length / 2)]) {
      assert.throws(() => codec.decompressGzip(gz.subarray(0, gz.length - cut)), /gzip/i, 'truncation rejected');
    }
    const badCRC = gz.slice(); badCRC[badCRC.length - 8] ^= 1;
    assert.throws(() => codec.decompressGzip(badCRC), /gzip/i, 'checksum rejected');
    const badSize = gz.slice(); badSize[badSize.length - 4] ^= 1;
    assert.throws(() => codec.decompressGzip(badSize), /gzip/i, 'size trailer rejected');
    assert.throws(() => codec.decompressGzip(deflateSync(raw)), /gzip/i, 'zlib is not gzip');
    console.log(`PASS ${length} bytes: independent round-trips, corruption, truncation, limit (${Math.round(performance.now() - t)} ms)`);
  }
  const next = await saves.encodeSave(original.header, original.track);
  const decoded = await saves.decodeSave(next);
  assert.deepEqual(decoded.header, original.header);
  assert.deepEqual(decoded.track, original.track);
  assert.equal(gunzipSync(next).subarray(0, 4).toString(), 'TWG1');
  await assert.rejects(saves.decodeSave(next.subarray(0, -8)), /not a trainsworld save/);
  assert.equal(nativeCalls, 0, 'save paths do not call faulty browser codecs');
  console.log(`PASS real NYC save: existing format preserved; ${real.length} -> ${next.length} compressed bytes; faulty native codecs never called`);
} finally {
  Object.assign(globalThis, native);
}
