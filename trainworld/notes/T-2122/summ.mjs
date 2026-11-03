// T-2122: per step of a phone.mjs run, the renderer's peak and what sat in it at that sample
import { readFile } from 'node:fs/promises';
for (const f of process.argv.slice(2)) {
  const r = JSON.parse(await readFile(f, 'utf8'));
  console.log(`== ${r.tag} (${r.url}) dm ${r.dm} hc ${r.hc} crashed ${r.crashed}`);
  for (const [k, s] of Object.entries(r.steps)) {
    if (!s) continue;
    const js = (s.atPeak?.js ?? []).map((h) => `${h.url}:${h.total}${h.ab !== undefined ? '+ab' + h.ab : ''}`).join(' ');
    if (s.uasm) console.log(`      ua-memory ${s.uasm.total} MB: ${Object.entries(s.uasm.by ?? {}).sort((a, b) => b[1] - a[1]).filter(([, v]) => v >= 5).map(([k, v]) => `${k} ${v}`).join(', ')}`);
    const jsSum = (s.atPeak?.js ?? []).reduce((a, h) => a + (h.total ?? 0), 0);
    const wasm = (s.atPeak?.workers ?? []).reduce((a, w) => a + w.wasm, 0);
    console.log(`${k.padEnd(5)} ok ${s.ok ?? '-'} final ${s.finalS ?? '-'}s rail ${s.rail ?? '-'} | renderer peak ${s.peakRenderer} MB, all ${s.peakAll} | at peak: wasm ${wasm} (${(s.atPeak?.workers ?? []).map((w) => w.wasm).join('+')}), js ${jsSum} [${js}] | wasm peaks ${JSON.stringify(s.wasmPeak)}`);
  }
}
