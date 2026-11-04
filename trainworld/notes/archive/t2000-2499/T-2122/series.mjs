// T-2122: a phone.mjs run's samples over time: renderer, WASM per worker, JS heap and backing stores per target
import { readFile } from 'node:fs/promises';
const r = JSON.parse(await readFile(process.argv[2], 'utf8'));
const every = Number(process.argv[3] ?? 1);
r.samples.forEach((s, i) => {
  if (i % every) return;
  const w = (s.page?.workers ?? []).map((x) => x.wasm).join('+');
  const f = (s.page?.focus ?? []).join('+');
  const js = (s.js ?? []).map((h) => `${h.url.replace(/\.(worker\.)?(ts|mjs)$/, '').slice(0, 10)}:${h.total}/${h.ab ?? '?'}`).join(' ');
  const jsum = (s.js ?? []).reduce((a, h) => a + (h.total ?? 0) + (h.ab ?? 0), 0);
  console.log(`${s.t.toFixed(0).padStart(4)} ${s.phase.padEnd(5)} rend ${String(s.rend).padStart(5)} all ${String(s.total).padStart(5)} wasm ${w}${f ? ' focus ' + f : ''} js+ab ${jsum} | ${js}`);
});
