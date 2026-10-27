// T-1504 probe command: every pair of places along line 113 (Beijing - Shanghai), both ways.
const reg = await (await fetch('/src/game/registry.json')).json().catch(() => null);
const ids = ['bei', 'tia', 'jin', 'nan', 'chz', 'suz', 'sha'];
const P = Object.fromEntries((reg?.cities ?? []).filter((c) => ids.includes(c.id)).map((c) => [c.id, c.place]));
const out = { P };
for (const a of ids) for (const b of ids) if (a < b || true) if (a !== b) {
  const r = await window.__t.ask(P[a], P[b]);
  out[`${a}-${b}`] = r && { day: r.day, served: r.big?.served, why: r.big?.why, railP: r.big?.railP, carP: r.big?.carP, airShare: r.day.air / (r.day.trips || 1), pairs: r.big?.pairs?.slice(0, 2) };
}
return out;
