// T-3269: the flags beside players' names. Copies each flag in app/src/game/flags.json from an unpacked circle-flags
// package (npm pack circle-flags@2.8.3, then untar) into app/public/flags/<code>.svg, draws the three it lacks, and
// writes shonei-server's trainworld_flags.json (the codes it accepts). Rerun, and redeploy the server, after a change
// to flags.json. Usage (maps root):
//   node trainworld/tools/flags.mjs <circle-flags package folder> [path to shonei-server]
import { copyFileSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { join } from "node:path";

const here = fileURLToPath(new URL(".", import.meta.url));
const src = process.argv[2];
if (!src) { console.error("usage: node trainworld/tools/flags.mjs <circle-flags package folder> [shonei-server]"); process.exit(1); }
const server = process.argv[3] ?? "C:/Users/anita/projects/shonei-server";
const list = JSON.parse(readFileSync(join(here, "../app/src/game/flags.json"), "utf8"));
const out = join(here, "../app/public/flags");
mkdirSync(out, { recursive: true });
for (const f of readdirSync(out)) if (f.endsWith(".svg")) rmSync(join(out, f));

const r1 = (n) => Math.round(n * 10) / 10;
const wrap = (body) => `<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512"><mask id="a"><circle cx="256" cy="256" r="256" fill="#fff"/></mask><g mask="url(#a)">${body}</g></svg>\n`;
function star6(cx, cy, r) {
  const pts = [];
  for (let i = 0; i < 12; i++) {
    const a = -Math.PI / 2 + (i * Math.PI) / 6, rr = i % 2 ? r * 0.55 : r;
    pts.push(`${r1(cx + rr * Math.cos(a))} ${r1(cy + rr * Math.sin(a))}`);
  }
  return "M" + pts.join("L") + "Z";
}
function saltire(cx, cy, s, w) {
  const bar = (dy) => {
    const len = Math.hypot(2 * s, 2 * s * dy), nx = (-2 * s * dy / len) * (w / 2), ny = (2 * s / len) * (w / 2);
    const x0 = cx - s, y0 = cy - s * dy, x1 = cx + s, y1 = cy + s * dy;
    return `M${r1(x0 + nx)} ${r1(y0 + ny)}L${r1(x1 + nx)} ${r1(y1 + ny)}L${r1(x1 - nx)} ${r1(y1 - ny)}L${r1(x0 - nx)} ${r1(y0 - ny)}Z`;
  };
  return bar(1) + bar(-1);
}
// drawn in circle-flags' style (its palette, a 512 box, the circle mask): simple public domain designs
const drawn = {
  // white, two light blue stripes, four red six-pointed stars
  chicago: wrap(`<path fill="#eee" d="M0 0h512v512H0z"/><path fill="#41b6e6" d="M0 96h512v76H0zm0 244h512v76H0z"/>`
    + `<path fill="#d80027" d="${[118, 204, 308, 394].map((x) => star6(x, 256, 46)).join("")}"/>`),
  // red, black, red, three white saltires on the black
  amsterdam: wrap(`<path fill="#d80027" d="M0 0h512v512H0z"/><path fill="#333" d="M0 171h512v170H0z"/>`
    + `<path fill="#eee" d="${[146, 256, 366].map((x) => saltire(x, 256, 44, 26)).join("")}"/>`),
  // a train front on, on its rails, under a blue sky
  train: wrap(`<path fill="#338af3" d="M0 0h512v512H0z"/><path fill="#333" d="M196 400h28l-28 112h-32zm92 0h28l20 112h-32zM240 76h32v40h-32z"/>`
    + `<rect x="150" y="108" width="212" height="300" rx="44" fill="#eee"/><rect x="178" y="146" width="156" height="104" rx="18" fill="#333"/>`
    + `<path fill="#d80027" d="M150 282h212v34H150z"/><circle cx="196" cy="358" r="20" fill="#ffda44"/><circle cx="316" cy="358" r="20" fill="#ffda44"/>`),
};

const codes = [];
for (const m of list.more) {
  if (m.from) copyFileSync(join(src, "flags", m.from), join(out, m.code + ".svg"));
  else if (drawn[m.code]) writeFileSync(join(out, m.code + ".svg"), drawn[m.code]);
  else throw new Error(`no picture for ${m.code}`);
  codes.push(m.code);
}
for (const cc of list.countries) { copyFileSync(join(src, "flags", cc + ".svg"), join(out, cc + ".svg")); codes.push(cc); }
copyFileSync(join(src, "LICENSE.md"), join(out, "LICENSE.md"));
const dest = join(server, "trainworld_flags.json");
writeFileSync(dest, JSON.stringify(codes) + "\n");
console.log(`${codes.length} flags -> ${out}; codes -> ${dest}`);
