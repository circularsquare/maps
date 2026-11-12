// T-3269, T-3279: the flags beside players' names, all 4:3 (640 x 480). From the list in app/src/game/flags.json:
// countries, the EU and the UN from flag-icons (MIT); Japanese prefectures from Wikimedia Commons (public domain,
// fitted to 4:3 by cropping the sides); Earth and Pirate from circle-flags (MIT) with the circle taken off; the rest
// drawn here. Writes app/public/flags/<code>.svg, both licences beside them, shonei-server's trainworld_flags.json
// (the codes it accepts) and app/src/game/tzCountry.json (time zone to country, for the picker's first country, from
// the venv's tzdata). Rerun, and redeploy the server, after a change to flags.json. Usage (maps root):
//   node trainworld/tools/flags.mjs <flag-icons package> <circle-flags package> <commons folder> [shonei-server]
// The packages: `npm pack flag-icons@7.6.0` and `npm pack circle-flags@2.8.3`, untarred. A Commons file missing from
// the folder is fetched into it.
import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { join } from "node:path";

const here = fileURLToPath(new URL(".", import.meta.url));
const [icons, circle, commons] = process.argv.slice(2);
if (!commons) { console.error("usage: node trainworld/tools/flags.mjs <flag-icons> <circle-flags> <commons folder> [shonei-server]"); process.exit(1); }
const server = process.argv[5] ?? "C:/Users/anita/projects/shonei-server";
const tzdata = join(here, "../../venv/Lib/site-packages/tzdata/zoneinfo");
const list = JSON.parse(readFileSync(join(here, "../app/src/game/flags.json"), "utf8"));
const out = join(here, "../app/public/flags");
mkdirSync(out, { recursive: true });
for (const f of readdirSync(out)) rmSync(join(out, f));

const r1 = (n) => Math.round(n * 10) / 10;
const svg = (body) => `<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480" viewBox="0 0 640 480">${body}</svg>\n`;
const rect = (fill, x, y, w, h) => `<path fill="${fill}" d="M${x} ${y}h${w}v${h}H${x}z"/>`;
const stripes = (colours) => svg(colours.map((c, i) => rect(c, 0, r1((480 * i) / colours.length), 640, r1(480 / colours.length) + 1)).join(""));
function star(cx, cy, r, n, inner) {
  const pts = [];
  for (let i = 0; i < 2 * n; i++) {
    const a = -Math.PI / 2 + (i * Math.PI) / n, rr = i % 2 ? r * inner : r;
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
/** a circle-flags 512 square without its circle, widened to 4:3 on its own ground colour */
function unround(file, ground) {
  const s = readFileSync(join(circle, "flags", file), "utf8");
  const body = s.replace(/^<svg[^>]*>/, "").replace(/<\/svg>\s*$/, "").replace(/<mask[\s\S]*?<\/mask>/, "").replace(/ mask="url\(#a\)"/, "");
  return `<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480" viewBox="-85.3 0 682.7 512"><path fill="${ground}" d="M-86 0h684v512H-86z"/>${body}</svg>\n`;
}
/** a Commons flag, cropped at the sides (or top and bottom) to 4:3 */
function fit(s) {
  s = s.replace(/<\?xml[\s\S]*?\?>/, "").replace(/<!DOCTYPE[\s\S]*?>/, "").replace(/<!--[\s\S]*?-->/g, "").replace(/<metadata[\s\S]*?<\/metadata>/g, "").trim();
  const head = s.match(/^<svg[^>]*>/)[0];
  const attr = (k) => head.match(new RegExp(`\\s${k}="([^"]*)"`))?.[1];
  const vb = attr("viewBox") ?? `0 0 ${parseFloat(attr("width"))} ${parseFloat(attr("height"))}`;
  if (/NaN/.test(vb)) throw new Error("no size: " + head);
  const keep = head.replace(/\s(width|height|viewBox|preserveAspectRatio)="[^"]*"/g, "").replace(/>$/, "");
  return `${keep} width="640" height="480" viewBox="${vb}" preserveAspectRatio="xMidYMid slice">` + s.slice(head.length) + "\n";
}
async function commonsFile(name) {
  const p = join(commons, name.replaceAll(" ", "_"));
  if (!existsSync(p)) {
    const url = "https://commons.wikimedia.org/wiki/Special:FilePath/" + encodeURIComponent(name.replaceAll(" ", "_"));
    for (let i = 0; ; i++) {
      const r = await fetch(url, { headers: { "User-Agent": "flag-list-builder/0.1" } });
      if (r.ok) { writeFileSync(p, Buffer.from(await r.arrayBuffer())); break; }
      if (i >= 5) throw new Error(`${name}: ${r.status}`);
      await new Promise((ok) => setTimeout(ok, 4000 * (i + 1)));
    }
  }
  const s = readFileSync(p, "utf8");
  if (/<script|<image|<foreignObject|href="(?!#)/i.test(s)) throw new Error(`${name}: not a plain drawing`);
  return s;
}

// drawn for the game: simple designs that are public domain, or the game's own (train)
const drawn = {
  train: svg(rect("#338af3", 0, 0, 640, 480) + `<path fill="#333" d="M264 375h26l-26 105h-30zm86 0h26l19 105h-30zM305 71h30v38h-30z"/>`
    + `<rect x="221" y="101" width="199" height="281" rx="41" fill="#eee"/><rect x="247" y="137" width="146" height="98" rx="17" fill="#333"/>`
    + rect("#d80027", 221, 264, 199, 32) + `<circle cx="264" cy="335" r="19" fill="#ffda44"/><circle cx="377" cy="335" r="19" fill="#ffda44"/>`),
  rainbow: stripes(["#e40303", "#ff8c00", "#ffed00", "#008026", "#24408e", "#732982"]),
  trans: stripes(["#5bcefa", "#f5a9b8", "#fff", "#f5a9b8", "#5bcefa"]),
  chequered: svg(rect("#fff", 0, 0, 640, 480) + `<path fill="#111" d="${Array.from({ length: 48 }, (_, i) => [i % 8, Math.floor(i / 8)])
    .filter(([x, y]) => (x + y) % 2).map(([x, y]) => `M${x * 80} ${y * 80}h80v80h-80z`).join("")}"/>`),
  // red, black, red, three white saltires on the black
  amsterdam: svg(rect("#ec0000", 0, 0, 640, 480) + rect("#111", 0, 160, 640, 160)
    + `<path fill="#fff" d="${[200, 320, 440].map((x) => saltire(x, 240, 46, 26)).join("")}"/>`),
  // white, two light blue stripes, four red six-pointed stars
  chicago: svg(rect("#fff", 0, 0, 640, 480) + rect("#41b6e6", 0, 80, 640, 80) + rect("#41b6e6", 0, 320, 640, 80)
    + `<path fill="#e4002b" d="${[170, 270, 370, 470].map((x) => star(x, 240, 44, 6, 0.55)).join("")}"/>`),
  // white, three red stars over two red bars
  washington: svg(rect("#fff", 0, 0, 640, 480) + rect("#e91436", 0, 196, 640, 92) + rect("#e91436", 0, 334, 640, 92)
    + `<path fill="#e91436" d="${[170, 320, 470].map((x) => star(x, 104, 54, 5, 0.382)).join("")}"/>`),
  // blue, white, orange upright; the seal, simplified to a blue ring round a disc, in the white
  nyc: svg(rect("#003884", 0, 0, 214, 480) + rect("#fff", 213, 0, 214, 480) + rect("#ff6319", 426, 0, 214, 480)
    + `<circle cx="320" cy="240" r="70" fill="none" stroke="#003884" stroke-width="12"/><circle cx="320" cy="240" r="40" fill="#003884"/>`),
  // Hiroshima: the white emblem, a crescent open to the upper right, on deep rose
  "jp-34": svg(rect("#ab3748", 0, 0, 640, 480)
    + `<circle cx="320" cy="240" r="132" fill="#fff"/><circle cx="352" cy="208" r="92" fill="#ab3748"/>`),
  // Kagawa: the white emblem, a ka in an olive leaf (an arc and two arcs meeting at the top), on green
  "jp-37": svg(rect("#005933", 0, 0, 640, 480) + `<clipPath id="k"><path d="M160 86h320v288H160z"/></clipPath>`
    + `<g clip-path="url(#k)" fill="none" stroke="#fff" stroke-width="64"><circle cx="80" cy="406" r="272"/><path d="M448 406A368 368 0 0 0 301 112m38 0A368 368 0 0 0 192 406"/></g>`),
};

const codes = [];
const put = (code, body) => { writeFileSync(join(out, code + ".svg"), body); codes.push(code); };
const fromIcons = (code) => copyFileSync(join(icons, "flags/4x3", code + ".svg"), join(out, code + ".svg"));
for (const cc of list.countries) { fromIcons(cc); codes.push(cc); }
for (const m of [...list.places, ...list.more]) {
  if (m.code === "eu" || m.code === "un") { fromIcons(m.code); codes.push(m.code); }
  else if (m.code === "earth") put(m.code, unround("other/earth.svg", "#0052b4"));
  else if (m.code === "pirate") put(m.code, unround("other/jolly_roger.svg", "#333"));
  else if (m.commons) put(m.code, fit(await commonsFile(m.commons)));
  else if (drawn[m.code]) put(m.code, drawn[m.code]);
  else throw new Error(`no picture for ${m.code}`);
}
copyFileSync(join(icons, "LICENSE"), join(out, "LICENSE-flag-icons.txt"));
copyFileSync(join(circle, "LICENSE.md"), join(out, "LICENSE-circle-flags.md"));
const dest = join(server, "trainworld_flags.json");
writeFileSync(dest, JSON.stringify(codes) + "\n");

// time zone to country: zone.tab's own zones, then the links (Asia/Calcutta) to them
const tz = {};
for (const line of readFileSync(join(tzdata, "zone.tab"), "utf8").split("\n")) {
  const f = line.split("\t");
  if (f.length >= 3 && !line.startsWith("#")) tz[f[2]] = f[0].toLowerCase();
}
for (const line of readFileSync(join(tzdata, "tzdata.zi"), "utf8").split("\n")) {
  const [l, target, alias] = line.split(" ");
  if (l === "L" && tz[target] && !tz[alias]) tz[alias] = tz[target];
}
const known = new Set(list.countries);
const tzOut = Object.fromEntries(Object.entries(tz).filter(([, cc]) => known.has(cc)).sort(([a], [b]) => (a < b ? -1 : 1)));
writeFileSync(join(here, "../app/src/game/tzCountry.json"), JSON.stringify(tzOut) + "\n");
console.log(`${codes.length} flags -> ${out}; codes -> ${dest}; ${Object.keys(tzOut).length} time zones`);
