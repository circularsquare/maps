// T-3269, T-3279, T-3328: the flags beside players' names, all 4:3 (640 x 480). From the list in app/src/game/flags.json:
// countries, the EU, the UN and three Spanish regions (`icons`) from flag-icons (MIT); Japanese prefectures and the Spanish, Portuguese and Indonesian places from Wikimedia Commons (public domain or CC0,
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
  // Andalusia: green, white, green (the official flag has the arms on the white; Commons' drawing of them is CC BY-SA)
  "es-an": stripes(["#00843d", "#fff", "#00843d"]),
  // Madeira: blue, gold, blue upright with the red Cross of the Order of Christ on the gold (simplified; Commons' file is CC BY-SA)
  "pt-30": svg(rect("#00247d", 0, 0, 640, 480) + rect("#ffcc00", 150, 0, 340, 480)
    + `<path fill="#d80027" d="M270 130h100v100h100v100H370v100H270V330H170V230h100z"/><path fill="#fff" d="M308 168h24v100h100v24H332v100h-24V292H208v-24h100z"/>`),
  // Brittany, Gwenn-ha-du: five black and four white stripes, ermine spots in a white canton (Commons' file is CC BY-SA)
  "fr-bre": svg(rect("#fff", 0, 0, 640, 480) + Array.from({ length: 5 }, (_, i) => rect("#000", 0, r1((i * 2 * 480) / 9), 640, r1(480 / 9) + 0.5)).join("")
    + rect("#fff", 0, 0, 249, 267) + `<path fill="#000" d="${[[42, 36], [104, 36], [166, 36], [228, 36], [73, 98], [135, 98], [197, 98], [42, 160], [104, 160], [166, 160], [228, 160]]
    .filter(([x]) => x < 240).map(([x, y]) => `M${x} ${y + 40}l-6-26h12zM${x} ${y - 6}m-9 6a9 9 0 1 0 18 0a9 9 0 1 0-18 0`).join("")}"/>`),
  // Paris: Commons' own flag, a blue left half and a red right half (a square there, cropped here)
  paris: svg(rect("#ed2939", 0, 0, 640, 480) + rect("#002395", 0, 0, 320, 480)),
  // Corsica: the Moor's head in profile, a white band on his brow, on white (simplified from the CC0 drawing)
  "fr-cor": svg(rect("#fff", 0, 0, 640, 480) + `<path fill="#000" d="M300 70c70-26 150 8 150 90 0 36-14 60-30 80l24 58-62 30-14 56H300l-12-50c-60-12-92-60-92-122 0-60 34-120 104-142z"/>`
    + `<path fill="#fff" d="M200 140c36-22 148-28 244 12l-6 36c-94-38-196-30-244-4z"/><path fill="#000" d="M356 196l22-14 4 24z"/>`),
  // Lisbon and Porto: eight triangles from the centre, black and white (Lisbon) or green and white (Porto), the city arms
  // on the middle (Commons' drawings of the arms are 1 to 2 MB, so the arms are simplified here)
  lisbon: svg(rect("#fff", 0, 0, 640, 480) + `<path fill="#000" d="M320 240L0 0h320zM320 240L640 0v240zM320 240L640 480H320zM320 240L0 480V240z"/>`
    + `<path fill="none" stroke="#fff" stroke-width="30" d="M200 330A140 140 0 0 1 200 150M440 330A140 140 0 0 0 440 150"/>`
    + `<path fill="#e8b400" stroke="#000" stroke-width="5" d="M244 170h152v82q0 44-76 74-76-30-76-74z"/><path fill="#2a8a45" d="M244 262q19 12 38 0t38 0 38 0 38 0v8q0 8-12 22-64 20-140 0-4-14-4-22z"/>`
    + `<path fill="#000" d="M300 200h40v36h-40z"/><path fill="#e8b400" stroke="#000" stroke-width="4" d="M262 150l12-24 18 14 28-24 28 24 18-14 12 24z"/>`),
  porto: svg(rect("#fff", 0, 0, 640, 480) + `<path fill="#447a4b" d="M320 240L0 0h320zM320 240L640 0v240zM320 240L640 480H320zM320 240L0 480V240z"/>`
    + `<path fill="none" stroke="#fff" stroke-width="30" d="M200 330A140 140 0 0 1 200 150M440 330A140 140 0 0 0 440 150"/>`
    + `<path fill="#2a3bf0" stroke="#aaa" stroke-width="5" d="M244 170h152v82q0 44-76 74-76-30-76-74z"/><path fill="#c9a227" d="M278 190h26v18h16v-18h16v18h16v-18h26v80h-100z"/><path fill="#c0201f" d="M310 238h20v32h-20z"/>`
    + `<path fill="#2a8a45" d="M244 282q19 12 38 0t38 0 38 0 38 0q0 22-76 44-76-22-76-44z"/><path fill="#999" stroke="#555" stroke-width="4" d="M262 150l12-24 18 14 28-24 28 24 18-14 12 24z"/>`),
  // Navarre: red, a gold crown over a red shield with gold chains (simplified)
  "es-nc": svg(rect("#d8101a", 0, 0, 640, 480) + `<path fill="#e8b400" d="M262 170l12-44 24 20 22-34 22 34 24-20 12 44z"/>`
    + `<path fill="#d8101a" stroke="#e8b400" stroke-width="10" d="M268 190h104v110q0 36-52 52-52-16-52-52z"/><path fill="none" stroke="#e8b400" stroke-width="9" d="M280 206l80 130M360 206l-80 130M320 200v140M276 270h88"/><circle cx="320" cy="270" r="12" fill="#2a8a45"/>`),
  // Jakarta: the arms on white, a yellow-edged blue shield with a white monument and a red band for its name (simplified)
  "id-jk": svg(rect("#fff", 0, 0, 640, 480) + `<path fill="#f8c300" d="M218 108h204v26l-10 10v100q0 66-92 104-92-38-92-104V144l-10-10z"/>`
    + `<path fill="#fff" d="M236 124h168v16l-10 8v96q0 52-74 86-74-34-74-86v-96l-10-8z"/><rect x="236" y="124" width="168" height="18" fill="#fff"/><rect x="248" y="126" width="144" height="12" fill="#d8101a"/>`
    + `<path fill="#0b4ea2" d="M248 156h144v82q0 46-72 80-72-34-72-80z"/><path fill="#fff" d="M314 170h12v110h-12z"/><path fill="#fff" d="M296 280h48v14h-48z"/>`),
  // T-3351: the four Spanish communities and two Indonesian provinces whose Commons files are CC BY / BY-SA or heavy, redrawn
  // from the official descriptions with plain shapes (the arms become a small crowned shield, or a simple emblem)
  // Aragon: four red bars on gold, a small quartered shield under a crown on the hoist side
  "es-ar": svg(Array.from({ length: 9 }, (_, i) => rect(i % 2 ? "#da121a" : "#fcdd09", 0, r1((i * 480) / 9), 640, r1(480 / 9) + 1)).join("")
    + `<g transform="translate(180 270) scale(1.35) translate(-180 -270)"><clipPath id="a"><path d="M130 200h100v80q0 50-50 70-50-20-50-70z"/></clipPath><path fill="#fcdd09" stroke="#7a6500" stroke-width="4" d="M134 190l8-30 18 14 20-26 20 26 18-14 8 30z"/>`
    + `<g clip-path="url(#a)"><path fill="#fff" d="M130 200h100v150H130z"/><path fill="#0072bc" d="M180 200h50v75h-50z"/><path fill="#fcdd09" d="M180 275h50v75h-50z"/>`
    + `<path fill="#da121a" d="M130 258h50v14h-50zM148 200h14v75h-14zM130 330h50v12h-50zM190 275h6v75h-6zM204 275h6v75h-6zM218 275h6v75h-6z"/><path fill="#fff" d="M200 215h10v50h-10zM192 228h26v10h-26z"/></g>`
    + `<path fill="none" stroke="#333" stroke-width="5" d="M130 200h100v80q0 50-50 70-50-20-50-70z"/></g>`),
  // Cantabria: white over red, the arms in the middle: a crown on a blue and red shield with a grey disc
  "es-cb": svg(rect("#fff", 0, 0, 640, 240) + rect("#da121a", 0, 240, 640, 240)
    + `<g transform="translate(320 270) scale(1.35) translate(-320 -270)"><path fill="#da121a" stroke="#7a1a1a" stroke-width="4" d="M278 196l-6-40 22 14 26-34 26 34 22-14-6 40z"/><path fill="#fcdd09" d="M278 192h84v16h-84z"/>`
    + `<path fill="#0072bc" d="M272 208h96v62h-96z"/><path fill="#da121a" d="M272 270h96v18q0 50-48 70-48-20-48-70z"/><circle cx="320" cy="304" r="30" fill="#c8c8c8" stroke="#888" stroke-width="3"/>`
    + `<path fill="#fff" d="M290 244h60v10h-60zM304 222h10v22h-10z"/><path fill="none" stroke="#333" stroke-width="5" d="M272 208h96v80q0 50-48 70-48-20-48-70z"/></g>`),
  // Castile and Leon: quarters, a gold castle on red and a purple lion on white
  "es-cl": svg(rect("#da121a", 0, 0, 640, 480) + rect("#fff", 320, 0, 320, 240) + rect("#fff", 0, 240, 320, 240)
    + [[160, 120], [480, 360]].map(([x, y]) => `<g transform="translate(${x} ${y}) scale(1.45)"><path fill="#fcdd09" stroke="#7a6500" stroke-width="4" d="M-56-14h20v-18h14v18h14v-34h12v-14h12v14h12v34h14v-18h14v18h20v78H-56z"/>`
      + `<path fill="#0072bc" d="M-10 28h20v36h-20zM-44 0h10v18h-10zM34 0h10v18H34z"/></g>`).join("")
    + [[480, 120], [160, 360]].map(([x, y]) => `<g transform="translate(${x} ${y}) scale(1.12)" fill="#7b2f6b" stroke="#7b2f6b" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="6" cy="10" rx="28" ry="42" transform="rotate(-14 6 10)" stroke="none"/>`
      + `<circle cx="-12" cy="-48" r="30" stroke="none"/><circle cx="43" cy="-34" r="9" stroke="none"/><path fill="none" stroke-width="17" d="M-18-14L-54-40M-6-4L-52-14M10 44L-8 82M30 44L46 80"/><path fill="none" stroke-width="9" d="M26 34C64 28 58-10 42-32"/><path stroke="none" d="M-34-60l-6-18 16 6zM-40-44l-24 4 18 10z"/></g>`).join("")),
  // La Rioja: red, white, green, yellow, a small blue shield under a crown on the white and green
  "es-ri": svg(rect("#da291c", 0, 0, 640, 120) + rect("#fff", 0, 120, 640, 120) + rect("#6abf2a", 0, 240, 640, 120) + rect("#ffc22d", 0, 360, 640, 120)
    + `<g transform="translate(320 230) scale(1.3) translate(-320 -230)"><path fill="#da121a" stroke="#7a1a1a" stroke-width="4" d="M282 138l-6-36 22 12 22-28 22 28 22-12-6 36z"/><path fill="#fcdd09" d="M282 134h76v14h-76z"/>`
    + `<path fill="#0a6ebd" d="M276 152h88v82q0 54-44 72-44-18-44-72z"/><path fill="#6abf2a" d="M276 250q22-20 44-6t44-6q0 40-44 62-44-18-44-50z"/>`
    + `<path fill="#da121a" d="M296 170h10v34h-10zM288 180h26v9h-26z"/><path fill="#fcdd09" d="M330 176h24v36h-24zM330 170h6v6h-6zM348 170h6v6h-6z"/><path fill="none" stroke="#333" stroke-width="5" d="M276 152h88v82q0 54-44 72-44-18-44-72z"/></g>`),
  // Bali: orange with a blue pentagon holding a gold tiered temple roof under a star
  "id-ba": svg(rect("#ff9955", 0, 0, 640, 480) + `<path fill="#4a90d9" stroke="#2f6aa8" stroke-width="5" d="${[0, 1, 2, 3, 4].map((i) => `${r1(320 + 120 * Math.sin((i * 2 * Math.PI) / 5))} ${r1(250 - 120 * Math.cos((i * 2 * Math.PI) / 5))}`).join("L").replace(/^/, "M")}Z"/>`
    + `<path fill="#f0b800" d="${star(320, 170, 14, 5, 0.4)}"/><path fill="#e0a400" d="M318 186h4v14h-4zM304 200h32l-6 18h-20zM292 220h56l-8 20h-40zM280 242h80l-10 24h-60zM292 268h56v46h-56z"/><path fill="#4a90d9" d="M310 284h20v30h-20z"/><path fill="#e0a400" d="M268 306h104v14H268z"/>`),
  // Yogyakarta: dark blue, a red disc in a gold wreath under a gold star, a white tower and gold wings, a white ribbon
  "id-yo": svg(rect("#00004f", 0, 0, 640, 480) + `<path fill="#f8c300" d="${star(320, 92, 42, 5, 0.4)}"/><circle cx="320" cy="262" r="136" fill="none" stroke="#f8c300" stroke-width="22" stroke-dasharray="14 5"/>`
    + `<circle cx="320" cy="262" r="112" fill="#e32b24"/><path fill="none" stroke="#f8c300" stroke-width="12" stroke-linecap="round" d="M300 270C250 250 240 200 252 168M340 270C390 250 400 200 388 168M300 296C240 280 224 230 232 196M340 296C400 280 416 230 408 196"/>`
    + `<path fill="#fff" d="M314 160h12l10 150h-32z"/><path fill="#f8c300" d="M298 310h44v22h-44z"/><path fill="#fff" stroke="#999" stroke-width="3" d="M240 372h160l-14 34H254z"/>`),
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
  else if (m.icons) { fromIcons(m.code); codes.push(m.code); }
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
