import fs from "node:fs";
const d = JSON.parse(fs.readFileSync(new URL("./cases.json", import.meta.url)));
const bands = ["<1", "1-2", "2-3", "3-5", "5-8", "8-12"];
const cats = [...new Set(d.cases.map((c) => c.cat))].sort();
console.log("cat".padEnd(46), bands.map((b) => b.padStart(6)).join(""));
for (const cat of cats) console.log(cat.padEnd(46), bands.map((b) => String(d.cases.filter((c) => c.cat === cat && c.band === b).length).padStart(6)).join(""));
console.log("\nper city, street-end-near-street by band");
for (const city of Object.keys(d.perCity)) {
  const cs = d.cases.filter((c) => c.city === city && c.cat === "street end near street");
  console.log(city.padEnd(12), bands.map((b) => String(cs.filter((c) => c.band === b).length).padStart(6)).join(""), " ends", d.perCity[city].ends, " free", d.perCity[city].free);
}
console.log("\nbridging service pieces per city", Object.entries(d.perCity).map(([c, p]) => c + ":" + p.bridging.length).join(" "));
console.log("links", Object.entries(d.perCity).map(([c, p]) => c + ":" + p.links.kept + "/" + p.links.raw).join(" "));
