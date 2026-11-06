// node merge.mjs out.json in1.json in2.json ...
import { readFileSync, writeFileSync } from 'node:fs';
const [out, ...ins] = process.argv.slice(2);
writeFileSync(out, JSON.stringify(ins.flatMap(f => JSON.parse(readFileSync(f, 'utf8')))));
