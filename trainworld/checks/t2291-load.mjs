// T-2291: load a pure app module (game/colourPick.ts, game/palette.ts) into Node: its types stripped
// (node:module, Node 22.13+), its relative imports pointed at stripped siblings, the signals library
// (palette.ts's `modeScheme`, unused by checks) stubbed. Load `palette` before a module that imports it.
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { stripTypeScriptTypes } from 'node:module';

export const dir = mkdtempSync(join(tmpdir(), 'tw-t2291-'));
const GAME = fileURLToPath(new URL('../app/src/game/', import.meta.url));

export async function loadGame(name) {
  const src = readFileSync(GAME + `${name}.ts`, 'utf8').replace(/^import \{ signal \} from "@preact\/signals";$/m, 'const signal = (value) => ({ value });');
  const js = stripTypeScriptTypes(src).replace(/from "\.\/(\w+)"/g, 'from "./$1.mjs"');
  writeFileSync(join(dir, name + '.mjs'), js);
  return import(pathToFileURL(join(dir, name + '.mjs')).href);
}
