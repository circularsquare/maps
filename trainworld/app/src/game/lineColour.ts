import { edit, stationsById, world } from "./world";
import { newLine } from "./ui";
import { stationZone } from "./listScope";
import { besideColours, settledColour, type PlacedLine } from "./colourPick";

/** T-2291: the line whose colour the game picked, and which colour. Only that line, only while it is the
 * new line (`newLine`), and only while it still has that colour: a colour the player sets stays. */
let picked: { line: number; colour: string } | null = null;

export function rememberPicked(line: number, colour: string) {
  picked = { line, colour: colour.toLowerCase() };
}

/** T-2291: after a stop edit on the new line, give it a palette colour farthest from the lines beside it
 * if the one it has reads as one of theirs. A line whose colour the player changed is left alone. */
export async function settleNewLineColour(num: number) {
  const w = world.peek();
  const line = w?.save.lines.find((l) => l.num === num);
  if (!w || !line || picked?.line !== num || newLine.peek()?.line !== num || line.colour.toLowerCase() !== picked.colour) return;
  const at = stationsById(w);
  const where = (s: string): [number, number] | null => { const p = at.get(s); return p ? [p.lng, p.lat] : null; };
  const placed: PlacedLine[] = w.save.lines.map((l) => ({ id: l.id, colour: l.colour, stops: l.stops }));
  const self = placed.find((l) => l.id === line.id)!;
  const beside = besideColours(self, placed, w.edges.map((e) => e.lines), where);
  const first = at.get(line.stops[0]);
  const city = first ? stationZone(first) : undefined;
  const mates = city === undefined ? Infinity : w.save.lines.filter((l) => { const s = at.get(l.stops[0]); return s && stationZone(s) === city; }).length - 1;
  const colour = settledColour(line.colour, beside, mates);
  if (colour.toLowerCase() === picked.colour) return;
  const a = await edit({ op: "lineLook", line: num, colour }, true);
  if (a.ok) picked = { line: num, colour: colour.toLowerCase() };
}
