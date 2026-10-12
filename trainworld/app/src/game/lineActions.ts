import { edit, nextFreeColour } from "./world";
import { lineDraft, renameLinePrompt, selection, tool } from "./ui";
import { live } from "../map/live";

/** Reveal the line before asking for stops, so even an empty line can be named and edited. */
export async function createLine() {
  const a = await edit({ op: "addLine", stops: [], name: "", colour: nextFreeColour() });
  if (!a.ok || a.line === undefined) return;
  live.tools?.exit();
  lineDraft.value = { line: a.line, stops: [] };
  renameLinePrompt.value = String(a.line);
  selection.value = { kind: "line", line: String(a.line) };
  tool.value = "line";
}
