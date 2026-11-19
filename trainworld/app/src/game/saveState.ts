// T-3774: what a feedback report says about the player's saves: counts and short reason codes only, never a game's
// name, contents or an account id. "Lost my save" reports are read against it (tools/ops/feedback.py prints it).

import { session } from "./account";
import { readOpenFailures } from "./openFailures";
import { countUnreadable, listSlots } from "../workers/saveFile";

const short = (s: string | undefined) => (s ?? "").replace(/\s+/g, " ").trim().slice(0, 80);

export async function saveStateReport(active: string): Promise<string> {
  const parts: string[] = [];
  try {
    const slots = await listSlots();
    const playing = slots.find((s) => s.key === active);
    parts.push(`slots ${slots.length}`, `damaged ${slots.filter((s) => s.damaged).length}`);
    parts.push(`playing ${playing ? (playing.cloud?.error ? "sync error: " + short(playing.cloud.error) : playing.cloud?.acct ? "synced" : "local only") : "none"}`);
  } catch (e) {
    parts.push("slots unreadable: " + short((e as Error | null)?.name ?? String(e)));
  }
  try { parts.push(`unreadable copies ${await countUnreadable()}`); } catch { parts.push("unreadable copies ?"); }
  parts.push(`signed in ${session.peek() ? "yes" : "no"}`);
  try { const p = await navigator.storage?.persisted?.(); parts.push(`persisted ${p === undefined ? "?" : p ? "yes" : "no"}`); } catch { parts.push("persisted ?"); }
  const f = readOpenFailures(), last = f.recent[0];
  parts.push(`open failures ${f.count}` + (last ? `, last ${last.from}: ${short(last.why !== undefined ? last.error + " (" + last.why + ")" : last.error)} (${last.version})` : ""));
  return parts.join("; ");
}
