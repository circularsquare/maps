// T-1447: a selection copied into another saved game as blueprint (notes/T-1447.md). The Select tool's pick goes to the
// clock worker, which keeps it as a clip waiting for that game's next load in this browser; a game that opens with
// clips waiting, in the tab that saves it, pastes each as one undoable edit, and the pasted pieces become the pick.

import { effect, signal } from "@preact/signals";
import { bpPicked, type BpPicked } from "./blueprintPick";
import { gameReady } from "./loading";
import { edit, say, world } from "./world";
import { copyTo, gameStamp, saveSlotsState } from "../workers/clockClient";
import { listClips } from "../workers/saveFile";
import { groupVersions } from "../workers/versions";
import { locateNodes } from "../ui/locate";
import { t } from "../i18n";

/** The pick being copied while the saves panel shows its copy mode; null otherwise. */
export const copying = signal<BpPicked | null>(null);

/** The pick being copied, copied into game `origin`. */
export function copyPickTo(origin: string): Promise<{ ok: boolean; left: number }> {
  const p = copying.peek(), s = saveSlotsState.peek();
  if (!p) return Promise.resolve({ ok: false, left: 0 });
  return copyTo(origin, s.slots.find((x) => x.key === s.active)?.name ?? s.viewing?.name ?? "", p.edges, p.stations);
}

/** The clips waiting for game `origin`, pasted in the order they were copied. */
async function pasteWaiting(origin: string) {
  const clips = await listClips(origin);
  const edges: number[] = [], stations: number[] = [];
  for (const c of clips) {
    const a = await edit({ op: "pasteClip", key: c.key }, true, true);
    const p = a.pasted, why = a.issues[0];
    if (a.ok && p) {
      const ne = p[0], ns = p[1 + ne], left = p[p.length - 1];
      edges.push(...p.slice(1, 1 + ne));
      stations.push(...p.slice(2 + ne, 2 + ne + ns));
      // (a test page's game has no name to give)
      say(!c.from ? t("tools.blueprint") : left ? t("clips.addedLeft", { name: c.from, count: left }) : t("clips.added", { name: c.from }));
    } else if (why === "ClipNewer") say(t("clips.newer"));
    else if (why === "ClipUnreadable") say(t("clips.unreadable"), true);
    else if (why !== "ClipGone" && why !== "ClipNotHere") say(t("clips.noneAdded", { name: c.from }), true);
  }
  if (!edges.length && !stations.length) return;
  bpPicked.value = { edges, stations };
  const nodes = new Set(stations), picked = new Set(edges);
  for (const e of world.peek()?.edges ?? []) if (picked.has(e.id)) nodes.add(e.a).add(e.b);
  locateNodes([...nodes]);
}

// once a game is open, in the tab that saves it (a test page or a viewed network has no slot of its own)
let pastedFor = -1;
effect(() => {
  if (!gameReady.value) return;
  const s = saveSlotsState.value, game = gameStamp.value;
  if (game === pastedFor || !s.writable || !s.active || s.viewing) return;
  // the origin its card has, which the copy panel keys the clip by
  const origin = s.slots.find((x) => x.key === s.active)?.origin
    ?? groupVersions(s.slots, [], {}, s.active).find((g) => g.top.kind === "slot" && g.top.slot.key === s.active)?.origin;
  if (!origin) return;
  pastedFor = game;
  void pasteWaiting(origin);
});
