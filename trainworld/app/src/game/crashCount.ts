// T-4409: counts the game's failure screens on Anita's own server (shonei-server's POST /trainworld/crash, its
// docs/trainworld-api.md "Failure counts"). One place watches the signals behind every failure notice (a load or
// demand failure, lost graphics, the stopped track model, figures that stopped updating) instead of each screen
// sending its own. What is sent: the build, the first line of the error with numbers and paths stripped so equal
// errors group, and phone or computer. No session, no ids, nothing from the save. At most one send per kind per
// page load. Quiet in the same places as the plays ping (game/playtime.ts); failures are silent.

import { effect } from "@preact/signals";
import { BUILD, quiet } from "./playtime";
import { demandStopped, graphicsGone, loadFailure } from "./loading";
import { figuresStopped } from "./figuresStopped";
import { modelStopped } from "./modelStopped";
import { mobileDevice } from "../workers/deviceMemory";

export interface CrashReport { v: string; line: string; device: "phone" | "computer" }

const MAX_LINE = 120;

/** An error's first line as sent: quoted text (a name a player typed could be in it), URLs and paths, long hex names and
 * numbers replaced, so one error groups. */
export function failureLine(raw: string): string {
  return (raw.split(/\r?\n/)[0] ?? "")
    .replace(/"[^"]*"|'[^']*'|`[^`]*`|“[^”]*”/g, "<text>")
    .replace(/\b[a-z][a-z0-9+.-]*:\/\/\S+/gi, "<url>")
    .replace(/(?<![\w<])(?:[A-Za-z]:)?(?:[\\/][^\s:"'()[\]]+)+/g, "<path>")
    .replace(/\b[0-9a-f]{8,}\b/gi, "N")
    .replace(/\b\d[\d.,]*\b/g, "N")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, MAX_LINE);
}

let started = false;
/** Begin counting through `send`. Idempotent; does nothing on a page that sends nothing. A failure that already
 * happened is sent at once, since each effect runs on its current value. */
export function startCrashCount(send: (r: CrashReport) => Promise<unknown>) {
  if (started || quiet()) return;
  started = true;
  const sent = new Set<string>();
  const report = (kind: string, line: string) => {
    if (sent.has(kind) || !line) return;
    sent.add(kind);
    void send({ v: BUILD, line, device: mobileDevice() ? "phone" : "computer" }).catch(() => {});
  };
  effect(() => { const why = loadFailure.value; if (why) report("load", failureLine(why)); });
  effect(() => { if (demandStopped.value === "graphics") report("graphics", "graphics context lost"); });
  effect(() => { if (graphicsGone.value) report("graphicsGone", "graphics context refused"); });
  effect(() => { if (modelStopped.value) report("model", "track model stopped"); });
  effect(() => { if (figuresStopped.value) report("figures", "figures stopped updating"); });
}
