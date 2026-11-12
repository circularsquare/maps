// T-3313: what a feedback report carries about the game's own state, so a report of figures that stopped,
// a button that never came back or trains that vanished says where it stuck: the clock worker's oldest
// unanswered edit or read and the last hour of trips (main.ts registers its client), the demand pool's
// newest solve and downloads, and how long the page has been open. Plain English, for Anita and agents.

import { demandDiagnosis } from "../workers/demandClient";

let clockLine: () => string = () => "";
export function setClockDiagnosis(f: () => string) { clockLine = f; }

/** One block of text, at most `max` characters; never throws */
export function stateReport(max = 1500): string {
  const part = (name: string, f: () => string) => {
    try { const s = f(); return s ? `${name}: ${s}` : ""; } catch (err) { return `${name}: (${String(err).slice(0, 80)})`; }
  };
  return [
    `page open ${Math.round(performance.now() / 60000)} min`, // i18n-ignore
    part("clock", clockLine),
    part("demand", demandDiagnosis),
  ].filter(Boolean).join("\n").slice(0, max);
}
