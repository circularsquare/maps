import { t } from "../i18n";
import { FEW_MS, type Stage } from "./loadBarCore";

export * from "./loadBarCore";

/** what the card says while each stage is the one in hand (T-1194) */
export const STAGE_TEXT: Record<Stage, string> = { get game() { return t("loadBar.game"); }, get data() { return t("loadBar.data"); }, get demand() { return t("loadBar.demand"); } };

/** T-1479: the time left as the card says it, rough on purpose (T-1654: always a figure, never a count
 * that gives up) */
export function leftText(ms: number): string {
  const s = ms / 1000;
  if (ms < FEW_MS) return t("loadBar.fewSeconds");
  if (s < 55) return t("loadBar.seconds", { n: Math.round(s / 10) * 10 });
  if (s < 90) return t("loadBar.oneMinute");
  return t("loadBar.minutes", { n: Math.round(s / 60) });
}
