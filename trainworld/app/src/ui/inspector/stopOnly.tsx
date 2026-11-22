// T-4013: a stop where riders only board or only get off, as the line panel and the schedule show it
import { ALIGHT_ONLY, BOARD_ONLY, type StopOnly } from "../../game/types";
import type { MessageKey } from "../../i18n";
import { Icon } from "../widgets";

/** the stop's mark's hover, by kind */
export const ONLY_TIP: Record<StopOnly, MessageKey> = { [BOARD_ONLY]: "lineInspector.boardOnly", [ALIGHT_ONLY]: "lineInspector.alightOnly" };
/** the line panel's switch's hover while stops are edited: what a click makes the stop, from its kind now (0 both) */
export const ONLY_NEXT: Record<0 | StopOnly, MessageKey> = { 0: "lineInspector.setBoardOnly", [BOARD_ONLY]: "lineInspector.setAlightOnly", [ALIGHT_ONLY]: "lineInspector.setBoardBoth" };

/** the line diagram's mark after a stop's name: an arrow toward the station dot on the left (boarding only) or away from it (getting off only) */
export function OnlyArrow({ kind }: { kind: StopOnly }) {
  return <svg viewBox="0 0 16 16" aria-hidden="true"><path d={kind === BOARD_ONLY ? "M13 8H3M6.5 4.5L3 8l3.5 3.5" : "M3 8h10M9.5 4.5L13 8l-3.5 3.5"} /></svg>;
}

export function OnlyIcon({ kind }: { kind: 0 | StopOnly }) {
  return kind === BOARD_ONLY ? <Icon.boardOnly /> : kind === ALIGHT_ONLY ? <Icon.alightOnly /> : <Icon.boardBoth />;
}
