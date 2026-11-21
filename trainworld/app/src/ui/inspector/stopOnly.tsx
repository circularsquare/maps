// T-4013: a stop where riders only board or only get off, as the line panel and the schedule show it
import { ALIGHT_ONLY, BOARD_ONLY, type StopOnly } from "../../game/types";
import type { MessageKey } from "../../i18n";
import { Icon } from "../widgets";

/** the stop's mark's hover, by kind */
export const ONLY_TIP: Record<StopOnly, MessageKey> = { [BOARD_ONLY]: "lineInspector.boardOnly", [ALIGHT_ONLY]: "lineInspector.alightOnly" };
/** the line panel's switch's hover while stops are edited: what a click makes the stop, from its kind now (0 both) */
export const ONLY_NEXT: Record<0 | StopOnly, MessageKey> = { 0: "lineInspector.setBoardOnly", [BOARD_ONLY]: "lineInspector.setAlightOnly", [ALIGHT_ONLY]: "lineInspector.setBoardBoth" };

export function OnlyIcon({ kind }: { kind: 0 | StopOnly }) {
  return kind === BOARD_ONLY ? <Icon.boardOnly /> : kind === ALIGHT_ONLY ? <Icon.alightOnly /> : <Icon.boardBoth />;
}
