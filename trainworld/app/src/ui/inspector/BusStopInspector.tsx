// T-2949: a stop that several bus lines share, shown together as a station complex is: its name, then
// the lines that stop there. A click on a line selects it.

import { t } from "../../i18n";
import { selection } from "../../game/ui";
import { linesById, stationNames, world } from "../../game/world";
import { Chip } from "../widgets";

export function BusStopInspector({ lines, stop }: { lines: string[]; stop: string }) {
  const w = world.value;
  if (!w) return null;
  const byId = linesById(w);
  const here = lines.flatMap((id) => byId.get(id) ?? []);
  return (
    <>
      <div class="title">
        <span class="name">{stationNames(w).get(stop) ?? t("world.busStop")}</span>
      </div>
      <div class="h">{t("stationInspector.lines")}</div>
      <ul class="rows pick">
        {here.map((l) => (
          <li onClick={() => (selection.value = { kind: "line", line: l.id })}>
            <Chip line={l} />
            <span class="grow">{l.name}</span>
          </li>
        ))}
      </ul>
    </>
  );
}
