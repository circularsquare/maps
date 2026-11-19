// T-3740 stage Z2: the Fare zones tool's bar on the map (the fare area and the zone it gives) and the station
// panel's Fare zone row, shown only while the fares have an area. docs/ui-panels.md "Advanced fares".

import { MAX_ZONES, zoneCode, zoneOfCode, areaOfCode } from "../game/fareZones";
import { complexCode, currentArea, fareAreas, paintArea, paintZone, RING_STEPS, ringsArmed, ringWidth, ringWidthShown, setStationCodes } from "../game/fareZonePaint";
import { unitShort } from "../game/format";
import { CLICK_MODE_COLOURS, fareZoneColour, inkOn } from "../game/palette";
import type { StationInput } from "../game/types";
import { t, tNumber } from "../i18n";
import { Icon, Stepper } from "./widgets";

/** A zone number on a square of its colour; "none" for 0. */
export function ZoneChip({ zone }: { zone: number }) {
  if (!zone) return <>{t("fareZoneParts.none")}</>;
  const c = fareZoneColour(zone);
  return <span class="fz-chip" style={{ background: c, color: inkOn(c) }}>{zone}</span>;
}

/** The tool's bar: the area (a picker once there are two), and the zone a click or loop gives (none takes
 * stations out of their fare zone). Neither is an edit. */
export function FareBar() {
  const areas = fareAreas(), a = currentArea(areas), z = paintZone.value, armed = ringsArmed.value, width = ringWidthShown();
  if (!a) return null;
  return (
    <div class="tool-bar fare-bar" role="group" aria-label={t("fareZoneParts.title")} style={{ "--mode": CLICK_MODE_COLOURS.fares }}>
      <div class="tb-row">
        {areas.length > 1
          ? <select aria-label={t("fareZoneParts.area")} value={a.id} onChange={(e) => (paintArea.value = Number(e.currentTarget.value))}>
              {areas.map((x) => <option value={x.id}>{x.name}</option>)}
            </select>
          : <span class="fz-area">{a.name}</span>}
        <span class="muted">{t("fareZoneParts.zone")}</span>
        <Stepper value={<ZoneChip zone={z} />} label={t("fareZoneParts.zoneLabel")} onStep={(d) => {
          const to = Math.max(0, Math.min(MAX_ZONES, z + d));
          if (to === z) return false;
          paintZone.value = to;
          return undefined;
        }} />
      </div>
      <div class="tb-row">
        <button class={"btn text mini" + (armed ? " on" : "")} aria-pressed={armed} onClick={() => (ringsArmed.value = !armed)}>
          <Icon.rings />{t("fareZoneParts.rings")}
        </button>
        {armed && <>
          <span class="muted">{t("fareZoneParts.ringWidth")}</span>
          <Stepper value={t("fareGraph.tickUnit", { value: tNumber(width), unit: unitShort() })} label={t("fareZoneParts.ringWidthLabel")} onStep={(d) => {
            const i = RING_STEPS.indexOf(width), to = RING_STEPS[Math.max(0, Math.min(RING_STEPS.length - 1, (i < 0 ? RING_STEPS.findIndex((s) => s >= width) : i) + d))];
            if (to === undefined || to === width) return false;
            ringWidth.value = to;
            return undefined;
          }} />
        </>}
      </div>
    </div>
  );
}

/** The station panel's row: the fare zone of `members` (a station, or its complex's stations), stepped for all of
 * them at once, in their own area (or the tool's when they have none). Each step is one edit. */
export function FareZoneRow({ members }: { members: StationInput[] }) {
  const areas = fareAreas();
  if (!areas.length || !members.length) return null;
  const code = complexCode(members, areas);
  const area = code ? areas.find((x) => x.id === areaOfCode(code))! : currentArea(areas);
  const z = code ? zoneOfCode(code) : 0;
  return (
    <li>
      <span class="grow" data-tip={areas.length > 1 && area ? area.name : undefined}>{t("fareZoneParts.zone")}</span>
      <Stepper value={<ZoneChip zone={z} />} wide label={t("fareZoneParts.zoneLabel")} onStep={(d) => {
        const to = Math.max(0, Math.min(MAX_ZONES, z + d));
        if (to === z || !area) return false;
        return setStationCodes(members.map((s) => s.num), to ? zoneCode(area.id, to) : 0);
      }} />
    </li>
  );
}
