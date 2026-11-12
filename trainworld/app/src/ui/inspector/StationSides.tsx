// T-3332: the station view's selector for a station on double track: platform on the left track, on both,
// or on the right track. Each icon draws the two tracks with a station bracket on the side it picks; left
// and right are as the map shows the station with its axis turned to point up the screen (north-up map).
import { t } from "../../i18n";
import type { StationInput } from "../../game/types";
import { money } from "../../game/format";
import { edit, rightHand, say, world } from "../../game/world";
import { useClockQuery } from "../useClockQuery";

const TAU = 2 * Math.PI;

/** The heading (rad, counter-clockwise from east) the station's axis is read in: the way along the track that
 * points up the screen, east for a track running exactly east-west. */
function upAxis(heading: number): number {
  const s = Math.sin(heading), c = Math.cos(heading);
  return s > 1e-6 || (Math.abs(s) <= 1e-6 && c > 0) ? heading : heading + Math.PI;
}

/** The compass word of a heading, the nearest of the four. */
export function compass(h: number): "east" | "north" | "west" | "south" {
  const a = ((h % TAU) + TAU) % TAU;
  return a < Math.PI / 4 || a >= 7 * Math.PI / 4 ? "east" : a < 3 * Math.PI / 4 ? "north" : a < 5 * Math.PI / 4 ? "west" : "south";
}

/** A track pair with a bracket beside the track(s) that have a platform. */
function Sides({ left, right }: { left: boolean; right: boolean }) {
  return (
    <svg class="tracks" viewBox="0 0 20 20">
      <path d="M7 3v14M13 3v14" />
      {left && <path d="M4.5 6.5h-1.5v7h1.5" />}
      {right && <path d="M15.5 6.5h1.5v7h-1.5" />}
    </svg>
  );
}

export function StationSides({ s }: { s: StationInput }) {
  const cost = useClockQuery({ what: "tracksCost", node: s.num }, [s.num, s.level, s.built, s.one, s.length], (d) => d[0] ?? 0);
  const rh = rightHand.value;
  const up = upAxis(s.heading);
  // the heading of the trains that stop at a platform on the left (right) track of the axis `up`: trains keep
  // right, so those on the left track run against it
  const against = rh ? up + Math.PI : up;
  const towardsLeft = against, towardsRight = against + Math.PI;
  const way = (h: number) => t("stationInspector.oneWay", { way: compass(h) });
  const now = s.one === 1 && s.oneHeading !== undefined ? s.oneHeading : null;
  const serves = (h: number) => now !== null && Math.cos(now - h) > 0;
  const onLeft = serves(towardsLeft), onRight = serves(towardsRight), both = !s.one;
  const free = t("stationInspector.switchFree");
  const costLine = cost && cost > 0 ? t("stationUpgrades.costs", { price: money(cost * 1e6) }) : free;
  const pickOne = async (h: number) => {
    if (serves(h)) return;
    await edit({ op: "stationWay", node: s.num, heading: h });
  };
  const pickBoth = async () => {
    if (both) return;
    const a = await edit({ op: "stationTracks", node: s.num, one: false });
    if (a.ok && a.charge > 0) say(t("stationInspector.twoWayBuilt", { station: s.name, price: money(a.charge * 1e6) }));
  };
  void world.value;
  return (
    <div class="group tb-count" role="group" aria-label={t("toolBar.platformTracks")}>
      <button class={"btn icon" + (onLeft ? " on" : "")} aria-label={way(towardsLeft)} aria-pressed={onLeft} data-tip={`${way(towardsLeft)}\n${free}`}
        onClick={() => void pickOne(towardsLeft)}><Sides left right={false} /></button>
      <button class={"btn icon" + (both ? " on" : "")} aria-label={t("toolBar.bothTracksStation")} aria-pressed={both}
        data-tip={`${t("toolBar.bothTracksStation")}${both ? "" : `\n${costLine}`}`}
        onClick={() => void pickBoth()}><Sides left right /></button>
      <button class={"btn icon" + (onRight ? " on" : "")} aria-label={way(towardsRight)} aria-pressed={onRight} data-tip={`${way(towardsRight)}\n${free}`}
        onClick={() => void pickOne(towardsRight)}><Sides left={false} right /></button>
    </div>
  );
}
