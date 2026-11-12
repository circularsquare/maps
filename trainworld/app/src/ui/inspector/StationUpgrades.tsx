// The station panel's row of upgrades (T-1281): each icon a toggle, outlined when not built, dashed
// while in the blueprint, filled when built. On a built station a click pays at once, as the
// junction inspector's flyover does. The passage (the player's link, T-197) starts a pick of the
// other station instead; its passages are listed below the row.

import { millionsRange, money, perArea, roughDistance } from "../../game/format";
import { LINK_MAX_M } from "../../game/complexes";
import { BUILDING_PER_M2, offeredRectangle } from "../../game/buildings";
import { buildingDraft, buildingMembersById, buildingOf } from "../../game/buildingView";
import { complexOf } from "../../game/complexView";
import { groundScale } from "../../game/coords";
import { tool } from "../../game/ui";
import { t } from "../../i18n";
import { live } from "../../map/live";
import { cancelLink, linking, linksAmong, startLink } from "../../game/links";
import type { StationInput } from "../../game/types";
import { hasUpgrade, PASSAGE_BASE, PASSAGE_PER_KM, UPGRADE, UPGRADE_BITS, UPGRADE_TEXT, type UpgradeText } from "../../game/upgrades";
import { edit, say, stationsById, world } from "../../game/world";
import { useClockQuery } from "../useClockQuery";

const ICONS: Record<number, () => preact.JSX.Element> = {
  // two sidings running on past a platform's end
  [UPGRADE.terminal]: () => <svg viewBox="0 0 16 16"><path d="M1.5 6h6M1.5 10h6M7.5 6l3 1.5h4M7.5 10l3-1.5h4" /><path class="solid" d="M1.5 7h5v2h-5z" /></svg>,
  [UPGRADE.parkRide]: () => <svg viewBox="0 0 16 16"><path d="M5.5 13V3h3.5a3 3 0 010 6H5.5" /></svg>,
  // a bench under a roof
  [UPGRADE.waitingRoom]: () => <svg viewBox="0 0 16 16"><path d="M2 6l6-3.5L14 6M3.5 10.5h9M4.5 10.5v3M11.5 10.5v3M4.5 8h7" /></svg>,
  // a through track, and a track each side bowing out round the platforms and back
  [UPGRADE.passing]: () => <svg viewBox="0 0 16 16"><path d="M1 8h14M1 8c2.5 0 2.5-4 5-4h4c2.5 0 2.5 4 5 4M1 8c2.5 0 2.5 4 5 4h4c2.5 0 2.5-4 5-4" /></svg>,
};
// two platforms and the walkway between them
const PassageIcon = () => <svg viewBox="0 0 16 16"><path class="solid" d="M1.5 3.5h3v9h-3zM11.5 3.5h3v9h-3z" /><path d="M4.5 8h7" /></svg>;
/** Whether any of `ids` (`every`: each of them) is a station building's member. */
const housedText = () => t("stationUpgrades.housed");
function inBuilding(ids: readonly string[], every = false): boolean {
  const housed = new Set([...buildingMembersById.value.values()].flat());
  return every ? ids.every((id) => housed.has(id)) : ids.some((id) => housed.has(id));
}
// a hall round a platform
const BuildingIcon =() => <svg viewBox="0 0 16 16"><path d="M1.5 3.5h13v9h-13z" /><path class="solid" d="M4 7h8v2H4z" /></svg>;

/** `members`: the complex's stations in complex view, where park and ride and the waiting room are
 * built for all of them at once as one shared facility (T-1528), and the passage. */
export function StationUpgrades({ s, members }: { s: StationInput; members?: string[] }) {
  const prices = useClockQuery({ what: "upgradeCost", node: s.num }, [s.num, s.level, s.built], (d) => Array.from(d));
  if (members) return (
    <div class="upgrades" role="group" aria-label={t("stationUpgrades.complexGroup")}>
      {UPGRADE_TEXT.filter((u) => u.doesAll).map((u) => <ComplexUpgrade u={u} members={members} price={prices?.[UPGRADE_BITS.indexOf(u.bit)] ?? 0} />)}
      <PassageButton s={s} />
      <BuildingButton s={s} />
    </div>
  );
  // T-2231: passing tracks work on double track only
  const double = !!world.value?.edges.some((e) => (e.a === s.num || e.b === s.num) && e.tracks === 2);
  // T-2308: a station building is its members' waiting room (T-2276), so one bought there adds nothing
  const housed = inBuilding([s.id]);
  return (
    <div class="upgrades" role="group" aria-label={t("stationUpgrades.group")}>
      {/* T-3332: a one-track station has no terminal and no passing tracks */}
      {UPGRADE_TEXT.filter((u) => (!s.tram || u.bit === UPGRADE.waitingRoom) && (u.bit !== UPGRADE.passing || double || hasUpgrade(s.upgrades, u.bit))
        && !(s.one && (u.bit === UPGRADE.terminal || u.bit === UPGRADE.passing) && !hasUpgrade(s.upgrades, u.bit))).map((u) => (
        <UpgradeButton s={s} u={u} price={prices?.[UPGRADE_BITS.indexOf(u.bit)] ?? 0} housed={housed} />
      ))}
      <PassageButton s={s} />
      {!s.tram && <BuildingButton s={s} />}
    </div>
  );
}

/** One station's own upgrade as a toggle: build or add it, or take it away. */
function UpgradeButton({ s, u, price, housed = false }: { s: StationInput; u: UpgradeText; price: number; housed?: boolean }) {
  const on = hasUpgrade(s.upgrades, u.bit);
  const state = !on ? "" : s.built ? " on" : " planned";
  const act = on
    ? t(s.built ? "stationUpgrades.actBuilt" : "stationUpgrades.actPlanned")
    : t(s.built ? "stationUpgrades.actBuild" : "stationUpgrades.actAdd");
  const Ic = ICONS[u.bit];
  const toggle = async () => {
    const a = await edit({ op: "stationUpgrade", nodes: [s.num], bit: u.bit, on: !on });
    if (!a.ok) return;
    if (!on) say(a.charge > 0 ? t("stationUpgrades.built", { name: u.name, station: s.name, price: money(a.charge * 1e6) }) : t("stationUpgrades.added", { name: u.name }));
    else if (s.built) say(t("stationUpgrades.removed", { name: u.name, station: s.name }));
  };
  if (u.bit === UPGRADE.waitingRoom && housed && !on) return (
    <button class="btn icon upgrade" aria-label={u.name} disabled data-tip={`${u.name}\n${u.does(s.tram)}\n${housedText()}`}>
      <Ic />
    </button>
  );
  return (
    <button class={"btn icon upgrade" + state} aria-label={u.name} aria-pressed={on}
      data-tip={`${u.name}\n${u.does(s.tram)}\n${t("stationUpgrades.costs", { price: money(price * 1e6) })}\n${act}`}
      onClick={() => void toggle()}>
      <Ic />
    </button>
  );
}

/** T-3154: a station's terminal button in a complex's list, the one the station view has (a tram stop
 * or a port has none). */
export function TerminalButton({ s }: { s: StationInput }) {
  const u = UPGRADE_TEXT.find((x) => x.bit === UPGRADE.terminal)!;
  const prices = useClockQuery({ what: "upgradeCost", node: s.num }, [s.num, s.level, s.built], (d) => Array.from(d));
  if (s.tram || s.port || (s.one && !hasUpgrade(s.upgrades, UPGRADE.terminal))) return null;
  return <UpgradeButton s={s} u={u} price={prices?.[UPGRADE_BITS.indexOf(u.bit)] ?? 0} />;
}

/** T-2218: the station building tool, started on the complex's building, or on the smallest rectangle
 * round its boxes when it has none. */
function BuildingButton({ s }: { s: StationInput }) {
  const members = complexOf(s.id);
  const b = buildingOf(members);
  const d = buildingDraft.value;
  const drawing = tool.value === "building" && !!d && (b ? d.id === b.id : d.id === null);
  const state = drawing || b ? " on" : "";
  const tip = t("stationUpgrades.buildingTip", { price: perArea(BUILDING_PER_M2 * 1e6), act: drawing ? "stop" : b ? "built" : "new" });
  const click = () => {
    if (drawing) return live.tools?.exit();
    const w = world.peek();
    if (!w || !live.tools) return;
    if (b) return live.tools.building.begin(b.corners, b.id, s.id);
    const byId = stationsById(w);
    const rect = offeredRectangle(members.map((id) => byId.get(id)).filter((x): x is StationInput => !!x), groundScale);
    live.tools.building.begin(rect, null, s.id);
  };
  return (
    <button class={"btn icon upgrade" + state} aria-label={t("stationUpgrades.building")} aria-pressed={drawing} data-tip={tip} onClick={click}>
      <BuildingIcon />
    </button>
  );
}

/** Park and ride or the waiting room for a whole complex (T-1528): on at every station or none, one
 * station's price at each station that has none yet (the same everywhere); a complex where only some
 * have it shows as not built, and a click builds it at the rest. */
function ComplexUpgrade({ u, members, price }: { u: UpgradeText; members: string[]; price: number }) {
  const w = world.value;
  const byId = w ? stationsById(w) : new Map<string, StationInput>();
  const st = members.map((id) => byId.get(id)).filter((x): x is StationInput => !!x);
  const have = st.filter((x) => hasUpgrade(x.upgrades, u.bit)), missing = st.filter((x) => !hasUpgrade(x.upgrades, u.bit));
  const all = missing.length === 0, built = st.some((x) => x.built), name = st[0]?.name ?? "", allTram = st.length > 0 && st.every((x) => x.tram);
  const state = !all ? "" : have.some((x) => x.built) ? " on" : " planned";
  const Ic = ICONS[u.bit];
  if (u.bit === UPGRADE.waitingRoom && !all && st.length && inBuilding(st.map((x) => x.id), true)) return (
    <button class="btn icon upgrade" aria-label={u.name} disabled data-tip={`${u.name}\n${u.doesAll!(allTram)}\n${housedText()}`}>
      <Ic />
    </button>
  );
  const cost = (all ? st.length : missing.length) * price;
  const act = all
    ? built ? t("stationUpgrades.allBuilt", { count: st.length }) : t("stationUpgrades.actPlanned")
    : t("stationUpgrades.someAct", { some: have.length > 0, have: have.length, count: st.length, missing: missing.length, blueprint: !built });
  const click = async () => {
    const a = await edit({ op: "stationUpgrade", nodes: (all ? st : missing).map((x) => x.num), bit: u.bit, on: !all });
    if (!a.ok) return;
    if (!all) say(a.charge > 0 ? t("stationUpgrades.built", { name: u.name, station: name, price: money(a.charge * 1e6) }) : t("stationUpgrades.added", { name: u.name }));
    else if (built) say(t("stationUpgrades.removed", { name: u.name, station: name }));
  };
  return (
    <button class={"btn icon upgrade" + state} aria-label={u.name} aria-pressed={all}
      data-tip={`${u.name}\n${u.doesAll!(allTram)}\n${t("stationUpgrades.costs", { price: money(cost * 1e6) })}\n${act}`} onClick={() => void click()}>
      <Ic />
    </button>
  );
}

function PassageButton({ s }: { s: StationInput }) {
  const picking = linking.value === s.id;
  const w = world.value;
  const byId = w ? stationsById(w) : new Map<string, StationInput>();
  const pairs = linksAmong([s.id]);
  const built = pairs.some(([a, b]) => byId.get(a)?.built && byId.get(b)?.built);
  const state = picking ? " on" : built ? " on" : pairs.length ? " planned" : "";
  const tip = t("stationUpgrades.passageTip", { distance: roughDistance(LINK_MAX_M), range: millionsRange(PASSAGE_BASE, Math.round(PASSAGE_BASE + PASSAGE_PER_KM * LINK_MAX_M / 1000)), picking });
  return (
    <button class={"btn icon upgrade" + state} aria-label={t("stationUpgrades.passage")} aria-pressed={picking} data-tip={tip}
      onClick={() => (picking ? cancelLink() : startLink(s.id))}>
      <PassageIcon />
    </button>
  );
}
