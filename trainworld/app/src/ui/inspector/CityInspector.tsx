// A city picked on the map zoomed far out (T-186, SPEC 8): its trips within the zone and its longer
// trips to and from other places, each a figure with a mode bar, and the three biggest places
// (T-600, Anita: brief statistics of what the hover has, no table of every place). The map draws
// the city outlined and one bubble per other city (map/intercityView.ts). A click on the city in the
// Cities tab opens it too, the city view (T-1438), with CityDetails.tsx's rows around these.

import { cityById, type CityId } from "../../game/cities";
import { cityDemandViews, intercityDemand, INTERCITY_PLACES } from "../../game/demand";
import { placeFigures } from "../../game/corridorFigures";
import { count, exact } from "../../game/format";
import { zoneLoads } from "../../game/loading";
import { t, tSeries } from "../../i18n";
import { tPlace } from "../../i18n/places";
import { BUBBLE_ALPHA, bubbleColour, setBubbleColours, setBubbleShade } from "../../game/palette";
import { display, purposeFilter, mapShade, shownDayPhrase } from "../../game/ui";
import { MIN_PAIR_TRIPS } from "../../map/intercityView";
import { localModes } from "../../map/modeBar";
import { goTo } from "../Search";
import { Icon, SegBar } from "../widgets";
import { pctOf } from "./tripText";
import type { City } from "../../game/cities";
import { builtCities } from "../../game/worldTotals";
import { GoalTicks } from "../GoalTicks";
import { CityFigures, CityNetwork } from "./CityDetails";
import { HomeButton } from "./HomeButton";
import { TripGraph } from "./TripGraph";

/** `note`: a second line for the part's hover */
type Part = { name: string; n: number; cls: string; note?: string };
/** A figure with its mode bar: train blue, walking green, car red, No route grey, as the map's bubbles.
 * `kind`: whether the trips are within the city or to and from other places. */
function TripsBar({ label, tip, kind, city, parts, goals }: { label: string; tip?: string; kind: "within" | "inter"; city: string; parts: Part[]; goals?: City }) {
  const all = parts.reduce((s, p) => s + p.n, 0);
  const phrase = shownDayPhrase.value;
  const bar = <SegBar kind="split-bar" label={tSeries(parts.map((p) => `${p.name} ${pctOf(p.n, all)}`))}
    parts={parts.map((p) => ({ share: p.n / all, cls: p.cls, tip: t("cityInspector.partTip", { name: p.name, count: exact(p.n), kind, city, phrase, share: pctOf(p.n, all) }) + (p.note ? `\n${p.note}` : "") }))} />;
  return (
    <li class={"zone-trips" + (goals ? " with-goals" : "")}>
      <div class="line">
        <span class="grow" data-tip={tip}>{label}</span>
        <span class="val" data-tip={t("cityInspector.totalTip", { count: exact(all), kind, city, phrase })}>{count(all)}</span>
      </div>
      {/* T-1437: the goals over the city's own split (T-1551: only here, not on the top bar) */}
      {goals ? <div class="goal-bar">{bar}<GoalTicks city={goals} /></div> : bar}
    </li>
  );
}

/** T-3503: a place's everyday outings across its zone lines (`IntercityDemand.everyday`): legs, rail, walking. */
function everydayOf(e: number[] | undefined, place: number): [number, number, number] {
  for (let i = 0; e && i + 3 < e.length; i += 4) if (e[i] === place) return [e[i + 1], e[i + 2], e[i + 3]];
  return [0, 0, 0];
}

export function CityInspector({ id }: { id: string }) {
  const city = cityById(id);
  // T-421: a zone still loading (its bar on the map) has no figures to state yet
  const shown = zoneLoads.value.has(id as CityId) ? null : intercityDemand.value;
  if (!city) return null;
  // T-2136: the zone's rows come from a worker; until they do, its trips to other places wait as a loading zone's do
  const byPlace = shown ? placeFigures(shown.corridors, city.place) : null;
  const market = byPlace ? shown : null;
  const places = byPlace ?? [];
  // T-3503: with the everyday outings across the zone's lines (walking some of them)
  const ev = everydayOf(market?.everyday, city.place);
  const interAll = places.reduce((s, [, p]) => s + p.all, 0) + ev[0];
  const interRail = places.reduce((s, [, p]) => s + p.rail, 0) + ev[1];
  const interWalk = ev[2];
  const interAir = places.reduce((s, [, p]) => s + p.air, 0);
  // T-378: both ways, as the corridors count them
  const elsewhere = 2 * (market?.elsewhere?.[city.place] ?? 0);
  // within the zone: the open zone's modelled split, or only a count for a zone not opened
  const view = cityDemandViews.value.get(id as CityId);
  const lm = localModes(view);
  const localAll = view ? view.tripsPerDay : city.localTrips ?? 0;
  // the title mark is the zone's bubble colour (map/intercityView.ts): local and inter-region together
  const walk = (lm?.walk ?? 0) + interWalk, none = lm?.none ?? 0;
  const rail = interRail + (lm?.train ?? 0);
  const d = display.value;
  setBubbleShade(mapShade.value);
  setBubbleColours(d.bubbleColours, d.rampFrom, d.rampTo);
  const mix = bubbleColour(rail, walk, Math.max(0, interAll + localAll - rail - walk - none), none);
  const filter = purposeFilter.value;
  const drawn = filter === "all";
  const top = places.filter(([, p]) => p.all >= MIN_PAIR_TRIPS).slice(0, 3); // T-2770
  const built = builtCities().has(city.id);
  const shownName = tPlace(city.id, city.name);
  return (
    <>
      <div class="title">
        <i class="mk" style={{ background: `rgba(${mix[0]}, ${mix[1]}, ${mix[2]}, ${BUBBLE_ALPHA})` }} />
        <span class="name">{shownName}</span>
        <HomeButton place={city.id} />
        <button class="btn icon" data-tip={t("cityInspector.goTo", { city: shownName })} aria-label={t("cityInspector.goTo", { city: shownName })} onClick={() => goTo({ kind: "city", name: city.name, city })}>
          <Icon.locate />
        </button>
      </div>
      <CityFigures city={city} built={built} />
      {!market && !lm ? (
        <p class="muted">{t("cityInspector.calculating")}</p>
      ) : (
        <>
          <ul class="rows">
            {lm ? (
              <TripsBar
                label={t("cityInspector.tripsWithin", { city: shownName })}
                kind="within"
                city={shownName}
                parts={[
                  { name: t("cityInspector.train"), n: lm.train, cls: "seg-train" },
                  { name: t("cityInspector.walk"), n: lm.walk, cls: "seg-walk" },
                  { name: t("cityInspector.car"), n: lm.drive, cls: "seg-drive" },
                  ...(lm.none > 0 ? [{ name: t("cityInspector.noRoute"), n: lm.none, cls: "seg-none", note: t("inspector.noRoadEnds") }] : []),
                ]}
                goals={city}
              />
            ) : localAll > 0 ? (
              <li class="zone-trips">
                <div class="line">
                  <span class="grow">{t("cityInspector.tripsWithin", { city: shownName })}</span>
                  <span class="val" data-tip={t("cityInspector.totalTip", { count: exact(localAll), kind: "within", city: shownName, phrase: shownDayPhrase.value })}>{count(localAll)}</span>
                </div>
              </li>
            ) : null}
            {market && (
              <TripsBar
                label={t("cityInspector.toOther")}
                tip={t("cityInspector.toOtherTip")}
                kind="inter"
                city={shownName}
                parts={[
                  { name: t("cityInspector.train"), n: interRail, cls: "seg-train" },
                  ...(interWalk > 0 ? [{ name: t("cityInspector.walk"), n: interWalk, cls: "seg-walk" }] : []),
                  { name: t("cityInspector.car"), n: Math.max(0, interAll - interRail - interWalk - interAir), cls: "seg-drive" },
                  ...(interAir > 0 ? [{ name: t("cityInspector.plane"), n: interAir, cls: "seg-other" }] : []),
                ]}
              />
            )}
          </ul>
          {/* T-2500: a zone still loading whose own figures are in (a load's round of its cluster) shows them
              and its graph; its trips to other places wait */}
          {!market && <p class="muted">{t("cityInspector.calculating")}</p>}
          {lm && <TripGraph city={city} />}
          {market && top.length > 0 && (
            <ul class="rows">
              <li class="muted small">{t("cityInspector.topDestinations")}</li>
              {top.map(([place, p]) => (
                <li key={place}>
                  <span class="grow">{INTERCITY_PLACES[place]}</span>
                  <span class="val" data-tip={t("cityInspector.destinationTip", { count: exact(p.all), phrase: shownDayPhrase.value, city: shownName, place: INTERCITY_PLACES[place] })}>{count(p.all)}</span>
                </li>
              ))}
            </ul>
          )}
          {market && !places.length && elsewhere <= 0 && <p class="muted small">{t("cityInspector.noInter")}</p>}
          {!drawn && <p class="muted small">{t("cityInspector.notDrawn")}</p>}
        </>
      )}
      {built && <CityNetwork city={city} />}
    </>
  );
}
