// A station complex from above (T-147), drawn by the map's own network renderer (T-1289, Anita: "could
// we just copy the view in the world"): platforms, lines, trains, crowds and riders walking between
// platforms exactly as the map draws them close in, north up, whatever the map is looking at. The
// renderer draws it with its own camera along with each map frame (map/overlay.ts `setSnapshot`);
// this component picks the camera and lays the hovers, name labels and scale bar over it. Shown in
// the station inspector's complex view (StationInspector.tsx `ComplexSummary`).
//
// Hovers: each platform's names its lines with their boardings and riders, the riders waiting there
// now, and the biggest changes to other lines (T-176, `complexTransfers`). Platforms on top of each
// other (parallel within 15 degrees, centre lines within 12 m; the real New York save gives every
// service its own station) share one hover listing them all. Stations the player linked (T-197) are
// the renderer's bridges; their hover gives the distance.

import { useEffect, useLayoutEffect, useRef, useState } from "preact/hooks";
import { useLatestAnswer } from "../useClockQuery";
import { complexTransfers, demandView, lineDemand, type LineDemand } from "../../game/demand";
import { count, length, scaleBars } from "../../game/format";
import { segmentGap } from "../../game/complexes";
import type { LineInput, StationInput } from "../../game/types";
import { world } from "../../game/world";
import type { ComplexNow } from "../../game/complexLive";
import { live } from "../../map/live";
import { anchor, EARTH_C } from "../../map/geo";
import { stationIndexOf } from "../../map/network";
import { STATION_BOX_ZOOM } from "../../map/stationGeometry";

/** Riders on a line's trains at its `k`th stop a day, both directions: in each direction the
 * busier of the arriving and leaving train loads, every period summed. */
function ridersAt(dm: LineDemand, n: number, k: number): number {
  let sum = 0;
  for (const seg of dm.loads) {
    if (seg.length !== 2 * (n - 1)) continue;
    // run 0 goes stop 0 to n - 1 (segment k is stop k to k + 1); run 1 comes back, its segment j
    // is stop n - 1 - j to n - 2 - j, stored after run 0's
    const in0 = k > 0 ? seg[k - 1] : 0, out0 = k < n - 1 ? seg[k] : 0;
    const in1 = k < n - 1 ? seg[n - 1 + (n - 2 - k)] : 0, out1 = k > 0 ? seg[n - 1 + (n - 1 - k)] : 0;
    sum += Math.max(in0, out0) + Math.max(in1, out1);
  }
  return sum;
}

const PAD = 24; // px kept clear round the platforms
/** the closest the view comes: a small complex is not blown up past the map's own close-up */
const MAX_ZOOM = 17.5;
const BUNDLE_M = 12; // centre lines this close, and parallel, share a hover
const PARALLEL = Math.sin((15 * Math.PI) / 180);
const MIN_HIT = 10; // px, a platform's hover at the least across

/** One line at one platform as transfers name it. */
const strokeKey = (station: string, line: string) => `${station}|${line}`;

let measureCtx: CanvasRenderingContext2D | null | undefined;
function textWidth(t: string): number {
  if (measureCtx === undefined) {
    measureCtx = document.createElement("canvas").getContext("2d");
  }
  if (measureCtx) measureCtx.font = '12px "Nunito", "Zen Maru Gothic", sans-serif';
  return measureCtx ? measureCtx.measureText(t).width : t.length * 6.6;
}

/** The name as the mini-map labels it: the part before a slash ("Times Sq-42 St" of "Times
 * Sq-42 St/42 St-Port Authority Bus Terminal"), the whole name on hover and in the list. */
const shortName = (n: string) => n.split("/")[0].trim() || n;

/** Stations whose platforms lie on top of each other, as groups of ids. */
function overlaps(st: StationInput[]): Map<string, StationInput[]> {
  const n = st.length, parent = st.map((_, i) => i);
  const find = (i: number): number => (parent[i] === i ? i : (parent[i] = find(parent[i])));
  const ends = st.map((s) => {
    const h = s.length / 2, dx = Math.cos(s.heading) * h, dy = Math.sin(s.heading) * h;
    return [s.x - dx, s.y - dy, s.x + dx, s.y + dy];
  });
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      if (Math.abs(Math.sin(st[i].heading - st[j].heading)) > PARALLEL) continue;
      const [a, b] = [ends[i], ends[j]];
      if (segmentGap(a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]) <= BUNDLE_M) parent[find(j)] = find(i);
    }
  const groups = new Map<number, StationInput[]>();
  st.forEach((s, i) => groups.set(find(i), [...(groups.get(find(i)) ?? []), s]));
  const out = new Map<string, StationInput[]>();
  for (const g of groups.values()) for (const s of g) out.set(s.id, g);
  return out;
}

/** Each member's platform on the map: its two ends in Mercator 0..1 and half its width, from the
 * renderer's station buffers; absent for a station the drawn network does not have yet. */
function platforms(members: string[]): Map<string, { a: [number, number]; b: [number, number]; half: number }> {
  const out = new Map<string, { a: [number, number]; b: [number, number]; half: number }>();
  const n = live.overlay?.renderer.net;
  if (!n) return out;
  const o = (n.anchor ?? anchor()).merc, S = n.stations, sh = n.stationShape, ax = n.stationAxis;
  for (const m of members) {
    const i = stationIndexOf(n, m);
    if (i < 0) continue;
    const cx = S[i * 3] + (sh?.[i * 4] ?? 0), cy = S[i * 3 + 1] + (sh?.[i * 4 + 1] ?? 0);
    const dx = ax?.[i * 2] ?? 0, dy = ax?.[i * 2 + 1] ?? 0;
    const at = (x: number, y: number): [number, number] => [o[0] + x / EARTH_C, o[1] + y / EARTH_C];
    out.set(m, { a: at(cx - dx, cy - dy), b: at(cx + dx, cy + dy), half: (sh?.[i * 4 + 3] ?? 0) / EARTH_C });
  }
  return out;
}

export function ComplexMap({ id, members, stations, lines, now, onOpen }: {
  id: string; members: string[]; stations: Map<string, StationInput>; lines: LineInput[];
  /** riders waiting at the platforms now (T-246, game/complexLive.ts) */
  now?: ComplexNow | null;
  /** a click on a platform: open that station */
  onOpen: (station: string) => void;
}) {
  const box = useRef<HTMLDivElement>(null);
  const cv = useRef<HTMLCanvasElement>(null);
  const [W, setW] = useState(300);
  useLayoutEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(Math.max(160, Math.round(el.clientWidth))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // the complex's transfers, asked again for each published round; the last answer stays drawn
  // while a newer one is on its way (a worker busy with the next round answers nothing)
  const dv = demandView.value;
  const scope = members.join(",");
  const solved = dv ? `${dv.version}:${dv.round}` : "";
  const xf = useLatestAnswer(solved ? () => complexTransfers(members) : null, [scope, solved]);
  const transfers = xf && dv && xf.scope === scope && xf.version === dv.version ? xf : null;

  const st = members.map((m) => stations.get(m)).filter((s): s is StationInput => !!s);
  const plat = platforms(members);

  // the view: every platform's ends in, north up, at most MAX_ZOOM and never further out than the
  // map's platform boxes
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const p of plat.values())
    for (const [x, y] of [p.a, p.b]) {
      x0 = Math.min(x0, x); x1 = Math.max(x1, x);
      y0 = Math.min(y0, y); y1 = Math.max(y1, y);
    }
  const dx = Math.max(1e-9, x1 - x0), dy = Math.max(1e-9, y1 - y0);
  const H = Math.round(Math.max(120, Math.min(240, ((W - 2 * PAD) * dy) / dx + 2 * PAD)));
  const zoom = Math.max(STATION_BOX_ZOOM, Math.min(MAX_ZOOM, Math.log2(Math.min((W - 2 * PAD) / dx, (H - 2 * PAD) / dy) / 512)));
  const S = 512 * 2 ** zoom, mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
  const ok = plat.size > 0 && Number.isFinite(mx);
  const px = (p: [number, number]): [number, number] => [W / 2 + (p[0] - mx) * S, H / 2 + (p[1] - my) * S];

  useEffect(() => {
    const o = live.overlay, c = cv.current;
    if (!o || !c || !ok) return;
    o.setSnapshot({ canvas: c, w: W, h: H, mx, my, zoom });
    return () => o.setSnapshot(null);
  }, [ok, W, H, mx, my, zoom]);

  if (st.length < 2) return null;

  // what stops at each platform
  const at = new Map<string, { line: LineInput; boardings: number; riders: number; known: boolean }[]>(st.map((s) => [s.id, []]));
  for (const l of lines) {
    const dm = lineDemand(l.id);
    l.stops.forEach((sid, k) => at.get(sid)?.push({ line: l, boardings: dm?.boardings[k] ?? 0, riders: dm ? ridersAt(dm, l.stops.length, k) : 0, known: !!dm }));
  }

  // each line's biggest changes at each platform, both directions, inside the complex only
  const partners = new Map<string, Map<string, number>>();
  const lineById = new Map(lines.map((l) => [l.id, l]));
  for (const p of transfers?.pairs ?? []) {
    const a = stations.get(p.fromStation), b = stations.get(p.toStation), la = lineById.get(p.fromLine), lb = lineById.get(p.toLine);
    if (!a || !b || !la || !lb || !at.has(a.id) || !at.has(b.id)) continue;
    if (a.id === b.id && la.id === lb.id) continue; // back the way they came on the same line
    for (const [s, l, os, ol] of [[a, la, b, lb], [b, lb, a, la]] as const) {
      const k = strokeKey(s.id, l.id), m = partners.get(k) ?? new Map<string, number>();
      const name = ol.name + (os.id === s.id ? "" : ` at ${shortName(os.name)}`);
      m.set(name, (m.get(name) ?? 0) + p.riders);
      partners.set(k, m);
    }
  }

  // one platform's part of a hover: its lines, the riders waiting now (T-246) and the changes
  const platTip = (s: StationInput) => {
    let text = s.name + (s.built ? "" : " (blueprint)");
    const rows = at.get(s.id) ?? [];
    if (!rows.length) return `${text}\nNo line stops here`;
    for (const r of rows) {
      text += `\n${r.line.name}` + (r.known ? `: ${count(r.boardings)} boardings, ${count(r.riders)} riders on its trains here per day` : "");
      const m = partners.get(strokeKey(s.id, r.line.id));
      if (m) {
        const top = [...m].sort((x, y) => y[1] - x[1]);
        text += `, ${count(top.reduce((a, [, v]) => a + v, 0))} change to or from other lines, most with ` +
          top.slice(0, 3).map(([n, v]) => `${n} (${count(v)})`).join(", ");
      }
    }
    const p = now?.platforms.get(s.id);
    if (now && p) {
      text += `\n${count(p.side[0] + p.side[1])} waiting on the platform now`;
      text += p.oneSide ? `, room for ${count(p.room)}` : `, room for ${count(p.room)} on each side`;
    }
    return text;
  };
  const together = overlaps(st);
  const tip = (s: StationInput) => (together.get(s.id) ?? [s]).map(platTip).join("\n\n");

  // Name labels, when the complex has more than one name: one per name (T-169), busiest first,
  // beside its platforms, clear of every platform, other label and the scale bar.
  const lat = Math.atan(Math.sinh(Math.PI * (1 - 2 * my)));
  const k = S / EARTH_C / Math.cos(lat); // px per metre on the ground
  const bars = scaleBars();
  const [bar, barLabel] = bars.find(([m]) => m * k >= 40) ?? bars[bars.length - 1];
  const barRect: [number, number, number, number] = [W - 14 - bar * k, H - 24, W, H];
  const taken: [number, number, number, number][] = [barRect];
  const samples: [number, number, number][] = []; // points along each platform: x, y, radius
  for (const p of plat.values()) {
    const [ax, ay] = px(p.a), [bx, by] = px(p.b), len = Math.hypot(bx - ax, by - ay), r = Math.max(MIN_HIT / 2, p.half * S) + 2;
    for (let d = 0; d <= len; d += 3) samples.push([ax + ((bx - ax) * d) / Math.max(1, len), ay + ((by - ay) * d) / Math.max(1, len), r]);
  }
  const clear = (r: [number, number, number, number]) =>
    r[0] >= 2 && r[2] <= W - 2 && r[1] >= 2 && r[3] <= H - 2 &&
    !taken.some((q) => r[0] < q[2] && q[0] < r[2] && r[1] < q[3] && q[1] < r[3]) &&
    !samples.some(([x, y, rad]) => x > r[0] - rad && x < r[2] + rad && y > r[1] - rad && y < r[3] + rad);
  const byName = new Map<string, StationInput[]>();
  for (const s of st) byName.set(s.name, [...(byName.get(s.name) ?? []), s]);
  const labels: { text: string; x: number; y: number; anchor: string }[] = [];
  if (ok && byName.size > 1) {
    const weight = (ss: StationInput[]) => ss.reduce((a, s) => a + (at.get(s.id)?.length ?? 0), 0);
    for (const [name, all] of [...byName].sort((a, b) => weight(b[1]) - weight(a[1]))) {
      // beside the platform group holding most of the name's lines: a name can be on two platforms
      // apart (Penn's 34 St-Penn Station is the 1/2/3 and the A/C/E), and their midpoint is
      // somewhere else
      const lead = all.reduce((a, b) => ((at.get(b.id)?.length ?? 0) > (at.get(a.id)?.length ?? 0) ? b : a));
      const ss = (together.get(lead.id) ?? [lead]).filter((s) => s.name === name);
      let l = Infinity, r = -Infinity, t = Infinity, b = -Infinity;
      for (const s of ss) {
        const p = plat.get(s.id);
        if (!p) continue;
        for (const q of [px(p.a), px(p.b)]) {
          l = Math.min(l, q[0]); r = Math.max(r, q[0]); t = Math.min(t, q[1]); b = Math.max(b, q[1]);
        }
      }
      if (!Number.isFinite(l)) continue;
      const text = shortName(name), w = textWidth(text), cx = (l + r) / 2, cy = (t + b) / 2;
      // baseline positions: above, below, right, left of the name's platforms
      const tries: [number, number, string][] = [[cx, t - 12, "middle"], [cx, b + 22, "middle"], [r + 12, cy + 4, "start"], [l - 12, cy + 4, "end"]];
      for (const [x, y, anchor] of tries) {
        const x0 = anchor === "middle" ? x - w / 2 : anchor === "start" ? x : x - w;
        const rect: [number, number, number, number] = [x0 - 1, y - 12, x0 + w + 1, y + 3];
        if (!clear(rect)) continue;
        taken.push(rect);
        labels.push({ text, x, y, anchor });
        break;
      }
    }
  }

  // the player's links inside the complex (T-197): drawn by the renderer as bridges, hovered here
  const inside = new Set(members);
  const links = (world.value?.save.links ?? []).filter(([a, b]) => inside.has(a) && inside.has(b) && plat.has(a) && plat.has(b)).map(([a, b]) => {
    const sa = stations.get(a)!, sb = stations.get(b)!, pa = plat.get(a)!, pb = plat.get(b)!;
    const [x1, y1] = px([(pa.a[0] + pa.b[0]) / 2, (pa.a[1] + pa.b[1]) / 2]), [x2, y2] = px([(pb.a[0] + pb.b[0]) / 2, (pb.a[1] + pb.b[1]) / 2]);
    return { x1, y1, x2, y2, title: `You linked ${sa.name} with ${sb.name} ${length(Math.hypot(sa.x - sb.x, sa.y - sb.y))} away` };
  });

  // the selected station's platform last, so its hover wins where platforms cross
  const order = [...st].sort((a, b) => (a.id === id ? 1 : 0) - (b.id === id ? 1 : 0));
  return (
    <div class="complex-map" ref={box} style={{ position: "relative" }}>
      <canvas ref={cv} style={{ position: "absolute", left: 0, top: 0, width: W + "px", height: H + "px", pointerEvents: "none", background: "var(--map-land)" }} />
      <svg width={W} height={H} style={{ width: W + "px", height: H + "px", position: "relative" }} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Map of the complex's platforms">
        <rect x="0.5" y="0.5" width={W - 1} height={H - 1} fill="none" stroke="var(--rule)" />
        {ok && links.map((l) => (
          <line x1={l.x1} y1={l.y1} x2={l.x2} y2={l.y2} stroke="#fff" stroke-opacity="0" stroke-width="8" class="complex-map-link" data-tip={l.title} />
        ))}
        {ok && order.map((s) => {
          const p = plat.get(s.id);
          if (!p) return null;
          const [ax, ay] = px(p.a), [bx, by] = px(p.b);
          return <line x1={ax} y1={ay} x2={bx} y2={by} stroke="#fff" stroke-opacity="0" stroke-width={Math.max(MIN_HIT, 2 * p.half * S + 4)} stroke-linecap="round"
            class="complex-map-platform" onClick={() => onOpen(s.id)} data-tip={tip(s)} />;
        })}
        {labels.map((l) => (
          <text x={l.x} y={l.y} text-anchor={l.anchor} class="complex-map-label">{l.text}</text>
        ))}
        {ok && (
          <g transform={`translate(${W - 8 - bar * k} ${H - 8})`}>
            <line x1="0" x2={bar * k} y1="0" y2="0" stroke="var(--muted)" stroke-width="1.5" />
            <text x={(bar * k) / 2} y="-4" text-anchor="middle" class="complex-map-scale">{barLabel}</text>
          </g>
        )}
      </svg>
    </div>
  );
}
