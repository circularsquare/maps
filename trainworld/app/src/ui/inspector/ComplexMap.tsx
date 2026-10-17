// A station complex from above (T-147): each member's platform at its true relative position and
// heading, with the lines stopping there drawn through it, each as wide as the riders on that
// line's trains at that platform a day. North is up; one scale for the whole complex. Shown in the
// station inspector's complex view (others.tsx `ComplexSummary`).
//
// Platforms that lie on top of each other (parallel within 15 degrees, centre lines within 12 m;
// the real New York save gives every service its own station, so Grand Central's Metro-North
// lines are eight platforms in one place) are drawn as one bundle with their lines side by side,
// in the order they lie across, the way the main map draws lines sharing track.
//
// Transfers (T-176): the demand solve counts riders changing from each line at each platform to
// each other (`complexTransfers`, notes/T-147.md). The busiest changes between lines in different
// bundles, both directions together, are drawn as grey curves over the platforms, across the gap
// between the two lines' bands (middle to middle where they cross), as wide as the riders
// changing a day; at most 8, and none under 2% of the complex's changes.
// Changes within one bundle (across a platform) are not drawn; each line's hover lists its
// biggest changes, these included. Stations the player linked (T-197) are joined by a dashed line.

import { useEffect, useLayoutEffect, useRef, useState } from "preact/hooks";
import { complexTransfers, demandView, lineDemand, type ComplexTransfers, type LineDemand } from "../../game/demand";
import { count, length, scaleBars } from "../../game/format";
import { segmentGap } from "../../game/complexes";
import type { LineInput, StationInput } from "../../game/types";
import { selection } from "../../game/ui";
import { world } from "../../game/world";
import { PULLING_IN_S, type ComplexNow } from "../../game/complexLive";
import { PLATFORM_BLUE } from "../../game/palette";

/** Riders on a line's trains at its `k`th stop a day, both directions: in each direction the
 * busier of the arriving and leaving train loads, every period summed. */
export function ridersAt(dm: LineDemand, n: number, k: number): number {
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

/** One line's stop at one platform (line null: a platform no line stops at). */
interface Stroke {
  s: StationInput;
  line: LineInput | null;
  boardings: number;
  riders: number;
  known: boolean;
  w: number;
  /** the platform itself (T-246): a strip after the station's lines, filling with its riders */
  plat?: boolean;
}

/** Platforms drawn as one: world centre, heading, length; strokes across, in order. */
interface Bundle {
  x: number;
  y: number;
  heading: number;
  length: number;
  strokes: Stroke[];
  across: number;
}

const PAD = 16; // px kept clear round the platforms
const STUB = 7; // px the lines run past each platform end
const MIN_W = 1.5, MAX_W = 6; // px, a line's width at the least and the most riders
const MAX_PX_PER_M = 1.2;
const BUNDLE_M = 12; // centre lines this close, and parallel, are drawn as one bundle
const PARALLEL = Math.sin((15 * Math.PI) / 180);
const TOP_CHANGES = 8; // transfer curves drawn at most
const MIN_CHANGE_SHARE = 0.02; // and none carrying less than this share of the complex's changes
const XMIN_W = 1.5, XMAX_W = 8; // px, a transfer curve's width at the least and the most riders
const MIN_GAP = 16; // px: bands nearer than this are joined middle to middle
const PLAT_W = 5; // px, a platform's strip beside its lines (T-246)
const CROWD = "#77726d", FULL = "#d23c3c"; // the map's platform fill and its over-full red (renderer.ts)

type Pt = [number, number];
/** The nearest points of segments p0-p1 and q0-q1 (the midpoints of their overlap if parallel). */
function nearest(p0: Pt, p1: Pt, q0: Pt, q1: Pt): [Pt, Pt] {
  const ux = p1[0] - p0[0], uy = p1[1] - p0[1], vx = q1[0] - q0[0], vy = q1[1] - q0[1];
  const wx = p0[0] - q0[0], wy = p0[1] - q0[1];
  const a = ux * ux + uy * uy, b = ux * vx + uy * vy, c = vx * vx + vy * vy, d = ux * wx + uy * wy, e = vx * wx + vy * wy;
  const clamp = (t: number) => Math.max(0, Math.min(1, t));
  const at = (p: Pt, x: number, y: number, t: number): Pt => [p[0] + x * t, p[1] + y * t];
  const den = a * c - b * b;
  let s: number, t: number;
  if (a < 1e-9 && c < 1e-9) return [p0, q0];
  if (a < 1e-9) return [p0, at(q0, vx, vy, clamp(e / c))];
  if (c < 1e-9) return [at(p0, ux, uy, clamp(-d / a)), q0];
  if (den > 1e-9 * a * c) {
    s = clamp((b * e - c * d) / den);
  } else {
    // parallel: the middle of the stretch of p that lies alongside q
    const t0 = clamp(-d / a), t1 = clamp((b - d) / a);
    s = (t0 + t1) / 2;
  }
  t = (b * s + e) / c;
  if (t < 0 || t > 1) {
    t = clamp(t);
    s = clamp((b * t - d) / a);
  }
  return [at(p0, ux, uy, s), at(q0, vx, vy, t)];
}

/** One line at one platform as transfers name it. */
const strokeKey = (station: string, line: string) => `${station}|${line}`;

/** The changes between two strokes, both directions. */
interface Change {
  a: Stroke;
  b: Stroke;
  ab: number;
  ba: number;
}

let measureCtx: CanvasRenderingContext2D | null | undefined;
function textWidth(t: string): number {
  if (measureCtx === undefined) {
    measureCtx = document.createElement("canvas").getContext("2d");
    if (measureCtx) measureCtx.font = `12px "Zen Maru Gothic", sans-serif`;
  }
  return measureCtx ? measureCtx.measureText(t).width : t.length * 6.6;
}

/** The name as the mini-map labels it: the part before a slash ("Times Sq-42 St" of "Times
 * Sq-42 St/42 St-Port Authority Bus Terminal"), the whole name on hover and in the list. */
const shortName = (n: string) => n.split("/")[0].trim() || n;

function bundles(st: StationInput[], at: Map<string, Stroke[]>): Bundle[] {
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
  return [...groups.values()].map((ss) => {
    const lead = ss.reduce((a, b) => (b.length > a.length ? b : a));
    const h = lead.heading, ux = Math.cos(h), uy = Math.sin(h), nx = -uy, ny = ux;
    let lo = Infinity, hi = -Infinity;
    const side = new Map<string, number>();
    for (const s of ss) {
      const along = (s.x - lead.x) * ux + (s.y - lead.y) * uy;
      const half = (s.length / 2) * Math.abs(Math.cos(s.heading - h));
      lo = Math.min(lo, along - half);
      hi = Math.max(hi, along + half);
      side.set(s.id, (s.x - lead.x) * nx + (s.y - lead.y) * ny);
    }
    // across the bundle in the order the platforms lie (left of the heading first)
    const order = [...ss].sort((a, b) => side.get(b.id)! - side.get(a.id)!);
    const perp = ss.reduce((a, s) => a + side.get(s.id)!, 0) / ss.length;
    const mid = (lo + hi) / 2;
    const strokes = order.flatMap((s) => at.get(s.id)!);
    return {
      x: lead.x + ux * mid + nx * perp, y: lead.y + uy * mid + ny * perp, heading: h, length: hi - lo,
      strokes, across: strokes.reduce((a, t) => a + t.w, 0),
    };
  });
}

/** A line colour darkened as the map's trains are (the full gauge, 48% of the line's rgb). */
function trainShade(hex: string): string {
  const v = [1, 3, 5].map((i) => Math.round(parseInt(hex.slice(i, i + 2), 16) * 0.48));
  return `rgb(${v[0]},${v[1]},${v[2]})`;
}

export function ComplexMap({ id, members, stations, lines, now, onOpen }: {
  id: string; members: string[]; stations: Map<string, StationInput>; lines: LineInput[];
  /** riders waiting and trains at the platforms now (T-246, game/complexLive.ts) */
  now?: ComplexNow | null;
  /** a click on a platform: open that station */
  onOpen?: (station: string) => void;
}) {
  const open = onOpen ?? ((m: string) => (selection.value = { kind: "station", station: m }));
  const box = useRef<HTMLDivElement>(null);
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
  const [xf, setXf] = useState<ComplexTransfers | null>(null);
  useEffect(() => {
    if (!solved) return;
    let on = true;
    void complexTransfers(members).then((r) => { if (on && r) setXf(r); });
    return () => { on = false; };
  }, [scope, solved]);
  const transfers = xf && dv && xf.scope === scope && xf.version === dv.version ? xf : null;

  const st = members.map((m) => stations.get(m)).filter((s): s is StationInput => !!s);
  if (st.length < 2) return null;
  // what stops at each platform
  const at = new Map<string, Stroke[]>(st.map((s) => [s.id, []]));
  for (const l of lines) {
    const dm = lineDemand(l.id);
    l.stops.forEach((sid, k) => {
      const here = at.get(sid);
      if (here) here.push({ s: stations.get(sid)!, line: l, boardings: dm?.boardings[k] ?? 0, riders: dm ? ridersAt(dm, l.stops.length, k) : 0, known: !!dm, w: 0 });
    });
  }
  for (const s of st) if (!at.get(s.id)!.length) at.get(s.id)!.push({ s, line: null, boardings: 0, riders: 0, known: false, w: 0 });
  const all = [...at.values()].flat();
  const most = Math.max(0, ...all.map((t) => t.riders));
  for (const t of all) t.w = !t.line ? 4 : t.known && most > 0 ? MIN_W + (MAX_W - MIN_W) * Math.sqrt(t.riders / most) : 3;
  // each platform's strip after its lines (T-246), where its riders waiting are drawn
  for (const s of st) if (at.get(s.id)![0].line) at.get(s.id)!.push({ s, line: null, boardings: 0, riders: 0, known: false, w: PLAT_W, plat: true });
  const bs = bundles(st, at);

  // fit every bundle's ends, north up
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const b of bs) {
    const h = b.length / 2, dx = Math.abs(Math.cos(b.heading) * h), dy = Math.abs(Math.sin(b.heading) * h);
    x0 = Math.min(x0, b.x - dx); x1 = Math.max(x1, b.x + dx);
    y0 = Math.min(y0, b.y - dy); y1 = Math.max(y1, b.y + dy);
  }
  const edge = PAD + STUB;
  const bw = Math.max(1, x1 - x0), bh = Math.max(1, y1 - y0);
  const H = Math.round(Math.max(120, Math.min(240, ((W - 2 * edge) * bh) / bw + 2 * edge)));
  const k = Math.min(MAX_PX_PER_M, (W - 2 * edge) / bw, (H - 2 * edge) / bh);
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  const px = (x: number) => W / 2 + (x - cx) * k;
  const py = (y: number) => H / 2 - (y - cy) * k;
  const drawn = bs.map((b) => ({ b, len: Math.max(6, b.length * k), deg: (-b.heading * 180) / Math.PI, sx: px(b.x), sy: py(b.y) }));

  // Transfers: each stroke's band as a segment on screen (its centre line, platform length), then
  // the changes between strokes, both directions together, inside the complex only (a walk to a
  // station outside it is left out).
  const place = new Map<Stroke, { d: (typeof drawn)[number]; a: [number, number]; b: [number, number] }>();
  const byKey = new Map<string, Stroke>();
  for (const d of drawn) {
    let off = -d.b.across / 2;
    const sin = Math.sin(d.b.heading), cos = Math.cos(d.b.heading), half = d.len / 2;
    for (const t of d.b.strokes) {
      const c = off + t.w / 2;
      off += t.w;
      // local (x, c) under translate(sx, sy) rotate(-heading) is (sx + x cos + c sin, sy - x sin + c cos)
      const x = d.sx + c * sin, y = d.sy + c * cos;
      place.set(t, { d, a: [x - half * cos, y + half * sin], b: [x + half * cos, y - half * sin] });
      if (t.line && !byKey.has(strokeKey(t.s.id, t.line.id))) byKey.set(strokeKey(t.s.id, t.line.id), t);
    }
  }
  const changes = new Map<string, Change>();
  const partners = new Map<Stroke, Map<string, number>>();
  let changesInside = 0;
  for (const p of transfers?.pairs ?? []) {
    const a = byKey.get(strokeKey(p.fromStation, p.fromLine)), b = byKey.get(strokeKey(p.toStation, p.toLine));
    if (!a || !b) continue;
    changesInside += p.riders;
    if (a === b) continue; // back the way they came on the same line
    const [x, y] = strokeKey(a.s.id, a.line!.id) < strokeKey(b.s.id, b.line!.id) ? [a, b] : [b, a];
    const key = `${strokeKey(x.s.id, x.line!.id)}#${strokeKey(y.s.id, y.line!.id)}`;
    const c = changes.get(key) ?? { a: x, b: y, ab: 0, ba: 0 };
    if (x === a) c.ab += p.riders; else c.ba += p.riders;
    changes.set(key, c);
    for (const [s, o] of [[a, b], [b, a]] as const) {
      const m = partners.get(s) ?? new Map<string, number>();
      const name = o.line!.name + (o.s.id === s.s.id ? "" : ` at ${shortName(o.s.name)}`);
      m.set(name, (m.get(name) ?? 0) + p.riders);
      partners.set(s, m);
    }
  }
  const between = [...changes.values()]
    .filter((c) => place.get(c.a)!.d !== place.get(c.b)!.d && c.ab + c.ba >= MIN_CHANGE_SHARE * changesInside)
    .sort((x, y) => y.ab + y.ba - (x.ab + x.ba))
    .slice(0, TOP_CHANGES);
  const busiest = Math.max(0, ...between.map((c) => c.ab + c.ba));
  const curves = between.map((c) => {
    // across the gap between two platforms, from where their bands come nearest; bands that cross
    // or nearly touch would make a knot of loops there, so those join their middles instead
    const pa = place.get(c.a)!, pb = place.get(c.b)!;
    let [A, B] = nearest(pa.a, pa.b, pb.a, pb.b);
    if (Math.hypot(B[0] - A[0], B[1] - A[1]) < MIN_GAP) {
      A = [(pa.a[0] + pa.b[0]) / 2, (pa.a[1] + pa.b[1]) / 2];
      B = [(pb.a[0] + pb.b[0]) / 2, (pb.a[1] + pb.b[1]) / 2];
    }
    const dx = B[0] - A[0], dy = B[1] - A[1], len = Math.hypot(dx, dy);
    // a gentle bow to the left of a to b, so changes along one line of sight stay apart; two
    // middles at one spot get a small loop
    const bow = Math.max(10, Math.min(30, 0.2 * len));
    const path = len < 4
      ? `M${A[0]},${A[1]} C${A[0] - 0.8 * bow},${A[1] - 1.6 * bow} ${B[0] + 0.8 * bow},${B[1] - 1.6 * bow} ${B[0]},${B[1]}`
      : (() => {
        const nx = -dy / len, ny = dx / len;
        return `M${A[0]},${A[1]} C${A[0] + dx / 3 + nx * bow},${A[1] + dy / 3 + ny * bow} ${A[0] + (2 * dx) / 3 + nx * bow},${A[1] + (2 * dy) / 3 + ny * bow} ${B[0]},${B[1]}`;
      })();
    const w = XMIN_W + (XMAX_W - XMIN_W) * Math.sqrt((c.ab + c.ba) / busiest);
    const name = (t: Stroke) => `${t.line!.name} at ${shortName(t.s.name)}`;
    const title = `${count(c.ab + c.ba)} riders a day change between ${name(c.a)} and ${name(c.b)}\n` +
      `${count(c.ab)} from ${c.a.line!.name} to ${c.b.line!.name}, ${count(c.ba)} the other way`;
    return { path, w, title };
  });

  const lineById = new Map(lines.map((l) => [l.id, l]));
  const nameOf = (sid: string) => stations.get(sid)?.name ?? "";
  // the platform's riders now (T-246): each line and direction, as the station view's Waiting now
  const platTip = (t: Stroke) => {
    const name = t.s.name + (t.s.built ? "" : " (blueprint)");
    const p = now?.platforms.get(t.s.id);
    if (!now || !p) return `${name}\nNobody waiting: no train calls here this hour`;
    // one figure a line name, its directions and branches summed (T-250)
    const byName = new Map<string, number>();
    for (const q of now.queues) {
      const l = q.station === t.s.id ? lineById.get(q.line) : undefined;
      if (l) byName.set(l.name, (byName.get(l.name) ?? 0) + q.waiting);
    }
    const total = p.side[0] + p.side[1];
    let text = `${name}\n${count(total)} waiting on the platform now`;
    if (byName.size > 1) for (const [n, v] of [...byName].sort((a, b) => b[1] - a[1])) text += `\n${count(v)} for ${n}`;
    text += p.oneSide ? `\nRoom for ${count(p.room)} on the platform` : `\nRoom for ${count(p.room)} on each side`;
    return text;
  };
  const trainTip = (x: ComplexNow["trains"][number]) => {
    const l = lineById.get(x.line);
    const at = shortName(nameOf(x.station)), to = x.to !== x.station ? `to ${nameOf(x.to)}` : x.inS > 0 ? "ending its trip here" : "at the end of its trip";
    return x.inS > 0 ? `${l?.name ?? ""} train ${to}\nPulling in to ${at} in ${Math.max(1, Math.round(x.inS))} s` : `${l?.name ?? ""} train ${to}\nStanding at ${at}`;
  };
  const tip = (t: Stroke) => {
    const name = t.s.name + (t.s.built ? "" : " (blueprint)");
    if (t.plat) return platTip(t);
    if (!t.line) return `${name}\nNo line stops here`;
    let text = `${name}\n${t.line.name}` + (t.known ? `: ${count(t.boardings)} boardings, ${count(t.riders)} riders on its trains here a day` : "");
    const m = partners.get(t);
    if (m) {
      const top = [...m].sort((x, y) => y[1] - x[1]);
      const all = top.reduce((a, [, v]) => a + v, 0);
      text += `\n${count(all)} change to or from other lines here a day, most with ` +
        top.slice(0, 3).map(([n, v]) => `${n} (${count(v)})`).join(", ");
    }
    return text;
  };

  // Name labels, when the complex has more than one name: one per name (T-169), busiest first,
  // beside its platforms, clear of every platform, other label and the scale bar.
  const bars = scaleBars();
  const [bar, barLabel] = bars.find(([m]) => m * k >= 40) ?? bars[bars.length - 1];
  const barRect: [number, number, number, number] = [W - 14 - bar * k, H - 24, W, H];
  const taken: [number, number, number, number][] = [barRect];
  const samples: [number, number, number][] = []; // points along each bundle: x, y, radius
  for (const d of drawn) {
    const ux = Math.cos(d.b.heading), uy = -Math.sin(d.b.heading), half = d.len / 2 + STUB;
    for (let a = -half; a <= half; a += 3) samples.push([d.sx + ux * a, d.sy + uy * a, d.b.across / 2 + 2]);
  }
  const clear = (r: [number, number, number, number]) =>
    r[0] >= 2 && r[2] <= W - 2 && r[1] >= 2 && r[3] <= H - 2 &&
    !taken.some((q) => r[0] < q[2] && q[0] < r[2] && r[1] < q[3] && q[1] < r[3]) &&
    !samples.some(([x, y, rad]) => x > r[0] - rad && x < r[2] + rad && y > r[1] - rad && y < r[3] + rad);
  const byName = new Map<string, StationInput[]>();
  for (const s of st) byName.set(s.name, [...(byName.get(s.name) ?? []), s]);
  const labels: { text: string; x: number; y: number; anchor: string }[] = [];
  if (byName.size > 1) {
    const weight = (ss: StationInput[]) => ss.reduce((a, s) => a + at.get(s.id)!.length, 0);
    for (const [name, all] of [...byName].sort((a, b) => weight(b[1]) - weight(a[1]))) {
      // beside the bundle holding most of the name's lines: a name can be on two platforms
      // apart (Penn's 34 St-Penn Station is the 1/2/3 and the A/C/E), and their midpoint is
      // somewhere else
      const home = bs.reduce((a, b) => (b.strokes.filter((t) => t.s.name === name).length > a.strokes.filter((t) => t.s.name === name).length ? b : a));
      const ss = all.filter((s) => home.strokes.some((t) => t.s.id === s.id));
      let l = Infinity, r = -Infinity, t = Infinity, b = -Infinity;
      for (const s of ss) {
        const h = s.length / 2, dx = Math.cos(s.heading) * h, dy = Math.sin(s.heading) * h;
        for (const [x, y] of [[s.x - dx, s.y - dy], [s.x + dx, s.y + dy]]) {
          l = Math.min(l, px(x)); r = Math.max(r, px(x)); t = Math.min(t, py(y)); b = Math.max(b, py(y));
        }
      }
      const text = shortName(name), w = textWidth(text), mx = (l + r) / 2, my = (t + b) / 2;
      // baseline positions: above, below, right, left of the name's platforms
      const tries: [number, number, string][] = [[mx, t - 8, "middle"], [mx, b + 18, "middle"], [r + 8, my + 4, "start"], [l - 8, my + 4, "end"]];
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

  // the player's links inside the complex (T-197): a dashed line between the two stations
  const inside = new Set(members);
  const links = (world.value?.save.links ?? []).filter(([a, b]) => inside.has(a) && inside.has(b) && stations.has(a) && stations.has(b)).map(([a, b]) => {
    const sa = stations.get(a)!, sb = stations.get(b)!;
    return { x1: px(sa.x), y1: py(sa.y), x2: px(sb.x), y2: py(sb.y), title: `You linked ${sa.name} with ${sb.name} ${length(Math.hypot(sa.x - sb.x, sa.y - sb.y))} away` };
  });

  const band = (d: (typeof drawn)[number], from: number, w: number, extra: number) => ({ x: -d.len / 2 - extra, y: from, width: d.len + 2 * extra, height: w });
  return (
    <div class="complex-map" ref={box}>
      <svg width={W} height={H} style={{ width: W + "px", height: H + "px" }} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Map of the complex's platforms">
        <rect x="0.5" y="0.5" width={W - 1} height={H - 1} fill="var(--map-land)" stroke="var(--rule)" />
        {links.map((l) => (
          <g>
            <line x1={l.x1} y1={l.y1} x2={l.x2} y2={l.y2} stroke="var(--net-ink)" stroke-opacity="0.6" stroke-width="1.5" stroke-dasharray="4 3" />
            <line x1={l.x1} y1={l.y1} x2={l.x2} y2={l.y2} stroke="#fff" stroke-opacity="0" stroke-width="8" class="complex-map-link" data-tip={l.title} />
          </g>
        ))}
        {drawn.map((d) => {
          let off = -d.b.across / 2;
          const built = d.b.strokes.some((t) => t.s.built);
          const mine = d.b.strokes.filter((t) => t.s.id === id);
          let mineFrom = NaN;
          const rows = d.b.strokes.map((t) => {
            const y = off;
            off += t.w;
            if (t.s.id === id && Number.isNaN(mineFrom)) mineFrom = y;
            return { t, y };
          });
          return (
            <g transform={`translate(${d.sx} ${d.sy}) rotate(${d.deg})`}>
              {rows.map(({ t, y }) => (
                <rect {...band(d, y, t.w, t.line ? STUB : 0)} fill={t.line ? t.line.colour : "#fff"} opacity={t.s.built ? 1 : 0.6} />
              ))}
              {/* each platform's riders waiting now (T-246), as the map fills its boxes: each
                  direction's half from the middle out by its share of the room, red once full
                  (T-559); then the platform's blue edges (T-236) */}
              {rows.filter(({ t }) => t.plat).map(({ t, y }) => {
                const p = now?.platforms.get(t.s.id);
                const half = d.len / 2 - 0.5, sg = Math.cos(t.s.heading - d.b.heading) >= 0 ? 0 : 1;
                const fills = !p ? [] : p.oneSide
                  ? [{ share: (p.side[0] + p.side[1]) / p.room, y0: y + 0.5, h: PLAT_W - 1 }]
                  : [0, 1].map((k) => ({ share: p.side[k ^ sg] / p.room, y0: y + 0.5 + k * (PLAT_W - 1) / 2, h: (PLAT_W - 1) / 2 }));
                return (
                  <g>
                    {fills.map((f) => {
                      const fl = Math.min(1, f.share) * half;
                      return fl > 0.05 ? <rect x={-fl} y={f.y0} width={2 * fl} height={f.h} fill={f.share >= 1 ? FULL : CROWD} /> : null;
                    })}
                    <line x1={-half} x2={half} y1={y + 0.5} y2={y + 0.5} stroke={PLATFORM_BLUE} stroke-width="1" />
                    <line x1={-half} x2={half} y1={y + PLAT_W - 0.5} y2={y + PLAT_W - 0.5} stroke={PLATFORM_BLUE} stroke-width="1" />
                  </g>
                );
              })}
              <rect {...band(d, -d.b.across / 2 - 1, d.b.across + 2, 0)} rx="1" fill="none" stroke="var(--net-ink)" stroke-width="1" stroke-opacity="0.7"
                stroke-dasharray={built ? undefined : "3 2"} />
              {mine.length > 0 && (
                <rect {...band(d, mineFrom - 1.5, mine.reduce((a, t) => a + t.w, 0) + 3, 1.5)} rx="1.5" fill="none" stroke="var(--net-ink)" stroke-width="2" />
              )}
              {rows.map(({ t, y }) => (
                <rect {...band(d, y - (t.w < 4 ? (4 - t.w) / 2 : 0), Math.max(4, t.w), t.plat ? 0 : STUB)} fill="#fff" fill-opacity="0" class="complex-map-platform"
                  onClick={() => open(t.s.id)} data-tip={tip(t)} />
              ))}
              {/* trains standing at a platform, or pulling in from the end they arrive by (T-246) */}
              {(now?.trains ?? []).map((x) => {
                const row = rows.find(({ t }) => t.s.id === x.station && t.line?.id === x.line);
                if (!row) return null;
                const l = row.t.line!, h = row.t.w + 1, y = row.y - 0.5;
                const dirB = x.dir * (Math.cos(row.t.s.heading - d.b.heading) >= 0 ? 1 : -1);
                // pulling in: a paler train coming in from the end it arrives by, its front further
                // along the platform the sooner it stops
                const L = Math.max(6, 0.35 * d.len), front = 3 + (1 - x.inS / PULLING_IN_S) * 0.45 * d.len;
                const a0 = Math.max(-STUB, front - L), a1 = front;
                const x0 = x.inS > 0 ? (dirB > 0 ? -d.len / 2 + a0 : d.len / 2 - a1) : -d.len / 2 + 1;
                const wd = x.inS > 0 ? a1 - a0 : Math.max(4, d.len - 2);
                return <rect x={x0} y={y} width={wd} height={h} rx={h / 2} fill={trainShade(l.colour)} fill-opacity={x.inS > 0 ? 0.5 : 1} stroke="#fff" stroke-width="1" class="complex-map-train" data-tip={trainTip(x)} />;
              })}
            </g>
          );
        })}
        {/* over the platforms, each with a thin halo of the land so it reads across their colours */}
        {curves.map((c) => (
          <path d={c.path} fill="none" stroke="var(--map-land)" stroke-opacity="0.8" stroke-width={c.w + 2} stroke-linecap="round" />
        ))}
        {curves.map((c) => (
          <g>
            <path d={c.path} fill="none" stroke="var(--net-ink)" stroke-opacity="0.55" stroke-width={c.w} stroke-linecap="round" />
            <path d={c.path} fill="none" stroke="#fff" stroke-opacity="0" stroke-width={Math.max(6, c.w + 2)} class="complex-map-change" data-tip={c.title} />
          </g>
        ))}
        {labels.map((l) => (
          <text x={l.x} y={l.y} text-anchor={l.anchor} class="complex-map-label">{l.text}</text>
        ))}
        <g transform={`translate(${W - 8 - bar * k} ${H - 8})`}>
          <line x1="0" x2={bar * k} y1="0" y2="0" stroke="var(--muted)" stroke-width="1.5" />
          <text x={(bar * k) / 2} y="-4" text-anchor="middle" class="complex-map-scale">{barLabel}</text>
        </g>
      </svg>
      {most > 0 && <p class="muted small">Each line is as wide as the riders on its trains at that platform a day.</p>}
      {curves.length > 0 && <p class="muted small">Grey curves join the lines with the most riders changing between platforms, as wide as those riders a day.</p>}
      {links.length > 0 && <p class="muted small">Dashed lines join stations you linked.</p>}    </div>
  );
}
