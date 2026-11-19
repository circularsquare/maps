// T-1791: the player's line groups, a grouping for the Lines tab that the service never reads. A
// group is the lines sharing a group number; one left with a single member is no group. T-3423: a
// group can sit in another, to any depth; each line carries the groups around its own (`outer`).

import { signal } from "@preact/signals";
import type { LineInput, OuterGroup, Selection } from "./types";

/** A line's groups, its own first, then each one around it. */
export function groupChain(l: LineInput): OuterGroup[] {
  if (!l.group) return [];
  return [{ group: l.group, name: l.groupName ?? "", symbol: l.groupSymbol ?? "", colour: l.groupColour ?? "" }, ...(l.outer ?? [])];
}

/** Each group's lines at any depth, oldest (lowest number) first; only groups of two or more. */
export function lineGroups(lines: readonly LineInput[]): Map<number, LineInput[]> {
  const by = new Map<number, LineInput[]>();
  for (const l of lines) for (const o of groupChain(l)) by.set(o.group, [...(by.get(o.group) ?? []), l]);
  for (const [g, ls] of by) {
    if (ls.length < 2) by.delete(g);
    else ls.sort((a, b) => a.num - b.num);
  }
  return by;
}

/** T-3423: a group's place in the tree: the group around it (0 for none), and what sits straight in it. */
export interface GroupNode { parent: number; lines: LineInput[]; groups: number[] }

/** Every group's place in the tree; a group with fewer than two members is left out. */
export function groupTree(lines: readonly LineInput[]): Map<number, GroupNode> {
  const by = new Map<number, GroupNode>();
  const node = (g: number) => by.get(g) ?? by.set(g, { parent: 0, lines: [], groups: [] }).get(g)!;
  for (const l of lines) {
    const c = groupChain(l);
    if (!c.length) continue;
    node(c[0].group).lines.push(l);
    for (let k = 0; k < c.length; k++) {
      const n = node(c[k].group), up = c[k + 1]?.group ?? 0;
      n.parent = up;
      if (up && !node(up).groups.includes(c[k].group)) node(up).groups.push(c[k].group);
    }
  }
  for (const [g, n] of by) if (n.lines.length + n.groups.length < 2) by.delete(g);
  return by;
}

/** A group's own entry (name and bullet) as its oldest line has it. */
export function groupMeta(members: readonly LineInput[], g: number): OuterGroup | undefined {
  return groupChain(members[0] ?? ({} as LineInput)).find((o) => o.group === g);
}

/** The group's name: the player's, else its oldest line's (`members`: its lines, oldest first). */
export function groupName(members: readonly LineInput[], g: number): string {
  return groupMeta(members, g)?.name || members[0]?.name || "";
}

/** T-1813: a group's bullet text from its name: a trailing " line" or " main line" goes ("Tokaido
 * Main Line" reads "Tokaido"), and so does 線, 线, 本線, ライン or 선 ("東海道本線" reads "東海道") unless a
 * number is left ("1号线", "1호선" stay); a name that is only that stays. */
export function groupLabel(name: string): string {
  const n = name.trim();
  const t = n.replace(/\s+(main\s+)?line$/i, "").replace(/(本線|ライン|[線线선])$/u, "");
  return t && !/[\d号號호]$/u.test(t) ? t : n.replace(/\s+(main\s+)?line$/i, "") || n;
}

/** T-1813: a group's bullet, its text and colour: the player's where set, else the label of its
 * name and the colour most of its lines (at any depth) share, the busiest line's (`riders`) deciding a tie. */
export function groupBullet(members: readonly LineInput[], g: number, riders: (id: string) => number): { text: string; colour: string } {
  const m = groupMeta(members, g);
  const text = m?.symbol || groupLabel(groupName(members, g));
  if (m?.colour) return { text, colour: m.colour };
  const n = new Map<string, number>();
  for (const l of members) n.set(l.colour.toLowerCase(), (n.get(l.colour.toLowerCase()) ?? 0) + 1);
  const most = Math.max(0, ...n.values());
  let best: LineInput | undefined;
  for (const l of members) if (n.get(l.colour.toLowerCase()) === most && (!best || riders(l.id) > riders(best.id))) best = l;
  return { text, colour: best?.colour ?? "#888888" };
}

/** What Copy line opens: the copy's own panel (T-1894). */
export function afterCopy(_lines: readonly LineInput[], copy: number): Selection {
  return { kind: "line", line: String(copy) };
}

/** The line whose column the pointer is on in the group view: the map draws it full, the rest faint. */
export const groupHover = signal<number | null>(null);

/** T-1809: the group view names every stop, not only the key ones; kept while the page is open. */
export const groupAllStops = signal(false);

/** The groups the player closed in the Lines tab, kept while the page is open; the rest show open (T-1818). */
export const closedGroups = signal<ReadonlySet<number>>(new Set());

export function toggleGroup(g: number) {
  const s = new Set(closedGroups.peek());
  if (!s.delete(g)) s.add(g);
  closedGroups.value = s;
}
