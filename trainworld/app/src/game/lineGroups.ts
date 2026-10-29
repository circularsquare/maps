// T-1791: the player's line groups, a grouping for the Lines tab that the service never reads. A
// group is the lines sharing a group number; one left with a single line is no group.

import { signal } from "@preact/signals";
import type { LineInput, Selection } from "./types";

/** Each group's lines, oldest (lowest number) first; only groups of two or more. */
export function lineGroups(lines: readonly LineInput[]): Map<number, LineInput[]> {
  const by = new Map<number, LineInput[]>();
  for (const l of lines) if (l.group) by.set(l.group, [...(by.get(l.group) ?? []), l]);
  for (const [g, ls] of by) {
    if (ls.length < 2) by.delete(g);
    else ls.sort((a, b) => a.num - b.num);
  }
  return by;
}

/** The group's name: the player's, else its oldest line's. */
export function groupName(members: readonly LineInput[]): string {
  return members[0]?.groupName || members[0]?.name || "";
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
 * name and the colour most of its lines share, the busiest line's (`riders`) deciding a tie. */
export function groupBullet(members: readonly LineInput[], riders: (id: string) => number): { text: string; colour: string } {
  const m = members[0];
  const text = m?.groupSymbol || groupLabel(groupName(members));
  if (m?.groupColour) return { text, colour: m.groupColour };
  const n = new Map<string, number>();
  for (const l of members) n.set(l.colour.toLowerCase(), (n.get(l.colour.toLowerCase()) ?? 0) + 1);
  const most = Math.max(0, ...n.values());
  let best: LineInput | undefined;
  for (const l of members) if (n.get(l.colour.toLowerCase()) === most && (!best || riders(l.id) > riders(best.id))) best = l;
  return { text, colour: best?.colour ?? "#888888" };
}

/** What Copy line opens (T-1791): the copy's group view, or the copy where it has no group. */
export function afterCopy(lines: readonly LineInput[], copy: number): Selection {
  const g = lines.find((l) => l.num === copy)?.group;
  return g ? { kind: "group", group: g } : { kind: "line", line: String(copy) };
}

/** The line whose column the pointer is on in the group view: the map draws it full, the rest faint. */
export const groupHover = signal<number | null>(null);

/** T-1809: the group view names every stop, not only the key ones; kept while the page is open. */
export const groupAllStops = signal(false);

/** The groups opened in the Lines tab, kept while the page is open. */
export const openGroups = signal<ReadonlySet<number>>(new Set());

export function toggleGroup(g: number) {
  const s = new Set(openGroups.peek());
  if (!s.delete(g)) s.add(g);
  openGroups.value = s;
}
