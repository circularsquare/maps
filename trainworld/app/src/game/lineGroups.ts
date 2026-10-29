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
