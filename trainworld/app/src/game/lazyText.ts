// A list of names read by index (a day kind's name, a month's) that follows the language: each item reads its
// message when it is read, not when the module loads. Use it where a code indexes an array exported to many files.

import { t, type MessageKey } from "../i18n";

export function lazyTexts<T extends string = string>(keys: readonly MessageKey[]): readonly T[] {
  const a = new Array<string>(keys.length);
  keys.forEach((k, i) => Object.defineProperty(a, i, { get: () => t(k), enumerable: true }));
  return a as unknown as readonly T[];
}
