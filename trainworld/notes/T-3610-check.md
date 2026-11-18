# T-3610 check (scheck), PASS

Tree .tw-trees/saveread, Vite 6184. Scripts: scratchpad/scheck/many.mjs, del.mjs.

- Agent's unreadable.mjs: PASS (2 cards, 1 damaged, no unread message). Damaged card's name shows "Saved game" (the real name is in the record, so unknown).
- t2703-storeread.mjs: PASS (all 4).
- 30 slots, no fault: 30 cards in date order. Bad slot-15 (get fails NotFoundError): 30 cards, 29 in exact date order with date and day, damaged one last ("This save is damaged."), no unread message. Its backup still listed by listBackups; an unreadable backup of another slot is left out, no throw.
- With the fault live: slot-14 opens (109 lines), a save writes slot-14 again, slot-15's stored record untouched. Restore of the bad slot's backup via manageSaveSlot('restore') only set up a pending load (not followed through; inconclusive).
- Transient (first get fails UnknownError, then ok): 30 cards, 0 damaged.
- Delete of the damaged card through the start menu (UI): slot and its backup removed.
- Timing, 30 slots of 75 KB, in-page: listSlots 211,43,38,29,46 ms vs old-pattern one-transaction 89,18,26,19,20 ms. About 20 ms more warm, 30 transactions instead of 1. Menu with 30 cards ready in 2.5-4.4 s overall.
- t1826-saves.mjs on this tree: tiers, pull, close prompts, no page errors PASS. 3 FAIL: two are expected-string drift (check says "saved by a newer version", game says "saved on a newer game version"); the third is "live backups ... listed": listBackups returned t0..t3 (4 tiers) but the saves panel showed 1 version row. Panel code (useBackups, stamp per minute) is not in this diff, so likely old; baseline not obtainable (scratch HEAD worktree timed out loading a game, it was removed).
- Skipped: cloud/signed-in paths (stampAll filter read only), t3395 (needs server).
