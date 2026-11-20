# gearfix (silent, Anita 2026-11-20)
Two faults, both from T-3630 (3533773a, 2026-11-18), already in v0.4.3 (8c4c00b2); v0.4.4 did not cause them.
1. Start-menu gear: `.start-foot-line > *` sets display:inline-flex (no justify-content), overriding .btn's inline-grid centring, so the svg sat at the left: 4.4 px off. Fix: `justify-content:center` on `.start-foot .start-gear`.
2. Start settings "Panel and text size" value: `.slider span { min-width: 3.6em }` hit the value span inside the stepper (the start panel wraps it in .slider), so the text hugged the left of the 47 px cell. Fix: `.slider .stepper span { min-width: 0 }`. The Settings tab stepper is not in a .slider and was always fine.
Checked by measuring svg/text centre against box for every icon button and stepper on the start page and Settings tab: all within 0.5 px after. Account panel not reachable signed out; its icon buttons use plain .btn.icon (no flex override).
Dark shots look the same as light: the game has no prefers-color-scheme theme.
Shots: artifacts/trainworld/gearfix.png. Scratch: shoot.mjs, sheet.py in the scratchpad.
