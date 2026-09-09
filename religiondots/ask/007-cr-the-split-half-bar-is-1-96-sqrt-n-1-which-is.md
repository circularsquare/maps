# 007 — cr: The split-half bar is 1.96/sqrt(n-1), which is a 0.017-level test at seven units and not the 0.05 it says it is

*Filed 2026-09-08 by session `967ffe99-93b1-4ff9-8169-4a6d5ffa084e-cr`. Anita's call; nothing is waiting on it.*

## What I did

I applied the bar exactly as written. Costa Rica's `Católico`, 63.06% of the country, comes
back at +0.7857 against the bar's +0.8002, so it fails, and it is drawn at the national rate
inside each province's residual rather than on its own province shares. Costa Rica shipped
that way. I did not add an `OVERRIDE`, because `lapop.stability`'s own docstring says the bar
is never moved to make something pass and that an override is a person's decision.

## What it costs to reverse

For Costa Rica alone: add `1` to `CARRIES` in `sources/cr.py`, re-run it, re-scatter and
re-run the build tail. About twenty minutes, no new data.

For the rule: a one-line change in `sources/lapop.py::stability` plus a re-run of the five
countries that import it. **It would change exactly two categories in the whole module**, and
I have checked all five (see below), so the blast radius is known rather than guessed.

## Why it is yours rather than mine

AGENT_BRIEF §3, *"a rule everyone shares"* and *"something that changes an already-drawn
country's numbers"*. This is not a Costa Rica mapping call; it is the threshold that decides
which categories carry a geography in Guatemala, El Salvador, Ecuador, Panama and Costa Rica,
and flipping it would re-draw El Salvador as well as Costa Rica.

## The detail

`lapop.stability` sets the bar at `1.96/sqrt(n-1)` and its docstring says why: *"A Spearman
correlation over `n` units has a standard error of about 1/sqrt(n-1), so the bar is what it
takes to be distinguishable from zero at 95%."* **The intent is a 0.05-level test. The
implementation is not one at any unit count this module uses.** `1/sqrt(n-1)` is the
asymptotic standard error, and it is a poor approximation at small `n`.

Enumerating the exact null (every ordering up to n=10, 400,000 samples above):

```
   n   fixed bar   exact null 95th   exact 97.5th   what the fixed bar actually is
   7      0.8002        0.6786          0.7500      a 0.017-level test
  10      0.6533        0.5515          0.6364      a 0.022-level test
  14      0.5436        0.4593          0.5341      a 0.023-level test
  22      0.4277        0.3586          0.4241      a 0.024-level test
  23      0.4179        0.3508          0.4140      a 0.024-level test
```

So the bar is stricter than advertised everywhere, and worst where there are fewest units.
At seven it rejects a correlation that the exact null clears at both the 5% and the 2.5% level.

**What would change, across all five countries.** Only categories that clear the exact 95th
percentile and fail the fixed bar. There are two:

```
  cr  Católico                      63.06%   rho=+0.7857   exact bar +0.6786   fixed +0.8002
  sv  Protestante Tradicional        7.97%   rho=+0.5165   exact bar +0.4593   fixed +0.5436
```

Nothing else moves. Guatemala, Ecuador and Panama are unaffected at any of the three
thresholds; their passes pass comfortably and their failures fail comfortably.

**Two things pulling the other way, so this is genuinely a judgement and not an oversight.**

1. Costa Rica's `Católico` failure is not a borderline artefact. Its rank sum of squared
   differences between the wave halves is 12, and **nine of the twelve are Guanacaste alone**,
   which falls from the fourth most Catholic province to the seventh as its Catholic share
   goes 63.41% to 48.22% between the halves. That is a real fifteen-point move in one
   province, and the split-half is arguably right to withdraw the claim regardless of where
   the bar sits.
2. A stricter-than-advertised bar may be what this project wants. §14.16 treats a low value as
   *a failure to demonstrate signal rather than a demonstration of noise*, and the cost of a
   false pass here is a map that claims to know where a religion is when it does not, which is
   worse than the cost of a false fail, which is the national rate inside a residual.

**And the cost of the current call is small either way for Costa Rica.** `Católico` is 94.9%
of the tail that gets spread through each province's residual, so it is drawn as very nearly
one minus the four categories that did pass: at most 2.54 points from its measured share in
any province, 1.47 on average, with two adjacent pairs of the province ordering swapped. El
Salvador's `Protestante Tradicional` at 7.97% would move more, because it is a much smaller
share of its own country's residual.

**The precedent that made me look.** §9co (Kyrgyzstan, same day) found this same fixed bar
rejecting Islam at 89% of the country, and built the null instead — its own note says
*"Moving a bar to make something pass fits the test to the answer... Building the null does
the opposite: it also raises the bar where the fixed one is too generous."* That country's
statistic was different (a median over 400 PSU splits), so its fix does not transfer. But the
question it raised does, and for the ordinary single-split statistic the exact null is
computable in closed form and cheap.

If you want it changed, the one-line version is to replace `bar = 1.96 / np.sqrt(n_units - 1)`
with the exact null's 95th percentile, enumerated for `n <= 10` and sampled above. I have that
code in `<scratchpad>/967ffe99-cr-spearman-null.py` and it takes about ten seconds at n=23.

---

## Correction, 2026-09-09 — one row of the table above is mislabelled

*Appended by session `967ffe99-93b1-4ff9-8169-4a6d5ffa084e-crfix`, from the independent check
written up in `sources/cr.md` §7. **The ask is not being re-argued and its conclusion is
unaffected**; this is here so the ruling is made on accurate numbers.*

**The `n = 23` row is not a bar Ecuador uses.** Ecuador's split-half runs on the **20**
provinces present in both halves, so its bar is `1.96/sqrt(19) = +0.45`, not the +0.4179 that
23 units would give. At 20 units the exact null's 95th percentile is **+0.379**, and Ecuador's
nearest recorded value is `Testigos de Jehová` at **+0.34**, below both bars. **So Ecuador
still moves nothing under the change proposed above.**

**"Exactly two categories" therefore stands**, and it was re-checked against the five
countries' own records rather than against the list above: `cr` `Católico` at **+0.7857** and
`sv` `Protestante Tradicional` at **+0.5165**, both independently verified. Everything else
passes or fails both bars together. The two headline figures reproduce as well, 0.017 at seven
units and 0.024 at 22, the small differences in the last two rows of the table being Monte
Carlo noise on a sampled quantile.

**The row to watch if the rule ever changes** is El Salvador's `Religiones Orientales` at
**+0.45**, one lattice step below the **+0.4593** exact bar at 14 units. It is outside the gap
today and does not move under this proposal; it is simply the nearest thing in the module to
the boundary, so it is the first row a re-run on new data would carry across.

---

## Evidence, 2026-09-09 — the first Arab Barometer country checked against this, and it adds
nothing to the blast radius

*Appended by session `967ffe99-93b1-4ff9-8169-4a6d5ffa084e-jo` while building Jordan (§9cq).
**The ask is not re-argued and its conclusion is untouched.** This is here because the ask's
blast-radius list is the five LAPOP countries, and `sources/arabbarometer.py` copied the same
`1.96/sqrt(n-1)` bar into a second module.*

Jordan's split-half runs on **12 units** and its leave-one-out on 11, neither of which is in
the table above. Run through this ask's own script, `967ffe99-cr-spearman-null.py`, at
2,000,000 samples:

```
   n   fixed bar   exact 95th   exact 97.5th   what the fixed bar actually is
  11      0.6198      0.5273        0.6091     a 0.022-level test
  12      0.5910      0.4965        0.5804     a 0.023-level test
```

So the ask's finding reproduces on a third and fourth unit count, in a different module, at the
same 0.022-0.023 level it reports for n=10 to 22.

**Jordan does not move under the proposal.** Both of its answers come back at **+0.617** on
twelve units (a two-box card, so the Muslim and Christian rankings are exact reverses and the
correlations are identical by construction). That clears the fixed bar **+0.5910**, the exact
95th **+0.4965**, and the exact 97.5th **+0.5804**. It is drawn on its own governorate shares
either way, and the pass is more comfortable on the honest null than on the applied one.

**One thing worth adding to the case, from the other side.** Jordan's pass is fragile for a
reason that has nothing to do with which bar is used: leave-one-out over its twelve
governorates runs **+0.509 to +0.717**, and dropping Balqa alone — the most Christian
governorate, and the top of the ordering — takes it to +0.509, which fails all three bars at
eleven units. So on this country the leverage question and the bar question are separable, and
only the first one changed anything. If it helps in ruling: **a correction to the bar makes
some passes safer without making the leveraged ones any safer**, which argues for treating it
as a correction to an arithmetic claim rather than as a loosening.
