// T-4213: the measuring budget (trainworld/CLAUDE.md "Testing and measuring budget") as code, shared by perf_bench and
// edit_bench. Inside .tw-trees/<name>/ a tree gets 2 benchmark invocations in all, 1 run each, no heavy saves without
// a reason, no base build to compare against (master's stored numbers are diffed automatically). On the main tree
// (the orchestrator) nothing is refused. It also reads the core ledger to stamp a run NOISY.
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = fileURLToPath(new URL('.', import.meta.url));
export const TREE = resolve(HERE, '../..');
export const MAPS = TREE.replace(/[\\/]\.tw-trees[\\/][^\\/]+$/, '');
export const TREE_NAME = TREE === MAPS ? null : basename(TREE);
export const MAX_RUNS = 2;
export const NOISY_OTHERS = 4;
const BUDGET = join(TREE, '.bench-runs.json');
const HEAVY = /tokyo|belgrade|asia/i;

const budgetLine = 'a tree gets ' + MAX_RUNS + ' benchmark runs in all, 1 run each, no Tokyo, Belgrade or Asia save without --heavy "<reason>", no base build';

/** one plain line, non-zero exit */
export function refuse(why) {
  console.error('REFUSED in tree ' + TREE_NAME + ': ' + why + ' (' + budgetLine + '). Report what you have and hand back.');
  process.exit(2);
}

const readBudget = () => { try { return JSON.parse(readFileSync(BUDGET, 'utf8')); } catch { return { runs: [] }; } };

/** In a tree: refuses what the budget forbids, then (unless dryRun) counts this invocation. Elsewhere: nothing.
 * saves: names or paths; runs: --runs; compare: --compare given; heavy: --heavy value (string) or false. */
export function gateRun({ tool, saves, runs, compare, heavy, dryRun }) {
  if (!TREE_NAME) return;
  if (runs > 1) refuse('--runs ' + runs + ' is more than 1 run per call');
  if (compare) refuse('no compare against a base build or an older run here: the run is diffed against master\'s stored numbers by itself');
  const heavySaves = saves.filter((s) => HEAVY.test(String(s)));
  if (heavySaves.length && (typeof heavy !== 'string' || !heavy.trim())) refuse(heavySaves.join(', ') + ' is a heavy save and needs --heavy "<reason>"');
  const b = readBudget();
  if (b.runs.length >= MAX_RUNS) refuse(b.runs.length + ' invocations already used (' + b.runs.map((r) => r.tool + ' ' + r.saves.join(',')).join('; ') + ')');
  if (dryRun) {
    console.log('dry run: allowed, would be invocation ' + (b.runs.length + 1) + ' of ' + MAX_RUNS + ' in ' + TREE_NAME + (heavySaves.length ? ', heavy: ' + heavy : ''));
    process.exit(0);
  }
  b.runs.push({ at: new Date().toISOString().slice(0, 19), tool, saves, ...(heavySaves.length ? { heavy } : {}) });
  writeFileSync(BUDGET, JSON.stringify(b, null, 1) + '\n');
  console.log('bench budget: invocation ' + b.runs.length + ' of ' + MAX_RUNS + ' in ' + TREE_NAME + (heavySaves.length ? ' (heavy: ' + heavy + ')' : ''));
}

/** cores claimed by anyone, from the ledger (cores.py list's last line); null when unreadable */
export function claimedNow() {
  const r = spawnSync('python', [join(MAPS, 'cores.py'), 'list'], { encoding: 'utf8', windowsHide: true, cwd: MAPS });
  const m = /claimed (\d+) of cap (\d+)/.exec(r.stdout ?? '');
  return m ? Number(m[1]) : null;
}

/** Cores claimed beyond the run's own, tracked at the start and the end: sample() at each, then stamp() */
export function noiseWatch(own) {
  const seen = [];
  return {
    sample() { const c = claimedNow(); if (c != null) seen.push(Math.max(0, c - own)); },
    /** the most others seen when it was over the limit, else null */
    stamp() { const m = Math.max(0, ...seen); return m > NOISY_OTHERS ? m : null; },
  };
}
