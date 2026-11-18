// Agent worktrees (T-792): a git worktree on its own branch for a code agent whose edits would collide
// with others' in shared files. Run from anywhere; trees live in maps/.tw-trees/<name>, branch tw/<name>.
//   node trainworld/checks/agent-tree.mjs new <name> [commit]   from master's tip unless a commit is given
//   node trainworld/checks/agent-tree.mjs list
//   node trainworld/checks/agent-tree.mjs land <name> "<one sentence>"   orchestrator: one commit on master
//   node trainworld/checks/agent-tree.mjs remove <name> [--force]
//   node trainworld/checks/agent-tree.mjs finish <name> "<one sentence>" [--task T-1,T-2 --outcome "..." [--notes notes/T-1.md]]
//        [--keep-tree] [--dry-run] [--log <file>]   orchestrator: land + archive in one go, one line per step (T-3554)
//        [--coauthor "Name <email>"]   repeatable: a contributor credited in the commit's trailer
// `land` takes everything changed under trainworld/ in the tree (committed on the branch or not), rebases it
// onto master, squashes it into one commit and fast-forwards master to it. Master's own uncommitted edits
// stay; git stops the fast-forward if one of them is in a file the commit touches. A landing that
// changed sim/ rebuilds the main tree's WASM before it returns. `new` and `land` warn (T-2631) when the
// tree's app/src/game/*.json differ from master's committed or the main tree's working copies.
// `new` also copies the main tree's uncommitted app/src/game/*.json into the tree (the packs were built for
// them) and records which in the tree's git dir; `land` puts back, before landing, any of those the agent did not
// edit itself, so they never ride along.
import { execFileSync, spawnSync } from 'node:child_process';
import os from 'node:os';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';

const git = (cwd, ...args) => execFileSync('git', ['-C', cwd, ...args], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim();
// The main tree, also when this copy of the script sits in an agent's tree.
const here = path.resolve(fileURLToPath(new URL('../..', import.meta.url)));
const maps = path.dirname(path.resolve(here, git(here, 'rev-parse', '--git-common-dir')));
const trees = path.join(maps, '.tw-trees');
// Untracked in git, so a new tree gets them as junctions to the main tree's.
const LINKS = ['trainworld/data', 'trainworld/app/node_modules', 'artifacts/trainworld', 'venv'];

// `finish` sets LOG: everything the steps print (git warnings, wasm build) goes to that file, and a failure throws.
let LOG = null;
let warned = 0;
const say = (m) => (LOG ? fs.appendFileSync(LOG, `${m}
`) : console.log(m));
const warn = (m) => { if (LOG) { warned++; fs.appendFileSync(LOG, `WARNING ${m}
`); } else console.warn(m); };
// Runs a command; with LOG its output is appended there and a non-zero exit throws with the stderr in full.
function run(cmd, args, opts = {}) {
  if (!LOG) { execFileSync(cmd, args, { stdio: 'inherit', ...opts }); return ''; }
  const r = spawnSync(cmd, args, { encoding: 'utf8', maxBuffer: 1 << 30, ...opts });
  fs.appendFileSync(LOG, `$ ${cmd} ${args.join(' ')}
${r.stdout ?? ''}${r.stderr ?? ''}`);
  if (r.error) throw r.error;
  if (r.status !== 0) throw new Error(`${cmd} ${args.join(' ')} failed (exit ${r.status}):
${(r.stderr || r.stdout || '').trim()}`);
  return r.stdout ?? '';
}
const gitLoud = (cwd, ...args) => run('git', ['-C', cwd, ...args]);
const fail = (msg) => { if (LOG) throw new Error(msg); console.error(msg); process.exit(1); };
const isLink = (p) => { try { return fs.lstatSync(p).isSymbolicLink(); } catch { return false; } };

function treeOf(name) {
  if (!/^[a-z0-9][a-z0-9-]{0,30}$/.test(name ?? '')) fail(`a name is lowercase letters, digits and dashes: "${name ?? ''}"`);
  return { dir: path.join(trees, name), branch: `tw/${name}` };
}

function excludeTrees() {
  const file = path.join(maps, '.git', 'info', 'exclude');
  const text = fs.existsSync(file) ? fs.readFileSync(file, 'utf8') : '';
  if (!/^\.tw-trees\/$/m.test(text)) fs.appendFileSync(file, (text.endsWith('\n') || !text ? '' : '\n') + '.tw-trees/\n');
}

function unlinkJunctions(dir) {
  for (const rel of LINKS) {
    const p = path.join(dir, rel);
    if (isLink(p)) fs.unlinkSync(p);
  }
}

// Changed paths in the tree against master, split into trainworld/ and the rest.
function changes(dir) {
  const base = git(dir, 'merge-base', 'master', 'HEAD');
  const files = new Set([
    ...git(dir, 'diff', '--name-only', base).split('\n'),
    ...git(dir, 'ls-files', '--others', '--exclude-standard').split('\n'),
  ].filter(Boolean));
  const tw = [...files].filter((f) => f.startsWith('trainworld/'));
  return { tw, other: [...files].filter((f) => !f.startsWith('trainworld/')) };
}

// T-2631: the tree's data/ is the main tree's, and it is built from master's app/src/game/*.json (registry,
// market rows, realms ...); a tree whose copies differ runs with packs its json does not match, and a refused
// market pack is only a console warning. Warns, never stops: the tree may have changed one on purpose.
const GAME_JSON = 'trainworld/app/src/game';
function warnGameJson(dir, when, copied = []) {
  const rels = git(maps, 'ls-tree', '--name-only', 'master', `${GAME_JSON}/`).split('\n').filter((f) => f.endsWith('.json'));
  const behind = [];
  for (const rel of rels) {
    const mine = path.join(dir, rel);
    if (!fs.existsSync(mine)) { behind.push(`${rel} (missing in the tree)`); continue; }
    // git show gives LF, a checkout may be CRLF
    const lf = (b) => Buffer.from(b.toString('latin1').replace(/\r\n/g, '\n'), 'latin1');
    const have = lf(fs.readFileSync(mine));
    let committed = null;
    try { committed = execFileSync('git', ['-C', maps, 'show', `master:${rel}`], { maxBuffer: 1 << 30, stdio: ['ignore', 'pipe', 'ignore'] }); } catch { /* not committed */ }
    const main = path.join(maps, rel);
    const live = fs.existsSync(main) ? lf(fs.readFileSync(main)) : null;
    const notes = [];
    if (committed && !copied.includes(rel) && !have.equals(lf(committed))) notes.push("differs from master's committed copy");
    if (live && !have.equals(live)) notes.push("differs from the main tree's working copy");
    if (notes.length) behind.push(`${path.basename(rel)} ${notes.join(' and ')}`);
  }
  if (!behind.length) return;
  warn(`WARNING: ${when}, app/src/game json in ${path.basename(dir)} is out of step with master:\n  ${behind.join('\n  ')}\n` +
    `The tree shares master's data/packs, so a page here may refuse a market pack ("demand: market pack X would not open") and run with no regional market; smoke fails on that.\n` +
    `Unless you changed these on purpose, copy the main tree's version of each (the packs were built for it).`);
}

// The main tree's uncommitted game json, copied into a new tree. The record lives in the tree's git dir (never
// committed): the files copied and a hash of each copy, so land can tell a copy from the agent's own edit.
const copiedFile = (dir) => path.join(git(dir, 'rev-parse', '--absolute-git-dir'), 'tw-copied-game-json.json');
const sha = (p) => createHash('sha256').update(fs.readFileSync(p)).digest('hex');
function copyGameJson(dir) {
  const dirty = git(maps, 'status', '--porcelain', '--', GAME_JSON).split('\n').filter(Boolean)
    .map((l) => l.replace(/^\s*[A-Z?]+\s+/, '')).filter((f) => f.endsWith('.json') && fs.existsSync(path.join(maps, f)));
  const record = {};
  for (const rel of dirty) {
    fs.mkdirSync(path.dirname(path.join(dir, rel)), { recursive: true });
    fs.copyFileSync(path.join(maps, rel), path.join(dir, rel));
    record[rel] = sha(path.join(dir, rel));
  }
  fs.writeFileSync(copiedFile(dir), JSON.stringify(record, null, 2));
  if (dirty.length) console.log(`copied the main tree's uncommitted game json into the tree (land puts them back):\n  ${dirty.join('\n  ')}`);
  return dirty;
}
function restoreGameJson(dir) {
  let record = {};
  try { record = JSON.parse(fs.readFileSync(copiedFile(dir), 'utf8')); } catch { return; }
  for (const [rel, hash] of Object.entries(record)) {
    const mine = path.join(dir, rel);
    if (!fs.existsSync(mine)) continue;
    if (sha(mine) === hash) { gitLoud(dir, 'checkout', 'HEAD', '--', rel); say(`put back ${rel} (a copy from the main tree, not an edit)`); }
    else warn(`WARNING: ${rel} was edited in the tree after the copy, so it lands with the work`);
  }
  fs.rmSync(copiedFile(dir), { force: true });
}

// Puts the tree's work on master as one commit; returns the trainworld/ files it changed.
function landTree(dir, branch, message, coauthors = []) {
  if (!message) fail('land needs a commit message');
  if (!fs.existsSync(dir)) fail(`no tree ${dir}`);
  restoreGameJson(dir);
  const { tw, other } = changes(dir);
  if (other.length) fail(`changes outside trainworld/, land them by hand:\n  ${other.join('\n  ')}`);
  if (!tw.length) fail('nothing to land');
  warnGameJson(dir, 'land');
  // The links are untracked junctions; `add trainworld` must not see them as content.
  gitLoud(dir, 'add', '--all', '--', 'trainworld');
  if (git(dir, 'diff', '--cached', '--name-only')) gitLoud(dir, 'commit', '-q', '-m', 'agent-tree: work in progress');
  try { gitLoud(dir, 'rebase', '-q', 'master'); } catch (e) {
    fail(`the rebase onto master stopped on a conflict in ${dir}: resolve there and \`git rebase --continue\`, or \`git rebase --abort\`; then land again`);
  }
  gitLoud(dir, 'reset', '-q', '--soft', 'master');
  // one trailer paragraph, so GitHub credits a contributor (--coauthor) beside the agent
  const trailers = [...coauthors.map((c) => `Co-Authored-By: ${c}`), 'Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>'];
  gitLoud(dir, 'commit', '-q', '-m', message, '-m', trailers.join('\n'));
  try { gitLoud(maps, 'merge', '-q', '--ff-only', branch); } catch (e) {
    fail(`master did not fast-forward to ${branch} (an uncommitted edit in master's tree touches the same file, or master moved): the commit waits on the branch`);
  }
  say(`master is now ${git(maps, 'log', '-1', '--format=%h %s')}`);
  return tw;
}

// Agents' pages in the main tree load its WASM bindings, which a landed Rust change leaves stale
// (three agents hit "set_part is not a function" after T-1543, 2026-10-27).
function rebuildWasm() {
  say("Rust changed: rebuilding the main tree's WASM (npm run wasm)");
  const cargoBin = path.join(process.env.USERPROFILE ?? '', '.cargo', 'bin');
  run('npm', ['run', 'wasm', '--prefix', path.join(maps, 'trainworld', 'app')], {
    shell: true,
    env: { ...process.env, CARGO_BUILD_JOBS: '3', PATH: `${cargoBin}${path.delimiter}${process.env.PATH}` },
  });
}

// T-3554: land + archive in one command. Steps: 1 changelog items move to the main tree, 2 land, 3 WASM when sim/
// changed, 4 remove the tree, 5 tasks.py done per task. Stops at the first failure and says what state things are in.
const CHANGELOG = 'trainworld/changelog/next.json';
const pythonExe = () => {
  if (process.env.TW_PYTHON) return process.env.TW_PYTHON;
  const v = path.join(maps, 'venv', 'Scripts', 'python.exe');
  return fs.existsSync(v) ? v : 'python';
};
const readJsonText = (text) => { try { return JSON.parse(text); } catch (e) { throw new Error(`${CHANGELOG} is not valid JSON: ${e.message}`); } };

// New items in the tree's next.json (ids the tree's base does not have) and whether anything else in it differs.
function treeChangelog(dir) {
  const file = path.join(dir, CHANGELOG);
  if (!fs.existsSync(file)) return { exists: false, items: [], edited: 0 };
  const base = git(dir, 'merge-base', 'master', 'HEAD');
  let baseText = null;
  try { baseText = execFileSync('git', ['-C', dir, 'show', `${base}:${CHANGELOG}`], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }); } catch { /* not in base */ }
  const mine = fs.readFileSync(file, 'utf8');
  if (baseText !== null && mine.replace(/\r\n/g, '\n') === baseText.replace(/\r\n/g, '\n')) return { exists: true, base, baseText, items: [], edited: 0, differs: false };
  const baseItems = new Map((baseText ? readJsonText(baseText).items ?? [] : []).map((i) => [i.id, JSON.stringify(i)]));
  const all = readJsonText(mine).items ?? [];
  return {
    exists: true, base, baseText, differs: true,
    items: all.filter((i) => !baseItems.has(i.id)),
    edited: all.filter((i) => baseItems.has(i.id) && baseItems.get(i.id) !== JSON.stringify(i)).length,
  };
}

function finishTree(args) {
  const o = { keep: false, dry: false, coauthor: [] };
  const words = [];
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--keep-tree') o.keep = true;
    else if (a === '--dry-run') o.dry = true;
    else if (a === '--coauthor') {
      if (i + 1 >= args.length) { console.error('--coauthor needs "Name <email>"'); process.exit(2); }
      o.coauthor.push(args[++i]);
    }
    else if (['--task', '--outcome', '--notes', '--log'].includes(a)) {
      if (i + 1 >= args.length) { console.error(`${a} needs a value`); process.exit(2); }
      o[a.slice(2)] = args[++i];
    } else if (a.startsWith('--')) { console.error(`unknown option ${a}`); process.exit(2); } else words.push(a);
  }
  const message = words.join(' ').trim();
  const tasks = o.task ? String(o.task).split(',').map((t) => t.trim()).filter(Boolean) : [];
  if (!message) { console.error('finish needs a commit sentence'); process.exit(2); }
  if (tasks.length && !o.outcome) { console.error('--task needs --outcome "..."'); process.exit(2); }
  const { dir, branch } = treeOf(name);
  if (!fs.existsSync(dir)) { console.error(`no tree ${dir}`); process.exit(1); }
  const logDir = fs.existsSync(path.join(maps, 'artifacts', 'trainworld')) ? path.join(maps, 'artifacts', 'trainworld') : os.tmpdir();
  const stamp = new Date().toISOString().replace(/[-:]/g, '').slice(0, 15);
  LOG = o.log ? path.resolve(o.log) : path.join(logDir, `finish-${name}-${stamp}.log`);
  if (!o.dry) fs.writeFileSync(LOG, `finish ${name} ${new Date().toISOString()}\n`);
  const tag = o.dry ? '(dry run) ' : '';
  const line = (n, text) => console.log(`[${n}/5] ${tag}${text}`);
  const state = { moved: [], landed: false, wasm: false, removed: false, doneTasks: [], simChanged: false };
  let step = 0;
  const later = () => {
    const todo = [];
    if (!state.landed) return todo;
    if (!state.wasm && state.simChanged) todo.push('npm run wasm --prefix trainworld/app');
    if (!state.removed && !o.keep) todo.push(`node trainworld/checks/agent-tree.mjs remove ${name}`);
    for (const t of tasks.filter((t) => !state.doneTasks.includes(t))) todo.push(`venv\\Scripts\\python.exe trainworld/tools/tasks.py done ${t} --outcome "..."`);
    return todo;
  };
  try {
    // 1: changelog items an agent put in its own tree
    step = 1;
    const cl = treeChangelog(dir);
    if (!cl.exists || !cl.differs) line(1, 'changelog: tree untouched');
    else {
      const mainFile = path.join(maps, CHANGELOG);
      const have = fs.existsSync(mainFile) ? readJsonText(fs.readFileSync(mainFile, 'utf8')).items ?? [] : [];
      const fresh = cl.items.filter((i) => !have.some((h) => h.text === i.text && (h.task ?? null) === (i.task ?? null)));
      const dup = cl.items.length - fresh.length;
      if (!o.dry) {
        // back to the base content, committed or not
        if (cl.baseText === null) { fs.rmSync(path.join(dir, CHANGELOG), { force: true }); try { git(dir, 'rm', '-q', '--cached', '--', CHANGELOG); } catch { /* untracked */ } }
        else gitLoud(dir, 'checkout', cl.base, '--', CHANGELOG);
        for (const it of fresh) {
          const a2 = [path.join(maps, 'trainworld', 'tools', 'changelog.py'), 'add', it.text, '--area', it.area ?? 'other'];
          if (it.fix) a2.push('--fix');
          if (it.task) a2.push('--task', it.task);
          if (it.by) a2.push('--by', it.by);
          const out = run(pythonExe(), a2, { cwd: maps });
          it.finalId = (/added (c\d+)/.exec(out) ?? [])[1];   // the id the main tree gave it: the one to quote
          state.moved.push(it);
        }
      }
      line(1, `changelog: reverted the tree's next.json, ${fresh.length} item(s) ${o.dry ? 'to move' : 'moved'} to the main tree${dup ? `, ${dup} already there` : ''}${cl.edited ? `, ${cl.edited} edit(s) to existing items discarded` : ''}`);
      for (const it of fresh) console.log(`      ${it.finalId ? `${it.finalId} ` : ''}${it.fix ? 'fix ' : ''}${it.area ?? 'other'} ${it.task ?? '-'} ${it.by ?? '-'}: ${it.text}`);
    }
    // 2: land
    step = 2;
    if (o.dry) {
      const ch = changes(dir);
      const tw = ch.tw.filter((f) => !(f === CHANGELOG && cl.exists && cl.differs));
      if (ch.other.length) throw new Error(`changes outside trainworld/, land them by hand:\n  ${ch.other.join('\n  ')}`);
      if (!tw.length) throw new Error('nothing to land');
      state.simChanged = tw.some((f) => f.startsWith('trainworld/sim/'));
      line(2, `land ${branch}: ${tw.length} trainworld file(s), ${git(dir, 'rev-list', '--count', 'master..HEAD')} branch commit(s) plus any uncommitted work, squashed into one commit "${message}"`);
    } else {
      const tw = landTree(dir, branch, message, o.coauthor);
      state.landed = true;
      state.simChanged = tw.some((f) => f.startsWith('trainworld/sim/'));
      line(2, `landed ${branch}: ${git(maps, 'log', '-1', '--format=%h %s')} (${tw.length} trainworld file(s))`);
    }
    // 3: wasm
    step = 3;
    if (state.simChanged) {
      if (!o.dry) { rebuildWasm(); state.wasm = true; }
      line(3, `wasm: sim/ changed, main tree's WASM ${o.dry ? 'would be rebuilt' : 'rebuilt'}`);
    } else line(3, 'wasm: sim/ unchanged, no rebuild');
    // 4: remove the tree
    step = 4;
    if (o.keep) line(4, `tree kept: ${dir}`);
    else {
      if (!o.dry) {
        unlinkJunctions(dir);
        gitLoud(maps, 'worktree', 'remove', '--force', dir);
        try { git(maps, 'branch', '-D', branch); } catch { /* already gone */ }
        state.removed = true;
      }
      line(4, `tree ${o.dry ? 'would be removed' : 'removed'}: ${name}`);
    }
    // 5: tasks
    step = 5;
    if (!tasks.length) line(5, 'tasks: none given');
    else {
      const archived = fs.existsSync(path.join(maps, 'trainworld', 'ARCHIVE.md')) ? fs.readFileSync(path.join(maps, 'trainworld', 'ARCHIVE.md'), 'utf8') : '';
      const already = [];
      for (const t of tasks) {
        if (o.dry) continue;
        // the agent archived its own task: that counts as done
        if (new RegExp(`^- ${t} \\[`, 'm').test(archived)) { already.push(t); state.doneTasks.push(t); continue; }
        const a2 = [path.join(maps, 'trainworld', 'tools', 'tasks.py'), 'done', t, '--outcome', o.outcome];
        if (o.notes) a2.push('--notes', o.notes);
        run(pythonExe(), a2, { cwd: maps });
        state.doneTasks.push(t);
      }
      line(5, `tasks: ${tasks.join(', ')} ${o.dry ? 'would be archived' : 'archived'}${already.length ? ` (${already.join(', ')} already archived)` : ''}`);
    }
    if (!o.dry) console.log(`log: ${LOG}${warned ? ` (${warned} warning(s) in it)` : ''}`);
    else console.log('dry run: nothing changed');
  } catch (e) {
    const names = ['', 'changelog', 'land', 'wasm', 'remove tree', 'tasks'];
    console.error(`FAILED at step ${step}/5 (${names[step]}):\n${e.message}`);
    console.error(`state: ${state.moved.length ? `${state.moved.length} changelog item(s) already moved to the main tree and reverted in the tree (do not move again). ` : ''}${state.landed ? `master has the landing commit (${git(maps, 'log', '-1', '--format=%h', 'master')}). ` : 'master is unchanged by this run. '}${state.removed ? 'tree removed. ' : 'tree still there. '}${state.doneTasks.length ? `tasks archived: ${state.doneTasks.join(', ')}. ` : ''}`);
    const left = later();
    if (left.length) console.error(`still to do by hand:\n  ${left.join('\n  ')}`);
    if (!o.dry) console.error(`log: ${LOG}`);
    process.exit(1);
  }
}

const [cmd, name, ...rest] = process.argv.slice(2);

if (cmd === 'new') {
  const { dir, branch } = treeOf(name);
  if (fs.existsSync(dir)) fail(`${dir} exists already`);
  const from = rest[0] ?? 'master';
  excludeTrees();
  fs.mkdirSync(trees, { recursive: true });
  gitLoud(maps, 'worktree', 'add', '-q', '-b', branch, dir, from);
  for (const rel of LINKS) {
    const target = path.join(maps, rel);
    if (!fs.existsSync(target)) continue;
    fs.mkdirSync(path.dirname(path.join(dir, rel)), { recursive: true });
    fs.symlinkSync(target, path.join(dir, rel), 'junction');
  }
  // Generated WASM bindings, so the app runs before the first `npm run wasm`; a copy, since they differ by branch.
  const wasm = path.join(maps, 'trainworld', 'app', 'src', 'wasm');
  if (fs.existsSync(wasm)) fs.cpSync(wasm, path.join(dir, 'trainworld', 'app', 'src', 'wasm'), { recursive: true });
  console.log(`\n${dir}\nbranch ${branch} from ${git(dir, 'rev-parse', '--short', 'HEAD')}`);
  console.log(`Rust builds into the tree's own trainworld/sim/target (first wasm build a few minutes).`);
  const copied = copyGameJson(dir);
  warnGameJson(dir, 'new', copied);
} else if (cmd === 'list') {
  if (!fs.existsSync(trees)) { console.log('no trees'); process.exit(0); }
  for (const n of fs.readdirSync(trees)) {
    const dir = path.join(trees, n);
    if (!fs.existsSync(path.join(dir, '.git'))) continue;
    const { tw, other } = changes(dir);
    const ahead = git(dir, 'rev-list', '--count', 'master..HEAD');
    const behind = git(dir, 'rev-list', '--count', 'HEAD..master');
    console.log(`${n.padEnd(20)} ${git(dir, 'rev-parse', '--short', 'HEAD')}  ${ahead} ahead, ${behind} behind master, ${tw.length} trainworld files changed${other.length ? `, ${other.length} outside` : ''}`);
  }
} else if (cmd === 'land') {
  const { dir, branch } = treeOf(name);
  const tw = landTree(dir, branch, rest.join(' ').trim());
  if (tw.some((f) => f.startsWith('trainworld/sim/'))) rebuildWasm();
} else if (cmd === 'finish') {
  finishTree(rest);
} else if (cmd === 'remove') {
  const { dir, branch } = treeOf(name);
  const force = rest.includes('--force');
  if (fs.existsSync(dir)) {
    const { tw, other } = changes(dir);
    if (!force && (tw.length || other.length)) fail(`${dir} has ${tw.length + other.length} changed files not on master; land them or pass --force`);
    unlinkJunctions(dir);
    gitLoud(maps, 'worktree', 'remove', '--force', dir);
  }
  try { git(maps, 'branch', '-D', branch); } catch { /* already gone */ }
  console.log(`removed ${name}`);
} else {
  console.log(fs.readFileSync(fileURLToPath(import.meta.url), 'utf8').split('\n').slice(0, 12).join('\n'));
  process.exit(cmd ? 2 : 0);
}
