// Agent worktrees (T-792): a git worktree on its own branch for a code agent whose edits would collide
// with others' in shared files. Run from anywhere; trees live in maps/.tw-trees/<name>, branch tw/<name>.
//   node trainworld/checks/agent-tree.mjs new <name> [commit]   from master's tip unless a commit is given
//   node trainworld/checks/agent-tree.mjs list
//   node trainworld/checks/agent-tree.mjs land <name> "<one sentence>" [--into tw/track] [--dry-run]   orchestrator: one commit on master
//   node trainworld/checks/agent-tree.mjs remove <name> [--force]
//   node trainworld/checks/agent-tree.mjs finish <name> "<one sentence>" [--task T-1,T-2 --outcome "..." [--notes notes/T-1.md]]
//        [--keep-tree] [--dry-run] [--log <file>] [--into tw/track]   orchestrator: land + archive in one go, one line per step (T-3554)
//        [--coauthor "Name <email>"]   repeatable: a contributor credited in the commit's trailer
//   node trainworld/checks/agent-tree.mjs sync-track [tw/track] [--dry-run]   merge master into tw/track (T-4016)
// `--into tw/<x>` (default master; `finish` and `land`) lands on that branch instead: rebased onto it, squashed into one
// commit, fast-forwarded in the branch's own worktree maps/.tw-trees/_<x> (made on first use, with the same junctions),
// so the main tree's checkout is never touched. The WASM rebuild runs in that worktree. Changelog items of the tree go to
// trainworld/changelog/next-track.json on the branch, for the release merge. `tasks.py done` still runs in the main tree.
// `sync-track` merges master into the branch there, with the same conflict rule as land; a conflict elsewhere leaves the
// merge open in the worktree and says so.
// A rebase conflict in DECISIONS.md, ARCHIVE.md, notes/*.md or research/*.md keeps both sides (master's first); any other
// conflict stops. If master will not fast-forward over the main tree's uncommitted edits, the message names the files
// and the command. `remove` and `finish` name the processes that may hold a tree (never kill them), refuse when this
// shell's cwd is inside it, and clean up an empty folder a failed removal left (T-3665).
// T-4180: `land` and `finish` stop (before changing anything) when the tree has a merge, rebase, cherry-pick or revert in
// progress. Merge commits already on the branch (master merged into the tree) are handled: the branch is squashed to its net
// diff against the merge base first, because the rebase would drop them and their work.
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

// T-4121: a new tree starts from a warm copy of an earlier tree's trainworld/sim/target (agent profile, native and wasm32),
// so its first test and wasm build recompile only the sim crate, incrementally. A copy, never a link or a shared
// CARGO_TARGET_DIR: trees on different branches overwrite each other's artifacts, and cargo itself refuses to share
// one target between parallel builds. Cargo fingerprints source mtimes, so stale artifacts only cost a rebuild.
// The seed is the target of the last tree removed that had built both (renamed, so it costs no copy and no disk).
const seedDir = path.join(trees, '_seed', 'target');
const stamp = path.join(trees, '_seed', 'stamp');
const SEED_MAX_AGE_H = 12;
function seedTarget(dir) {
  if (!fs.existsSync(seedDir) || process.env.TW_NO_SEED) return false;
  const t0 = Date.now();
  const r = spawnSync('robocopy', [seedDir, path.join(dir, 'trainworld', 'sim', 'target'), '/E', '/MT:8', '/NFL', '/NDL', '/NJH', '/NJS', '/NP', '/NC', '/NS'], { stdio: 'ignore', windowsHide: true });
  if (r.status === null || r.status >= 8) { warn('the warm Rust cache could not be copied; the first build starts from nothing'); return false; }
  say(`warm Rust cache copied in ${Math.round((Date.now() - t0) / 1000)} s`);
  return true;
}
function keepSeed(dir) {
  const tgt = path.join(dir, 'trainworld', 'sim', 'target');
  const built = ['agent', path.join('wasm32-unknown-unknown', 'agent')].every((d) => fs.existsSync(path.join(tgt, d)));
  if (!built) return;
  try {
    if (fs.existsSync(seedDir) && fs.existsSync(stamp) && Date.now() - fs.statSync(stamp).mtimeMs < SEED_MAX_AGE_H * 3600e3) return;
    fs.rmSync(seedDir, { recursive: true, force: true });
    fs.mkdirSync(path.dirname(seedDir), { recursive: true });
    fs.renameSync(tgt, seedDir);
    fs.writeFileSync(stamp, dir);
    say(`kept ${path.basename(dir)}'s Rust target as the warm cache for new trees`);
  } catch (e) { warn(`could not keep the Rust cache: ${e.message.split('\n')[0]}`); }
}

function treeOf(name) {
  if (!/^[a-z0-9][a-z0-9-]{0,30}$/.test(name ?? '')) fail(`a name is lowercase letters, digits and dashes: "${name ?? ''}"`);
  return { dir: path.join(trees, name), branch: `tw/${name}` };
}

function excludeTrees() {
  const file = path.join(maps, '.git', 'info', 'exclude');
  const text = fs.existsSync(file) ? fs.readFileSync(file, 'utf8') : '';
  if (!/^\.tw-trees\/$/m.test(text)) fs.appendFileSync(file, (text.endsWith('\n') || !text ? '' : '\n') + '.tw-trees/\n');
}

// Every junction in the tree, not only LINKS: `worktree remove --force` deletes through any it meets, and an agent's
// own scratch junction to app/node_modules emptied the main tree's on 2026-11-18.
function unlinkJunctions(dir) {
  for (const rel of LINKS) {
    const p = path.join(dir, rel);
    if (isLink(p)) fs.unlinkSync(p);
  }
  const stack = [dir];
  while (stack.length) {
    const d = stack.pop();
    let ents;
    try { ents = fs.readdirSync(d, { withFileTypes: true }); } catch { continue; }
    for (const e of ents) {
      if (e.name === '.git') continue;
      const p = path.join(d, e.name);
      const link = e.isSymbolicLink() || (e.isDirectory() && isLink(p));
      if (link) { try { fs.unlinkSync(p); } catch { fs.rmdirSync(p); } }
      else if (e.isDirectory()) stack.push(p);
    }
  }
}

// Changed paths in the tree against master, split into trainworld/ and the rest.
function changes(dir, into = 'master') {
  const base = git(dir, 'merge-base', into, 'HEAD');
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
function warnGameJson(dir, when, copied = [], ref = 'master') {
  const rels = git(maps, 'ls-tree', '--name-only', ref, `${GAME_JSON}/`).split('\n').filter((f) => f.endsWith('.json'));
  const behind = [];
  for (const rel of rels) {
    const mine = path.join(dir, rel);
    if (!fs.existsSync(mine)) { behind.push(`${rel} (missing in the tree)`); continue; }
    // git show gives LF, a checkout may be CRLF
    const lf = (b) => Buffer.from(b.toString('latin1').replace(/\r\n/g, '\n'), 'latin1');
    const have = lf(fs.readFileSync(mine));
    let committed = null;
    try { committed = execFileSync('git', ['-C', maps, 'show', `${ref}:${rel}`], { maxBuffer: 1 << 30, stdio: ['ignore', 'pipe', 'ignore'] }); } catch { /* not committed */ }
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

// T-3665: a rebase conflict in a doc that only ever grows (both sides added lines, or both added the same note) keeps
// both texts, master's first; any other conflict stops as before. If one side already holds the other, that side is kept.
const APPEND_ONLY = /^trainworld\/(DECISIONS\.md|ARCHIVE\.md|notes\/.+\.md|research\/.+\.md)$/;
const CONFLICT = /^<<<<<<< [^\n]*\n([\s\S]*?)(?:^\|\|\|\|\|\|\| [^\n]*\n[\s\S]*?)?^=======\r?\n([\s\S]*?)^>>>>>>> [^\n]*(?:\n|$)/gm;
function keepBoth(text) {
  return text.replace(CONFLICT, (_, a, b) => {
    const has = (big, small) => big.replace(/\r/g, '').includes(small.replace(/\r/g, '').trim());
    if (has(b, a)) return b;
    if (has(a, b)) return a;
    return a + (a.endsWith('\n') || !a ? '' : '\n') + b;
  });
}
function rebaseOnto(dir, onto = 'master') {
  let last;
  try { gitLoud(dir, 'rebase', '-q', onto); return; } catch (e) { last = e; }
  for (let round = 0; round < 30; round++) {
    const unmerged = git(dir, 'status', '--porcelain').split('\n').filter((l) => /^(AA|UU|DD|AU|UA|DU|UD) /.test(l));
    if (!unmerged.length) fail(`the rebase onto ${onto} failed in ${dir} without a conflict: ${last.message}`);
    const bad = unmerged.filter((l) => !/^(AA|UU) /.test(l) || !APPEND_ONLY.test(l.slice(3).trim()));
    if (bad.length) {
      fail(`the rebase onto ${onto} stopped on a conflict in ${dir}: ${bad.map((l) => l.slice(3).trim()).join(', ')}. Resolve there and \`git rebase --continue\`, or \`git rebase --abort\`; then land again ` +
        `(only a conflict in DECISIONS.md, ARCHIVE.md, notes/*.md or research/*.md is merged by keeping both sides)`);
    }
    for (const l of unmerged) {
      const rel = l.slice(3).trim();
      const file = path.join(dir, rel);
      const out = keepBoth(fs.readFileSync(file, 'utf8'));
      if (/^(<<<<<<<|=======|>>>>>>>)/m.test(out)) fail(`could not merge the conflict markers in ${rel} (${dir}); resolve there by hand`);
      fs.writeFileSync(file, out);
      gitLoud(dir, 'add', '--', rel);
      say(`rebase conflict in ${rel}: kept both sides, ${onto}'s first`);
    }
    try { run('git', ['-C', dir, '-c', 'core.editor=true', 'rebase', '--continue']); return; } catch (e) { last = e; }
  }
  fail(`the rebase onto ${onto} in ${dir} kept conflicting after 30 rounds; resolve there by hand`);
}

// T-3665: what could be holding a tree open. A shell's cwd is not readable from outside, but this process inherits the
// shell's cwd, which is the usual culprit (2026-11-18: an orchestrator shell inside the tree, "Permission denied").
function cwdInside(dir) {
  const norm = (p) => path.resolve(p).toLowerCase().replace(/\//g, '\\');
  const here = norm(process.cwd()), mine = norm(dir);
  return here === mine || here.startsWith(mine + '\\') ? `this shell's current directory is ${process.cwd()} (cd out of the tree first)` : null;
}
function holders(dir) {
  const norm = (p) => path.resolve(p).toLowerCase().replace(/\//g, '\\');
  const mine = norm(dir);
  const found = [];
  const c = cwdInside(dir);
  if (c) found.push(c);
  if (process.platform === 'win32') {
    const r = spawnSync('powershell', ['-NoProfile', '-Command',
      'Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Compress'], { encoding: 'utf8', maxBuffer: 1 << 28 });
    let list = [];
    try { const j = JSON.parse(r.stdout); list = Array.isArray(j) ? j : [j]; } catch { /* no listing */ }
    if (!list.length) found.push('(could not list processes)');
    for (const p of list) {
      if (p.ProcessId === process.pid || p.ProcessId === process.ppid || !p.CommandLine) continue;
      if (/Get-CimInstance Win32_Process/.test(p.CommandLine)) continue;
      if (norm(p.CommandLine.replace(/\//g, '\\')).includes(mine)) found.push(`pid ${p.ProcessId} ${p.Name}: ${p.CommandLine.slice(0, 160)}`);
    }
    found.push('(a shell sitting in the tree has no command line to find: look at your open shells too)');
  }
  return found;
}

const relink = (dir) => {
  for (const rel of LINKS) {
    const target = path.join(maps, rel), p = path.join(dir, rel);
    if (!fs.existsSync(target) || fs.existsSync(p)) continue;
    fs.mkdirSync(path.dirname(p), { recursive: true });
    fs.symlinkSync(target, p, 'junction');
  }
};

// Removes a tree and its branch. Never kills anything; names what may hold it when git cannot remove it.
function removeTree(dir, branch) {
  const cwd = cwdInside(dir);
  if (cwd) fail(`${dir} was not removed: ${cwd}`);
  if (fs.existsSync(dir) && !fs.existsSync(path.join(dir, '.git'))) {
    // left by an earlier failed removal
    try { fs.rmdirSync(dir); say(`cleaned up the leftover folder ${dir}`); } catch {
      fail(`${dir} is not a worktree and not empty (${fs.readdirSync(dir).slice(0, 5).join(', ')}); delete it by hand. Possibly held by:\n  ${holders(dir).join('\n  ')}`);
    }
    gitLoud(maps, 'worktree', 'prune');
  }
  if (fs.existsSync(path.join(dir, '.git'))) keepSeed(dir);
  if (fs.existsSync(dir)) {
    unlinkJunctions(dir);
    try { gitLoud(maps, 'worktree', 'remove', '--force', dir); } catch (e) {
      // git can drop the checkout and leave the folder (empty, or holding what it could not delete)
      if (fs.existsSync(dir) && !fs.existsSync(path.join(dir, '.git'))) { try { fs.rmdirSync(dir); } catch { /* not empty */ } }
      if (fs.existsSync(dir)) {
        if (fs.existsSync(path.join(dir, '.git'))) relink(dir);
        const h = holders(dir);
        fail(`${dir} was not removed (${e.message.split('\n')[0]}). Possibly held by:\n  ${h.join('\n  ')}\nClose or stop them yourself, then: node trainworld/checks/agent-tree.mjs remove ${path.basename(dir)}`);
      }
      gitLoud(maps, 'worktree', 'prune');
    }
  }
  try { git(maps, 'branch', '-D', branch); } catch { /* already gone */ }
}

// T-4016: where a landing goes. master is the main tree's checkout; any other tw/<x> branch (tw/track) is checked out in
// its own persistent worktree maps/.tw-trees/_<x>, so the main tree's files and index are never touched.
function intoOf(v) {
  if (!v || v === 'master') return { branch: 'master', dir: maps, track: false };
  const short = String(v).replace(/^tw\//, '');
  if (!/^[a-z0-9][a-z0-9-]{0,30}$/.test(short)) fail(`--into takes master or a branch tw/<name>: "${v}"`);
  return { branch: `tw/${short}`, dir: path.join(trees, `_${short}`), track: true };
}
const hasBranch = (b) => { try { git(maps, 'rev-parse', '--verify', '-q', `refs/heads/${b}`); return true; } catch { return false; } };
// The worktree for a target branch, made on first use; refuses one with uncommitted edits to tracked files.
function ensureTarget(t) {
  if (!t.track) return;
  if (!hasBranch(t.branch)) fail(`no branch ${t.branch}`);
  if (!fs.existsSync(path.join(t.dir, '.git'))) {
    if (fs.existsSync(t.dir)) { try { fs.rmdirSync(t.dir); } catch { fail(`${t.dir} exists and is not a worktree; delete it by hand`); } }
    excludeTrees();
    fs.mkdirSync(trees, { recursive: true });
    gitLoud(maps, 'worktree', 'add', '-q', t.dir, t.branch);
    say(`made the worktree for ${t.branch}: ${t.dir}`);
  }
  relink(t.dir);
  const wasm = path.join(maps, 'trainworld', 'app', 'src', 'wasm');
  if (fs.existsSync(wasm) && !fs.existsSync(path.join(t.dir, 'trainworld', 'app', 'src', 'wasm'))) fs.cpSync(wasm, path.join(t.dir, 'trainworld', 'app', 'src', 'wasm'), { recursive: true });
  const head = git(t.dir, 'symbolic-ref', '--short', '-q', 'HEAD');
  if (head !== t.branch) fail(`${t.dir} has ${head} checked out, not ${t.branch}`);
  const dirty = git(t.dir, 'status', '--porcelain', '--untracked-files=no');
  if (dirty) fail(`${t.dir} (the ${t.branch} checkout) has uncommitted edits, so nothing was changed there:\n${dirty}\nCommit or discard them there first.`);
}

// T-4016: changelog items of a tree headed for a branch other than master wait in trainworld/changelog/next-track.json
// on that branch (the release merge turns them into next.json items). `take` reverts the tree's next.json to its base and
// parks the new items in the tree's git dir; landTree writes them into the file after the rebase, so the file never conflicts.
const TRACK_LOG = 'trainworld/changelog/next-track.json';
const pendingFile = (dir) => path.join(git(dir, 'rev-parse', '--absolute-git-dir'), 'tw-pending-changelog.json');
const readPending = (dir) => { try { return JSON.parse(fs.readFileSync(pendingFile(dir), 'utf8')); } catch { return []; } };
function takeChangelog(dir, into, cl) {
  const parked = readPending(dir);
  const fresh = cl.items.map((i) => ({ text: i.text, area: i.area ?? 'other', ...(i.fix ? { fix: true } : {}), ...(i.task ? { task: i.task } : {}), ...(i.by ? { by: i.by } : {}) }));
  if (cl.baseText === null) { fs.rmSync(path.join(dir, CHANGELOG), { force: true }); try { git(dir, 'rm', '-q', '--cached', '--', CHANGELOG); } catch { /* untracked */ } }
  else gitLoud(dir, 'checkout', cl.base, '--', CHANGELOG);
  fs.writeFileSync(pendingFile(dir), JSON.stringify([...parked, ...fresh], null, 2));
  return fresh;
}
function writeTrackLog(dir, items) {
  const file = path.join(dir, TRACK_LOG);
  const doc = fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, 'utf8')) : { release: 'next-track', items: [] };
  const at = new Date().toISOString().slice(0, 10);
  let n = 0;
  for (const it of items) {
    if (doc.items.some((h) => h.text === it.text && (h.task ?? null) === (it.task ?? null))) continue;
    doc.items.unshift({ ...it, at }); n++;
  }
  fs.writeFileSync(file, JSON.stringify(doc, null, 2) + '\n');
  return n;
}

// T-4180: a tree mid-merge, mid-rebase or mid-cherry-pick must not be landed: committing it would make a merge commit
// that the rebase below drops (lost work, 2026-11-24). Returns the problem as a sentence, or '' when the tree is clean.
function inProgress(dir) {
  const has = (n) => fs.existsSync(path.resolve(dir, git(dir, 'rev-parse', '--git-path', n)));
  const found = [['MERGE_HEAD', 'a merge'], ['CHERRY_PICK_HEAD', 'a cherry-pick'], ['REVERT_HEAD', 'a revert'], ['rebase-merge', 'a rebase'], ['rebase-apply', 'a rebase']].filter(([n]) => has(n)).map(([, w]) => w);
  if (!found.length) return '';
  return `${dir} has ${[...new Set(found)].join(' and ')} in progress, so nothing was landed. Finish it there (resolve and commit, or --abort), then land again`;
}
// Merge commits on the branch since it left the target (an agent merged master into its tree). Rebase drops merges, so landTree
// first squashes the branch to its net diff against the merge base, which is the work without what the merge brought in.
const mergesOnBranch = (dir, into) => Number(git(dir, 'rev-list', '--merges', '--count', `${git(dir, 'merge-base', into, 'HEAD')}..HEAD`));

// Puts the tree's work on the target as one commit; returns the trainworld/ files it changed.
function landTree(dir, branch, message, coauthors = [], t = intoOf('master')) {
  if (!message) fail('land needs a commit message');
  if (!fs.existsSync(dir)) fail(`no tree ${dir}`);
  if (t.track && !hasBranch(t.branch)) fail(`no branch ${t.branch}`);
  { const p = inProgress(dir); if (p) fail(p); }
  restoreGameJson(dir);
  if (t.track) {
    // a plain `land` has no changelog step: take the items here (finish already did and parked them)
    const cl = treeChangelog(dir, t.branch);
    if (cl.exists && cl.differs) say(`changelog: ${takeChangelog(dir, t, cl).length} item(s) parked for ${TRACK_LOG}`);
  }
  const { tw, other } = changes(dir, t.branch);
  if (other.length) fail(`changes outside trainworld/, land them by hand:\n  ${other.join('\n  ')}`);
  const pending = t.track ? readPending(dir) : [];
  if (!tw.length && !pending.length) fail('nothing to land');
  warnGameJson(dir, 'land', [], t.track ? t.branch : 'master');
  ensureTarget(t);
  // The links are untracked junctions; `add trainworld` must not see them as content.
  gitLoud(dir, 'add', '--all', '--', 'trainworld');
  if (git(dir, 'diff', '--cached', '--name-only')) gitLoud(dir, 'commit', '-q', '-m', 'agent-tree: work in progress');
  if (mergesOnBranch(dir, t.branch)) {
    const n = mergesOnBranch(dir, t.branch);
    gitLoud(dir, 'reset', '-q', '--soft', git(dir, 'merge-base', t.branch, 'HEAD'));
    gitLoud(dir, 'commit', '-q', '-m', 'agent-tree: work in progress');
    say(`${n} merge commit(s) on the branch: squashed its net diff against ${t.branch} before the rebase`);
  }
  rebaseOnto(dir, t.branch);
  if (pending.length) {
    say(`changelog: ${writeTrackLog(dir, pending)} item(s) added to ${TRACK_LOG}`);
    gitLoud(dir, 'add', '--', TRACK_LOG);
    tw.push(TRACK_LOG);
  }
  gitLoud(dir, 'reset', '-q', '--soft', t.branch);
  // one trailer paragraph, so GitHub credits a contributor (--coauthor) beside the agent
  const trailers = [...coauthors.map((c) => `Co-Authored-By: ${c}`), 'Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>'];
  gitLoud(dir, 'commit', '-q', '-m', message, '-m', trailers.join('\n'));
  if (t.track) {
    try { gitLoud(t.dir, 'merge', '-q', '--ff-only', branch); } catch (e) {
      fail(`${t.branch} did not fast-forward to ${branch} (${t.branch} moved while landing, or git refused; see the log): the commit waits on the branch; land again`);
    }
    fs.rmSync(pendingFile(dir), { force: true });
    say(`${t.branch} is now ${git(t.dir, 'log', '-1', '--format=%h %s')}`);
    return tw;
  }
  try { gitLoud(maps, 'merge', '-q', '--ff-only', branch); } catch (e) {
    const name = path.basename(dir);
    const touched = new Set(git(maps, 'diff', '--name-only', `master..${branch}`).split('\n').filter(Boolean));
    // not git(): its trim would eat the first line's leading space
    const dirty = execFileSync('git', ['-C', maps, 'status', '--porcelain', '--untracked-files=no'], { encoding: 'utf8' }).split('\n').filter(Boolean).map((l) => l.slice(3).replace(/^"|"$/g, ''));
    const blockers = dirty.filter((f) => touched.has(f));
    if (blockers.length) {
      fail(`master did not fast-forward to ${branch}: the main tree has uncommitted edits in ${blockers.join(', ')}, which the landing commit also changes. The commit waits on the branch; nothing of the main tree was touched.\n` +
        `Commit them first (they are the orchestrator's): git -C ${maps} commit -m "<one sentence>" -- ${blockers.join(' ')}\n` +
        `then land again: node trainworld/checks/agent-tree.mjs ${LOG ? 'finish' : 'land'} ${name} <the same arguments as before>`);
    }
    fail(`master did not fast-forward to ${branch} (master moved while landing, or git refused for another reason; see the log): the commit waits on the branch; land again`);
  }
  say(`master is now ${git(maps, 'log', '-1', '--format=%h %s')}`);
  return tw;
}

// T-4571: a landing changes app/ at the merge and the WASM only minutes later, so a gate (world_run.py freeze_app) copying app/
// in between would serve a page its WASM can't run. Landing holds this file from before the merge until the WASM is built;
// freeze_app waits while it exists (a dead owner's or a 45-minute-old one is ignored).
const landingLock = path.join(maps, 'trainworld', 'data', 'work', 'landing.lock');
let holdingLanding = false;
function releaseLanding() {
  if (!holdingLanding) return;
  holdingLanding = false;
  try { if (JSON.parse(fs.readFileSync(landingLock, 'utf8')).pid === process.pid) fs.rmSync(landingLock, { force: true }); } catch { /* gone */ }
}
function holdLanding() {
  fs.mkdirSync(path.dirname(landingLock), { recursive: true });
  fs.writeFileSync(landingLock, JSON.stringify({ pid: process.pid, since: new Date().toISOString() }));
  if (!holdingLanding) process.on('exit', releaseLanding);
  holdingLanding = true;
}

// Agents' pages in the main tree load its WASM bindings, which a landed Rust change leaves stale
// (three agents hit "set_part is not a function" after T-1543, 2026-10-27).
function rebuildWasm(root = maps) {
  say(`Rust changed: rebuilding the WASM in ${root === maps ? "the main tree" : root} (npm run wasm)`);
  const cargoBin = path.join(process.env.USERPROFILE ?? '', '.cargo', 'bin');
  run('npm', ['run', 'wasm', '--prefix', path.join(root, 'trainworld', 'app')], {
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
function treeChangelog(dir, into = 'master') {
  const file = path.join(dir, CHANGELOG);
  if (!fs.existsSync(file)) return { exists: false, items: [], edited: 0 };
  const base = git(dir, 'merge-base', into, 'HEAD');
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
    else if (a === '--into') {
      if (i + 1 >= args.length) { console.error('--into needs a branch'); process.exit(2); }
      o.into = args[++i];
    }
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
  const t = intoOf(o.into);
  const into = t.branch;
  if (!fs.existsSync(dir)) { console.error(`no tree ${dir}`); process.exit(1); }
  if (t.track && !hasBranch(into)) { console.error(`no branch ${into}`); process.exit(1); }
  const inside = o.keep ? null : cwdInside(dir);
  if (inside) { console.error(`finish stopped before changing anything: ${inside}`); process.exit(1); }
  const logDir = fs.existsSync(path.join(maps, 'artifacts', 'trainworld')) ? path.join(maps, 'artifacts', 'trainworld') : os.tmpdir();
  const stamp = new Date().toISOString().replace(/[-:]/g, '').slice(0, 15);
  LOG = o.log ? path.resolve(o.log) : path.join(logDir, `finish-${name}-${stamp}.log`);
  if (!o.dry) fs.writeFileSync(LOG, `finish ${name} ${new Date().toISOString()}\n`);
  const tag = o.dry ? '(dry run) ' : '';
  const targetName = t.track ? `${into} (${path.basename(t.dir)})` : 'master';
  const line = (n, text) => console.log(`[${n}/5] ${tag}${text}`);
  const state = { moved: [], landed: false, wasm: false, removed: false, doneTasks: [], simChanged: false };
  let step = 0;
  const later = () => {
    const todo = [];
    if (!state.landed) return todo;
    if (!state.wasm && state.simChanged) todo.push(`npm run wasm --prefix ${path.join(t.dir, 'trainworld', 'app')}`);
    if (!state.removed && !o.keep) todo.push(`node trainworld/checks/agent-tree.mjs remove ${name}`);
    for (const t of tasks.filter((t) => !state.doneTasks.includes(t))) todo.push(`venv\\Scripts\\python.exe trainworld/tools/tasks.py done ${t} --outcome "..."`);
    return todo;
  };
  { const p = inProgress(dir); if (p) { console.error(`finish stopped before changing anything: ${p}`); process.exit(1); } }
  try {
    // 1: changelog items an agent put in its own tree
    step = 1;
    const cl = treeChangelog(dir, into);
    if (!cl.exists || !cl.differs) line(1, 'changelog: tree untouched');
    else if (t.track) {
      const fresh = cl.items;
      if (!o.dry) takeChangelog(dir, t, cl);
      line(1, `changelog: reverted the tree's next.json, ${fresh.length} item(s) ${o.dry ? 'to park' : 'parked'} for ${TRACK_LOG} on ${into} (the release merge moves them to next.json)${cl.edited ? `, ${cl.edited} edit(s) to existing items discarded` : ''}`);
      for (const it of fresh) console.log(`      ${it.fix ? 'fix ' : ''}${it.area ?? 'other'} ${it.task ?? '-'} ${it.by ?? '-'}: ${it.text}`);
    } else {
      const mainFile = path.join(maps, CHANGELOG);
      const have = fs.existsSync(mainFile) ? readJsonText(fs.readFileSync(mainFile, 'utf8')).items ?? [] : [];
      const fresh = cl.items.filter((i) => !have.some((h) => h.text === i.text && (h.task ?? null) === (i.task ?? null)));
      const dup = cl.items.length - fresh.length;
      if (!o.dry) {
        // back to the base content, committed or not
        if (cl.baseText === null) { fs.rmSync(path.join(dir, CHANGELOG), { force: true }); try { git(dir, 'rm', '-q', '--cached', '--', CHANGELOG); } catch { /* untracked */ } }
        else gitLoud(dir, 'checkout', cl.base, '--', CHANGELOG);
        for (const it of fresh) {
          // an item reworded inside the tree moves with its new wording
          const a2 = [path.join(maps, 'trainworld', 'tools', 'changelog.py'), 'add', it.edited || it.text, '--area', it.area ?? 'other'];
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
      const pr = inProgress(dir);
      if (pr) throw new Error(pr);
      const ch = changes(dir, into);
      const tw = ch.tw.filter((f) => !(f === CHANGELOG && cl.exists && cl.differs));
      if (ch.other.length) throw new Error(`changes outside trainworld/, land them by hand:\n  ${ch.other.join('\n  ')}`);
      if (!tw.length && !(t.track && (cl.items.length || readPending(dir).length))) throw new Error('nothing to land');
      state.simChanged = tw.some((f) => f.startsWith('trainworld/sim/'));
      line(2, `land ${branch}: ${tw.length} trainworld file(s), ${git(dir, 'rev-list', '--count', `${into}..HEAD`)} branch commit(s) plus any uncommitted work, rebased onto ${targetName} and squashed into one commit "${message}"`);
      if (t.track && fs.existsSync(path.join(t.dir, '.git')) && git(t.dir, 'status', '--porcelain', '--untracked-files=no')) throw new Error(`${t.dir} (the ${into} checkout) has uncommitted edits; commit or discard them there first`);
    } else {
      holdLanding();
      const tw = landTree(dir, branch, message, o.coauthor, t);
      state.landed = true;
      state.simChanged = tw.some((f) => f.startsWith('trainworld/sim/'));
      line(2, `landed ${branch} on ${targetName}: ${git(t.dir, 'log', '-1', '--format=%h %s')} (${tw.length} trainworld file(s))`);
    }
    // 3: wasm
    step = 3;
    if (state.simChanged) {
      if (!o.dry) { rebuildWasm(t.dir); state.wasm = true; }
      line(3, `wasm: sim/ changed, ${t.track ? `${path.basename(t.dir)}'s` : "main tree's"} WASM ${o.dry ? 'would be rebuilt' : 'rebuilt'}`);
    } else line(3, 'wasm: sim/ unchanged, no rebuild');
    releaseLanding();
    // 4: remove the tree
    step = 4;
    if (o.keep) line(4, `tree kept: ${dir}`);
    else {
      if (!o.dry) {
        removeTree(dir, branch);
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
    releaseLanding();
    const names = ['', 'changelog', 'land', 'wasm', 'remove tree', 'tasks'];
    console.error(`FAILED at step ${step}/5 (${names[step]}):\n${e.message}`);
    console.error(`state: ${state.moved.length ? `${state.moved.length} changelog item(s) already moved to the main tree and reverted in the tree (do not move again). ` : ''}${state.landed ? `${into} has the landing commit (${git(maps, 'log', '-1', '--format=%h', into)}). ` : `${into} is unchanged by this run. `}${state.removed ? 'tree removed. ' : 'tree still there. '}${state.doneTasks.length ? `tasks archived: ${state.doneTasks.join(', ')}. ` : ''}`);
    const left = later();
    if (left.length) console.error(`still to do by hand:\n  ${left.join('\n  ')}`);
    if (!o.dry) console.error(`log: ${LOG}`);
    process.exit(1);
  }
}

// T-4016: merge master into the track branch, in its own worktree. A conflict in an append-only doc keeps both sides; any
// other conflict stops with the merge left open there. master is only read.
function syncTrack(args) {
  const dry = args.includes('--dry-run');
  const log = args.indexOf('--log');
  if (log >= 0) LOG = path.resolve(args[log + 1]);
  const t = intoOf(args.find((a, i) => !a.startsWith('--') && args[i - 1] !== '--log') ?? 'tw/track');
  if (!t.track) fail('sync-track merges master into a branch tw/<name>, default tw/track');
  if (!hasBranch(t.branch)) fail(`no branch ${t.branch}`);
  const behind = Number(git(maps, 'rev-list', '--count', `${t.branch}..master`));
  const touched = git(maps, 'diff', '--name-only', `${git(maps, 'merge-base', t.branch, 'master')}`, 'master').split('\n').filter(Boolean);
  if (!behind) { console.log(`${t.branch} already has everything on master`); return; }
  if (dry) {
    const base = git(maps, 'merge-base', t.branch, 'master');
    const out = execFileSync('git', ['-C', maps, 'merge-tree', base, t.branch, 'master'], { encoding: 'utf8', maxBuffer: 1 << 30 });
    const conflicts = /^\+<<<<<<< /m.test(out) ? 'would conflict (merge-tree shows markers; docs under notes/DECISIONS/ARCHIVE/research are merged automatically)' : 'would merge cleanly';
    console.log(`(dry run) merge master into ${t.branch}: ${behind} master commit(s) to bring in, ${touched.length} file(s) changed on master since the fork; ${conflicts}`);
    return;
  }
  ensureTarget(t);
  const before = git(t.dir, 'rev-parse', 'HEAD');
  let last;
  try { gitLoud(t.dir, 'merge', '-q', '--no-edit', '-m', 'Merge master into ' + t.branch, 'master'); } catch (e) { last = e; }
  for (let round = 0; last && round < 30; round++) {
    const unmerged = git(t.dir, 'status', '--porcelain').split('\n').filter((l) => /^(AA|UU|DD|AU|UA|DU|UD) /.test(l));
    if (!unmerged.length) fail(`the merge of master into ${t.branch} failed in ${t.dir} without a conflict: ${last.message}`);
    const bad = unmerged.filter((l) => !/^(AA|UU) /.test(l) || !APPEND_ONLY.test(l.slice(3).trim()));
    if (bad.length) {
      fail(`the merge of master into ${t.branch} stopped on a conflict in ${t.dir}: ${bad.map((l) => l.slice(3).trim()).join(', ')}. Resolve there and \`git commit\`, or \`git merge --abort\`; then sync-track again ` +
        `(only a conflict in DECISIONS.md, ARCHIVE.md, notes/*.md or research/*.md is merged by keeping both sides)`);
    }
    for (const l of unmerged) {
      const rel = l.slice(3).trim();
      const file = path.join(t.dir, rel);
      const out = keepBoth(fs.readFileSync(file, 'utf8'));
      if (/^(<<<<<<<|=======|>>>>>>>)/m.test(out)) fail(`could not merge the conflict markers in ${rel} (${t.dir}); resolve there by hand`);
      fs.writeFileSync(file, out);
      gitLoud(t.dir, 'add', '--', rel);
      say(`merge conflict in ${rel}: kept both sides, master's first`);
    }
    try { run('git', ['-C', t.dir, '-c', 'core.editor=true', 'commit', '--no-edit']); last = null; } catch (e) { last = e; }
  }
  const after = git(t.dir, 'rev-parse', 'HEAD');
  console.log(`${t.branch} is now ${git(t.dir, 'log', '-1', '--format=%h %s')} (${behind} master commit(s) merged)`);
  if (before !== after && git(t.dir, 'diff', '--name-only', before, after).split('\n').some((f) => f.startsWith('trainworld/sim/'))) rebuildWasm(t.dir);
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
  const warm = seedTarget(dir);
  console.log(`Rust builds into the tree's own trainworld/sim/target (${warm ? 'started from a warm copy' : 'first build from nothing, several minutes'}).`);
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
  const words = [];
  let intoArg, dry = false;
  for (let i = 0; i < rest.length; i++) {
    if (rest[i] === '--into') { intoArg = rest[++i]; if (!intoArg) fail('--into needs a branch'); }
    else if (rest[i] === '--dry-run') dry = true;
    else words.push(rest[i]);
  }
  const t = intoOf(intoArg);
  if (dry) {
    if (!fs.existsSync(dir)) fail(`no tree ${dir}`);
    if (t.track && !hasBranch(t.branch)) fail(`no branch ${t.branch}`);
    { const p = inProgress(dir); if (p) fail(p); }
    const ch = changes(dir, t.branch);
    const nm = mergesOnBranch(dir, t.branch);
    if (nm) console.log(`${nm} merge commit(s) on the branch: the real land squashes its net diff first`);
    console.log(`(dry run) land ${branch} onto ${t.branch}: ${ch.tw.length} trainworld file(s), ${git(dir, 'rev-list', '--count', `${t.branch}..HEAD`)} branch commit(s) plus any uncommitted work${ch.other.length ? `; ${ch.other.length} change(s) outside trainworld/ would stop it` : ''}`);
    if (t.track && fs.existsSync(path.join(t.dir, '.git')) && git(t.dir, 'status', '--porcelain', '--untracked-files=no')) console.log(`${t.dir} has uncommitted edits: the real land would stop`);
    process.exit(0);
  }
  holdLanding();
  const tw = landTree(dir, branch, words.join(' ').trim(), [], t);
  if (tw.some((f) => f.startsWith('trainworld/sim/'))) rebuildWasm(t.dir);
  releaseLanding();
} else if (cmd === 'sync-track') {
  syncTrack([name, ...rest].filter(Boolean));
} else if (cmd === 'finish') {
  finishTree(rest);
} else if (cmd === 'remove') {
  const { dir, branch } = treeOf(name);
  const force = rest.includes('--force');
  if (fs.existsSync(path.join(dir, '.git'))) {
    const { tw, other } = changes(dir);
    if (!force && (tw.length || other.length)) fail(`${dir} has ${tw.length + other.length} changed files not on master; land them or pass --force`);
  }
  removeTree(dir, branch);
  console.log(`removed ${name}`);
} else {
  console.log(fs.readFileSync(fileURLToPath(import.meta.url), 'utf8').split('\n').slice(0, 12).join('\n'));
  process.exit(cmd ? 2 : 0);
}
