# Notes for Gemini

You are working in Anita's maps repository alongside Claude agents and a Claude orchestrator. The rules that hold
for them hold for you too; Claude's hooks that enforce some of them don't run for you, so follow them by hand.

Read first: `CLAUDE.md` at this root, then `trainworld/CLAUDE.md` for anything under `trainworld/`.

Hard rules (no exceptions without Anita saying so in her own words):
- **No AWS at all.** Never run `aws`, boto3 or anything that reaches an AWS account.
- **No commits, no pushes, no publishing, no server deploys.** Leave your changes uncommitted; the trainworld
  orchestrator lands them. Say which files you changed when you're done.
- **Worktrees only through** `node trainworld/checks/agent-tree.mjs new|remove <name>`; never `git worktree remove`,
  never delete a `.tw-trees` folder by hand (it deletes shared `node_modules` and `data` for everyone).
- **Never** a bare `git stash pop` (the stash is shared by every worktree), `git reset`, `git checkout -- <files>` on
  files you didn't change, or `git clean`.
- **Don't edit** `trainworld/TASKS.md`, `DECISIONS.md`, `ARCHIVE.md`, `LINES.md`, `notes/handoff.md` or
  `notes/asks.json`, nor `trainworld/TODO.md` (Anita's own); tell Anita what should change and the orchestrator does it.
- **CPU**: claim cores before anything heavy (`python cores.py claim <n> <name> --pid <pid> "what"`, release when done;
  the cap is `python cores.py list`). Never kill processes broadly; target one PID you started.
- **Nothing on her screen**: no windows, no browsers outside a headless profile in a temp folder.
- **Never put Anita's name or email** in a request to an outside service.
- Before editing a file, check `git status` and the orchestrator's board (`venv\Scripts\python.exe
  trainworld/tools/handoff.py status`) for an agent already working on it; if one is, pick something else.
