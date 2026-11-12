# Contributing

Bug reports and pull requests are welcome. For a big change, open an issue first so we can talk about it.

## How changes get in

This repo is a copy of the game's working tree, which is private because it holds notes and tooling that are not
for publishing. Pull requests are reviewed here but not merged with the GitHub button. When one is accepted, the
maintainer applies it to the working tree. The next copy of the tree is committed here with you named as a
co-author, and the pull request is closed with a link to that commit.

So keep each pull request small and on one thing, based on the latest `main`. The commit that ends up in `main`
has a different hash from yours.

## Before you send one

- `cd trainworld/app && npm run check` passes.
- After a change in `trainworld/sim/`, from `trainworld/sim` run `cargo test --release` and
  `cargo test --release --features demand --lib`.
- If you changed what players see, say what changed. Text in the game follows `trainworld/docs/ui-panels.md`,
  the sections "Look" and "Hover tags". Messages live in `trainworld/app/src/i18n/messages/en.json`.
- If you changed a rule, change the doc that states it (`trainworld/SPEC.md` or a file in `trainworld/docs/`) in the
  same pull request. The docs say what is true now.
- Comments say why, not what.

## Licence

By sending a pull request you agree that your contribution is licensed under the AGPL-3.0, like the rest of the code.
Do not add data, fonts, pictures or sounds without saying where they come from and under what licence.
