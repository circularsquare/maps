# renameui: three small tweaks from her 11-25

Her words: "for renaming, we should put a (R) hotkey indicator as usual, on the rename button tooltip so people know / for the
bulk rename lets tweak tooltip to '... Paste a list of names to fill multiple' / for speedometer lets put the name fo the next
stop on the line below next stop, so that we can put the speedometer to the right of the next stop".

## What changed

1. **Station rename hover** (`ui/inspector/NameEdit.tsx`): the pencil's hover names the key when the name box has one
   (only a station's: `hotkey={RENAME_KEY}`, T-4273). "As usual" is the game's pattern "Label (K)" (`buildTab.busTip`,
   `linesTab.newLineTip`, `blueprint.disableTip`), which `ui/tip.ts` KEY_HINT letters in the accent and drops on phones;
   it matches the bracket anywhere in the line, so "(R)" goes right after "Rename", the action it does. The message is a
   `keyed` select as `blueprint.disableTip` is, so a line's and a building's pencil keep the plain wording.
2. **Bulk rename hover**, second line (`lineInspector.renamePaste`): her wording.
3. **Train panel** (`ui/inspector/TrainInspector.tsx`, `ui/style.css` `.next-stop`): the Next stop row became a block:
   the label and the stop's name as two 24 px rows at the left (a long name wraps, never cut), the speed dial beside them
   at the panel's right, the two centred on each other; 4 px over it, 2 px under it as the dial had (14 px to On board,
   the heading's own gap). `.speedo` lost its auto margin (it right-aligned the dial under the row).

Docs: ui-inspectors.md "The speed dial" (where it sits); ui-panels.md key table gained R (T-4273 had left it out; the
rule over the table says each button's hover shows its key).

## Read before run: what could go wrong

- A pencil without a key (line, building) must not show "()": select on `keyed`, false there.
- Phones: tip.ts removes " (R)" with its space, leaving "Rename, or double click the name".
- Japanese and Chinese write keys in full-width brackets, which KEY_HINT does not match: not lettered, and kept on
  phones. True of every keyed hover in those three languages, not only this one: filed T-4327 (P3).
- Long names: the text column has min-width 0 and wraps anywhere; the dial is flex: none.
- The finished-trip panel (no `at`) is unchanged; "Terminus" takes the name's line.

## Strings (old -> new, as shown)

- Station pencil hover: "Rename, or double click the name" -> "Rename (R), or double click the name" (lines and buildings
  keep the old wording). `nameEdit.renameTip`.
- Line panel rename icon hover, line 2: "Paste a list of names into a box to fill it and the ones below" -> "Paste a list
  of names to fill multiple" (her wording). `lineInspector.renamePaste`. Line 1 "Rename stations" unchanged.
- Train panel: no text change ("Next stop" and the stop's name, moved).

## Translations

Done as docs/localization.md "Translating" says, for the 3 keys of the two hovers (renameStations translated too, so the
hover is not half English) in all 12 offered languages: pass 1 by me (l10n.mjs batch, merge, lint), pass 2 a fresh
Sonnet back-translator, pass 3 a fresh Opus reviewer (no meaning errors; 5 style rewrites merged: ja's plain branch made
parallel to the keyed one, zh-Hans/zh-TW paste line ends on 站名, ru both halves infinitive, ko 더블클릭 as the help
strings write it). `checks/i18n.mjs` clean. Lint's only warnings are width ones, from the word "button" in the notes; these
are hovers, which wrap.

| lang | nameEdit.renameTip | lineInspector.renameStations | lineInspector.renamePaste |
|---|---|---|---|
| ja | {keyed, select, true {名前を変更（{key}）、または名前をダブルクリック} other {名前を変更、または名前をダブルクリック}} | 駅名を変更 | 名前のリストを貼り付けて複数の欄に入力 |
| zh-Hans | {keyed, select, true {重命名（{key}），或双击名称} other {重命名，或双击名称}} | 重命名车站 | 粘贴名称列表，一次填入多个站名 |
| zh-TW | {keyed, select, true {重新命名（{key}），也可以按兩下名稱} other {重新命名，也可以按兩下名稱}} | 重新命名車站 | 貼上名稱清單，一次填入多個站名 |
| es | {keyed, select, true {Renombra ({key}), o haz doble clic en el nombre} other {Renombra, o haz doble clic en el nombre}} | Renombra las estaciones | Pega una lista de nombres para rellenar varias casillas |
| pt | {keyed, select, true {Renomear ({key}), ou clique duas vezes no nome} other {Renomear, ou clique duas vezes no nome}} | Renomear estações | Cole uma lista de nomes para preencher vários campos |
| de | {keyed, select, true {Umbenennen ({key}) oder auf den Namen doppelklicken} other {Umbenennen oder auf den Namen doppelklicken}} | Stationen umbenennen | Namensliste einfügen, um mehrere Felder auszufüllen |
| fr | {keyed, select, true {Renommer ({key}), ou double-cliquer sur le nom} other {Renommer, ou double-cliquer sur le nom}} | Renommer les stations | Coller une liste de noms pour remplir plusieurs champs |
| tr | {keyed, select, true {Ad değiştir ({key}) ya da adına çift tıkla} other {Ad değiştir ya da adına çift tıkla}} | İstasyon adlarını değiştir | Birden çok kutuyu doldurmak için ad listesi yapıştır |
| ru | {keyed, select, true {Переименовать ({key}) или дважды щёлкнуть по названию} other {Переименовать или дважды щёлкнуть по названию}} | Переименовать станции | Вставьте список названий, чтобы заполнить сразу несколько полей |
| ko | {keyed, select, true {이름 변경 ({key}) 또는 이름 더블클릭} other {이름 변경 또는 이름 더블클릭}} | 역 이름 변경 | 이름 목록을 붙여 넣어 여러 칸 채우기 |
| ca | {keyed, select, true {Canvia el nom ({key}) o fes doble clic al nom} other {Canvia el nom o fes doble clic al nom}} | Canvia el nom de les estacions | Enganxa una llista de noms per omplir diverses caselles |
| id | {keyed, select, true {Ganti nama ({key}), atau klik dua kali namanya} other {Ganti nama, atau klik dua kali namanya}} | Ganti nama stasiun | Tempel daftar nama untuk mengisi beberapa kolom sekaligus |

## Checks

Typecheck clean; `checks/i18n.mjs` clean. One new NYC sandbox game on my Vite (6189), a three-station line, stations
named by edit ops (one long: "34 St-Hudson Yards/Javits Center Terminal"). Screenshots in artifacts/trainworld/review/:
rename-tooltip.png (tag reads "Rename (R), or double click the name", R in the accent), bulk-rename-hover.png,
train-panel-speed.png (a train at 44 km/h heading for the long name).

Measured on the panel (CSS px; the panel draws at 0.9): long name wraps to two lines in a 231 px column, nothing cut;
the dial's right edge on the panel's value column; 4 px from the Towards line to the block; 14 px from the block to On
board. Short name ("Times Sq-42 St"): label and name 48 px centred on the 65 px dial, so the label's row starts 8 px
under the dial's top; block height 65 px, against 97 px for the old row plus dial under it.

No advisor call (trainworld CLAUDE.md: not for small UI or wording). Benchmarks: none. Game checks: about 9 minutes.
