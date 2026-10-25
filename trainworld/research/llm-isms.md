# LLM-isms in interface text and site copy

T-1242, 2026-10-24. Research on what makes text read as machine-written, aimed at short interface
text (labels, tooltips, buttons, empty states, status lines, errors, short descriptions, link
previews, map labels), and a proposed rewrite of the "User-facing text" section of Anita's global
CLAUDE.md. Her file is not edited; the proposal is section 4. The scan of trainworld's own strings
is in `notes/T-1242.md`.

## 1. Summary for Anita

- Your current rules hold up. Every one of them is backed by at least one style guide or study,
  and none is contradicted by anything I found. The proposal keeps all of them and adds about a
  dozen.
- The best evidence is on **words** (studies of millions of paper abstracts since 2023 show
  "delve", "underscore", "showcase", "crucial", "intricate", "notably" and a few hundred others
  rising sharply) and on **sentence shapes**: "not X, it's Y" and its cousins ("X rather than Y",
  "more than an X"), a trailing "-ing" clause that adds importance ("..., highlighting its role"),
  and lists of three. These turn up in every independent source.
- **Word lists go stale.** Each new model has its own favourites; one 2026 study found only a
  fifth to a half of the tells carry over from one model version to the next. So the durable rule
  is "use the plain word", not "avoid this list". The list is useful for checking, not as the rule.
- **Em dashes are no longer what models overuse.** Wikipedia's editors now call the em dash a
  historical sign (up to September 2026), and a 2026 count puts Claude Opus 5.5 at about 1/200 of
  the human rate. Your dash rule is still right, because readers react to dashes on sight, but the
  fix matters: swapping a dash for a colon or semicolon keeps the same shape, and models leaning on
  colons and commas is the newer tell.
- **The tells of the model writing for you now** (Claude Opus 5.5, measured against human articles):
  "this matters" (116 times the human rate), "why X matters", "is more than an X, it's a Y",
  "rather than simply", "adds another layer", "dependable", "can help you", "makes it easier". The
  last two are exactly your "objects don't act" rule.
- For interface text specifically, the strongest new rules come from the UX style guides rather
  than the AI research: say what it is, one word per thing, a tooltip never repeats its label,
  errors say what happened and what to do, no chatty assistant voice. (The guides also drop
  "please"; Anita keeps it.)
- One real conflict: a column of labels written in one grammar is good interface writing, while a
  set of free-text lines that all share one shape reads as generated. The proposal separates the two.
- trainworld's text is already plain. The scan has 41 rows: 26 proposed changes, most of them
  small, and 15 borderline ones for you; see `notes/T-1242.md`.

How the marks below read: **well attested** means three or more independent sources (studies,
style guides, editors) name it; **single source** means one source or one person; **her rule**
means it is already in her CLAUDE.md or memory notes. Section 5 is the list of sources, with what I
read in full and what only through summaries.

## 2. Patterns

Each pattern: what it is, an example, the fix, how well attested, sources (numbers refer to
section 5).

### 2.1 Word choice

**W-a. "AI vocabulary" clusters.** Well attested [1, 3, 4, 6, 12, 14, 20].
Words measured to spike since 2023: delve, underscore, showcase, highlight (verb), crucial,
pivotal, key, vital, intricate, tapestry, landscape (abstract), testament, foster, garner,
bolster, enhance, robust, meticulous, comprehensive, notably, additionally, interplay, vibrant,
valuable insights, align with. Kobak et al. found 379 such "style words" in 2024 PubMed abstracts
against almost none before, two thirds of them verbs, and estimate at least 13.5% of 2024 abstracts
were processed by an LLM [3]. Juzek and Ward tie the overuse to human-feedback training: raters
prefer text with these words [4]. Expert readers who use ChatGPT a lot named vocabulary in 53% of
their (very accurate) judgements [6].
GOV.UK's "words to avoid", written before LLMs, bans many of the same words (key, robust, foster,
deliver, empower, leverage, streamline, transform, facilitate) because they are "vague or
inflated" [14]. That is the better reason for an interface: the plain word is clearer, whoever
wrote it.
- Example: "A robust tool to showcase your network's key metrics"
- Fix: "Riders, money and train counts for each line"
- Caveat: one occurrence means nothing; density does [1]. The words change by model generation
  (Wikipedia keeps era lists [1]; Graphite found 19-46% overlap between consecutive versions [9]).

**W-b. Promotional and travel-guide words.** Well attested [1, 13, 19, 25].
boasts, nestled, in the heart of, rich, renowned, vibrant, bustling, scenic, breathtaking,
groundbreaking, diverse array, commitment to; on landing pages: unlock, unleash, elevate,
empower, supercharge, game-changer, seamless. Rettberg calls the slip into ad copy inside plain
text a "genre glitch": a word strongly tied to tourism ("train", "Chicago") pulls the model into
travel-brochure prose [13]. Relevant to a rail game: city and station descriptions are where it
would happen.
- Example: "Explore the vibrant heart of Osaka's bustling rail network"
- Fix: "Osaka: 3.4M trips a day, 41% by train"

**W-c. Avoiding "is" and "has".** Well attested [1, 7, 12].
serves as, stands as, functions as, represents, marks, boasts, features, offers, maintains.
Wikipedia cites measured declines in plain "is/are" in AI copyedits [1]; Reinhart et al. found
instruction-tuned models use nominalisations at 1.5-2 times the human rate [7].
- Example: "The Lines tab serves as a hub for managing your routes"
- Fix: "Every line, with riders and money" (or no description at all)

**W-d. Helper phrasing.** Well attested [9, 15, her rule].
can help you, makes it easier, lets you, allows you, is designed to, is especially helpful. These
rose in Claude Opus 5.5 against Opus 5 [9]. Microsoft's guide: start with a verb and edit out
"you can" [15]. Her rule: objects don't act.
- Example: "This view makes it easier to spot crowded lines"
- Fix: "Crowded lines in red"

**W-e. Softeners and magic adverbs.** Well attested [11, 12, her rule].
simply, just, easily, quickly, seamlessly, effortlessly, quietly, truly, genuinely, actually,
honestly. Google bans "simply", "easy" and "quickly" in instructions [11]; tropes.fyi and Kriss
name "quietly" and the habit of calling things subtle [10, 12].

**W-f. Model-specific tics (2026).** Single source (one measured study, secondary coverage) [9].
Claude Opus 5.5: "this matters", "why X matters", "is more than an X, it's a Y", "rather than
simply", "adds another layer", "dependable". GPT-6 era: "another dimension", "may/can provide",
"not simply X". Claude users also report "load-bearing" [21]. Worth knowing because the agents
writing her text are Claude models; worth not over-trusting because these lists turn over.

**W-g. Synonym cycling.** Well attested as a UX fault [1, 12, 18]; weakening as an AI tell.
Calling one thing the dashboard, the interface, the panel and the hub in turn. Wikipedia now files
"elegant variation" under historical signs [1]. In an interface it is a plain usability fault
whoever wrote it: the reader wonders whether two words mean two things. One term per concept is a
standing UX rule.
- Example (trainworld): "Sign in" and "Log in" for the same action; "build" and "construct".

**W-h. Generic wording where a specific fact exists.** Well attested [1, 16, 18, her rule].
Vague attributions ("studies show", "experts say"), "something went wrong", "learn more", "submit",
"changes applied", "available data". Mailchimp and Grafana both ask for labels that name the action
or destination [16, 18]; her panel note: "a wording general enough to fit every case describes
none of them".

### 2.2 Sentence shapes

**S-a. Negative parallelism and corrective framing.** Well attested, the strongest structural
tell [1, 2, 6, 8, 9, 10, 12, 17].
"It's not X, it's Y", "not just X but Y", "not only... but also", "X rather than Y", "no X, no Y,
just Z", "is more than an X", "not simply X", and the 2026 hedge form "should be treated as X rather
than Y". Kriss opens his essay on it [10]; Russell et al.'s expert readers cite "not only... but
also" [6]; Graphite measures "corrective framing" at over 100 times the human rate in GPT-6 Astra
and finds Claude has swapped "it's not X, it's Y" for "is more than an X" [9]. Wikipedia added a
"Y rather than X" section and a mid-2026 "corrective framings" section [1, 2].
- Example: "Not just a map. A living network."
- Fix: drop the tagline, or say the one positive fact it stands for.
- Caveat: it is an ordinary construction [10]. A "not" that marks a real boundary the reader needs
  ("Visits, shopping and business, not commutes") is fine. The tell is the reflex, and the swap
  of one form for another ("rather than" for "not").

**S-b. A trailing "-ing" clause that adds significance.** Well attested [1, 7, 12].
"..., highlighting its importance", "..., ensuring smooth service", "..., making it ideal for
commuters", "..., contributing to the city's growth". Reinhart et al. measured present participial
clauses at 2-5 times the human rate in instruction-tuned models (GPT-4o 5.3 times) [7]; Wikipedia
calls it "superficial analysis" [1].
- Example: "Trains wait at the ends, ensuring the timetable stays even"
- Fix: "Trains wait at the ends to keep the gaps even" or cut the second half.

**S-c. Rule of three.** Well attested [1, 6, 10, 12, 17, her rule].
Three adjectives, three parallel phrases, three bullets by habit. Strongest where people don't
normally reach for rhetoric (Wikipedia: edit summaries [1]), which includes interface text.
- Caveat: the device is older than print [17]. A real list of three things is fine.

**S-d. Two-part lines.** Her rule; supported by [2, 12, 21] (single sources each).
A short line split into a label and an add-on, "The X? A Y.", a title with a colon subtitle, and
colon or semicolon pauses where "because" or "but" would do. The supporting evidence is thinner
than for S-a to S-c, but it is the shape models fall back on once dashes are banned (see P-b).

**S-e. Punchy fragments.** Well attested in prose [8, 12, 21]; relevant to taglines.
"Not a bug. A design decision." "Fast. Simple. Yours." Silentbob on LessWrong: clusters of very
short sentences as a punchline rhythm [8].

**S-f. Hedges, preambles and disclaimers.** Well attested [1, 11, 12, 17].
"It's worth noting", "Importantly", "Note that", "Please note", "at this time", "based on available
information", "figures should be treated as indicative". Google bans "please note" and "at this
time" [11].
- Fix: delete and lead with the fact. If an estimate is an estimate, say what it is estimated
  from, once: "Visits estimated from each place's size".

**S-g. One shape for every line.** Well attested [6, her rule, 11].
Russell et al.: AI text has "less length variation" [6]. Her map-label note: "a corpus where
nearly every line is 'fragment; fragment' reads as machine-written even when each line is fine".
Google: avoid starting sentences with the same phrase [11].
- Conflict: interface labels in a list should share one grammar (all nouns, or all verb
  phrases). That is consistency, not a tell. The tell is in free text: descriptions, tooltips,
  notes, map captions.

### 2.3 Structure

**T-a. Telling the reader what they can see.** Well attested [5, 8, 22, her rule].
Tooltips that repeat the label, a line summarising the panel's own counts, an explanation of how a
feature works shown every time. PatternFly: "Don't repeat information that is already available in
the UI" [22]. The slop study's strongest predictors of a reader calling text slop were low
relevance, low density and tone, and verbosity was the one all annotators agreed on [5]. Her rule
(trainworld "Panels show what can change"): explanations go in a hover if anywhere.

**T-b. Summaries, signposts and tie-backs.** Well attested [1, 12], mostly prose.
"In short", "Here's what you need to know", "Five things to know", a closing line that sums up. In
an interface: a status aside that restates counts ("814 touched, 150 finished", cut by Anita in
noritetsu).

**T-c. Chatbot formatting.** Well attested [1, 12, 15, her rule].
Title Case Headings, a bold label at the start of every bullet, emoji before headings, decorative
arrows and checkmarks.

### 2.4 Punctuation

**P-a. Em dashes.** Well attested as a 2023-2025 tell; now historical [1, 9, 17].
Wikipedia lists em dash overuse under historical indicators "2022 to September 2026" [1]. Graphite's
counts per 1,000 words: human articles 3.14, GPT-4.1 7.13, Claude Opus 5 2.92, Gemini 2.5 Pro 1.43,
Claude Opus 5.5 0.015 [9] (other measurements of the same models differ a lot by prompt). Readers
still associate the mark with AI, which is the reason her rule gives, and that still holds.
- Conflict: Microsoft's style guide uses closed em dashes in its own examples [15].

**P-b. The substitute problem.** Single sources, consistent [1, 17, 21].
Wikipedia: since models dropped the em dash, "commas and periods now look more machine-like" and
GPT-6-era text runs to sentences of 8-20 words [1]. Proofed: swapping in an en dash or hyphen
"doesn't fix the rhythm" [17]. The "Claudisms" list names colon and semicolon pauses [21]. So a dash
becomes a rewrite, not a colon.

**P-c. Exclamation marks.** Well attested in style guides [11, 18, her rule]; a marketing tell
more than an LLM one.

**P-d. Unicode decoration and curly quotes.** Weak [1, 12].
Arrows (→), checkmarks, sparkles. Curly quotes point to ChatGPT, but word processors make them too
and Claude and Gemini typically don't [1]. In a game, an arrow can be the interface's own symbol.

**P-e. End punctuation on fragments.** Style guides, not an AI tell [15, 18, 23].
No period on a single-fragment label, button, heading or status; periods when the string is a
full sentence or several [18, 23]. Mixing both for the same kind of string is untidy.

### 2.5 Tone

**O-a. Significance inflation.** Well attested [1, 12].
"plays a pivotal role", "marks a turning point", "reshapes how you think about transit". On a map
or in a game this shows up as importance claims in descriptions.

**O-b. Sales pitch and invitation.** Well attested [12, 19, her rule].
"Discover", "Explore", "Imagine a world where", "Ready to build?". Her rule: instructions yes,
pitches no; describe the fact ("runs in your browser").
- Conflict: Microsoft rewrites a sentence as "Ready to buy? Contact us." as good practice [15].
  Corporate guides favour a friendly, chatty voice; her rule is stricter, and the AI research is on
  her side.

**O-c. Chatty assistant voice.** Well attested [1, 11, 18].
"Let's", "Great!", "Certainly", "Oops", "Hang tight", "We're crunching the numbers", "Welcome to
...!". Wikipedia's "collaborative communication" [1]; Google bans "let's" [11]; Grafana: "Use:
'The server is unresponsive.' Don't use: 'Oops, the server appears to be on a break.'" [18].

**O-d. Forced wit and metaphor.** Well attested [8, 11, 12, 14].
GOV.UK: metaphors "don't say what you mean and slow comprehension" [14].

**O-e. Safe, frictionless sameness.** Well attested as the overall impression [6, 10].
Expert readers flagged text as AI when it was "safe, straightforward... with few surprises", and
flagged human text by its odd specific words and imperfect grammar [6].
- Caveat: non-experts wrongly called neutral tone an AI sign [6]. The cure is specific facts, not
  added personality.

### 2.6 Interface-specific

**I-a. Interface nouns as subjects.** Her rule; supported [9, 15].
"The panel shows", "this button lets you", "a click opens". See W-d.

**I-b. Buttons name the action.** Well attested [16, 18, 15].
Verb plus object, one to four words, specific: "Delete line", not "OK" or "Submit"; the pair "Save"
and "Don't save" over "Save" and "Cancel" [18].

**I-c. Errors and refusals.** Well attested [15, 18].
Say what happened and, if possible, what to do; no blame, no apology, no "please", no "invalid"
[18]. Microsoft: "Invalid ID" becomes "You need an ID that looks like this: someone@example.com"
[15].

**I-d. Tooltips.** Supported [18, 22, 23].
Under about 120 characters [18]; don't repeat the label [22]; period only when a full sentence
[23]; don't hard-code a shortcut key the player can remap [23].

**I-e. Confirmations.** My own suggestion from common UX practice, not a fetched source.
A confirmation's heading names the action ("Delete line?") rather than "Are you sure?"; the
buttons repeat the verb ("Delete line" / "Keep line").

**I-f. Link previews and page titles.** Inferred from W-b, O-b and her rule; no source of its own.
The title is the name; the description states what it is and one fact ("A rail building game on
real maps. Runs in the browser."), no "Discover", no taglines.

### 2.7 Why it matters

Readers who believe a text is AI-written rate it lower, even when a person wrote it: an industry
study (Raptive, 2025) found ratings and the trust of nearby ads fell [24]; a set of preregistered
experiments found the effect strongest for emotional copy and weak for factual messages [24]. That
fits her aim: plain factual text is both less likely to read as AI and less penalised if it does.

### 2.8 Where the sources disagree

- **Detection versus quality.** The Hacker News thread on Wikipedia's page argues the signs are
  bad writing whoever wrote them, and that you cannot detect AI from style [25]. For her
  purposes this doesn't matter: every pattern above is a fault on its own terms.
- **Humanizer lists as house style.** One editor warns that a "humanizer" checklist turns into a
  single prescribed style of its own [17]. The proposal keeps each rule tied to a reason.
- **Em dashes**: historical for detection, still noticed by readers (P-a).
- **Friendly voice**: Microsoft and Grafana want contractions and warmth; GOV.UK avoids negative
  contractions ("cannot"); her rule and trainworld's text are on GOV.UK's side.
- **Uniform grammar**: tell in prose, virtue in label lists (S-g).
- **Rule of three, "not X"**: old devices; only reflexive use is the tell (S-a, S-c).
- **Capitalisation**: Apple's guidelines use title case for some controls (from memory; the page
  did not load). Her sentence-case rule matches Microsoft, Grafana and Google.

## 3. What changes against her current section

Kept as they are: avoid dashes between phrases (reason updated), people act, instructions not
pitches, facts over vibes, reflex structures, two-part lines, no hype punctuation, no softeners,
sentence case and few sizes and weights.

New: say what it is (copula); plain word, with a check list and the reason the list is not the
rule; name the specific thing (promoted from her panel note); one word per thing; positive
statements (the "not X" family, widened); no significance tails; no hedges; no assistant voice;
don't tell the reader what they can see; vary free text but keep lists parallel; rewrite, don't
substitute; end punctuation; errors and buttons; link previews.

Cut: nothing. The hyphen advice ("sideview") has no source for or against; it stays as her
preference. The font sizes and weights line is a design rule, not wording; it stays but could move
to a design section.

## 4. Proposed "User-facing text" section

Superseded 2026-10-24: Anita took a shorter version into her global CLAUDE.md, with a line on her
own voice added, and "please" is fine (her call). This long form stays as the reference behind it.

Written to drop in for the current section of `C:\Users\anita\.claude\CLAUDE.md`. "(new)" marks
what her current section doesn't have; "(changed)" marks a rule that is reworded or widened. The
ids in brackets are what `notes/T-1242.md` refers to.

```markdown
## User-facing text

Style guidelines for anything a player or site visitor reads in her projects: UI labels,
buttons, tooltips, alerts, errors, empty states, status lines, site copy, link previews, map
labels. The aim is text that doesn't read as AI-written. Chat replies, code comments, specs and
commit messages are out of scope.

These are guidelines, not hard rules. If following one makes a sentence clunky, use your
judgement and leave it. When a line breaks one, rewrite the line; don't swap the flagged word or
mark for its nearest substitute (a colon for a dash, "instead of" for "rather than",
"utilise" for "leverage"), which keeps the shape readers notice. (new)

Words
- [W1] **Use the plain word.** (changed) "is", not "serves as", "stands as" or "represents"; "has",
  not "boasts", "features" or "offers"; "use", not "leverage" or "utilise". Words to check for:
  delve, showcase, underscore, highlight (as a verb), crucial, pivotal, key, vital, robust,
  seamless, comprehensive, intricate, tapestry, landscape, testament, foster, enhance, elevate,
  empower, unlock, streamline, dependable, vibrant, bustling, nestled, rich, insights. These are
  the words measured to have spiked since 2023, but each model has its own and the list goes
  stale, so the rule is the plain word, not the list.
- [W2] **No softeners or magic adverbs:** simply, easily, just, quickly, seamlessly,
  effortlessly, quietly, truly, genuinely, actually.
- [W3] **Name the specific thing.** (new) "from the 2010 census", not "from available data";
  "Track removed", not "Changes applied". A wording general enough to fit every case reads as
  evasion.
- [W4] **One word per thing.** (new) Pick one of build/construct, sign in/log in,
  riders/passengers, and use it everywhere. Two words make the reader wonder whether they are two
  things.
- [W5] **Facts over vibes.** Say what the thing is or does. Skip cozy, charming, immersive,
  vibrant and the like.

Sentences
- [S1] **Say what it is, not what it isn't.** (changed) No "not X, but Y", "X rather than Y", "more
  than an X", "not just", "not simply", "no X, no Y, just Z". This family is the most reliable
  tell there is, and models swap one form for another when told to stop. Keep a "not" only where
  it marks a boundary the reader needs ("Visits and shopping, not commutes").
- [S2] **No significance tails.** (new) Don't end a sentence with ", making it...", ", ensuring...",
  ", highlighting...", ", which helps...". Models use these at several times the human rate. If
  the consequence matters, state it as its own fact, ideally with a number.
- [S3] **People act, objects mostly don't.** Prefer the reader or a character in the world as the
  subject, never a click, panel, button, feature or update: "click to open", not "a click opens
  the panel". Watch for "shows", "lets you", "allows you", "can help you", "makes it easier",
  "reveals", "brings", "keeps track of" after an interface noun. (changed: the helper phrases are
  measured tells of current Claude models.) In-world machines are a softer case: "flour is ground
  at the mill" often reads better than "the mill grinds flour", but keep the active version when
  the rewrite gets awkward.
- [S4] **Instructions yes, sales pitches no.** "Click to open" is fine. In a description, state the
  fact ("runs in your browser") instead of inviting ("play in your browser"). No "Discover",
  "Explore", "Imagine", "Ready to...?". (changed)
- [S5] **No hedges or disclaimers.** (new) Cut "It's worth noting", "Note that", "Please note",
  "at this time", and "may" or "can" where the thing simply does. Where a figure is an estimate,
  say once what it is estimated from ("visits estimated from each place's size"), not that it
  "should be treated as indicative".
- [S6] **No assistant voice.** (new) No "Let's", "Great", "Oops", "Hang tight", "We're crunching
  the numbers", "Welcome to ...!". Errors say what happened and what to do, with no apology and no
  "please".

Structure
- [T1] **Lean away from reflex structures:** lists of three by habit, "whether you're X or Y", a
  closing line that sums up, stacked fragments for effect ("Fast. Simple. Yours."), "The X? A Y.",
  announcing what is coming. (changed)
- [T2] **Avoid two-part lines.** A short line split into a label and an add-on ("one row per
  mouse: how it spends its day") or with a second idea tacked on after a comma ("every kind of
  alert, and whether it shows") reads as generated. Say one thing. A label with its value ("Cash
  $4.2M") is fine.
- [T3] **Don't tell the reader what they can see.** (new) A tooltip never repeats its label; no
  line summarises the panel's own counts; a one-time explanation goes in a hover if anywhere.
  Tooltips under about 120 characters where they can be.
- [T4] **Vary free text, keep lists parallel.** (new) A column of labels in one grammar is good.
  A set of descriptions, tooltips or captions where every line has the same shape ("fragment;
  fragment", "X: Y", the same opening word) reads as machine-written even when each line is fine.
  Read a batch together before handing it over.
- [T5] **Buttons name the action.** (new) Verb and object, one to four words: "Delete line", not
  "OK". A confirmation's heading or question names the action ("Delete line?"), not "Are you
  sure?".

Punctuation and type
- [P1] **Avoid dashes between phrases.** Em dashes, en dashes and spaced hyphens: readers react to
  them on sight, even now that current models rarely use them. Rewrite the sentence; a colon or
  semicolon in the dash's place keeps the same two-part rhythm. Number ranges ("5-7") are fine.
  Hyphens inside a word are fine where they help, but drop them where the word reads fine without
  ("sideview" over "side-view"). (changed: reason and the rewrite)
- [P2] **No hype punctuation**, such as exclamation marks. No emoji or decorative symbols (arrows,
  checkmarks, sparkles) in text, unless the symbol is the interface's own icon. (changed)
- [P3] **End punctuation by kind.** (new) No period on a single label, button, heading or status
  fragment; periods when the string is a full sentence or several. Keep it the same for every
  string of one kind.
- [P4] **No all-caps headings or labels.** Section titles, tabs and panel headers go in sentence
  case ("Ridden in Japan", not "RIDDEN IN JAPAN"); no `text-transform: uppercase` with
  letter-spacing as a heading style. Keep the number of distinct font sizes and weights on a page
  to a minimum.

Link previews and page titles (new)
- [L1] The page title is the name. The description says what it is and one fact, in a plain
  sentence or two: no tagline, no invitation, no list of three features.

Before showing a draft, run the checks in trainworld/research/llm-isms.md section 6.
```

## 5. Sources

Read in full or in large part unless marked "summary only".

1. Wikipedia, [Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)
   (WikiProject AI Cleanup), read 2026-10-24, all three parts: word lists by model era, copula
   avoidance, negative parallelism, "Y rather than X", superficial analyses, promotional language,
   rule of three, collaborative communication, hedging disclaimers including mid-2026 corrective
   framings, ineffective indicators, historical indicators (em dashes to September 2026, elegant
   variation).
2. Wikipedia, [talk page](https://en.wikipedia.org/wiki/Wikipedia_talk:Signs_of_AI_writing): the
   "rather than" disclaimer variant, a Claude habit of compressing ideas into abstract nouns
   ("a disputed authorship claim"), "this matters" as a Claude tell, false positives in a 2014
   human text.
3. Kobak et al., [Delving into LLM-assisted writing in biomedical publications through excess
   vocabulary](https://arxiv.org/html/2406.07016v5), Science Advances 2025. 15.1M PubMed abstracts.
4. Juzek and Ward, [Why does ChatGPT "delve" so much?](https://arxiv.org/abs/2412.11385), COLING
   2025, and the [2025 follow-up](https://arxiv.org/abs/2508.01930) (abstracts and summaries only).
5. [Measuring AI "slop" in text](https://arxiv.org/abs/2509.19163), 2025: a taxonomy of what
   readers call slop and which features predict it.
6. Russell, Karpinska and Iyyer, [People who frequently use ChatGPT for writing tasks are accurate
   and robust detectors of AI-generated text](https://arxiv.org/html/2501.15654), ACL 2025.
7. Reinhart et al., [Do LLMs write like humans? Variation in grammatical and rhetorical
   styles](https://arxiv.org/html/2410.16107v2), PNAS 2025 (doi 10.1073/pnas.2422455122).
8. silentbob, [LLM style slop is absolutely
   everywhere](https://www.greaterwrong.com/posts/yBM2rQ6AJY6MoRGFQ/llm-style-slop-is-absolutely-everywhere),
   LessWrong, with comments.
9. Graphite, "AI Tells" (2026-09-16) and "AI Tells: Opus 5.5 Update" (2026-10-01), through a
   [recheck by Pebblous](https://blog.pebblous.ai/report/opus-5-5-em-dash-ai-tells-drift/en/) and
   [press coverage](https://www.winzheng.com/en/article/opus-5-5-ai-writing-tells). A company's own
   research, not peer reviewed; data released CC BY 4.0. Summary only.
10. Sam Kriss, "Why Does A.I. Write Like ... That?", New York Times Magazine, 2025-12-03. Summary
    only: [Longreads](https://longreads.com/2025/12/04/why-does-a-i-write-like-that) and press
    coverage; the full text was paywalled or blocked.
11. Google developer documentation style guide, [Voice and tone](https://developers.google.com/style/tone).
12. [tropes.fyi](https://tropes.fyi/), a catalogue of AI writing tropes (one person's list, posted
    to Hacker News).
13. Jill Walker Rettberg, [Genre glitches and unexpected promotional phrases as a sign of AI
    writing](https://jilltxt.net/genre-glitches-and-unexpected-promotional-phrases-as-a-sign-of-ai-writing/),
    2026.
14. GOV.UK, [A to Z style guide](https://guidance.publishing.service.gov.uk/writing-to-gov-uk-standards/style-guides/a-to-z-style-guide/),
    "Words to avoid", contractions.
15. Microsoft Writing Style Guide, [Top 10 tips](https://learn.microsoft.com/en-us/style-guide/top-10-tips-style-voice).
16. Mailchimp content style guide, [Writing for accessibility](https://styleguide.mailchimp.com/writing-for-accessibility/).
17. Editors' checklists on AI drafts: [Proofed](https://proofed.com/knowledge-hub/ai-editing-checklist-how-to-spot-and-fix-ai-writing-patterns/)
    and a DropTimes piece, "A humanizer should not become a house style manual". Summary only (both
    blocked the fetch).
18. Grafana Writers' Toolkit, [UX writing](https://grafana.com/docs/writers-toolkit/write/style-guide/ux-writing/).
19. Microcopy Examples, [Avoid "landing page words"](https://microcopyexamples.substack.com/p/avoid-landing-page-words). Summary only.
20. Yakura et al., [Empirical evidence of large language models' influence on human spoken
    communication](https://arxiv.org/abs/2409.01754): "delve" and similar words rose in podcasts
    and YouTube talks after ChatGPT's release, then fell back in several podcast categories.
    Abstract and summaries only.
21. [explainx, Claudisms](https://explainx.ai/blog/claude-opus-5-load-bearing-claudisms-writing-tells-2026),
    2026, and the "load-bearing" bug reports. Blog, single source.
22. PatternFly, [Tooltips](https://patternfly.org/ux-writing/tooltips). Summary only.
23. Unity Foundations, [Tooltip](https://foundations.unity.com/components/tooltip). Summary only.
24. Trust penalty: [Raptive study coverage](https://ppc.land/raptive-study-shows-ai-content-cuts-reader-trust-by-half/)
    (industry-commissioned) and a preregistered seven-experiment study reported by Newswise.
    Summary only.
25. Hacker News, [Wikipedia: Signs of AI Writing](https://news.ycombinator.com/item?id=44700558).

Searched and not used: Language Log (no post on "delve" found), Nielsen Norman Group pages (not
returned by search), Apple's Human Interface Guidelines writing page (did not load), Material
Design's UX writing page (did not load).

## 6. Checks to run on a draft

Run these on the list of strings before showing it. Most are a grep; the regexes are for ripgrep
(`rg -n -i`) over the strings or files.

1. **Dashes.** `—|–| - [a-z]`. Each hit: rewrite the sentence, not just the mark.
2. **Word list.** `\b(delve|showcas|underscor|crucial|pivotal|vital|robust|seamless|comprehensive|intricate|tapestry|landscape|testament|foster|enhanc|elevat|empower|unlock|streamlin|dependable|vibrant|bustling|nestled|insights?|leverag|utilis|serves as|stands as|boasts)\w*`
3. **Softeners and hedges.** `\b(simply|just|easily|quickly|seamlessly|effortlessly|quietly|truly|genuinely|actually|please|worth noting|note that|at this time)\b`
4. **Negative parallelism.** `\b(not just|not only|not simply|rather than|more than an?|instead of|it's not)\b` and lines matching `\bnot\b.*,.*\b(it's|but)\b`. Keep only real boundaries.
5. **Significance tails.** `, (making|ensuring|highlighting|underscoring|allowing|helping|letting|giving|contributing|reflecting)\b`.
6. **Interface nouns acting.** `\b(panel|tab|button|view|map|tool|feature|page|screen|list|chart)s? (shows|lets|allows|helps|makes|displays|reveals|brings|keeps)\b`, and `\b(can help|makes it easier|lets you|allows you)\b`.
7. **Two-part lines.** Short strings (under about 60 characters) with `: ` or `; ` in the middle,
   and strings ending `, and <clause>`. Is it a label with its value (fine) or a label with an
   explanation (rewrite)?
8. **Hype.** `!` in any string; emoji; `→`, `✓`, `✨`.
9. **Tooltips.** For each: does it repeat its label or aria-label? Over 120 characters?
10. **Same shape.** Read every free-text line of the batch together, sorted by first word. Many
    lines with the same opening, the same "X: Y" or "fragment; fragment" shape: vary them.
11. **One word per thing.** List the nouns and verbs for each concept (build/construct, sign in/log
    in, rider/passenger) across the batch and pick one.
12. **The plain-word test.** Replace each verb with "is" or "has" and each claim with the number
    behind it. If the sentence still says the same, use that version.
