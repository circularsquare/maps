# Replies to the four open PRs, 2026-11-16

Drafts for Anita to read. Nothing has been posted.

## #7, peanutmow: reuse scratch buffers for profile builds

Decision: taken, with credit; already in the main repo, reaches players with the next game publish (v0.4.0).

Draft reply:

> Thanks, this one is in. A 240-line network now loads in about half the time here too, and the figures come out identical. We kept your 240-line save as a benchmark for load and edit times, so thank you for attaching it. You are credited for it, and it will reach players with the next release.
>
> We are closing the PR because the change went into our main copy directly, and the open-source repo gets it with that release.

## #1, Renkon: Make bus routes follow one-way streets

Decision (revised 2026-11-16): taken as a setting, off by default, at low priority (T-3480). A both-ways or looping line still drives a one-way street backwards, which is why it stays an option.

Draft reply:

> Thanks for this, and for the write-up and the tests. The search change is clean and the arrows are easy to read.
>
> We'd like to take it as a setting that is off by default. Buses in the game are settled and we want the default to stay simple, and some players will want buses to keep to one-way streets. A line set to both ways, or a loop, still drives some one-way streets backwards, so it fits better as an option for now.
>
> It's on our list at low priority. We'll bring it into our main copy with you credited, and close this PR once it reaches the open-source repo.

## #12, CinnamonVII: Combine partial line service, merge lane upgrade, demand CPU/WebGPU optimizations, and MSAA stroke fix

Decision: partly built lines taken as a low-priority task (no date); demand CPU speedups being measured; merge lane upgrade and WebGPU left out; MSAA fix already done here; small fixes part not decided (see ask).

Draft reply:

> Thanks for the amount of work in here, and for folding your earlier two PRs into one with tests. Part by part, so you know where each landed.
>
> Partly built lines (from #3): we want this. A line running its built stretch while the rest stays planned fits how the game plays. It is on our list at low priority, so there is no date, and it will not land from this PR as it is: the files it touches have changed a lot, so we will build it on current code. You will be credited when it lands.
>
> Demand CPU speedups: we are measuring these on our own saves now. The parts that keep every sum in the same order and show a real gain are the ones we are looking to take, with credit. Anything that changes figures, even in the last bits, or that adds threads, needs a clear gain before it goes in. We will not know which parts make it until the timings are in.
>
> Merge lane upgrade: we are leaving this out. It adds a new thing to build, and the 0.6 occupancy it rests on has no source behind it that we could point to, so we would rather not add a capacity rule we cannot back up.
>
> WebGPU demand mode: we are leaving this out too. It is a second copy of the mode choice, about 6,400 lines, that would have to be kept in step with every change to the demand rules, and the GPU results only match the CPU within a tolerance, so figures and scores would differ from one machine to the next.
>
> Stroke fix past zoom 14: we fixed the same problem ourselves today by giving each stroke cache layer its own multisampled buffer, so there is nothing to take. Thank you for finding it.
>
> The one-line debug check in the mode choice code, for when the walk rule is off, we will take, with credit.
>
> We are closing this PR, since it is too big to land as one piece. If you want to send any of these parts again on their own, a smaller PR would be easier to review.

## #13, GabrielMoraez: Weekend and holiday train counts per line

Decision: not merged; its sim parts are reused when Detailed-mode weekends are built (T-3346 stage 4). Main reasons: it puts weekend counts on the line panel and gives holidays their own counts, against the agreed design, and could triple a London settle time.

Draft reply:

> Thanks for this. The PR is thorough, the tests cover a lot, and your note that it changes an earlier rule and that you would rework it is appreciated.
>
> We are not going to merge it as it is. Weekend schedules are planned for the schedule panel in Detailed mode, where weekends can run the same as weekdays, scaled, or custom, and holidays run as weekends. Your version puts separate weekend counts on the line panel and gives holidays a third set of their own, which goes against that plan. It also solves each kind of day on its own, which on a very large network such as London, where one solve already takes several seconds, could multiply an edit's wait by up to three.
>
> We do want the sim side. The per-line calendar, laying departures by each day's kind, crowding and running cost per kind of day, fleet sizing to the busiest day and the quad-track handling are what we need for that panel, and we plan to reuse them when it is built, with you credited as co-author.
>
> We are closing this PR on that basis.
