# Release findings: lower findings left for the release review, one line each (task id, the line, what breaks); cleared after each publish
- T-4408: Try again's 10 s fallback reload could cut a very large save mid-write; check that a slot write is one IndexedDB transaction (else a slot that won't load). Also low: a slot switch clicked with Try again loses to the reload; saveThen (oom retry) has no timeout.
- T-4510: one of two timed lines released alone, or a pair timed later, averages that pair on weekends only (saves fine).
- T-4510: on a weekday-pattern line with a weekday offset, the weekend row shows its own step, not what runs (weekday +90 s, weekend +30 s shows +30, runs +120).
- T-4510: a build from before T-4510 misreads a save with weekend holds instead of refusing it (rollback only).
