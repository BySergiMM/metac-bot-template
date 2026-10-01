# Detection cadence (R8)

How quickly the bot notices a newly opened question, why it is slower than the
cron says, and what can and cannot be done about it.

## The measurement

200 most recent runs of `run_bot_on_tournament.yaml`,
2026-08-18T21:59Z .. 2026-08-23T21:31Z, declared schedule `*/5 * * * *`
(12 requested events per hour):

| statistic | value |
| --- | --- |
| gap, minimum | 16.6 min |
| gap, p25 | 23.6 min |
| gap, **median** | **32.2 min** |
| gap, p75 | 41.5 min |
| gap, p90 | 55.7 min |
| gap, maximum | 109.3 min |
| gap, mean | 36.0 min |
| gaps ≤ 6 min | 0 of 199 |
| gaps ≤ 10 min | 0 of 199 |

Effective delivery: about 1.7 runs/hour against 12 requested, ≈14%.

## The cause

Not this repository. Three independent measurements rule out every local
explanation:

* **Queue delay was 0 s on all 200 runs** (`startedAt - createdAt`). The
  `concurrency` group is not holding runs back, and runners are not scarce.
* **Run duration is 62 s median, 292 s max.** Runs are not overlapping into
  one another, and even the longest is far shorter than the shortest gap.
* Events simply never arrive. The gap distribution is what dropped events look
  like, not what delayed events look like — a delayed event still arrives.

GitHub documents this exact behaviour for the `schedule` event:

> The `schedule` event can be delayed during periods of high loads of GitHub
> Actions workflow runs. High load times include the start of every hour. If
> the load is sufficiently high enough, some queued jobs may be dropped.

There is no delivery guarantee, and no repository-side setting that raises the
delivery rate.

## What was changed

The cron's **phase**, which is the one lever GitHub actually offers — it
recommends "scheduling workflows at different times within the hour rather
than at the start".

```diff
- - cron: "*/5 * * * *"                                     # :00 :05 :10 ...
+ - cron: "2,7,12,17,22,27,32,37,42,47,52,57 * * * *"       # :02 :07 :12 ...
```

Same 12 requests per hour, same 5-minute spacing, moved off the `:00`/`:05`
grid that every other `*/5` and `*/15` cron on the platform also asks for.

**This mitigation was UNVERIFIED, and has now been falsified.**

## Re-measurement, 2026-09-07

The baseline above was recorded precisely so the change could be tested. It
was, over the 200 most recent runs (2026-08-23T01:46Z .. 2026-09-07T06:05Z):

| window | gap median | p75 | max | delivery |
| --- | ---: | ---: | ---: | ---: |
| 18–23 Aug (`*/5`, the baseline) | 32.2 min | 41.5 | 109.3 | **14%** |
| 23 Aug – 1 Sep (phase-shifted) | 41.1 min | 60.0 | 741.4 | **5.9%** |
| since 1 Sep (phase-shifted) | **174.6 min** | 267.6 | 321.5 | **2.6%** |

The distribution moved, in the wrong direction, by roughly 5x. Whether the
phase shift caused that or merely failed to prevent a platform-wide
degradation cannot be separated from this data — but either way the phase is
not the lever, and the honest reading is that cron cannot be tuned into
working.

The cron is kept as-is: `*/5` is already GitHub's documented ceiling ("the
shortest interval you can run scheduled workflows is once every 5 minutes"),
so there is nothing above it to ask for, and reverting the phase would trade
one unverified guess for another.

The concrete cost, observed: on the night of 2026-09-06/07 MiniBench opened a
batch of 2 questions at 01:07Z and a batch of 4 at 06:07Z. The gap between the
two runs was **299 minutes**. The bot caught both batches by luck, not by
coverage; questions 45518/45519 had already closed by the time it looked again.

## What was changed instead: an internal poll loop

Option 1 from the table below, which this document had already identified as
the only one that does not require a permission escalation or new
infrastructure. `run_bot_on_tournament.yaml`'s "Run bot" step now polls for
`POLL_WINDOW_SECONDS` (3 h) at `POLL_INTERVAL_SECONDS` (5 min) instead of
running once and exiting.

The schedule is unchanged and still mostly dropped. What changed is what one
*delivered* event buys: 3 hours of genuine 5-minute detection instead of ~60
seconds. Since the median gap between delivered events is 2.9 h, runs now hand
over to one another most of the time.

Sizing is deliberate: the window is about one median gap, not the 6 h job
ceiling. Enough for near-continuous coverage, without turning a scheduled job
into a permanently resident process.

Failure semantics, because a loop makes them non-obvious:

* a poll that fails does **not** abort the window — one bad question or one
  provider blip must not blind the bot for the remaining hours, which would be
  strictly worse than the single shot this replaced;
* a window in which **every** poll failed exits non-zero, so a systemic break
  (dead token, all providers down) still fails the job and still fires the
  WhatsApp failure notifier.

Both are pinned by tests in `tests/test_production_invariants.py`, along with
the window fitting inside the step timeout and the timeout staying under
GitHub's 6 h ceiling.

**Validation, and its limit.** The loop script was extracted from the workflow
and exercised against a stub bot for all three cases (clean, all-fail,
intermittent), confirming the exit codes and that the window is respected. It
had **not** been observed across a full window in CI before shipping.

## The 3-hour window was wrong, and it cost the Actions allowance

Re-measured 2026-09-12, after two days in production. The account ran out of
GitHub Actions for roughly three weeks.

Observed run durations, back to back:

    3h13m  3h00m  3h01m  3h32m  5h13m  4h10m  3h17m  3h00m  3h01m  3h00m

That is ~24 h of runner time per day. The mechanism was in the workflow the
whole time and this document failed to reason about it: `cancel-in-progress:
false` **queues** the scheduled events that arrive during a run, so the moment
a 3 h run ends, the queued one starts. "Runs hand over to one another" — the
stated goal — and "a permanently resident process" — the stated thing to avoid
— are the same outcome. The note above claimed the sizing avoided it; it
guaranteed it.

What the occupancy bought was close to nothing. Tournament 33022 has had no
open questions for weeks and MiniBench opens a batch roughly every two weeks,
so the overwhelming majority of polls discovered **zero** questions. Paying
continuous occupancy to catch a bi-weekly event is the wrong trade at any
detection latency.

### What it is now

| | before | after |
| --- | --- | --- |
| window | 3 h | **40 min** |
| early exit | none | after **2** consecutive polls finding 0 open questions |
| step timeout | 215 min | 55 min |

The early exit matters more than the window. When the tournaments are quiet a
run now costs about two polls; when a batch is genuinely open the loop keeps
its 5-minute granularity for the whole window, which is the only circumstance
in which that granularity was ever worth paying for.

A run that dies *before* discovery does not count as idle — it says nothing
about whether the tournaments are quiet, and treating it as idle would end the
window on exactly the runs that deserve another attempt.

`tests/test_production_invariants.py` now caps the window at one hour, requires
the early exit, and pins the `discovery_complete ... questions=<n>` marker the
loop reads, so renaming it in `discovery.py` fails loudly rather than silently
restoring continuous polling.

### The lesson worth keeping

The cost of a scheduled workflow is not the cost of one run. It is one run
times how often the platform will start another, and a concurrency group that
queues rather than drops turns "occasionally" into "always".

## Re-measurement, 2026-10-01: about six runs a day

How often GitHub actually started `run_bot_on_tournament.yaml`, over its whole
history: 568 runs, 2026-08-17T09:06Z .. 2026-10-01T02:10Z (566 by `schedule`,
2 by `workflow_dispatch`).

Method. Read from the Actions API on 2026-10-01T07:20Z, every page until the
last (`total_count` 568; 568 runs read, no duplicate ids):

```
GET /repos/BySergiMM/metac-bot-template/actions/workflows/run_bot_on_tournament.yaml/runs?per_page=100&page=1..6
```

A day is a UTC calendar day of the run's `created_at`. A gap is the time between
the `created_at` of consecutive runs. Percentiles are linear interpolations. The
cron requests 12 events an hour, 288 a day. The same computation over 18-23 Aug
gives 39.3 runs a day and a median gap of 32.2 min, which is the baseline at the
top of this document, so it is the same measure.

| window | runs per day, mean (min–max) | delivered of 288 | gap median | gap p90 | gap max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1–30 Sep (30 days) | 6.6 (4–9) | 2.3% | 208.5 min | 323.1 min | 471.2 min |
| 17–30 Sep (the last 14 full days) | 6.0 (4–8) | 2.1% | 240.6 min | 341.5 min | 471.2 min |
| 24–30 Sep (7 days) | 5.4 (4–6) | 1.9% | 291.3 min | 360.4 min | 471.2 min |

Runs per day, 17 to 30 Sep: 6, 7, 8, 7, 5, 7, 6, 6, 6, 6, 6, 4, 5, 5 (84 runs).

So the scheduled job starts about six times a day, and four or five over the
last days: not every 5 minutes as the cron asks (about 2% of the 288 events a
day are delivered), and not every 20 minutes as the README inherited from the
upstream template said. A run is short, median 6.2 min over 1–30 Sep
(`updated_at - run_started_at`, an approximation), which is consistent with the
early exit after two empty polls.

What that means for coverage is arithmetic on numbers already in this
document, not a new measurement: the median gap between delivered runs over the
last 14 days is about 4 hours, longer than the 1.5–3 h MiniBench forecast
windows quoted below, so a question that opens just after a run can close
before the next one starts. A green run says the job was delivered, not that the
tournaments are being watched continuously.

## What would actually give ~5-minute detection

Cron alone cannot, because delivery is upstream. Three options:

| option | detection | cost / risk |
| --- | --- | --- |
| **Internal poll loop** — ADOPTED 2026-09-07, see above | genuine ~5 min while a run is alive | Runner minutes are free on a public repo. A long-lived runner is a new failure mode, which is why the failure semantics are pinned by tests. |
| **Self-re-dispatch** — the run triggers the next one via the API | ~5 min | Requires `actions: write`, i.e. a permission escalation on the one workflow that holds the Metaculus token. Rejected on that basis alone. |
| **External scheduler** calling `workflow_dispatch` | ~5 min | New infrastructure to run, monitor and secure, holding a GitHub token. Not justified by the size of the problem. |

## How much this actually costs

Coverage, not compliance. Nothing here risks disqualification.

MiniBench forecast windows have been measured at 1.5–3 h wide (418/418
questions, none shorter). Against the 32-minute median gap of 18–23 Aug the bot
had ample margin on a typical question, and the exposure was the tail: a p90
gap of 55.7 min and a worst observed gap of 109.3 min could consume most of, or
exceed, a 90-minute window. Against the median gap of about 4 hours measured on
2026-10-01 (above) that margin is gone: the median gap is longer than the
windows, so a question can open and close between two delivered runs.

Missed questions are a scoring loss on those questions, not a rules breach —
Metaculus requires a comment on each question the bot *forecasts*, not that it
forecast every question.

## What to watch

* `discovery_complete ... questions=N` on each run — a sudden jump means a
  burst arrived and the gap that preceded it is the one that mattered.
* Re-run the gap measurement periodically; the numbers above are the baseline
  to compare against.
