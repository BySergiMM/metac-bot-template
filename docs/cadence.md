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

**This mitigation is UNVERIFIED.** The baseline above is recorded precisely so
the change can be falsified: re-measure after several days and revert if the
distribution has not moved. Do not report it as a fix until it is measured.

## What would actually give ~5-minute detection

Cron alone cannot, because delivery is upstream. Three options, none adopted:

| option | detection | cost / risk |
| --- | --- | --- |
| **Internal poll loop** — one scheduled run per hour that loops internally, sleeping ~5 min between passes | genuine ~5 min, independent of GitHub's scheduler | ~50 runner-min/hour instead of ~1. Free on a public repo, but a large step up in consumption, and a long-lived runner is a new failure mode. Needs validation on `bot-testing-area` first. |
| **Self-re-dispatch** — the run triggers the next one via the API | ~5 min | Requires `actions: write`, i.e. a permission escalation on the one workflow that holds the Metaculus token. Rejected on that basis alone. |
| **External scheduler** calling `workflow_dispatch` | ~5 min | New infrastructure to run, monitor and secure, holding a GitHub token. Not justified by the size of the problem. |

## How much this actually costs

Coverage, not compliance. Nothing here risks disqualification.

MiniBench forecast windows have been measured at 1.5–3 h wide (418/418
questions, none shorter). Against a 32-minute median gap the bot has ample
margin on a typical question. The exposure is the tail: a p90 gap of 55.7 min
and a worst observed gap of 109.3 min can consume most of, or exceed, a
90-minute window.

Missed questions are a scoring loss on those questions, not a rules breach —
Metaculus requires a comment on each question the bot *forecasts*, not that it
forecast every question.

## What to watch

* `discovery_complete ... questions=N` on each run — a sudden jump means a
  burst arrived and the gap that preceded it is the one that mattered.
* Re-run the gap measurement periodically; the numbers above are the baseline
  to compare against.
