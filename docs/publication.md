# Publication (R1, R3, R6)

What guarantees the bot makes when it publishes a forecast, what it still does
not guarantee, and why.

## The requirement

Metaculus' AI Forecasting Benchmark rules:

> Your bot must autonomously generate forecasts to questions included in the
> Tournament as well as written comments accompanying each forecast.

> In order to be eligible for the prize, the participating bot needs to have
> written a comment response (including a display of its forecast) under each
> question.

So a forecast without its comment is an **eligibility failure**, not a
cosmetic one.

## What upstream does

`forecasting_tools` 0.2.90, `binary_report.py:66-71` and the three sibling
report types:

```python
metaculus_client.post_binary_question_prediction(id_of_question, self.prediction)
metaculus_client.post_question_comment(id_of_post, self.explanation)
```

Two independent POSTs. Each wrapped in
`@retry_with_exponential_backoff(max_retries=3)`, which retries on
`RequestException`, `Timeout` and `ConnectionError`. No transaction, no
idempotency key, no reconciliation.

Three defects follow.

### R1 — a forecast can be left permanently without its comment

```
prediction POST succeeds        → my_forecasts.history is now non-empty
comment POST fails 4x           → exception propagates, report becomes an error
next run                        → already_forecasted == True   (questions.py:138)
                                → forecast_questions drops it  (forecast_bot.py:233)
                                → the comment is never written, ever
```

The dedup key is derived from the **prediction**, so the very thing that
succeeded is what prevents the repair.

### R3 — a retried POST can duplicate

Server accepts the POST, the response is lost, the client raises `Timeout`, the
decorator retries. For a comment that is visible duplicate content on the post.

### R6 — group posts get one comment per subquestion

`_unpack_group_question` (`metaculus_client.py:685`) deep-copies the parent
post once per subquestion, so N subquestions share one `id_of_post` while
having distinct `id_of_question`. The prediction goes to the question, the
comment goes to the **post**, and `forecast_questions` runs every question
concurrently through a bare `asyncio.gather`. N comments land on one post.

## What `publication.PublishingClient` does

Injected via `ForecastBot(metaculus_client=...)` (`forecast_bot.py:89`, used at
`:166` for discovery and `:416` for publication). The client is the seam
because `_run_individual_question` offers no hook around publication, and the
report type — not the caller — fixes the order of the two POSTs. Both POSTs
are client methods, so overriding them covers every question type, every
report class, and any future upstream refactor of the calling code.

| defect | mitigation | strength |
| --- | --- | --- |
| R1 | comment gets its own retry budget (`COMMENT_ATTEMPTS = 3`, each already containing the SDK's 3) — up to 9 chances instead of 3 | **reduced, not eliminated** |
| R1 | on permanent failure the question enters `ORPHANED` and logs `PUBLICATION_ORPHAN post_id=… question_ids=… comment_attempts=…` | detection is now certain |
| R1 | `print_publication_report` prints every orphan on stdout at end of run and returns the count | detection is now certain |
| R3 | one prediction per `question_id`, one comment per `post_id`, per process | **in-run only** |
| R6 | one comment per `post_id` regardless of how many subquestions publish | **fixed** |

`question_id → post_id` is recorded during discovery, which the same client
performs, so the mapping is known by publication time without threading it
through upstream code.

## What is still not guaranteed

**R1 is reduced, not eliminated.** If all nine comment attempts fail, the
forecast is live with no comment and the bot will not retry it on a later run,
because `already_forecasted` is now true. Recovery is an operator action.

**R3 is in-run only.** A duplicate arising from a lost response *across* runs
is still possible; cross-run duplicate prevention is `already_forecasted`,
which is upstream's mechanism and unchanged.

### Why the order was not reversed

Publishing the comment first would make R1 structural rather than
probabilistic: the prediction would only ever be written after the comment
succeeded, so "a prediction exists" would imply "a comment exists", and
`already_forecasted` would become a correct completion marker. The residual
failure inverts into a self-healing one — comment written, prediction not,
question still `already_forecasted == False`, so the next run retries it in
full, at the cost of one duplicate comment.

It was not done, for two reasons:

1. The order is fixed inside each report type's `publish_report_to_metaculus`.
   Changing it means reimplementing publication for four report types — a far
   larger blast radius than the bug.
2. `post_question_comment` defaults to `included_forecast=True`, and whether
   Metaculus accepts that on a question the author has not yet forecast could
   not be verified without production credentials. Putting an unverified
   server interaction into the critical publication path — when the current
   order has published successfully in production three times out of three
   (runs 32239144510, 32268972325, 32366841649) — trades a rare, detectable
   failure for an unmeasured one.

Note the comment **body** carries `*Final Prediction*: …`
(`forecast_bot._create_comment`), so the rules' "display of its forecast" does
not depend on the `included_forecast` widget. The blocker is server behaviour,
not the requirement.

### Why there is no automatic cross-run reconciliation

It would need to answer "does this post already carry my comment?".
`MetaculusClient` has `post_question_comment` and **no comment read method**,
and whether a `GET /api/comments/` endpoint exists could not be established:
an unauthenticated probe returned `403` for that path *and* for a deliberately
nonexistent one, so the result carries no information.

Building reconciliation on an unverified endpoint, in the publication path,
would be worse than the bug. The alternative — persisting an orphan ledger on
the runner — is rejected because GitHub runners are ephemeral, and writing it
back to the repository would require `contents: write` on the one workflow
holding the Metaculus token.

## Operator procedure when `PUBLICATION_ORPHAN` appears

1. The run's stdout lists `post_id`, `question_id` and `comment_attempts`.
2. Open the post and confirm the forecast is present and the comment is not.
3. Post the comment manually, before the question closes. The forecast itself
   is already correct and must not be re-posted.
4. If it recurs more than once in a season, revisit the ordering decision
   above — validate `included_forecast` behaviour on `bot-testing-area` first.

## Tests

`tests/test_publication.py` — 27 cases driving the real client with only the
two network methods replaced: happy path, prediction failure, prediction
timeout, comment failure, comment timeout, comment retry succeeding, comment
retries permanently failing, orphan marker content, duplicate suppression on
both paths, group posts with 1/2/N/concurrent subquestions, rerun after
partial publication, and process restart.
