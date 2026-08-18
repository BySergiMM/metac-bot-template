# Track record: measuring what the competition measures

This document describes the offline research lab in `research/`. It exists to
answer one question before we change anything about the forecaster:

> Can we measure whether a change actually improved out-of-sample performance?

The Phase 0 audit concluded we could not. `backtest/minibench_backtest.py`
never runs the forecaster, has zero resolution labels, no train/test split, and
scores Brier while the tournament scores spot peer. This lab is the first
instrument that measures the real thing on real data.

**Nothing here touches production.** No module imports `main.py`. The API
client only issues GET and refuses any other verb at the call site. No
forecast is ever produced or submitted.

---

## 1. How to obtain the dataset

### Prerequisites

A Metaculus API token for the bot account, from
<https://www.metaculus.com/accounts/settings/account/>. The lab needs **no**
other dependency — it is standard library only, Python 3.9+, no virtualenv, no
`poetry install`.

```bash
export METACULUS_TOKEN='...'          # or put it in .env, or use --token-file
python3 research/fetch_own_track_record.py --check    # auth probe, writes nothing
python3 research/fetch_own_track_record.py            # build a dataset
python3 research/analyze_track_record.py              # report on the newest one
```

### What the fetch does

| Step | Endpoint | Purpose |
|---|---|---|
| 1 | `GET /api/users/me/` | identify our own account id |
| 2 | `GET /api/get-data-access-status/` | record which data tier this token has |
| 3 | `GET /api/posts/?forecaster_id=<us>&statuses=resolved,closed` | every post we have forecast on |
| 4 | `GET /api/posts/?tournaments=<id>` | the question universe, for coverage |
| 5 | `GET /api/data/download/?post_ids=...` | a ZIP of CSVs, in chunks of 50 posts |

### Why this is allowed

Metaculus grants every account unrestricted access to its own data, and
resolution values for *"every question it has forecasted at least once"*
([API docs](https://www.metaculus.com/api/), "All Authenticated Accounts"). The
server enforces this independently of us — in
`utils/csv_utils.py::export_data_for_questions`:

```python
if not (has_data_access or is_staff):
    user_forecasts = user_forecasts.filter(author=user)
```

so another forecaster's rows cannot reach us even if we asked for them. We
deliberately never send `user_ids` (the serializer rejects it for accounts
without the data tier, and the server already scopes the export to us).

### Competition-rules safety

Reading our own already-submitted forecasts on already-closed questions is the
sanctioned use — Metaculus offers this data so bot makers can perform
*"retrospective assessment of performance"*. The tournament rules prohibit
*"previewing bot forecasts on open questions, then updating the bot based on
results"*.

Accordingly the fetch defaults to **closed questions only**. `--include-open`
exists but prints a warning and should stay off; using it and then changing the
bot is exactly the human-in-the-loop pattern that is banned.

---

## 2. What a dataset contains

Datasets are **immutable directories** under `research/datasets/`, named by
their own id:

```
research/datasets/track-record-20260819T081500Z-1a2b3c4d/
├── manifest.json              provenance: hashes, source, git commit, limits
├── question_data.csv          one row per question
├── forecast_data.csv          one row per forecast interval (ours + aggregates)
├── score_data.csv             Metaculus' own scores for our account
├── tournament_questions.json  the question universe, for the coverage denominator
├── account.json               which account this is, and its data tier
└── fetch_log.json             every API call and every failure
```

The id embeds a UTC timestamp and the first 8 hex of a digest over all file
hashes, so two fetches of unchanged data are visibly the same data.
`write_manifest` refuses to overwrite an existing manifest; a re-fetch always
creates a new directory. `--verify DIR` re-hashes everything.

Column names come from the Metaculus server's own exporter
(`utils/csv_utils.py::generate_data`), transcribed rather than inferred. If
Metaculus renames a column the reader raises with the missing name instead of
silently producing `None` everywhere — the failure mode the old backtest
harness spent four commits chasing.

The columns that matter most:

- `question_data.csv` → `Open Time`, `CP Reveal Time`, `Scheduled Close Time`,
  `Actual Close Time` (these determine the spot instant), `Resolution`,
  `Question Weight`.
- `forecast_data.csv` → `Start Time` / `End Time` (the forecast's validity
  interval), and **`Probability of Resolution`**: the probability mass our
  forecast placed on the outcome that actually happened. The server's README
  calls it *"the value used in scoring"*. It is defined for every question
  type, which is why the lab can score numeric and multiple-choice questions
  without reimplementing CDF integration.
- `score_data.csv` → `Score Type` (`spot_peer`, `peer`, `baseline`,
  `spot_baseline`), `Score`, `Coverage`. This is the ground truth.

---

## 3. What we can and cannot reproduce

The tournament metric, transcribed from
`scoring/score_math.py::evaluate_forecasts_peer_spot_forecast`:

```
spot_peer = 100 * (N / (N - 1)) * ln(p / gmp)      # halved for continuous types
coverage  = 1.0 if a forecast of ours was live at the spot instant else 0.0
```

- `p` — our probability on the resolved outcome. **We have this.**
- `gmp` — the same quantity for the geometric mean of every eligible
  forecaster. **We do not have this.**
- `N` — the number of forecasters in that geometric mean. **We do not have this.**

and the spot instant, from `questions/models.py::get_spot_scoring_time`:

```
spot = question.spot_scoring_time
    or (cp_reveal_time if cp_reveal_time > open_time)
    or actual_close_time
    or scheduled_close_time
spot_timestamp = min(spot, actual_close_time)
```

### Tier table

| Quantity | Tier | Notes |
|---|---|---|
| Log score on the resolved outcome | **EXACT** | `ln(p)`, all question types |
| Coverage at the spot instant | **EXACT** | validated against Metaculus' `Coverage` column |
| Spot peer for a forfeited question | **EXACT** | it is 0 by definition |
| Brier | **EXACT**, binary only | secondary metric; deliberately not generalised |
| Spot peer, full | **EXACT if** the dataset has `geometric_mean` aggregate rows, otherwise **UNAVAILABLE** | never approximated silently |
| Tournament total | EXACT only if every question was exact | a total over a subset is not the total |
| Forfeited points | **PROXY** | realised mean × missed questions |

The lab **never** substitutes a proxy for an unavailable exact value. A missing
input is reported as `UNAVAILABLE` with the missing term named.

### The one open possibility

`/api/data/download/` accepts `aggregation_methods`, and `geometric_mean` is an
accepted value (`utils/serializers.py::DataPostRequestSerializer`). For
questions that are already resolved, `export_data_for_questions` puts them in
`questions_with_revealed_cp` even for an ordinary account:

```python
questions_with_revealed_cp = questions.filter(
    Q(resolution__isnull=False) | Q(cp_reveal_time__isnull=True)
    | Q(cp_reveal_time__lte=timezone.now())
)
```

If that path survives the API gateway's aggregation restrictions, the exact
denominator is retrievable for resolved questions and **the competition metric
becomes exactly reproducible offline**. The fetch script therefore *attempts*
the `geometric_mean` request first and falls back cleanly, recording the exact
failure in `fetch_log.json`.

This is a **hypothesis, not a claim**. It has not been tested against a real
token. Do not repeat it as fact until `manifest.json` shows
`aggregation_methods_present: ["geometric_mean"]`.

### If it fails: the valid proxy

Peer score is relative, so ranking two of our own variants needs a *reference
population*, not the real one. The scientifically defensible substitute is a
frozen reference panel:

```
peer_proxy(i) = ln(p_ours,i) - mean_j[ ln(p_reference_j,i) ]
```

with 3–5 baseline configurations held fixed across all comparisons. Its
limitations, which must be restated in every report that uses it:

- the absolute level is not comparable to the leaderboard;
- only *differences* between variants are informative, and only while the panel
  is unchanged;
- a stale panel drifts, so "we beat the panel" degrades into "the panel got old".

---

## 4. How the analysis is validated

`research/validate.py` runs two independent checks.

**1. Coverage reproduction — the strong test.** For every question Metaculus
scored, compare our derived "was a forecast of ours live at the spot instant?"
against Metaculus' own `Coverage` column. This is exact, binary and falsifiable,
it tests the spot instant *and* the forecast-selection rule, and — crucially —
it works even when the peer denominator is unavailable. If this check passes at
100%, the timing half of the reconstruction is verified.

**2. Inversion diagnostic — falsifiability without the denominator.** Given
Metaculus' published score `S` and our own `p`, the peer aggregate is pinned:

```
gmp = p * exp(-S * k / (100 * N/(N-1)))        k = 2 for continuous
```

Every implied value must be a probability in `(0, 1]`. A value outside that
range *proves* something upstream is wrong. Evaluated in the large-N limit
(factor → 1), so the recovered value is slightly biased low; it is a necessary
condition, not a sufficient one.

When the denominator *is* available, check 2 is replaced by a direct numeric
comparison: MAE, max absolute error, Pearson correlation, and the fraction
within tolerance.

---

## 5. Coverage reporting

Two distinct notions, kept apart:

**Production coverage** — of the questions the tournament actually posed, how
many did the live bot forecast, and how many was it still standing on at the
spot instant? Group and conditional posts are expanded into their subquestions,
because those are scored individually; counting posts would understate the
tournament. Posts that cannot be decomposed are still counted, since dropping
them would flatter the number.

**Benchmark coverage** — of the questions in the dataset, how many can the lab
score at all, and why not the rest? Every competition question type is listed
even when its count is zero: a type that silently never appears is precisely
the blind spot this report exists to surface.

The forfeited-points figure is `mean realised spot peer on covered questions ×
number of missed questions`. It is labelled `PROXY` and assumes missed
questions were no harder than covered ones — optimistic if misses cluster on
bursty question drops, which is exactly when they do cluster.

---

## 6. Reproducing an analysis

```bash
python3 research/analyze_track_record.py --dataset research/datasets/<id> --json report.json
python3 research/fetch_own_track_record.py --verify research/datasets/<id>
python3 -m unittest discover -s tests -t .
```

Analysis is a pure function of the dataset directory: no clocks, no randomness,
no network. Running it twice on the same dataset produces byte-identical
output, and there is a test asserting exactly that. Every report begins with
the `dataset_id`, the git commit that fetched it, and whether that tree was
dirty — a dirty tree means the commit alone does not identify the code that ran.

---

## 7. Current results

**Status: BLOCKED on a credential. The lab is built, tested and verified against
synthetic data; it has never been run against real data.**

| Item | State |
|---|---|
| Lab implemented | yes — 6 modules, 2 CLIs |
| Test suite | 148 tests, all passing |
| Endpoints verified reachable | yes — both `/users/me/` and `/data/download/` answer, returning HTTP 403 with an explicit "authenticated users only" message for an invalid token |
| API contract | verified against the Metaculus server source, not guessed |
| Real dataset fetched | **no** |
| Spot peer reproduced | **unknown** — untestable without a dataset |

The blocker is narrow and specific: **there is no `METACULUS_TOKEN` available on
this machine.** The repository has no `.env`; the token exists only as a GitHub
Actions secret, which by design cannot be read back out. The two ways to
unblock:

1. Put the bot token in `.env` or `METACULUS_TOKEN` locally, then run the two
   commands in section 1. Nothing else is required.
2. Or run the fetch inside GitHub Actions, where the secret already exists, and
   download the dataset as an artifact. This needs a new manual-dispatch
   workflow to be committed and pushed first.

Until then, everything in this document about *our* numbers is a
capability statement, not a result. The section will be replaced with real
figures on the first successful fetch.

---

## 8. Known limitations

- **`spot_scoring_time` overrides are invisible.** The CSV export does not
  include the per-question override, so where one exists our derived spot
  instant is wrong. The coverage check detects this; it does not fix it.
- **The peer denominator is not available** to an ordinary account. See §3.
- **The universe is a snapshot.** Questions created and resolved between
  fetches are invisible to the coverage denominator, so true coverage may be
  lower than reported, never higher.
- **Brier is binary-only** by choice. Reporting it for continuous questions
  would invite optimising the wrong thing where it is hardest to notice.
- **Model knowledge cutoff is not controlled here.** This lab measures forecasts
  the bot already made; it says nothing about whether a *replay* on historical
  questions would be contaminated by the model remembering the outcome. That is
  a separate problem for the next milestone.
- **Metaculus Terms of Use** state that API data *"may not be used to train,
  evaluate, or otherwise create or develop AI/ML models or algorithms without
  Metaculus's prior written permission"*. Datasets are gitignored and must stay
  local. The Bot Benchmarking access tier is described as *"intended for
  training and evaluation purposes"* — requesting it via the
  [Metaculus Data Needs form](https://docs.google.com/forms/d/e/1FAIpQLSeJhtZzHl5qMvBjbXbatyaqoS4IU7RE0GGw_vlhs6I9syqn1g/viewform)
  is the sanctioned route and should be done before the lab is used to justify
  changes to the forecaster.

---

## 9. Module map

| File | Responsibility |
|---|---|
| `research/provenance.py` | dataset identity, hashing, manifests, tamper detection |
| `research/metaculus_read_api.py` | read-only API client; refuses non-GET |
| `research/track_record.py` | typed readers for the three CSVs; spot-time derivation |
| `research/scorer.py` | log score, Brier, coverage, spot peer; EXACT/PROXY/UNAVAILABLE tiers |
| `research/validate.py` | reconstruction vs Metaculus; inversion diagnostic |
| `research/coverage.py` | production and benchmark coverage |
| `research/fetch_own_track_record.py` | CLI: build an immutable dataset |
| `research/analyze_track_record.py` | CLI: score, validate, report |
| `tests/` | 148 tests, standard library only |

`backtest/minibench_backtest.py` is untouched and is superseded by this lab for
anything involving our own performance.
