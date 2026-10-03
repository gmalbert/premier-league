# PitchAPI rollout

This consumer pins core **1.5.0** at `e31c5b9703ef05bad486fc348894fedd64655033`.
The shared implementation is reviewed in [core PR #30](https://github.com/gmalbert/pitch-oracle-core/pull/30).

The daily pipeline builds observation-aware historical features and a separate
fitted baseline. The hourly workflow refreshes primary fixtures, captures
lineups and actual successful forecast revisions, then publishes the optional
analytics index and validates the refreshed consumer cache. Jobs share a
repository concurrency group.

Match analytics, Team analytics and Feature validation are available from the
sidebar. Optional provider absence retains baseline serving. No enhanced family
is currently promoted: fixture mapping and historical observation eligibility
must pass strict per-league gates before enabling it. Backfill timestamps remain
actual capture times.

This rollout passed the local consumer tests, all twelve actual-app Playwright
checks and nine real-payload pilot checks. The shared final core suite passed
421 tests, with all six cross-platform CI jobs passing. Runtime Python
compilation passed.

`PITCH_API_KEY` was configured as a protected repository GitHub Actions secret
and its presence verified on October 3 after explicit user approval. The daily
and hourly integrations can use it once the rollout PR is merged. Jobs without
the optional credential still produce baseline forecasts and unavailable
provider health.

See the [production runbook](https://github.com/gmalbert/pitch-oracle-core/blob/e31c5b9703ef05bad486fc348894fedd64655033/docs/pitchapi-production-runbook.md) for
backfill budgets, correction policy, strict validation, promotion and rollback.
