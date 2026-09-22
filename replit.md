# EPL Quantitative Research Engine

Python-only research and backtesting engine that rebuilds leakage-controlled EPL pre-match features and evaluates them chronologically.

## Run & Operate

- `pnpm --filter @workspace/api-server run dev` — run the API server (port 5000)
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- Required env: `DATABASE_URL` — Postgres connection string

## Python CLI

- `python -m src.main full_analysis` — run cleaning, feature rebuild, leakage audit, model evaluation, backtest, and report generation
- `python -m src.main audit` — generate the cleaning audit
- `python -m src.main build_features` — rebuild pre-match features
- `python -m src.main leakage_audit` — classify predictors as SAFE, LEAKAGE, or UNKNOWN
- `python -m src.main validate_features` — generate manual spot-check traces
- `python -m src.main backtest` — run the fixed theoretical one-unit backtest
- `python -m src.main analyze_current current_sportybet_odds.csv` — analyze manually supplied current odds

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- API: Express 5
- DB: PostgreSQL + Drizzle ORM
- Validation: Zod (`zod/v4`), `drizzle-zod`
- API codegen: Orval (from OpenAPI spec)
- Build: esbuild (CJS bundle)

## Where things live

- `src/` — Python engine modules and CLI
- `data/raw/` — authoritative historical data and audit-only legacy files
- `exports/` — cleaning, leakage, feature, model, and calibration outputs
- `backtests/` — selection-level and grouped theoretical backtest outputs
- `reports/` — final research report
- `research_brief.txt` — source requirements for this project

## Architecture decisions

- Same-date fixtures are processed as a batch because kickoff times are unavailable; no same-date result is used as pre-match information.
- The historical-clean CSV is authoritative; legacy engineered CSVs are never model predictors.
- Suspicious source records are reported rather than silently corrected.
- Out-of-sample parameters are frozen; one fixed edge threshold is used for the research backtest.

## Product

The CLI produces reproducible pre-match features, leakage classifications, chronological model metrics,
calibration tables, manual validation traces, and clearly labeled theoretical backtest results.

## User preferences

The work must remain Python-only and research-focused; it must never place bets or connect to betting accounts.

## Gotchas

- Run `python -m src.main full_analysis` after changing feature logic.
- Current odds are inference-only and must stay in a separate manually supplied CSV.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
