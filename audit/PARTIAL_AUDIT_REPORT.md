# WQ Research Engine — Partial Audit Report (session 1)

Target: its-meAmrit/WQ-Research-Engine @ 5a7d16d9f8494cccf47afaeeebceef5da9de0d7a (shallow clone, branch as cloned, tree clean)
Env: Linux, Python 3.13 (uv-managed interpreter 3.14 used by `uv run`), Node 22, pnpm. Sandbox is offline for non-proxied hosts.
Upstream residual-lab/alpha-harness: NOT fetched or compared in this session.

## Baseline commands (all executed)

| Command | Result |
|---|---|
| `uv sync --frozen` (backend) | OK |
| `uv run ruff check .` | exit 0, all checks passed |
| `uv run pytest -q` | exit 0, 503 passed in 78.6s |
| `pnpm install --frozen-lockfile` | OK |
| `pnpm exec tsc --noEmit -p .` | exit 0 |
| `pnpm exec biome check .` | exit 0 (168 files) |
| `pnpm exec vite build` | exit 0 |
| Pyrefly / Basedpyright | NOT RUN |
| Desktop (Tauri) build | NOT RUN |
| OpenAPI-vs-frontend type drift | NOT TESTED |

## 2.1 Research router / AppState wiring — hypothesis REFUTED at this commit

- `research.router` is imported and included (`main.py:200`); `super_research`, `discover`, `alpha_builder` also included.
- `AppState` builds `self.web`, `self.research`, `self.discovery` (`state.py:118-141`); `research.reconcile()` runs in startup, `research.stop()` in shutdown.
- Live OpenAPI (app booted via TestClient with lifespan): all of `/api/research/{options,papers,papers/upload,papers/{id}/analyze,jobs,ideas,search,discover/*,builder/*}` and `/api/super-lab/advanced/*` are registered.
- `GET /api/research/options` → 200 on a fresh data dir, with no keys set: `models: []`, `webSearch: false`. Startup does not crash without credentials.
- Navigation: `/labs/papers` ("Research Intelligence") is routed (`router.tsx:80`) and listed in `nav.ts:42`.
- Therefore no router/AppState fix was needed or made. No code was changed.

## 2.2 Paper upload — partial, with evidence

Fresh data dir, single-page generated PDF, headers `X-Harness-Client: 1`:
- `POST /api/research/papers/upload` → 202, import job created, completed (1 page, 1 section), paper persisted and listed. PASS.
- `POST /api/research/papers/{id}/analyze` → 422 `no_model` ("No model is set up yet… API Integration › Models"). This is a clear, correct refusal, not a wiring fault.
- Observation (unverified severity): the PDF title was taken from the first text line (the whole abstract sentence). Minor metadata-quality issue; `metadataVerified: false`.
- NOT TESTED (no LLM credentials; paid calls need your authorization): LLM analysis, structured output parsing, evidence quote verification, idea dedup, assess/handoff to Alpha Builder, retries/cancel/timeout, restart recovery, scanned/corrupt/large PDFs, multi-column/tables.

## Most likely cause of the reported failures (hypothesis, not confirmed)

With no model configured, analysis cannot run; with no Tavily/Brave key, `webSearch` is false. Check **API Integration › Models** and search keys first. Also: DuckDB FTS extension install fails when offline (`catalog.fts_unavailable`, logged, non-fatal) — may degrade field search on offline machines.

## Not done in this session (status NOT TESTED)

2.3 public-source adapters; 2.4 SuperAlpha (note: "Advanced research" maps to `api/super_research.py` → `/api/super-lab/advanced/*`; whether a paper→SuperAlpha bridge exists is not yet traced); Phase 3 per-feature matrix; Phase 4 security audit (observed only: TrustedHost + Sec-Fetch-Site + custom-header write guard in `main.py`); Phase 5 upstream parity; Phase 6 new tests; Phase 7 remaining deliverables (FEATURE_INVENTORY, TEST_MATRIX, BUG_REPORT, SECURITY_AUDIT, UPSTREAM_PARITY, FIX_CHANGELOG, REMAINING_LIMITATIONS, RUNBOOK).

## Reproduce

```
cd backend && uv sync --frozen && uv run ruff check . && uv run pytest -q
HOME=$(mktemp -d) uv run python <probe>   # boots create_app() under TestClient; keeps ~/.alpha-harness untouched
cd ../frontend && pnpm install --frozen-lockfile && pnpm exec tsc --noEmit -p . && pnpm exec biome check . && pnpm exec vite build
```
