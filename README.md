<p align="center">
  <img src=".github/banner.png" alt="Alpha Harness" width="1280" />
</p>

<a href="https://www.star-history.com/?repos=residual-lab%2Falpha-harness&type=date&legend=bottom-right">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=residual-lab/alpha-harness&type=date&theme=dark&legend=bottom-right" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=residual-lab/alpha-harness&type=date&legend=bottom-right" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=residual-lab/alpha-harness&type=date&legend=bottom-right" />
 </picture>
</a>

## SPC Research Lab

A workspace for WorldQuant's **Systematic Prediction Challenge (SPC)**, under *BRAIN → SPC Research Lab*. It is unrelated to alpha simulation: an SPC prompt is a plain-text investment thesis that an LLM turns into a portfolio of global equity predictions. SPC never uses BRAIN simulations, simulation settings, Fast Expression operators, dataset sync or alpha submission, and spends no simulation quota.

**What it does**

- **Prompt Studio**: write prompts (up to 10,000 characters) or generate candidates from an idea; every save of changed text is a new immutable version, with duplicate, archive, restore, compare and a preferred version.
- **Test Runs**: *Run now* sends the saved prompt to an LLM you configured under *API Integration*, optionally after web research. A run is stored with a copy of the prompt, model and answer it used, goes through `queued → researching → generating → validating → completed | failed | cancelled`, can be cancelled, and is marked interrupted if the app restarts mid-run. Starting a second run for the same prompt while one is active is refused.
- **Validation**: the answer must be a top-level JSON array whose entries match the schema, with ISO 6166 ISIN check digits, MIC format, no duplicates and confidence/horizon/return ranges. Problems are listed with entry, field and message. Nothing is silently repaired; a *corrective pass* is a separate new run and the original is kept.
- **Portfolio Analytics**: arithmetic on saved output only (long/short counts, confidence exposure, concentration, country and MIC shares, change from the previous run, overlap between runs). No performance is shown: it reads *Performance data unavailable*, because no licensed dated price data is loaded.
- **Submission Export**: readiness checks, copy/download of the prompt, a sample JSON and a package file. **Submission is manual.** Alpha Harness has no SPC submission endpoint and never submits for you.

**Configuration**

- LLM: add a provider key and model under *API Integration*; pick the model per prompt.
- Web research (optional): open *API Integration › Web Research*, paste a Tavily or Brave Search key, and test it. Tavily supports search and page extraction; Brave supports search. Keys are stored encrypted (AES-GCM) in the local database and are never sent to the UI, only a masked hint. Each service has an on/off switch, a daily request budget you set (a local count, not the provider's balance), and one can be made the default. A connectivity test may use one paid credit and asks first. The older `ALPHA_HARNESS_SPC_SEARCH` / `ALPHA_HARNESS_SPC_SEARCH_KEY` variables still work as a fallback when no enabled key is stored in the app; they are never copied into the database and are not budgeted. Without any key, runs are labelled as using the model's own knowledge only. Result URLs are screened against private/loopback addresses; pages are not fetched by SPC.
- Optional reference data in the data directory: `spc_mic_reference.csv` (ISO 10383) and `spc_security_master.csv` (`isin,mic,ticker,name`). Without them, MIC and company mappings are shown as *unverified*.

**Limits worth knowing**

- A valid ISIN checksum shows the code is well formed, not that the security exists or matches the company.
- **The output key names are unconfirmed.** They are candidates taken from the brief; the complete official example could not be read. The app says so wherever they matter. Compare against WorldQuant's *What do I need to submit* article before submitting.
- A run costs LLM requests (and search requests if configured) and is confirmed before it starts. Local numbers are not leaderboard scores. This is an independent tool, not endorsed by WorldQuant.

Scoring, as WorldQuant describes it and shown only for orientation: prompts are evaluated per update frequency, active portfolios are combined by prompt weight, daily PnL is calculated, and ranking uses cumulative monthly PnL with PnL needed on at least 50% of trading days.

## Research Intelligence › Discover

Finds public research on a topic, lets you pick what to read, and saves the testable ideas in it to the same idea library the Paper Analyzer uses. Open *Research Intelligence → Discover*. It reads and records only: it never starts a simulation, signs in to BRAIN, or touches your own alphas or account data.

**Workflow.** Enter a topic (or pick a suggestion) → optionally edit the proposed search queries, choose sources, dates, market scope, search service and max results (the estimated API usage is shown before anything is sent) → *Discover* → review results with attribution → tick the pages to read → *Extract ideas* → save, reject, mark reviewed, or *Send to Alpha Builder* (a draft with no expression and no simulation).

**Two separate jobs.** *Discover* reads search results only (title, snippet, date, address). *Extract* reads the pages you chose, one at a time. Both are persisted jobs: they survive a page reload, show progress, can be cancelled, are marked *interrupted* if the app stops mid-run, and an identical job already running is returned instead of started twice. Nothing runs in the background; every job starts from a button.

**Sources** (*Sources* panel, `research_registry` table). Academic indexes (OpenAlex, arXiv), web search (Tavily or Brave) with optional site filters, arXiv q-fin RSS feeds, feeds you add, and URLs you type. Each entry has an on/off switch, a minimum fetch interval, and records its last success and last error. Web-source site lists are starting filters, not endorsements. Nothing is fabricated: a source whose key is missing says so.

**What is never done.** robots.txt is checked before any page is read (an unreadable robots.txt means *not allowed*); paywalled, login-gated and CAPTCHA pages are kept as metadata only and never bypassed; only public `http(s)` addresses are opened (private, loopback, link-local and reserved addresses are refused, DNS is resolved once and pinned, every redirect is re-checked, bodies are size-limited, no executables are downloaded). Page text is handed to the model between markers as untrusted data and any instructions in it are ignored.

**Tavily.** Search supports depth, max results, date range and site filters; extraction is a separate call with a separate cost. The crawl and research endpoints are deliberately not used. Results are cached for six hours (feeds for their minimum interval). Auth failures, exhausted quota, 429 (honouring `Retry-After`, at most three attempts with backoff), timeouts and pages that fail to extract are each reported distinctly. Your own daily request budget (*API Integration*) is enforced before a request is sent.

**Ideas.** The model returns structured ideas that are validated: every quote is searched for in the page, and an idea with no quote found is labelled *speculative*. Performance a page quotes is stored as the source's own claim and shown as unchecked. Several distinct hypotheses in one article stay separate; the same hypothesis from another source is linked to the existing idea (DOI, canonical URL, content hash and normalised hypothesis similarity are used) so every source reference is kept. Pages that are not research are rejected with a reason.

**Ranking.** Five separate indicators (source quality, research specificity, BRAIN feasibility, novelty, evidence quality), each *High / Medium / Low / Unknown* with its reasons. They are never combined and no Sharpe ratio or return is estimated. Novelty compares against your local library only.

**Limitations.** No crawl: only pages you select are read. Academic results are imported through the existing paper pipeline rather than the page reader. Tavily credit figures are estimates (check the provider dashboard). Brave has no date filter, so dates are applied to its results afterwards. Novelty is wording overlap, not meaning.

**Endpoints** (`/api/research/discover`): `GET /options`, `POST /queries`, `POST /estimate`, `POST /` (start), `GET /jobs/{id}/hits`, `POST /extract`, and `GET|POST|PATCH|DELETE /sources`. Progress and cancellation use `/api/research/jobs`.
