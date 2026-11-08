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
