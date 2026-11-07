import { useMutation } from '@tanstack/react-query'
import { toast } from 'sonner'
import { fmt } from '@/lib/format'
import {
  Badge,
  Button,
  Disclosure,
  Empty,
  ErrorNotice,
  KV,
  Metric,
  Notice,
  Panel,
  Skeleton,
} from '@/ui/kit'
import { type Column, DataTable } from '@/ui/table'
import {
  type Evidence,
  type Issue,
  isActive,
  type Position,
  type RunBrief,
  type RunDetail,
  spc,
  validationOf,
} from './api'
import {
  copy,
  StatusBadge,
  shortModel,
  useInvalidate,
  useRun,
  useRuns,
  useSelection,
} from './shared'

const RUN_COLUMNS: Column<RunBrief>[] = [
  { key: 'id', header: 'Run', width: '64px', cell: (r) => <span className="num">{r.id}</span> },
  {
    key: 'prompt',
    header: 'Prompt',
    width: 'minmax(160px,2fr)',
    cell: (r) => <span className="truncate">{r.promptName}</span>,
  },
  {
    key: 'v',
    header: 'Ver.',
    width: '56px',
    align: 'right',
    cell: (r) => <span className="num">v{r.versionNumber}</span>,
  },
  {
    key: 'model',
    header: 'Model',
    width: 'minmax(120px,1.4fr)',
    cell: (r) => <span className="num truncate">{shortModel(r.modelRef)}</span>,
  },
  {
    key: 'status',
    header: 'Status',
    width: '110px',
    cell: (r) => <StatusBadge status={r.status} valid={r.valid} />,
  },
  {
    key: 'pos',
    header: 'Positions',
    width: '84px',
    align: 'right',
    cell: (r) => <span className="num">{r.positions || '—'}</span>,
  },
  {
    key: 'time',
    header: 'Duration',
    width: '84px',
    align: 'right',
    cell: (r) => <span className="num">{fmt.duration(r.durationSeconds)}</span>,
  },
  {
    key: 'at',
    header: 'Started',
    width: '120px',
    align: 'right',
    cell: (r) => <span className="num">{fmt.dateTime(r.createdAt)}</span>,
  },
]

export function Runs() {
  const { prompt, run, set } = useSelection()
  const runs = useRuns()
  return (
    <div className="flex min-w-0 flex-col gap-4">
      <Panel
        title="Test runs"
        description="Every test of a prompt, kept across restarts. A run records the exact prompt, model and answer it saw."
        bodyClassName="p-0"
      >
        <DataTable
          label="SPC test runs"
          rows={runs.data ?? []}
          rowKey={(r) => String(r.id)}
          columns={RUN_COLUMNS}
          activeKey={run === null ? null : String(run)}
          onRowClick={(r) => set({ tab: 'runs', run: r.id, prompt: r.promptId })}
          loading={runs.isPending}
          error={runs.error}
          maxHeight="32vh"
          empty="No runs yet. Write a prompt in Prompt Studio and press Run now."
        />
      </Panel>
      {run === null ? (
        <Panel>
          <Empty title="Choose a run">
            Pick one above to see its status, validation and positions.
          </Empty>
        </Panel>
      ) : (
        <RunPanel id={run} onOpenPrompt={() => set({ tab: 'studio', prompt: prompt })} />
      )}
    </div>
  )
}

function RunPanel({ id, onOpenPrompt }: { id: number; onOpenPrompt: () => void }) {
  const q = useRun(id)
  const invalidate = useInvalidate()
  const { set } = useSelection()
  const cancel = useMutation({
    mutationFn: () => spc.cancel(id),
    onSuccess: () => {
      toast.success('Run cancelled')
      invalidate()
    },
  })
  const correct = useMutation({
    mutationFn: () => spc.correct(id, { research: 'off' }),
    onSuccess: (next) => {
      toast.success(`Corrective run ${next.id} started. Run ${id} is kept as it was.`)
      invalidate()
      set({ tab: 'runs', run: next.id })
    },
  })
  if (q.isPending) return <Skeleton className="h-64" />
  if (q.isError) return <ErrorNotice error={q.error} title="Could not load this run" />
  const r = q.data
  const v = validationOf(r)
  const active = isActive(r.status)
  const canCorrect = r.status === 'failed' && r.errorCode === 'invalid_output' && !!r.rawOutput

  return (
    <div className="flex min-w-0 flex-col gap-4">
      <Panel
        title={
          <span className="flex flex-wrap items-center gap-2">
            Run {r.id} <StatusBadge status={r.status} valid={r.valid} />
          </span>
        }
        description={`${r.promptName} · version ${r.versionNumber} · ${r.modelRef}`}
        actions={
          <>
            <Button size="sm" variant="ghost" onClick={onOpenPrompt}>
              Open prompt
            </Button>
            {active && (
              <Button
                size="sm"
                variant="danger"
                loading={cancel.isPending}
                onClick={() => cancel.mutate()}
              >
                Cancel run
              </Button>
            )}
            {canCorrect && (
              <Button
                size="sm"
                variant="primary"
                loading={correct.isPending}
                onClick={() => correct.mutate()}
              >
                Try a corrective pass
              </Button>
            )}
          </>
        }
      >
        <div className="flex flex-col gap-4">
          {active && (
            <Notice tone="info" title={r.message}>
              This updates by itself. The run is working on the server; you can leave this page.
            </Notice>
          )}
          {r.status === 'cancelled' && (
            <Notice tone="warn" title="Cancelled">
              {r.message}
            </Notice>
          )}
          {r.status === 'failed' && (
            <Notice tone="error" title={FAILURE[r.errorCode ?? ''] ?? 'The run failed'}>
              {r.errorMessage}
              {r.errorCode === 'invalid_output' &&
                ' The model’s answer is kept below; nothing was repaired or guessed.'}
            </Notice>
          )}
          {canCorrect && (
            <p className="text-body-compact text-ink-subtle">
              A corrective pass shows the model its own errors and asks again. It is recorded as a
              new run and this one stays as it is. A sample produced that way may not be
              reproducible from the prompt alone, and the export says so.
            </p>
          )}
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Metric boxed size="sm" label="Positions" value={r.positions || '—'} />
            <Metric boxed size="sm" label="Duration" value={fmt.duration(r.durationSeconds)} />
            <Metric
              boxed
              size="sm"
              label="Tokens"
              value={fmt.int((r.usage as { totalTokens?: number } | null)?.totalTokens)}
              hint={
                r.usage
                  ? `${r.usage['promptTokens'] ?? '—'} in · ${r.usage['outputTokens'] ?? '—'} out`
                  : undefined
              }
            />
            <Metric
              boxed
              size="sm"
              label="Validation"
              value={v ? (v.valid ? 'Valid' : `${v.errorCount} errors`) : '—'}
              tone={v ? (v.valid ? 'profit' : 'loss') : 'muted'}
              hint={v && v.warningCount > 0 ? `${v.warningCount} warnings` : undefined}
            />
          </div>
          <KV
            items={[
              ['Started', fmt.dateTime(r.startedAt ?? r.createdAt)],
              [
                'Finished',
                r.finishedAt ? fmt.dateTime(r.finishedAt) : active ? 'in progress' : '—',
              ],
              ['Model', r.modelRef],
              ['Update frequency', String(r.meta['update_frequency'] ?? '—')],
              [
                'Weight',
                r.meta['weight_hundredths'] != null
                  ? (Number(r.meta['weight_hundredths']) / 100).toFixed(2)
                  : '—',
              ],
              ...(r.correctsRunId
                ? ([['Corrects run', String(r.correctsRunId)]] as [string, string][])
                : []),
            ]}
          />
        </div>
      </Panel>

      {v && v.issues.length > 0 && <Issues issues={v.issues} schemaVerified={v.schemaVerified} />}
      {r.portfolio.length > 0 && <Portfolio rows={r.portfolio} />}
      <Research run={r} />
      {r.rawOutput != null && <Raw run={r} />}
      <Disclosure summary="Prompt this run tested (saved copy)">
        <pre className="num max-h-72 overflow-auto whitespace-pre-wrap break-words text-body-compact text-ink-muted">
          {r.promptSnapshot}
        </pre>
      </Disclosure>
    </div>
  )
}

/** The headline for each failure code the backend can give: specific, never "something went wrong". */
const FAILURE: Record<string, string> = {
  no_api_key: 'No API key for this model',
  budget_exhausted: 'Request allowance used up',
  provider_timeout: 'The model took too long',
  provider_unavailable: 'The provider is unavailable',
  llm_error: 'The model provider reported an error',
  empty_response: 'The model returned nothing',
  invalid_output: 'The answer does not match the SPC schema',
  research_not_configured: 'Web research is not configured',
  research_failed: 'Web research failed',
  interrupted: 'Interrupted by a restart',
  internal_error: 'Unexpected error in Alpha Harness',
}

const ISSUE_COLUMNS: Column<Issue>[] = [
  {
    key: 's',
    header: 'Level',
    width: '84px',
    cell: (i) => <Badge tone={i.severity === 'error' ? 'loss' : 'warn'}>{i.severity}</Badge>,
  },
  {
    key: 'i',
    header: 'Entry',
    width: '60px',
    align: 'right',
    cell: (i) => <span className="num">{i.index ?? '—'}</span>,
  },
  {
    key: 'f',
    header: 'Field',
    width: '150px',
    cell: (i) => <span className="num">{i.field ?? '—'}</span>,
  },
  {
    key: 'c',
    header: 'Code',
    width: '170px',
    cell: (i) => <span className="num text-ink-subtle">{i.code}</span>,
  },
  {
    key: 'm',
    header: 'What is wrong',
    width: 'minmax(260px,3fr)',
    cell: (i) => (
      <span className="whitespace-normal break-words">
        {i.message}
        {i.suggestion && <span className="text-ink-subtle"> {i.suggestion}</span>}
      </span>
    ),
  },
]

function Issues({ issues, schemaVerified }: { issues: Issue[]; schemaVerified: boolean }) {
  return (
    <Panel
      title="Validation"
      description={`Checked against the SPC portfolio schema${schemaVerified ? '' : ' (field names unconfirmed, see the notice at the top)'}. Entry numbers start at 0.`}
      bodyClassName="p-0"
    >
      <DataTable
        label="Validation issues"
        rows={issues}
        rowKey={(i) => `${i.code}-${i.index}-${i.field}-${i.message}`}
        columns={ISSUE_COLUMNS}
        rowHeight={44}
        maxHeight="40vh"
      />
    </Panel>
  )
}

const ID_WORD: Record<string, string> = {
  checksum_valid: 'checksum ok',
  verified: 'verified',
  structure_valid: 'format ok',
  recognized: 'recognized',
}

const POSITION_COLUMNS: Column<Position>[] = [
  {
    key: 'n',
    header: '#',
    width: '44px',
    align: 'right',
    cell: (p) => <span className="num text-ink-subtle">{p.position}</span>,
  },
  {
    key: 'name',
    header: 'Company',
    width: 'minmax(180px,2fr)',
    cell: (p) => <span className="truncate">{p.companyName}</span>,
  },
  {
    key: 't',
    header: 'Ticker',
    width: '90px',
    cell: (p) => <span className="num">{p.ticker}</span>,
  },
  {
    key: 'isin',
    header: 'ISIN',
    width: '130px',
    cell: (p) => <span className="num">{p.isin}</span>,
  },
  { key: 'mic', header: 'MIC', width: '70px', cell: (p) => <span className="num">{p.mic}</span> },
  {
    key: 'side',
    header: 'Side',
    width: '70px',
    cell: (p) => (
      <Badge tone={p.confidenceScore > 0 ? 'profit' : 'loss'}>
        {p.confidenceScore > 0 ? 'long' : 'short'}
      </Badge>
    ),
  },
  {
    key: 'c',
    header: 'Confidence',
    width: '96px',
    align: 'right',
    cell: (p) => (
      <span className="num">
        {p.confidenceScore > 0 ? '+' : ''}
        {p.confidenceScore.toFixed(2)}
      </span>
    ),
  },
  {
    key: 'h',
    header: 'Horizon',
    width: '84px',
    align: 'right',
    cell: (p) => <span className="num">{p.investmentHorizon}d</span>,
  },
  {
    key: 'r',
    header: 'Return',
    width: '84px',
    align: 'right',
    cell: (p) => <span className="num">{p.returnPrediction}</span>,
  },
  {
    key: 'ids',
    header: 'Identifier checks',
    width: 'minmax(200px,1.6fr)',
    cell: (p) => (
      <span
        className="truncate text-ink-subtle"
        title="A good checksum shows an ISIN is well formed, not that the security exists."
      >
        ISIN {ID_WORD[p.isinStatus] ?? p.isinStatus} · MIC {ID_WORD[p.micStatus] ?? p.micStatus} ·{' '}
        {p.mappingStatus === 'verified'
          ? 'mapping verified'
          : p.mappingStatus === 'mismatch'
            ? 'mapping mismatch'
            : 'mapping unverified'}
      </span>
    ),
  },
]

function Portfolio({ rows }: { rows: Position[] }) {
  return (
    <Panel
      title="Validated portfolio"
      description="Positions exactly as the model returned them. “Unverified” means no security master was loaded to confirm the company, ticker, ISIN and MIC belong together."
      bodyClassName="p-0"
    >
      <DataTable
        label="Portfolio positions"
        rows={rows}
        rowKey={(p) => String(p.position)}
        columns={POSITION_COLUMNS}
        maxHeight="50vh"
      />
    </Panel>
  )
}

const EVIDENCE_COLUMNS: Column<Evidence>[] = [
  {
    key: 't',
    header: 'Source',
    width: 'minmax(220px,3fr)',
    cell: (e) => (
      <a
        href={e.url}
        target="_blank"
        rel="noopener noreferrer"
        className="truncate text-link underline underline-offset-2 hover:text-ink"
        title={e.url}
      >
        {e.title || e.url}
      </a>
    ),
  },
  {
    key: 'k',
    header: 'Type',
    width: '150px',
    cell: (e) => <span className="text-ink-subtle">{e.sourceType.replaceAll('_', ' ')}</span>,
  },
  {
    key: 'p',
    header: 'Published',
    width: '150px',
    cell: (e) =>
      e.publishedAt ? (
        <span className="num">
          {e.publishedAt}
          {e.stale ? ' · stale' : ''}
        </span>
      ) : (
        <span className="text-ink-subtle">date unknown</span>
      ),
  },
  {
    key: 'r',
    header: 'Retrieved',
    width: '120px',
    align: 'right',
    cell: (e) => <span className="num">{fmt.dateTime(e.retrievedAt)}</span>,
  },
]

function Research({ run }: { run: RunDetail }) {
  const r = run.research as Record<string, unknown> | null
  if (!r || r['status'] === 'off') return null
  if (r['status'] === 'not_configured')
    return (
      <Notice tone="warn" title="Ran without web research">
        {String(r['note'])} {String(r['help'])}
      </Notice>
    )
  if (r['status'] === 'failed')
    return (
      <Notice tone="warn" title="Web research failed, so the model worked from its own knowledge">
        {String(r['error'])}
      </Notice>
    )
  return (
    <Panel
      title="Evidence"
      description={`${run.evidence.length} sources from ${String(r['provider'])}. Dates are the ones the source gave; “date unknown” is not the same as old.`}
      bodyClassName="p-0"
    >
      <DataTable
        label="Research evidence"
        rows={run.evidence}
        rowKey={(e) => e.url}
        columns={EVIDENCE_COLUMNS}
        maxHeight="36vh"
        empty="The search returned nothing usable."
      />
    </Panel>
  )
}

function Raw({ run }: { run: RunDetail }) {
  const text = run.rawOutput ?? ''
  return (
    <Disclosure
      summary={`Raw model output (${text.length.toLocaleString()} characters)`}
      defaultOpen={run.status === 'failed'}
    >
      <div className="flex flex-col gap-2">
        <div>
          <Button size="sm" onClick={() => copy(text, 'Output', toast.success, toast.error)}>
            Copy raw output
          </Button>
        </div>
        <pre className="num max-h-96 overflow-auto whitespace-pre-wrap break-words text-body-compact text-ink-muted">
          {text || '(empty)'}
        </pre>
      </div>
    </Disclosure>
  )
}
