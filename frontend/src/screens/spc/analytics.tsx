import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { Badge, Disclosure, Empty, ErrorNotice, Metric, Notice, Panel, Skeleton } from '@/ui/kit'
import { type Column, DataTable } from '@/ui/table'
import { type Overlap, spc } from './api'
import { shortModel, useRuns, useSelection } from './shared'

type Obj = Record<string, unknown>
type Share = { key: string; positions: number; share: number }
type Bin = { from: number; to: number; count: number }

const pct = (x: unknown) => (typeof x === 'number' ? `${(x * 100).toFixed(1)}%` : '—')
const num = (x: unknown, d = 2) => (typeof x === 'number' ? x.toFixed(d) : '—')

const SHARE_COLUMNS: Column<Share>[] = [
  { key: 'k', header: 'Key', width: '1fr', cell: (s) => <span className="num">{s.key}</span> },
  {
    key: 'p',
    header: 'Positions',
    width: '90px',
    align: 'right',
    cell: (s) => <span className="num">{s.positions}</span>,
  },
  {
    key: 's',
    header: 'Share of gross',
    width: '120px',
    align: 'right',
    cell: (s) => <span className="num">{pct(s.share)}</span>,
  },
]

export function Analytics() {
  const { run, set } = useSelection()
  const runs = useRuns()
  const usable = (runs.data ?? []).filter((r) => r.status === 'completed' && r.positions > 0)
  const q = useQuery({
    queryKey: ['spc', 'analytics', run],
    queryFn: () => spc.analytics(run as number),
    enabled: run !== null,
  })
  const [picked, setPicked] = useState<number[]>([])
  const overlap = useQuery({
    queryKey: ['spc', 'overlap', picked],
    queryFn: () => spc.overlap(picked),
    enabled: picked.length >= 2,
  })

  return (
    <div className="flex min-w-0 flex-col gap-4">
      <Notice tone="info" title="Figures from saved output only">
        Everything here is arithmetic on the positions a run produced. It is not a forecast of
        performance and not an estimate of a leaderboard score.
      </Notice>

      <Panel
        title="Choose a run"
        description="Only completed runs with a valid portfolio can be analysed."
      >
        {runs.isPending ? (
          <Skeleton className="h-16" />
        ) : usable.length === 0 ? (
          <Empty title="No analysable runs yet">Complete a valid test run first.</Empty>
        ) : (
          <div className="flex flex-wrap gap-2" role="radiogroup" aria-label="Run to analyse">
            {usable.map((r) => (
              <button
                key={r.id}
                type="button"
                role="radio"
                aria-checked={run === r.id}
                onClick={() => set({ tab: 'analytics', run: r.id, prompt: r.promptId })}
                className={`rounded border px-3 py-1.5 text-body-compact ${run === r.id ? 'border-ink bg-surface-raised' : 'border-line hover:bg-surface-raised'}`}
              >
                Run {r.id} · {r.promptName} v{r.versionNumber} · {shortModel(r.modelRef)}
              </button>
            ))}
          </div>
        )}
      </Panel>

      {run !== null && q.isPending && <Skeleton className="h-64" />}
      {q.isError && <ErrorNotice error={q.error} title="Could not compute these figures" />}
      {q.data && <Figures data={q.data} />}

      <Panel
        title="Overlap between runs"
        description="Pick two or more runs to see securities they share and where they disagree on direction."
      >
        <div className="flex flex-col gap-3">
          <div className="flex flex-wrap gap-2">
            {usable.map((r) => {
              const on = picked.includes(r.id)
              return (
                <button
                  key={r.id}
                  type="button"
                  aria-pressed={on}
                  onClick={() =>
                    setPicked(on ? picked.filter((x) => x !== r.id) : [...picked, r.id])
                  }
                  className={`rounded border px-3 py-1.5 text-body-compact ${on ? 'border-ink bg-surface-raised' : 'border-line hover:bg-surface-raised'}`}
                >
                  Run {r.id}
                </button>
              )
            })}
          </div>
          {picked.length < 2 && (
            <p className="text-body-compact text-ink-subtle">Select at least two runs.</p>
          )}
          {overlap.isError && (
            <ErrorNotice error={overlap.error} title="Could not compare these runs" />
          )}
          {overlap.data && <OverlapView data={overlap.data} />}
        </div>
      </Panel>
    </div>
  )
}

function Figures({
  data,
}: {
  data: NonNullable<ReturnType<typeof useQuery<Awaited<ReturnType<typeof spc.analytics>>>>['data']>
}) {
  const s = data.summary as Obj
  const perf = data.performance as { available?: boolean; reason?: string }
  if (s['empty'])
    return (
      <Panel>
        <Empty title="This run has no positions">There is nothing to summarise.</Empty>
      </Panel>
    )
  const hist = (s['confidenceHistogram'] as Bin[]) ?? []
  const peak = Math.max(1, ...hist.map((b) => b.count))
  const cmp = data.comparison as Obj | null
  return (
    <>
      <Panel
        title={`Run ${data.runId} · ${data.promptName}`}
        description="Exposure figures are sums of the model’s confidence scores, not portfolio weights."
      >
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Metric
            boxed
            size="sm"
            label="Positions"
            value={String(s['positions'])}
            hint={`${s['longs']} long · ${s['shorts']} short`}
          />
          <Metric boxed size="sm" label="Gross confidence" value={num(s['grossConfidence'])} />
          <Metric
            boxed
            size="sm"
            label="Net confidence"
            value={num(s['netConfidence'])}
            hint={`${pct(s['netShare'])} of gross`}
          />
          <Metric boxed size="sm" label="Mean |confidence|" value={num(s['meanAbsConfidence'])} />
          <Metric
            boxed
            size="sm"
            label="Largest position"
            value={pct(s['largestShare'])}
            hint="of gross"
          />
          <Metric boxed size="sm" label="Top 5 share" value={pct(s['top5Share'])} />
          <Metric
            boxed
            size="sm"
            label="Effective positions"
            value={num(s['effectivePositions'], 1)}
            hint={`Herfindahl ${num(s['herfindahl'], 3)}`}
          />
          <Metric
            boxed
            size="sm"
            label="Horizon (days)"
            value={num(s['meanHorizonDays'], 1)}
            hint={`${s['minHorizonDays']}–${s['maxHorizonDays']}`}
          />
        </div>
      </Panel>

      <Panel
        title="Confidence distribution"
        description="Count of positions in each confidence band, from −1 to +1."
      >
        <div
          className="flex items-end gap-1"
          style={{ height: 120 }}
          role="img"
          aria-label="Histogram of confidence scores"
        >
          {hist.map((b) => (
            <div
              key={b.from}
              className="flex flex-1 flex-col items-center justify-end gap-1"
              title={`${b.from} to ${b.to}: ${b.count}`}
            >
              <span className="num text-caption text-ink-subtle">{b.count}</span>
              <div
                className={b.to <= 0 ? 'w-full bg-loss' : 'w-full bg-profit'}
                style={{ height: `${(b.count / peak) * 80}px`, minHeight: b.count ? 2 : 0 }}
              />
              <span className="num text-caption text-ink-subtle">{b.from}</span>
            </div>
          ))}
        </div>
      </Panel>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel
          title="By ISIN country"
          description="The ISIN’s prefix: where it was issued, not necessarily where it trades."
          bodyClassName="p-0"
        >
          <DataTable
            label="Exposure by ISIN country"
            rows={(s['byIsinCountry'] as Share[]) ?? []}
            rowKey={(x) => x.key}
            columns={SHARE_COLUMNS}
            maxHeight="30vh"
          />
        </Panel>
        <Panel title="By exchange (MIC)" bodyClassName="p-0">
          <DataTable
            label="Exposure by MIC"
            rows={(s['byMic'] as Share[]) ?? []}
            rowKey={(x) => x.key}
            columns={SHARE_COLUMNS}
            maxHeight="30vh"
          />
        </Panel>
      </div>

      {cmp ? (
        <Panel
          title={`Compared with run ${data.previousRunId}`}
          description="The previous completed run of the same prompt."
        >
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Metric
              boxed
              size="sm"
              label="Added"
              value={String((cmp['added'] as unknown[]).length)}
            />
            <Metric
              boxed
              size="sm"
              label="Removed"
              value={String((cmp['removed'] as unknown[]).length)}
            />
            <Metric
              boxed
              size="sm"
              label="Direction flipped"
              value={String((cmp['flipped'] as unknown[]).length)}
            />
            <Metric
              boxed
              size="sm"
              label="Similarity"
              value={num(cmp['jaccard'], 2)}
              hint={`${cmp['common']} in common`}
            />
          </div>
        </Panel>
      ) : (
        <Notice tone="info" title="No earlier run to compare with">
          Run the same prompt again to see what changed.
        </Notice>
      )}

      <Notice tone="warn" title="Performance data unavailable">
        {perf.reason}
      </Notice>

      <Disclosure summary="How WorldQuant scores the challenge (for orientation only)">
        <ul className="list-disc pl-5 text-body-compact text-ink-muted">
          {data.scoring.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
        <p className="mt-2 text-body-compact text-ink-subtle">
          Alpha Harness does not compute this score. Confirm the rules on the official page.
        </p>
      </Disclosure>
      <Disclosure summary="Notes on these figures">
        <ul className="list-disc pl-5 text-body-compact text-ink-muted">
          {((s['notes'] as string[]) ?? []).map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      </Disclosure>
    </>
  )
}

function OverlapView({ data }: { data: Overlap }) {
  const o = data.overlap as {
    pairs: { runA: number; runB: number; common: number; jaccard: number; conflicts: number }[]
    sharedSecurities: number
    uniqueSecurities: number
    conflicting: { isin: string; ticker: string; name: string; byRun: Record<string, number> }[]
  }
  return (
    <div className="flex flex-col gap-3">
      <div className="grid gap-4 sm:grid-cols-3">
        <Metric boxed size="sm" label="Distinct securities" value={String(o.uniqueSecurities)} />
        <Metric boxed size="sm" label="Held in 2+ runs" value={String(o.sharedSecurities)} />
        <Metric boxed size="sm" label="Opposite directions" value={String(o.conflicting.length)} />
      </div>
      <ul className="text-body-compact text-ink-muted">
        {o.pairs.map((p) => (
          <li key={`${p.runA}-${p.runB}`}>
            Runs {p.runA} and {p.runB}: {p.common} in common, similarity {p.jaccard.toFixed(2)},{' '}
            {p.conflicts} opposed
          </li>
        ))}
      </ul>
      {o.conflicting.length > 0 && (
        <div>
          <Badge tone="warn">Opposed</Badge>
          <ul className="mt-1 text-body-compact text-ink-muted">
            {o.conflicting.map((c) => (
              <li key={c.isin}>
                {c.name} ({c.ticker}, {c.isin}):{' '}
                {Object.entries(c.byRun)
                  .map(([r, v]) => `run ${r} ${v > 0 ? '+' : ''}${v.toFixed(2)}`)
                  .join(', ')}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
