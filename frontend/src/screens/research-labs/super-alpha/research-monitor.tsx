/** Super Alpha Lab, Advanced Research: the running task. The Research Dashboard, the generated
 *  candidates (inspect, copy, edit and rerun), and the task's real Start / Pause / Resume / Stop,
 *  which are the Tasks tab's own endpoints. Nothing here starts a simulation by itself. */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { http } from '@/api/http'
import { cn } from '@/lib/cn'
import { fmt } from '@/lib/format'
import { useRefetchOn } from '@/lib/ws'
import { type LabTask, labTasks } from '@/screens/tasks/api'
import {
  Badge,
  Button,
  Disclosure,
  Empty,
  ErrorNotice,
  Field,
  Metric,
  Notice,
  Panel,
  Progress,
  Textarea,
} from '@/ui/kit'
import { CoverageMetrics } from './advanced'
import type { Candidate, TaskDetail, Validation } from './research-types'

const B = '/api/super-lab/advanced'
const SAMPLER = 'super-research'
const ACTIVE = new Set(['RUNNING', 'QUEUED'])

type Outcome = Candidate['outcome']
const OUTCOME_TONE: Record<Outcome, 'profit' | 'loss' | 'warn' | 'muted'> = {
  accepted: 'profit',
  submitted: 'warn',
  waiting: 'muted',
  rejected: 'loss',
}

export function ResearchMonitor() {
  const queryClient = useQueryClient()
  const tasks = useQuery({ queryKey: ['lab-tasks'], queryFn: labTasks.list })
  useRefetchOn('studies', ['lab-tasks'], 2_000)
  const mine: LabTask[] = (tasks.data?.tasks ?? []).filter((t) => t.lab === SAMPLER)
  const [picked, setPicked] = useState<number | null>(null)
  const task = mine.find((t) => t.id === picked) ?? mine.at(-1) ?? null

  useEffect(() => {
    if (picked == null && task) setPicked(task.id)
  }, [picked, task])

  const detail = useQuery({
    queryKey: ['super-research-task', task?.id],
    queryFn: () => http.get<TaskDetail>(`${B}/tasks/${task?.id}`),
    enabled: task != null,
    refetchInterval: (q) => (ACTIVE.has(q.state.data?.status ?? '') ? 4_000 : false),
  })

  const control = useMutation({
    mutationFn: ({ id, action }: { id: number; action: 'run' | 'pause' | 'stop' }) =>
      labTasks[action](id),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['lab-tasks'] })
      void queryClient.invalidateQueries({ queryKey: ['super-research-task'] })
    },
  })

  if (tasks.isError) return <ErrorNotice error={tasks.error} title="Could not load the tasks" />
  if (!task) {
    return (
      <Panel title="Research Dashboard">
        <Empty title="No Advanced Research task yet">
          Add one above. It stays idle until you press Start.
        </Empty>
      </Panel>
    )
  }

  const d = detail.data
  const status = d?.status ?? task.status
  const active = ACTIVE.has(status)
  const act = (action: 'run' | 'pause' | 'stop') => control.mutate({ id: task.id, action })

  return (
    <div className="flex flex-col gap-4">
      <Panel
        title="Research Dashboard"
        description="Counts come from what this task has recorded. Submittable means BRAIN's own checks passed; nothing is submitted for you."
        actions={
          <div className="flex flex-wrap items-center gap-2">
            {mine.length > 1 && (
              <select
                aria-label="Research task"
                className="h-7 rounded-sm border border-hairline-strong bg-surface-1 px-2 text-body-compact text-ink"
                value={task.id}
                onChange={(e) => setPicked(Number(e.target.value))}
              >
                {mine.map((t) => (
                  <option key={t.id} value={t.id}>
                    {t.name || `Task ${t.id}`}
                  </option>
                ))}
              </select>
            )}
            <Badge tone={status === 'FAILED' ? 'loss' : active ? 'profit' : 'muted'}>
              {status}
            </Badge>
            {(status === 'IDLE' || status === 'FAILED') && (
              <Button
                variant="primary"
                size="sm"
                loading={control.isPending}
                onClick={() => act('run')}
              >
                Start
              </Button>
            )}
            {status === 'PAUSED' && (
              <Button
                variant="primary"
                size="sm"
                loading={control.isPending}
                onClick={() => act('run')}
              >
                Resume
              </Button>
            )}
            {active && (
              <Button size="sm" loading={control.isPending} onClick={() => act('pause')}>
                Pause
              </Button>
            )}
            {(active || status === 'PAUSED') && (
              <Button size="sm" loading={control.isPending} onClick={() => act('stop')}>
                Stop
              </Button>
            )}
          </div>
        }
      >
        {control.isError && <ErrorNotice error={control.error} />}
        {detail.isError && <ErrorNotice error={detail.error} title="Could not load the task" />}
        {d?.message && (
          <Notice tone="info" className="mb-3">
            {d.message}
          </Notice>
        )}
        {d && <DashboardBody detail={d} />}
      </Panel>

      {d && <Candidates detail={d} />}

      {d && (
        <Panel
          title="Operator coverage on this task"
          description="Which operators were put into candidates, which BRAIN accepted, and which it refused."
        >
          <CoverageMetrics summary={d.coverageSummary} />
          {Object.keys(d.avoided).length > 0 && (
            <Notice tone="warn" title="Left out after BRAIN refused them" className="mt-3">
              {Object.entries(d.avoided)
                .map(([name, why]) => `${name}: ${why}`)
                .join('; ')}
            </Notice>
          )}
        </Panel>
      )}
    </div>
  )
}

function DashboardBody({ detail }: { detail: TaskDetail }) {
  const { dashboard: s } = detail
  const share = s.budget > 0 ? s.budgetUsed / s.budget : null
  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
        <Metric label="Generated" value={fmt.int(s.generated)} />
        <Metric label="Locally valid" value={fmt.int(s.locallyValid)} />
        <Metric label="Simulated" value={fmt.int(s.simulated)} />
        <Metric label="Failed" value={fmt.int(s.failed)} tone={s.failed > 0 ? 'loss' : 'neutral'} />
        <Metric label="Waiting" value={fmt.int(s.waiting)} />
        <Metric
          label="Submittable"
          value={fmt.int(s.submittable)}
          tone={s.submittable > 0 ? 'profit' : 'neutral'}
        />
      </div>
      <div className="flex flex-col gap-1">
        <span className="text-body-compact text-ink-subtle">
          Budget: {fmt.int(s.budgetUsed)} of {fmt.int(s.budget)} simulations
        </span>
        <Progress value={share} label="Simulation budget used" />
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="flex flex-col gap-2">
          <h3 className="text-body font-medium text-ink">Best so far</h3>
          {s.best.length === 0 && (
            <p className="text-body-compact text-ink-subtle">Nothing scored yet.</p>
          )}
          {s.best.map((c) => (
            <div
              key={c.trialId}
              className="flex items-center justify-between gap-2 text-body-compact"
            >
              <span className="truncate text-ink-muted">{c.name ?? `#${c.number}`}</span>
              <span className="mono-metric text-ink">
                {fmt.ratio(c.score)} · Sharpe {fmt.ratio(c.sharpe)}
                {c.submittable ? ' · submittable' : ''}
              </span>
            </div>
          ))}
        </div>
        <div className="flex flex-col gap-2">
          <h3 className="text-body font-medium text-ink">Generations</h3>
          {s.history.length === 0 && (
            <p className="text-body-compact text-ink-subtle">No generation has run.</p>
          )}
          {s.history.map((g) => (
            <div key={g.generation} className="flex items-center justify-between text-body-compact">
              <span className="text-ink-muted">Generation {g.generation}</span>
              <span className="mono-metric text-ink">
                {g.scored}/{g.candidates} scored · best {fmt.ratio(g.best)}
              </span>
            </div>
          ))}
        </div>
      </div>
      <div className="flex flex-wrap gap-x-6 gap-y-2 text-body-compact text-ink-subtle">
        <span>Distinct selections {fmt.int(s.diversity.distinctSelections)}</span>
        <span>Distinct combos {fmt.int(s.diversity.distinctCombos)}</span>
        <span>Combo uniqueness {fmt.pct(s.diversity.comboUniqueness, 0)}</span>
        {Object.entries(s.diversity.families).map(([family, n]) => (
          <span key={family}>
            {family.replace(/_/g, ' ')} {n}
          </span>
        ))}
      </div>
      {Object.keys(s.failureKinds).length > 0 && (
        <div className="flex flex-wrap gap-2">
          {Object.entries(s.failureKinds).map(([kind, n]) => (
            <Badge key={kind} tone="loss">
              {kind.replace(/_/g, ' ')} {n}
            </Badge>
          ))}
        </div>
      )}
    </div>
  )
}

function Candidates({ detail }: { detail: TaskDetail }) {
  const [only, setOnly] = useState<Outcome | 'all'>('all')
  const rows = detail.candidates.filter((c) => only === 'all' || c.outcome === only)
  return (
    <Panel
      title="Generated Candidates"
      description="Every candidate with how it was made and what happened to it, failures included. Open one to inspect, copy, or edit and rerun it."
      actions={
        <select
          aria-label="Filter candidates"
          className="h-7 rounded-sm border border-hairline-strong bg-surface-1 px-2 text-body-compact text-ink"
          value={only}
          onChange={(e) => setOnly(e.target.value as Outcome | 'all')}
        >
          <option value="all">All ({detail.candidates.length})</option>
          <option value="accepted">Accepted</option>
          <option value="submitted">Submitted to BRAIN</option>
          <option value="waiting">Waiting</option>
          <option value="rejected">Rejected</option>
        </select>
      }
      bodyClassName="flex flex-col gap-2"
    >
      {rows.length === 0 && <Empty title="No candidates yet" />}
      {rows.map((c) => (
        <CandidateRow key={c.trialId} taskId={detail.id} c={c} />
      ))}
    </Panel>
  )
}

function CandidateRow({ taskId, c }: { taskId: number; c: Candidate }) {
  const summary = (
    <span className="flex min-w-0 flex-1 flex-wrap items-center gap-2">
      <span className="text-ink">{c.name ?? `Candidate ${c.number}`}</span>
      <Badge tone={OUTCOME_TONE[c.outcome]}>{c.outcome}</Badge>
      {c.origin && <Badge tone="outline">{c.origin}</Badge>}
      {c.baseline && <Badge tone="muted">baseline</Badge>}
      {c.submittable && <Badge tone="profit">submittable</Badge>}
      <span className="mono-metric ml-auto text-ink-subtle">
        Sharpe {fmt.ratio(c.sharpe)} · Fitness {fmt.ratio(c.fitness)} · TO {fmt.pct(c.turnover)}
      </span>
    </span>
  )
  return (
    <Disclosure summary={summary}>
      <CandidateBody taskId={taskId} c={c} />
    </Disclosure>
  )
}

function CandidateBody({ taskId, c }: { taskId: number; c: Candidate }) {
  const queryClient = useQueryClient()
  const [selection, setSelection] = useState(c.selection)
  const [combo, setCombo] = useState(c.combo)
  const [copied, setCopied] = useState<string | null>(null)
  const edited = selection !== c.selection || combo !== c.combo

  const rerun = useMutation({
    mutationFn: async () => {
      const checked = await http.post<Validation>(`${B}/validate`, { selection, combo })
      const bad = [checked.selection, checked.combo].flatMap((x) => (x && !x.ok ? x.problems : []))
      if (bad.length > 0) throw new Error(bad[0])
      return http.post<unknown>(`${B}/tasks/${taskId}/proposals`, {
        selection,
        combo,
        note: `Edited from candidate ${c.number}`,
      })
    },
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['super-research-task', taskId] })
    },
  })

  const copy = async (what: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(what)
    } catch {
      setCopied(null)
    }
  }

  return (
    <div className="flex flex-col gap-3">
      {c.explanation && <p className="text-body-compact text-ink-muted">{c.explanation}</p>}
      <Field label="Selection expression">
        <Textarea
          className="mono-metric min-h-16 py-2"
          value={selection}
          onChange={(e) => setSelection(e.target.value)}
          spellCheck={false}
        />
      </Field>
      <Field label="Combo expression">
        <Textarea
          className="mono-metric min-h-24 py-2"
          value={combo}
          onChange={(e) => setCombo(e.target.value)}
          spellCheck={false}
        />
      </Field>
      <div className="flex flex-wrap gap-2">
        <Button size="sm" onClick={() => void copy('selection', c.selection)}>
          Copy selection
        </Button>
        <Button size="sm" onClick={() => void copy('combo', c.combo)}>
          Copy combo
        </Button>
        <Button
          variant="primary"
          size="sm"
          disabled={!edited}
          loading={rerun.isPending}
          onClick={() => rerun.mutate()}
        >
          Queue edited version
        </Button>
        {copied && (
          <span className="self-center text-body-compact text-ink-subtle">
            Copied the {copied}.
          </span>
        )}
        {rerun.isSuccess && (
          <span className="self-center text-body-compact text-ink-subtle">
            Queued. It runs on the next refill while the task runs.
          </span>
        )}
      </div>
      {rerun.isError && <ErrorNotice error={rerun.error} title="That edit was not queued" />}
      <div className={cn('grid gap-x-6 gap-y-1 text-body-compact sm:grid-cols-2')}>
        <Line k="Operators" v={c.operators.join(', ') || '—'} />
        <Line k="Parents" v={c.parents.join(', ') || '—'} />
        <Line k="Alpha" v={c.alphaId ?? '—'} />
        <Line
          k="Settings"
          v={Object.entries(c.settings)
            .map(([k, v]) => `${k} ${String(v)}`)
            .join(' · ')}
        />
        <Line k="Score" v={fmt.ratio(c.score)} />
        <Line k="Drawdown" v={fmt.pct(c.drawdown)} />
        <Line
          k="Turnover in 2%–40%"
          v={c.turnoverInRange == null ? '—' : c.turnoverInRange ? 'yes' : 'no'}
        />
        <Line k="Versus equal weight" v={fmt.ratio(c.vsEqualWeight)} />
        <Line k="Failed checks" v={c.failedChecks.join(', ') || 'none'} />
        {c.rejection && <Line k="Rejection" v={`${c.rejectionKind ?? ''} ${c.rejection}`.trim()} />}
      </div>
    </div>
  )
}

function Line({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex min-w-0 gap-2">
      <span className="shrink-0 text-ink-subtle">{k}</span>
      <span className="min-w-0 break-words text-ink-muted">{v}</span>
    </div>
  )
}
