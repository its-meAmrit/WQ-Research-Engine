/** Alpha Builder candidates: review, edit, save, preview, create a task, results, refinement. */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import { CopyIcon, DownloadIcon } from 'lucide-react'
import { useId, useState } from 'react'
import { toast } from 'sonner'
import {
  Badge,
  Button,
  Checkbox,
  Disclosure,
  ErrorNotice,
  KV,
  Notice,
  Skeleton,
  Textarea,
} from '@/ui/kit'
import { Confirm } from '@/ui/overlay'
import {
  type BuilderCandidate,
  type BuilderResults,
  builder,
  type SimSettings,
  type Validation,
} from './api'
import { useJobWatch } from './builder-jobs'

const OUTCOME = {
  valid: { tone: 'profit', label: 'Valid for task preview' },
  needs_review: { tone: 'warn', label: 'Needs review' },
  blocked: { tone: 'loss', label: 'Blocked' },
  not_validated: { tone: 'muted', label: 'Not validated' },
  invalid: { tone: 'loss', label: 'Blocked' },
} as const

const KIND: Record<string, string> = {
  replication: 'Closest replication',
  simplified: 'Simplified',
  proxy: 'Documented proxy',
  direction: 'Opposite direction',
  normalization: 'Different normalisation',
  event_aware: 'Event-aware',
  robust: 'Robustness-oriented',
  manual_edit: 'Your edit',
  refinement: 'Refinement draft',
}

export const OutcomeBadge = ({ outcome }: { outcome: string }) => {
  const o = OUTCOME[outcome as keyof typeof OUTCOME] ?? OUTCOME.not_validated
  return <Badge tone={o.tone}>{o.label}</Badge>
}

const validationOf = (c: BuilderCandidate): Validation | null =>
  (c.validationDetail as Validation | null) ?? null

const fmtMetric = (key: string, v: unknown): string => {
  if (typeof v !== 'number') return 'not reported'
  return ['turnover', 'returns', 'drawdown', 'margin'].includes(key)
    ? `${(v * 100).toFixed(key === 'margin' ? 3 : 2)}%`
    : v.toFixed(2)
}

const METRIC_LABEL: [string, string][] = [
  ['sharpe', 'Sharpe'],
  ['fitness', 'Fitness'],
  ['turnover', 'Turnover'],
  ['returns', 'Returns'],
  ['drawdown', 'Drawdown'],
  ['margin', 'Margin'],
  ['train_sharpe', 'In-sample Sharpe'],
  ['test_sharpe', 'Out-of-sample Sharpe'],
  ['self_correlation', 'Self-correlation'],
  ['prod_correlation', 'Production correlation'],
]

/** Real BRAIN metrics for the candidate, labelled as BRAIN's; commentary is set apart. */
export function ResultsPanel({ candidate }: { candidate: BuilderCandidate }) {
  const results = useQuery({
    queryKey: ['research', 'builder', 'results', candidate.id],
    queryFn: () => builder.results(candidate.id),
    refetchInterval: (q) => (q.state.data?.state === 'waiting' ? 5000 : false),
  })
  if (results.isPending) return <Skeleton className="h-16" label="Loading results" />
  if (results.isError) return <ErrorNotice error={results.error} title="Results did not load" />
  const r: BuilderResults = results.data
  return (
    <section className="flex flex-col gap-2 rounded-md bg-surface-2 p-3">
      <h5 className="text-body font-medium text-ink">Results</h5>
      <p className="text-caption text-ink-subtle">{r.message}</p>
      {r.taskId != null && (
        <Button
          variant="ghost"
          className="self-start"
          render={<Link to="/tasks" search={{ task: r.taskId }} />}
        >
          Open the task
        </Button>
      )}
      {(r.settingsMismatch ?? []).length > 0 && (
        <Notice tone="warn" title="Settings differ from the candidate">
          BRAIN ran with different values for: {(r.settingsMismatch ?? []).join(', ')}. Do not
          compare this with the candidate as saved.
        </Notice>
      )}
      {r.metrics && (
        <>
          <p className="text-caption text-ink-subtle">
            Metrics returned by BRAIN (in-sample unless marked).
          </p>
          <KV items={METRIC_LABEL.map(([k, label]) => [label, fmtMetric(k, r.metrics?.[k])])} />
          {r.checks.length > 0 && (
            <ul className="flex flex-col gap-1 text-body-compact">
              {r.checks.map((c, i) => (
                <li key={`${String(c['name'])}-${i}`} className="flex gap-2">
                  <Badge tone={String(c['result']).toUpperCase() === 'FAIL' ? 'loss' : 'muted'}>
                    {String(c['result'])}
                  </Badge>
                  <span>
                    {String(c['name'])}: {String(c['value'] ?? '')} (limit{' '}
                    {String(c['limit'] ?? '—')})
                  </span>
                </li>
              ))}
            </ul>
          )}
          {r.diagnostics.length > 0 && (
            <div className="flex flex-col gap-1">
              <h6 className="text-caption font-medium text-ink-muted">
                What the numbers may point to (rule-based reading of BRAIN's figures, not BRAIN's)
              </h6>
              <ul className="list-disc pl-5 text-body-compact text-ink-muted">
                {r.diagnostics.map((d, i) => (
                  <li key={`${String(d['source'])}-${i}`}>{String(d['text'])}</li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}
    </section>
  )
}

function Refinement({
  candidate,
  modelRef,
  onDone,
}: {
  candidate: BuilderCandidate
  modelRef: string | null
  onDone: () => void
}) {
  const [commentary, setCommentary] = useState<string | null>(null)
  const watch = useJobWatch((job) => {
    setCommentary((job.result?.['commentary'] as string | undefined) ?? null)
    onDone()
  })
  const lineage = useQuery({
    queryKey: ['research', 'builder', 'lineage', candidate.id],
    queryFn: () => builder.lineage(candidate.id),
  })
  const results = useQuery({
    queryKey: ['research', 'builder', 'results', candidate.id],
    queryFn: () => builder.results(candidate.id),
  })
  const complete = results.data?.state === 'complete'
  const reason = !complete
    ? 'Refinement starts from real BRAIN results, and this candidate has none yet.'
    : !modelRef
      ? 'Choose a model first.'
      : null
  return (
    <section className="flex flex-col gap-2">
      <h5 className="text-body font-medium text-ink">Refinement</h5>
      {lineage.data?.multipleTestingWarning && (
        <Notice tone="warn" title="Repeated searching">
          {lineage.data.multipleTestingWarning}
        </Notice>
      )}
      <p className="text-caption text-ink-subtle">
        Refinements are drafts. Each becomes a new version and the tested one is never changed. A
        better in-sample number on a refinement is not evidence it is better.
      </p>
      <div className="flex flex-wrap items-center gap-2">
        <Button
          disabled={reason != null}
          loading={watch.busy}
          onClick={() => watch.start(() => builder.refine(candidate.id, modelRef))}
        >
          Ask for refinement drafts
        </Button>
        {watch.busy && (
          <Button variant="ghost" onClick={watch.cancel}>
            Cancel
          </Button>
        )}
      </div>
      {reason && <p className="text-caption text-ink-subtle">Unavailable: {reason}</p>}
      {watch.startError != null && <ErrorNotice error={watch.startError} />}
      {watch.job?.status === 'failed' && (
        <Notice
          tone="error"
          title="Refinement failed"
          action={<Button onClick={watch.clear}>Dismiss</Button>}
        >
          {watch.job.errorMessage}
        </Notice>
      )}
      {commentary && (
        <Notice tone="info" title="Model commentary (not a BRAIN result)">
          {commentary}
        </Notice>
      )}
    </section>
  )
}

export function CandidateCard({
  candidate,
  all,
  selected,
  onSelect,
  modelRef,
}: {
  candidate: BuilderCandidate
  all: BuilderCandidate[]
  selected: boolean
  onSelect: (on: boolean) => void
  modelRef: string | null
}) {
  const client = useQueryClient()
  const refresh = () => {
    void client.invalidateQueries({ queryKey: ['research', 'builder'] })
    void client.invalidateQueries({ queryKey: ['research', 'idea', candidate.ideaId] })
  }
  const [expression, setExpression] = useState(candidate.expression ?? '')
  const [note, setNote] = useState('')
  const editId = useId()
  const v = validationOf(candidate)
  const changed = expression.trim() !== (candidate.expression ?? '').trim()

  const check = useMutation({ mutationFn: () => builder.check(candidate.id, { expression }) })
  const revise = useMutation({
    mutationFn: () => builder.revise(candidate.id, { expression, note }),
    onSuccess: (next) => {
      refresh()
      toast(`Saved as version ${next.version}`, {
        description: 'The earlier version is kept unchanged.',
      })
    },
  })
  const save = useMutation({ mutationFn: () => builder.save(candidate.id), onSuccess: refresh })
  const revalidate = useMutation({
    mutationFn: () => builder.revalidate(candidate.id),
    onSuccess: refresh,
  })
  const exporter = useMutation({
    mutationFn: () => builder.export(candidate.id),
    onSuccess: (data) => {
      const url = URL.createObjectURL(
        new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }),
      )
      const a = document.createElement('a')
      a.href = url
      a.download = `alpha-candidate-v${candidate.version}.json`
      a.click()
      URL.revokeObjectURL(url)
    },
  })

  const live = check.data ?? null
  const shown = changed && live ? live : v
  const parent = all.find((c) => c.id === candidate.parentId)
  const blocked =
    candidate.validationOutcome === 'blocked' || candidate.validationOutcome === 'invalid'
  const saveReason =
    candidate.status !== 'draft'
      ? null
      : changed
        ? 'You have unsaved edits: save them as a new version first.'
        : blocked
          ? 'This candidate is blocked; fix what the validation lists.'
          : null
  const explanation = (candidate.explanation ?? {}) as Record<string, unknown>

  return (
    <article className="flex flex-col gap-3 rounded-md border border-hairline p-3">
      <header className="flex flex-wrap items-center gap-2">
        <Checkbox
          checked={selected}
          onChange={onSelect}
          disabled={candidate.status === 'draft' || candidate.status === 'tasked'}
          label={`Select version ${candidate.version} for a task`}
        />
        <span className="text-body font-medium text-ink">
          v{candidate.version} ·{' '}
          {String(explanation['name'] ?? KIND[candidate.variantKind ?? ''] ?? 'Candidate')}
        </span>
        <OutcomeBadge outcome={candidate.validationOutcome} />
        <Badge tone="outline">{candidate.status}</Badge>
        {candidate.variantKind && (
          <Badge tone="muted">{KIND[candidate.variantKind] ?? candidate.variantKind}</Badge>
        )}
        {parent && <span className="text-caption text-ink-subtle">from v{parent.version}</span>}
      </header>

      <label htmlFor={editId} className="text-caption font-medium text-ink-muted">
        Fast Expression
      </label>
      <Textarea
        id={editId}
        className="num min-h-20"
        spellCheck={false}
        value={expression}
        disabled={candidate.status === 'tasked'}
        onChange={(e) => {
          setExpression(e.target.value)
          check.reset()
        }}
      />
      <div className="flex flex-wrap items-center gap-2">
        <Button
          onClick={() => {
            void navigator.clipboard.writeText(expression)
            toast('Expression copied')
          }}
        >
          <CopyIcon className="size-4" aria-hidden /> Copy
        </Button>
        <Button loading={check.isPending} disabled={!changed} onClick={() => check.mutate()}>
          Check edit
        </Button>
        <Button
          variant="primary"
          loading={revise.isPending}
          disabled={!changed || !expression.trim() || candidate.status === 'tasked'}
          onClick={() => revise.mutate()}
        >
          Save as new version
        </Button>
        <Button loading={revalidate.isPending} onClick={() => revalidate.mutate()}>
          Revalidate
        </Button>
        <Button loading={exporter.isPending} onClick={() => exporter.mutate()}>
          <DownloadIcon className="size-4" aria-hidden /> Export
        </Button>
      </div>
      {changed && (
        <Textarea
          aria-label="Note for the new version"
          placeholder="Why this edit (optional)"
          value={note}
          onChange={(e) => setNote(e.target.value)}
        />
      )}
      {[check.error, revise.error, save.error, revalidate.error, exporter.error].map(
        (e, i) => e != null && <ErrorNotice key={`${i}-${String(e)}`} error={e} />,
      )}

      {changed && live && (
        <p className="text-caption text-ink-subtle">
          Checking your unsaved edit: <OutcomeBadge outcome={live.outcome} />
        </p>
      )}
      {shown && (
        <div className="flex flex-col gap-1">
          {shown.checks.length === 0 && (
            <p className="text-caption text-ink-subtle">Every static check passed.</p>
          )}
          {shown.checks.map((c) => (
            <Notice key={`${c.code}-${c.message}`} tone={c.severity === 'block' ? 'error' : 'warn'}>
              {c.message}
            </Notice>
          ))}
          <Disclosure summary="What this check cannot tell you">
            <ul className="list-disc pl-5 text-body-compact text-ink-muted">
              {shown.notChecked.map((n) => (
                <li key={n}>{n}</li>
              ))}
            </ul>
          </Disclosure>
        </div>
      )}

      <Disclosure summary="Provenance, fields and settings">
        <div className="flex flex-col gap-2 p-3">
          <KV
            items={[
              ['Research idea', candidate.ideaId],
              ['Candidate', `${candidate.id} · version ${candidate.version}`],
              [
                'Scope',
                `${candidate.scope['region']} · D${candidate.scope['delay']} · ${candidate.scope['universe']} (${candidate.scope['instrumentType']})`,
              ],
              ['BRAIN fields', candidate.fieldsUsed.join(', ') || '—'],
              ['Operators', candidate.operatorsUsed.join(', ') || '—'],
              ['Source', String(candidate.provenance['sourceTitle'] ?? '—')],
              ['Hypothesis', String(candidate.provenance['hypothesis'] ?? '—')],
              ['Settings', JSON.stringify(candidate.settings)],
            ]}
          />
          {explanation['relationToSource'] != null && (
            <p className="text-body-compact text-ink-muted">
              <b>Relation to the source:</b> {String(explanation['relationToSource'])}
            </p>
          )}
          {explanation['implementation'] != null && (
            <p className="text-body-compact text-ink-muted">
              <b>Implementation:</b> {String(explanation['implementation'])}
            </p>
          )}
          {candidate.changes.length > 0 && (
            <ul className="list-disc pl-5 text-body-compact text-ink-muted">
              {candidate.changes.map((c) => (
                <li key={c}>{c}</li>
              ))}
            </ul>
          )}
          {(
            candidate.provenance['evidence'] as { quote: string; page: number | null }[] | undefined
          )?.map((e) => (
            <blockquote
              key={e.quote}
              className="border-l-2 border-hairline pl-3 text-caption text-ink-subtle"
            >
              “{e.quote}” {e.page != null && `(page ${e.page})`}
            </blockquote>
          ))}
          {candidate.limitations.length > 0 && (
            <div>
              <h6 className="text-caption font-medium text-ink-muted">Known limitations</h6>
              <ul className="list-disc pl-5 text-body-compact text-ink-muted">
                {candidate.limitations.map((l) => (
                  <li key={l}>{l}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </Disclosure>

      <div className="flex flex-wrap items-center gap-2">
        <Button
          variant="primary"
          loading={save.isPending}
          disabled={candidate.status !== 'draft' || saveReason != null}
          onClick={() => save.mutate()}
        >
          {candidate.status === 'draft' ? 'Save candidate' : 'Saved'}
        </Button>
        {saveReason && (
          <span className="text-caption text-ink-subtle">Unavailable: {saveReason}</span>
        )}
      </div>

      {candidate.studyId != null && (
        <>
          <ResultsPanel candidate={candidate} />
          <Refinement candidate={candidate} modelRef={modelRef} onDone={refresh} />
        </>
      )}
    </article>
  )
}

/** Preview the task, confirm it, create it. Previewing is read-only and free. */
export function TaskPanel({ selected }: { selected: BuilderCandidate[] }) {
  const client = useQueryClient()
  const ids = selected.map((c) => c.id)
  const [open, setOpen] = useState(false)
  const [ack, setAck] = useState(false)
  // One key per opening of the dialog: a double click re-sends the same key and gets the same
  // task back instead of a second one.
  const [key, setKey] = useState(() => crypto.randomUUID())
  const preview = useQuery({
    queryKey: ['research', 'builder', 'preview', ids],
    queryFn: () => builder.preview(ids),
    enabled: ids.length > 0,
  })
  const create = useMutation({
    mutationFn: () =>
      builder.createTask({
        candidateIds: ids,
        idempotencyKey: key,
        confirm: true,
        acknowledgeReview: ack,
      }),
    onSuccess: (task) => {
      setOpen(false)
      void client.invalidateQueries({ queryKey: ['research', 'builder'] })
      void client.invalidateQueries({ queryKey: ['lab-tasks'] })
      toast(task.existing ? 'That task already exists' : 'Task created, not started', {
        description: 'Open Tasks and press Run when you want it to use simulations.',
      })
    },
  })
  const p = preview.data
  const reason =
    ids.length === 0
      ? 'Select one or more saved candidates.'
      : preview.isPending
        ? 'Checking the task…'
        : preview.isError
          ? 'The preview failed.'
          : p && p.problems.length > 0
            ? p.problems[0]
            : p && p.simulations === 0
              ? 'Nothing here can run.'
              : null
  return (
    <section className="flex flex-col gap-2 rounded-md bg-surface-2 p-3">
      <h4 className="text-body font-medium text-ink">Create a simulation task</h4>
      {preview.isError && <ErrorNotice error={preview.error} />}
      {p && (
        <>
          <p className="text-body-compact text-ink-muted">
            {p.quota} Nothing is simulated until you press Run on the task in Tasks.
          </p>
          <ul className="flex flex-col gap-1 text-body-compact">
            {p.items.map((i) => (
              <li key={i.candidateId} className="flex flex-wrap items-center gap-2">
                <span className="num">v{i.version}</span>
                <OutcomeBadge outcome={i.outcome} />
                <span className="num text-ink-subtle">
                  {String(i.settings?.['region'])} · D{String(i.settings?.['delay'])} ·{' '}
                  {String(i.settings?.['universe'])}
                </span>
                {i.blockers.map((b) => (
                  <span key={b} className="text-loss">
                    {b}
                  </span>
                ))}
              </li>
            ))}
          </ul>
        </>
      )}
      <div className="flex flex-wrap items-center gap-2">
        <Button
          variant="primary"
          disabled={reason != null}
          onClick={() => {
            setKey(crypto.randomUUID())
            setAck(false)
            setOpen(true)
          }}
        >
          Create simulation task
        </Button>
        {reason && <span className="text-caption text-ink-subtle">Unavailable: {reason}</span>}
      </div>
      <Confirm
        open={open}
        onOpenChange={setOpen}
        title="Create this task?"
        confirmLabel={create.isPending ? 'Creating…' : 'Create task (does not start it)'}
        pending={create.isPending}
        onConfirm={() => {
          if (p?.needsReview && !ack) return
          create.mutate()
        }}
      >
        <div className="flex flex-col gap-2">
          <p>
            {p?.simulations ?? 0} simulation(s) in {p?.batches ?? 0} batch(es) will be stored as a
            task. They count against your BRAIN allowance only when you run it.
          </p>
          {p?.needsReview && (
            <Checkbox
              checked={ack}
              onChange={setAck}
              label="I have read the review warnings on the candidates I selected."
            />
          )}
          {p?.needsReview && !ack && (
            <p className="text-caption text-ink-subtle">Tick the box to continue.</p>
          )}
          {create.error != null && <ErrorNotice error={create.error} />}
        </div>
      </Confirm>
    </section>
  )
}

export type { SimSettings }
