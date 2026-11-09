/** One paper: where it came from, how much of it is readable, and the analysis run on it. */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { fmt } from '@/lib/format'
import {
  Badge,
  Button,
  Disclosure,
  Empty,
  ErrorNotice,
  KV,
  Notice,
  Panel,
  Skeleton,
} from '@/ui/kit'
import { Select } from '@/ui/overlay'
import { type Job, type Options, research } from './api'
import { Ideas } from './ideas'
import { AccessBadge, JOB_KIND, KEYS } from './shared'

const STALLED = new Set(['failed', 'interrupted', 'cancelled'])

export function PaperView({
  id,
  options,
  jobs,
}: {
  id: string
  options: Options | undefined
  jobs: Job[]
}) {
  const paper = useQuery({ queryKey: KEYS.paper(id), queryFn: () => research.paper(id) })
  const mine = jobs.filter((j) => j.sourceId === id)
  const latest = mine.find((j) => !j.active)

  if (paper.isPending) return <Skeleton className="h-64" label="Loading the paper" />
  if (paper.isError)
    return <ErrorNotice error={paper.error} title="The paper could not be loaded" />
  const p = paper.data
  const doc = p.document

  return (
    <div className="flex min-w-0 flex-col gap-4">
      <Panel title={p.title} description={p.authors.join(', ') || 'Authors not found'}>
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap items-center gap-1.5">
            <AccessBadge access={p.accessStatus} />
            {p.topics.map((t) => (
              <Badge key={t} tone="outline">
                {t}
              </Badge>
            ))}
            {p.license && <Badge tone="muted">{p.license}</Badge>}
          </div>
          {p.accessStatus !== 'full_text' && (
            <Notice tone="warn" title="Metadata/abstract only">
              {p.accessDetail} Ideas from this paper are read from the abstract alone, so none can
              be marked source-backed. Upload the PDF for a full read.
            </Notice>
          )}
          {p.abstract && <p className="text-body text-pretty text-ink-muted">{p.abstract}</p>}
          <KV
            items={[
              ['Published', p.publishedAt?.slice(0, 10) ?? '—'],
              ['Venue', p.venue ?? '—'],
              ['DOI', p.doi ?? '—'],
              ['arXiv', p.arxivId ?? '—'],
              ['Imported by', p.importMethod],
              ['Retrieved', p.retrievedAt ? fmt.date(p.retrievedAt) : '—'],
            ]}
          />
          {p.canonicalUrl && (
            <a
              href={p.canonicalUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="truncate text-body-compact text-link underline underline-offset-2"
            >
              {p.canonicalUrl}
            </a>
          )}
          {doc && (
            <Disclosure
              summary={`${doc.pages} pages · ${doc.sections.length} sections · ${doc.equations.length} equations`}
            >
              <div className="flex flex-col gap-3 text-body-compact">
                <ul className="flex flex-col gap-1">
                  {doc.sections.map((s) => (
                    <li key={s.id} className="flex justify-between gap-3 text-ink-muted">
                      <span className="truncate">{s.heading}</span>
                      <span className="num shrink-0 text-ink-subtle">
                        p.{s.pageStart}
                        {s.pageEnd !== s.pageStart && `–${s.pageEnd}`}
                      </span>
                    </li>
                  ))}
                </ul>
                {doc.equations.some((e) => e['needsReview'] ?? e['needs_review']) && (
                  <Notice tone="warn">
                    Some equations could not be read reliably from the PDF and are flagged for
                    review. Check them in the original before relying on a formula.
                  </Notice>
                )}
              </div>
            </Disclosure>
          )}
        </div>
      </Panel>

      {mine
        .filter((j) => j.active)
        .map((job) => (
          <Notice key={job.id} tone="info" title={`${JOB_KIND[job.kind] ?? job.kind} running`}>
            {job.message}
          </Notice>
        ))}
      {latest && STALLED.has(latest.status) && <JobProblem job={latest} />}

      {p.hasText && (
        <Analyze
          id={id}
          options={options}
          busy={mine.some((j) => j.active)}
          abstractOnly={p.accessStatus !== 'full_text'}
        />
      )}
      {!p.hasText && (
        <Empty title="Nothing to read yet">
          This paper has no readable text or abstract. Upload its PDF to analyze it.
        </Empty>
      )}

      <Ideas sourceId={id} />
    </div>
  )
}

function JobProblem({ job }: { job: Job }) {
  const title =
    job.status === 'interrupted'
      ? 'The last run was interrupted'
      : job.status === 'cancelled'
        ? 'The last run was cancelled'
        : 'The last run failed'
  return (
    <Notice tone={job.status === 'failed' ? 'error' : 'warn'} title={title}>
      {job.status === 'interrupted'
        ? 'The app stopped while it was running, so nothing from it was saved. Run it again.'
        : (job.errorMessage ?? 'Nothing was saved.')}
      {job.errorDetail && job.errorDetail.length > 0 && (
        <ul className="mt-1 list-disc pl-4 text-caption">
          {job.errorDetail.slice(0, 5).map((d, i) => (
            <li key={i}>{String(d['msg'] ?? d['message'] ?? JSON.stringify(d))}</li>
          ))}
        </ul>
      )}
    </Notice>
  )
}

function Analyze({
  id,
  options,
  busy,
  abstractOnly,
}: {
  id: string
  options: Options | undefined
  busy: boolean
  abstractOnly: boolean
}) {
  const client = useQueryClient()
  const models = options?.models ?? []
  const [model, setModel] = useState<string | null>(null)
  const ref = model ?? models[0]?.ref ?? null
  const chosen = models.find((m) => m.ref === ref)

  const preview = useQuery({
    queryKey: ['research', 'preview', id],
    queryFn: () => research.preview(id),
    staleTime: 60_000,
  })
  const run = useMutation({
    mutationFn: (force: boolean) => research.analyze(id, { modelRef: ref, force }),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: KEYS.jobs })
      void client.invalidateQueries({ queryKey: KEYS.ideas })
    },
  })
  const p = preview.data

  return (
    <Panel
      title="Extract ideas"
      description="Reads the paper once and lists the testable ideas in it. Nothing is simulated."
    >
      <div className="flex flex-col gap-3">
        {models.length === 0 ? (
          <Notice tone="warn">
            No model is available to run this. Add a provider key in Settings.
          </Notice>
        ) : (
          <Select
            value={ref}
            onChange={setModel}
            items={models.map((m) => ({ value: m.ref, label: `${m.id} (${m.provider})` }))}
            label="Model"
          />
        )}
        {preview.isError && <ErrorNotice error={preview.error} title="No estimate available" />}
        {p && (
          <div className="flex flex-col gap-2 text-body-compact text-ink-muted">
            <KV
              items={[
                [
                  'Reads',
                  abstractOnly
                    ? 'The abstract only'
                    : `${p.sections.filter((s) => s.included).length} sections`,
                ],
                ['Model calls', String(p.requests)],
                ['Input', `≈ ${p.estimatedInputTokens.toLocaleString()} tokens`],
                ['Output', `≤ ${p.estimatedOutputTokens.toLocaleString()} tokens`],
                ...(chosen
                  ? ([['Left today', String(chosen.remainingToday)]] as [string, string][])
                  : []),
              ]}
            />
            {p.truncated && (
              <Notice tone="warn">
                {p.charsLeftOut.toLocaleString()} characters will not be sent, to stay within the
                budget. Later sections may be missing from the result.
              </Notice>
            )}
          </div>
        )}
        {run.isError && <ErrorNotice error={run.error} title="The analysis did not start" />}
        {run.data?.existing && (
          <Notice tone="info">
            {run.data.message} Run again with “Re-run” to bypass the saved result.
          </Notice>
        )}
        <div className="flex gap-2">
          <Button
            variant="primary"
            disabled={busy || !ref}
            loading={run.isPending}
            onClick={() => run.mutate(false)}
          >
            Analyze
          </Button>
          <Button disabled={busy || !ref} onClick={() => run.mutate(true)}>
            Re-run
          </Button>
        </div>
        <p className="text-caption text-ink-subtle">
          Saved, rejected and annotated ideas are kept when you re-run; unreviewed ones are
          replaced.
        </p>
      </div>
    </Panel>
  )
}
