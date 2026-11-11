/** The ideas taken from a paper, each checked against the market the person has selected. */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { CheckIcon, XIcon } from 'lucide-react'
import { type ReactNode, useEffect, useState } from 'react'
import { toast } from 'sonner'
import type { Scope } from '@/api/types'
import { scopeLabel } from '@/api/types'
import { useScope } from '@/lib/scope'
import {
  Badge,
  Button,
  Disclosure,
  Empty,
  ErrorNotice,
  LINK,
  Notice,
  Panel,
  Skeleton,
  Textarea,
} from '@/ui/kit'
import { Select } from '@/ui/overlay'
import { ScopePicker } from '@/ui/scope-picker'
import type { Assessment, Idea, IdeaDetail, IdeaStatus } from './api'
import { research } from './api'
import { KEYS, RankingView, ReadinessBadge, SupportBadge, useBuilderLauncher } from './shared'

const ANY = 'any'
const STATUS = [
  { value: ANY, label: 'All ideas' },
  { value: 'new', label: 'Unreviewed' },
  { value: 'saved', label: 'Saved' },
  { value: 'rejected', label: 'Rejected' },
]

const same = (a: Assessment['scope'], b: Scope) =>
  a.instrumentType === b.instrumentType &&
  a.region === b.region &&
  a.delay === b.delay &&
  a.universe === b.universe

export function Ideas({ sourceId }: { sourceId: string }) {
  const [scope, setScope] = useScope('labs/papers')
  const [status, setStatus] = useState(ANY)
  const query = useQuery({
    queryKey: [...KEYS.ideas, sourceId, status],
    queryFn: () => research.ideas({ sourceId, ...(status === ANY ? {} : { status }) }),
  })

  return (
    <Panel
      title="Ideas"
      description="Each idea separates what the authors claim from what the model suggests."
      actions={<Select value={status} onChange={setStatus} items={STATUS} label="Show" />}
    >
      <div className="flex flex-col gap-4">
        <div className="flex flex-col gap-1.5">
          <span className="text-caption font-medium text-ink-muted">Check against</span>
          <div className="flex flex-wrap items-center gap-2">
            <ScopePicker scope={scope} onChange={setScope} />
          </div>
          <p className="text-caption text-ink-subtle">
            Only this market is searched. It is never switched for you.
          </p>
        </div>
        {query.isPending && <Skeleton className="h-32" label="Loading ideas" />}
        {query.isError && <ErrorNotice error={query.error} title="Ideas could not be loaded" />}
        {query.data?.length === 0 && (
          <Empty title="No ideas to show">Run the analysis above to extract them.</Empty>
        )}
        {query.data?.map((idea) => (
          <IdeaCard key={idea.id} summary={idea} scope={scope} />
        ))}
      </div>
    </Panel>
  )
}

function List({
  title,
  items,
  hint,
}: {
  title: string
  items: string[] | undefined
  hint?: string | undefined
}) {
  if (!items || items.length === 0) return null
  return (
    <div className="flex flex-col gap-1">
      <h4 className="text-caption font-medium text-ink-muted">{title}</h4>
      {hint && <p className="text-caption text-ink-subtle">{hint}</p>}
      <ul className="list-disc pl-5 text-body text-pretty text-ink">
        {items.map((t) => (
          <li key={t}>{t}</li>
        ))}
      </ul>
    </div>
  )
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  if (!children) return null
  return (
    <div className="flex flex-col gap-0.5">
      <h4 className="text-caption font-medium text-ink-muted">{label}</h4>
      <p className="text-body text-pretty text-ink">{children}</p>
    </div>
  )
}

export function IdeaCard({ summary, scope }: { summary: Idea; scope: Scope }) {
  const client = useQueryClient()
  // The list already carries the whole idea; the candidates (drafts) come from the detail.
  const detail = useQuery({
    queryKey: ['research', 'idea', summary.id],
    queryFn: () => research.idea(summary.id),
  })
  const idea: IdeaDetail = detail.data ?? {
    ...summary,
    candidates: [],
    linkedSources: [],
    related: [],
  }
  const refresh = (next: IdeaDetail) => {
    client.setQueryData(['research', 'idea', idea.id], next)
    void client.invalidateQueries({ queryKey: KEYS.ideas })
    void client.invalidateQueries({ queryKey: KEYS.papers })
  }

  const [note, setNote] = useState(idea.note)
  useEffect(() => setNote(idea.note), [idea.note])

  const patch = useMutation({
    mutationFn: (body: {
      status?: IdeaStatus
      note?: string
      favorite?: boolean
      reviewed?: boolean
    }) => research.patchIdea(idea.id, body),
    onSuccess: refresh,
  })
  const assess = useMutation({
    mutationFn: () => research.assess(idea.id, scope),
    onSuccess: refresh,
  })
  const openBuilder = useBuilderLauncher()
  const handOff = useMutation({
    mutationFn: () => research.handOff(idea.id, scope),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ['research', 'idea', idea.id] })
      toast('Draft sent to the Alpha Builder', {
        description:
          'It carries the idea and its source only. No expression was written and nothing was simulated.',
        action: { label: 'Open Alpha Builder', onClick: () => openBuilder(idea.id) },
      })
    },
  })

  const a = idea.brainAssessment
  const current = a != null && same(a.scope, scope)
  const claims = idea.claims
  const rejected = idea.status === 'rejected'

  return (
    <article
      className={`flex flex-col gap-4 rounded-md border border-hairline p-4 ${rejected ? 'opacity-70' : ''}`}
    >
      <header className="flex flex-col gap-2">
        <div className="flex flex-wrap items-start justify-between gap-2">
          <h3 className="min-w-0 text-body font-semibold break-words text-ink">{idea.title}</h3>
          <div className="flex flex-wrap gap-1.5">
            <SupportBadge support={idea.support} />
            <ReadinessBadge readiness={idea.readiness} />
            {idea.favorite && <Badge tone="warn">Favourite</Badge>}
            {idea.reviewedAt && <Badge tone="muted">Reviewed</Badge>}
            {idea.status !== 'new' && (
              <Badge tone={idea.status === 'saved' ? 'profit' : 'muted'}>
                {idea.status === 'saved' ? 'Saved' : 'Rejected'}
              </Badge>
            )}
          </div>
        </div>
        <p className="text-caption text-ink-subtle">{idea.topic}</p>
        <SourceLine idea={idea} />
      </header>

      {idea.evidenceScope === 'web_page' && (
        <Notice tone="info">
          Taken from a public web page. Performance figures it quotes are the page's own claims;
          they have not been checked or reproduced here.
        </Notice>
      )}

      {idea.evidenceScope === 'abstract' && (
        <Notice tone="warn">
          Read from the abstract only, so this is a lead, not a finding the paper is shown to make.
        </Notice>
      )}
      {idea.support === 'speculative' && idea.evidenceScope !== 'abstract' && (
        <Notice tone="warn">
          No quote the model cited could be found in the paper. Treat this as speculation.
        </Notice>
      )}

      <Row label="Main finding">{idea.mainFinding}</Row>
      <List
        title="Claimed by the authors"
        items={claims.authorClaims}
        hint="Each needs a quote from the paper below."
      />
      <List
        title="The model's interpretation"
        items={claims.modelInterpretations}
        hint="The model's reading; the paper does not say this."
      />
      <List
        title="Proposed BRAIN hypotheses"
        items={claims.proposedHypotheses}
        hint="New, and not tested. Speculative until a simulation you run says otherwise."
      />
      <Row label="Economic explanation">{idea.economicHypothesis}</Row>
      <Row label="Expected relationship">{idea.expectedRelationship}</Row>
      <Row label="Signal">{idea.signalDefinition}</Row>
      <Row label="How it might be built on BRAIN">{idea.brainImplementation}</Row>
      <List title="Assumptions" items={idea.assumptions} />
      <List
        title="Performance the source reports"
        items={idea.claims.reportedPerformance}
        hint={idea.claims.reportedPerformanceNote}
      />
      <RankingView ranking={idea.ranking} />
      <LinkedSources idea={idea} />

      <Disclosure summary="Methodology and design">
        <div className="flex flex-col gap-3">
          <Row label="Method">{idea.methodology}</Row>
          <Row label="Direction">{idea.signalDirection}</Row>
          <Row label="Lookback">{idea.lookback}</Row>
          <Row label="Holding period">{idea.holdingPeriod}</Row>
          <Row label="Universe studied">{idea.universe}</Row>
          <Row label="Point-in-time needs">{idea.pointInTime}</Row>
          {idea.reportedFormula && (
            <div className="flex flex-col gap-0.5">
              <h4 className="text-caption font-medium text-ink-muted">Formula as extracted</h4>
              <code className="rounded-sm bg-surface-2 px-2 py-1 text-body-compact break-all">
                {idea.reportedFormula}
              </code>
              {idea.formulaNeedsReview && (
                <p className="text-caption text-ink-subtle">
                  Extracted from PDF text and may be garbled. Check it in the original.
                </p>
              )}
            </div>
          )}
          <List title="Implementation variants" items={idea.implementationVariants} />
          <List title="Statistical limitations" items={idea.limitations} />
          <List title="Look-ahead concerns" items={idea.lookaheadConcerns} />
          <List title="Missing information" items={idea.missingInformation} />
        </div>
      </Disclosure>

      <Disclosure
        summary={`Evidence (${idea.evidence.filter((e) => e.verified).length} of ${idea.evidence.length} quotes found)`}
      >
        {idea.evidence.length === 0 ? (
          <p className="text-body-compact text-ink-subtle">No passage was cited for this idea.</p>
        ) : (
          <ul className="flex flex-col gap-3">
            {idea.evidence.map((e, i) => (
              <li key={i} className="flex flex-col gap-1">
                <blockquote className="border-l-2 border-hairline-strong pl-3 text-body-compact text-pretty text-ink-muted">
                  “{e.quote}”
                </blockquote>
                <p className="flex flex-wrap items-center gap-1.5 text-caption text-ink-subtle">
                  {e.verified ? (
                    <Badge tone="profit">
                      <CheckIcon className="size-3" aria-hidden /> Found
                    </Badge>
                  ) : (
                    <Badge tone="warn">
                      <XIcon className="size-3" aria-hidden /> Not verified
                    </Badge>
                  )}
                  <span>{e.location}</span>
                  <span>· supports: {e.supports}</span>
                </p>
              </li>
            ))}
          </ul>
        )}
      </Disclosure>

      <section className="flex flex-col gap-3 rounded-md bg-surface-2 p-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h4 className="text-body font-medium text-ink">Fit with your BRAIN data</h4>
          <Button size="sm" loading={assess.isPending} onClick={() => assess.mutate()}>
            {a ? 'Check again' : 'Check'} in {scopeLabel(scope)}
          </Button>
        </div>
        {assess.isError && <ErrorNotice error={assess.error} title="The check failed" />}
        {a && !current && (
          <Notice tone="info">
            This result is for {scopeLabel(a.scope)}, not the market selected above.
          </Notice>
        )}
        {a ? (
          <AssessmentView a={a} />
        ) : (
          <p className="text-body-compact text-ink-subtle">
            Not checked yet. The check looks up fields and operators in your local catalogue. It
            runs no simulation.
          </p>
        )}
      </section>

      <footer className="flex flex-col gap-3">
        <Textarea
          value={note}
          onChange={(e) => setNote(e.target.value)}
          onBlur={() => note !== idea.note && patch.mutate({ note })}
          placeholder="Your notes on this idea"
          aria-label="Your notes on this idea"
        />
        {patch.isError && <ErrorNotice error={patch.error} title="That change was not saved" />}
        {handOff.isError && <ErrorNotice error={handOff.error} title="Could not hand this off" />}
        <div className="flex flex-wrap gap-2">
          <Button
            variant={idea.favorite ? 'primary' : 'secondary'}
            onClick={() => patch.mutate({ favorite: !idea.favorite })}
          >
            {idea.favorite ? 'Favourite' : 'Mark favourite'}
          </Button>
          <Button onClick={() => patch.mutate({ reviewed: !idea.reviewedAt })}>
            {idea.reviewedAt ? 'Reviewed' : 'Mark reviewed'}
          </Button>
          <Button
            variant={idea.status === 'saved' ? 'primary' : 'secondary'}
            onClick={() => patch.mutate({ status: idea.status === 'saved' ? 'new' : 'saved' })}
          >
            {idea.status === 'saved' ? 'Saved' : 'Save'}
          </Button>
          <Button onClick={() => patch.mutate({ status: rejected ? 'new' : 'rejected' })}>
            {rejected ? 'Restore' : 'Reject'}
          </Button>
          <Button
            disabled={!current || a?.readiness === 'blocked' || rejected}
            loading={handOff.isPending}
            onClick={() => handOff.mutate()}
          >
            Send to Alpha Builder
          </Button>
          <Button disabled={rejected} onClick={() => openBuilder(idea.id)}>
            Open in Alpha Builder
          </Button>
        </div>
        {idea.candidates.length > 0 && (
          <p className="text-caption text-ink-subtle">
            {idea.candidates.length} draft{idea.candidates.length === 1 ? '' : 's'} sent. Drafts
            hold no expression and have not been simulated.
          </p>
        )}
      </footer>
    </article>
  )
}

/** Where the idea came from: the page or paper, its site, when it was retrieved, and a link. */
function SourceLine({ idea }: { idea: IdeaDetail }) {
  const s = idea.source
  if (!s) return null
  const link = s.canonicalUrl ?? (s.doi ? `https://doi.org/${s.doi}` : null)
  return (
    <p className="flex flex-wrap items-center gap-x-2 gap-y-1 text-caption text-ink-subtle">
      <span className="break-words">{s.title}</span>
      {s.domain && <Badge tone="outline">{s.domain}</Badge>}
      {s.retrievedAt && <span>retrieved {s.retrievedAt.slice(0, 10)}</span>}
      {link && (
        <a href={link} target="_blank" rel="noopener noreferrer" className={LINK}>
          Open source
        </a>
      )}
    </p>
  )
}

/** Other sources that state the same hypothesis, and neighbours worth comparing. */
function LinkedSources({ idea }: { idea: IdeaDetail }) {
  if (idea.linkedSources.length === 0 && idea.related.length === 0) return null
  return (
    <section className="flex flex-col gap-2 rounded-md border border-hairline p-3">
      {idea.linkedSources.length > 0 && (
        <div className="flex flex-col gap-1">
          <h4 className="text-caption font-medium text-ink-muted">
            Also stated by {idea.linkedSources.length} other source
            {idea.linkedSources.length === 1 ? '' : 's'}
          </h4>
          <ul className="flex flex-col gap-1 text-body-compact">
            {idea.linkedSources.map((l) => (
              <li key={l.sourceId} className="break-words">
                {l.canonicalUrl ? (
                  <a
                    href={l.canonicalUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className={LINK}
                  >
                    {l.title}
                  </a>
                ) : (
                  l.title
                )}
                <span className="text-caption text-ink-subtle">
                  {' '}
                  · matched by {l.matchedBy}
                  {l.similarity != null ? ` (${Math.round(l.similarity * 100)}%)` : ''}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
      {idea.related.length > 0 && (
        <div className="flex flex-col gap-1">
          <h4 className="text-caption font-medium text-ink-muted">Related ideas in your library</h4>
          <ul className="list-disc pl-5 text-body-compact text-ink-muted">
            {idea.related.map((r) => (
              <li key={r.id}>
                {r.title}{' '}
                <span className="text-caption text-ink-subtle">
                  ({r.relation}, {Math.round(r.similarity * 100)}%)
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  )
}

const SEVERITY: Record<string, 'muted' | 'warn' | 'loss'> = {
  info: 'muted',
  adapt: 'warn',
  block: 'loss',
}

function AssessmentView({ a }: { a: Assessment }) {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <ReadinessBadge readiness={a.readiness} />
        <span className="text-caption text-ink-subtle">for {scopeLabel(a.scope)}</span>
      </div>
      {a.reasons.length > 0 && (
        <ul className="flex flex-col gap-1.5">
          {a.reasons.map((r, i) => (
            <li
              key={i}
              className="flex items-start gap-2 text-body-compact text-pretty text-ink-muted"
            >
              <Badge tone={SEVERITY[r.severity] ?? 'muted'}>{r.severity}</Badge>
              <span>{r.message}</span>
            </li>
          ))}
        </ul>
      )}
      {a.variables.length > 0 && (
        <div className="flex flex-col gap-2">
          <h5 className="text-caption font-medium text-ink-muted">Required data</h5>
          {a.variables.map((v) => (
            <div key={v.name} className="flex flex-col gap-1 rounded-sm border border-hairline p-2">
              <div className="flex flex-wrap items-center gap-1.5 text-body-compact text-ink">
                <span className="font-medium">{v.name}</span>
                <Badge
                  tone={
                    v.status === 'candidates'
                      ? 'profit'
                      : v.status === 'missing' || v.status === 'unsupported'
                        ? v.essential
                          ? 'loss'
                          : 'warn'
                        : 'warn'
                  }
                >
                  {v.status === 'candidates'
                    ? 'Candidates found'
                    : v.status === 'other_scope'
                      ? 'Other markets only'
                      : v.status === 'unsupported'
                        ? 'Not on BRAIN'
                        : 'Missing'}
                </Badge>
                {!v.essential && <Badge tone="muted">optional</Badge>}
              </div>
              {v.candidates.length > 0 && (
                <ul className="flex flex-col gap-0.5 text-caption text-ink-subtle">
                  {v.candidates.map((c) => (
                    <li key={c.fieldId} className="truncate" title={c.description}>
                      <code className="text-ink-muted">{c.fieldId}</code> · {c.description}
                      {c.pointInTime !== 'point_in_time' && ` · point-in-time: ${c.pointInTime}`}
                    </li>
                  ))}
                </ul>
              )}
              {v.availableElsewhere.length > 0 && (
                <p className="text-caption text-ink-subtle">
                  Exists in:{' '}
                  {v.availableElsewhere
                    .slice(0, 4)
                    .map((p) => `${p.fieldId} ${p.region}/D${p.delay}/${p.universe}`)
                    .join(', ')}
                  . Not substituted.
                </p>
              )}
            </div>
          ))}
          <p className="text-caption text-ink-subtle">{a.note}</p>
        </div>
      )}
      {a.operators.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {a.operators.map((o) => (
            <Badge
              key={o.name}
              tone={
                o.status === 'supported' ? 'profit' : o.status === 'unsupported' ? 'loss' : 'warn'
              }
            >
              {o.name}: {o.status}
            </Badge>
          ))}
        </div>
      )}
    </div>
  )
}
