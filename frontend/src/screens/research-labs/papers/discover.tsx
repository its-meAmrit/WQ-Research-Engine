/**
 * Discover: find public research on a topic, read the results you pick, and save the ideas in
 * them. Searching reads results only; reading a page is a separate step you start. Nothing here
 * runs a simulation or reaches BRAIN.
 */

import { useMutation, useQueries, useQuery, useQueryClient } from '@tanstack/react-query'
import { SearchIcon, SparklesIcon } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { toast } from 'sonner'
import { useScope } from '@/lib/scope'
import {
  Badge,
  Button,
  Checkbox,
  Chips,
  Empty,
  ErrorNotice,
  Field,
  Input,
  LINK,
  Notice,
  Panel,
  Progress,
  Skeleton,
  Spinner,
  Switch,
  Textarea,
} from '@/ui/kit'
import { Select } from '@/ui/overlay'
import { ScopePicker } from '@/ui/scope-picker'
import {
  type DiscoverBody,
  type DiscoverOptions,
  discover,
  type Hit,
  type Job,
  research,
} from './api'
import { IdeaCard } from './ideas'
import { KEYS } from './shared'

const HIT_TONE: Record<Hit['status'], 'profit' | 'warn' | 'loss' | 'muted' | 'neutral'> = {
  found: 'neutral',
  extracted: 'profit',
  imported: 'profit',
  duplicate: 'muted',
  rejected: 'warn',
  restricted: 'warn',
  failed: 'loss',
}

const HIT_LABEL: Record<Hit['status'], string> = {
  found: 'Found',
  extracted: 'Ideas saved',
  imported: 'Imported as a paper',
  duplicate: 'Already have it',
  rejected: 'No usable idea',
  restricted: 'Metadata only',
  failed: 'Could not read',
}

const DEPTHS = [
  { value: 'basic' as const, label: 'Basic search (fewer credits)' },
  { value: 'advanced' as const, label: 'Advanced search (more credits)' },
]

/** Polls one job until it stops; the screen reads its progress and result from here. */
function useJob(id: string | null) {
  return useQuery({
    queryKey: ['research', 'job', id],
    queryFn: () => research.job(id as string),
    enabled: id != null,
    refetchInterval: (q) => (q.state.data && !q.state.data.active ? false : 1000),
  })
}

function JobStrip({ job, label }: { job: Job | undefined; label: string }) {
  const client = useQueryClient()
  const cancel = useMutation({
    mutationFn: () => research.cancelJob(job?.id as string),
    onSuccess: (next) => client.setQueryData(['research', 'job', next.id], next),
  })
  if (!job) return null
  return (
    <div className="flex flex-col gap-2 rounded-md border border-hairline p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-body-compact text-ink">
          {label}: {job.message}
        </span>
        {job.active && (
          <Button size="sm" loading={cancel.isPending} onClick={() => cancel.mutate()}>
            Cancel
          </Button>
        )}
      </div>
      {job.active && <Progress value={job.progress} label={`${label} progress`} />}
      {job.status === 'failed' && (
        <Notice tone="warn" title="This did not finish">
          {job.errorMessage}
        </Notice>
      )}
      {job.status === 'interrupted' && (
        <Notice tone="warn" title="Interrupted">
          {job.errorMessage ?? 'The app closed while this was running. Nothing from it was saved.'}
        </Notice>
      )}
      {job.status === 'cancelled' && <p className="text-caption text-ink-subtle">Cancelled.</p>}
    </div>
  )
}

export function Discover({ options }: { options: DiscoverOptions | undefined }) {
  const [topic, setTopic] = useState('')
  const [market, setMarket] = useState('')
  const [queryText, setQueryText] = useState('')
  const [sources, setSources] = useState<string[]>([])
  const [provider, setProvider] = useState<string>('auto')
  const [maxResults, setMaxResults] = useState(10)
  const [depth, setDepth] = useState<'basic' | 'advanced'>('basic')
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [domains, setDomains] = useState('')
  const [urls, setUrls] = useState('')
  const [jobId, setJobId] = useState<string | null>(null)

  const searchable = useMemo(
    () => (options?.sources ?? []).filter((s) => s.enabled && s.adapter !== 'direct_url'),
    [options],
  )
  // Until the person chooses, every enabled search source other than feeds is used.
  useEffect(() => {
    if (sources.length === 0 && options)
      setSources(
        options.sources
          .filter((s) => s.enabled && (s.adapter === 'scholarly' || s.adapter === 'tavily'))
          .map((s) => s.id),
      )
  }, [options, sources.length])

  const lines = (text: string) =>
    text
      .split('\n')
      .map((l) => l.trim())
      .filter(Boolean)
  const body: DiscoverBody = {
    topic: topic.trim(),
    market: market.trim(),
    queries: lines(queryText),
    sources,
    provider: provider === 'auto' ? null : provider,
    maxResults,
    depth,
    startDate: startDate || null,
    endDate: endDate || null,
    includeDomains: lines(domains.replaceAll(',', '\n')),
    urls: lines(urls),
  }
  const ready =
    body.topic.length >= 3 || (body.urls?.length ?? 0) > 0 || (body.queries?.length ?? 0) > 0

  const estimate = useQuery({
    queryKey: ['research', 'discover', 'estimate', JSON.stringify(body)],
    queryFn: ({ signal }) => discover.estimate(body, signal),
    enabled: ready && sources.length > 0,
    retry: false,
    staleTime: 30_000,
  })

  const suggest = useMutation({
    mutationFn: () => discover.queries(topic.trim(), market.trim()),
    onSuccess: (proposals) => setQueryText(proposals.map((p) => p.text).join('\n')),
  })
  const start = useMutation({
    mutationFn: () => discover.start(body),
    onSuccess: (started) => {
      setJobId(started.jobId)
      toast(started.existing ? 'That search is already running' : 'Searching', {
        description: started.message,
      })
    },
  })

  const job = useJob(jobId)
  const hits = useQuery({
    queryKey: KEYS.hits(jobId ?? ''),
    queryFn: () => discover.hits(jobId as string),
    enabled: jobId != null && job.data?.status === 'completed',
  })

  const providerChoices = [
    { value: 'auto', label: 'Automatic (your default)' },
    ...(options?.providers ?? []).map((p) => ({
      value: p.id,
      label: `${p.label}${p.configured ? '' : ' (no key)'}`,
    })),
  ]
  const budget = (options?.providers ?? []).find((p) => p.id === estimate.data?.provider)?.usage

  return (
    <div className="flex min-w-0 flex-col gap-4">
      <Panel
        title="Discover"
        description="Search public research for ideas. Only search results are read until you choose pages to read."
      >
        <form
          className="flex flex-col gap-4"
          onSubmit={(e) => {
            e.preventDefault()
            if (ready) start.mutate()
          }}
        >
          <Field label="Topic">
            <div className="flex gap-2">
              <Input
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder="e.g. earnings surprise cross-sectional equity anomaly"
                aria-label="Topic"
              />
              <Button
                loading={suggest.isPending}
                disabled={topic.trim().length < 3}
                onClick={() => suggest.mutate()}
              >
                <SparklesIcon aria-hidden /> Suggest queries
              </Button>
            </div>
          </Field>
          <div className="flex flex-wrap gap-1.5">
            {(options?.topics ?? []).map((t) => (
              <Button key={t} size="sm" onClick={() => setTopic(t)}>
                {t}
              </Button>
            ))}
          </div>
          {suggest.isError && <ErrorNotice error={suggest.error} title="No queries suggested" />}
          <Field
            label="Search queries (one per line, editable)"
            hint={`Leave empty to search the topic itself. At most ${options?.limits.maxQueries ?? 5}.`}
          >
            <Textarea
              value={queryText}
              onChange={(e) => setQueryText(e.target.value)}
              rows={3}
              aria-label="Search queries"
            />
          </Field>

          <div className="flex flex-col gap-1.5">
            <span className="text-caption font-medium text-ink-muted">Sources</span>
            <Chips
              label="Sources"
              value={sources}
              onChange={setSources}
              items={searchable.map((s) => ({
                value: s.id,
                label: s.label,
                title: s.lastError ?? undefined,
              }))}
            />
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <Field label="Market scope (added to each query)">
              <Input
                value={market}
                onChange={(e) => setMarket(e.target.value)}
                placeholder="e.g. US equities"
              />
            </Field>
            <Field label="Search service">
              <Select
                value={provider}
                onChange={setProvider}
                items={providerChoices}
                label="Search service"
              />
            </Field>
            <Field label="From date">
              <Input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
            </Field>
            <Field label="To date">
              <Input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} />
            </Field>
            <Field label={`Max results (up to ${options?.limits.maxResults ?? 20})`}>
              <Input
                type="number"
                min={1}
                max={options?.limits.maxResults ?? 20}
                value={maxResults}
                onChange={(e) =>
                  setMaxResults(
                    Math.min(
                      options?.limits.maxResults ?? 20,
                      Math.max(1, Number(e.target.value) || 1),
                    ),
                  )
                }
              />
            </Field>
            <Field label="Search depth">
              <Select value={depth} onChange={setDepth} items={DEPTHS} label="Search depth" />
            </Field>
          </div>
          <Field label="Only these sites (comma separated, optional)">
            <Input value={domains} onChange={(e) => setDomains(e.target.value)} />
          </Field>
          <Field label="Or pages you name (one public URL per line, optional)">
            <Textarea value={urls} onChange={(e) => setUrls(e.target.value)} rows={2} />
          </Field>

          <EstimateLine
            loading={estimate.isFetching}
            error={estimate.error}
            data={estimate.data}
            budget={budget}
          />

          <div className="flex flex-wrap items-center gap-2">
            <Button
              type="submit"
              variant="primary"
              loading={start.isPending}
              disabled={!ready || sources.length === 0 || job.data?.active === true}
            >
              <SearchIcon aria-hidden /> Discover
            </Button>
            <span className="text-caption text-ink-subtle">
              Starts only when you press this. Nothing runs in the background.
            </span>
          </div>
          {start.isError && <ErrorNotice error={start.error} title="The search did not start" />}
        </form>
      </Panel>

      <JobStrip job={job.data} label="Search" />
      {jobId && job.data?.status === 'completed' && (
        <Results jobId={jobId} hits={hits.data} loading={hits.isPending} options={options} />
      )}
      <SourceList options={options} />
    </div>
  )
}

function EstimateLine({
  loading,
  error,
  data,
  budget,
}: {
  loading: boolean
  error: unknown
  data: import('./api').Estimate | undefined
  budget: { remaining_today?: number; daily_budget?: number } | null | undefined
}) {
  if (error) return <ErrorNotice error={error} title="Cannot search yet" />
  if (!data) return loading ? <Spinner /> : null
  return (
    <div className="flex flex-col gap-1 rounded-md bg-surface-2 p-3 text-body-compact">
      <p className="text-ink">
        Estimated usage: {data.requests} paid search request{data.requests === 1 ? '' : 's'}
        {data.provider ? ` via ${data.provider}` : ''} (about {data.searchCredits} credit
        {data.searchCredits === 1 ? '' : 's'}), plus {data.free} free lookup
        {data.free === 1 ? '' : 's'}.
      </p>
      {data.extractCreditsIfAllRead != null && (
        <p className="text-ink-muted">
          Reading pages is separate: up to about {data.extractCreditsIfAllRead} credits if you read
          all {data.maxResults}.
        </p>
      )}
      {budget?.remaining_today != null && (
        <p className="text-ink-muted">
          {budget.remaining_today} of {budget.daily_budget} requests left today.
        </p>
      )}
      <p className="text-caption text-ink-subtle">{data.note}</p>
      {data.notes.map((n) => (
        <p key={n} className="text-caption text-ink-subtle">
          {n}
        </p>
      ))}
    </div>
  )
}

function Results({
  jobId,
  hits,
  loading,
  options,
}: {
  jobId: string
  hits: Hit[] | undefined
  loading: boolean
  options: DiscoverOptions | undefined
}) {
  const client = useQueryClient()
  const [picked, setPicked] = useState<string[]>([])
  const [model, setModel] = useState<string | null>(null)
  const [withScope, setWithScope] = useState(false)
  const [scope, setScope] = useScope('labs/papers')
  const [extractId, setExtractId] = useState<string | null>(null)

  const models = options?.models ?? []
  useEffect(() => {
    if (model == null && models[0]) setModel(models[0].ref)
  }, [model, models])

  const extract = useMutation({
    mutationFn: () =>
      discover.extract({
        hitIds: picked,
        modelRef: model,
        scope: withScope ? scope : null,
      }),
    onSuccess: (started) => {
      setExtractId(started.jobId)
      setPicked([])
      toast(started.jobId ? 'Reading started' : 'Done', { description: started.message })
      void client.invalidateQueries({ queryKey: KEYS.papers })
    },
  })
  const job = useJob(extractId)
  const finished = job.data?.status === 'completed'
  useEffect(() => {
    if (finished) void client.invalidateQueries({ queryKey: KEYS.hits(jobId) })
  }, [finished, client, jobId])

  if (loading) return <Skeleton className="h-40" label="Loading results" />
  const list = hits ?? []
  const open = list.filter((h) => h.status === 'found' || h.status === 'failed')

  return (
    <Panel
      title={`Results (${list.length})`}
      description="Attribution is shown for every result. Reading a page checks the site's robots.txt first; paywalled or login pages are kept as metadata only."
    >
      <div className="flex flex-col gap-3">
        {list.length === 0 && (
          <Empty title="No results">
            Try different words, a wider date range or another source.
          </Empty>
        )}
        {list.map((h) => (
          <HitCard
            key={h.id}
            hit={h}
            checked={picked.includes(h.id)}
            onCheck={(on) => setPicked((p) => (on ? [...p, h.id] : p.filter((id) => id !== h.id)))}
          />
        ))}
        {open.length > 0 && (
          <div className="flex flex-col gap-3 border-t border-hairline pt-3">
            <div className="flex flex-wrap items-center gap-3">
              <Select
                value={model}
                onChange={setModel}
                items={models.map((m) => ({ value: m.ref, label: m.ref }))}
                label="Model for extraction"
              />
              <span className="flex items-center gap-2 text-body-compact text-ink-muted">
                <Switch
                  checked={withScope}
                  onChange={setWithScope}
                  aria-label="Check ideas against my BRAIN data"
                />
                Check ideas against my BRAIN data
              </span>
              {withScope && <ScopePicker scope={scope} onChange={setScope} />}
            </div>
            {models.length === 0 && (
              <Notice tone="warn">
                No model has an enabled key, so ideas cannot be extracted. Add one under API
                Integration.
              </Notice>
            )}
            <div className="flex flex-wrap items-center gap-2">
              <Button
                variant="primary"
                loading={extract.isPending}
                disabled={picked.length === 0 || models.length === 0 || job.data?.active === true}
                onClick={() => extract.mutate()}
              >
                Extract ideas from {picked.length || 'selected'}
              </Button>
              <span className="text-caption text-ink-subtle">
                At most {options?.limits.maxExtractPerJob ?? 10} at a time.
              </span>
            </div>
            {extract.isError && <ErrorNotice error={extract.error} title="Reading did not start" />}
          </div>
        )}
        <JobStrip job={job.data} label="Reading" />
        {finished && <Preview job={job.data as Job} hits={list} scope={scope} />}
      </div>
    </Panel>
  )
}

function HitCard({
  hit,
  checked,
  onCheck,
}: {
  hit: Hit
  checked: boolean
  onCheck: (on: boolean) => void
}) {
  const selectable = hit.status === 'found' || hit.status === 'failed'
  return (
    <article className="flex flex-col gap-1.5 rounded-md border border-hairline px-3 py-2">
      <div className="flex items-start gap-3">
        <Checkbox
          checked={checked}
          onChange={onCheck}
          disabled={!selectable}
          label={<span className="sr-only">{`Select ${hit.title || hit.url}`}</span>}
        />
        <div className="flex min-w-0 flex-1 flex-col gap-1">
          <p className="text-body font-medium break-words text-ink">{hit.title || hit.url}</p>
          <p className="flex flex-wrap items-center gap-x-2 gap-y-1 text-caption text-ink-subtle">
            {hit.domain && <Badge tone="outline">{hit.domain}</Badge>}
            <span>via {hit.provider}</span>
            {hit.published && <span>published {hit.published}</span>}
            <span>retrieved {hit.retrievedAt?.slice(0, 16).replace('T', ' ')}</span>
            {hit.fromCache && <span>(cached)</span>}
          </p>
          {hit.snippet && (
            <p className="text-body-compact text-pretty text-ink-muted">{hit.snippet}</p>
          )}
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone={HIT_TONE[hit.status]}>{HIT_LABEL[hit.status]}</Badge>
            {hit.statusDetail && (
              <span className="text-caption text-ink-subtle">{hit.statusDetail}</span>
            )}
            {hit.status !== 'restricted' || hit.canonicalUrl ? (
              <a href={hit.url} target="_blank" rel="noopener noreferrer" className={LINK}>
                Open source
              </a>
            ) : null}
          </div>
        </div>
      </div>
    </article>
  )
}

/** The ideas an extraction produced, each with Save, Reject, Mark reviewed and Send to Alpha
 *  Builder, and the pages that produced none, with why. */
function Preview({
  job,
  hits,
  scope,
}: {
  job: Job
  hits: Hit[]
  scope: import('@/api/types').Scope
}) {
  const result = (job.result ?? {}) as {
    ideaIds?: string[]
    outcomes?: { hitId: string; status: string; detail: string; linked: unknown[] }[]
  }
  const ids = result.ideaIds ?? []
  const ideas = useQueries({
    queries: ids.map((id) => ({
      queryKey: ['research', 'idea', id],
      queryFn: () => research.idea(id),
    })),
  })
  const skipped = (result.outcomes ?? []).filter((o) => o.status !== 'extracted')
  const title = (id: string) => hits.find((h) => h.id === id)?.title ?? id

  return (
    <div className="flex flex-col gap-3 border-t border-hairline pt-3">
      <h3 className="text-body font-semibold text-ink">
        {ids.length} idea{ids.length === 1 ? '' : 's'} saved to your library
      </h3>
      {skipped.map((o) => (
        <Notice key={o.hitId} tone="info" title={title(o.hitId)}>
          {o.detail}
        </Notice>
      ))}
      {ideas.map((q, i) =>
        q.data ? (
          <IdeaCard key={ids[i]} summary={q.data} scope={scope} />
        ) : (
          <Skeleton key={ids[i]} className="h-24" label="Loading idea" />
        ),
      )}
    </div>
  )
}

function SourceList({ options }: { options: DiscoverOptions | undefined }) {
  const client = useQueryClient()
  const [label, setLabel] = useState('')
  const [feed, setFeed] = useState('')
  const refresh = () => client.invalidateQueries({ queryKey: ['research', 'discover', 'options'] })
  const toggle = useMutation({
    mutationFn: (v: { id: string; enabled: boolean }) =>
      discover.updateSource(v.id, { enabled: v.enabled }),
    onSuccess: refresh,
  })
  const add = useMutation({
    mutationFn: () => discover.addFeed(label.trim(), feed.trim()),
    onSuccess: () => {
      setLabel('')
      setFeed('')
      return refresh()
    },
  })
  const remove = useMutation({ mutationFn: discover.removeSource, onSuccess: refresh })

  return (
    <Panel
      title="Sources"
      description="Where Discover looks. Each can be switched off. The sites listed for web sources are starting filters, not endorsements."
    >
      <div className="flex flex-col gap-3">
        <ul className="flex flex-col gap-2">
          {(options?.sources ?? []).map((s) => (
            <li
              key={s.id}
              className="flex flex-wrap items-center justify-between gap-3 rounded-md border border-hairline px-3 py-2"
            >
              <div className="flex min-w-0 flex-col gap-0.5">
                <span className="text-body text-ink">{s.label}</span>
                <span className="text-caption text-ink-subtle">
                  {s.category} · {s.adapter}
                  {s.lastSuccessAt ? ` · last worked ${s.lastSuccessAt.slice(0, 10)}` : ''}
                </span>
                {s.lastError && (
                  <span className="text-caption text-loss break-words">
                    Last error: {s.lastError}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2">
                {!s.builtin && (
                  <Button size="sm" onClick={() => remove.mutate(s.id)}>
                    Remove
                  </Button>
                )}
                <Switch
                  checked={s.enabled}
                  onChange={(enabled) => toggle.mutate({ id: s.id, enabled })}
                  aria-label={`${s.label} enabled`}
                />
              </div>
            </li>
          ))}
        </ul>
        {(toggle.isError || remove.isError) && (
          <ErrorNotice error={toggle.error ?? remove.error} title="That change failed" />
        )}
        <form
          className="flex flex-wrap gap-2"
          onSubmit={(e) => {
            e.preventDefault()
            add.mutate()
          }}
        >
          <Input
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            placeholder="Feed name"
            aria-label="Feed name"
          />
          <Input
            value={feed}
            onChange={(e) => setFeed(e.target.value)}
            placeholder="https://example.com/feed.xml"
            aria-label="RSS or Atom feed URL"
          />
          <Button type="submit" loading={add.isPending} disabled={!label.trim() || !feed.trim()}>
            Add feed
          </Button>
        </form>
        {add.isError && <ErrorNotice error={add.error} title="That feed was not added" />}
      </div>
    </Panel>
  )
}
