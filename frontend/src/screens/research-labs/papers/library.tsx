/** The saved papers, with the filters the task asks for, and the jobs still running. */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Trash2Icon } from 'lucide-react'
import { useState } from 'react'
import { Badge, Button, Empty, ErrorNotice, Input, Panel, Progress, Skeleton } from '@/ui/kit'
import { Confirm, Select } from '@/ui/overlay'
import { type Job, type Paper, research } from './api'
import { AccessBadge, JOB_KIND, KEYS, READY_LABEL } from './shared'

const ANY = 'any'

export interface Filters {
  search: string
  source: string
  topic: string
  readiness: string
  access: string
  from: string
  saved: boolean
}

export const NO_FILTERS: Filters = {
  search: '',
  source: ANY,
  topic: ANY,
  readiness: ANY,
  access: ANY,
  from: '',
  saved: false,
}

const SOURCES = [
  { value: ANY, label: 'Any source' },
  { value: 'upload', label: 'Uploaded PDF' },
  { value: 'doi', label: 'DOI' },
  { value: 'arxiv', label: 'arXiv' },
  { value: 'url', label: 'URL' },
  { value: 'search', label: 'Search' },
]

const READINESS = [
  { value: ANY, label: 'Any readiness' },
  ...Object.entries(READY_LABEL).map(([value, label]) => ({ value, label })),
]

const ACCESS = [
  { value: ANY, label: 'Any access' },
  { value: 'full_text', label: 'Full text' },
  { value: 'abstract_only', label: 'Abstract only' },
]

const pick = (value: string) => (value === ANY || value === '' ? undefined : value)

export function Library({
  topics,
  selected,
  onSelect,
  jobs,
}: {
  topics: string[]
  selected: string | null
  onSelect: (id: string | null) => void
  jobs: Job[]
}) {
  const client = useQueryClient()
  const [filters, setFilters] = useState<Filters>(NO_FILTERS)
  const [doomed, setDoomed] = useState<Paper | null>(null)
  const change = (patch: Partial<Filters>) => setFilters((f) => ({ ...f, ...patch }))

  const query = useQuery({
    queryKey: [...KEYS.papers, filters],
    queryFn: () =>
      research.papers({
        ...(filters.search.trim() ? { search: filters.search.trim() } : {}),
        ...(pick(filters.source) ? { source: filters.source } : {}),
        ...(pick(filters.topic) ? { topic: filters.topic } : {}),
        ...(pick(filters.readiness) ? { readiness: filters.readiness } : {}),
        ...(pick(filters.access) ? { access: filters.access } : {}),
        ...(filters.from ? { publishedFrom: filters.from } : {}),
        ...(filters.saved ? { saved: true } : {}),
      }),
  })
  const remove = useMutation({
    mutationFn: (id: string) => research.deletePaper(id),
    onSuccess: (_, id) => {
      if (selected === id) onSelect(null)
      setDoomed(null)
      void client.invalidateQueries({ queryKey: KEYS.papers })
      void client.invalidateQueries({ queryKey: KEYS.ideas })
    },
  })
  const dirty = JSON.stringify(filters) !== JSON.stringify(NO_FILTERS)

  return (
    <Panel
      title="Library"
      actions={
        dirty && (
          <Button size="sm" variant="ghost" onClick={() => setFilters(NO_FILTERS)}>
            Clear filters
          </Button>
        )
      }
    >
      <div className="flex flex-col gap-3">
        <Input
          value={filters.search}
          onChange={(e) => change({ search: e.target.value })}
          placeholder="Filter by title, author or idea"
          aria-label="Filter papers by title, author or idea"
        />
        <div className="grid grid-cols-2 gap-2">
          <Select
            value={filters.source}
            onChange={(v) => change({ source: v })}
            items={SOURCES}
            label="Source"
          />
          <Select
            value={filters.topic}
            onChange={(v) => change({ topic: v })}
            items={[
              { value: ANY, label: 'Any topic' },
              ...topics.map((t) => ({ value: t, label: t })),
            ]}
            label="Topic"
          />
          <Select
            value={filters.readiness}
            onChange={(v) => change({ readiness: v })}
            items={READINESS}
            label="Readiness"
          />
          <Select
            value={filters.access}
            onChange={(v) => change({ access: v })}
            items={ACCESS}
            label="Access"
          />
        </div>
        <div className="flex flex-wrap items-center gap-3 text-body-compact text-ink-muted">
          <label className="flex items-center gap-2">
            Published since
            <Input
              type="date"
              value={filters.from}
              onChange={(e) => change({ from: e.target.value })}
              className="w-40"
            />
          </label>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={filters.saved}
              onChange={(e) => change({ saved: e.target.checked })}
            />
            Has saved ideas
          </label>
        </div>

        {jobs.map((job) => (
          <JobRow key={job.id} job={job} onOpen={onSelect} />
        ))}

        {query.isPending && <Skeleton className="h-24" label="Loading papers" />}
        {query.isError && <ErrorNotice error={query.error} title="Papers could not be loaded" />}
        {query.data?.length === 0 && (
          <Empty title={dirty ? 'No paper matches these filters' : 'No papers yet'}>
            {dirty ? 'Clear a filter to see more.' : 'Upload a PDF or import one above.'}
          </Empty>
        )}
        <ul className="flex flex-col gap-1.5">
          {query.data?.map((paper) => (
            <li key={paper.id} className="group relative">
              <button
                type="button"
                onClick={() => onSelect(paper.id)}
                aria-current={selected === paper.id}
                className={`flex w-full flex-col gap-1.5 rounded-md border px-3 py-2 text-left transition-colors hover:border-hairline-strong ${selected === paper.id ? 'border-primary bg-surface-2' : 'border-hairline'}`}
              >
                <span className="pr-8 text-body font-medium break-words text-ink">
                  {paper.title}
                </span>
                <span className="truncate text-caption text-ink-subtle">
                  {[paper.authors.slice(0, 2).join(', '), paper.publishedAt?.slice(0, 4)]
                    .filter(Boolean)
                    .join(' · ')}
                </span>
                <span className="flex flex-wrap gap-1.5">
                  <AccessBadge access={paper.accessStatus} />
                  <Badge tone="muted">
                    {paper.ideaCount} {paper.ideaCount === 1 ? 'idea' : 'ideas'}
                  </Badge>
                  {paper.savedIdeaCount > 0 && (
                    <Badge tone="profit">{paper.savedIdeaCount} saved</Badge>
                  )}
                </span>
              </button>
              <Button
                size="sm"
                variant="ghost"
                aria-label={`Delete ${paper.title}`}
                className="absolute top-1.5 right-1.5 opacity-0 group-focus-within:opacity-100 group-hover:opacity-100"
                onClick={() => setDoomed(paper)}
              >
                <Trash2Icon aria-hidden />
              </Button>
            </li>
          ))}
        </ul>
      </div>
      <Confirm
        open={doomed != null}
        onOpenChange={(open) => !open && setDoomed(null)}
        title="Delete this paper?"
        confirmLabel="Delete"
        danger
        pending={remove.isPending}
        onConfirm={() => doomed && remove.mutate(doomed.id)}
      >
        “{doomed?.title}” and the ideas taken from it, including saved ones, are removed from your
        library.
      </Confirm>
    </Panel>
  )
}

function JobRow({ job, onOpen }: { job: Job; onOpen: (id: string) => void }) {
  const client = useQueryClient()
  const cancel = useMutation({
    mutationFn: () => research.cancelJob(job.id),
    onSettled: () => void client.invalidateQueries({ queryKey: KEYS.jobs }),
  })
  return (
    <div className="flex flex-col gap-1.5 rounded-md border border-hairline-strong bg-surface-2 px-3 py-2">
      <div className="flex items-center justify-between gap-2 text-body-compact">
        <span className="text-ink">{JOB_KIND[job.kind] ?? job.kind}</span>
        <span className="flex gap-1">
          {job.sourceId && (
            <Button size="sm" variant="ghost" onClick={() => onOpen(job.sourceId as string)}>
              Open
            </Button>
          )}
          <Button
            size="sm"
            variant="ghost"
            loading={cancel.isPending}
            onClick={() => cancel.mutate()}
          >
            {job.cancelRequested ? 'Cancelling' : 'Cancel'}
          </Button>
        </span>
      </div>
      <Progress value={job.progress} label={`${JOB_KIND[job.kind] ?? job.kind} progress`} />
      <p className="truncate text-caption text-ink-subtle">{job.message}</p>
    </div>
  )
}
