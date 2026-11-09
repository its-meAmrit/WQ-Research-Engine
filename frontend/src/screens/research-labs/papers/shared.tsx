/** Labels, badges and the job-polling hook the Paper Analyzer's parts share. */

import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useRef } from 'react'
import { Badge } from '@/ui/kit'
import { type Access, type Job, type Readiness, research, type Support } from './api'

export const KEYS = {
  options: ['research', 'options'] as const,
  papers: ['research', 'papers'] as const,
  paper: (id: string) => ['research', 'paper', id] as const,
  ideas: ['research', 'ideas'] as const,
  jobs: ['research', 'jobs'] as const,
}

export const ACCESS_LABEL: Record<Access, string> = {
  full_text: 'Full text',
  abstract_only: 'Metadata/abstract only',
  metadata_only: 'Metadata/abstract only',
}

export const READY_LABEL: Record<Readiness, string> = {
  ready: 'Ready',
  adaptation: 'Needs adaptation',
  blocked: 'Blocked',
}

export const SUPPORT_LABEL: Record<Support, string> = {
  source_backed: 'Source-backed',
  partially_supported: 'Partly supported',
  speculative: 'Speculative',
}

export function AccessBadge({ access }: { access: Access }) {
  return <Badge tone={access === 'full_text' ? 'profit' : 'warn'}>{ACCESS_LABEL[access]}</Badge>
}

export function ReadinessBadge({ readiness }: { readiness: Readiness | null }) {
  if (!readiness) return <Badge tone="muted">Not checked</Badge>
  const tone = readiness === 'ready' ? 'profit' : readiness === 'blocked' ? 'loss' : 'warn'
  return <Badge tone={tone}>{READY_LABEL[readiness]}</Badge>
}

export function SupportBadge({ support }: { support: Support }) {
  const tone =
    support === 'source_backed' ? 'profit' : support === 'speculative' ? 'warn' : 'neutral'
  return <Badge tone={tone}>{SUPPORT_LABEL[support]}</Badge>
}

export const JOB_KIND: Record<string, string> = {
  import: 'Import',
  analyze: 'Analysis',
  search: 'Search',
}

/**
 * Every job, polled while any is running. When one finishes the lists it changed are
 * refetched, so a paper or idea appears without a reload.
 */
export function useJobs(): { jobs: Job[]; active: Job[]; isError: boolean } {
  const client = useQueryClient()
  const query = useQuery({
    queryKey: KEYS.jobs,
    queryFn: () => research.jobs(),
    refetchInterval: (q) => (q.state.data?.some((j) => j.active) ? 1500 : 15_000),
  })
  const jobs = query.data ?? []
  const active = jobs.filter((j) => j.active)
  const running = useRef(new Set<string>())
  useEffect(() => {
    const now = new Set(active.map((j) => j.id))
    const finished = [...running.current].some((id) => !now.has(id))
    running.current = now
    if (finished) {
      void client.invalidateQueries({ queryKey: KEYS.papers })
      void client.invalidateQueries({ queryKey: KEYS.ideas })
      void client.invalidateQueries({ queryKey: ['research', 'paper'] })
    }
  }, [active, client])
  return { jobs, active, isError: query.isError }
}
