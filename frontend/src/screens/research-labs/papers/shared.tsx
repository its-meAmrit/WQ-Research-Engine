/** Labels, badges and the job-polling hook the Paper Analyzer's parts share. */

import { useQuery, useQueryClient } from '@tanstack/react-query'
import { createContext, useContext, useEffect, useRef } from 'react'
import { Badge, Disclosure } from '@/ui/kit'
import {
  type Access,
  type Indicator,
  type IndicatorLevel,
  type Job,
  type Ranking,
  type Readiness,
  research,
  type Support,
} from './api'

export const KEYS = {
  options: ['research', 'options'] as const,
  papers: ['research', 'papers'] as const,
  paper: (id: string) => ['research', 'paper', id] as const,
  ideas: ['research', 'ideas'] as const,
  jobs: ['research', 'jobs'] as const,
  discover: ['research', 'discover'] as const,
  hits: (jobId: string) => ['research', 'discover', 'hits', jobId] as const,
}

/** Opens the Alpha Builder on an idea. Provided by the Research Intelligence screen. */
export const BuilderLauncher = createContext<(ideaId: string) => void>(() => undefined)
export const useBuilderLauncher = () => useContext(BuilderLauncher)

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
  discover: 'Discovery',
  extract: 'Extraction',
  design: 'Signal design',
  variants: 'Alpha variants',
  refine: 'Refinement',
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
      void client.invalidateQueries({ queryKey: ['research', 'builder'] })
    }
  }, [active, client])
  return { jobs, active, isError: query.isError }
}

const LEVEL_LABEL: Record<IndicatorLevel, string> = {
  high: 'High',
  medium: 'Medium',
  low: 'Low',
  unknown: 'Unknown',
}
const LEVEL_TONE: Record<IndicatorLevel, 'profit' | 'neutral' | 'warn' | 'muted'> = {
  high: 'profit',
  medium: 'neutral',
  low: 'warn',
  unknown: 'muted',
}
const INDICATORS: { key: keyof Ranking; label: string }[] = [
  { key: 'sourceQuality', label: 'Source quality' },
  { key: 'specificity', label: 'Research specificity' },
  { key: 'brainFeasibility', label: 'BRAIN feasibility' },
  { key: 'novelty', label: 'Novelty (your library)' },
  { key: 'evidenceQuality', label: 'Evidence quality' },
]

function IndicatorRow({ label, indicator }: { label: string; indicator: Indicator }) {
  return (
    <Disclosure
      summary={
        <span className="flex flex-wrap items-center gap-2">
          <span className="text-body-compact text-ink">{label}</span>
          <Badge tone={LEVEL_TONE[indicator.level]}>{LEVEL_LABEL[indicator.level]}</Badge>
        </span>
      }
    >
      <p className="text-caption text-ink-subtle">{indicator.basis}</p>
      <ul className="list-disc pl-5 text-body-compact text-ink-muted">
        {indicator.reasons.map((r) => (
          <li key={r}>{r}</li>
        ))}
      </ul>
    </Disclosure>
  )
}

/** The five ranking indicators, side by side and never combined. */
export function RankingView({ ranking }: { ranking: Ranking | null }) {
  if (!ranking) return null
  return (
    <section className="flex flex-col gap-1 rounded-md bg-surface-2 p-3">
      <h4 className="text-body font-medium text-ink">Ranking indicators</h4>
      <p className="text-caption text-ink-subtle">
        Five separate indicators. None is combined into a score, and none estimates a Sharpe ratio
        or a return. “Unknown” means it could not be worked out from what is stored.
      </p>
      {INDICATORS.map(({ key, label }) => (
        <IndicatorRow key={key} label={label} indicator={ranking[key]} />
      ))}
    </section>
  )
}
