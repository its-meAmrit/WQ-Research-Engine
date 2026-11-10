/**
 * Paper Analyzer: turns research papers into testable ideas and checks which of them the
 * account's BRAIN data can support. It reads and records; it never runs a simulation.
 */

import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { Notice, Page, PageHeader, Segmented } from '@/ui/kit'
import { discover, research } from './api'
import { Discover } from './discover'
import { IdeaLibrary } from './idea-library'
import { Intake } from './intake'
import { Library } from './library'
import { PaperView } from './paper'
import { KEYS, useJobs } from './shared'

type View = 'papers' | 'discover' | 'ideas'

export function PaperAnalyzerScreen() {
  const [view, setView] = useState<View>('papers')
  const [selected, setSelected] = useState<string | null>(null)
  const options = useQuery({
    queryKey: KEYS.options,
    queryFn: research.options,
    staleTime: 60_000,
  })
  const found = useQuery({
    queryKey: ['research', 'discover', 'options'],
    queryFn: discover.options,
    staleTime: 30_000,
    enabled: view === 'discover',
  })
  const { active } = useJobs()

  return (
    <Page>
      <PageHeader
        title="Research Intelligence"
        description="Read papers, discover ideas on the public web, and see which ones your BRAIN data can test."
      />
      <Segmented
        label="Research view"
        value={view}
        onChange={setView}
        items={[
          { value: 'papers', label: 'Papers' },
          { value: 'discover', label: 'Discover' },
          { value: 'ideas', label: 'Idea library' },
        ]}
      />
      <Notice tone="info">
        This reads and records only. It does not run simulations, submit Alphas or write results.
      </Notice>
      {view === 'discover' && <Discover options={found.data} />}
      {view === 'ideas' && <IdeaLibrary />}
      {view === 'papers' && (
        <div className="grid min-w-0 items-start gap-4 lg:grid-cols-[minmax(0,26rem)_minmax(0,1fr)]">
          <div className="flex min-w-0 flex-col gap-4">
            <Intake
              maxMb={options.data?.limits.maxPdfMb}
              webSearch={options.data?.webSearch ?? false}
              onOpen={setSelected}
            />
            <Library
              topics={options.data?.topics ?? []}
              selected={selected}
              onSelect={setSelected}
              jobs={active}
            />
          </div>
          {selected ? (
            <PaperView id={selected} options={options.data} jobs={active} />
          ) : (
            <Notice tone="info" title="Choose a paper">
              Pick one from the library, or add one on the left.
            </Notice>
          )}
        </div>
      )}
    </Page>
  )
}
