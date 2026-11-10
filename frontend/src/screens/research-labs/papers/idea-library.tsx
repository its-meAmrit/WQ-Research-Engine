/** Every saved idea, from papers and from web pages alike, in one searchable list. */

import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { useScope } from '@/lib/scope'
import { Empty, ErrorNotice, Field, Input, Panel, Skeleton, Switch } from '@/ui/kit'
import { Select } from '@/ui/overlay'
import { ScopePicker } from '@/ui/scope-picker'
import { research } from './api'
import { IdeaCard } from './ideas'
import { KEYS } from './shared'

const ANY = 'any'
const STATUS = [
  { value: ANY, label: 'Any status' },
  { value: 'new', label: 'Unreviewed' },
  { value: 'saved', label: 'Saved' },
  { value: 'rejected', label: 'Rejected' },
]
const ORIGIN = [
  { value: ANY, label: 'Papers and web' },
  { value: 'academic_paper', label: 'Papers' },
  { value: 'web_article', label: 'Web pages' },
]
const CLASS = [
  { value: ANY, label: 'Any source class' },
  { value: 'research_paper', label: 'Research paper' },
  { value: 'documented_factor', label: 'Documented factor' },
  { value: 'trading_strategy', label: 'Trading strategy' },
  { value: 'market_hypothesis', label: 'Market hypothesis' },
  { value: 'code_implementation', label: 'Code implementation' },
  { value: 'unverified_claim', label: 'Unverified claim' },
]
const REVIEW = [
  { value: ANY, label: 'Reviewed or not' },
  { value: 'yes', label: 'Reviewed' },
  { value: 'no', label: 'Not reviewed' },
]

export function IdeaLibrary() {
  const [scope, setScope] = useScope('labs/papers')
  const [search, setSearch] = useState('')
  const [status, setStatus] = useState(ANY)
  const [origin, setOrigin] = useState(ANY)
  const [sourceClass, setSourceClass] = useState(ANY)
  const [reviewed, setReviewed] = useState(ANY)
  const [domain, setDomain] = useState('')
  const [favorite, setFavorite] = useState(false)

  const filters = {
    ...(search.trim() ? { search: search.trim() } : {}),
    ...(status === ANY ? {} : { status }),
    ...(origin === ANY ? {} : { sourceType: origin }),
    ...(sourceClass === ANY ? {} : { sourceClass }),
    ...(reviewed === ANY ? {} : { reviewed: reviewed === 'yes' }),
    ...(domain.trim() ? { domain: domain.trim().toLowerCase() } : {}),
    ...(favorite ? { favorite: true } : {}),
  }
  const query = useQuery({
    queryKey: [...KEYS.ideas, 'library', filters],
    queryFn: () => research.ideas(filters),
  })

  return (
    <Panel
      title="Idea library"
      description="Ideas from papers and from the web. One hypothesis found in several places is kept once, with every source listed."
    >
      <div className="flex flex-col gap-4">
        <div className="grid gap-3 sm:grid-cols-2">
          <Field label="Search ideas">
            <Input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Keyword in the title, finding or signal"
            />
          </Field>
          <Field label="Site">
            <Input
              value={domain}
              onChange={(e) => setDomain(e.target.value)}
              placeholder="e.g. quant.example.com"
            />
          </Field>
          <Select value={status} onChange={setStatus} items={STATUS} label="Status" />
          <Select value={origin} onChange={setOrigin} items={ORIGIN} label="Source kind" />
          <Select
            value={sourceClass}
            onChange={setSourceClass}
            items={CLASS}
            label="Source class"
          />
          <Select value={reviewed} onChange={setReviewed} items={REVIEW} label="Review state" />
        </div>
        <span className="flex items-center gap-2 text-body-compact text-ink-muted">
          <Switch checked={favorite} onChange={setFavorite} aria-label="Favourites only" />
          Favourites only
        </span>
        <div className="flex flex-col gap-1.5">
          <span className="text-caption font-medium text-ink-muted">Check against</span>
          <ScopePicker scope={scope} onChange={setScope} />
        </div>
        {query.isPending && <Skeleton className="h-32" label="Loading ideas" />}
        {query.isError && <ErrorNotice error={query.error} title="Ideas could not be loaded" />}
        {query.data?.length === 0 && (
          <Empty title="No ideas match">
            Change the filters, or discover more on the Discover tab.
          </Empty>
        )}
        {query.data?.map((idea) => (
          <IdeaCard key={idea.id} summary={idea} scope={scope} />
        ))}
      </div>
    </Panel>
  )
}
