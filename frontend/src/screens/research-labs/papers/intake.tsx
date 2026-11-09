/** Getting a paper in: a PDF, a DOI, arXiv id or URL, or a search of the academic indexes. */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { FileUpIcon, SearchIcon } from 'lucide-react'
import { type DragEvent, useRef, useState } from 'react'
import { toast } from 'sonner'
import { Badge, Button, Empty, ErrorNotice, Input, Notice, Panel, Spinner, Switch } from '@/ui/kit'
import { research, type SearchPaper, type Started } from './api'
import { AccessBadge, KEYS } from './shared'

/** Tells the person what an import did, including when it found the paper already there. */
function useAfterImport(onOpen: (sourceId: string) => void) {
  const client = useQueryClient()
  return (started: Started) => {
    void client.invalidateQueries({ queryKey: KEYS.papers })
    void client.invalidateQueries({ queryKey: KEYS.jobs })
    if (started.sourceId) onOpen(started.sourceId)
    toast(started.existing ? 'Already in your library' : 'Import started', {
      description: started.message,
    })
  }
}

export function Intake({
  maxMb,
  webSearch,
  onOpen,
}: {
  maxMb: number | undefined
  webSearch: boolean
  onOpen: (sourceId: string) => void
}) {
  const after = useAfterImport(onOpen)
  const file = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)
  const [reference, setReference] = useState('')

  const upload = useMutation({ mutationFn: research.upload, onSuccess: after })
  const fromReference = useMutation({
    mutationFn: research.importReference,
    onSuccess: (started) => {
      setReference('')
      after(started)
    },
  })

  const take = (files: FileList | null) => {
    const first = files?.[0]
    if (first) upload.mutate(first)
  }
  const drop = (event: DragEvent) => {
    event.preventDefault()
    setOver(false)
    take(event.dataTransfer.files)
  }

  return (
    <Panel
      title="Add a paper"
      description="Full text is read only from a PDF you upload or a copy that is openly available."
    >
      <div className="flex flex-col gap-4">
        <button
          type="button"
          onClick={() => file.current?.click()}
          onDragOver={(e) => {
            e.preventDefault()
            setOver(true)
          }}
          onDragLeave={() => setOver(false)}
          onDrop={drop}
          disabled={upload.isPending}
          className={`flex flex-col items-center gap-1.5 rounded-md border border-dashed px-4 py-6 text-body text-ink-subtle transition-colors hover:border-hairline-strong hover:text-ink ${over ? 'border-primary text-ink' : 'border-hairline-strong'}`}
        >
          {upload.isPending ? <Spinner /> : <FileUpIcon className="size-5" aria-hidden />}
          <span>Drop a PDF here, or choose one</span>
          {maxMb != null && <span className="text-caption">Up to {maxMb} MB</span>}
        </button>
        <input
          ref={file}
          type="file"
          accept="application/pdf,.pdf"
          className="sr-only"
          aria-label="Upload a PDF"
          onChange={(e) => {
            take(e.target.files)
            e.target.value = ''
          }}
        />
        {upload.isError && <ErrorNotice error={upload.error} title="That PDF could not be added" />}

        <form
          className="flex gap-2"
          onSubmit={(e) => {
            e.preventDefault()
            if (reference.trim()) fromReference.mutate(reference.trim())
          }}
        >
          <Input
            value={reference}
            onChange={(e) => setReference(e.target.value)}
            placeholder="DOI, arXiv id or paper URL"
            aria-label="DOI, arXiv id or paper URL"
          />
          <Button
            type="submit"
            variant="primary"
            loading={fromReference.isPending}
            disabled={!reference.trim()}
          >
            Import
          </Button>
        </form>
        {fromReference.isError && (
          <ErrorNotice error={fromReference.error} title="That reference could not be imported" />
        )}

        <Search webSearch={webSearch} onImported={after} />
      </div>
    </Panel>
  )
}

function Search({
  webSearch,
  onImported,
}: {
  webSearch: boolean
  onImported: (started: Started) => void
}) {
  const [q, setQ] = useState('')
  const [submitted, setSubmitted] = useState('')
  const [semantic, setSemantic] = useState(false)
  const [web, setWeb] = useState(false)

  const results = useQuery({
    queryKey: ['research', 'search', submitted, semantic, web],
    queryFn: ({ signal }) => research.search(submitted, semantic, web, signal),
    enabled: submitted.length >= 2,
    staleTime: 5 * 60_000,
    retry: false,
  })
  const add = useMutation({
    mutationFn: (paper: SearchPaper) =>
      research.importResult({ doi: paper.doi, arxivId: paper.arxivId, url: paper.url }),
    onSuccess: onImported,
  })

  return (
    <div className="flex flex-col gap-3 border-t border-hairline pt-4">
      <form
        className="flex gap-2"
        onSubmit={(e) => {
          e.preventDefault()
          setSubmitted(q.trim())
        }}
      >
        <Input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search papers by keyword"
          aria-label="Search papers by keyword"
        />
        <Button type="submit" disabled={q.trim().length < 2} loading={results.isFetching}>
          <SearchIcon aria-hidden /> Search
        </Button>
      </form>
      <div className="flex flex-wrap gap-x-5 gap-y-2 text-body-compact">
        <span className="flex items-center gap-2 text-ink-muted">
          <Switch checked={semantic} onChange={setSemantic} aria-label="Include Semantic Scholar" />
          Include Semantic Scholar
        </span>
        {webSearch && (
          <span className="flex items-center gap-2 text-ink-muted">
            <Switch checked={web} onChange={setWeb} aria-label="Add supplementary web results" />
            Add supplementary web results
          </span>
        )}
      </div>

      {results.isError && <ErrorNotice error={results.error} title="The search failed" />}
      {add.isError && <ErrorNotice error={add.error} title="That paper could not be imported" />}
      {results.data?.notes.map((note) => (
        <Notice key={note} tone="warn">
          {note}
        </Notice>
      ))}
      {results.data && results.data.papers.length === 0 && (
        <Empty title="No papers found">Try different words, or import by DOI.</Empty>
      )}
      <ul className="flex flex-col gap-2">
        {results.data?.papers.map((paper) => (
          <li
            key={paper.doi ?? paper.arxivId ?? paper.url ?? paper.title}
            className="flex flex-col gap-1.5 rounded-md border border-hairline px-3 py-2"
          >
            <div className="flex items-start justify-between gap-3">
              <p className="min-w-0 text-body font-medium break-words text-ink">{paper.title}</p>
              <Button
                size="sm"
                disabled={add.isPending || !(paper.doi || paper.arxivId || paper.url)}
                onClick={() => add.mutate(paper)}
              >
                {paper.importedAs ? 'Open' : 'Import'}
              </Button>
            </div>
            <p className="truncate text-caption text-ink-subtle">
              {[paper.authors.slice(0, 3).join(', '), paper.venue, paper.publishedAt?.slice(0, 4)]
                .filter(Boolean)
                .join(' · ')}
            </p>
            <div className="flex flex-wrap gap-1.5">
              <AccessBadge access={paper.access} />
              <Badge tone="outline">{paper.provider}</Badge>
              {paper.citations != null && <Badge tone="muted">{paper.citations} citations</Badge>}
            </div>
          </li>
        ))}
      </ul>
      {results.data && results.data.web.length > 0 && (
        <div className="flex flex-col gap-1.5">
          <p className="text-caption font-medium text-ink-muted">
            Supplementary web results (not papers; nothing is imported from them)
          </p>
          {results.data.web.map((hit) => (
            <a
              key={hit.url}
              href={hit.url}
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-md border border-hairline px-3 py-2 text-body-compact text-link hover:text-ink"
            >
              {hit.title}
              <span className="block truncate text-caption text-ink-subtle">{hit.snippet}</span>
            </a>
          ))}
        </div>
      )}
    </div>
  )
}
