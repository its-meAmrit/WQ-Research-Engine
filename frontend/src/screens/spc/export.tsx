import { useQuery } from '@tanstack/react-query'
import { toast } from 'sonner'
import { saveText } from '@/lib/download'
import { Button, Disclosure, Empty, ErrorNotice, Notice, Panel, Skeleton } from '@/ui/kit'
import { spc } from './api'
import { CHECK_WORD, CheckDot, copy, usePrompts, useRuns, useSelection } from './shared'

const slug = (s: string) =>
  s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '') || 'prompt'

export function Export() {
  const { prompt, run, set } = useSelection()
  const prompts = usePrompts()
  const runs = useRuns(prompt ?? undefined)
  const q = useQuery({
    queryKey: ['spc', 'export', prompt, run],
    queryFn: () => spc.export(prompt as number, run ?? undefined),
    enabled: prompt !== null,
  })
  const good = (runs.data ?? []).filter((r) => r.status === 'completed' && r.valid)
  const ok = toast.success
  const bad = toast.error

  return (
    <div className="flex min-w-0 flex-col gap-4">
      <Notice tone="info" title="Alpha Harness never submits for you">
        Submission to WorldQuant is manual. Nothing here sends anything anywhere: you copy or
        download the prompt, then submit it yourself on the official SPC page.
      </Notice>

      <Panel
        title="Choose what to export"
        description="A prompt, and optionally a run whose sample you want to attach."
      >
        <div className="flex flex-col gap-3">
          {prompts.isPending ? (
            <Skeleton className="h-10" />
          ) : (prompts.data ?? []).length === 0 ? (
            <Empty title="No prompts yet">Write one in Prompt Studio.</Empty>
          ) : (
            <div className="flex flex-wrap gap-2" role="radiogroup" aria-label="Prompt to export">
              {(prompts.data ?? []).map((p) => (
                <button
                  key={p.id}
                  type="button"
                  role="radio"
                  aria-checked={prompt === p.id}
                  onClick={() => set({ tab: 'export', prompt: p.id, run: null })}
                  className={`rounded border px-3 py-1.5 text-body-compact ${prompt === p.id ? 'border-ink bg-surface-raised' : 'border-line hover:bg-surface-raised'}`}
                >
                  {p.name}
                </button>
              ))}
            </div>
          )}
          {prompt !== null && (
            <div
              className="flex flex-wrap gap-2"
              role="radiogroup"
              aria-label="Run whose sample to attach"
            >
              <button
                type="button"
                role="radio"
                aria-checked={run === null}
                onClick={() => set({ tab: 'export', prompt, run: null })}
                className={`rounded border px-3 py-1.5 text-body-compact ${run === null ? 'border-ink bg-surface-raised' : 'border-line hover:bg-surface-raised'}`}
              >
                Latest valid run
              </button>
              {good.map((r) => (
                <button
                  key={r.id}
                  type="button"
                  role="radio"
                  aria-checked={run === r.id}
                  onClick={() => set({ tab: 'export', prompt, run: r.id })}
                  className={`rounded border px-3 py-1.5 text-body-compact ${run === r.id ? 'border-ink bg-surface-raised' : 'border-line hover:bg-surface-raised'}`}
                >
                  Run {r.id}
                </button>
              ))}
            </div>
          )}
        </div>
      </Panel>

      {prompt !== null && q.isPending && <Skeleton className="h-48" />}
      {q.isError && <ErrorNotice error={q.error} title="Could not prepare the export" />}
      {q.data && (
        <>
          <Panel
            title="Readiness"
            description={
              q.data.ready
                ? 'Nothing blocks this export.'
                : 'Something blocks the package. Prompt text can still be copied.'
            }
          >
            <ul className="flex flex-col gap-2">
              {q.data.checks.map((c) => (
                <li key={c.id} className="flex items-start gap-2 text-body-compact">
                  <CheckDot status={c.status} />
                  <span>
                    <span className="font-medium">{c.label}</span>{' '}
                    <span className="text-ink-subtle">({CHECK_WORD[c.status]})</span>
                    <span className="block text-ink-muted">{c.detail}</span>
                  </span>
                </li>
              ))}
            </ul>
          </Panel>
          <Panel
            title="Take it with you"
            description={
              q.data.versionNumber
                ? `${q.data.promptName}, version ${q.data.versionNumber}`
                : q.data.promptName
            }
          >
            <div className="flex flex-wrap gap-2">
              <Button
                disabled={!q.data.promptText}
                onClick={() => copy(q.data.promptText ?? '', 'Prompt', ok, bad)}
              >
                Copy prompt
              </Button>
              <Button
                disabled={!q.data.promptText}
                onClick={() =>
                  saveText(
                    `${slug(q.data.promptName)}-prompt.txt`,
                    q.data.promptText ?? '',
                    'text/plain',
                  )
                }
              >
                Download prompt
              </Button>
              <Button
                disabled={!q.data.sampleJson}
                onClick={() => copy(q.data.sampleJson ?? '', 'JSON', ok, bad)}
              >
                Copy sample JSON
              </Button>
              <Button
                disabled={!q.data.sampleJson}
                onClick={() =>
                  saveText(
                    `${slug(q.data.promptName)}-sample.json`,
                    q.data.sampleJson ?? '',
                    'application/json',
                  )
                }
              >
                Download sample JSON
              </Button>
              <Button
                variant="primary"
                disabled={!q.data.package}
                onClick={() =>
                  saveText(
                    `${slug(q.data.promptName)}-spc-package.json`,
                    JSON.stringify(q.data.package, null, 2),
                    'application/json',
                  )
                }
              >
                Download submission package
              </Button>
            </div>
            {!q.data.package && (
              <p className="mt-2 text-body-compact text-ink-subtle">
                The package needs every check above to pass.
              </p>
            )}
          </Panel>
          {q.data.promptText && (
            <Disclosure summary="Prompt text">
              <pre className="num max-h-72 overflow-auto whitespace-pre-wrap break-words text-body-compact text-ink-muted">
                {q.data.promptText}
              </pre>
            </Disclosure>
          )}
        </>
      )}
    </div>
  )
}
