import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useSearch } from '@tanstack/react-router'
import { useCallback } from 'react'
import { cn } from '@/lib/cn'
import { Badge, type Tone } from '@/ui/kit'
import { isActive, spc } from './api'

/** Keys are namespaced so one invalidation refreshes the whole lab and nothing else. */
export const KEY = ['spc'] as const

export const useOptions = () =>
  useQuery({ queryKey: [...KEY, 'options'], queryFn: spc.options, staleTime: 30_000 })

export const usePrompts = (archived = false) =>
  useQuery({ queryKey: [...KEY, 'prompts', archived], queryFn: () => spc.prompts(archived) })

export const usePrompt = (id: number | null) =>
  useQuery({
    queryKey: [...KEY, 'prompt', id],
    queryFn: () => spc.prompt(id as number),
    enabled: id !== null,
  })

/** A run being worked on is read every second, and left alone once it has finished: the screen
 *  follows the row the backend writes, so what it shows is what happened. */
export const useRun = (id: number | null) =>
  useQuery({
    queryKey: [...KEY, 'run', id],
    queryFn: () => spc.run(id as number),
    enabled: id !== null,
    refetchInterval: (query) => {
      const status = query.state.data?.status
      return status && isActive(status) ? 1000 : false
    },
  })

export const useRuns = (promptId?: number) =>
  useQuery({
    queryKey: [...KEY, 'runs', promptId ?? null],
    queryFn: () => spc.runs(promptId),
    refetchInterval: (query) => (query.state.data?.some((r) => isActive(r.status)) ? 1500 : false),
  })

export function useInvalidate() {
  const client = useQueryClient()
  return useCallback(() => client.invalidateQueries({ queryKey: KEY }), [client])
}

/** Which prompt and run are open. Held in the URL, so a reload or a link lands where you were. */
export function useSelection() {
  const search = useSearch({ strict: false }) as { prompt?: number; run?: number }
  const navigate = useNavigate()
  const set = useCallback(
    (next: { prompt?: number | null; run?: number | null; tab?: string }) =>
      navigate({
        to: '/spc/$tab',
        params: { tab: next.tab ?? 'studio' },
        search: (prev: { prompt?: number; run?: number }) => {
          const prompt = next.prompt === null ? undefined : (next.prompt ?? prev.prompt)
          const run = next.run === null ? undefined : (next.run ?? prev.run)
          return {
            ...(prompt === undefined ? {} : { prompt }),
            ...(run === undefined ? {} : { run }),
          }
        },
      }),
    [navigate],
  )
  return { prompt: search.prompt ?? null, run: search.run ?? null, set }
}

/** A run's state in words and a tone. The word always carries it; the tone only helps. */
export const STATUS_TONE: Record<string, Tone> = {
  queued: 'muted',
  researching: 'neutral',
  generating: 'neutral',
  validating: 'neutral',
  completed: 'profit',
  failed: 'loss',
  cancelled: 'warn',
}

export function StatusBadge({ status, valid }: { status: string; valid?: boolean | null }) {
  const label = status === 'completed' && valid === false ? 'invalid' : status
  return <Badge tone={STATUS_TONE[status] ?? 'muted'}>{label}</Badge>
}

/** One dot and one word per readiness check: a state is never colour alone. */
export function CheckDot({ status }: { status: 'pass' | 'warn' | 'fail' }) {
  return (
    <span
      aria-hidden
      className={cn(
        'mt-1.5 size-1.5 shrink-0 rounded-full',
        status === 'pass' && 'bg-pnl-positive',
        status === 'warn' && 'bg-status-warning',
        status === 'fail' && 'bg-pnl-negative',
      )}
    />
  )
}

export const CHECK_WORD = { pass: 'Pass', warn: 'Warning', fail: 'Blocked' } as const

/** `owner/model` as a short name for a table cell. */
export const shortModel = (ref: string) => ref.split('/').slice(-1)[0] ?? ref

export async function copy(
  text: string,
  what: string,
  ok: (m: string) => void,
  bad: (m: string) => void,
) {
  try {
    await navigator.clipboard.writeText(text)
    ok(`${what} copied`)
  } catch {
    bad(`Could not copy the ${what.toLowerCase()}. Select it and copy by hand.`)
  }
}
