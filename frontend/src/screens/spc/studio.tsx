import { useMutation } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import { CopyIcon, PlayIcon, PlusIcon } from 'lucide-react'
import { useEffect, useState } from 'react'
import { toast } from 'sonner'
import { cn } from '@/lib/cn'
import { fmt } from '@/lib/format'
import {
  Badge,
  Button,
  Chips,
  Disclosure,
  Empty,
  ErrorNotice,
  Field,
  Fieldset,
  Input,
  LINK,
  Notice,
  Panel,
  Segmented,
  Skeleton,
  Textarea,
} from '@/ui/kit'
import { Confirm, Dialog, Select } from '@/ui/overlay'
import {
  type GenerationOptions,
  isActive,
  type PromptCandidate,
  type PromptDetail,
  type PromptOut,
  type PromptVersion,
  spc,
} from './api'
import {
  StatusBadge,
  shortModel,
  useInvalidate,
  useOptions,
  usePrompt,
  usePrompts,
  useSelection,
} from './shared'

const WEIGHT = /^(0(\.\d{1,2})?|1(\.0{1,2})?)$/

export function Studio() {
  const { prompt, set } = useSelection()
  const [archived, setArchived] = useState(false)
  const prompts = usePrompts(archived)
  return (
    <div className="grid min-w-0 gap-4 lg:grid-cols-[18rem_minmax(0,1fr)]">
      <Panel
        title="Prompts"
        actions={
          <Button size="sm" onClick={() => set({ prompt: null, run: null })}>
            <PlusIcon /> New
          </Button>
        }
        bodyClassName="p-2"
      >
        {prompts.isPending ? (
          <Skeleton className="h-32" />
        ) : prompts.isError ? (
          <ErrorNotice error={prompts.error} title="Could not load the prompts" />
        ) : prompts.data.length === 0 ? (
          <Empty title={archived ? 'No archived prompts' : 'No prompts yet'}>
            {archived
              ? 'Archived prompts appear here.'
              : 'Describe an idea on the right to write your first.'}
          </Empty>
        ) : (
          <ul className="flex flex-col gap-1">
            {prompts.data.map((p) => (
              <PromptRow
                key={p.id}
                p={p}
                open={p.id === prompt}
                onOpen={() => set({ prompt: p.id })}
              />
            ))}
          </ul>
        )}
        <div className="mt-2 border-t border-hairline px-1 pt-2">
          <Button variant="ghost" size="sm" onClick={() => setArchived((a) => !a)}>
            {archived ? 'Show active prompts' : 'Show archived'}
          </Button>
        </div>
      </Panel>
      <Editor key={prompt ?? 'new'} promptId={prompt} />
    </div>
  )
}

function PromptRow({ p, open, onOpen }: { p: PromptOut; open: boolean; onOpen: () => void }) {
  return (
    <li>
      <button
        type="button"
        onClick={onOpen}
        aria-current={open}
        className={cn(
          'flex w-full min-w-0 flex-col gap-1 rounded-sm border border-transparent px-2 py-2 text-left transition-colors hover:bg-surface-2',
          open && 'border-hairline-strong bg-surface-3',
        )}
      >
        <span className="truncate text-body font-medium text-ink">{p.name}</span>
        <span className="flex flex-wrap items-center gap-1.5 text-body-compact text-ink-subtle">
          <span className="num">v{p.latestVersion}</span>
          <span>·</span>
          <span className="num">weight {p.weight.toFixed(2)}</span>
          {p.lastRun && <StatusBadge status={p.lastRun.status} valid={p.lastRun.valid} />}
        </span>
      </button>
    </li>
  )
}

// ── the editor ──────────────────────────────────────────────────────────────────────────

function Editor({ promptId }: { promptId: number | null }) {
  const detail = usePrompt(promptId)
  if (promptId !== null && detail.isPending) return <Skeleton className="h-96" />
  if (promptId !== null && detail.isError)
    return <ErrorNotice error={detail.error} title="Could not load this prompt" />
  return <EditorBody saved={detail.data ?? null} />
}

function EditorBody({ saved }: { saved: PromptDetail | null }) {
  const { set } = useSelection()
  const options = useOptions()
  const invalidate = useInvalidate()
  const [name, setName] = useState(saved?.name ?? '')
  const [text, setText] = useState(saved?.promptText ?? '')
  const [frequency, setFrequency] = useState(saved?.updateFrequency ?? 'weekly')
  const [weight, setWeight] = useState(saved ? saved.weight.toFixed(2) : '0.00')
  const [model, setModel] = useState<string | null>(saved?.preferredModel ?? null)
  const [modelVersion, setModelVersion] = useState(saved?.modelVersion ?? '')
  const [note, setNote] = useState('')
  const [settings, setSettings] = useState<Record<string, unknown> | null>(null)
  const [confirmRun, setConfirmRun] = useState(false)
  const [research, setResearch] = useState<'off' | 'auto' | 'required'>('auto')
  const [compare, setCompare] = useState<[number, number] | null>(null)

  // A model the lab can run: the one on the prompt if it is still set up, else the first.
  const models = options.data?.models ?? []
  const chosen = model ?? options.data?.defaultModel ?? null
  const limits = options.data?.limits as
    | { promptMaxChars: number; modelVersionMaxChars: number }
    | undefined
  const max = limits?.promptMaxChars ?? 10_000

  useEffect(() => {
    // The server's copy is the baseline: a refetch after a save must not reset what is typed.
  }, [])

  const weightOk = WEIGHT.test(weight)
  const overLength = text.length > max
  const dirtyText = saved ? text !== saved.promptText : text.trim() !== ''
  const dirtyMeta =
    !!saved &&
    (name !== saved.name ||
      frequency !== saved.updateFrequency ||
      (weightOk && Number(weight) !== saved.weight) ||
      (model ?? null) !== (saved.preferredModel ?? null) ||
      modelVersion !== (saved.modelVersion ?? ''))
  const dirty = dirtyText || dirtyMeta
  const problem = !name.trim()
    ? 'Give the prompt a name.'
    : !text.trim()
      ? 'The prompt is empty.'
      : overLength
        ? `The prompt is ${text.length.toLocaleString()} characters; WorldQuant accepts ${max.toLocaleString()}.`
        : !weightOk
          ? 'Weight must be 0.00 to 1.00 with at most two decimals.'
          : modelVersion.length > (limits?.modelVersionMaxChars ?? 100)
            ? 'The model version is too long.'
            : null

  const body = {
    name: name.trim(),
    promptText: text,
    updateFrequency: frequency as 'weekly',
    weight: Number(weight),
    modelVersion,
    note,
    ...(settings ? { settings } : {}),
  }
  const save = useMutation({
    mutationFn: () =>
      saved
        ? spc.update(
            saved.id,
            model ? { ...body, preferredModel: model } : { ...body, clearModel: true },
          )
        : spc.create(model ? { ...body, preferredModel: model } : body),
    onSuccess: (next) => {
      toast.success(saved ? `Saved. Version ${next.latestVersion}.` : 'Prompt saved')
      invalidate()
      setNote('')
      if (!saved) set({ prompt: next.id })
    },
  })
  const run = useMutation({
    mutationFn: () => {
      if (!saved || !chosen) throw new Error('Save the prompt and choose a model first.')
      return spc.test(saved.id, { model: chosen, research })
    },
    onSuccess: (started) => {
      setConfirmRun(false)
      toast.success(`Run ${started.id} started`)
      invalidate()
      set({ tab: 'runs', run: started.id })
    },
    onError: () => setConfirmRun(false),
  })
  const duplicate = useMutation({
    mutationFn: () => spc.duplicate((saved as PromptDetail).id),
    onSuccess: (copy) => {
      toast.success('Duplicated, with weight 0 so it cannot double this one')
      invalidate()
      set({ prompt: copy.id })
    },
  })
  const archive = useMutation({
    mutationFn: () => spc.update((saved as PromptDetail).id, { archived: !saved?.archived }),
    onSuccess: (next) => {
      toast.success(next.archived ? 'Archived' : 'Restored')
      invalidate()
    },
  })

  const researchStatus = options.data?.research as
    | {
        configured: boolean
        help: string | null
        message?: string | null
        source?: 'store' | 'environment' | null
        label?: string | null
      }
    | undefined
  const running = saved?.lastRun && isActive(saved.lastRun.status)
  const blocked = !saved
    ? 'Save the prompt first: a run tests a saved version.'
    : dirty
      ? 'You have unsaved changes. Save them first, so the run tests exactly what you see.'
      : !chosen
        ? 'Choose a model. Set one up under API Integration if the list is empty.'
        : running
          ? 'A run of this prompt is already in progress.'
          : research === 'required' && researchStatus && !researchStatus.configured
            ? 'Web research is required but not set up. Add a key under API Integration › Web Research.'
            : null

  return (
    <div className="flex min-w-0 flex-col gap-4">
      <Generator
        onUse={(c, s) => {
          setText(c.promptText)
          setSettings(s as Record<string, unknown>)
          setFrequency(s.updateFrequency ?? frequency)
          if (!name.trim()) setName(s.thesis.slice(0, 60))
          toast.success(`${c.label} prompt loaded. Edit it, then save.`)
        }}
        hasText={text.trim() !== ''}
      />

      <Panel
        title={saved ? 'Edit prompt' : 'New prompt'}
        description={
          saved
            ? `Saving a changed text adds version ${saved.latestVersion + 1}; earlier versions and their runs are never changed.`
            : 'Write the prompt by hand, or generate one above. It is entered into WorldQuant exactly as it is here.'
        }
      >
        <div className="flex flex-col gap-4">
          <div className="grid gap-4 md:grid-cols-2">
            <Field label="Prompt name">
              <Input value={name} onChange={(e) => setName(e.target.value)} maxLength={200} />
            </Field>
            <Field label="Update frequency" hint="How often the platform re-runs it.">
              <Select
                label="Update frequency"
                value={frequency}
                onChange={setFrequency}
                items={(options.data?.frequencies ?? ['weekly']).map((f) => ({
                  value: f,
                  label: f[0]?.toUpperCase() + f.slice(1),
                }))}
              />
            </Field>
            <Field
              label="Portfolio weight"
              hint="0.00 to 1.00. Zero makes the prompt inactive: it does not run for SPC."
              error={weightOk ? undefined : 'Use 0.00 to 1.00, at most two decimals.'}
            >
              <Input
                inputMode="decimal"
                value={weight}
                onChange={(e) => setWeight(e.target.value)}
                aria-invalid={!weightOk}
              />
            </Field>
            <Field label="Model" hint="One you have set up under API Integration.">
              <Select
                label="Model"
                value={chosen}
                onChange={setModel}
                mono
                disabled={models.length === 0}
                items={models.map((m) => ({ value: m.ref, label: m.ref }))}
              />
            </Field>
            <Field
              label="Model version (optional)"
              hint={`Up to ${limits?.modelVersionMaxChars ?? 100} characters, as WorldQuant records it.`}
            >
              <Input value={modelVersion} onChange={(e) => setModelVersion(e.target.value)} />
            </Field>
            {saved && (
              <Field
                label="Version note (optional)"
                hint="Why this change, kept with the new version."
              >
                <Input value={note} onChange={(e) => setNote(e.target.value)} maxLength={300} />
              </Field>
            )}
          </div>
          {models.length === 0 && options.isSuccess && (
            <Notice tone="warn" title="No model is ready to run">
              Add an API key and set up a model under API Integration. You can still write, save and
              version prompts without one.
            </Notice>
          )}

          <Field
            label={
              <span className="flex items-baseline justify-between gap-2">
                <span>Prompt text</span>
                <span
                  className={cn(
                    'num text-body-compact',
                    overLength ? 'text-pnl-negative-text' : 'text-ink-subtle',
                  )}
                >
                  {text.length.toLocaleString()} / {max.toLocaleString()}
                </span>
              </span>
            }
            error={
              overLength
                ? `Over the limit by ${(text.length - max).toLocaleString()} characters.`
                : undefined
            }
          >
            <Textarea
              value={text}
              onChange={(e) => {
                setText(e.target.value)
                // Hand edits make the options that generated the text a poor record of it.
                if (settings && e.target.value !== text) setSettings(settings)
              }}
              rows={18}
              spellCheck={false}
              aria-invalid={overLength}
              className="num leading-relaxed"
            />
          </Field>

          {problem && dirty && <Notice tone="warn">{problem}</Notice>}
          {save.isError && <ErrorNotice error={save.error} title="Could not save" />}

          <div className="flex flex-wrap items-center gap-2">
            <Button
              variant="primary"
              loading={save.isPending}
              disabled={!!problem || (!!saved && !dirty)}
              onClick={() => save.mutate()}
            >
              {saved
                ? dirtyText
                  ? `Save as version ${saved.latestVersion + 1}`
                  : 'Save changes'
                : 'Save prompt'}
            </Button>
            {saved && (
              <>
                <Button loading={duplicate.isPending} onClick={() => duplicate.mutate()}>
                  <CopyIcon /> Duplicate
                </Button>
                <Button
                  variant="ghost"
                  loading={archive.isPending}
                  onClick={() => archive.mutate()}
                >
                  {saved.archived ? 'Restore from archive' : 'Archive'}
                </Button>
              </>
            )}
            {dirty && saved && (
              <Button variant="ghost" onClick={() => set({ prompt: saved.id })}>
                Discard changes
              </Button>
            )}
          </div>
        </div>
      </Panel>

      <Panel
        title="Test it"
        description="Runs the saved version on the model, checks the answer against the SPC schema and keeps the result. Spends model requests (and searches, if configured); never a BRAIN simulation."
      >
        <div className="flex flex-col gap-3">
          <Fieldset
            legend="Web research"
            hint="A model has no browser. Research is a separate step that searches, then hands the evidence to the model."
          >
            <Segmented
              label="Web research"
              value={research}
              onChange={setResearch}
              items={[
                { value: 'auto', label: 'Use if configured' },
                { value: 'required', label: 'Required' },
                { value: 'off', label: 'Off' },
              ]}
            />
          </Fieldset>
          {researchStatus && !researchStatus.configured && research !== 'off' && (
            <Notice tone="warn" title="Web research is not set up">
              {researchStatus.help}{' '}
              <Link to="/ai/$tab" params={{ tab: 'web' }} className={LINK}>
                Open Web Research setup
              </Link>
            </Notice>
          )}
          {researchStatus?.configured &&
            researchStatus.source === 'environment' &&
            research !== 'off' && (
              <Notice
                tone="info"
                title={`Research uses ${researchStatus.label} from an environment variable`}
              >
                {researchStatus.message}{' '}
                <Link to="/ai/$tab" params={{ tab: 'web' }} className={LINK}>
                  Manage it in Web Research
                </Link>
              </Notice>
            )}
          <div className="flex flex-wrap items-center gap-3">
            <Button variant="primary" disabled={!!blocked} onClick={() => setConfirmRun(true)}>
              <PlayIcon /> Run now
            </Button>
            {blocked && <span className="text-body-compact text-ink-subtle">{blocked}</span>}
          </div>
        </div>
      </Panel>

      {saved && (
        <Versions
          saved={saved}
          onCompare={(a, b) => setCompare([a, b])}
          onLoad={(v) => {
            setText(v.promptText)
            setSettings(null)
          }}
        />
      )}

      <Confirm
        open={confirmRun}
        onOpenChange={setConfirmRun}
        title="Run this prompt?"
        confirmLabel="Run now"
        pending={run.isPending}
        onConfirm={() => run.mutate()}
      >
        Sends version {saved?.latestVersion} to <span className="num text-ink">{chosen}</span>: one
        model request
        {research !== 'off' && researchStatus?.configured ? ', plus up to four searches' : ''}. It
        starts at once and you can follow it on the Test Runs tab.
      </Confirm>
      {compare && saved && (
        <CompareDialog promptId={saved.id} pair={compare} onClose={() => setCompare(null)} />
      )}
    </div>
  )
}

// ── generator ───────────────────────────────────────────────────────────────────────────

function Generator({
  onUse,
  hasText,
}: {
  onUse: (c: PromptCandidate, s: GenerationOptions) => void
  hasText: boolean
}) {
  const options = useOptions()
  const [thesis, setThesis] = useState('')
  const [universe, setUniverse] = useState('Global listed equities')
  const [sector, setSector] = useState('')
  const [direction, setDirection] = useState<GenerationOptions['direction']>('long_short')
  const [target, setTarget] = useState('50')
  const [horizon, setHorizon] = useState('30')
  const [freshness, setFreshness] = useState('90')
  const [sectorCap, setSectorCap] = useState('30')
  const [countryCap, setCountryCap] = useState('40')
  const [constraints, setConstraints] = useState('')
  const [returnUnit, setReturnUnit] = useState<GenerationOptions['returnUnit']>('fraction')
  const [frequency, setFrequency] = useState<GenerationOptions['updateFrequency']>('weekly')
  const [approaches, setApproaches] = useState<string[]>(['hybrid'])
  const [pending, setPending] = useState<{ c: PromptCandidate; s: GenerationOptions } | null>(null)

  const request = {
    thesis,
    universe,
    sector,
    direction,
    targetEquities: Number(target),
    horizonDays: Number(horizon),
    freshnessDays: Number(freshness),
    maxSectorPercent: Number(sectorCap),
    maxCountryPercent: Number(countryCap),
    constraints,
    returnUnit,
    updateFrequency: frequency,
    approaches,
  } as GenerationOptions
  const generate = useMutation({ mutationFn: () => spc.generate(request) })
  const choices = Object.entries(options.data?.approaches ?? {}).map(([value, label]) => ({
    value,
    label,
  }))
  const use = (c: PromptCandidate) => {
    if (hasText) setPending({ c, s: request })
    else onUse(c, request)
  }

  return (
    <Panel
      title="Start from an idea"
      description="Describe the thesis in plain English and choose the options. This only assembles text: no model is called and nothing is saved."
    >
      <div className="flex flex-col gap-4">
        <Field label="Investment idea">
          <Textarea
            rows={3}
            value={thesis}
            onChange={(e) => setThesis(e.target.value)}
            maxLength={4000}
            placeholder="Identify global semiconductor infrastructure companies benefiting from AI investment, with strong balance sheets and attractive forward earnings, while shorting overvalued companies with declining margins."
          />
        </Field>
        <div className="grid gap-4 md:grid-cols-2">
          <Field label="Universe or region">
            <Input value={universe} onChange={(e) => setUniverse(e.target.value)} />
          </Field>
          <Field label="Sector or industry (optional)">
            <Input value={sector} onChange={(e) => setSector(e.target.value)} />
          </Field>
        </div>
        <Fieldset legend="Direction">
          <Segmented
            label="Direction"
            value={direction}
            onChange={setDirection}
            items={[
              { value: 'long_short', label: 'Long / short' },
              { value: 'long_only', label: 'Long only' },
              { value: 'short_only', label: 'Short only' },
            ]}
          />
        </Fieldset>
        <Fieldset
          legend="Approaches"
          hint="One prompt is written per approach, so you can compare them."
        >
          <Chips label="Approaches" items={choices} value={approaches} onChange={setApproaches} />
        </Fieldset>
        <Disclosure summary="More options">
          <div className="grid gap-4 md:grid-cols-3">
            <Field label="Securities" hint="A guideline: about 50 per prompt is suggested.">
              <Input
                inputMode="numeric"
                value={target}
                onChange={(e) => setTarget(e.target.value)}
              />
            </Field>
            <Field label="Horizon, days">
              <Input
                inputMode="numeric"
                value={horizon}
                onChange={(e) => setHorizon(e.target.value)}
              />
            </Field>
            <Field label="Prefer evidence from the last, days">
              <Input
                inputMode="numeric"
                value={freshness}
                onChange={(e) => setFreshness(e.target.value)}
              />
            </Field>
            <Field label="Max sector, %">
              <Input
                inputMode="numeric"
                value={sectorCap}
                onChange={(e) => setSectorCap(e.target.value)}
              />
            </Field>
            <Field label="Max country, %">
              <Input
                inputMode="numeric"
                value={countryCap}
                onChange={(e) => setCountryCap(e.target.value)}
              />
            </Field>
            <Field label="Update frequency">
              <Select
                label="Generated update frequency"
                value={frequency}
                onChange={setFrequency}
                items={['daily', 'weekly', 'monthly', 'quarterly'].map((f) => ({
                  value: f as typeof frequency,
                  label: f,
                }))}
              />
            </Field>
            <Fieldset
              legend="Return prediction unit"
              hint="The official examples could not be read in full, so the prompt states the unit explicitly. Check it against WorldQuant's example."
              className="md:col-span-3"
            >
              <Segmented
                label="Return unit"
                value={returnUnit}
                onChange={setReturnUnit}
                items={[
                  { value: 'fraction', label: 'Fraction (0.05)' },
                  { value: 'percent', label: 'Percent (5)' },
                ]}
              />
            </Fieldset>
            <Field label="Extra constraints, in your words" className="md:col-span-3">
              <Textarea
                rows={2}
                value={constraints}
                onChange={(e) => setConstraints(e.target.value)}
                maxLength={1500}
              />
            </Field>
          </div>
        </Disclosure>
        <div>
          <Button
            variant="primary"
            loading={generate.isPending}
            disabled={!thesis.trim() || approaches.length === 0}
            onClick={() => generate.mutate()}
          >
            Generate prompts
          </Button>
        </div>
        {generate.isError && <ErrorNotice error={generate.error} title="Could not generate" />}
        {generate.data && (
          <div className="flex flex-col gap-3">
            {generate.data.candidates.map((c) => (
              <div key={c.approach} className="rounded-md border border-hairline bg-surface-2 p-3">
                <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                  <span className="flex items-center gap-2 text-body font-medium text-ink">
                    {c.label} <Badge tone="muted">{c.characters.toLocaleString()} characters</Badge>
                  </span>
                  <Button size="sm" variant="primary" onClick={() => use(c)}>
                    Use this prompt
                  </Button>
                </div>
                {c.warnings.map((w) => (
                  <Notice key={w} tone="warn" className="mb-2">
                    {w}
                  </Notice>
                ))}
                <pre className="num max-h-48 overflow-auto whitespace-pre-wrap break-words text-body-compact text-ink-muted">
                  {c.promptText}
                </pre>
              </div>
            ))}
          </div>
        )}
      </div>
      <Confirm
        open={pending !== null}
        onOpenChange={(o) => !o && setPending(null)}
        title="Replace the prompt text?"
        confirmLabel="Replace"
        onConfirm={() => {
          if (pending) onUse(pending.c, pending.s)
          setPending(null)
        }}
      >
        The editor already holds text. Replacing it does not lose anything that is saved: earlier
        versions stay as they are.
      </Confirm>
    </Panel>
  )
}

// ── versions ────────────────────────────────────────────────────────────────────────────

function Versions({
  saved,
  onCompare,
  onLoad,
}: {
  saved: PromptDetail
  onCompare: (a: number, b: number) => void
  onLoad: (v: PromptVersion) => void
}) {
  const invalidate = useInvalidate()
  const [a, setA] = useState<number | null>(null)
  const restore = useMutation({
    mutationFn: (n: number) => spc.restore(saved.id, n),
    onSuccess: (p) => {
      toast.success(`Restored as version ${p.latestVersion}`)
      invalidate()
    },
  })
  const prefer = useMutation({
    mutationFn: (id: number) => spc.update(saved.id, { preferredVersionId: id }),
    onSuccess: () => {
      toast.success('Marked as the preferred version')
      invalidate()
    },
  })
  return (
    <Panel
      title="Versions"
      description="Every saved text, newest first. A version never changes; a run keeps its own copy of what it tested."
    >
      <ul className="flex flex-col divide-y divide-hairline-subtle">
        {saved.versions.map((v) => (
          <li key={v.id} className="flex flex-wrap items-center justify-between gap-2 py-2">
            <div className="flex min-w-0 flex-col gap-0.5">
              <span className="flex flex-wrap items-center gap-2 text-body text-ink">
                <span className="num font-medium">Version {v.number}</span>
                {v.id === saved.preferredVersionId && <Badge tone="profit">preferred</Badge>}
                <span className="text-body-compact text-ink-subtle">
                  {fmt.dateTime(v.createdAt)} · {v.characters.toLocaleString()} characters ·{' '}
                  {v.runs} run{v.runs === 1 ? '' : 's'}
                  {v.runs > 0 ? `, ${v.validRuns} valid` : ''}
                </span>
              </span>
              {v.note && <span className="text-body-compact text-ink-muted">{v.note}</span>}
            </div>
            <div className="flex flex-wrap gap-1.5">
              <Button size="sm" variant="ghost" onClick={() => onLoad(v)}>
                Load into editor
              </Button>
              <Button
                size="sm"
                variant="ghost"
                aria-pressed={a === v.number}
                onClick={() => {
                  if (a === null) setA(v.number)
                  else if (a === v.number) setA(null)
                  else {
                    onCompare(Math.min(a, v.number), Math.max(a, v.number))
                    setA(null)
                  }
                }}
              >
                {a === null ? 'Compare…' : a === v.number ? 'Cancel compare' : `Compare with ${a}`}
              </Button>
              <Button
                size="sm"
                variant="ghost"
                loading={restore.isPending}
                onClick={() => restore.mutate(v.number)}
              >
                Restore
              </Button>
              <Button
                size="sm"
                variant="ghost"
                disabled={v.id === saved.preferredVersionId}
                loading={prefer.isPending}
                onClick={() => prefer.mutate(v.id)}
              >
                Mark preferred
              </Button>
            </div>
          </li>
        ))}
      </ul>
    </Panel>
  )
}

function CompareDialog({
  promptId,
  pair,
  onClose,
}: {
  promptId: number
  pair: [number, number]
  onClose: () => void
}) {
  const [a, b] = pair
  const diff = useMutation({ mutationFn: () => spc.compare(promptId, a, b) })
  const { mutate } = diff
  // biome-ignore lint/correctness/useExhaustiveDependencies: fetch once per pair
  useEffect(() => mutate(), [a, b])
  return (
    <Dialog
      open
      onOpenChange={(o) => !o && onClose()}
      title={`Version ${a} → version ${b}`}
      description={
        diff.data
          ? `${diff.data.added} lines added, ${diff.data.removed} removed, ${diff.data.characterChange >= 0 ? '+' : ''}${diff.data.characterChange} characters.`
          : undefined
      }
      className="max-w-4xl"
    >
      {diff.isPending && <Skeleton className="h-48" />}
      {diff.isError && <ErrorNotice error={diff.error} title="Could not compare" />}
      {diff.data && (
        <pre className="num max-h-[60vh] overflow-auto text-body-compact">
          {diff.data.lines.map((l, i) => (
            <div
              key={i}
              className={cn(
                'whitespace-pre-wrap break-words px-2',
                l.kind === 'add' && 'bg-pnl-positive-tint text-pnl-positive-text',
                l.kind === 'del' && 'bg-pnl-negative-tint text-pnl-negative-text',
                l.kind === 'ctx' && 'text-ink-subtle',
              )}
            >
              {/* The sign and the colour say the same thing: a change is never colour alone. */}
              {l.kind === 'add' ? '+ ' : l.kind === 'del' ? '− ' : '  '}
              {l.text}
            </div>
          ))}
        </pre>
      )}
    </Dialog>
  )
}

export { shortModel }
