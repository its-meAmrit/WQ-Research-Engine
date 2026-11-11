/**
 * Alpha Builder: an approved idea, written as reviewable alpha candidates.
 *
 * Nothing on this screen simulates. Designing and generating call the model; validating,
 * saving and previewing are local. A task is only stored, idle, when the person confirms it,
 * and it runs only when they press Run in Tasks.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { MAX_DECAY } from '@/api/platform'
import { type Scope, scopeLabel } from '@/api/types'
import { useScope, useScopeOptions } from '@/lib/scope'
import {
  Badge,
  Button,
  Checkbox,
  Chips,
  Disclosure,
  ErrorNotice,
  Field,
  Input,
  Notice,
  Panel,
  Segmented,
  Skeleton,
} from '@/ui/kit'
import { Select } from '@/ui/overlay'
import { ScopePicker } from '@/ui/scope-picker'
import {
  builder,
  type Design,
  type MappingEntry,
  type Options,
  research,
  type SimSettings,
} from './api'
import { CandidateCard, TaskPanel } from './builder-candidates'
import { useJobWatch } from './builder-jobs'
import { KEYS } from './shared'

const KINDS: { value: string; label: string }[] = [
  { value: 'replication', label: 'Closest replication' },
  { value: 'simplified', label: 'Simplified' },
  { value: 'proxy', label: 'Documented proxy' },
  { value: 'direction', label: 'Opposite direction' },
  { value: 'normalization', label: 'Different normalisation' },
  { value: 'event_aware', label: 'Event-aware' },
  { value: 'robust', label: 'Robustness-oriented' },
]

const STATUS: Record<
  MappingEntry['status'],
  { tone: 'profit' | 'warn' | 'loss' | 'muted'; label: string }
> = {
  exact: { tone: 'profit', label: 'Wording match' },
  manual: { tone: 'warn', label: 'Your substitution' },
  proxy: { tone: 'warn', label: 'Proxy' },
  missing: { tone: 'loss', label: 'Not found' },
  other_scope: { tone: 'loss', label: 'Other markets only' },
  unsupported: { tone: 'loss', label: 'Not on BRAIN' },
}

/** The ideas worth building from: saved first, the rest on request. */
function IdeaPicker({ value, onChange }: { value: string | null; onChange: (id: string) => void }) {
  const [all, setAll] = useState(false)
  const ideas = useQuery({
    queryKey: [...KEYS.ideas, 'builder', all],
    queryFn: () => research.ideas(all ? {} : { status: 'saved' }),
  })
  const list = (ideas.data ?? []).filter((i) => i.status !== 'rejected')
  return (
    <Field label="Research idea">
      <div className="flex flex-wrap items-center gap-2">
        {ideas.isPending ? (
          <Skeleton className="h-8 w-64" label="Loading ideas" />
        ) : list.length === 0 ? (
          <span className="text-body-compact text-ink-subtle">
            No {all ? '' : 'saved '}ideas yet. Save an idea in the Idea library first.
          </span>
        ) : (
          <Select
            label="Research idea"
            value={value}
            onChange={onChange}
            items={list.map((i) => ({ value: i.id, label: i.title }))}
          />
        )}
        <Checkbox checked={all} onChange={setAll} label="Include ideas I have not saved" />
      </div>
      {ideas.isError && <ErrorNotice error={ideas.error} />}
    </Field>
  )
}

interface SettingsState {
  neutralization: string | null
  decay: string
  truncation: string
  pasteurization: 'ON' | 'OFF'
  nanHandling: 'ON' | 'OFF'
}

const settingsBody = (scope: Scope, s: SettingsState): SimSettings => ({
  instrumentType: scope.instrumentType,
  region: scope.region,
  delay: scope.delay,
  universe: scope.universe,
  language: 'FASTEXPR',
  neutralization: s.neutralization,
  decay: s.decay.trim() === '' ? null : Number(s.decay),
  truncation: s.truncation.trim() === '' ? null : Number(s.truncation),
  pasteurization: s.pasteurization,
  nanHandling: s.nanHandling,
  maxTrade: 'OFF',
  maxPosition: 'OFF',
})

function MappingTable({ design, scope }: { design: Design; scope: Scope }) {
  const client = useQueryClient()
  const [editing, setEditing] = useState<string | null>(null)
  const [text, setText] = useState('')
  const patch = useMutation({
    mutationFn: (body: { substitutions?: Record<string, string>; acceptedProxies?: string[] }) =>
      builder.patchMapping(design.id, body),
    onSuccess: () => {
      setEditing(null)
      void client.invalidateQueries({ queryKey: ['research', 'builder', 'designs'] })
    },
  })
  const hits = useQuery({
    queryKey: ['research', 'builder', 'fields', scopeLabel(scope), text],
    queryFn: () => builder.searchFields(scope, text),
    enabled: editing != null && text.trim().length > 1,
  })
  return (
    <div className="flex flex-col gap-2">
      <p className="text-caption text-ink-subtle">
        Fields come from your downloaded catalogue for exactly {scopeLabel(scope)}. A proxy is not
        an exact match and is not used until you accept it.
      </p>
      {design.mapping.map((m) => (
        <div key={m.name} className="flex flex-col gap-1 rounded-md border border-hairline p-3">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-body font-medium text-ink">{m.name}</span>
            <Badge tone={STATUS[m.status].tone}>{STATUS[m.status].label}</Badge>
            {!m.essential && <Badge tone="muted">optional</Badge>}
            <Badge tone="outline">{m.expectedType}</Badge>
            {!m.sourceBacked && <Badge tone="warn">model's addition</Badge>}
          </div>
          <p className="text-caption text-ink-subtle">{m.meaning}</p>
          {m.final ? (
            <dl className="grid grid-cols-[auto_1fr] gap-x-3 text-body-compact">
              <dt className="text-ink-muted">BRAIN field</dt>
              <dd className="num">{m.final.fieldId}</dd>
              <dt className="text-ink-muted">Dataset</dt>
              <dd>{m.final.datasetName ?? m.final.datasetId ?? '—'}</dd>
              <dt className="text-ink-muted">Type</dt>
              <dd>{m.final.fieldType ?? 'unknown'}</dd>
              <dt className="text-ink-muted">Scope</dt>
              <dd>Available in {scopeLabel(scope)}</dd>
              <dt className="text-ink-muted">Description</dt>
              <dd>{m.final.description}</dd>
            </dl>
          ) : (
            <p className="text-body-compact text-loss">{m.blockedReason ?? m.rationale}</p>
          )}
          <p className="text-caption text-ink-subtle">{m.rationale}</p>
          {m.approximation && <p className="text-caption text-warn-text">{m.approximation}</p>}
          {m.typeNote && <p className="text-caption text-warn-text">{m.typeNote}</p>}
          {m.availableElsewhere.length > 0 && (
            <p className="text-caption text-ink-subtle">
              Exists elsewhere (not used):{' '}
              {m.availableElsewhere
                .slice(0, 4)
                .map((e) => `${e.region} D${e.delay} ${e.universe}`)
                .join('; ')}
            </p>
          )}
          <div className="flex flex-wrap items-center gap-2">
            {m.status === 'proxy' && (
              <Button
                loading={patch.isPending}
                onClick={() =>
                  patch.mutate({
                    acceptedProxies: m.usable
                      ? design.acceptedProxies.filter((n) => n !== m.name)
                      : [...design.acceptedProxies, m.name],
                  })
                }
              >
                {m.usable ? 'Withdraw proxy' : 'Accept proxy'}
              </Button>
            )}
            <Button
              onClick={() => {
                setEditing(editing === m.name ? null : m.name)
                setText('')
              }}
            >
              Choose another field
            </Button>
            {design.substitutions[m.name] && (
              <Button onClick={() => patch.mutate({ substitutions: { [m.name]: '' } })}>
                Remove my substitution
              </Button>
            )}
          </div>
          {editing === m.name && (
            <div className="flex flex-col gap-2 rounded-md bg-surface-2 p-2">
              <Input
                aria-label={`Search fields in ${scopeLabel(scope)}`}
                placeholder="Search field ids and descriptions"
                value={text}
                onChange={(e) => setText(e.target.value)}
              />
              {hits.isFetching && <Skeleton className="h-6" label="Searching" />}
              {hits.data?.length === 0 && (
                <p className="text-caption text-ink-subtle">No field here matches that.</p>
              )}
              {hits.data?.map((h) => (
                <button
                  type="button"
                  key={h.fieldId}
                  className="flex flex-col items-start rounded-sm px-2 py-1 text-left hover:bg-surface-1"
                  onClick={() =>
                    patch.mutate({
                      substitutions: { ...design.substitutions, [m.name]: h.fieldId },
                    })
                  }
                >
                  <span className="num text-body-compact text-ink">
                    {h.fieldId}{' '}
                    <span className="text-ink-subtle">
                      · {h.fieldType ?? '?'} · {h.datasetId}
                    </span>
                  </span>
                  <span className="text-caption text-ink-subtle">{h.description}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      ))}
      {patch.error != null && <ErrorNotice error={patch.error} />}
    </div>
  )
}

export function AlphaBuilder({
  options,
  initialIdea,
}: {
  options: Options | undefined
  initialIdea: string | null
}) {
  const client = useQueryClient()
  const [ideaId, setIdeaId] = useState<string | null>(initialIdea)
  const [scope, setScope] = useScope('labs/builder')
  const scopeOptions = useScopeOptions(scope)
  const meta = useQuery({ queryKey: ['research', 'builder', 'options'], queryFn: builder.options })
  const [settings, setSettings] = useState<SettingsState>({
    neutralization: null,
    decay: '',
    truncation: '',
    pasteurization: 'ON',
    nanHandling: 'OFF',
  })
  const [confirmed, setConfirmed] = useState(false)
  const [count, setCount] = useState(3)
  const [kinds, setKinds] = useState<string[]>(['replication', 'simplified', 'robust'])
  const models = options?.models ?? []
  const [model, setModel] = useState<string | null>(null)
  const modelRef = model ?? models[0]?.ref ?? null
  const [selected, setSelected] = useState<string[]>([])

  const idea = useQuery({
    queryKey: ['research', 'idea', ideaId],
    queryFn: () => research.idea(ideaId as string),
    enabled: ideaId != null,
  })
  const designs = useQuery({
    queryKey: ['research', 'builder', 'designs', ideaId],
    queryFn: () => builder.designs(ideaId as string),
    enabled: ideaId != null,
  })
  const candidates = useQuery({
    queryKey: ['research', 'builder', 'candidates', ideaId],
    queryFn: () => builder.candidates(ideaId as string),
    enabled: ideaId != null,
  })
  const refresh = () => void client.invalidateQueries({ queryKey: ['research', 'builder'] })
  const designWatch = useJobWatch(refresh)
  const variantsWatch = useJobWatch(refresh)

  const synced = (meta.data?.syncedScopes ?? []).some(
    (s) =>
      s.instrumentType === scope.instrumentType &&
      s.region === scope.region &&
      s.delay === scope.delay &&
      s.universe === scope.universe,
  )
  // The design the person is working from: the newest one made in the scope on screen.
  const design =
    designs.data?.find(
      (d) =>
        d.scope['region'] === scope.region &&
        d.scope['delay'] === scope.delay &&
        d.scope['universe'] === scope.universe &&
        d.scope['instrumentType'] === scope.instrumentType,
    ) ?? null
  const otherScopes = (designs.data ?? []).length - (design ? 1 : 0)

  const decay = Number(settings.decay)
  const truncation = Number(settings.truncation)
  const settingsReason =
    settings.neutralization == null
      ? 'Choose a neutralization.'
      : settings.decay.trim() === '' || !Number.isInteger(decay) || decay < 0 || decay > MAX_DECAY
        ? `Enter a decay from 0 to ${MAX_DECAY}.`
        : settings.truncation.trim() === '' || !(truncation >= 0 && truncation <= 1)
          ? 'Enter a truncation from 0 to 1.'
          : !confirmed
            ? 'Confirm the settings.'
            : null
  const designReason = !ideaId
    ? 'Choose a research idea.'
    : meta.isPending
      ? 'Loading your markets…'
      : !synced
        ? `${scopeLabel(scope)} is not downloaded. Sync it in the Data Explorer; the scope is never switched for you.`
        : !modelRef
          ? 'Add and enable an LLM key under API Integration.'
          : null
  const blockedEssential = design?.blocked.filter((b) => b.essential) ?? []
  const variantsReason = !design
    ? 'Make a signal design first.'
    : blockedEssential[0]
      ? `${blockedEssential[0].variable}: ${blockedEssential[0].reason}`
      : (settingsReason ?? (kinds.length === 0 ? 'Choose at least one kind of variant.' : null))
  const maxVariants = meta.data?.maxVariants ?? 7

  // The empty drafts the Paper Analyzer hands off carry no expression; they are not candidates.
  const drafts = (candidates.data ?? []).filter((c) => c.expression)
  const selectedCandidates = drafts.filter((c) => selected.includes(c.id))
  const pickNeutral = scopeOptions.neutralizations

  return (
    <div className="flex flex-col gap-4">
      <Notice tone="info">
        An expression that passes these checks is only well formed for the metadata here. It can
        still fail BRAIN's own checks or perform badly. Nothing is simulated until you create a task
        and press Run in Tasks.
      </Notice>
      <Panel
        title="1 · Research idea"
        description="The source and the evidence the candidates are built from."
      >
        <div className="flex flex-col gap-3 p-4">
          <IdeaPicker value={ideaId} onChange={setIdeaId} />
          {idea.isError && <ErrorNotice error={idea.error} />}
          {idea.data && (
            <div className="flex flex-col gap-1">
              <h3 className="text-body font-semibold text-ink">{idea.data.title}</h3>
              <p className="text-body-compact text-ink-muted">{idea.data.economicHypothesis}</p>
              {idea.data.source && (
                <p className="text-caption text-ink-subtle">Source: {idea.data.source.title}</p>
              )}
              <Disclosure summary={`Evidence (${idea.data.evidence.length})`}>
                <div className="flex flex-col gap-2 p-3">
                  {idea.data.evidence.map((e) => (
                    <blockquote
                      key={e.quote}
                      className="border-l-2 border-hairline pl-3 text-caption text-ink-subtle"
                    >
                      “{e.quote}” (page {e.page})
                    </blockquote>
                  ))}
                </div>
              </Disclosure>
            </div>
          )}
        </div>
      </Panel>

      <Panel
        title="2 · Market and simulation settings"
        description="Confirmed by you and kept through every step. Choices come from BRAIN's own settings list."
      >
        <div className="flex flex-col gap-3 p-4">
          <div className="flex flex-wrap items-center gap-3">
            <Badge tone="muted">{scope.instrumentType}</Badge>
            <ScopePicker scope={scope} onChange={setScope} />
            <Badge tone="outline">Language: FASTEXPR</Badge>
          </div>
          {!synced && !meta.isPending && (
            <Notice tone="error" title="This market is not downloaded">
              The catalogue holds no fields for {scopeLabel(scope)}, so no field can be matched or
              checked. Sync it in the Data Explorer, or pick a market that is synced. It will not be
              replaced with another one.
            </Notice>
          )}
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <Field label="Neutralization">
              <Select
                label="Neutralization"
                value={settings.neutralization}
                onChange={(neutralization) => setSettings({ ...settings, neutralization })}
                items={pickNeutral}
              />
            </Field>
            <Field label={`Decay (0–${MAX_DECAY})`}>
              <Input
                type="number"
                min={0}
                max={MAX_DECAY}
                value={settings.decay}
                onChange={(e) => setSettings({ ...settings, decay: e.target.value })}
              />
            </Field>
            <Field label="Truncation (0–1)">
              <Input
                type="number"
                min={0}
                max={1}
                step={0.01}
                value={settings.truncation}
                onChange={(e) => setSettings({ ...settings, truncation: e.target.value })}
              />
            </Field>
            <Field label="Pasteurization">
              <Segmented
                label="Pasteurization"
                value={settings.pasteurization}
                onChange={(pasteurization) => setSettings({ ...settings, pasteurization })}
                items={[
                  { value: 'ON', label: 'On' },
                  { value: 'OFF', label: 'Off' },
                ]}
              />
            </Field>
          </div>
          <Checkbox
            checked={confirmed}
            onChange={setConfirmed}
            label={`I confirm ${scopeLabel(scope)} with these settings. Nothing is filled in for me.`}
          />
        </div>
      </Panel>

      <Panel
        title="3 · Signal design and data fields"
        description="What the model proposes, and which real BRAIN fields stand for it."
      >
        <div className="flex flex-col gap-3 p-4">
          <div className="flex flex-wrap items-center gap-2">
            {models.length > 0 && (
              <Select
                label="Model"
                value={modelRef}
                onChange={setModel}
                items={models.map((m) => ({ value: m.ref, label: `${m.id} (${m.provider})` }))}
              />
            )}
            <Button
              variant="primary"
              disabled={designReason != null}
              loading={designWatch.busy}
              onClick={() =>
                ideaId && designWatch.start(() => builder.design(ideaId, scope, modelRef))
              }
            >
              {design ? 'Make a new signal design' : 'Make a signal design'}
            </Button>
            {designWatch.busy && (
              <Button variant="ghost" onClick={designWatch.cancel}>
                Cancel
              </Button>
            )}
            {designReason && (
              <span className="text-caption text-ink-subtle">Unavailable: {designReason}</span>
            )}
          </div>
          {designWatch.busy && designWatch.job && (
            <p className="text-caption text-ink-subtle">{designWatch.job.message}</p>
          )}
          {designWatch.startError != null && <ErrorNotice error={designWatch.startError} />}
          {designWatch.job?.status === 'failed' && (
            <Notice
              tone="error"
              title="The design was not kept"
              action={<Button onClick={designWatch.clear}>Dismiss</Button>}
            >
              {designWatch.job.errorMessage}
            </Notice>
          )}
          {designs.isPending && ideaId && <Skeleton className="h-24" label="Loading designs" />}
          {!design && designs.data && ideaId && (
            <p className="text-body-compact text-ink-subtle">
              No signal design for {scopeLabel(scope)} yet.
              {otherScopes > 0 &&
                ` (${otherScopes} exist for other markets and are not used here.)`}
            </p>
          )}
          {design && (
            <>
              <div className="flex flex-col gap-1 text-body-compact text-ink-muted">
                <p>
                  <b>Hypothesis:</b> {design.design.economic_hypothesis}
                </p>
                <p>
                  <b>Construction:</b> {design.design.signal_construction}
                </p>
                <p>
                  <b>Direction:</b> {design.design.expected_direction} —{' '}
                  {design.design.direction_basis}
                </p>
                <p>
                  <b>Lookback:</b> {design.design.lookback}
                </p>
                <p>
                  <b>Timing assumptions:</b> {design.design.data_timing_assumptions}
                </p>
                <p>
                  <b>Operators:</b> {design.design.relevant_operators.join(', ')}
                </p>
                {(design.design.unknown_operators?.length ?? 0) > 0 && (
                  <p className="text-warn-text">
                    Not in your BRAIN operator list: {design.design.unknown_operators?.join(', ')}.
                  </p>
                )}
                {design.design.adaptations.length > 0 && (
                  <div>
                    <b>Adaptations (the model's own, not the source's):</b>
                    <ul className="list-disc pl-5">
                      {design.design.adaptations.map((a) => (
                        <li key={a.change}>
                          {a.change}: {a.why}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {design.design.implementation_limitations.length > 0 && (
                  <div>
                    <b>Limitations:</b>
                    <ul className="list-disc pl-5">
                      {design.design.implementation_limitations.map((l) => (
                        <li key={l}>{l}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
              <MappingTable design={design} scope={scope} />
            </>
          )}
        </div>
      </Panel>

      <Panel
        title="4 · Candidates"
        description="A small, bounded set of documented variants. Drafts until you save them."
      >
        <div className="flex flex-col gap-3 p-4">
          <Chips label="Kinds of variant" value={kinds} onChange={setKinds} items={KINDS} />
          <div className="flex flex-wrap items-center gap-3">
            <Field label={`How many (1–${maxVariants})`}>
              <Input
                type="number"
                min={1}
                max={maxVariants}
                value={count}
                onChange={(e) =>
                  setCount(Math.min(maxVariants, Math.max(1, Number(e.target.value) || 1)))
                }
              />
            </Field>
            <Button
              variant="primary"
              disabled={variantsReason != null}
              loading={variantsWatch.busy}
              onClick={() =>
                design &&
                variantsWatch.start(() =>
                  builder.variants(design.id, {
                    settings: settingsBody(scope, settings),
                    settingsConfirmed: confirmed,
                    count,
                    kinds,
                    modelRef,
                  }),
                )
              }
            >
              Generate candidates
            </Button>
            {variantsWatch.busy && (
              <Button variant="ghost" onClick={variantsWatch.cancel}>
                Cancel
              </Button>
            )}
          </div>
          {variantsReason && (
            <p className="text-caption text-ink-subtle">Unavailable: {variantsReason}</p>
          )}
          {variantsWatch.busy && variantsWatch.job && (
            <p className="text-caption text-ink-subtle">{variantsWatch.job.message}</p>
          )}
          {variantsWatch.startError != null && <ErrorNotice error={variantsWatch.startError} />}
          {variantsWatch.job?.status === 'failed' && (
            <Notice
              tone="error"
              title="Nothing was saved"
              action={<Button onClick={variantsWatch.clear}>Dismiss</Button>}
            >
              {variantsWatch.job.errorMessage} You can try again; the earlier candidates are
              untouched.
            </Notice>
          )}
          {variantsWatch.job?.status === 'completed' &&
            variantsWatch.job.result?.['fewerReason'] != null && (
              <Notice tone="info">{String(variantsWatch.job.result['fewerReason'])}</Notice>
            )}
          {candidates.isPending && ideaId && (
            <Skeleton className="h-24" label="Loading candidates" />
          )}
          {candidates.isError && <ErrorNotice error={candidates.error} />}
          {drafts.length === 0 && candidates.data && (
            <p className="text-body-compact text-ink-subtle">No candidates for this idea yet.</p>
          )}
          {drafts.map((c) => (
            <CandidateCard
              key={`${c.id}-${c.status}-${c.validationOutcome}`}
              candidate={c}
              all={drafts}
              selected={selected.includes(c.id)}
              onSelect={(on) =>
                setSelected((s) => (on ? [...new Set([...s, c.id])] : s.filter((x) => x !== c.id)))
              }
              modelRef={modelRef}
            />
          ))}
        </div>
      </Panel>

      {drafts.length > 0 && <TaskPanel selected={selectedCandidates} />}
    </div>
  )
}
