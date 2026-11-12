/** Super Alpha Lab, Advanced Research: operator-aware SuperAlpha research. Configuration and the
 *  Operator Explorer here; the running task (candidates, dashboard, Run/Pause/Stop) in
 *  `research-monitor.tsx`. Everything shown comes from `/api/super-lab/advanced`. */

import { keepPreviousData, useQuery } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { components } from '@/api/generated'
import { http, qs } from '@/api/http'
import {
  REGION_AGNOSTIC_REGION,
  SUPER_ALPHA_MIN_SELECTION,
  SUPER_ALPHA_SLOTS,
} from '@/api/platform'
import { cn } from '@/lib/cn'
import { fmt } from '@/lib/format'
import { useScope, useScopeOptions } from '@/lib/scope'
import { followOtherTabs } from '@/lib/tabs'
import { AddTaskButtons, useAddTask } from '@/screens/research-labs/add-task'
import {
  Badge,
  Checkbox,
  Chips,
  Disclosure,
  ErrorNotice,
  Field,
  Fieldset,
  Input,
  Metric,
  Notice,
  Page,
  PageHeader,
  Panel,
  Segmented,
  Switch,
} from '@/ui/kit'
import { Select } from '@/ui/overlay'
import { ScopePicker } from '@/ui/scope-picker'
import { ResearchMonitor } from './research-monitor'
import type { CoverageSummary, OperatorCatalog, OperatorRow, ResearchPlan } from './research-types'

type Schemas = components['schemas']
type Body = Schemas['ResearchTask']
type Activation = NonNullable<Body['activation']>
type Strategy = NonNullable<Body['strategy']>
type Objective = NonNullable<Body['objective']>

const B = '/api/super-lab/advanced'
const MAX_LIMIT = 200

const ACTIVATIONS: { value: Activation; label: string }[] = [
  { value: 'BOTH', label: 'IS and OS' },
  { value: 'IS', label: 'IS' },
  { value: 'OS', label: 'OS' },
]
const STRATEGIES: { value: Strategy; label: string }[] = [
  { value: 'mixed', label: 'Mixed' },
  { value: 'templates', label: 'Templates' },
  { value: 'grammar', label: 'Grammar' },
  { value: 'evolve', label: 'Evolve' },
]
const STRATEGY_HINT: Record<Strategy, string> = {
  mixed: 'Research templates, grammar-built expressions, and mutations of what scored well.',
  templates: 'Only the economically motivated templates, with their parameters explored.',
  grammar: 'Typed expressions grown from every verified operator, with no template.',
  evolve: 'Breeds what scored well; falls back to templates until there are results.',
}
const OBJECTIVES: { value: Objective; label: string }[] = [
  { value: 'balanced', label: 'Balanced' },
  { value: 'sharpe', label: 'Sharpe' },
  { value: 'fitness', label: 'Fitness' },
  { value: 'robust', label: 'Robust' },
]
const OBJECTIVE_HINT: Record<Objective, string> = {
  balanced:
    'Sharpe and Fitness, less drawdown, turnover outside 2%–40%, and structural check failures; credit for beating equal weight.',
  sharpe: 'Sharpe alone.',
  fitness: 'Fitness alone.',
  robust:
    'Balanced, with heavier penalties for drawdown and turnover and more credit for beating equal weight.',
}
const CORES = Array.from({ length: SUPER_ALPHA_SLOTS }, (_, i) => ({
  value: i + 1,
  label: String(i + 1),
}))

/** Operators and families switched off, kept on this computer. */
interface Switches {
  operators: string[]
  families: string[]
  preferred: string[]
}
const useSwitches = create<Switches>()(
  persist((): Switches => ({ operators: [], families: [], preferred: [] }), {
    name: 'alpha-harness-super-research',
    version: 1,
  }),
)
followOtherTabs(useSwitches)

const flip = (list: string[], item: string) =>
  list.includes(item) ? list.filter((x) => x !== item) : [...list, item]

/** `4, 10` → [4, 10]; `null` when any part is not a number. */
function numbers(text: string, whole: boolean): number[] | null {
  const parts = text
    .split(/[\s,]+/)
    .map((p) => p.trim())
    .filter(Boolean)
  if (parts.length === 0) return null
  const values = parts.map(Number)
  if (values.some((v) => !Number.isFinite(v) || (whole && !Number.isInteger(v)))) return null
  return values
}

const words = (family: string) => family.replace(/_/g, ' ')

export function AdvancedLab({ switcher }: { switcher: React.ReactNode }) {
  const [scope, setScope] = useScope('/labs/super-alpha')
  const options = useScopeOptions(scope)
  const [picked, setPicked] = useState('')
  const [limit, setLimit] = useState(30)
  const [activation, setActivation] = useState<Activation>('BOTH')
  const [decayText, setDecayText] = useState('4, 10')
  const [truncText, setTruncText] = useState('0.05')
  const [simulations, setSimulations] = useState(60)
  const [cores, setCores] = useState<number>(SUPER_ALPHA_SLOTS)
  const [strategy, setStrategy] = useState<Strategy>('mixed')
  const [objective, setObjective] = useState<Objective>('balanced')
  const [maxDepth, setMaxDepth] = useState(5)
  const [maxOperators, setMaxOperators] = useState(14)
  const [maxVariables, setMaxVariables] = useState(3)
  const [seed, setSeed] = useState(0)
  const [exploration, setExploration] = useState(0.5)
  const [mutation, setMutation] = useState(0.6)
  const [patience, setPatience] = useState(4)
  const [relaxed, setRelaxed] = useState(false)
  const switches = useSwitches()

  const neutralization = options.neutralizations.some((n) => n.value === picked)
    ? picked
    : (options.neutralizations.find((n) => n.value === 'SUBINDUSTRY')?.value ??
      options.neutralizations[0]?.value ??
      '')
  const agnostic = scope.region === REGION_AGNOSTIC_REGION
  const decays = numbers(decayText, true)
  const truncations = numbers(truncText, false)
  const problem = (() => {
    if (!Number.isInteger(limit) || limit < SUPER_ALPHA_MIN_SELECTION || limit > MAX_LIMIT)
      return `Selection Limit must be a whole number from ${SUPER_ALPHA_MIN_SELECTION} to ${MAX_LIMIT}.`
    if (!decays || decays.length > 6 || decays.some((d) => d < 0 || d > 512))
      return 'Decay must be up to six whole numbers from 0 to 512.'
    if (!truncations || truncations.length > 4 || truncations.some((t) => t < 0 || t > 1))
      return 'Truncation must be up to four numbers from 0 to 1.'
    if (!Number.isInteger(simulations) || simulations < 1 || simulations > 2000)
      return 'Simulation budget must be a whole number from 1 to 2,000.'
    if (!Number.isInteger(seed) || seed < 0) return 'Seed must be a whole number, 0 or more.'
    if (!Number.isInteger(maxDepth) || maxDepth < 1 || maxDepth > 8)
      return 'Maximum depth must be a whole number from 1 to 8.'
    if (!Number.isInteger(maxOperators) || maxOperators < 3 || maxOperators > 30)
      return 'Maximum operators must be a whole number from 3 to 30.'
    if (!Number.isInteger(maxVariables) || maxVariables < 0 || maxVariables > 6)
      return 'Maximum variables must be a whole number from 0 to 6.'
    if (!Number.isInteger(patience) || patience < 1 || patience > 20)
      return 'Patience must be a whole number from 1 to 20.'
    return null
  })()

  const body: Body = {
    region: scope.region,
    delay: scope.delay,
    universe: scope.universe,
    neutralization,
    selectionLimit: limit,
    activation,
    decays: decays ?? [],
    truncations: truncations ?? [],
    simulations,
    cores,
    objective,
    strategy,
    maxDepth,
    maxOperators,
    maxVariables,
    seed,
    exploration,
    mutationProbability: mutation,
    disabledOperators: switches.operators,
    disabledFamilies: switches.families,
    preferredFamilies: switches.preferred,
    patience,
    relaxedScope: relaxed,
    run: false,
  }
  const valid = !agnostic && problem === null && options.ready && neutralization !== ''

  const plan = useQuery({
    queryKey: ['super-lab', 'advanced', 'plan', body],
    queryFn: () => http.post<ResearchPlan>(`${B}/plan`, body),
    enabled: valid,
    placeholderData: keepPreviousData,
  })
  const add = useAddTask(() => http.post<Schemas['AddedTask']>(`${B}/tasks`, body))

  return (
    <Page>
      <PageHeader
        sticky
        title="Super Alpha Lab"
        description="Writes, simulates and breeds SuperAlphas from the documented operator library"
        actions={
          <AddTaskButtons
            add={add}
            disabled={!valid || plan.data === undefined}
            blocked={problem}
          />
        }
      />
      {switcher}

      <Panel
        title="Market"
        description="Only your submitted ACTIVE Alphas of this region and delay can be selected. Nothing here changes your region, delay or universe."
      >
        <div className="flex flex-col gap-4">
          <ScopePicker scope={scope} onChange={setScope} />
          {agnostic && (
            <Notice
              tone="error"
              title="A SuperAlpha combines Alphas of one region. Pick a region other than All Regions."
            />
          )}
          <Field label="Neutralization" className="max-w-xs">
            <Select
              label="Neutralization"
              items={options.neutralizations}
              value={neutralization}
              onChange={setPicked}
            />
          </Field>
        </div>
      </Panel>

      <Panel
        title="Research Configuration"
        description="Each simulation is a Full-mode SuperAlpha with Selection Handling Positive. The task writes a generation at a time and learns from BRAIN's results; it never submits anything."
      >
        <div className="flex flex-col gap-5">
          <div className="flex flex-wrap items-end gap-6">
            <Field
              label="Selection Limit"
              hint={`Alphas kept from each ranking, ${SUPER_ALPHA_MIN_SELECTION} to ${MAX_LIMIT}.`}
              className="w-44"
            >
              <NumberInput value={limit} onChange={setLimit} />
            </Field>
            <Fieldset legend="Component Activation">
              <Segmented
                label="Component Activation"
                items={ACTIVATIONS}
                value={activation}
                onChange={setActivation}
              />
            </Fieldset>
            <Field label="Decay" hint="Whole numbers, 0 to 512, comma-separated." className="w-44">
              <Input
                value={decayText}
                onChange={(e) => setDecayText(e.target.value)}
                className="num"
              />
            </Field>
            <Field label="Truncation" hint="0 to 1, comma-separated." className="w-44">
              <Input
                value={truncText}
                onChange={(e) => setTruncText(e.target.value)}
                className="num"
              />
            </Field>
          </div>

          <div className="flex flex-wrap items-end gap-6">
            <Field
              label="Simulation Budget"
              hint="The most SuperAlpha simulations the task may spend."
              className="w-48"
            >
              <NumberInput value={simulations} onChange={setSimulations} />
            </Field>
            <Fieldset legend="Concurrency">
              <Segmented label="Concurrency" items={CORES} value={cores} onChange={setCores} />
            </Fieldset>
            <Field
              label="Early-Stop Patience"
              hint="Generations without a new best before it stops."
              className="w-48"
            >
              <NumberInput value={patience} onChange={setPatience} />
            </Field>
          </div>

          <Fieldset legend="Candidate Generation">
            <Segmented
              label="Candidate Generation"
              items={STRATEGIES}
              value={strategy}
              onChange={setStrategy}
            />
            <p className="mt-1 text-body-compact text-ink-subtle">{STRATEGY_HINT[strategy]}</p>
          </Fieldset>

          <Fieldset legend="Research Objective">
            <Segmented
              label="Research Objective"
              items={OBJECTIVES}
              value={objective}
              onChange={setObjective}
            />
            <p className="mt-1 text-body-compact text-ink-subtle">{OBJECTIVE_HINT[objective]}</p>
          </Fieldset>

          <Disclosure summary="Expression Complexity and Search Controls">
            <div className="flex flex-wrap items-end gap-6 pt-3">
              <Field label="Maximum Depth" hint="1 to 8." className="w-40">
                <NumberInput value={maxDepth} onChange={setMaxDepth} />
              </Field>
              <Field label="Maximum Operators" hint="3 to 30 per expression." className="w-44">
                <NumberInput value={maxOperators} onChange={setMaxOperators} />
              </Field>
              <Field label="Intermediate Variables" hint="0 to 6." className="w-44">
                <NumberInput value={maxVariables} onChange={setMaxVariables} />
              </Field>
              <Field
                label="Random Seed"
                hint="The same seed and results rebuild the same candidates."
                className="w-48"
              >
                <NumberInput value={seed} onChange={setSeed} />
              </Field>
              <Field
                label={`Exploration ${fmt.pct(exploration, 0)}`}
                hint="Low follows what scored; high favours rarely used operators."
                className="w-56"
              >
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={exploration}
                  aria-label="Exploration"
                  onChange={(e) => setExploration(e.target.valueAsNumber)}
                />
              </Field>
              <Field
                label={`Mutation ${fmt.pct(mutation, 0)}`}
                hint="How often a strong candidate is changed rather than replaced."
                className="w-56"
              >
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={mutation}
                  aria-label="Mutation probability"
                  onChange={(e) => setMutation(e.target.valueAsNumber)}
                />
              </Field>
            </div>
            <div className="flex items-center gap-3 pt-4">
              <Switch
                checked={relaxed}
                onChange={setRelaxed}
                aria-label="Accept operators by arity alone"
              />
              <span className="text-body-compact text-ink-muted">
                Verify operators BRAIN lists without a Combo or Selection scope by their argument
                count alone. Less strict: leave off unless the Operator Explorer shows too few
                verified.
              </span>
            </div>
          </Disclosure>
          {problem && <Notice tone="warn" title={problem} />}
        </div>
      </Panel>

      <PlanPanel plan={plan} />
      <OperatorExplorer relaxed={relaxed} />
      <ResearchMonitor />
    </Page>
  )
}

function NumberInput({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  return (
    <Input
      type="number"
      value={Number.isNaN(value) ? '' : String(value)}
      onChange={(e) => onChange(e.target.valueAsNumber)}
      className="num"
    />
  )
}

function PlanPanel({ plan }: { plan: ReturnType<typeof useQuery<ResearchPlan>> }) {
  return (
    <Panel
      title="What Will Run"
      description="Checked locally against the operator library. BRAIN stays the authority: a candidate that passes here can still be refused there."
    >
      {plan.isError && <ErrorNotice error={plan.error} title="Could not plan the task" />}
      {plan.data && (
        <div className="flex flex-col gap-4">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Metric boxed label="Simulation Budget" value={fmt.int(plan.data.simulations)} />
            <Metric boxed label="At Once" value={String(plan.data.concurrent)} />
            <Metric
              boxed
              label="Operators Usable"
              value={`${fmt.int(plan.data.operatorSummary.eligible)} of ${fmt.int(plan.data.operatorSummary.registered)}`}
            />
            <Metric
              boxed
              label="Turnover Range"
              value={`${fmt.pct(plan.data.turnoverRange[0], 0)}–${fmt.pct(plan.data.turnoverRange[1], 0)}`}
            />
          </div>
          <p className="text-body-compact text-ink-muted">
            {fmt.int(plan.data.comboTemplates.length)} combo and{' '}
            {fmt.int(plan.data.selectionTemplates.length)} selection templates are usable now.
            {plan.data.unavailableTemplates.length > 0 &&
              ` ${fmt.int(plan.data.unavailableTemplates.length)} wait on operators still pending verification.`}
          </p>
          {plan.data.unavailableTemplates.length > 0 && (
            <Disclosure summary="Templates waiting on operators">
              <ul className="flex flex-col gap-1 pt-2 text-body-compact text-ink-muted">
                {plan.data.unavailableTemplates.map((t) => (
                  <li key={t.key}>
                    <code className="num text-ink">{t.key}</code> needs{' '}
                    {t.needs.length > 0
                      ? t.needs.join(', ')
                      : 'a selection property not yet verified'}
                  </li>
                ))}
              </ul>
            </Disclosure>
          )}
          {plan.data.warnings.map((w) => (
            <Notice key={w} tone="info" title={w} />
          ))}
        </div>
      )}
    </Panel>
  )
}

type Show = 'all' | 'eligible' | 'blocked'

function OperatorExplorer({ relaxed }: { relaxed: boolean }) {
  const switches = useSwitches()
  const [search, setSearch] = useState('')
  const [show, setShow] = useState<Show>('all')
  const [context, setContext] = useState<'any' | 'selection' | 'combo'>('any')
  const catalog = useQuery({
    queryKey: [
      'super-lab',
      'advanced',
      'operators',
      relaxed,
      switches.operators,
      switches.families,
    ],
    queryFn: () =>
      http.get<OperatorCatalog>(
        `${B}/operators${qs({
          relaxedScope: relaxed,
          disabled: switches.operators.join(','),
          familiesOff: switches.families.join(','),
        })}`,
      ),
    placeholderData: keepPreviousData,
  })
  const rows = useMemo(() => {
    const needle = search.trim().toLowerCase()
    return (catalog.data?.operators ?? []).filter(
      (r) =>
        (show === 'all' || (show === 'eligible') === r.eligible) &&
        (context === 'any' || r.contexts.includes(context)) &&
        (needle === '' ||
          r.name.includes(needle) ||
          r.family.includes(needle) ||
          r.description.toLowerCase().includes(needle)),
    )
  }, [catalog.data, search, show, context])

  return (
    <Panel
      title="Operator Explorer"
      description="Every documented SuperAlpha operator, where it can run, and whether the generator can use it. Untick an operator or a family to leave it out of this task."
    >
      {catalog.isError && (
        <ErrorNotice error={catalog.error} title="Could not load the operators" />
      )}
      {catalog.data && (
        <div className="flex flex-col gap-4">
          <CoverageMetrics summary={catalog.data.summary} />
          {catalog.data.note && <Notice tone="warn" title={catalog.data.note} />}
          <Chips
            label="Operator families"
            items={catalog.data.families.map((f) => ({ value: f, label: words(f) }))}
            value={catalog.data.families.filter((f) => !switches.families.includes(f))}
            onChange={(on) =>
              useSwitches.setState({
                families: catalog.data.families.filter((f) => !on.includes(f)),
              })
            }
          />
          <div className="flex flex-wrap items-center gap-3">
            <Input
              aria-label="Search operators"
              placeholder="Search by name, family or description"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="max-w-sm"
            />
            <Segmented
              label="Show"
              items={[
                { value: 'all', label: 'All' },
                { value: 'eligible', label: 'Usable' },
                { value: 'blocked', label: 'Blocked' },
              ]}
              value={show}
              onChange={setShow}
            />
            <Segmented
              label="Context"
              items={[
                { value: 'any', label: 'Any' },
                { value: 'selection', label: 'Selection' },
                { value: 'combo', label: 'Combo' },
              ]}
              value={context}
              onChange={setContext}
            />
            <span className="num text-body-compact text-ink-subtle">
              {fmt.int(rows.length)} of {fmt.int(catalog.data.operators.length)}
            </span>
          </div>
          <ul className="flex max-h-[32rem] flex-col gap-1 overflow-y-auto">
            {rows.map((r) => (
              <OperatorItem key={r.name} row={r} />
            ))}
          </ul>
        </div>
      )}
    </Panel>
  )
}

export function CoverageMetrics({ summary }: { summary: CoverageSummary }) {
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6">
      <Metric boxed size="sm" label="Registered" value={fmt.int(summary.registered)} />
      <Metric
        boxed
        size="sm"
        label="Signature Verified"
        value={fmt.int(summary.signatureVerified)}
      />
      <Metric boxed size="sm" label="Locally Validated" value={fmt.int(summary.locallyValidated)} />
      <Metric boxed size="sm" label="Generation-Eligible" value={fmt.int(summary.eligible)} />
      <Metric
        boxed
        size="sm"
        label="Pending Verification"
        value={fmt.int(summary.pendingVerification)}
      />
      <Metric boxed size="sm" label="Blocked" value={fmt.int(summary.blocked)} />
    </div>
  )
}

const STATUS_TONE = {
  verified: 'profit',
  pending_verification: 'warn',
  incompatible: 'loss',
} as const
const STATUS_LABEL = {
  verified: 'Verified',
  pending_verification: 'Pending verification',
  incompatible: 'Incompatible',
} as const

function OperatorItem({ row }: { row: OperatorRow }) {
  const off = useSwitches((s) => s.operators.includes(row.name))
  const familyOff = useSwitches((s) => s.families.includes(row.family))
  const canToggle = row.status === 'verified'
  return (
    <li
      className={cn(
        'flex flex-col gap-1 rounded-md border p-2',
        row.eligible ? 'border-hairline bg-surface-2' : 'border-hairline bg-surface-1',
      )}
    >
      <div className="flex flex-wrap items-center gap-2">
        <Checkbox
          checked={canToggle && !off && !familyOff}
          disabled={!canToggle || familyOff}
          onChange={() => useSwitches.setState((s) => ({ operators: flip(s.operators, row.name) }))}
          aria-label={`Allow ${row.name}`}
          label={<code className="num font-medium text-ink">{row.name}</code>}
        />
        {row.contexts.map((c) => (
          <Badge key={c} tone="outline">
            {c}
          </Badge>
        ))}
        <Badge
          tone={STATUS_TONE[row.status]}
          {...(row.verifiedBy ? { title: row.verifiedBy } : {})}
        >
          {STATUS_LABEL[row.status]}
        </Badge>
        <Badge tone={row.eligible ? 'profit' : 'muted'}>
          {row.eligible ? 'Usable' : 'Not generated'}
        </Badge>
        <span className="text-body-compact text-ink-subtle">{words(row.family)}</span>
      </div>
      {!row.eligible && row.reason && (
        <p className="text-body-compact text-ink-muted">{row.reason}</p>
      )}
      <Disclosure
        summary={<span className="text-body-compact text-ink-muted">{row.description}</span>}
      >
        <div className="flex flex-col gap-1 pt-2 text-body-compact">
          <div className="text-ink-subtle">
            Signature{' '}
            <code className="num text-ink">{row.signature ?? 'none documented here'}</code>
          </div>
          {row.brainDefinition && (
            <div className="text-ink-subtle">
              BRAIN's definition <code className="num text-ink">{row.brainDefinition}</code>
            </div>
          )}
          {row.example && (
            <div className="text-ink-subtle">
              Example{' '}
              <code className="num break-all whitespace-pre-wrap text-ink">{row.example}</code>
            </div>
          )}
          {row.note && <div className="text-ink-muted">{row.note}</div>}
        </div>
      </Disclosure>
    </li>
  )
}
