/**
 * Web Research: Tavily and Brave Search. Not language models, so nothing here touches Models or
 * Keys. The backend holds every key sealed and sends back only a masked hint; this screen can
 * add, replace and remove one, turn it on or off, test it, cap it and choose the default.
 *
 * A test may spend a paid credit, so it always asks first and says what it costs.
 */

import { useMutation } from '@tanstack/react-query'
import { type FormEvent, useState } from 'react'
import { toast } from 'sonner'
import { DASH, fmt } from '@/lib/format'
import { type WebCheck, type WebProvider, type WebResearchStatus, web } from '@/screens/ai/api'
import {
  Badge,
  Button,
  Empty,
  ErrorNotice,
  Field,
  Input,
  KV,
  LINK,
  Notice,
  Panel,
  Skeleton,
  Switch,
} from '@/ui/kit'
import { Confirm } from '@/ui/overlay'
import { useInvalidateWeb, useWebResearch } from './shared'

const CAPABILITY: Record<string, string> = { search: 'Search', extract: 'Page extraction' }

export function WebResearch() {
  const status = useWebResearch()

  if (status.isError)
    return <ErrorNotice title="Could not load web research" error={status.error} />
  if (!status.data) return <Skeleton className="h-64" />
  const { data } = status
  const configured = data.providers.filter((p) => p.configured)

  return (
    <div className="flex min-w-0 flex-col gap-4">
      <ActiveNotice data={data} />
      {configured.length === 0 && (
        <Panel>
          <Empty title="No web research service has a key yet">
            Add a Tavily or Brave Search key below. SPC runs use it to look up current sources
            before the model answers. It is optional: without one, runs rely on the model's own
            knowledge.
          </Empty>
        </Panel>
      )}
      <div className="grid min-w-0 gap-4 lg:grid-cols-2">
        {data.providers.map((p) => (
          <ProviderCard key={p.id} provider={p} canPrefer={configured.length > 1} />
        ))}
      </div>
    </div>
  )
}

function ActiveNotice({ data }: { data: WebResearchStatus }) {
  const { active, legacyEnvironment: env } = data
  if (active.configured) {
    return (
      <Notice
        tone="info"
        title={`SPC research will use ${active.label} ${active.source === 'environment' ? '(from an environment variable)' : ''}`.trim()}
      >
        {active.message ??
          'Searches use the key stored here, counted against the daily budget you set.'}
      </Notice>
    )
  }
  return (
    <Notice tone="warn" title="Web research is not set up">
      {active.help}
      {env.present && !env.usable && env.note ? ` ${env.note}` : ''}
    </Notice>
  )
}

function StatusBadge({ p }: { p: WebProvider }) {
  if (!p.configured) return <Badge tone="muted">No key</Badge>
  if (!p.enabled) return <Badge tone="muted">Disabled</Badge>
  if (p.lastError)
    return (
      <Badge tone="loss" title={p.lastError}>
        Failing
      </Badge>
    )
  if (p.lastOkAt) return <Badge tone="profit">Working · {fmt.ago(p.lastOkAt)}</Badge>
  return <Badge tone="muted">Not tested</Badge>
}

function ProviderCard({ provider: p, canPrefer }: { provider: WebProvider; canPrefer: boolean }) {
  const invalidate = useInvalidateWeb()
  const [replacing, setReplacing] = useState(false)
  const [confirmTest, setConfirmTest] = useState(false)
  const [confirmRemove, setConfirmRemove] = useState(false)
  const [result, setResult] = useState<WebCheck | null>(null)

  const toggle = useMutation({
    mutationFn: (enabled: boolean) => web.update(p.id, { enabled }),
    onSuccess: (next) => {
      toast.success(`${p.label} ${next.enabled ? 'enabled' : 'disabled'}`)
      invalidate()
    },
  })
  const prefer = useMutation({
    mutationFn: (id: WebProvider['id'] | null) => web.setPreferred(id),
    onSuccess: (_, id) => {
      toast.success(id ? `${p.label} is now the default` : 'Default cleared')
      invalidate()
    },
  })
  const remove = useMutation({
    mutationFn: () => web.removeKey(p.id),
    onSuccess: () => {
      toast.success(`Removed the ${p.label} key`)
      setConfirmRemove(false)
      setResult(null)
      invalidate()
    },
  })
  const test = useMutation({
    mutationFn: () => web.check(p.id, true),
    // Shown inline as well as toasted: a toast is gone before it is read.
    onSuccess: (r) => {
      setResult(r)
      if (r.ok) toast.success(`${p.label} accepted the key`)
      else toast.error(`${p.label}: ${r.message}`)
      invalidate()
    },
    onSettled: () => setConfirmTest(false),
  })

  return (
    <Panel
      title={p.label}
      description={p.description}
      actions={
        <div className="flex flex-wrap items-center gap-1.5">
          {p.isPreferred && <Badge tone="profit">Default</Badge>}
          <StatusBadge p={p} />
        </div>
      }
    >
      <div className="flex min-w-0 flex-col gap-3">
        <div className="flex flex-wrap gap-1.5">
          {p.capabilities.map((c) => (
            <Badge key={c} tone="outline">
              {CAPABILITY[c] ?? c}
            </Badge>
          ))}
        </div>

        {!p.configured || replacing ? (
          <KeyForm
            provider={p}
            replacing={replacing}
            onDone={() => {
              setReplacing(false)
              setResult(null)
            }}
          />
        ) : (
          <>
            <KV
              items={[
                [
                  'Key',
                  <span key="hint" className="num text-ink-muted">
                    {p.hint}
                  </span>,
                ],
                [
                  'Enabled',
                  <EnabledSwitch
                    key="enabled"
                    p={p}
                    pending={toggle.isPending}
                    onChange={toggle.mutate}
                  />,
                ],
                ['Last successful check', p.lastOkAt ? fmt.ago(p.lastOkAt) : DASH],
              ]}
            />
            <UsageBlock p={p} />
            {p.lastError && !result && (
              <Notice tone="error" title="The last request failed">
                {p.lastError}
              </Notice>
            )}
            {result && <CheckResult result={result} />}
            <div className="flex flex-wrap gap-2">
              <Button
                size="sm"
                disabled={!p.enabled}
                loading={test.isPending}
                onClick={() => setConfirmTest(true)}
              >
                Test connection
              </Button>
              {canPrefer && (
                <Button
                  size="sm"
                  disabled={!p.enabled || prefer.isPending}
                  onClick={() => prefer.mutate(p.isPreferred ? null : p.id)}
                >
                  {p.isPreferred ? 'Clear default' : 'Make default'}
                </Button>
              )}
              <Button size="sm" variant="ghost" onClick={() => setReplacing(true)}>
                Replace key
              </Button>
              <Button size="sm" variant="danger" onClick={() => setConfirmRemove(true)}>
                Remove key
              </Button>
            </div>
            <BudgetEditor p={p} />
          </>
        )}

        <a
          href={p.keyUrl}
          target="_blank"
          rel="noopener noreferrer"
          className={`${LINK} text-body-compact`}
        >
          Get a {p.label} key
        </a>
      </div>

      <Confirm
        open={confirmTest}
        onOpenChange={setConfirmTest}
        title={`Test ${p.label}?`}
        confirmLabel={p.checkCost.mayConsumeCredit ? 'Test (may use a credit)' : 'Test'}
        pending={test.isPending}
        onConfirm={() => test.mutate()}
      >
        {p.checkCost.description} It also counts as one request against your daily budget here.
      </Confirm>
      <Confirm
        open={confirmRemove}
        onOpenChange={setConfirmRemove}
        title={`Remove the ${p.label} key?`}
        confirmLabel="Remove key"
        danger
        pending={remove.isPending}
        onConfirm={() => remove.mutate()}
      >
        The key is deleted from the local vault together with its usage count
        {p.isPreferred ? ', and it stops being the default' : ''}. Saved SPC runs and their evidence
        are kept. You can add the key again later.
      </Confirm>
    </Panel>
  )
}

function EnabledSwitch({
  p,
  pending,
  onChange,
}: {
  p: WebProvider
  pending: boolean
  onChange: (enabled: boolean) => void
}) {
  return (
    <Switch
      aria-label={`Enable ${p.label}`}
      checked={p.enabled}
      disabled={pending}
      onChange={onChange}
    />
  )
}

/** The paste box. The key lives in this component's state only until it is sent; it is never
 *  written to storage, and the field is cleared the moment it is saved. */
function KeyForm({
  provider: p,
  replacing,
  onDone,
}: {
  provider: WebProvider
  replacing: boolean
  onDone: () => void
}) {
  const invalidate = useInvalidateWeb()
  const [key, setKey] = useState('')
  const save = useMutation({
    mutationFn: () => web.putKey(p.id, key),
    meta: { inline: true },
    onSuccess: () => {
      toast.success(
        replacing
          ? `${p.label} key replaced. Test it to check it works.`
          : `${p.label} key saved. Test it to check it works.`,
      )
      setKey('')
      onDone()
      invalidate()
    },
  })
  const submit = (e: FormEvent) => {
    e.preventDefault()
    if (key.trim()) save.mutate()
  }
  return (
    <form onSubmit={submit} className="flex min-w-0 flex-col gap-2">
      <Field
        label={replacing ? `New ${p.label} API key` : `${p.label} API key`}
        hint="Stored encrypted on this computer and never shown again. Saving does not contact the service."
      >
        <Input
          type="password"
          autoComplete="off"
          spellCheck={false}
          placeholder="Paste the key"
          value={key}
          onChange={(e) => setKey(e.target.value)}
          aria-invalid={save.isError}
        />
      </Field>
      {save.isError && <ErrorNotice title="The key was not saved" error={save.error} />}
      <div className="flex gap-2">
        <Button
          variant="primary"
          size="sm"
          type="submit"
          loading={save.isPending}
          disabled={!key.trim()}
        >
          {replacing ? 'Replace key' : 'Save key'}
        </Button>
        {replacing && (
          <Button size="sm" variant="ghost" onClick={onDone}>
            Cancel
          </Button>
        )}
      </div>
    </form>
  )
}

function UsageBlock({ p }: { p: WebProvider }) {
  const u = p.usage
  const r = p.reportedUsage
  if (!u) return null
  return (
    <div className="flex min-w-0 flex-col gap-1 rounded-sm border border-hairline bg-surface-2 px-3 py-2 text-body-compact">
      <p className="text-ink">
        Sent from here today: <span className="num">{fmt.int(u.requests)}</span> of{' '}
        <span className="num">{fmt.int(u.dailyBudget)}</span> requests
        <span className="text-ink-subtle"> · resets in {fmt.duration(u.resetsInSeconds)}</span>
      </p>
      <p className="text-ink-subtle">
        A local count of this app's requests, not {p.label}'s balance.
      </p>
      {r && (
        <p className="text-ink-muted">
          {p.label} reported{' '}
          {r.used != null && r.limit != null
            ? `${fmt.int(r.used)} of ${fmt.int(r.limit)} used`
            : 'usage'}
          {r.plan ? ` on ${r.plan}` : ''}
          {r.period ? ` (${r.period})` : ''}, as of {fmt.ago(r.checkedAt)}.
        </p>
      )}
    </div>
  )
}

function BudgetEditor({ p }: { p: WebProvider }) {
  const invalidate = useInvalidateWeb()
  const current = p.usage?.dailyBudget ?? 100
  const [draft, setDraft] = useState(String(current))
  const [seen, setSeen] = useState(current)
  if (seen !== current) {
    setSeen(current)
    setDraft(String(current))
  }
  const value = Number(draft)
  const valid = Number.isInteger(value) && value >= 1 && value <= 100_000
  const save = useMutation({
    mutationFn: () => web.update(p.id, { dailyRequestBudget: value }),
    onSuccess: (next) => {
      toast.success(
        `${p.label} limited to ${fmt.int(next.usage?.dailyBudget ?? value)} requests a day`,
      )
      invalidate()
    },
  })
  return (
    <form
      className="flex flex-wrap items-end gap-2"
      onSubmit={(e) => {
        e.preventDefault()
        if (valid && value !== current) save.mutate()
      }}
    >
      <Field
        label="Daily request budget"
        hint="Your own ceiling on paid requests per day (UTC). SPC stops searching when it is spent."
        error={draft !== '' && !valid ? 'Enter a whole number from 1 to 100,000.' : undefined}
        className="w-56"
      >
        <Input
          inputMode="numeric"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          aria-invalid={draft !== '' && !valid}
        />
      </Field>
      <Button
        size="sm"
        type="submit"
        loading={save.isPending}
        disabled={!valid || value === current}
      >
        Save budget
      </Button>
    </form>
  )
}

function CheckResult({ result }: { result: WebCheck }) {
  if (result.ok)
    return (
      <Notice tone="info" title={result.detail}>
        {result.reported?.remaining != null
          ? `Reported remaining: ${fmt.int(result.reported.remaining)}.`
          : 'The service did not report its remaining allowance.'}
      </Notice>
    )
  return (
    <Notice tone="error" title="The test failed">
      {result.message}
      {result.retryAfter != null ? ` (retry in ${fmt.duration(result.retryAfter)})` : ''}
    </Notice>
  )
}
