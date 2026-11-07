/**
 * SPC Research Lab (WorldQuant Systematic Prediction Challenge). A prompt here is a text thesis
 * an LLM turns into a stock portfolio: it is not a Fast Expression, and nothing on these tabs
 * reaches BRAIN. Tabs: `/spc/$tab` with `?prompt=` and `?run=` in the URL.
 */

import { useParams } from '@tanstack/react-router'
import { SPC_TABS } from '@/shell/nav'
import {
  Empty,
  ErrorNotice,
  LINK,
  Notice,
  Page,
  PageHeader,
  Panel,
  TabBar,
  TabLink,
} from '@/ui/kit'
import { Analytics } from './analytics'
import { Export } from './export'
import { Runs } from './runs'
import { useOptions } from './shared'
import { Studio } from './studio'

const SCREENS = { studio: Studio, runs: Runs, analytics: Analytics, export: Export } as const

export function SpcScreen() {
  const { tab } = useParams({ strict: false })
  const options = useOptions()
  const Screen = tab && tab in SCREENS ? SCREENS[tab as keyof typeof SCREENS] : null
  return (
    <Page>
      <PageHeader
        title="SPC Research Lab"
        description="Write, test and export prompts for WorldQuant's Systematic Prediction Challenge. A prompt is an investment thesis an LLM turns into a global equity portfolio; nothing here runs a BRAIN simulation or submits to WorldQuant."
      />
      <TabBar>
        {SPC_TABS.map((t) => (
          <TabLink key={t.tab} to="/spc/$tab" params={{ tab: t.tab }} search={(prev) => prev}>
            {t.label}
          </TabLink>
        ))}
      </TabBar>
      {options.isError && (
        <ErrorNotice title="Could not load the lab's options" error={options.error} />
      )}
      {options.data && !options.data.schemaProfile['verified'] && (
        <Notice tone="warn" title="Output field names are not confirmed">
          {String(options.data.schemaProfile['note'])}
        </Notice>
      )}
      {Screen ? (
        <Screen />
      ) : (
        <Panel>
          <Empty title={`There is no “${tab}” tab`}>
            <TabLink to="/spc/$tab" params={{ tab: 'studio' }} className={LINK}>
              Go to Prompt Studio
            </TabLink>
          </Empty>
        </Panel>
      )}
    </Page>
  )
}
