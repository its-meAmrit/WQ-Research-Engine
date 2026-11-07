/**
 * SPC Research Lab: prompts for WorldQuant's Systematic Prediction Challenge. Nothing here
 * reaches BRAIN; a run spends LLM (and optionally search) requests, never a simulation slot.
 * Every body is camelCase.
 */

import type { components } from '@/api/generated'
import { http, qs } from '@/api/http'

type Schemas = components['schemas']

export type SpcOptions = Schemas['SpcOptions']
export type PromptOut = Schemas['PromptOut']
export type PromptDetail = Schemas['PromptDetail']
export type PromptVersion = Schemas['PromptVersionOut']
export type PromptCreate = Schemas['PromptCreate']
export type PromptPatch = Schemas['PromptPatch']
export type GenerationOptions = Schemas['GenerationOptions']
export type PromptCandidate = Schemas['PromptCandidate']
export type RunBrief = Schemas['RunBrief']
export type RunDetail = Schemas['RunDetail']
export type RunRequest = Schemas['RunRequest']
export type Position = Schemas['PositionOut']
export type Evidence = Schemas['EvidenceOut']
export type VersionComparison = Schemas['VersionComparison']
export type ExportOut = Schemas['ExportOut']
export type Analytics = Schemas['Analytics']
export type Overlap = Schemas['Overlap']

/** What the validator reports for one thing wrong, and where. */
export interface Issue {
  severity: 'error' | 'warning'
  code: string
  message: string
  index: number | null
  field: string | null
  suggestion: string | null
}

export interface Validation {
  valid: boolean
  itemCount: number
  schemaVerified: boolean
  errorCount: number
  warningCount: number
  issues: Issue[]
}

/** `validation` arrives as an open object; this is the shape the backend writes into it. */
export const validationOf = (run: Pick<RunDetail, 'validation'>): Validation | null =>
  (run.validation as Validation | null) ?? null

export const ACTIVE = new Set(['queued', 'researching', 'generating', 'validating'])
export const isActive = (status: string) => ACTIVE.has(status)

export const spc = {
  options: () => http.get<SpcOptions>('/api/spc/options'),
  generate: (body: Partial<GenerationOptions> & { thesis: string }) =>
    http.post<{ candidates: PromptCandidate[] }>('/api/spc/generate', body),

  prompts: (archived = false) => http.get<PromptOut[]>(`/api/spc/prompts${qs({ archived })}`),
  prompt: (id: number) => http.get<PromptDetail>(`/api/spc/prompts/${id}`),
  create: (body: Partial<PromptCreate> & { name: string; promptText: string }) =>
    http.post<PromptDetail>('/api/spc/prompts', body),
  update: (id: number, body: Partial<PromptPatch>) =>
    http.patch<PromptDetail>(`/api/spc/prompts/${id}`, body),
  duplicate: (id: number) => http.post<PromptDetail>(`/api/spc/prompts/${id}/duplicate`),
  restore: (id: number, version: number) =>
    http.post<PromptDetail>(`/api/spc/prompts/${id}/versions/${version}/restore`),
  compare: (id: number, a: number, b: number) =>
    http.get<VersionComparison>(`/api/spc/prompts/${id}/compare${qs({ a, b })}`),

  test: (id: number, body: Partial<RunRequest>) =>
    http.post<RunDetail>(`/api/spc/prompts/${id}/test`, body),
  runs: (promptId?: number) => http.get<RunBrief[]>(`/api/spc/runs${qs({ promptId })}`),
  run: (id: number) => http.get<RunDetail>(`/api/spc/runs/${id}`),
  cancel: (id: number) => http.post<RunDetail>(`/api/spc/runs/${id}/cancel`),
  correct: (id: number, body: Partial<RunRequest>) =>
    http.post<RunDetail>(`/api/spc/runs/${id}/correct`, body),
  analytics: (id: number) => http.get<Analytics>(`/api/spc/runs/${id}/analytics`),
  overlap: (runIds: number[]) => http.post<Overlap>('/api/spc/overlap', { runIds }),
  export: (promptId: number, runId?: number) =>
    http.get<ExportOut>(`/api/spc/prompts/${promptId}/export${qs({ runId })}`),
}
