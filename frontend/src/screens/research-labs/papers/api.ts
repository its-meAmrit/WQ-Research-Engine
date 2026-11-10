/**
 * Paper Analyzer: the Research Intelligence API. Responses are camelCase; the free-form JSON
 * the backend stores whole (an assessment, the claims, the evidence) is narrowed here.
 */

import type { components } from '@/api/generated'
import { ApiError, http, normalise, qs } from '@/api/http'

type Schemas = components['schemas']

export type Paper = Schemas['SourceOut']
export type PaperDetail = Schemas['SourceDetail']
export type Job = Schemas['JobOut']
export type Started = Schemas['StartedOut']
export type SearchResults = Schemas['SearchOut']
export type SearchPaper = Schemas['SearchPaper']
export type Preview = Schemas['PreviewOut']
export type Options = Schemas['ResearchOptions']
export type PageText = Schemas['PageOut']
export type Candidate = Schemas['CandidateOut']
export type Readiness = 'ready' | 'adaptation' | 'blocked'
export type Support = 'source_backed' | 'partially_supported' | 'speculative'
export type IdeaStatus = 'new' | 'saved' | 'rejected'
export type Access = 'full_text' | 'abstract_only' | 'metadata_only'

export interface Evidence {
  sectionId?: string | null
  section_id?: string | null
  page: number
  quote: string
  supports: string
  verified: boolean
  location: string
}

export interface Claims {
  authorClaims: string[]
  modelInterpretations: string[]
  proposedHypotheses: string[]
  operatorsNeeded: string[]
  marketsStudied: string[]
}

export interface RequiredVariable {
  name: string
  description: string
  kind: string
  essential: boolean
  search_terms: string[]
  point_in_time_risk: string
}

export interface FieldCandidate {
  fieldId: string
  description: string
  fieldType: string | null
  datasetId: string | null
  coverage: number | null
  alphaCount: number | null
  pointInTime: string
}

export interface VariableCheck {
  name: string
  kind: string
  essential: boolean
  searchTerms: string[]
  candidates: FieldCandidate[]
  availableElsewhere: { fieldId: string; region: string; delay: number; universe: string }[]
  pointInTimeRisk: string
  status: 'candidates' | 'missing' | 'other_scope' | 'unsupported'
}

export interface Assessment {
  readiness: Readiness
  scope: { instrumentType: string; region: string; delay: number; universe: string }
  scopeHasData: boolean
  variables: VariableCheck[]
  operators: {
    name: string
    status: 'supported' | 'unsupported' | 'unverified'
    category?: string
  }[]
  reasons: { code: string; severity: 'info' | 'adapt' | 'block'; message: string }[]
  marketsStudied: string[]
  assessedAt: string
  note: string
}

export type IndicatorLevel = 'high' | 'medium' | 'low' | 'unknown'

/** One of the five ranking indicators. Each stands alone: there is no combined score, and none
 *  of them estimates a Sharpe ratio or a return. */
export interface Indicator {
  level: IndicatorLevel
  reasons: string[]
  basis: string
}

export interface Ranking {
  sourceQuality: Indicator
  specificity: Indicator
  brainFeasibility: Indicator
  novelty: Indicator
  evidenceQuality: Indicator
}

export type Idea = Omit<
  Schemas['IdeaOut'],
  'claims' | 'evidence' | 'brainAssessment' | 'requiredVariables' | 'ranking'
> & {
  claims: Partial<Claims> & { reportedPerformance?: string[]; reportedPerformanceNote?: string }
  evidence: Evidence[]
  brainAssessment: Assessment | null
  requiredVariables: RequiredVariable[]
  ranking: Ranking | null
}
export type LinkedSource = Schemas['LinkedSource']
export type RelatedIdea = Schemas['RelatedIdea']
export type IdeaDetail = Idea & {
  candidates: Candidate[]
  linkedSources: LinkedSource[]
  related: RelatedIdea[]
}

export type RegistryEntry = Schemas['RegistryEntryOut']
export type DiscoverOptions = Schemas['DiscoverOptions']
export type DiscoverBody = Schemas['DiscoverRequest']
export type Estimate = Schemas['EstimateOut']
export type Hit = Schemas['HitOut']
export type QueryProposal = Schemas['QueryProposal']
export type DiscoverStarted = Schemas['DiscoverStarted']
export type ExtractStarted = Schemas['ExtractStarted']

export interface PaperFilters {
  search?: string
  access?: string
  source?: string
  topic?: string
  readiness?: string
  saved?: boolean
  publishedFrom?: string
}

export interface IdeaFilters {
  sourceId?: string
  status?: string
  readiness?: string
  topic?: string
  support?: string
  search?: string
  domain?: string
  sourceClass?: string
  sourceType?: string
  kind?: string
  favorite?: boolean
  reviewed?: boolean
}

export interface ScopeBody {
  instrumentType: string
  region: string
  delay: number
  universe: string
}

const B = '/api/research'

/** A PDF goes as the raw body: the backend takes bytes, not a multipart form. */
async function uploadPdf(file: File): Promise<Started> {
  let response: Response
  try {
    response = await fetch(`${B}/papers/upload${qs({ filename: file.name })}`, {
      method: 'POST',
      headers: { 'X-Harness-Client': '1', 'Content-Type': 'application/pdf' },
      body: file,
    })
  } catch (failure) {
    if (!(failure instanceof TypeError)) throw failure
    throw new ApiError(0, {
      code: 'backend_unreachable',
      message: 'Cannot reach the Alpha Harness backend.',
    })
  }
  const raw = await response.text()
  let parsed: unknown = raw
  try {
    parsed = JSON.parse(raw)
  } catch {
    // Plain text error, kept as it is.
  }
  if (!response.ok) throw new ApiError(response.status, normalise(response.status, parsed))
  return parsed as Started
}

export const research = {
  options: () => http.get<Options>(`${B}/options`),
  papers: (filters: PaperFilters) => http.get<Paper[]>(`${B}/papers${qs({ ...filters })}`),
  paper: (id: string) => http.get<PaperDetail>(`${B}/papers/${id}`),
  deletePaper: (id: string) => http.del<void>(`${B}/papers/${id}`),
  page: (id: string, page: number) => http.get<PageText>(`${B}/papers/${id}/pages/${page}`),
  upload: uploadPdf,
  importReference: (reference: string) => http.post<Started>(`${B}/papers/import`, { reference }),
  search: (q: string, semantic: boolean, web: boolean, signal?: AbortSignal) =>
    http.get<SearchResults>(`${B}/search${qs({ q, semantic, web })}`, signal),
  importResult: (body: { doi?: string | null; arxivId?: string | null; url?: string | null }) =>
    http.post<Started>(`${B}/search/import`, body),
  preview: (id: string, charBudget?: number) =>
    http.get<Preview>(`${B}/papers/${id}/preview${qs({ charBudget })}`),
  analyze: (id: string, body: { modelRef?: string | null; charBudget?: number; force?: boolean }) =>
    http.post<Started>(`${B}/papers/${id}/analyze`, body),
  jobs: (sourceId?: string) => http.get<Job[]>(`${B}/jobs${qs({ sourceId })}`),
  job: (id: string) => http.get<Job>(`${B}/jobs/${id}`),
  cancelJob: (id: string) => http.post<Job>(`${B}/jobs/${id}/cancel`),
  ideas: (filters: IdeaFilters) => http.get<Idea[]>(`${B}/ideas${qs({ ...filters })}`),
  idea: (id: string) => http.get<IdeaDetail>(`${B}/ideas/${id}`),
  patchIdea: (
    id: string,
    body: { status?: IdeaStatus; note?: string; favorite?: boolean; reviewed?: boolean },
  ) => http.patch<IdeaDetail>(`${B}/ideas/${id}`, body),
  assess: (id: string, scope: ScopeBody) =>
    http.post<IdeaDetail>(`${B}/ideas/${id}/assess`, { scope }),
  handOff: (id: string, scope: ScopeBody) =>
    http.post<Candidate>(`${B}/ideas/${id}/handoff`, { scope }),
}

const D = `${B}/discover`

/** Discover: find public research, read what you pick, save its ideas. Searching reads results
 *  only; reading a page is a separate request. Jobs are followed through `research.job`. */
export const discover = {
  options: () => http.get<DiscoverOptions>(`${D}/options`),
  queries: (topic: string, market: string) =>
    http.post<QueryProposal[]>(`${D}/queries`, { topic, market }),
  estimate: (body: DiscoverBody, signal?: AbortSignal) =>
    http.post<Estimate>(`${D}/estimate`, body, signal),
  start: (body: DiscoverBody) => http.post<DiscoverStarted>(D, body),
  hits: (jobId: string) => http.get<Hit[]>(`${D}/jobs/${jobId}/hits`),
  extract: (body: {
    hitIds: string[]
    modelRef?: string | null
    scope?: ScopeBody | null
    depth?: 'basic' | 'advanced'
  }) => http.post<ExtractStarted>(`${D}/extract`, body),
  updateSource: (
    id: string,
    body: { enabled?: boolean; minIntervalSeconds?: number; domains?: string[] },
  ) => http.patch<RegistryEntry>(`${D}/sources/${id}`, body),
  addFeed: (label: string, feedUrl: string) =>
    http.post<RegistryEntry>(`${D}/sources`, { label, feedUrl }),
  removeSource: (id: string) => http.del<void>(`${D}/sources/${id}`),
}
