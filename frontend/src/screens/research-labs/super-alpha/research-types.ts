/** Shapes of the Advanced Research responses that are not request bodies. The backend builds
 *  these as plain JSON in `api/super_research.py`; every field here is one it writes. */

export type Context = 'selection' | 'combo'
export type ValidationStatus = 'verified' | 'pending_verification' | 'incompatible'

export interface OperatorRow {
  name: string
  contexts: Context[]
  family: string
  description: string
  signature: string | null
  example: string | null
  status: ValidationStatus
  verifiedBy: string | null
  registered: boolean
  signatureVerified: boolean
  locallyValidated: boolean
  eligible: boolean
  reason: string
  included: number
  submitted: number
  accepted: number
  rejected: number
  successful: number
  failureReasons: Record<string, number>
  lastFailure: string | null
  brainDefinition: string | null
  note: string
}

export interface CoverageSummary {
  registered: number
  signatureVerified: number
  pendingVerification: number
  incompatible: number
  locallyValidated: number
  eligible: number
  blocked: number
  included: number
  submitted: number
  accepted: number
  rejected: number
  successful: number
}

export interface OperatorCatalog {
  operators: OperatorRow[]
  summary: CoverageSummary
  families: string[]
  verifiedAgainstBrain: boolean
  note: string | null
}

export interface ResearchPlan {
  simulations: number
  concurrent: number
  operatorSummary: CoverageSummary
  comboTemplates: string[]
  selectionTemplates: string[]
  unavailableTemplates: { key: string; needs: string[] }[]
  operatorCap: number
  objectives: string[]
  turnoverRange: [number, number]
  warnings: string[]
}

export interface Candidate {
  trialId: number
  number: number
  generation: number | null
  state: string
  outcome: 'accepted' | 'rejected' | 'submitted' | 'waiting'
  name: string | null
  family: string | null
  origin: string | null
  explanation: string | null
  selection: string
  combo: string
  operators: string[]
  parents: string[]
  baseline: boolean
  key: string | null
  settings: Record<string, unknown>
  local: Record<string, number>
  alphaId: string | null
  score: number | null
  scoreParts: Record<string, number>
  sharpe: number | null
  fitness: number | null
  turnover: number | null
  drawdown: number | null
  returns: number | null
  margin: number | null
  turnoverInRange: boolean | null
  failedChecks: string[]
  vsEqualWeight: number | null
  submittable: boolean
  rejection: string | null
  rejectionKind: string | null
}

export interface Dashboard {
  generated: number
  locallyValid: number
  simulated: number
  submitted: number
  failed: number
  waiting: number
  budget: number
  budgetUsed: number
  submittable: number
  best: Candidate[]
  history: { generation: number; candidates: number; scored: number; best: number | null }[]
  diversity: {
    families: Record<string, number>
    distinctCombos: number
    distinctSelections: number
    comboUniqueness: number
  }
  failureKinds: Record<string, number>
}

export interface TaskDetail {
  id: number
  name: string
  status: 'IDLE' | 'QUEUED' | 'RUNNING' | 'PAUSED' | 'COMPLETE' | 'FAILED'
  message: string | null
  params: Record<string, unknown>
  dashboard: Dashboard
  candidates: Candidate[]
  coverage: OperatorRow[]
  coverageSummary: CoverageSummary
  avoided: Record<string, string>
}

export interface Validation {
  selection?: ExpressionCheck
  combo?: ExpressionCheck
}

export interface ExpressionCheck {
  ok: boolean
  problems: string[]
  operators: string[]
  text: string
  operatorCount: number
  depth: number
  type: string | null
}
