/** Following one Alpha Builder job from start to finish, with retry and cancel. */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { type Job, research } from './api'

export interface Watched {
  job: Job | null
  /** A job is running or queued. */
  busy: boolean
  start: (begin: () => Promise<{ jobId: string }>) => void
  starting: boolean
  startError: unknown
  cancel: () => void
  clear: () => void
}

/**
 * Starts a job through ``begin`` and polls it until it ends. ``onDone`` runs once when it
 * completes, with the job's result; a failed or cancelled job is left on screen with its
 * reason, never hidden.
 */
export function useJobWatch(onDone: (job: Job) => void): Watched {
  const client = useQueryClient()
  const [jobId, setJobId] = useState<string | null>(null)
  const [reported, setReported] = useState<string | null>(null)
  const query = useQuery({
    queryKey: ['research', 'builder', 'job', jobId],
    queryFn: () => research.job(jobId as string),
    enabled: jobId != null,
    refetchInterval: (q) => (q.state.data?.active === false ? false : 800),
  })
  const job = query.data ?? null
  if (job && !job.active && job.status === 'completed' && reported !== job.id) {
    setReported(job.id)
    onDone(job)
  }
  const starter = useMutation({
    mutationFn: (begin: () => Promise<{ jobId: string }>) => begin(),
    onSuccess: (started) => {
      setJobId(started.jobId)
      void client.invalidateQueries({ queryKey: ['research', 'jobs'] })
    },
  })
  const canceller = useMutation({
    mutationFn: () => research.cancelJob(jobId as string),
    onSuccess: (next) => client.setQueryData(['research', 'builder', 'job', jobId], next),
  })
  return {
    job,
    busy: starter.isPending || (jobId != null && (job == null || job.active)),
    start: (begin) => starter.mutate(begin),
    starting: starter.isPending,
    startError: starter.error,
    cancel: () => canceller.mutate(),
    clear: () => {
      setJobId(null)
      starter.reset()
    },
  }
}
