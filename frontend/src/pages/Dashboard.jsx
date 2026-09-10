import React, { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { subscribeToJob } from '../lib/ws.js'
import { api } from '../lib/api.js'

const STAGE_ICONS = {
  pending:   '○',
  running:   '⟳',
  completed: '✓',
  failed:    '✗',
  skipped:   '–',
}

const STAGE_COLORS = {
  pending:   'var(--text-muted)',
  running:   'var(--indigo)',
  completed: 'var(--emerald)',
  failed:    'var(--red)',
  skipped:   'var(--text-muted)',
}

export default function Dashboard() {
  const { jobId } = useParams()
  const navigate = useNavigate()
  const [job, setJob] = useState(null)
  const [stages, setStages] = useState([])
  const [progress, setProgress] = useState(0)
  const [liveData, setLiveData] = useState({})
  const [error, setError] = useState('')

  useEffect(() => {
    // Load initial job state
    api.getJob(jobId).then((data) => {
      setJob(data)
      setStages(data.stages || [])
      setProgress(data.progress || 0)
      if (data.status === 'completed') {
        navigate(`/jobs/${jobId}/viewer`)
      }
    }).catch(() => setError('Job not found'))

    // Subscribe to WebSocket events
    const sub = subscribeToJob(jobId, (event) => {
      if (event.event === 'snapshot') {
        setStages(event.stages || [])
        setProgress(event.progress || 0)
        if (event.status === 'completed') navigate(`/jobs/${jobId}/viewer`)
        if (event.status === 'failed') setError('Reconstruction failed. See details below.')
        return
      }

      setProgress(event.progress || 0)

      if (event.event === 'stage_started' || event.event === 'stage_completed' || event.event === 'stage_failed') {
        setStages((prev) => prev.map((s) => {
          if (s.name !== event.stage) return s
          return {
            ...s,
            status: event.event === 'stage_started' ? 'running'
                  : event.event === 'stage_completed' ? 'completed'
                  : 'failed',
            message: event.message || s.message,
          }
        }))
        if (event.data && Object.keys(event.data).length) {
          setLiveData((prev) => ({ ...prev, ...event.data }))
        }
      }

      if (event.event === 'done') navigate(`/jobs/${jobId}/viewer`)
      if (event.event === 'error') setError(event.message)
    })

    return () => sub.close()
  }, [jobId, navigate])

  const runningStage = stages.find((s) => s.status === 'running')
  const completedCount = stages.filter((s) => s.status === 'completed').length

  return (
    <div className="dashboard-page">
      {/* Header */}
      <div style={{ marginBottom: 32 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
          <h2>Reconstruction Progress</h2>
          {job && <span className="badge badge-indigo">{job.method}</span>}
        </div>
        <p style={{ fontFeatureSettings: '"tnum"', fontSize: 12 }}>
          Job: {jobId}
        </p>
      </div>

      {/* Progress bar */}
      <div className="card" style={{ padding: '20px 24px', marginBottom: 24 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
          <span style={{ fontSize: 13, fontWeight: 600 }}>
            {runningStage ? runningStage.label : progress === 100 ? 'Complete' : 'Initialising…'}
          </span>
          <span style={{ fontSize: 13, color: 'var(--indigo-l)', fontWeight: 600 }}>
            {progress}%
          </span>
        </div>
        <div className="progress-bar">
          <div className="progress-bar__fill" style={{ width: `${progress}%` }} />
        </div>
        {runningStage && (
          <p style={{ marginTop: 8, fontSize: 12 }}>{runningStage.message}</p>
        )}
      </div>

      {/* Stage list */}
      <div className="card" style={{ padding: 16, marginBottom: 24 }}>
        {stages.map((stage, i) => (
          <div key={stage.name} className={`stage-item ${stage.status}`} id={`stage-${stage.name}`}>
            <div className={`stage-dot ${stage.status}`} style={{ color: STAGE_COLORS[stage.status] }}>
              {stage.status === 'running' ? (
                <div className="spinner" style={{ width: 14, height: 14 }} />
              ) : (
                <span style={{ fontSize: 14 }}>{STAGE_ICONS[stage.status]}</span>
              )}
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 2 }}>
                {stage.label || stage.name}
              </div>
              {stage.message && (
                <p style={{ fontSize: 12, margin: 0 }}>{stage.message}</p>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Live stats */}
      {Object.keys(liveData).length > 0 && (
        <div className="card" style={{ padding: '16px 20px', marginBottom: 24 }}>
          <h3 style={{ marginBottom: 12 }}>Live Statistics</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: 12 }}>
            {Object.entries(liveData).map(([key, val]) => (
              <div key={key} style={{ padding: '10px 14px', borderRadius: 8, background: 'rgba(255,255,255,0.04)' }}>
                <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--indigo-l)' }}>
                  {typeof val === 'number' ? val.toLocaleString() : String(val)}
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
                  {key.replace(/_/g, ' ')}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{ padding: '16px 20px', borderRadius: 10, background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.25)' }}>
          <h3 style={{ color: 'var(--red)', marginBottom: 8 }}>Reconstruction Failed</h3>
          <p style={{ fontSize: 13, color: 'var(--text-primary)' }}>{error}</p>
        </div>
      )}
    </div>
  )
}
