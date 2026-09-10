import React, { useEffect, useState, Suspense } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { api } from '../lib/api.js'
import ThreeViewer from '../components/ThreeViewer.jsx'
import QualityPanel from '../components/QualityPanel.jsx'
import RecaptureAdvisor from '../components/RecaptureAdvisor.jsx'
import ComparisonTable from '../components/ComparisonTable.jsx'

const VIEW_MODES = [
  { id: 'pointcloud', label: 'Point Cloud' },
  { id: 'confidence', label: 'Confidence' },
  { id: 'cameras',    label: 'Cameras' },
]

const TABS = [
  { id: 'quality',     label: 'Quality' },
  { id: 'recapture',  label: 'Recapture' },
  { id: 'comparison', label: 'Comparison' },
]

export default function Viewer() {
  const { jobId } = useParams()
  const navigate = useNavigate()
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [viewMode, setViewMode] = useState('pointcloud')
  const [activeTab, setActiveTab] = useState('quality')
  const [cameraPoses, setCameraPoses] = useState([])

  useEffect(() => {
    api.getResults(jobId)
      .then(async (data) => {
        setResult(data)
        // Load camera poses
        if (data.artifacts?.camera_poses) {
          try {
            const res = await fetch(data.artifacts.camera_poses)
            const json = await res.json()
            setCameraPoses(json.cameras || [])
          } catch {}
        }
        setLoading(false)
      })
      .catch((err) => {
        if (err?.response?.status === 202) {
          // Still processing — redirect to dashboard
          navigate(`/jobs/${jobId}/progress`)
        } else {
          setError(err?.response?.data?.detail || 'Failed to load results')
          setLoading(false)
        }
      })
  }, [jobId, navigate])

  const plyUrl = result?.artifacts?.sparse_ply
  const confidencePlyUrl = result?.artifacts?.confidence_ply

  if (loading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 'calc(100vh - 65px)', gap: 12, flexDirection: 'column' }}>
        <div className="spinner" style={{ width: 36, height: 36 }} />
        <p>Loading reconstruction…</p>
      </div>
    )
  }

  if (error) {
    return (
      <div style={{ padding: 32 }}>
        <div style={{ padding: 20, borderRadius: 10, background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.25)', color: 'var(--red)' }}>
          {error}
        </div>
      </div>
    )
  }

  return (
    <div className="viewer-page">
      {/* ── 3D Canvas ────────────────────────────────────────────────── */}
      <div className="viewer-canvas-wrap">
        {/* View mode toggle */}
        <div style={{ position: 'absolute', top: 16, left: 16, zIndex: 10, display: 'flex', gap: 6 }}>
          {VIEW_MODES.map((m) => (
            <button
              key={m.id}
              id={`view-${m.id}`}
              className={`view-btn ${viewMode === m.id ? 'active' : ''}`}
              onClick={() => setViewMode(m.id)}
            >
              {m.label}
            </button>
          ))}
        </div>

        {/* Stats overlay */}
        {result && (
          <div style={{
            position: 'absolute', top: 16, right: 16, zIndex: 10,
            background: 'rgba(5,5,8,0.75)', backdropFilter: 'blur(8px)',
            border: '1px solid rgba(255,255,255,0.08)',
            borderRadius: 8, padding: '8px 12px', fontSize: 12,
            display: 'flex', gap: 16, alignItems: 'center',
          }}>
            <span style={{ color: 'var(--text-muted)' }}>Points</span>
            <span style={{ fontWeight: 700, fontFeatureSettings: '"tnum"' }}>
              {(result.colmap_stats?.sparse_point_count || 0).toLocaleString()}
            </span>
            <span style={{ color: 'var(--text-muted)' }}>Score</span>
            <span style={{ fontWeight: 700, color: 'var(--emerald)' }}>
              {result.quality_score?.overall?.toFixed(0) ?? '—'}/100
            </span>
          </div>
        )}

        {plyUrl ? (
          <Suspense fallback={
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', flexDirection: 'column', gap: 12 }}>
              <div className="spinner" style={{ width: 32, height: 32 }} />
              <p style={{ fontSize: 13 }}>Loading point cloud…</p>
            </div>
          }>
            <ThreeViewer
              plyUrl={plyUrl}
              confidencePlyUrl={confidencePlyUrl}
              cameraPoses={cameraPoses}
              viewMode={viewMode}
            />
          </Suspense>
        ) : (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
            <p style={{ color: 'var(--text-muted)' }}>No point cloud available</p>
          </div>
        )}
      </div>

      {/* ── Right sidebar ────────────────────────────────────────────── */}
      <div className="viewer-sidebar">
        {/* Job info */}
        <div className="sidebar-section">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
            <h3 style={{ fontSize: 14 }}>Reconstruction Results</h3>
            <span className="badge badge-indigo">{result?.method}</span>
          </div>
          <p style={{ fontSize: 11, margin: 0 }}>Job: {jobId}</p>
        </div>

        {/* Tab navigation */}
        <div className="sidebar-section" style={{ paddingBottom: 0 }}>
          <div className="view-controls">
            {TABS.map((t) => (
              <button
                key={t.id}
                id={`tab-${t.id}`}
                className={`view-btn ${activeTab === t.id ? 'active' : ''}`}
                onClick={() => setActiveTab(t.id)}
              >
                {t.label}
                {t.id === 'recapture' && result?.failure_regions?.length > 0 && (
                  <span style={{ marginLeft: 4, background: 'var(--red)', color: '#fff', borderRadius: '50%', width: 16, height: 16, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontSize: 10 }}>
                    {result.failure_regions.length}
                  </span>
                )}
              </button>
            ))}
          </div>
        </div>

        {/* Tab content */}
        {activeTab === 'quality' && <QualityPanel result={result} />}
        {activeTab === 'recapture' && (
          <RecaptureAdvisor
            failureRegions={result?.failure_regions || []}
            recommendations={result?.recommendations || []}
          />
        )}
        {activeTab === 'comparison' && (
          <div style={{ padding: 16 }}>
            <ComparisonTable rows={result?.comparison || []} />
          </div>
        )}

        {/* Frame stats footer */}
        {result?.frame_stats && (
          <div className="sidebar-section" style={{ marginTop: 'auto' }}>
            <h3 style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', marginBottom: 10 }}>
              Frame Selection
            </h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              {[
                ['Extracted', result.frame_stats.total_extracted],
                ['Selected', result.frame_stats.selected_for_reconstruction],
                ['Ratio', `${(result.frame_stats.selection_ratio * 100).toFixed(0)}%`],
                ['Registered', result.colmap_stats?.registered_images],
              ].map(([label, val]) => (
                <div key={label} style={{ padding: '8px 10px', borderRadius: 6, background: 'rgba(255,255,255,0.03)' }}>
                  <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--indigo-l)' }}>{val ?? '—'}</div>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 1 }}>{label}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
