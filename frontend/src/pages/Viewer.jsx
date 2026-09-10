import React, { useEffect, useState, Suspense } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { api } from '../lib/api.js'
import ThreeViewer from '../components/ThreeViewer.jsx'
import QualityPanel from '../components/QualityPanel.jsx'
import RecaptureAdvisor from '../components/RecaptureAdvisor.jsx'
import ComparisonTable from '../components/ComparisonTable.jsx'

const VIEW_MODES = [
  { id: 'sparse',     label: 'Sparse Cloud', icon: '⁖' },
  { id: 'dense',      label: 'Infilled Dense Cloud', icon: '፨' },
  { id: 'mesh',       label: '3D Surface Mesh', icon: '◬' },
  { id: 'confidence', label: 'Confidence Heatmap', icon: '◉' },
  { id: 'cameras',    label: 'Flight Trajectory', icon: '✈' },
]

const SPATIAL_PRESETS = [
  { id: 'iso',   label: 'Isometric' },
  { id: 'top',   label: 'Top View' },
  { id: 'front', label: 'Front' },
  { id: 'side',  label: 'Side' },
]

const TABS = [
  { id: 'quality',    label: 'Quality Score' },
  { id: 'recapture',  label: 'Recapture Advisor' },
  { id: 'comparison', label: 'Method Comparison' },
]

export default function Viewer() {
  const { jobId } = useParams()
  const navigate = useNavigate()
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [viewMode, setViewMode] = useState('dense')
  const [activeTab, setActiveTab] = useState('quality')
  const [cameraPoses, setCameraPoses] = useState([])
  
  // Interactive 3D spatial controls state
  const [pointSize, setPointSize] = useState(0.025)
  const [wireframe, setWireframe] = useState(false)
  const [autoRotate, setAutoRotate] = useState(false)
  const [spatialPreset, setSpatialPreset] = useState('iso')
  const [targetFocus, setTargetFocus] = useState(null)

  useEffect(() => {
    api.getResults(jobId)
      .then(async (data) => {
        setResult(data)
        // Default to dense/mesh if available, otherwise sparse
        if (data.artifacts?.dense_ply) {
          setViewMode('dense')
        } else if (data.artifacts?.mesh_ply) {
          setViewMode('mesh')
        } else {
          setViewMode('sparse')
        }

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
          navigate(`/jobs/${jobId}/progress`)
        } else {
          setError(err?.response?.data?.detail || 'Failed to load results')
          setLoading(false)
        }
      })
  }, [jobId, navigate])

  const sparsePlyUrl = result?.artifacts?.sparse_ply
  const densePlyUrl = result?.artifacts?.dense_ply
  const meshPlyUrl = result?.artifacts?.mesh_ply
  const confidencePlyUrl = result?.artifacts?.confidence_ply

  const handleFocusRegion = (region) => {
    if (region?.center) {
      setTargetFocus(region.center)
    }
  }

  if (loading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 'calc(100vh - 65px)', gap: 12, flexDirection: 'column' }}>
        <div className="spinner" style={{ width: 36, height: 36 }} />
        <p style={{ color: 'var(--text-secondary)' }}>Loading 3D model & spatial assets…</p>
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
      {/* ── 3D Interactive Canvas Area ──────────────────────────────── */}
      <div className="viewer-canvas-wrap">
        
        {/* Top Control Bar: Mode Selectors */}
        <div style={{
          position: 'absolute', top: 16, left: 16, zIndex: 10,
          display: 'flex', gap: 6, flexWrap: 'wrap', maxWidth: 'calc(100% - 280px)',
          background: 'rgba(8, 8, 15, 0.75)', backdropFilter: 'blur(10px)',
          padding: '6px', borderRadius: '10px', border: '1px solid rgba(255,255,255,0.08)',
        }}>
          {VIEW_MODES.map((m) => (
            <button
              key={m.id}
              id={`view-${m.id}`}
              className={`view-btn ${viewMode === m.id ? 'active' : ''}`}
              onClick={() => setViewMode(m.id)}
              style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 12, padding: '6px 10px' }}
            >
              <span>{m.icon}</span>
              <span>{m.label}</span>
            </button>
          ))}
        </div>

        {/* Top Right: Stats Overlay */}
        {result && (
          <div style={{
            position: 'absolute', top: 16, right: 16, zIndex: 10,
            background: 'rgba(8, 8, 15, 0.85)', backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255,255,255,0.08)',
            borderRadius: 10, padding: '10px 14px', fontSize: 12,
            display: 'flex', gap: 14, alignItems: 'center',
          }}>
            <div>
              <div style={{ color: 'var(--text-muted)', fontSize: 10, textTransform: 'uppercase' }}>Points</div>
              <div style={{ fontWeight: 700, color: 'var(--text-primary)', fontFeatureSettings: '"tnum"' }}>
                {(result.colmap_stats?.sparse_point_count || 0).toLocaleString()}
              </div>
            </div>
            <div style={{ width: 1, height: 20, background: 'rgba(255,255,255,0.1)' }} />
            <div>
              <div style={{ color: 'var(--text-muted)', fontSize: 10, textTransform: 'uppercase' }}>Error</div>
              <div style={{ fontWeight: 700, color: 'var(--emerald)', fontFeatureSettings: '"tnum"' }}>
                {result.colmap_stats?.mean_reprojection_error?.toFixed(2) || '—'} px
              </div>
            </div>
            <div style={{ width: 1, height: 20, background: 'rgba(255,255,255,0.1)' }} />
            <div>
              <div style={{ color: 'var(--text-muted)', fontSize: 10, textTransform: 'uppercase' }}>Quality</div>
              <div style={{ fontWeight: 700, color: 'var(--indigo-l)' }}>
                {result.quality_score?.overall?.toFixed(0) ?? '—'}/100
              </div>
            </div>
          </div>
        )}

        {/* Bottom Left: Spatial Zoom & Camera Angles Toolbar */}
        <div style={{
          position: 'absolute', bottom: 20, left: 16, zIndex: 10,
          background: 'rgba(8, 8, 15, 0.85)', backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255,255,255,0.08)',
          borderRadius: 10, padding: '8px 12px',
          display: 'flex', gap: 12, alignItems: 'center',
        }}>
          <span style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>
            Spatial Views:
          </span>
          <div style={{ display: 'flex', gap: 4 }}>
            {SPATIAL_PRESETS.map((p) => (
              <button
                key={p.id}
                className={`view-btn ${spatialPreset === p.id ? 'active' : ''}`}
                onClick={() => { setSpatialPreset(p.id); setTargetFocus(null); }}
                style={{ fontSize: 11, padding: '4px 8px' }}
              >
                {p.label}
              </button>
            ))}
          </div>
          <div style={{ width: 1, height: 18, background: 'rgba(255,255,255,0.1)' }} />
          <button
            className={`view-btn ${autoRotate ? 'active' : ''}`}
            onClick={() => setAutoRotate(!autoRotate)}
            style={{ fontSize: 11, padding: '4px 8px' }}
            title="Auto-rotate 3D model"
          >
            {autoRotate ? '⏸ Stop Spin' : '⟳ Auto Spin'}
          </button>
        </div>

        {/* Bottom Right: Point Size & Rendering Fine-Tuning */}
        <div style={{
          position: 'absolute', bottom: 20, right: 16, zIndex: 10,
          background: 'rgba(8, 8, 15, 0.85)', backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255,255,255,0.08)',
          borderRadius: 10, padding: '8px 14px',
          display: 'flex', gap: 14, alignItems: 'center',
        }}>
          {viewMode !== 'mesh' && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Point Size</span>
              <input
                type="range"
                min="0.008"
                max="0.08"
                step="0.002"
                value={pointSize}
                onChange={(e) => setPointSize(parseFloat(e.target.value))}
                style={{ width: 70, cursor: 'pointer' }}
              />
            </div>
          )}

          {viewMode === 'mesh' && (
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={wireframe}
                onChange={(e) => setWireframe(e.target.checked)}
              />
              <span>Wireframe</span>
            </label>
          )}
        </div>

        {/* 3D Canvas */}
        <Suspense fallback={
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', flexDirection: 'column', gap: 12 }}>
            <div className="spinner" style={{ width: 32, height: 32 }} />
            <p style={{ fontSize: 13, color: 'var(--text-secondary)' }}>Rendering 3D model…</p>
          </div>
        }>
          <ThreeViewer
            sparsePlyUrl={sparsePlyUrl}
            densePlyUrl={densePlyUrl}
            meshPlyUrl={meshPlyUrl}
            confidencePlyUrl={confidencePlyUrl}
            cameraPoses={cameraPoses}
            activeMode={viewMode}
            pointSize={pointSize}
            wireframe={wireframe}
            autoRotate={autoRotate}
            viewPreset={spatialPreset}
            targetFocus={targetFocus}
          />
        </Suspense>
      </div>

      {/* ── Right Sidebar ────────────────────────────────────────────── */}
      <div className="viewer-sidebar">
        <div className="sidebar-section">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
            <h3 style={{ fontSize: 14 }}>Reconstruction Results</h3>
            <span className="badge badge-indigo">{result?.method}</span>
          </div>
          <p style={{ fontSize: 11, margin: 0, color: 'var(--text-muted)' }}>
            Job: {jobId}
          </p>
        </div>

        {/* Tab Controls */}
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

        {/* Tab Content */}
        {activeTab === 'quality' && <QualityPanel result={result} />}
        {activeTab === 'recapture' && (
          <RecaptureAdvisor
            failureRegions={result?.failure_regions || []}
            recommendations={result?.recommendations || []}
            onSelectRegion={handleFocusRegion}
          />
        )}
        {activeTab === 'comparison' && (
          <div style={{ padding: 16 }}>
            <ComparisonTable rows={result?.comparison || []} />
          </div>
        )}

        {/* Frame Selection Stats Footer */}
        {result?.frame_stats && (
          <div className="sidebar-section" style={{ marginTop: 'auto' }}>
            <h3 style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', marginBottom: 10 }}>
              Reconstruction Pipeline Stats
            </h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              {[
                ['Extracted', result.frame_stats.total_extracted],
                ['Selected', result.frame_stats.selected_for_reconstruction],
                ['Selected %', `${(result.frame_stats.selection_ratio * 100).toFixed(0)}%`],
                ['3D Points', (result.colmap_stats?.sparse_point_count || 0).toLocaleString()],
              ].map(([label, val]) => (
                <div key={label} style={{ padding: '8px 10px', borderRadius: 6, background: 'rgba(255,255,255,0.03)' }}>
                  <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--indigo-l)' }}>{val ?? '—'}</div>
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
