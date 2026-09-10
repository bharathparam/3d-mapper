import React, { useCallback, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../lib/api.js'

const METHODS = [
  {
    id: 'optimized',
    label: 'Our Method',
    sub: 'Quality-aware frame selection + adaptive reconstruction',
    badge: 'Recommended',
    badgeClass: 'badge-emerald',
  },
  {
    id: 'baseline',
    label: 'Baseline',
    sub: 'All frames → COLMAP (for comparison)',
    badge: null,
    badgeClass: '',
  },
  {
    id: 'both',
    label: 'Compare Both',
    sub: 'Run both methods and show metrics side-by-side',
    badge: 'SIH Demo',
    badgeClass: 'badge-indigo',
  },
]

export default function Upload() {
  const navigate = useNavigate()
  const [file, setFile] = useState(null)
  const [method, setMethod] = useState('both')
  const [dragOver, setDragOver] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    const f = e.dataTransfer.files[0]
    if (f) setFile(f)
  }, [])

  const handleSubmit = async () => {
    if (!file) return
    setUploading(true)
    setError('')
    try {
      const { job_id } = await api.createJob(file, method)
      navigate(`/jobs/${job_id}/progress`)
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || 'Upload failed')
      setUploading(false)
    }
  }

  const formatSize = (bytes) => {
    if (bytes > 1e9) return `${(bytes/1e9).toFixed(1)} GB`
    if (bytes > 1e6) return `${(bytes/1e6).toFixed(1)} MB`
    return `${(bytes/1e3).toFixed(0)} KB`
  }

  return (
    <div className="upload-page">
      <div className="upload-hero">
        {/* Hero text */}
        <div style={{textAlign:'center'}}>
          <h1 style={{marginBottom:12}}>
            Drone Video<br/>
            <span className="gradient-text">→ 3D Reconstruction</span>
          </h1>
          <p style={{fontSize:15,maxWidth:480,margin:'0 auto'}}>
            Upload your drone footage. Our system intelligently selects frames,
            reconstructs the scene, evaluates quality, and identifies regions
            that need improved capture.
          </p>
        </div>

        {/* Drop zone */}
        <div
          className={`upload-dropzone ${dragOver ? 'drag-over' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => document.getElementById('file-input').click()}
          role="button"
          tabIndex={0}
          id="upload-dropzone"
        >
          <input
            id="file-input"
            type="file"
            accept=".mp4,.mov,.avi,.webm"
            style={{ display: 'none' }}
            onChange={(e) => setFile(e.target.files[0])}
          />
          <div className="icon-wrap">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#6366f1" strokeWidth="1.5">
              <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
              <polyline points="17 8 12 3 7 8"/>
              <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
          </div>
          {file ? (
            <div>
              <p style={{ color: 'var(--text-primary)', fontWeight: 600, marginBottom: 4 }}>
                {file.name}
              </p>
              <p style={{ fontSize: 13 }}>{formatSize(file.size)}</p>
            </div>
          ) : (
            <div>
              <p style={{ color: 'var(--text-primary)', fontWeight: 600, marginBottom: 6 }}>
                Drop drone video here
              </p>
              <p style={{ fontSize: 13 }}>MP4, MOV, AVI, WebM · up to 4 GB</p>
            </div>
          )}
        </div>

        {/* Method selector */}
        <div>
          <h3 style={{ marginBottom: 12, color: 'var(--text-secondary)', fontSize: 12, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
            Reconstruction Method
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {METHODS.map((m) => (
              <div
                key={m.id}
                className="card"
                id={`method-${m.id}`}
                onClick={() => setMethod(m.id)}
                style={{
                  padding: '14px 16px',
                  cursor: 'pointer',
                  border: method === m.id ? '1px solid rgba(99,102,241,0.5)' : undefined,
                  background: method === m.id ? 'rgba(99,102,241,0.08)' : undefined,
                  display: 'flex', alignItems: 'center', gap: 12,
                }}
              >
                <div style={{
                  width: 18, height: 18, borderRadius: '50%',
                  border: `2px solid ${method === m.id ? '#6366f1' : 'rgba(255,255,255,0.2)'}`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                  {method === m.id && <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#6366f1' }} />}
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 2 }}>
                    <span style={{ fontWeight: 600, fontSize: 14 }}>{m.label}</span>
                    {m.badge && <span className={`badge ${m.badgeClass}`}>{m.badge}</span>}
                  </div>
                  <p style={{ fontSize: 12, margin: 0 }}>{m.sub}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Error */}
        {error && (
          <div style={{ padding: '12px 16px', borderRadius: 8, background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.25)', color: 'var(--red)', fontSize: 13 }}>
            {error}
          </div>
        )}

        {/* Submit */}
        <button
          className="btn btn-primary"
          id="start-reconstruction"
          onClick={handleSubmit}
          disabled={!file || uploading}
          style={{ width: '100%', justifyContent: 'center', padding: '14px', fontSize: 15 }}
        >
          {uploading ? (
            <><div className="spinner" style={{width:16,height:16}} /> Uploading…</>
          ) : (
            <><span>Start Reconstruction</span>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M5 12h14M12 5l7 7-7 7"/>
              </svg>
            </>
          )}
        </button>

        {/* Info */}
        <div style={{ display: 'flex', gap: 24, justifyContent: 'center' }}>
          {[
            ['Intelligent Sampling', 'Our frame selector'],
            ['COLMAP SfM', 'Geometric baseline'],
            ['Quality Analysis', 'Per-region scoring'],
          ].map(([title, sub]) => (
            <div key={title} style={{ textAlign: 'center' }}>
              <p style={{ fontWeight: 600, fontSize: 13, color: 'var(--text-primary)', marginBottom: 2 }}>{title}</p>
              <p style={{ fontSize: 11, margin: 0 }}>{sub}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
