import React, { useState } from 'react'

const FAILURE_COLORS = {
  missing_viewpoint: 'var(--red)',
  low_overlap:       'var(--amber)',
  motion_blur:       'var(--purple)',
  textureless:       'var(--indigo)',
}

const FAILURE_ICONS = {
  missing_viewpoint: '📷',
  low_overlap:       '🔁',
  motion_blur:       '💨',
  textureless:       '🔲',
}

export default function RecaptureAdvisor({ failureRegions = [], recommendations = [] }) {
  const [open, setOpen] = useState(null)

  if (!failureRegions.length) {
    return (
      <div style={{ padding: 20 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '12px 16px', borderRadius: 8, background: 'rgba(16,185,129,0.08)', border: '1px solid rgba(16,185,129,0.2)' }}>
          <span>✓</span>
          <p style={{ margin: 0, color: 'var(--emerald)', fontSize: 13 }}>
            No significant low-confidence regions detected.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div style={{ padding: 20 }}>
      <p style={{ fontSize: 12, marginBottom: 12 }}>
        {failureRegions.length} region{failureRegions.length !== 1 ? 's' : ''} need improved capture:
      </p>
      {failureRegions.map((region, i) => {
        const rec = recommendations[i]
        const isOpen = open === i
        return (
          <div key={i} className={`region-item ${rec?.priority || 'low'}`} id={`region-${i}`}>
            <div
              style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 8 }}
              onClick={() => setOpen(isOpen ? null : i)}
            >
              <span style={{ fontSize: 16 }}>{FAILURE_ICONS[region.failure_type] || '⚠'}</span>
              <div style={{ flex: 1 }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: FAILURE_COLORS[region.failure_type] }}>
                  {region.failure_type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                  <span className={`badge badge-${rec?.priority === 'high' ? 'red' : rec?.priority === 'medium' ? 'amber' : 'indigo'}`}
                    style={{ marginLeft: 8 }}>
                    {rec?.priority || 'low'}
                  </span>
                </div>
                <p style={{ margin: '2px 0 0', fontSize: 11 }}>
                  {region.point_count} pts · {region.camera_count} cameras · conf {(region.confidence_score * 100).toFixed(0)}%
                </p>
              </div>
              <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>{isOpen ? '▲' : '▼'}</span>
            </div>
            {isOpen && rec && (
              <div style={{ marginTop: 10, paddingTop: 10, borderTop: '1px solid rgba(255,255,255,0.06)' }}>
                {rec.description.split('\n').map((line, j) => (
                  <p key={j} style={{ margin: '3px 0', fontSize: 12, color: line.startsWith('•') ? 'var(--text-primary)' : 'var(--text-secondary)' }}>
                    {line}
                  </p>
                ))}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
