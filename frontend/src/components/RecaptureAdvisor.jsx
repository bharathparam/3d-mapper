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

export default function RecaptureAdvisor({ failureRegions = [], recommendations = [], onSelectRegion }) {
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
    <div style={{ padding: 16 }}>
      <p style={{ fontSize: 12, marginBottom: 12, color: 'var(--text-secondary)' }}>
        {failureRegions.length} region{failureRegions.length !== 1 ? 's' : ''} diagnosed with drone recapture tips:
      </p>
      {failureRegions.slice(0, 50).map((region, i) => {
        const rec = recommendations[i]
        const isOpen = open === i
        return (
          <div key={i} className={`region-item ${rec?.priority || 'low'}`} id={`region-${i}`} style={{ marginBottom: 8 }}>
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
                <p style={{ margin: '2px 0 0', fontSize: 11, color: 'var(--text-muted)' }}>
                  {region.point_count} pts · {region.camera_count} cameras · conf {(region.confidence_score * 100).toFixed(0)}%
                </p>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                {onSelectRegion && region.center && (
                  <button
                    className="view-btn"
                    style={{ fontSize: 10, padding: '2px 6px', background: 'rgba(99,102,241,0.2)' }}
                    onClick={(e) => {
                      e.stopPropagation()
                      onSelectRegion(region)
                    }}
                    title="Focus 3D camera on this region"
                  >
                    🎯 Focus
                  </button>
                )}
                <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>{isOpen ? '▲' : '▼'}</span>
              </div>
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
