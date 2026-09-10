import React from 'react'

function ScoreRing({ score, size = 90 }) {
  const r = 36
  const circ = 2 * Math.PI * r
  const fill = (score / 100) * circ
  const color = score >= 70 ? '#10b981' : score >= 50 ? '#f59e0b' : '#ef4444'

  return (
    <svg width={size} height={size} viewBox="0 0 90 90" style={{ display: 'block' }}>
      <circle cx="45" cy="45" r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="8" />
      <circle
        cx="45" cy="45" r={r}
        fill="none"
        stroke={color}
        strokeWidth="8"
        strokeLinecap="round"
        strokeDasharray={`${fill} ${circ - fill}`}
        strokeDashoffset={circ * 0.25}
        style={{ transition: 'stroke-dasharray 0.6s ease' }}
      />
      <text x="45" y="49" textAnchor="middle" fill={color} fontSize="16" fontWeight="700" fontFamily="Inter">
        {Math.round(score)}
      </text>
      <text x="45" y="62" textAnchor="middle" fill="#475569" fontSize="9" fontFamily="Inter">
        /100
      </text>
    </svg>
  )
}

function MetricRow({ label, value, unit = '', type = 'measured' }) {
  return (
    <div className="metric-row">
      <span className="metric-label">{label}</span>
      <div style={{ textAlign: 'right' }}>
        <span className="metric-value">
          {typeof value === 'number' ? value.toLocaleString() : value} {unit}
        </span>
        <br />
        <span className="metric-badge">{type}</span>
      </div>
    </div>
  )
}

export default function QualityPanel({ result }) {
  if (!result) return (
    <div style={{ padding: 20, color: 'var(--text-muted)', fontSize: 13 }}>
      No quality data available.
    </div>
  )

  const qs = result.quality_score || {}
  const cs = result.colmap_stats || {}
  const conf = result.confidence || {}

  return (
    <div>
      {/* Overall score ring */}
      <div style={{ textAlign: 'center', padding: '20px 0 12px' }}>
        <ScoreRing score={qs.overall || 0} />
        <p style={{ fontWeight: 700, marginTop: 6, fontSize: 14 }}>
          {qs.label || '—'}
        </p>
      </div>

      {/* Sub-scores */}
      {[
        ['Camera Registration', qs.camera_registration, '%', 'measured'],
        ['Feature Coverage',    qs.feature_coverage,    '%', 'measured'],
        ['Point Density',       qs.point_density,       '%', 'measured'],
        ['Spatial Coverage',    qs.spatial_coverage,    '%', 'estimated'],
      ].map(([l, v, u, t]) => (
        <div key={l} style={{ padding: '6px 20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
            <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{l}</span>
            <span style={{ fontSize: 12, fontWeight: 600 }}>{v?.toFixed(1) ?? '—'}{u}</span>
          </div>
          <div className="progress-bar" style={{ height: 4 }}>
            <div className="progress-bar__fill" style={{ width: `${v || 0}%` }} />
          </div>
        </div>
      ))}

      {/* Raw metrics */}
      <div style={{ padding: '16px 20px' }}>
        <h3 style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', marginBottom: 10 }}>
          COLMAP Measurements
        </h3>
        <MetricRow label="Registered images"  value={`${cs.registered_images || 0}/${cs.total_images || 0}`} type="measured" />
        <MetricRow label="Sparse 3D points"   value={cs.sparse_point_count || 0} type="measured" />
        <MetricRow label="Reprojection error" value={(cs.mean_reprojection_error || 0).toFixed(3)} unit="px" type="measured" />
        <MetricRow label="Mean track length"  value={(cs.mean_track_length || 0).toFixed(2)} unit="imgs/pt" type="measured" />
        <MetricRow label="Processing time"    value={(cs.processing_time_seconds || 0).toFixed(0)} unit="s" type="measured" />
      </div>

      {/* Confidence breakdown */}
      <div style={{ padding: '0 20px 16px' }}>
        <h3 style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', marginBottom: 10 }}>
          Confidence Breakdown
        </h3>
        {[
          ['High', conf.high_ratio || 0, '#10b981'],
          ['Medium', conf.medium_ratio || 0, '#f59e0b'],
          ['Low', conf.low_ratio || 0, '#ef4444'],
        ].map(([label, ratio, color]) => (
          <div key={label} style={{ marginBottom: 8 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
              <span style={{ fontSize: 12, color }}>{label}</span>
              <span style={{ fontSize: 12, fontWeight: 600 }}>{(ratio * 100).toFixed(0)}%</span>
            </div>
            <div className="progress-bar" style={{ height: 4 }}>
              <div style={{ height: '100%', width: `${ratio * 100}%`, background: color, borderRadius: 3, transition: 'width 0.5s ease' }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
