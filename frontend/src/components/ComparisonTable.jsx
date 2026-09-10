import React from 'react'

export default function ComparisonTable({ rows = [] }) {
  if (!rows.length) return (
    <div style={{ padding: '16px 20px', color: 'var(--text-muted)', fontSize: 13 }}>
      Run with method="both" to see comparison.
    </div>
  )

  const getDeltaClass = (delta) => {
    if (!delta || delta === 'N/A') return ''
    if (delta.includes('↑')) return 'delta-positive'
    if (delta.includes('↓')) return 'delta-negative'
    return ''
  }

  return (
    <div style={{ padding: '0 4px' }}>
      <table className="comparison-table">
        <thead>
          <tr>
            <th>Metric</th>
            <th style={{ textAlign: 'right' }}>Baseline</th>
            <th style={{ textAlign: 'right' }}>Our Method</th>
            <th style={{ textAlign: 'right' }}>Δ</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.metric}>
              <td style={{ color: 'var(--text-secondary)' }}>
                {row.metric}
                {!row.is_measured && (
                  <span style={{ fontSize: 9, color: 'var(--text-muted)', marginLeft: 4 }}>[est]</span>
                )}
              </td>
              <td style={{ textAlign: 'right' }}>{row.baseline_value}</td>
              <td style={{ textAlign: 'right', fontWeight: 600 }}>{row.optimized_value}</td>
              <td style={{ textAlign: 'right' }}>
                <span className={getDeltaClass(row.delta)} style={{ fontSize: 11 }}>
                  {row.delta}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p style={{ marginTop: 10, fontSize: 10, color: 'var(--text-muted)' }}>
        All values are directly measured from COLMAP output. [est] = estimated.
      </p>
    </div>
  )
}
