import React from 'react'
import { useNavigate } from 'react-router-dom'

export default function Nav() {
  const navigate = useNavigate()
  return (
    <nav className="nav">
      <div className="nav-logo" style={{cursor:'pointer'}} onClick={() => navigate('/')}>
        <div className="nav-logo-icon">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2">
            <path d="M12 2L2 7l10 5 10-5-10-5z"/>
            <path d="M2 17l10 5 10-5"/>
            <path d="M2 12l10 5 10-5"/>
          </svg>
        </div>
        <span>Drone<span className="gradient-text">3D</span></span>
      </div>
      <div style={{display:'flex',gap:'8px',alignItems:'center'}}>
        <span className="badge badge-indigo">SIH 2024</span>
        <span style={{fontSize:'12px',color:'var(--text-muted)'}}>Quality-Aware Reconstruction</span>
      </div>
    </nav>
  )
}
