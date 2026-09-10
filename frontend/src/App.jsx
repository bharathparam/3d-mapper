import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Upload from './pages/Upload.jsx'
import Dashboard from './pages/Dashboard.jsx'
import Viewer from './pages/Viewer.jsx'
import Nav from './components/Nav.jsx'

export default function App() {
  return (
    <BrowserRouter>
      <div className="layout">
        <div className="bg-gradient-animated" />
        <Nav />
        <Routes>
          <Route path="/" element={<Upload />} />
          <Route path="/jobs/:jobId/progress" element={<Dashboard />} />
          <Route path="/jobs/:jobId/viewer" element={<Viewer />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </BrowserRouter>
  )
}
