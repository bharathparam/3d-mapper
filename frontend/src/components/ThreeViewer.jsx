import React, { Suspense, useRef, useState, useEffect } from 'react'
import { Canvas, useLoader, useThree } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera } from '@react-three/drei'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js'
import * as THREE from 'three'

/* ── Point cloud rendered from PLY geometry ──────────────────────────────── */
function PointCloud({ url, pointSize = 0.02 }) {
  const geometry = useLoader(PLYLoader, url)

  useEffect(() => {
    if (geometry) {
      geometry.center()
      if (!geometry.attributes.color) {
        // Default white if no colour attribute
        const count = geometry.attributes.position.count
        const colors = new Float32Array(count * 3).fill(1)
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
      }
    }
  }, [geometry])

  if (!geometry) return null

  return (
    <points>
      <bufferGeometry attach="geometry" {...geometry} />
      <pointsMaterial
        attach="material"
        size={pointSize}
        vertexColors
        sizeAttenuation
      />
    </points>
  )
}

/* ── Camera trajectory lines ─────────────────────────────────────────────── */
function CameraTrajectory({ poses }) {
  if (!poses || poses.length < 2) return null
  const points = poses.map((p) => new THREE.Vector3(...p.center))
  const geometry = new THREE.BufferGeometry().setFromPoints(points)
  return (
    <line>
      <bufferGeometry attach="geometry" {...geometry} />
      <lineBasicMaterial attach="material" color="#6366f1" linewidth={1} />
    </line>
  )
}

/* ── Camera frustum markers ──────────────────────────────────────────────── */
function CameraMarkers({ poses, visible }) {
  if (!visible || !poses?.length) return null
  return (
    <group>
      {poses.slice(0, 200).map((p, i) => (
        <mesh key={i} position={p.center}>
          <boxGeometry args={[0.05, 0.05, 0.05]} />
          <meshBasicMaterial color="#8b5cf6" />
        </mesh>
      ))}
    </group>
  )
}

/* ── Scene controller (centres camera on load) ──────────────────────────── */
function SceneSetup({ plyUrl }) {
  const { camera } = useThree()
  const geo = useLoader(PLYLoader, plyUrl)
  useEffect(() => {
    if (geo) {
      geo.computeBoundingSphere()
      const r = geo.boundingSphere?.radius || 5
      camera.position.set(r * 1.5, r, r * 1.5)
      camera.near = r * 0.001
      camera.far = r * 100
      camera.updateProjectionMatrix()
    }
  }, [geo, camera])
  return null
}

/* ── Main viewer component ───────────────────────────────────────────────── */
export default function ThreeViewer({ plyUrl, confidencePlyUrl, cameraPoses, viewMode }) {
  const activeUrl = viewMode === 'confidence' && confidencePlyUrl ? confidencePlyUrl : plyUrl

  return (
    <Canvas
      gl={{ antialias: true, alpha: false }}
      style={{ background: '#08080f' }}
      id="three-canvas"
    >
      <PerspectiveCamera makeDefault fov={60} />
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} intensity={0.5} />

      {activeUrl && (
        <Suspense fallback={null}>
          <SceneSetup plyUrl={activeUrl} />
          <PointCloud
            url={activeUrl}
            pointSize={viewMode === 'confidence' ? 0.025 : 0.018}
          />
        </Suspense>
      )}

      {(viewMode === 'cameras' || viewMode === 'confidence') && (
        <CameraTrajectory poses={cameraPoses} />
      )}
      <CameraMarkers poses={cameraPoses} visible={viewMode === 'cameras'} />

      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        rotateSpeed={0.8}
        zoomSpeed={1.2}
      />

      {/* Grid helper */}
      <gridHelper args={[20, 20, '#1e1e2e', '#1a1a2e']} />
    </Canvas>
  )
}
