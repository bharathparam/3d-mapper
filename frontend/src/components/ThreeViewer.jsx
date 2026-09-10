import React, { Suspense, useRef, useState, useEffect } from 'react'
import { Canvas, useLoader, useThree } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera } from '@react-three/drei'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js'
import * as THREE from 'three'

/* ── Point Cloud with Auto-Centering ──────────────────────────────── */
function PointCloud({ url, pointSize = 0.025 }) {
  const geometry = useLoader(PLYLoader, url)

  useEffect(() => {
    if (geometry) {
      geometry.center()
      if (!geometry.attributes.color) {
        const count = geometry.attributes.position.count
        const colors = new Float32Array(count * 3).fill(0.9)
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
        transparent
        opacity={0.96}
      />
    </points>
  )
}

/* ── Surface Mesh with Auto-Centering & Lighting ──────────────────── */
function SurfaceMesh({ url, wireframe = false }) {
  const geometry = useLoader(PLYLoader, url)

  useEffect(() => {
    if (geometry) {
      geometry.center()
      geometry.computeVertexNormals()
    }
  }, [geometry])

  if (!geometry) return null

  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial
        vertexColors={Boolean(geometry.attributes.color)}
        color={geometry.attributes.color ? undefined : '#818cf8'}
        roughness={0.35}
        metalness={0.08}
        wireframe={wireframe}
        side={THREE.DoubleSide}
      />
    </mesh>
  )
}

/* ── Camera Trajectory & Markers ─────────────────────────────────── */
function CameraTrajectory({ poses, visible = true }) {
  if (!visible || !poses || poses.length < 2) return null
  const points = poses.map((p) => new THREE.Vector3(...p.center))
  const geometry = new THREE.BufferGeometry().setFromPoints(points)
  return (
    <group>
      <line>
        <bufferGeometry attach="geometry" {...geometry} />
        <lineBasicMaterial attach="material" color="#6366f1" linewidth={2} />
      </line>
      {poses.map((p, i) => (
        <group key={i} position={p.center}>
          <mesh>
            <coneGeometry args={[0.08, 0.16, 4]} />
            <meshBasicMaterial color="#ec4899" wireframe />
          </mesh>
        </group>
      ))}
    </group>
  )
}

/* ── Scene Camera Auto-Fit Controller ────────────────────────────── */
function AutoFitScene({ url, viewPreset, controlsRef }) {
  const { camera } = useThree()
  const geo = useLoader(PLYLoader, url)

  useEffect(() => {
    if (geo) {
      geo.computeBoundingSphere()
      const radius = Math.max(geo.boundingSphere?.radius || 4, 1.5)
      const dist = radius * 2.2

      switch (viewPreset) {
        case 'top':
          camera.position.set(0, dist * 1.4, 0.001)
          break
        case 'front':
          camera.position.set(0, 0, dist)
          break
        case 'side':
          camera.position.set(dist, 0, 0)
          break
        case 'iso':
        default:
          camera.position.set(dist * 0.8, dist * 0.6, dist * 0.8)
          break
      }

      camera.near = radius * 0.005
      camera.far = radius * 50
      camera.lookAt(0, 0, 0)
      camera.updateProjectionMatrix()

      if (controlsRef.current) {
        controlsRef.current.target.set(0, 0, 0)
        controlsRef.current.maxDistance = radius * 10
        controlsRef.current.minDistance = radius * 0.1
        controlsRef.current.update()
      }
    }
  }, [geo, viewPreset, camera, controlsRef])

  return null
}

/* ── Camera Director for Target Coordinates Focus ─────────────────── */
function TargetFocusDirector({ targetPosition, controlsRef }) {
  const { camera } = useThree()

  useEffect(() => {
    if (targetPosition && controlsRef.current) {
      controlsRef.current.target.set(...targetPosition)
      camera.position.set(
        targetPosition[0] + 3,
        targetPosition[1] + 2,
        targetPosition[2] + 3
      )
      controlsRef.current.update()
    }
  }, [targetPosition, camera, controlsRef])

  return null
}

/* ── Main Interactive Three.js Canvas ─────────────────────────────── */
export default function ThreeViewer({
  sparsePlyUrl,
  densePlyUrl,
  meshPlyUrl,
  confidencePlyUrl,
  primaryObjectPlyUrl,
  cameraPoses,
  activeMode = 'dense',
  focusTargetOnly = true,
  pointSize = 0.025,
  wireframe = false,
  autoRotate = false,
  viewPreset = 'iso',
  targetFocus = null,
}) {
  const controlsRef = useRef()

  // Select active geometry URL based on filter and mode
  let activeUrl = sparsePlyUrl
  if (activeMode === 'mesh' && meshPlyUrl) {
    activeUrl = meshPlyUrl
  } else if (activeMode === 'dense' && densePlyUrl) {
    activeUrl = densePlyUrl
  } else if (activeMode === 'confidence' && confidencePlyUrl) {
    activeUrl = confidencePlyUrl
  } else if (focusTargetOnly && primaryObjectPlyUrl) {
    activeUrl = primaryObjectPlyUrl
  }

  return (
    <Canvas
      gl={{ antialias: true, alpha: false, powerPreference: 'high-performance' }}
      style={{ background: '#07070d', width: '100%', height: '100%' }}
      id="three-canvas"
    >
      <PerspectiveCamera makeDefault position={[5, 4, 5]} fov={45} />
      
      {/* Lighting */}
      <ambientLight intensity={0.7} />
      <directionalLight position={[15, 25, 20]} intensity={1.3} />
      <directionalLight position={[-15, -10, -15]} intensity={0.5} />
      <pointLight position={[0, 10, 0]} intensity={0.6} />

      <TargetFocusDirector targetPosition={targetFocus} controlsRef={controlsRef} />

      {activeUrl && (
        <Suspense fallback={null}>
          <AutoFitScene url={activeUrl} viewPreset={viewPreset} controlsRef={controlsRef} />

          {activeMode === 'mesh' ? (
            <SurfaceMesh url={activeUrl} wireframe={wireframe} />
          ) : (
            <PointCloud
              url={activeUrl}
              pointSize={
                activeMode === 'confidence'
                  ? pointSize * 1.2
                  : activeMode === 'dense'
                  ? pointSize * 0.8
                  : pointSize
              }
            />
          )}

          {activeMode === 'cameras' && (
            <CameraTrajectory poses={cameraPoses} visible={true} />
          )}
        </Suspense>
      )}

      <OrbitControls
        ref={controlsRef}
        enableDamping
        dampingFactor={0.08}
        rotateSpeed={0.8}
        zoomSpeed={1.4}
        panSpeed={0.9}
        autoRotate={autoRotate}
        autoRotateSpeed={1.5}
      />

      <gridHelper args={[24, 24, '#27273a', '#141424']} position={[0, -0.01, 0]} />
    </Canvas>
  )
}
