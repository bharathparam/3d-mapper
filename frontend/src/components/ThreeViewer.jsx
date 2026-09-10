import React, { Suspense, useRef, useState, useEffect } from 'react'
import { Canvas, useLoader, useThree } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera } from '@react-three/drei'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js'
import * as THREE from 'three'

/* ── Spatial Density Centering Helper ─────────────────────────────────
   Finds the medoid/median of the 3D point cloud so the densest region
   is pinned precisely at (0, 0, 0), completely ignoring far-off outliers.
─────────────────────────────────────────────────────────────────────── */
function centerGeometryOnDensityPeak(geometry) {
  if (!geometry || !geometry.attributes.position) return 5.0
  const pos = geometry.attributes.position.array
  const count = geometry.attributes.position.count
  if (count < 10) return 5.0

  const step = Math.max(1, Math.floor(count / 1500))
  const sampleX = []
  const sampleY = []
  const sampleZ = []

  for (let i = 0; i < count; i += step) {
    sampleX.push(pos[i * 3])
    sampleY.push(pos[i * 3 + 1])
    sampleZ.push(pos[i * 3 + 2])
  }

  sampleX.sort((a, b) => a - b)
  sampleY.sort((a, b) => a - b)
  sampleZ.sort((a, b) => a - b)

  const medX = sampleX[Math.floor(sampleX.length / 2)]
  const medY = sampleY[Math.floor(sampleY.length / 2)]
  const medZ = sampleZ[Math.floor(sampleZ.length / 2)]

  // Translate all points so peak density is exactly at (0, 0, 0)
  geometry.translate(-medX, -medY, -medZ)

  // Compute robust 90th percentile radius
  const dists = []
  const newPos = geometry.attributes.position.array
  for (let i = 0; i < count; i += step) {
    const x = newPos[i * 3]
    const y = newPos[i * 3 + 1]
    const z = newPos[i * 3 + 2]
    dists.push(Math.sqrt(x * x + y * y + z * z))
  }
  dists.sort((a, b) => a - b)
  const r90 = dists[Math.floor(dists.length * 0.9)] || 5.0

  geometry.computeBoundingSphere()
  return Math.max(r90, 2.0)
}

/* ── Point Cloud with Density-Centering ────────────────────────────── */
function PointCloud({ url, pointSize = 0.025, onRadiusCalculated }) {
  const geometry = useLoader(PLYLoader, url)

  useEffect(() => {
    if (geometry) {
      const radius = centerGeometryOnDensityPeak(geometry)
      if (onRadiusCalculated) onRadiusCalculated(radius)

      if (!geometry.attributes.color) {
        const count = geometry.attributes.position.count
        const colors = new Float32Array(count * 3).fill(0.9)
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
      }
    }
  }, [geometry, onRadiusCalculated])

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

/* ── Surface Mesh with Density-Centering ───────────────────────────── */
function SurfaceMesh({ url, wireframe = false, onRadiusCalculated }) {
  const geometry = useLoader(PLYLoader, url)

  useEffect(() => {
    if (geometry) {
      const radius = centerGeometryOnDensityPeak(geometry)
      if (onRadiusCalculated) onRadiusCalculated(radius)
      geometry.computeVertexNormals()
    }
  }, [geometry, onRadiusCalculated])

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

/* ── Camera Flight Trajectory & Markers ───────────────────────────── */
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

/* ── Camera Director & Auto-Fit Controller ────────────────────────── */
function CameraController({ radius = 8.0, viewPreset = 'iso', autoRotate = false, targetFocus = null, controlsRef }) {
  const { camera } = useThree()

  useEffect(() => {
    if (targetFocus && controlsRef.current) {
      controlsRef.current.target.set(...targetFocus)
      camera.position.set(
        targetFocus[0] + radius * 0.8,
        targetFocus[1] + radius * 0.5,
        targetFocus[2] + radius * 0.8
      )
      controlsRef.current.update()
      return
    }

    const dist = Math.max(radius * 1.8, 3.0)
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

    camera.near = Math.max(dist * 0.01, 0.01)
    camera.far = dist * 20
    camera.lookAt(0, 0, 0)
    camera.updateProjectionMatrix()

    if (controlsRef.current) {
      controlsRef.current.target.set(0, 0, 0)
      controlsRef.current.minDistance = dist * 0.05
      controlsRef.current.maxDistance = dist * 10
      controlsRef.current.update()
    }
  }, [radius, viewPreset, targetFocus, camera, controlsRef])

  return null
}

/* ── Main Three.js Viewer Export ───────────────────────────────────── */
export default function ThreeViewer({
  sparsePlyUrl,
  primaryObjectPlyUrl,
  densePlyUrl,
  meshPlyUrl,
  confidencePlyUrl,
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
  const [modelRadius, setModelRadius] = useState(8.0)

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
      <PerspectiveCamera makeDefault position={[6, 5, 6]} fov={45} />
      
      {/* Lights */}
      <ambientLight intensity={0.7} />
      <directionalLight position={[15, 25, 20]} intensity={1.3} />
      <directionalLight position={[-15, -10, -15]} intensity={0.5} />
      <pointLight position={[0, 10, 0]} intensity={0.6} />

      <CameraController
        radius={modelRadius}
        viewPreset={viewPreset}
        autoRotate={autoRotate}
        targetFocus={targetFocus}
        controlsRef={controlsRef}
      />

      {activeUrl && (
        <Suspense fallback={null}>
          {activeMode === 'mesh' ? (
            <SurfaceMesh
              url={activeUrl}
              wireframe={wireframe}
              onRadiusCalculated={setModelRadius}
            />
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
              onRadiusCalculated={setModelRadius}
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

      <gridHelper args={[30, 30, '#27273a', '#141424']} position={[0, -0.01, 0]} />
    </Canvas>
  )
}
