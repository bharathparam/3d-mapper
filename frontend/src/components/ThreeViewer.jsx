import React, { Suspense, useRef, useState, useEffect, useCallback } from 'react'
import { Canvas, useLoader, useThree, useFrame } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera } from '@react-three/drei'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js'
import * as THREE from 'three'

/* ── Point Cloud Component ────────────────────────────────────────── */
function PointCloud({ url, pointSize = 0.02 }) {
  const geometry = useLoader(PLYLoader, url)
  const pointsRef = useRef()

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
    <points ref={pointsRef}>
      <bufferGeometry attach="geometry" {...geometry} />
      <pointsMaterial
        attach="material"
        size={pointSize}
        vertexColors
        sizeAttenuation
        transparent
        opacity={0.95}
      />
    </points>
  )
}

/* ── Surface Mesh Component ────────────────────────────────────────── */
function SurfaceMesh({ url, wireframe = false, opacity = 1.0 }) {
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
        roughness={0.4}
        metalness={0.1}
        wireframe={wireframe}
        side={THREE.DoubleSide}
        transparent={opacity < 1.0}
        opacity={opacity}
      />
    </mesh>
  )
}

/* ── Camera Trajectory & Frustums ─────────────────────────────────── */
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

/* ── Camera Director (Spatial Presets & Focus) ────────────────────── */
function CameraController({ targetPosition, viewPreset, autoRotate, controlsRef }) {
  const { camera } = useThree()

  useEffect(() => {
    if (!viewPreset) return
    const dist = 6.0
    switch (viewPreset) {
      case 'top':
        camera.position.set(0, dist * 1.5, 0.001)
        camera.lookAt(0, 0, 0)
        break
      case 'front':
        camera.position.set(0, 0, dist)
        camera.lookAt(0, 0, 0)
        break
      case 'side':
        camera.position.set(dist, 0, 0)
        camera.lookAt(0, 0, 0)
        break
      case 'iso':
      default:
        camera.position.set(dist, dist * 0.8, dist)
        camera.lookAt(0, 0, 0)
        break
    }
    if (controlsRef.current) {
      controlsRef.current.target.set(0, 0, 0)
      controlsRef.current.update()
    }
  }, [viewPreset, camera, controlsRef])

  useEffect(() => {
    if (targetPosition && controlsRef.current) {
      controlsRef.current.target.set(...targetPosition)
      camera.position.set(
        targetPosition[0] + 2,
        targetPosition[1] + 1.5,
        targetPosition[2] + 2
      )
      controlsRef.current.update()
    }
  }, [targetPosition, camera, controlsRef])

  return null
}

/* ── Main Interactive 3D Viewer ───────────────────────────────────── */
export default function ThreeViewer({
  sparsePlyUrl,
  densePlyUrl,
  meshPlyUrl,
  confidencePlyUrl,
  cameraPoses,
  activeMode = 'sparse',
  pointSize = 0.025,
  wireframe = false,
  autoRotate = false,
  viewPreset = 'iso',
  targetFocus = null,
}) {
  const controlsRef = useRef()

  return (
    <Canvas
      gl={{ antialias: true, alpha: false, powerPreference: 'high-performance' }}
      style={{ background: '#07070d', width: '100%', height: '100%' }}
      id="three-canvas"
    >
      <PerspectiveCamera makeDefault position={[5, 4, 5]} fov={50} near={0.01} far={1000} />
      
      {/* Lighting Setup */}
      <ambientLight intensity={0.7} />
      <directionalLight position={[10, 20, 15]} intensity={1.2} />
      <directionalLight position={[-10, -10, -10]} intensity={0.4} />
      <pointLight position={[0, 10, 0]} intensity={0.5} />

      <CameraController
        targetPosition={targetFocus}
        viewPreset={viewPreset}
        autoRotate={autoRotate}
        controlsRef={controlsRef}
      />

      <Suspense fallback={null}>
        {activeMode === 'sparse' && sparsePlyUrl && (
          <PointCloud url={sparsePlyUrl} pointSize={pointSize} />
        )}

        {activeMode === 'dense' && (densePlyUrl || sparsePlyUrl) && (
          <PointCloud url={densePlyUrl || sparsePlyUrl} pointSize={pointSize * 0.85} />
        )}

        {activeMode === 'mesh' && (meshPlyUrl || sparsePlyUrl) && (
          meshPlyUrl ? (
            <SurfaceMesh url={meshPlyUrl} wireframe={wireframe} />
          ) : (
            <PointCloud url={sparsePlyUrl} pointSize={pointSize} />
          )
        )}

        {activeMode === 'confidence' && (confidencePlyUrl || sparsePlyUrl) && (
          <PointCloud url={confidencePlyUrl || sparsePlyUrl} pointSize={pointSize * 1.2} />
        )}

        {activeMode === 'cameras' && (
          <>
            {sparsePlyUrl && <PointCloud url={sparsePlyUrl} pointSize={pointSize * 0.7} />}
            <CameraTrajectory poses={cameraPoses} visible={true} />
          </>
        )}
      </Suspense>

      <OrbitControls
        ref={controlsRef}
        enableDamping
        dampingFactor={0.08}
        rotateSpeed={0.8}
        zoomSpeed={1.4}
        panSpeed={0.8}
        autoRotate={autoRotate}
        autoRotateSpeed={1.5}
      />

      <gridHelper args={[24, 24, '#27273a', '#141424']} position={[0, -0.01, 0]} />
    </Canvas>
  )
}
