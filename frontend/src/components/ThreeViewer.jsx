import React, { Suspense, useRef, useState, useEffect } from 'react'
import { Canvas, useLoader, useThree } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera } from '@react-three/drei'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import * as THREE from 'three'

/* ── Generate Soft Circular Gaussian Splat Texture ─────────────────── */
function createSplatTexture() {
  const size = 64
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')

  const gradient = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2)
  gradient.addColorStop(0, 'rgba(255, 255, 255, 1.0)')
  gradient.addColorStop(0.5, 'rgba(255, 255, 255, 0.85)')
  gradient.addColorStop(0.85, 'rgba(255, 255, 255, 0.25)')
  gradient.addColorStop(1, 'rgba(255, 255, 255, 0.0)')

  ctx.fillStyle = gradient
  ctx.fillRect(0, 0, size, size)

  const texture = new THREE.CanvasTexture(canvas)
  texture.needsUpdate = true
  return texture
}

const splatTexture = createSplatTexture()

/* ── Spatial Density Centering Helper ───────────────────────────────── */
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

  geometry.translate(-medX, -medY, -medZ)

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

/* ── Photorealistic Textured GLTF Model ────────────────────────────── */
function TexturedGLTFModel({ url, wireframe = false, onRadiusCalculated }) {
  const gltf = useLoader(GLTFLoader, url)
  const modelRef = useRef()

  useEffect(() => {
    if (gltf && gltf.scene) {
      const box = new THREE.Box3().setFromObject(gltf.scene)
      const center = box.getCenter(new THREE.Vector3())
      const size = box.getSize(new THREE.Vector3())
      const radius = Math.max(size.x, size.y, size.z) / 2.0

      gltf.scene.position.x = -center.x
      gltf.scene.position.y = -center.y
      gltf.scene.position.z = -center.z

      gltf.scene.traverse((child) => {
        if (child.isMesh) {
          child.material.wireframe = wireframe
          child.material.roughness = 0.35
          child.material.metalness = 0.05
          child.material.side = THREE.DoubleSide
        }
      })

      if (onRadiusCalculated) onRadiusCalculated(Math.max(radius, 3.0))
    }
  }, [gltf, wireframe, onRadiusCalculated])

  return <primitive ref={modelRef} object={gltf.scene} />
}

/* ── Photorealistic Neural Point Splat Component ───────────────────── */
function NeuralPointCloud({ url, pointSize = 0.025, onRadiusCalculated, isMLDense = false }) {
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
        size={isMLDense ? pointSize * 0.8 : pointSize}
        map={splatTexture}
        vertexColors
        sizeAttenuation
        transparent
        alphaTest={0.01}
        opacity={isMLDense ? 0.98 : 0.94}
        blending={THREE.NormalBlending}
      />
    </points>
  )
}

/* ── High-Fidelity Surface Mesh ────────────────────────────────────── */
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
        roughness={0.25}
        metalness={0.05}
        wireframe={wireframe}
        side={THREE.DoubleSide}
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
    camera.far = dist * 25
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

/* ── Canvas Capture Handler ───────────────────────────────────────── */
function CanvasCaptureBridge({ captureTrigger, onCaptureDone }) {
  const { gl, scene, camera } = useThree()

  useEffect(() => {
    if (captureTrigger > 0) {
      gl.render(scene, camera)
      const dataUrl = gl.domElement.toDataURL('image/png')
      const link = document.createElement('a')
      link.download = `drone3d-photorealistic-${Date.now()}.png`
      link.href = dataUrl
      link.click()
      if (onCaptureDone) onCaptureDone()
    }
  }, [captureTrigger, gl, scene, camera, onCaptureDone])

  return null
}

/* ── Main Three.js Viewer Export ───────────────────────────────────── */
export default function ThreeViewer({
  texturedGlbUrl,
  sparsePlyUrl,
  primaryObjectPlyUrl,
  densePlyUrl,
  meshPlyUrl,
  mlDensePlyUrl,
  mlMeshPlyUrl,
  confidencePlyUrl,
  cameraPoses,
  activeMode = 'textured',
  focusTargetOnly = true,
  pointSize = 0.024,
  wireframe = false,
  autoRotate = false,
  viewPreset = 'iso',
  targetFocus = null,
  captureTrigger = 0,
  onCaptureDone = null,
}) {
  const controlsRef = useRef()
  const [modelRadius, setModelRadius] = useState(8.0)

  const isTexturedGLB = activeMode === 'textured' && texturedGlbUrl
  let activePlyUrl = sparsePlyUrl

  if (activeMode === 'mldense' && (mlDensePlyUrl || densePlyUrl)) {
    activePlyUrl = mlDensePlyUrl || densePlyUrl
  } else if (activeMode === 'mlmesh' && (mlMeshPlyUrl || meshPlyUrl)) {
    activePlyUrl = mlMeshPlyUrl || meshPlyUrl
  } else if (activeMode === 'dense' && densePlyUrl) {
    activePlyUrl = densePlyUrl
  } else if (activeMode === 'mesh' && meshPlyUrl) {
    activePlyUrl = meshPlyUrl
  } else if (activeMode === 'confidence' && confidencePlyUrl) {
    activePlyUrl = confidencePlyUrl
  } else if (focusTargetOnly && primaryObjectPlyUrl) {
    activePlyUrl = primaryObjectPlyUrl
  }

  const isMeshMode = activeMode === 'mesh' || activeMode === 'mlmesh'
  const isMLDense = activeMode === 'mldense'

  return (
    <Canvas
      gl={{
        antialias: true,
        alpha: false,
        powerPreference: 'high-performance',
        preserveDrawingBuffer: true,
      }}
      style={{ background: '#07070d', width: '100%', height: '100%' }}
      id="three-canvas"
    >
      <PerspectiveCamera makeDefault position={[6, 5, 6]} fov={45} />
      
      {/* Studio Lighting Setup */}
      <ambientLight intensity={0.8} />
      <directionalLight position={[15, 25, 20]} intensity={1.5} />
      <directionalLight position={[-15, -10, -15]} intensity={0.6} />
      <directionalLight position={[0, -15, 10]} intensity={0.3} />
      <pointLight position={[0, 12, 0]} intensity={0.7} />

      <CameraController
        radius={modelRadius}
        viewPreset={viewPreset}
        autoRotate={autoRotate}
        targetFocus={targetFocus}
        controlsRef={controlsRef}
      />

      <CanvasCaptureBridge
        captureTrigger={captureTrigger}
        onCaptureDone={onCaptureDone}
      />

      <Suspense fallback={null}>
        {isTexturedGLB ? (
          <TexturedGLTFModel
            url={texturedGlbUrl}
            wireframe={wireframe}
            onRadiusCalculated={setModelRadius}
          />
        ) : activePlyUrl ? (
          isMeshMode ? (
            <SurfaceMesh
              url={activePlyUrl}
              wireframe={wireframe}
              onRadiusCalculated={setModelRadius}
            />
          ) : (
            <NeuralPointCloud
              url={activePlyUrl}
              pointSize={pointSize}
              isMLDense={isMLDense}
              onRadiusCalculated={setModelRadius}
            />
          )
        ) : null}

        {activeMode === 'cameras' && (
          <CameraTrajectory poses={cameraPoses} visible={true} />
        )}
      </Suspense>

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
