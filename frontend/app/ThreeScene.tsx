"use client"
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js"
import { TrackManager } from "./trackManager"
import { useEffect, useRef } from "react"
import type { FrameMessage } from "./wsClient"
import * as THREE from "three"
const bounds = { xMin: -5.0, xMax: 13.0, zMin: -12.0, zMax: 26.0, yMax: 8.5 }
function createLabelSprite(text: string, bgColor: string = "rgba(0,0,0,0.85)", textColor: string = "#ffffff"): THREE.Sprite {
  const canvas = document.createElement("canvas")
  canvas.width = 180
  canvas.height = 64
  const ctx = canvas.getContext("2d")!
  ctx.fillStyle = bgColor
  ctx.beginPath()
  ctx.roundRect(10, 10, 160, 44, 8)
  ctx.fill()
  ctx.strokeStyle = "rgba(255,255,255,0.6)"
  ctx.lineWidth = 2
  ctx.stroke()
  ctx.fillStyle = textColor
  ctx.font = "22px sans-serif"
  ctx.textAlign = "center"
  ctx.textBaseline = "middle"
  ctx.fillText(text, 90, 32)
  const texture = new THREE.CanvasTexture(canvas)
  const mat = new THREE.SpriteMaterial({ map: texture, transparent: true })
  const sprite = new THREE.Sprite(mat)
  sprite.scale.set(1.6, 0.57, 1.0)
  return sprite
}
function createTickSprite(text: string): THREE.Sprite {
  const canvas = document.createElement("canvas")
  canvas.width = 96
  canvas.height = 48
  const ctx = canvas.getContext("2d")!
  ctx.fillStyle = "#ffffff"
  ctx.font = "18px sans-serif"
  ctx.textAlign = "center"
  ctx.textBaseline = "middle"
  ctx.fillText(text, 48, 24)
  const texture = new THREE.CanvasTexture(canvas)
  const mat = new THREE.SpriteMaterial({ map: texture, transparent: true })
  const sprite = new THREE.Sprite(mat)
  sprite.scale.set(0.9, 0.45, 1.0)
  return sprite
}
interface Props {
  latestFrame: FrameMessage | null
  cameraView: "top" | "iso" | "side"
  showTrails?: boolean
  onStats: (stats: { fps: number; trackCount: number }) => void
  onControlsReady?: (api: {
    orbitBy: (deltaTheta: number, deltaPhi: number) => void
    zoomBy: (deltaZoom: number) => void
  }) => void
}
export default function ThreeScene({
  latestFrame,
  cameraView,
  showTrails = true,
  onStats,
  onControlsReady,
}: Props) {
  const mountRef = useRef<HTMLDivElement>(null)
  const camerasGroupRef = useRef<THREE.Group | null>(null)
  const starsRef = useRef<THREE.Points | null>(null)
  const knownCamIdsRef = useRef<string>("")
  const sceneRef = useRef<{
    renderer: THREE.WebGLRenderer
    scene: THREE.Scene
    camera: THREE.PerspectiveCamera
    controls: OrbitControls
    manager: TrackManager
    animId: number
    lastTime: number
    frameCount: number
  } | null>(null)
  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return
    let renderer: THREE.WebGLRenderer
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    } catch {
      renderer = new THREE.WebGLRenderer({ antialias: false })
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(mount.clientWidth, mount.clientHeight)
    renderer.setClearColor(0x000000, 1)
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFShadowMap
    mount.appendChild(renderer.domElement)
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x000000)
    const starCount = 3500
    const starGeo = new THREE.BufferGeometry()
    const starPositions = new Float32Array(starCount * 3)
    const starColors = new Float32Array(starCount * 3)
    for (let i = 0; i < starCount; i++) {
      const radius = 70 + Math.random() * 90
      const theta = 2 * Math.PI * Math.random()
      const phi = Math.acos(2 * Math.random() - 1)
      starPositions[i * 3] = radius * Math.sin(phi) * Math.cos(theta)
      starPositions[i * 3 + 1] = radius * Math.cos(phi)
      starPositions[i * 3 + 2] = radius * Math.sin(phi) * Math.sin(theta)
      const brightness = 0.5 + Math.random() * 0.5
      starColors[i * 3] = brightness
      starColors[i * 3 + 1] = brightness
      starColors[i * 3 + 2] = brightness + Math.random() * 0.15
    }
    starGeo.setAttribute("position", new THREE.BufferAttribute(starPositions, 3))
    starGeo.setAttribute("color", new THREE.BufferAttribute(starColors, 3))
    const starMat = new THREE.PointsMaterial({
      size: 0.35,
      vertexColors: true,
      transparent: true,
      opacity: 0.95,
    })
    const stars = new THREE.Points(starGeo, starMat)
    scene.add(stars)
    starsRef.current = stars
    const camerasGroup = new THREE.Group()
    scene.add(camerasGroup)
    camerasGroupRef.current = camerasGroup
    const camera = new THREE.PerspectiveCamera(
      55,
      mount.clientWidth / mount.clientHeight,
      0.1,
      500
    )
    camera.position.set(4, 22, 34)
    camera.lookAt(4, 1.5, 7)
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.minDistance = 3
    controls.maxDistance = 120
    controls.target.set(4, 1.5, 7)
    const { xMin, xMax, zMin, zMax, yMax } = bounds
    const w = xMax - xMin
    const d = zMax - zMin
    const cx = (xMin + xMax) / 2
    const cz = (zMin + zMax) / 2
    const floorGeo = new THREE.PlaneGeometry(w, d)
    const floorMat = new THREE.MeshStandardMaterial({
      color: 0x000000,
      transparent: true,
      opacity: 0.85,
      roughness: 0.8,
      metalness: 0.2,
    })
    const floor = new THREE.Mesh(floorGeo, floorMat)
    floor.rotation.x = -Math.PI / 2
    floor.position.set(cx, 0, cz)
    floor.receiveShadow = true
    scene.add(floor)
    const gridLines = new THREE.Group()
    const gridLineMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.35 })
    const mainLineMat = new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 2 })
    for (let x = Math.ceil(xMin); x <= Math.floor(xMax); x++) {
      const geo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(x, 0.01, zMin),
        new THREE.Vector3(x, 0.01, zMax),
      ])
      const line = new THREE.Line(geo, x === 0 ? mainLineMat : gridLineMat)
      gridLines.add(line)
    }
    for (let z = Math.ceil(zMin); z <= Math.floor(zMax); z++) {
      const geo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(xMin, 0.01, z),
        new THREE.Vector3(xMax, 0.01, z),
      ])
      const line = new THREE.Line(geo, z === 0 ? mainLineMat : gridLineMat)
      gridLines.add(line)
    }
    scene.add(gridLines)
    const oxyzGroup = new THREE.Group()
    const originGeo = new THREE.SphereGeometry(0.25, 16, 12)
    const originMat = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 0.3 })
    const originMesh = new THREE.Mesh(originGeo, originMat)
    originMesh.position.set(0, 0.1, 0)
    oxyzGroup.add(originMesh)
    const originLabel = createLabelSprite("O (0,0,0)")
    originLabel.position.set(0, 0.75, 0)
    oxyzGroup.add(originLabel)
    const xLen = xMax - xMin
    const xPoleGeo = new THREE.CylinderGeometry(0.045, 0.045, xLen, 8)
    const xPoleMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.2 })
    const xPole = new THREE.Mesh(xPoleGeo, xPoleMat)
    xPole.rotation.z = -Math.PI / 2
    xPole.position.set((xMin + xMax) / 2, 0.05, 0)
    oxyzGroup.add(xPole)
    const headGeo = new THREE.ConeGeometry(0.18, 0.50, 12)
    const xHead = new THREE.Mesh(headGeo, xPoleMat)
    xHead.rotation.z = -Math.PI / 2
    xHead.position.set(xMax + 0.25, 0.05, 0)
    oxyzGroup.add(xHead)
    const xLabel = createLabelSprite("+X (m)")
    xLabel.position.set(xMax + 1.2, 0.3, 0)
    oxyzGroup.add(xLabel)
    const zLen = zMax - zMin
    const zPoleGeo = new THREE.CylinderGeometry(0.045, 0.045, zLen, 8)
    const zPoleMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.2 })
    const zPole = new THREE.Mesh(zPoleGeo, zPoleMat)
    zPole.rotation.x = Math.PI / 2
    zPole.position.set(0, 0.05, (zMin + zMax) / 2)
    oxyzGroup.add(zPole)
    const zHead = new THREE.Mesh(headGeo, zPoleMat)
    zHead.rotation.x = Math.PI / 2
    zHead.position.set(0, 0.05, zMax + 0.25)
    oxyzGroup.add(zHead)
    const zLabel = createLabelSprite("+Z (m)")
    zLabel.position.set(0, 0.3, zMax + 1.2)
    oxyzGroup.add(zLabel)
    const yPoleGeo = new THREE.CylinderGeometry(0.055, 0.055, yMax, 8)
    const yPoleMat = new THREE.MeshStandardMaterial({ color: 0x4ade80, roughness: 0.2, emissive: 0x4ade80, emissiveIntensity: 0.3 })
    const yPole = new THREE.Mesh(yPoleGeo, yPoleMat)
    yPole.position.set(0, yMax / 2, 0)
    oxyzGroup.add(yPole)
    const yHead = new THREE.Mesh(headGeo, yPoleMat)
    yHead.position.set(0, yMax + 0.25, 0)
    oxyzGroup.add(yHead)
    const yLabel = createLabelSprite("+Y (m)")
    yLabel.position.set(0, yMax + 0.9, 0)
    oxyzGroup.add(yLabel)
    for (let y = 1; y <= Math.floor(yMax); y++) {
      const ringGeo = new THREE.TorusGeometry(0.12, 0.02, 8, 16)
      const ringMat = new THREE.MeshBasicMaterial({ color: 0x86efac })
      const ring = new THREE.Mesh(ringGeo, ringMat)
      ring.rotation.x = Math.PI / 2
      ring.position.set(0, y, 0)
      oxyzGroup.add(ring)
      if (y % 2 === 0) {
        const tick = createTickSprite(`${y}m`)
        tick.position.set(0.4, y, 0)
        oxyzGroup.add(tick)
      }
    }
    scene.add(oxyzGroup)
    for (let x = -4; x <= 12; x += 4) {
      if (x === 0) continue
      const tick = createTickSprite(`${x}m`)
      tick.position.set(x, 0.12, -0.6)
      scene.add(tick)
    }
    for (let z = -8; z <= 24; z += 8) {
      if (z === 0) continue
      const tick = createTickSprite(`${z}m`)
      tick.position.set(-0.7, 0.12, z)
      scene.add(tick)
    }
    const ambient = new THREE.AmbientLight(0xffffff, 2.5)
    scene.add(ambient)
    const sun = new THREE.DirectionalLight(0xffffff, 2.5)
    sun.position.set(10, 30, 10)
    sun.castShadow = true
    sun.shadow.mapSize.set(1024, 1024)
    scene.add(sun)
    const fill = new THREE.PointLight(0x86efac, 1.2, 40)
    fill.position.set(-5, 8, 15)
    scene.add(fill)
    const manager = new TrackManager(scene)
    const ro = new ResizeObserver(() => {
      if (!mount) return
      const w = mount.clientWidth
      const h = mount.clientHeight
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
    })
    ro.observe(mount)
    let lastTime = performance.now()
    let frameCount = 0
    let animId = 0
    const animate = () => {
      animId = requestAnimationFrame(animate)
      if (starsRef.current) {
        starsRef.current.rotation.y += 0.0003
      }
      controls.update()
      manager.step()
      renderer.render(scene, camera)
      frameCount++
      const now = performance.now()
      if (now - lastTime >= 1000) {
        onStats({ fps: frameCount, trackCount: manager.getActiveTracks().length })
        frameCount = 0
        lastTime = now
      }
    }
    animate()
    sceneRef.current = { renderer, scene, camera, controls, manager, animId, lastTime, frameCount }
    if (onControlsReady) {
      onControlsReady({
        orbitBy: (deltaTheta: number, deltaPhi: number) => {
          if (!sceneRef.current) return
          const { camera: cam, controls: ctrl } = sceneRef.current
          const offset = cam.position.clone().sub(ctrl.target)
          const spherical = new THREE.Spherical().setFromVector3(offset)
          spherical.theta += deltaTheta
          spherical.phi = Math.max(0.1, Math.min(Math.PI / 2 - 0.05, spherical.phi + deltaPhi))
          offset.setFromSpherical(spherical)
          cam.position.copy(ctrl.target).add(offset)
          cam.lookAt(ctrl.target)
          ctrl.update()
        },
        zoomBy: (deltaZoom: number) => {
          if (!sceneRef.current) return
          const { camera: cam, controls: ctrl } = sceneRef.current
          const offset = cam.position.clone().sub(ctrl.target)
          const dist = offset.length()
          const newDist = Math.max(5, Math.min(90, dist - deltaZoom))
          offset.setLength(newDist)
          cam.position.copy(ctrl.target).add(offset)
          ctrl.update()
        },
      })
    }
    return () => {
      cancelAnimationFrame(animId)
      ro.disconnect()
      manager.dispose()
      renderer.dispose()
      mount.removeChild(renderer.domElement)
      sceneRef.current = null
      camerasGroupRef.current = null
      starsRef.current = null
    }
  }, [])
  useEffect(() => {
    if (!latestFrame || !sceneRef.current) return
    const { manager } = sceneRef.current
    manager.update(latestFrame.tracks, latestFrame.frame)
    if (latestFrame.cameras && camerasGroupRef.current) {
      const camKey = latestFrame.cameras.map((c) => `${c.id}:${c.pos.join(",")}`).join("|")
      if (camKey !== knownCamIdsRef.current) {
        knownCamIdsRef.current = camKey
        const group = camerasGroupRef.current
        while (group.children.length > 0) {
          const child = group.children[0]
          group.remove(child)
        }
        const camBodyGeo = new THREE.BoxGeometry(0.40, 0.26, 0.55)
        const camBodyMat = new THREE.MeshStandardMaterial({ color: 0x052e16, roughness: 0.3, metalness: 0.2 })
        const camLensGeo = new THREE.CylinderGeometry(0.11, 0.11, 0.20, 12)
        const camLensMat = new THREE.MeshStandardMaterial({ color: 0x86efac, roughness: 0.1, metalness: 0.8 })
        for (const cInfo of latestFrame.cameras) {
          const camGroup = new THREE.Group()
          const poleHeight = cInfo.pos[1]
          const poleGeo = new THREE.CylinderGeometry(0.04, 0.04, poleHeight, 8)
          const poleMat = new THREE.MeshStandardMaterial({ color: 0x86efac, roughness: 0.5 })
          const pole = new THREE.Mesh(poleGeo, poleMat)
          pole.position.set(cInfo.pos[0], poleHeight / 2, cInfo.pos[2])
          group.add(pole)
          const body = new THREE.Mesh(camBodyGeo, camBodyMat)
          const lens = new THREE.Mesh(camLensGeo, camLensMat)
          lens.rotation.x = Math.PI / 2
          lens.position.z = 0.32
          camGroup.add(body)
          camGroup.add(lens)
          camGroup.position.set(...cInfo.pos)
          const targetVec = new THREE.Vector3(...cInfo.target)
          camGroup.lookAt(targetVec)
          group.add(camGroup)
          const label = createLabelSprite(cInfo.label)
          label.position.set(cInfo.pos[0], cInfo.pos[1] + 0.65, cInfo.pos[2])
          group.add(label)
        }
      }
    }
  }, [latestFrame])
  useEffect(() => {
    if (!sceneRef.current) return
    sceneRef.current.manager.setTrailsVisible(showTrails)
  }, [showTrails])
  useEffect(() => {
    if (!sceneRef.current) return
    const { camera, controls } = sceneRef.current
    const cx = (bounds.xMin + bounds.xMax) / 2
    const cz = (bounds.zMin + bounds.zMax) / 2
    if (cameraView === "top") {
      camera.position.set(cx, 38, cz)
      controls.target.set(cx, 0, cz)
    } else if (cameraView === "iso") {
      camera.position.set(cx - 18, 24, cz + 26)
      controls.target.set(cx, 2.0, cz)
    } else {
      camera.position.set(cx + 25, 12, cz)
      controls.target.set(cx, 2.0, cz)
    }
    controls.update()
  }, [cameraView])
  return (
    <div
      ref={mountRef}
      className="w-full h-full relative bg-black"
    />
  )
}