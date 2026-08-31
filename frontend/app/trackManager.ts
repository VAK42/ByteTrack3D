import * as THREE from "three"
import type { TrackData } from "./wsClient"
const maxInstances = 128
const trailLength = 60
const palette = [
  0x22c55e,
  0x4ade80,
  0x86efac,
  0x10b981,
  0x34d399,
  0x6ee7b7,
  0x14b8a6,
  0x2dd4bf,
  0x5eead4,
  0xa7f3d0,
]
export function trackColor(trackId: number): number {
  return palette[trackId % palette.length]
}
class TrailLine {
  private geometry: THREE.BufferGeometry
  line: THREE.Line
  private maxLen: number
  constructor(color: number, maxLen = trailLength) {
    this.maxLen = maxLen
    this.geometry = new THREE.BufferGeometry()
    const positions = new Float32Array(maxLen * 3)
    this.geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3))
    this.geometry.setDrawRange(0, 0)
    const mat = new THREE.LineBasicMaterial({
      color,
      transparent: true,
      opacity: 0.75,
      linewidth: 2,
    })
    this.line = new THREE.Line(this.geometry, mat)
    this.line.frustumCulled = false
  }
  update(trailData: [number, number, number][]) {
    const attr = this.geometry.attributes.position as THREE.BufferAttribute
    const arr = attr.array as Float32Array
    const n = Math.min(trailData.length, this.maxLen)
    for (let i = 0; i < n; i++) {
      arr[i * 3] = trailData[i][0]
      arr[i * 3 + 1] = 0.05
      arr[i * 3 + 2] = trailData[i][2]
    }
    this.geometry.setDrawRange(0, n)
    attr.needsUpdate = true
  }
  dispose() {
    this.geometry.dispose()
    ;(this.line.material as THREE.Material).dispose()
  }
}
interface TrackEntry {
  instanceIdx: number
  currPos: THREE.Vector3
  targetPos: THREE.Vector3
  isLost: boolean
  trail: TrailLine
  color: number
  lastSeen: number
}
export class TrackManager {
  private scene: THREE.Scene
  private sphereMesh: THREE.InstancedMesh
  private pinMesh: THREE.InstancedMesh
  private ringMesh: THREE.InstancedMesh
  private dummy = new THREE.Object3D()
  private tracks: Map<number, TrackEntry> = new Map()
  private freeSlots: number[] = []
  private frameIdx = 0
  private trailsVisible = true
  constructor(scene: THREE.Scene) {
    this.scene = scene
    const sphereGeo = new THREE.SphereGeometry(0.24, 20, 14)
    const sphereMat = new THREE.MeshStandardMaterial({
      roughness: 0.2,
      metalness: 0.1,
    })
    this.sphereMesh = new THREE.InstancedMesh(sphereGeo, sphereMat, maxInstances)
    this.sphereMesh.frustumCulled = false
    this.sphereMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
    const pinGeo = new THREE.CylinderGeometry(0.02, 0.02, 1.70, 8)
    const pinMat = new THREE.MeshStandardMaterial({
      roughness: 0.3,
      metalness: 0.1,
      transparent: true,
      opacity: 0.70,
    })
    this.pinMesh = new THREE.InstancedMesh(pinGeo, pinMat, maxInstances)
    this.pinMesh.frustumCulled = false
    this.pinMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
    const ringGeo = new THREE.TorusGeometry(0.26, 0.02, 6, 16)
    ringGeo.rotateX(Math.PI / 2)
    const ringMat = new THREE.MeshBasicMaterial({
      transparent: true,
      opacity: 0.85,
    })
    this.ringMesh = new THREE.InstancedMesh(ringGeo, ringMat, maxInstances)
    this.ringMesh.frustumCulled = false
    this.ringMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
    const colorArr1 = new Float32Array(maxInstances * 3)
    const colorArr2 = new Float32Array(maxInstances * 3)
    const colorArr3 = new Float32Array(maxInstances * 3)
    this.sphereMesh.instanceColor = new THREE.InstancedBufferAttribute(colorArr1, 3)
    this.pinMesh.instanceColor = new THREE.InstancedBufferAttribute(colorArr2, 3)
    this.ringMesh.instanceColor = new THREE.InstancedBufferAttribute(colorArr3, 3)
    for (let i = 0; i < maxInstances; i++) {
      this.dummy.position.set(0, -100, 0)
      this.dummy.scale.set(0, 0, 0)
      this.dummy.updateMatrix()
      this.sphereMesh.setMatrixAt(i, this.dummy.matrix)
      this.pinMesh.setMatrixAt(i, this.dummy.matrix)
      this.ringMesh.setMatrixAt(i, this.dummy.matrix)
      this.freeSlots.push(i)
    }
    this.sphereMesh.instanceMatrix.needsUpdate = true
    this.pinMesh.instanceMatrix.needsUpdate = true
    this.ringMesh.instanceMatrix.needsUpdate = true
    scene.add(this.ringMesh)
    scene.add(this.pinMesh)
    scene.add(this.sphereMesh)
  }
  update(tracks: TrackData[], frame: number) {
    this.frameIdx = frame
    const seenIds = new Set<number>()
    for (const t of tracks) {
      const tid = t.trackId
      seenIds.add(tid)
      let entry = this.tracks.get(tid)
      const [x, y, z] = t.pos
      if (!entry) {
        const slot = this.freeSlots.pop()
        if (slot === undefined) continue
        const color = trackColor(tid)
        const trail = new TrailLine(color)
        trail.line.visible = this.trailsVisible
        this.scene.add(trail.line)
        const c = new THREE.Color(color)
        this.sphereMesh.setColorAt(slot, c)
        this.pinMesh.setColorAt(slot, c)
        this.ringMesh.setColorAt(slot, c)
        if (this.sphereMesh.instanceColor) this.sphereMesh.instanceColor.needsUpdate = true
        if (this.pinMesh.instanceColor) this.pinMesh.instanceColor.needsUpdate = true
        if (this.ringMesh.instanceColor) this.ringMesh.instanceColor.needsUpdate = true
        const posVec = new THREE.Vector3(x, 0, z)
        entry = {
          instanceIdx: slot,
          currPos: posVec.clone(),
          targetPos: posVec.clone(),
          isLost: t.lost,
          trail,
          color,
          lastSeen: frame,
        }
        this.tracks.set(tid, entry)
      }
      entry.lastSeen = frame
      entry.targetPos.set(x, 0, z)
      entry.isLost = t.lost
      entry.trail.update(t.trail)
    }
    for (const [tid, entry] of this.tracks) {
      if (!seenIds.has(tid)) {
        this.dummy.position.set(0, -100, 0)
        this.dummy.scale.set(0, 0, 0)
        this.dummy.updateMatrix()
        this.sphereMesh.setMatrixAt(entry.instanceIdx, this.dummy.matrix)
        this.pinMesh.setMatrixAt(entry.instanceIdx, this.dummy.matrix)
        this.ringMesh.setMatrixAt(entry.instanceIdx, this.dummy.matrix)
        this.sphereMesh.instanceMatrix.needsUpdate = true
        this.pinMesh.instanceMatrix.needsUpdate = true
        this.ringMesh.instanceMatrix.needsUpdate = true
        this.scene.remove(entry.trail.line)
        entry.trail.dispose()
        this.freeSlots.push(entry.instanceIdx)
        this.tracks.delete(tid)
      }
    }
  }
  step() {
    if (this.tracks.size === 0) return
    for (const entry of this.tracks.values()) {
      entry.currPos.lerp(entry.targetPos, 0.4)
      const scale = entry.isLost ? 0.6 : 1.0
      const x = entry.currPos.x
      const z = entry.currPos.z
      this.dummy.position.set(x, 1.70, z)
      this.dummy.scale.set(scale, scale, scale)
      this.dummy.updateMatrix()
      this.sphereMesh.setMatrixAt(entry.instanceIdx, this.dummy.matrix)
      this.dummy.position.set(x, 0.85, z)
      this.dummy.scale.set(scale, scale, scale)
      this.dummy.updateMatrix()
      this.pinMesh.setMatrixAt(entry.instanceIdx, this.dummy.matrix)
      this.dummy.position.set(x, 0.03, z)
      this.dummy.scale.set(scale, scale, scale)
      this.dummy.updateMatrix()
      this.ringMesh.setMatrixAt(entry.instanceIdx, this.dummy.matrix)
    }
    this.sphereMesh.instanceMatrix.needsUpdate = true
    this.pinMesh.instanceMatrix.needsUpdate = true
    this.ringMesh.instanceMatrix.needsUpdate = true
  }
  setTrailsVisible(visible: boolean) {
    this.trailsVisible = visible
    for (const entry of this.tracks.values()) {
      entry.trail.line.visible = visible
    }
  }
  getActiveTracks(): Array<{ id: number; color: number }> {
    return Array.from(this.tracks.entries()).map(([id, e]) => ({
      id,
      color: e.color,
    }))
  }
  dispose() {
    this.sphereMesh.geometry.dispose()
    ;(this.sphereMesh.material as THREE.Material).dispose()
    this.pinMesh.geometry.dispose()
    ;(this.pinMesh.material as THREE.Material).dispose()
    this.ringMesh.geometry.dispose()
    ;(this.ringMesh.material as THREE.Material).dispose()
    for (const entry of this.tracks.values()) {
      entry.trail.dispose()
    }
  }
}