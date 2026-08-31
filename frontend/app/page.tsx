"use client"
import dynamic from "next/dynamic"
import { WSClient, type FrameMessage, type ConnectionStatus } from "./wsClient"
import { useState, useEffect, useRef, useCallback } from "react"
import { trackColor } from "./trackManager"
const GestureController = dynamic(() => import("./GestureController"), { ssr: false })
const ThreeScene = dynamic(() => import("./ThreeScene"), { ssr: false })
type CameraView = "top" | "iso" | "side"
function hexToCSS(hex: number): string {
  return "#" + hex.toString(16).padStart(6, "0")
}
const statusLabel: Record<ConnectionStatus, string> = {
  connecting: "Connecting…",
  connected: "Live",
  disconnected: "Disconnected",
}
export default function Home() {
  const [status, setStatus] = useState<ConnectionStatus>("connecting")
  const [frame, setFrame] = useState<FrameMessage | null>(null)
  const [cameraView, setCameraView] = useState<CameraView>("iso")
  const [serverFps, setServerFps] = useState(0)
  const [frameIndex, setFrameIndex] = useState(0)
  const [simTime, setSimTime] = useState(0)
  const [showTrails, setShowTrails] = useState(true)
  const [gestureEnabled, setGestureEnabled] = useState(false)
  const clientRef = useRef<WSClient | null>(null)
  const incomingFramesRef = useRef(0)
  const lastFpsTimeRef = useRef(performance.now())
  const controlsApiRef = useRef<{
    orbitBy: (deltaTheta: number, deltaPhi: number) => void
    zoomBy: (deltaZoom: number) => void
  } | null>(null)
  const initClient = useCallback(() => {
    if (clientRef.current) {
      clientRef.current.destroy()
    }
    const client = new WSClient(
      (msg) => {
        setFrame(msg)
        setFrameIndex(msg.frame)
        setSimTime(msg.t)
        if (typeof msg.fps === "number" && msg.fps > 0) {
          setServerFps(Math.round(msg.fps))
        } else {
          incomingFramesRef.current += 1
          const now = performance.now()
          const elapsed = now - lastFpsTimeRef.current
          if (elapsed >= 500) {
            setServerFps(Math.round((incomingFramesRef.current / elapsed) * 1000.0))
            incomingFramesRef.current = 0
            lastFpsTimeRef.current = now
          }
        }
      },
      (s) => {
        setStatus(s)
        if (s === "connected") {
          incomingFramesRef.current = 0
          lastFpsTimeRef.current = performance.now()
        } else {
          setServerFps(0)
          incomingFramesRef.current = 0
          setFrame(null)
          setFrameIndex(0)
          setSimTime(0)
        }
      },
    )
    clientRef.current = client
  }, [])
  useEffect(() => {
    initClient()
    return () => {
      clientRef.current?.destroy()
    }
  }, [initClient])
  const handleReconnect = () => {
    if (clientRef.current && typeof clientRef.current.reconnect === "function") {
      clientRef.current.reconnect()
    } else {
      initClient()
    }
  }
  const handleStats = useCallback(() => {}, [])
  const tracks = frame?.tracks ?? []
  const fmtTime = (t: number) => {
    const m = Math.floor(t / 60).toString().padStart(2, "0")
    const s = Math.floor(t % 60).toString().padStart(2, "0")
    return `${m}:${s}`
  }
  const fmtPos = (pos: [number, number, number]) =>
    `(${pos[0].toFixed(1)}, ${pos[2].toFixed(1)})`
  return (
    <main className="w-screen h-screen flex overflow-hidden relative bg-black text-white font-normal">
      <div className="flex-1 relative bg-black">
        <ThreeScene
          latestFrame={frame}
          cameraView={cameraView}
          showTrails={showTrails}
          onStats={handleStats}
          onControlsReady={(api) => {
            controlsApiRef.current = api
          }}
        />
        <GestureController
          enabled={gestureEnabled}
          onOrbit={(dt, dp) => controlsApiRef.current?.orbitBy(dt, dp)}
          onZoom={(dz) => controlsApiRef.current?.zoomBy(dz)}
          onSwitchView={() =>
            setCameraView((prev) =>
              prev === "iso" ? "top" : prev === "top" ? "side" : "iso"
            )
          }
          onClose={() => setGestureEnabled(false)}
        />
        <div className="absolute top-4 left-4 z-30 flex flex-col gap-2.5">
          <div className="p-3 flex items-center gap-3 bg-black/90 border border-white/20 rounded shadow backdrop-blur-md">
            <div className="w-7 h-7 rounded bg-green-950 text-white flex items-center justify-center text-xs font-normal">
              3D
            </div>
            <div>
              <div className="text-xs font-normal text-white">
                VAK
              </div>
              <div className="text-[10px] text-white/70 font-normal">
                YOLO26 · ByteTrack
              </div>
            </div>
          </div>
          <div className="p-2.5 bg-black/90 border border-white/20 rounded shadow backdrop-blur-md flex flex-col gap-1.5 w-48 text-xs text-white">
            <div className="text-[10px] text-white/60 font-normal border-b border-white/10 pb-1 flex items-center justify-between">
              <span>Hand Gestures</span>
              <span className="w-1.5 h-1.5 rounded bg-emerald-400" />
            </div>
            <div className="flex items-center justify-between text-[11px]">
              <span className="text-white/80">5F</span>
              <span className="text-emerald-300 font-mono text-[10px]">Orbit</span>
            </div>
            <div className="flex items-center justify-between text-[11px]">
              <span className="text-white/80">3F</span>
              <span className="text-emerald-300 font-mono text-[10px]">Zoom In</span>
            </div>
            <div className="flex items-center justify-between text-[11px]">
              <span className="text-white/80">Closed Fist</span>
              <span className="text-emerald-300 font-mono text-[10px]">Zoom Out</span>
            </div>
            <div className="flex items-center justify-between text-[11px]">
              <span className="text-white/80">2F</span>
              <span className="text-emerald-300 font-mono text-[10px]">Switch View</span>
            </div>
          </div>
        </div>
        <div className="absolute top-4 right-4 p-2 flex gap-4 items-center bg-black/90 border border-white/20 rounded shadow backdrop-blur-md">
          <div className="text-center">
            <div className="text-sm font-normal text-white">
              {status === "connected" ? serverFps : 0}
            </div>
            <div className="text-[10px] text-white/70 font-normal">FPS</div>
          </div>
          <div className="w-px h-6 bg-white/20" />
          <div className="text-center">
            <div className="text-sm font-normal text-white font-mono">
              {frameIndex.toString().padStart(5, "0")}
            </div>
            <div className="text-[10px] text-white/70 font-normal">Frame</div>
          </div>
          <div className="w-px h-6 bg-white/20" />
          <div className="text-center">
            <div className="text-sm font-normal text-white">{fmtTime(simTime)}</div>
            <div className="text-[10px] text-white/70 font-normal">Sim Time</div>
          </div>
        </div>
        <div className="absolute bottom-5 left-1/2 -translate-x-1/2 p-2 flex gap-2 items-center bg-black/90 border border-white/20 rounded shadow backdrop-blur-md">
          <span className="text-[10px] text-white/70 font-normal mr-1">
            View
          </span>
          {(["iso", "top", "side"] as CameraView[]).map((v) => (
            <button
              key={v}
              id={`cam-btn-${v}`}
              className={`px-3 py-1 text-xs font-normal rounded border cursor-pointer transition-colors ${
                cameraView === v
                  ? "bg-green-950 text-white border-green-800"
                  : "bg-black/70 text-white border-white/20 hover:border-white/50"
              }`}
              onClick={() => setCameraView(v)}
            >
              {v === "iso" ? "Isometric" : v === "top" ? "Top-Down" : "Side"}
            </button>
          ))}
          <div className="w-px h-4 bg-white/20 mx-1" />
          <button
            id="gesture-toggle-btn"
            className={`px-3 py-1 text-xs font-normal rounded border cursor-pointer transition-colors flex items-center gap-1.5 ${
              gestureEnabled
                ? "bg-green-950 text-emerald-300 border-green-700"
                : "bg-black/70 text-white border-white/20 hover:border-white/50"
            }`}
            onClick={() => setGestureEnabled((prev) => !prev)}
          >
            <span
              className={`w-1.5 h-1.5 rounded ${
                gestureEnabled ? "bg-emerald-400 animate-ping" : "bg-white/40"
              }`}
            />
            Hand Control
          </button>
        </div>
      </div>
      <aside className="w-64 h-full border-l border-white/20 bg-black/90 flex flex-col p-4 gap-3 overflow-y-auto font-normal shadow backdrop-blur-md">
        <div className="flex items-center justify-between pb-1 border-b border-white/10">
          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded ${
                status === "connected"
                  ? "bg-emerald-400"
                  : status === "connecting"
                  ? "bg-emerald-400/50 animate-pulse"
                  : "bg-white/20"
              }`}
            />
            <span className="text-xs font-normal text-white">
              {statusLabel[status]}
            </span>
          </div>
          <button
            id="reconnect-btn"
            className="text-[10px] px-2 py-0.5 border border-white/20 rounded hover:border-white/50 text-white/80 hover:text-white transition-colors cursor-pointer"
            onClick={handleReconnect}
            title="Reconnect WebSocket"
          >
            Reconnect
          </button>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <div className="p-2 border border-white/20 rounded bg-black/70">
            <div className="text-base font-normal text-white">{tracks.length}</div>
            <div className="text-[10px] text-white/70 font-normal">Active Tracks</div>
          </div>
          <div className="p-2 border border-white/20 rounded bg-black/70">
            <div className="text-base font-normal text-white">
              {tracks.filter((t) => !t.lost).length}
            </div>
            <div className="text-[10px] text-white/70 font-normal">Confirmed</div>
          </div>
          <div className="p-2 border border-white/20 rounded bg-black/70">
            <div className="text-base font-normal text-white">
              {tracks.length > 0
                ? (
                    tracks.reduce((acc, t) => acc + t.hits, 0) / tracks.length
                  ).toFixed(1)
                : "0.0"}
            </div>
            <div className="text-[10px] text-white/70 font-normal">Avg Hits</div>
          </div>
          <div className="p-2 border border-white/20 rounded bg-black/70">
            <div className="text-base font-normal text-white">
              {frame?.cameras ? frame.cameras.length : 0}
            </div>
            <div className="text-[10px] text-white/70 font-normal">Cameras</div>
          </div>
        </div>
        <div className="flex items-center justify-between text-xs text-white">
          <label className="flex items-center gap-1.5 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={showTrails}
              onChange={(e) => setShowTrails(e.target.checked)}
              className="accent-green-950 rounded cursor-pointer"
            />
            Show Trails
          </label>
        </div>
        <div className="flex flex-col gap-1.5 flex-1 min-h-0">
          <div className="text-[10px] text-white/70 font-normal">Active Tracks</div>
          <div className="flex flex-col gap-1 overflow-y-auto flex-1 pr-1">
            {tracks.length === 0 ? (
              <div className="text-xs text-white/40 text-center py-6">
                No Tracks Detected
              </div>
            ) : (
              tracks.map((t) => {
                const color = trackColor(t.trackId)
                return (
                  <div
                    key={t.trackId}
                    className="p-1.5 border border-white/10 rounded bg-black/60 flex items-center justify-between text-xs"
                  >
                    <div className="flex items-center gap-2">
                      <span
                        className="w-2 h-2 rounded shrink-0"
                        style={{ backgroundColor: hexToCSS(color) }}
                      />
                      <span className="text-white">#{t.trackId}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] text-white/70">
                        {fmtPos(t.pos)}
                      </span>
                      {t.lost && (
                        <span className="text-[9px] text-white/50 border border-white/20 px-1 py-0.5 rounded">
                          Lost
                        </span>
                      )}
                    </div>
                  </div>
                )
              })
            )}
          </div>
        </div>
        <div className="pt-2 border-t border-white/20 text-[10px] text-white/50 flex justify-between font-normal">
          <span>WildTrack</span>
          <span>RX 6550M</span>
        </div>
      </aside>
    </main>
  )
}