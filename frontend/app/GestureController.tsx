"use client"
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision"
import { useEffect, useRef, useState } from "react"
if (typeof window !== "undefined") {
  const origWarn = console.warn
  console.warn = (...args: unknown[]) => {
    const msg = typeof args[0] === "string" ? args[0] : ""
    if (
      msg.includes("gl_context") ||
      msg.includes("NORM_RECT") ||
      msg.includes("landmark_projection")
    ) {
      return
    }
    origWarn(...args)
  }
}
interface GestureControllerProps {
  enabled: boolean
  onOrbit: (deltaTheta: number, deltaPhi: number) => void
  onZoom: (deltaZoom: number) => void
  onSwitchView: () => void
  onClose: () => void
}
export default function GestureController({
  enabled,
  onOrbit,
  onZoom,
  onSwitchView,
  onClose,
}: GestureControllerProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const landmarkerRef = useRef<HandLandmarker | null>(null)
  const animRef = useRef<number>(0)
  const lastPalmPosRef = useRef<{ x: number; y: number } | null>(null)
  const zoomHoldCountRef = useRef<number>(0)
  const twoFingerStartTimeRef = useRef<number | null>(null)
  const lastSwitchTimeRef = useRef<number>(0)
  const [gestureState, setGestureState] = useState<string>("Initializing...")
  const [isReady, setIsReady] = useState(false)
  useEffect(() => {
    if (!enabled) return
    let active = true
    let mediaStream: MediaStream | null = null
    const initVision = async () => {
      try {
        setGestureState("Loading AI Model...")
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        )
        if (!active) return
        const landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numHands: 1,
          minHandDetectionConfidence: 0.75,
          minHandPresenceConfidence: 0.75,
          minTrackingConfidence: 0.75,
        })
        if (!active) return
        landmarkerRef.current = landmarker
        mediaStream = await navigator.mediaDevices.getUserMedia({
          video: { width: 320, height: 240, facingMode: "user" },
        })
        if (!active) return
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream
          await videoRef.current.play()
          setIsReady(true)
          setGestureState("Hand Ready")
        }
      } catch (e) {
        if (active) {
          setGestureState("Camera Not Available")
        }
      }
    }
    initVision()
    return () => {
      active = false
      if (animRef.current) cancelAnimationFrame(animRef.current)
      if (mediaStream) {
        mediaStream.getTracks().forEach((track) => track.stop())
      }
      if (landmarkerRef.current) {
        landmarkerRef.current.close()
        landmarkerRef.current = null
      }
    }
  }, [enabled])
  useEffect(() => {
    if (!enabled || !isReady) return
    let lastVideoTime = -1
    const loop = () => {
      animRef.current = requestAnimationFrame(loop)
      const video = videoRef.current
      const canvas = canvasRef.current
      const landmarker = landmarkerRef.current
      if (!video || !canvas || !landmarker || video.readyState < 2) return
      if (video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime
        const ctx = canvas.getContext("2d")
        if (!ctx) return
        const vw = video.videoWidth || 320
        const vh = video.videoHeight || 240
        const minDim = Math.min(vw, vh)
        const sx = (vw - minDim) / 2
        const sy = (vh - minDim) / 2
        ctx.clearRect(0, 0, canvas.width, canvas.height)
        ctx.drawImage(video, sx, sy, minDim, minDim, 0, 0, canvas.width, canvas.height)
        const results = landmarker.detectForVideo(canvas, performance.now())
        if (
          results.landmarks &&
          results.landmarks.length > 0 &&
          results.handedness &&
          results.handedness.length > 0 &&
          results.handedness[0][0]?.score >= 0.70
        ) {
          const lms = results.landmarks[0]
          const wrist = lms[0]
          const middleTip = lms[12]
          const middleMcp = lms[9]
          const totalLength = Math.hypot(wrist.x - middleTip.x, wrist.y - middleTip.y)
          const palmLength = Math.hypot(wrist.x - middleMcp.x, wrist.y - middleMcp.y)
          if (totalLength > 0.10 && palmLength > 0.04) {
            drawHandSkeleton(ctx, lms, canvas.width, canvas.height)
            processGesture(lms)
            return
          }
        }
        lastPalmPosRef.current = null
        zoomHoldCountRef.current = 0
        setGestureState("Show Hand")
      }
    }
    const processGesture = (lms: Array<{ x: number; y: number; z: number }>) => {
      const thumbTip = lms[4]
      const indexTip = lms[8]
      const middleTip = lms[12]
      const ringTip = lms[16]
      const pinkyTip = lms[20]
      const wrist = lms[0]
      const isIndexUp = indexTip.y < lms[6].y
      const isMiddleUp = middleTip.y < lms[10].y
      const isRingUp = ringTip.y < lms[14].y
      const isPinkyUp = pinkyTip.y < lms[18].y
      const thumbDist = Math.hypot(thumbTip.x - lms[2].x, thumbTip.y - lms[2].y)
      const isThumbOut = thumbDist > 0.075
      const now = performance.now()
      if (isIndexUp && isMiddleUp && !isRingUp && !isPinkyUp) {
        if (twoFingerStartTimeRef.current === null) {
          twoFingerStartTimeRef.current = now
        }
        const holdDuration = now - twoFingerStartTimeRef.current
        zoomHoldCountRef.current = 0
        lastPalmPosRef.current = null
        if (holdDuration < 350) {
          setGestureState("2F: Hold To Switch")
        } else {
          setGestureState("2F: Switched View")
          if (now - lastSwitchTimeRef.current > 1200) {
            lastSwitchTimeRef.current = now
            onSwitchView()
          }
        }
        return
      }
      twoFingerStartTimeRef.current = null
      if (isIndexUp && isMiddleUp && isRingUp && !isPinkyUp) {
        zoomHoldCountRef.current += 1
        const speed = Math.min(1.2, 0.25 + zoomHoldCountRef.current * 0.035)
        setGestureState(`3F: Zoom In (${speed.toFixed(1)}x)`)
        lastPalmPosRef.current = null
        onZoom(speed)
        return
      }
      const isFist = !isIndexUp && !isMiddleUp && !isRingUp && !isPinkyUp
      if (isFist) {
        zoomHoldCountRef.current += 1
        const speed = Math.min(1.2, 0.25 + zoomHoldCountRef.current * 0.035)
        setGestureState(`Closed Fist: Zoom Out (${speed.toFixed(1)}x)`)
        lastPalmPosRef.current = null
        onZoom(-speed)
        return
      }
      zoomHoldCountRef.current = 0
      if (isIndexUp && isMiddleUp && isRingUp && isPinkyUp && isThumbOut) {
        setGestureState("5F: Orbiting")
        const palmX = (wrist.x + lms[5].x + lms[9].x + lms[13].x + lms[17].x) / 5
        const palmY = (wrist.y + lms[5].y + lms[9].y + lms[13].y + lms[17].y) / 5
        if (lastPalmPosRef.current) {
          const dx = palmX - lastPalmPosRef.current.x
          const dy = palmY - lastPalmPosRef.current.y
          if (Math.hypot(dx, dy) > 0.007) {
            onOrbit(-dx * 3.2, dy * 2.4)
          }
        }
        lastPalmPosRef.current = { x: palmX, y: palmY }
        return
      }
      setGestureState("Show Gestures")
      lastPalmPosRef.current = null
    }
    const drawHandSkeleton = (
      ctx: CanvasRenderingContext2D,
      lms: Array<{ x: number; y: number; z: number }>,
      w: number,
      h: number
    ) => {
      ctx.strokeStyle = "#4ade80"
      ctx.lineWidth = 2
      const connections = [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [5, 9], [9, 10], [10, 11], [11, 12],
        [9, 13], [13, 14], [14, 15], [15, 16],
        [13, 17], [17, 18], [18, 19], [19, 20],
        [0, 17],
      ]
      for (const [i, j] of connections) {
        ctx.beginPath()
        ctx.moveTo(lms[i].x * w, lms[i].y * h)
        ctx.lineTo(lms[j].x * w, lms[j].y * h)
        ctx.stroke()
      }
      ctx.fillStyle = "#ffffff"
      for (const pt of lms) {
        ctx.beginPath()
        ctx.arc(pt.x * w, pt.y * h, 3, 0, Math.PI * 2)
        ctx.fill()
      }
    }
    animRef.current = requestAnimationFrame(loop)
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current)
    }
  }, [enabled, isReady, onOrbit, onZoom, onSwitchView])
  if (!enabled) return null
  return (
    <div className="absolute bottom-6 left-6 z-40 flex flex-col items-center bg-black/90 backdrop-blur-md border border-white/20 p-2 rounded shadow-2xl">
      <div className="flex items-center justify-between w-full pb-1 px-1 text-xs text-white">
        <span className="font-normal flex items-center gap-1.5">
          <span className="inline-block w-2 h-2 rounded bg-emerald-400 animate-pulse" />
          Hand Control
        </span>
        <button
          onClick={onClose}
          className="text-white/60 hover:text-white transition-colors cursor-pointer text-sm"
        >
          ✕
        </button>
      </div>
      <div className="relative w-40 h-40 bg-black/80 rounded overflow-hidden border border-white/10">
        <video
          ref={videoRef}
          className="hidden"
          playsInline
          muted
        />
        <canvas
          ref={canvasRef}
          width={160}
          height={160}
          className="w-full h-full scale-x-[-1]"
        />
      </div>
      <div className="mt-1.5 text-xs text-emerald-400 font-normal tracking-wide text-center">
        {gestureState}
      </div>
    </div>
  )
}