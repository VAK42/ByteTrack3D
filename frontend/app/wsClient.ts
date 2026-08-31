export type TrackData = {
  trackId: number
  pos: [number, number, number]
  vel: [number, number, number]
  cls: number
  age: number
  hits: number
  trail: [number, number, number][]
  lost: boolean
}
export type CameraData = {
  id: string
  label: string
  pos: [number, number, number]
  target: [number, number, number]
}
export type FrameMessage = {
  frame: number
  fps?: number
  t: number
  tracks: TrackData[]
  cameras?: CameraData[]
  world: {
    xMin: number
    xMax: number
    zMin: number
    zMax: number
  }
}
export type ConnectionStatus = "connecting" | "connected" | "disconnected"
type FrameCallback = (msg: FrameMessage) => void
type StatusCallback = (status: ConnectionStatus) => void
const wsUrl = "ws://localhost:8765"
const reconnectDelayMs = 2000
export class WsClient {
  private ws: WebSocket | null = null
  private onFrame: FrameCallback
  private onStatus: StatusCallback
  private destroyed = false
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null
  constructor(onFrame: FrameCallback, onStatus: StatusCallback) {
    this.onFrame = onFrame
    this.onStatus = onStatus
    this.connect()
  }
  private connect() {
    if (this.destroyed) return
    this.onStatus("connecting")
    try {
      this.ws = new WebSocket(wsUrl)
      this.ws.onopen = () => {
        this.onStatus("connected")
      }
      this.ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data as string) as FrameMessage
          this.onFrame(data)
        } catch {}
      }
      this.ws.onerror = () => {
        this.ws?.close()
      }
      this.ws.onclose = () => {
        this.onStatus("disconnected")
        this.scheduleReconnect()
      }
    } catch {
      this.onStatus("disconnected")
      this.scheduleReconnect()
    }
  }
  private scheduleReconnect() {
    if (this.destroyed) return
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer)
    this.reconnectTimer = setTimeout(() => {
      this.connect()
    }, reconnectDelayMs)
  }
  reconnect = () => {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer)
    if (this.ws) {
      this.ws.onclose = null
      this.ws.close()
      this.ws = null
    }
    this.connect()
  }
  destroy() {
    this.destroyed = true
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer)
    if (this.ws) {
      this.ws.onclose = null
      this.ws.close()
      this.ws = null
    }
  }
}
export const WSClient = WsClient
export type WSClient = WsClient