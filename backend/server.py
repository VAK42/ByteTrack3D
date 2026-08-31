import websockets
import traceback
import asyncio
import ctypes
import json
import time
import cv2
try:
  ctypes.windll.winmm.timeBeginPeriod(1)
except Exception:
  pass
from detector import buildBatchTensor, runInferenceOnBatchTensor, loadFrame
from calibration import loadAllCalibrations, computeGroundHomographies
from concurrent.futures import ThreadPoolExecutor
from correspondence import find3dCandidates
from tracker3d import Tracker3D
from pathlib import Path
from config import (
  cameraIds,
  datasetRoot,
  imageSubsetsDir,
  simulationFps,
  wsHost,
  wsPort,
  worldXMin,
  worldXMax,
  worldZMin,
  worldZMax,
)
connectedClients = set()
async def broadcast(payload: dict):
  if not connectedClients:
    return
  msg = json.dumps(payload, separators=(",", ":"))
  disconnected = set()
  for ws in list(connectedClients):
    try:
      await ws.send(msg)
    except Exception:
      disconnected.add(ws)
  connectedClients.difference_update(disconnected)
async def wsHandler(ws):
  connectedClients.add(ws)
  remote = ws.remote_address
  print(f"Client Connected: {remote}")
  try:
    await ws.wait_closed()
  finally:
    connectedClients.discard(ws)
    print(f"Client Disconnected: {remote}")
def readSingleCamera(item: tuple) -> tuple:
  camId, cap = item
  ret, frame = cap.read()
  return camId, ret, frame
async def simulationLoop(calibrations: dict, homographies: dict, tracker: Tracker3D):
  frameInterval = 1.0 / simulationFps if simulationFps > 0 else 0.0
  frameIdx = 0
  worldMeta = {
    "xMin": worldXMin,
    "xMax": worldXMax,
    "zMin": worldZMin,
    "zMax": worldZMax,
  }
  camerasMeta = []
  import numpy as np
  for camId, cal in calibrations.items():
    R = cal["R"]
    t = cal["t"]
    cWorld = (-R.T @ t).flatten()
    lookDir = (R.T @ np.array([[0], [0], [1]])).flatten()
    camNum = camId.replace("c", "").replace("C", "")
    targetPos = [
      round(float(cWorld[0] + 8.0 * lookDir[0]), 2),
      round(float(cWorld[2] + 8.0 * lookDir[2]), 2),
      round(float(cWorld[1] + 8.0 * lookDir[1]), 2),
    ]
    camerasMeta.append({
      "id": camId,
      "label": f"C{camNum}",
      "pos": [round(float(cWorld[0]), 2), round(float(cWorld[2]), 2), round(float(cWorld[1]), 2)],
      "target": targetPos,
    })
  videoCaps = {}
  minTotalFrames = None
  for camId in cameraIds:
    camNum = camId.replace("c", "").replace("C", "")
    videoPath = datasetRoot / f"cam{camNum}.mp4"
    if videoPath.exists():
      cap = cv2.VideoCapture(str(videoPath))
      if cap.isOpened():
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        videoCaps[camId] = cap
        if minTotalFrames is None or frameCount < minTotalFrames:
          minTotalFrames = frameCount
  useVideos = len(videoCaps) == len(cameraIds)
  readPool = ThreadPoolExecutor(max_workers=len(videoCaps)) if useVideos else None
  prepPool = ThreadPoolExecutor(max_workers=1) if useVideos else None
  inferPool = ThreadPoolExecutor(max_workers=1) if useVideos else None
  def fetchAndPrepNextBatch():
    readTasks = [(camId, cap) for camId, cap in videoCaps.items()]
    readResults = list(readPool.map(readSingleCamera, readTasks))
    rawFrames = {}
    for camId, ret, frame in readResults:
      if not ret or frame is None:
        return None
      rawFrames[camId] = frame
    if len(rawFrames) < len(videoCaps):
      return None
    batchTensor, camList, origShapes, scales = buildBatchTensor(rawFrames)
    return rawFrames, batchTensor, camList, origShapes, scales
  if useVideos:
    totalFrames = minTotalFrames
    print(f"Server Streaming From 7 MP4 Videos ({totalFrames} Frames)")
    nextBatchFuture = prepPool.submit(fetchAndPrepNextBatch)
    initItem = nextBatchFuture.result()
    if initItem is not None:
      nextBatchFuture = prepPool.submit(fetchAndPrepNextBatch)
      inferFuture = inferPool.submit(runInferenceOnBatchTensor, initItem[1], initItem[2], initItem[3], initItem[4])
  else:
    cam1Dir = Path(imageSubsetsDir) / (cameraIds[0] if cameraIds else "c1")
    totalFrames = len(list(cam1Dir.glob("*.png")) + list(cam1Dir.glob("*.jpg"))) if cam1Dir.exists() else 0
    print(f"Server Streaming From Image Subsets ({totalFrames} Frames)")
  while True:
    tStart = time.perf_counter()
    tSim = frameIdx / simulationFps if simulationFps > 0 else 0.0
    try:
      if useVideos and initItem is not None:
        t0 = time.perf_counter()
        detectionsPerCam = inferFuture.result()
        tInfer = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        nextItem = nextBatchFuture.result()
        tPrep = (time.perf_counter() - t1) * 1000.0
        if nextItem is not None:
          nextBatchFuture = prepPool.submit(fetchAndPrepNextBatch)
          inferFuture = inferPool.submit(runInferenceOnBatchTensor, nextItem[1], nextItem[2], nextItem[3], nextItem[4])
        t2 = time.perf_counter()
        candidates = find3dCandidates(detectionsPerCam, homographies, calibrations)
        tracks = tracker.update(candidates)
        tTrack = (time.perf_counter() - t2) * 1000.0
        if frameIdx % 30 == 0:
          instFps = 1.0 / max(0.001, time.perf_counter() - tStart)
          print(f"Frame {frameIdx:04d} | Infer: {tInfer:.1f}ms | Prep: {tPrep:.1f}ms | 3D Track: {tTrack:.1f}ms | Speed: {instFps:.1f} FPS")
        if nextItem is None:
          for cap in videoCaps.values():
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
          nextBatchFuture = prepPool.submit(fetchAndPrepNextBatch)
          initItem = nextBatchFuture.result()
          if initItem is not None:
            nextBatchFuture = prepPool.submit(fetchAndPrepNextBatch)
            inferFuture = inferPool.submit(runInferenceOnBatchTensor, initItem[1], initItem[2], initItem[3], initItem[4])
      elif totalFrames > 0:
        rawFrames = {}
        for camId in cameraIds:
          frame = loadFrame(camId, frameIdx, imageSubsetsDir)
          if frame is not None:
            rawFrames[camId] = frame
        batchTensor, camList, origShapes, scales = buildBatchTensor(rawFrames)
        detectionsPerCam = runInferenceOnBatchTensor(batchTensor, camList, origShapes, scales) if rawFrames else {}
        candidates = find3dCandidates(detectionsPerCam, homographies, calibrations)
        tracks = tracker.update(candidates)
      else:
        import math
        t = tSim
        candidates = []
        for i in range(5):
          angle = t * 0.4 + i * (2 * math.pi / 5)
          r = 3.0 + i * 0.5
          candidates.append({
            "pos3d": [r * math.cos(angle) + 3.0, 0.0, r * math.sin(angle) + 8.0],
            "cameras": ["C1"],
            "conf": 0.9,
            "cls": 0,
          })
        tracks = tracker.update(candidates)
    except Exception as e:
      print(f"Frame {frameIdx} Error: {e}")
      traceback.print_exc()
      tracks = []
    instFps = 1.0 / max(0.001, time.perf_counter() - tStart)
    await broadcast({
      "frame": frameIdx,
      "fps": round(instFps, 1),
      "t": round(tSim, 3),
      "tracks": tracks,
      "cameras": camerasMeta,
      "world": worldMeta,
    })
    frameIdx += 1
    if totalFrames > 0 and frameIdx >= totalFrames:
      print("End Of Dataset — Looping Back To Frame 0")
      frameIdx = 0
      tracker = Tracker3D()
    elapsed = time.perf_counter() - tStart
    sleepTime = frameInterval - elapsed
    if sleepTime > 0.002:
      await asyncio.sleep(sleepTime)
    else:
      await asyncio.sleep(0)
async def main():
  print("Loading Calibrations")
  calibrations = loadAllCalibrations()
  homographies = computeGroundHomographies(calibrations)
  tracker = Tracker3D()
  print(f"Loaded Calibration For Cameras: {list(calibrations.keys())}")
  async with websockets.serve(wsHandler, wsHost, wsPort):
    print(f"WebSocket Listening On ws://{wsHost}:{wsPort}")
    await simulationLoop(calibrations, homographies, tracker)
if __name__ == "__main__":
  asyncio.run(main())