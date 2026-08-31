import numpy as np
import time
import cv2
from calibration import loadAllCalibrations, computeGroundHomographies
from detector import buildBatchTensor, runInferenceOnBatchTensor
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw, ImageFont
from correspondence import find3dCandidates
from tracker3d import Tracker3D
from pathlib import Path
from config import (
  cameraIds,
  datasetRoot,
  yoloModelPath,
)
try:
  fontTitle = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 22)
  fontBody = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 16)
  fontSmall = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 14)
except Exception:
  fontTitle = ImageFont.load_default()
  fontBody = ImageFont.load_default()
  fontSmall = ImageFont.load_default()
def createDashboardTile(tileW: int, tileH: int, frameIdx: int, totalFrames: int, currentFps: float, elapsedMs: float, activeTracksCount: int, total2dCount: int, total3dCount: int, uniqueIdsCount: int) -> np.ndarray:
  pilImg = Image.new("RGB", (tileW, tileH), (28, 34, 42))
  draw = ImageDraw.Draw(pilImg)
  draw.rectangle([(0, 0), (tileW - 1, tileH - 1)], outline=(0, 255, 128), width=2)
  draw.rectangle([(16, 12), (tileW - 16, 48)], fill=(18, 22, 28))
  draw.text((24, 16), "AMD RX 6550M DirectML", fill=(0, 255, 128), font=fontTitle)
  draw.rectangle([(16, 54), (tileW - 16, 90)], fill=(18, 22, 28))
  draw.text((24, 59), f"Frame: {frameIdx}/{totalFrames - 1}", fill=(255, 255, 255), font=fontBody)
  draw.rectangle([(16, 96), (tileW - 16, 132)], fill=(18, 22, 28))
  draw.text((24, 101), f"Speed: {currentFps:.1f} FPS ({elapsedMs:.1f} ms)", fill=(0, 255, 128), font=fontBody)
  draw.rectangle([(16, 138), (tileW - 16, 174)], fill=(18, 22, 28))
  draw.text((24, 143), f"Active 3D Tracks: {activeTracksCount}", fill=(255, 255, 255), font=fontBody)
  draw.rectangle([(16, 180), (tileW - 16, 214)], fill=(18, 22, 28))
  draw.text((24, 184), f"Total 2D Dets: {total2dCount}", fill=(255, 255, 255), font=fontSmall)
  draw.rectangle([(16, 220), (tileW - 16, 254)], fill=(18, 22, 28))
  draw.text((24, 224), f"3D Candidates: {total3dCount}", fill=(0, 255, 128), font=fontSmall)
  draw.rectangle([(16, 260), (tileW - 16, 294)], fill=(18, 22, 28))
  draw.text((24, 264), f"Unique Pedestrian IDs: {uniqueIdsCount}", fill=(255, 255, 255), font=fontSmall)
  bgr = cv2.cvtColor(np.array(pilImg), cv2.COLOR_RGB2BGR)
  return bgr
def createInfoTile(tileW: int, tileH: int) -> np.ndarray:
  pilImg = Image.new("RGB", (tileW, tileH), (28, 34, 42))
  draw = ImageDraw.Draw(pilImg)
  draw.rectangle([(0, 0), (tileW - 1, tileH - 1)], outline=(0, 255, 128), width=2)
  draw.rectangle([(16, 12), (tileW - 16, 48)], fill=(18, 22, 28))
  draw.text((24, 16), "Model: YOLO 26", fill=(0, 255, 128), font=fontTitle)
  draw.rectangle([(16, 54), (tileW - 16, 90)], fill=(18, 22, 28))
  draw.text((24, 59), "Tracker: ByteTrack", fill=(255, 255, 255), font=fontBody)
  draw.rectangle([(16, 96), (tileW - 16, 132)], fill=(18, 22, 28))
  draw.text((24, 101), "Geometry: MRay DLT Triangulation", fill=(0, 255, 128), font=fontBody)
  draw.rectangle([(16, 138), (tileW - 16, 174)], fill=(18, 22, 28))
  draw.text((24, 143), "3D Tracking: 3D Kalman Filter", fill=(255, 255, 255), font=fontBody)
  draw.rectangle([(16, 180), (tileW - 16, 214)], fill=(18, 22, 28))
  draw.text((24, 184), "Ground Model: Planar Homography", fill=(255, 255, 255), font=fontSmall)
  draw.rectangle([(16, 220), (tileW - 16, 254)], fill=(18, 22, 28))
  draw.text((24, 224), "Clustering: Euclidean DBSCAN Radius", fill=(0, 255, 128), font=fontSmall)
  draw.rectangle([(16, 260), (tileW - 16, 294)], fill=(18, 22, 28))
  draw.text((24, 264), "Streams: 7 Synchronized Video Feeds", fill=(255, 255, 255), font=fontSmall)
  bgr = cv2.cvtColor(np.array(pilImg), cv2.COLOR_RGB2BGR)
  return bgr
def readSingleCamera(item: tuple) -> tuple:
  camId, cap = item
  ret, frame = cap.read()
  return camId, ret, frame
def runVideoCheck(maxFrames: int | None = None):
  tStart = time.perf_counter()
  calibrations = loadAllCalibrations()
  homographies = computeGroundHomographies(calibrations)
  print(f"Loaded Calibrations For {len(calibrations)} Cameras")
  videoCaps = {}
  minTotalFrames = None
  for camId in cameraIds:
    camNum = camId.replace("c", "").replace("C", "")
    videoPath = datasetRoot / f"cam{camNum}.mp4"
    if not videoPath.exists():
      print(f"Video Not Found: {videoPath}")
      continue
    cap = cv2.VideoCapture(str(videoPath))
    if not cap.isOpened():
      print(f"Could Not Open Video: {videoPath}")
      continue
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    videoCaps[camId] = cap
    if minTotalFrames is None or frameCount < minTotalFrames:
      minTotalFrames = frameCount
  if not videoCaps:
    print("No Camera Videos Found")
    return
  totalFrames = min(minTotalFrames, maxFrames) if maxFrames else minTotalFrames
  print(f"Full Pipeline Dense Verification On All {totalFrames} Frames Across 7 MP4 Videos")
  tracker = Tracker3D()
  total2dDets = 0
  total3dCandidates = 0
  maxSimultaneousTracks = 0
  multiCamCount = 0
  allLatencies = []
  frameIdx = 0
  tileW, tileH = 640, 346
  tileH3 = 348
  windowName = "AMD Radeon RX 6550M"
  cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
  infoTile = createInfoTile(tileW, tileH3)
  readPool = ThreadPoolExecutor(max_workers=len(videoCaps))
  prepPool = ThreadPoolExecutor(max_workers=1)
  inferPool = ThreadPoolExecutor(max_workers=1)
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
  nextBatchFuture = prepPool.submit(fetchAndPrepNextBatch)
  initItem = nextBatchFuture.result()
  if initItem is None:
    print("Failed To Initialize Video Streams")
    return
  nextBatchFuture = prepPool.submit(fetchAndPrepNextBatch)
  inferFuture = inferPool.submit(runInferenceOnBatchTensor, initItem[1], initItem[2], initItem[3], initItem[4])
  currRawFrames = initItem[0]
  try:
    while frameIdx < totalFrames:
      tFrameStart = time.perf_counter()
      tInferWait0 = time.perf_counter()
      detectionsPerCam = inferFuture.result()
      tInferWait1 = time.perf_counter()
      nextItem = nextBatchFuture.result()
      if nextItem is not None:
        nextBatchFuture = prepPool.submit(fetchAndPrepNextBatch)
        inferFuture = inferPool.submit(runInferenceOnBatchTensor, nextItem[1], nextItem[2], nextItem[3], nextItem[4])
      tRender0 = time.perf_counter()
      renderedTiles = {}
      frameDetsCount = 0
      for camId, frame in currRawFrames.items():
        dets = detectionsPerCam.get(camId, [])
        frameDetsCount += len(dets)
        origH, origW = frame.shape[:2]
        curTileH = tileH3 if camId == "c7" else tileH
        tileBgr = cv2.resize(frame, (tileW, curTileH), interpolation=cv2.INTER_LINEAR)
        scaleX = float(tileW) / float(origW)
        scaleY = float(curTileH) / float(origH)
        for d in dets:
          b = d["boxXyxy"]
          x1 = max(0, min(tileW - 1, int(b[0] * scaleX)))
          y1 = max(0, min(curTileH - 1, int(b[1] * scaleY)))
          x2 = max(0, min(tileW - 1, int(b[2] * scaleX)))
          y2 = max(0, min(curTileH - 1, int(b[3] * scaleY)))
          cv2.rectangle(tileBgr, (x1, y1), (x2, y2), (0, 255, 128), 1, cv2.LINE_AA)
          if (y2 - y1) >= 18:
            label = f"ID:{d['camTrackId']}"
            cv2.rectangle(tileBgr, (x1, max(0, y1 - 18)), (x1 + 50, max(12, y1)), (18, 22, 28), -1)
            cv2.putText(tileBgr, label, (x1 + 3, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 128), 1, cv2.LINE_AA)
        cv2.rectangle(tileBgr, (12, 12), (118, 38), (18, 22, 28), -1)
        cv2.rectangle(tileBgr, (12, 12), (118, 38), (0, 255, 128), 1, cv2.LINE_AA)
        cv2.putText(tileBgr, f"Camera {camId.upper()}", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        renderedTiles[camId] = tileBgr
      tRender1 = time.perf_counter()
      candidates = find3dCandidates(detectionsPerCam, homographies, calibrations)
      tCorr1 = time.perf_counter()
      tracks = tracker.update(candidates)
      tTrack1 = time.perf_counter()
      total2dDets += frameDetsCount
      total3dCandidates += len(candidates)
      maxSimultaneousTracks = max(maxSimultaneousTracks, len(tracks))
      for trk in tracks:
        if trk.get("numCameras", 1) >= 2:
          multiCamCount += 1
      camOrder = list(renderedTiles.keys())
      c1 = renderedTiles.get(camOrder[0]) if len(camOrder) > 0 else np.zeros((tileH, tileW, 3), dtype=np.uint8)
      c2 = renderedTiles.get(camOrder[1]) if len(camOrder) > 1 else np.zeros((tileH, tileW, 3), dtype=np.uint8)
      c3 = renderedTiles.get(camOrder[2]) if len(camOrder) > 2 else np.zeros((tileH, tileW, 3), dtype=np.uint8)
      c4 = renderedTiles.get(camOrder[3]) if len(camOrder) > 3 else np.zeros((tileH, tileW, 3), dtype=np.uint8)
      c5 = renderedTiles.get(camOrder[4]) if len(camOrder) > 4 else np.zeros((tileH, tileW, 3), dtype=np.uint8)
      c6 = renderedTiles.get(camOrder[5]) if len(camOrder) > 5 else np.zeros((tileH, tileW, 3), dtype=np.uint8)
      c7 = renderedTiles.get(camOrder[6]) if len(camOrder) > 6 else np.zeros((tileH3, tileW, 3), dtype=np.uint8)
      elapsedMs = (time.perf_counter() - tFrameStart) * 1000.0
      allLatencies.append(elapsedMs)
      currentFps = 1000.0 / elapsedMs if elapsedMs > 0 else 0.0
      dashTile = createDashboardTile(tileW, tileH3, frameIdx, totalFrames, currentFps, elapsedMs, len(tracks), total2dDets, total3dCandidates, tracker.nextId - 1)
      row1 = np.hstack([c1, c2, c3])
      row2 = np.hstack([c4, c5, c6])
      row3 = np.hstack([c7, dashTile, infoTile])
      mosaic = np.vstack([row1, row2, row3])
      cv2.imshow(windowName, mosaic)
      key = cv2.waitKey(1) & 0xFF
      tShow1 = time.perf_counter()
      if key == 27 or key == ord("q"):
        print("Playback Stopped!")
        break
      if frameIdx % 10 == 0:
        tInferMs = (tInferWait1 - tInferWait0) * 1000.0
        tRenderMs = (tRender1 - tRender0) * 1000.0
        tCorrMs = (tCorr1 - tRender1) * 1000.0
        tTrackMs = (tTrack1 - tCorr1) * 1000.0
        tShowMs = (tShow1 - tTrack1) * 1000.0
        print(f"Frame {frameIdx}: InferWait={tInferMs:.1f}ms | Render={tRenderMs:.1f}ms | 3D-Fusion={tCorrMs:.1f}ms | Tracker={tTrackMs:.1f}ms | Show={tShowMs:.1f}ms | Cycle={elapsedMs:.1f}ms ({currentFps:.1f} FPS)")
      if nextItem is None:
        break
      currRawFrames = nextItem[0]
      frameIdx += 1
  finally:
    prepPool.shutdown(wait=False)
    inferPool.shutdown(wait=False)
    readPool.shutdown(wait=False)
    for cap in videoCaps.values():
      cap.release()
    cv2.destroyAllWindows()
  totalElapsed = time.perf_counter() - tStart
  avgMsPerFrame = np.mean(allLatencies) if allLatencies else 0.0
  avgFps = 1000.0 / avgMsPerFrame if avgMsPerFrame > 0 else 0.0
  uniqueTracksCount = tracker.nextId - 1
  print("=" * 60)
  print("Summary Report")
  print("=" * 60)
  print(f"Total Frames Processed: {frameIdx}")
  print(f"Total Camera Video Streams: {len(videoCaps)}")
  print(f"Total 2D Detections: {total2dDets}")
  print(f"Total 3D Spatial Candidates: {total3dCandidates}")
  print(f"Total Unique Pedestrian Tracks Identified: {uniqueTracksCount}")
  print(f"Max Simultaneous Tracks In A Single Frame: {maxSimultaneousTracks}")
  print(f"Multi-Camera Consensus Hits: {multiCamCount}")
  print(f"Average Pipeline Latency Per Timestep: {avgMsPerFrame:.2f} ms")
  print(f"Average Throughput: {avgFps:.2f} FPS")
  print(f"Total Execution Time: {totalElapsed:.2f}s")
  print("=" * 60)
  print("Completed Successfully!")
if __name__ == "__main__":
  runVideoCheck()