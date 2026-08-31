import numpy as np
import time
from pathlib import Path
from config import (
  cameraIds,
  imageSubsetsDir,
  yoloModelPath,
)
from calibration import loadAllCalibrations, computeGroundHomographies
from detector import detectAndTrackBatch, loadFrame
from correspondence import find3dCandidates
from tracker3d import Tracker3D
def runImageCheck(maxFrames: int | None = None):
  tStart = time.perf_counter()
  calibrations = loadAllCalibrations()
  homographies = computeGroundHomographies(calibrations)
  print(f"Loaded Calibrations For {len(calibrations)} Cameras")
  firstCam = cameraIds[0] if cameraIds else "c1"
  camDir = Path(imageSubsetsDir) / firstCam
  availableFrames = len(list(camDir.glob("*.png")) + list(camDir.glob("*.jpg")))
  totalFrames = min(availableFrames, maxFrames) if maxFrames else availableFrames
  print(f"Full Pipeline Verification On All {totalFrames} Dataset Frames")
  tracker = Tracker3D()
  total2dDets = 0
  total3dCandidates = 0
  maxSimultaneousTracks = 0
  multiCamCount = 0
  allLatencies = []
  for frameIdx in range(totalFrames):
    tFrameStart = time.perf_counter()
    rawFrames = {}
    for camId in cameraIds:
      frame = loadFrame(camId, frameIdx, imageSubsetsDir)
      if frame is not None:
        rawFrames[camId] = frame
    detectionsPerCam = detectAndTrackBatch(rawFrames)
    frameDetsCount = sum(len(d) for d in detectionsPerCam.values())
    candidates = find3dCandidates(detectionsPerCam, homographies, calibrations)
    tracks = tracker.update(candidates)
    elapsedMs = (time.perf_counter() - tFrameStart) * 1000.0
    allLatencies.append(elapsedMs)
    total2dDets += frameDetsCount
    total3dCandidates += len(candidates)
    maxSimultaneousTracks = max(maxSimultaneousTracks, len(tracks))
    for trk in tracks:
      if trk.get("numCameras", 1) >= 2:
        multiCamCount += 1
    currentFps = 1000.0 / elapsedMs if elapsedMs > 0 else 0.0
    print(f"Frame {frameIdx}/{totalFrames - 1}: {frameDetsCount} 2D Dets → {len(candidates)} 3D Candidates → {len(tracks)} Active Tracks ({elapsedMs:.1f} ms | {currentFps:.1f} FPS)")
    for trk in tracks[:3]:
      pos = trk["pos"]
      vel = trk["vel"]
      speed = np.linalg.norm(vel)
      print(f"  Track ID {trk['trackId']}: Pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})m, Speed={speed:.2f}m/s, Cams={trk['numCameras']}, Hits={trk['hits']}")
  totalElapsed = time.perf_counter() - tStart
  avgMsPerFrame = np.mean(allLatencies) if allLatencies else 0.0
  avgFps = 1000.0 / avgMsPerFrame if avgMsPerFrame > 0 else 0.0
  uniqueTracksCount = tracker.nextId - 1
  print("=" * 60)
  print("Summary Report")
  print("=" * 60)
  print(f"Total Frames Processed: {totalFrames}")
  print(f"Total Camera Views Analyzed: {totalFrames * len(cameraIds)}")
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
  runImageCheck()