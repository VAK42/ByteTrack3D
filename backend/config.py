import os
from pathlib import Path
projectRoot = Path(__file__).parent.parent
datasetRoot = projectRoot / "wildtrack"
imageSubsetsDir = datasetRoot / "subsets"
calibrationDir = datasetRoot / "calibrations"
extrinsicDir = calibrationDir / "extrinsic"
intrinsicDir = calibrationDir / "intrinsic0"
annotationsDir = datasetRoot / "positions"
cameraIds = ["c1", "c2", "c3", "c4", "c5", "c6", "c7"]
cameraCalibMap = {
  "c1": "cvlab1",
  "c2": "cvlab2",
  "c3": "cvlab3",
  "c4": "cvlab4",
  "c5": "idiap1",
  "c6": "idiap2",
  "c7": "idiap3",
  "C1": "cvlab1",
  "C2": "cvlab2",
  "C3": "cvlab3",
  "C4": "cvlab4",
  "C5": "idiap1",
  "C6": "idiap2",
  "C7": "idiap3",
}
yoloModelPath = projectRoot / "model.onnx" if (projectRoot / "model.onnx").exists() else projectRoot / "model.pt"
def selectDevice():
  try:
    import torch
    if torch.cuda.is_available():
      return 0
  except Exception:
    pass
  return "cpu"
yoloDevice = selectDevice()
yoloConfThreshold = 0.20
yoloIouThreshold = 0.45
yoloTargetClasses = [0]
simulationFps = 60.0
groundClusterRadiusM = 0.70
byteTrack3dTrackThresh = 0.40
byteTrack3dLowThresh = 0.15
byteTrack3dNewTrackThresh = 0.50
byteTrack3dMatchDist1 = 1.80
byteTrack3dMatchDist2 = 1.20
byteTrack3dTrackBuffer = 15
minCamerasForConfirmation = 2
minCamerasForTrack = 2
wsHost = "localhost"
wsPort = 8765
wsUri = f"ws://{wsHost}:{wsPort}"
worldXMin, worldXMax = -3.0, 9.0
worldZMin, worldZMax = -9.0, 26.0
worldYPerson = 1.75