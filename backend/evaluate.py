import numpy as np
import json
import cv2
from calibration import loadAllCalibrations, computeGroundHomographies
from detector import buildBatchTensor, runInferenceOnBatchTensor
from scipy.optimize import linear_sum_assignment
from correspondence import find3dCandidates
from config import datasetRoot, cameraIds
from tracker3d import Tracker3D
from pathlib import Path
def evaluateVideoFrame(frameIdx: int = 0, calibrations: dict = None, homographies: dict = None, tracker: Tracker3D = None):
  if calibrations is None:
    calibrations = loadAllCalibrations()
  if homographies is None:
    homographies = computeGroundHomographies(calibrations)
  if tracker is None:
    tracker = Tracker3D()
  jsonPath = datasetRoot / "positions" / f"{frameIdx:08d}.json"
  if not jsonPath.exists():
    allJson = sorted((datasetRoot / "positions").glob("*.json"))
    jsonPath = allJson[0] if allJson else None
  if jsonPath is None or not jsonPath.exists():
    print(f"No Ground Truth Annotation For Frame {frameIdx}")
    return None
  gtData = json.load(open(jsonPath))
  gtPositions = []
  for p in gtData:
    pid = p["personID"]
    posID = p["positionID"]
    gx = (posID % 480) * 0.025 - 3.0
    gz = (posID // 480) * 0.025 - 9.0
    validViews = [v for v in p["views"] if v["xmin"] != -1]
    gtPositions.append({
      "pid": pid,
      "gtX": gx,
      "gtZ": gz,
      "numViews": len(validViews),
    })
  rawFrames = {}
  for camId in cameraIds:
    camNum = camId.replace("c", "").replace("C", "")
    videoPath = datasetRoot / f"cam{camNum}.mp4"
    if videoPath.exists():
      cap = cv2.VideoCapture(str(videoPath))
      cap.set(cv2.CAP_PROP_POS_FRAMES, frameIdx)
      ret, frame = cap.read()
      if ret and frame is not None:
        rawFrames[camId] = frame
      cap.release()
  if len(rawFrames) < len(cameraIds):
    print(f"Frame {frameIdx}: Could Only Read {len(rawFrames)}/{len(cameraIds)} Video Streams")
    return None
  batchTensor, camList, origShapes, scales = buildBatchTensor(rawFrames)
  detectionsPerCam = runInferenceOnBatchTensor(batchTensor, camList, origShapes, scales)
  candidates = find3dCandidates(detectionsPerCam, homographies, calibrations)
  tracks = tracker.update(candidates)
  predCoords = np.array([[t["pos"][0], t["pos"][2]] for t in tracks])
  gtCoords = np.array([[g["gtX"], g["gtZ"]] for g in gtPositions])
  costMatrix = np.zeros((len(tracks), len(gtPositions)))
  for i, p in enumerate(predCoords):
    for j, g in enumerate(gtCoords):
      costMatrix[i, j] = np.linalg.norm(p - g)
  rowInd, colInd = linear_sum_assignment(costMatrix)
  matchedErrors = []
  matchedPairs = []
  for r, c in zip(rowInd, colInd):
    err = costMatrix[r, c]
    t = tracks[r]
    g = gtPositions[c]
    if err <= 1.2:
      matchedErrors.append(err)
      matchedPairs.append((t, g, err, True))
    else:
      matchedPairs.append((t, g, err, False))
  precision = (len(matchedErrors) / len(tracks)) * 100 if tracks else 0.0
  recall = (len(matchedErrors) / len(gtPositions)) * 100 if gtPositions else 0.0
  f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
  meanErrorCm = np.mean(matchedErrors) * 100 if matchedErrors else 0.0
  medianErrorCm = np.median(matchedErrors) * 100 if matchedErrors else 0.0
  return {
    "frameIdx": frameIdx,
    "gtCount": len(gtPositions),
    "detCount": sum(len(d) for d in detectionsPerCam.values()),
    "candCount": len(candidates),
    "trackCount": len(tracks),
    "matchedCount": len(matchedErrors),
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "meanErrorCm": meanErrorCm,
    "medianErrorCm": medianErrorCm,
    "matchedPairs": matchedPairs,
  }
def runEvaluation(startFrame: int = 0, numFrames: int = 1, step: int = 5):
  print("Loading Camera Calibrations")
  calibrations = loadAllCalibrations()
  homographies = computeGroundHomographies(calibrations)
  tracker = Tracker3D()
  allResults = []
  print(f"Running Evaluation Across {numFrames} Frame(s)")
  for idx in range(numFrames):
    f = startFrame + idx * step
    res = evaluateVideoFrame(f, calibrations, homographies, tracker)
    if res is not None:
      allResults.append(res)
      if numFrames == 1:
        print("=" * 60)
        print(f"Summary Report {res['frameIdx']}")
        print("=" * 60)
        print(f"Ground Truth People In Dataset: {res['gtCount']}")
        print(f"Detections Across 7 Cameras: {res['detCount']}")
        print(f"3D Candidates Triangulated: {res['candCount']}")
        print(f"Active 3D Tracks Output: {res['trackCount']}")
        print("-" * 60)
        print(f"1-To-1 Ground Truth Matching Table")
        print("-" * 60)
        print(f"{'Track ID':<10} | {'Predicted (X, Z)':<20} | {'GT Person':<10} | {'GT (X, Z)':<20} | {'Error (cm)':<12} | {'Status'}")
        print("-" * 92)
        for t, g, err, isMatch in res["matchedPairs"]:
          statusStr = f"Match (Hits: {t['hits']})" if isMatch else "Out Of Threshold"
          print(f"Track #{t['trackId']:<4} | ({t['pos'][0]:5.2f}m, {t['pos'][2]:5.2f}m)    | Person #{g['pid']:<4} | ({g['gtX']:5.2f}m, {g['gtZ']:5.2f}m)    | {err*100:6.1f} cm   | {statusStr}")
  if allResults:
    avgPrec = np.mean([r["precision"] for r in allResults])
    avgRec = np.mean([r["recall"] for r in allResults])
    avgF1 = np.mean([r["f1"] for r in allResults])
    avgMeanErr = np.mean([r["meanErrorCm"] for r in allResults])
    avgMedianErr = np.mean([r["medianErrorCm"] for r in allResults])
    print("=" * 60)
    print(f"Summary Metrics ({len(allResults)} Frame(s) Evaluated)")
    print("=" * 60)
    print(f"3D Tracking Precision : {avgPrec:.1f}%")
    print(f"Ground Truth Recall   : {avgRec:.1f}%")
    print(f"F1 Accuracy Score     : {avgF1:.1f}%")
    print(f"Mean 3D Position Error: {avgMeanErr:.1f} cm ({avgMeanErr/100:.2f} m)")
    print(f"Median 3D Error       : {avgMedianErr:.1f} cm ({avgMedianErr/100:.2f} m)")
if __name__ == "__main__":
  runEvaluation(startFrame=5, numFrames=1, step=5)