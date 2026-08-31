import numpy as np
from config import (
  groundClusterRadiusM,
  minCamerasForTrack,
  worldXMin,
  worldXMax,
  worldZMin,
  worldZMax,
)
from calibration import projectToGround, triangulate3d
from sklearn.cluster import DBSCAN
GroundPoint = dict
Candidate3D = dict
def isInsideWorldBounds(x: float, z: float, margin: float = 2.0) -> bool:
  return (
    (worldXMin - margin) <= x <= (worldXMax + margin) and
    (worldZMin - margin) <= z <= (worldZMax + margin)
  )
def projectFootpoints(
  detectionsPerCam: dict[str, list[dict]],
  homographies: dict[str, np.ndarray],
) -> list[GroundPoint]:
  groundPts: list[GroundPoint] = []
  for camId, dets in detectionsPerCam.items():
    hMatrix = homographies.get(camId)
    if hMatrix is None:
      continue
    for det in dets:
      gpXz = projectToGround(det["footUv"], hMatrix)
      if not isInsideWorldBounds(gpXz[0], gpXz[1], margin=2.5):
        continue
      groundPts.append({
        "camId": camId,
        "camTrackId": det["camTrackId"],
        "footUv": det["footUv"],
        "gpXz": gpXz,
        "conf": det["conf"],
        "cls": det["cls"],
        "boxXyxy": det["boxXyxy"],
      })
  return groundPts
def clusterGroundPoints(groundPts: list[GroundPoint]) -> list[list[GroundPoint]]:
  if not groundPts:
    return []
  coords = np.array([p["gpXz"] for p in groundPts], dtype=np.float64)
  labels = DBSCAN(
    eps=groundClusterRadiusM,
    min_samples=1,
    metric="euclidean",
  ).fit_predict(coords)
  clusters: dict[int, list[GroundPoint]] = {}
  for label, pt in zip(labels, groundPts):
    if label == -1:
      continue
    clusters.setdefault(label, []).append(pt)
  return list(clusters.values())
def bestPerCamera(cluster: list[GroundPoint]) -> list[GroundPoint]:
  best: dict[str, GroundPoint] = {}
  for pt in cluster:
    cam = pt["camId"]
    if cam not in best or pt["conf"] > best[cam]["conf"]:
      best[cam] = pt
  return list(best.values())
def triangulateClusters(
  clusters: list[list[GroundPoint]],
  calibrations: dict[str, dict],
) -> list[Candidate3D]:
  candidates: list[Candidate3D] = []
  for cluster in clusters:
    pts = bestPerCamera(cluster)
    if len(pts) < minCamerasForTrack:
      pt = pts[0]
      xz = pt["gpXz"]
      if not isInsideWorldBounds(xz[0], xz[1], margin=1.5):
        continue
      pos3d = np.array([xz[0], 0.0, xz[1]], dtype=np.float64)
      cameras = [pt["camId"]]
    else:
      imagePoints = {p["camId"]: p["footUv"] for p in pts}
      pos3d = triangulate3d(imagePoints, calibrations)
      if pos3d is not None and isInsideWorldBounds(pos3d[0], pos3d[1], margin=2.0) and -0.5 <= pos3d[2] <= 2.2:
        pos3d = np.array([pos3d[0], 0.0, pos3d[1]], dtype=np.float64)
      else:
        meanXz = np.mean([p["gpXz"] for p in pts], axis=0)
        if not isInsideWorldBounds(meanXz[0], meanXz[1], margin=1.5):
          continue
        pos3d = np.array([meanXz[0], 0.0, meanXz[1]], dtype=np.float64)
      cameras = list(imagePoints.keys())
    avgConf = float(np.mean([p["conf"] for p in pts]))
    dominantCls = max(set(p["cls"] for p in pts), key=lambda c: sum(1 for p in pts if p["cls"] == c))
    candidates.append({
      "pos3d": pos3d,
      "cameras": cameras,
      "conf": avgConf,
      "cls": dominantCls,
      "imagePoints": {p["camId"]: p["footUv"] for p in pts},
      "camTrackIds": {p["camId"]: p["camTrackId"] for p in pts},
    })
  return candidates
def find3dCandidates(
  detectionsPerCam: dict[str, list[dict]],
  homographies: dict[str, np.ndarray],
  calibrations: dict[str, dict],
) -> list[Candidate3D]:
  groundPts = projectFootpoints(detectionsPerCam, homographies)
  clusters = clusterGroundPoints(groundPts)
  candidates = triangulateClusters(clusters, calibrations)
  return candidates