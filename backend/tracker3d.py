import numpy as np
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass, field
from enum import Enum
from config import (
  byteTrack3dTrackThresh,
  byteTrack3dLowThresh,
  byteTrack3dNewTrackThresh,
  byteTrack3dMatchDist1,
  byteTrack3dMatchDist2,
  byteTrack3dTrackBuffer,
  minCamerasForConfirmation,
  worldXMin,
  worldXMax,
  worldZMin,
  worldZMax,
)
class TrackState(Enum):
  New = 0
  Tracked = 1
  Lost = 2
  Removed = 3
class KalmanFilter3D:
  def __init__(self, initialPos: np.ndarray, dt: float = 0.033):
    self.dt = dt
    self.fMatrix = np.eye(6, dtype=np.float64)
    self.fMatrix[0, 3] = self.fMatrix[1, 4] = self.fMatrix[2, 5] = self.dt
    self.hMatrix = np.zeros((3, 6), dtype=np.float64)
    self.hMatrix[0, 0] = self.hMatrix[1, 1] = self.hMatrix[2, 2] = 1.0
    stdP = 0.02
    stdV = 0.04
    self.qMatrix = np.diag([stdP**2, stdP**2, stdP**2, stdV**2, stdV**2, stdV**2])
    stdM = 0.25
    self.rMatrix = np.eye(3, dtype=np.float64) * (stdM ** 2)
    self.stateVector = np.zeros(6, dtype=np.float64)
    self.stateVector[:3] = initialPos
    self.stateVector[1] = 0.0
    self.pMatrix = np.eye(6, dtype=np.float64) * 0.2
  def predict(self) -> np.ndarray:
    self.stateVector = self.fMatrix @ self.stateVector
    self.stateVector[3:] *= 0.90
    speed = np.linalg.norm(self.stateVector[3:])
    if speed > 2.5:
      self.stateVector[3:] = (self.stateVector[3:] / speed) * 2.5
    self.stateVector[1] = 0.0
    self.stateVector[4] = 0.0
    self.pMatrix = self.fMatrix @ self.pMatrix @ self.fMatrix.T + self.qMatrix
    return self.stateVector[:3].copy()
  def update(self, measurement: np.ndarray):
    meas = measurement.copy()
    meas[1] = 0.0
    y = meas - (self.hMatrix @ self.stateVector)
    sMatrix = self.hMatrix @ self.pMatrix @ self.hMatrix.T + self.rMatrix
    kGain = self.pMatrix @ self.hMatrix.T @ np.linalg.inv(sMatrix)
    self.stateVector = self.stateVector + (kGain @ y)
    self.stateVector[1] = 0.0
    self.stateVector[4] = 0.0
    self.pMatrix = (np.eye(6) - kGain @ self.hMatrix) @ self.pMatrix
  @property
  def pos(self) -> np.ndarray:
    return self.stateVector[:3].copy()
  @property
  def vel(self) -> np.ndarray:
    return self.stateVector[3:].copy()
@dataclass
class STrack3D:
  trackId: int
  kf: KalmanFilter3D
  cls: int = 0
  score: float = 0.0
  state: TrackState = TrackState.New
  age: int = 1
  hits: int = 1
  timeSinceUpdate: int = 0
  numCameras: int = 1
  trail: list = field(default_factory=list)
  maxTrail: int = 60
  def markTracked(self, pos: np.ndarray, score: float, numCameras: int):
    self.kf.update(pos)
    self.score = score
    self.hits += 1
    self.age += 1
    self.timeSinceUpdate = 0
    self.numCameras = numCameras
    self.state = TrackState.Tracked
    currPos = self.kf.pos.tolist()
    currPos[1] = 0.0
    self.trail.append(currPos)
    if len(self.trail) > self.maxTrail:
      self.trail.pop(0)
  def markLost(self):
    self.timeSinceUpdate += 1
    self.age += 1
    if self.state == TrackState.Tracked:
      self.state = TrackState.Lost
  def markRemoved(self):
    self.state = TrackState.Removed
class ByteTrack3D:
  def __init__(self):
    self.trackedTracks: list[STrack3D] = []
    self.lostTracks: list[STrack3D] = []
    self.nextId: int = 1
    self.frameCount: int = 0
  @staticmethod
  def computeDistanceMatrix(tracks: list[STrack3D], detections: list[dict]) -> np.ndarray:
    if not tracks or not detections:
      return np.empty((len(tracks), len(detections)), dtype=np.float64)
    trackPos = np.array([t.kf.pos for t in tracks], dtype=np.float64)
    detPos = np.array([d["pos3d"] for d in detections], dtype=np.float64)
    diff = trackPos[:, None, :] - detPos[None, :, :]
    return np.linalg.norm(diff, axis=2)
  def associate(
    self,
    tracks: list[STrack3D],
    detections: list[dict],
    distThreshold: float,
  ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if not tracks or not detections:
      return [], list(range(len(tracks))), list(range(len(detections)))
    dMatrix = self.computeDistanceMatrix(tracks, detections)
    rowInd, colInd = linear_sum_assignment(dMatrix)
    matched: list[tuple[int, int]] = []
    unmatchedTracks = list(range(len(tracks)))
    unmatchedDets = list(range(len(detections)))
    for r, c in zip(rowInd, colInd):
      if dMatrix[r, c] <= distThreshold:
        matched.append((r, c))
        if r in unmatchedTracks:
          unmatchedTracks.remove(r)
        if c in unmatchedDets:
          unmatchedDets.remove(c)
    return matched, unmatchedTracks, unmatchedDets
  def update(self, candidates: list[dict]) -> list[dict]:
    self.frameCount += 1
    dHigh = [c for c in candidates if c.get("conf", 0.0) >= byteTrack3dTrackThresh]
    dLow = [
      c for c in candidates
      if byteTrack3dLowThresh <= c.get("conf", 0.0) < byteTrack3dTrackThresh
    ]
    allActiveTracks = self.trackedTracks + self.lostTracks
    for trk in allActiveTracks:
      trk.kf.predict()
    matched1, uTracks1, uDets1 = self.associate(
      allActiveTracks, dHigh, distThreshold=byteTrack3dMatchDist1
    )
    for tIdx, dIdx in matched1:
      det = dHigh[dIdx]
      allActiveTracks[tIdx].markTracked(
        pos=np.array(det["pos3d"], dtype=np.float64),
        score=det.get("conf", 0.8),
        numCameras=len(det.get("cameras", [1])),
      )
    unmatchedTrackedFromS1 = [
      allActiveTracks[i] for i in uTracks1
      if allActiveTracks[i].state == TrackState.Tracked
    ]
    matched2, uTracks2, _ = self.associate(
      unmatchedTrackedFromS1, dLow, distThreshold=byteTrack3dMatchDist2
    )
    for tIdx, dIdx in matched2:
      det = dLow[dIdx]
      unmatchedTrackedFromS1[tIdx].markTracked(
        pos=np.array(det["pos3d"], dtype=np.float64),
        score=det.get("conf", 0.4),
        numCameras=len(det.get("cameras", [1])),
      )
    for i in uTracks2:
      unmatchedTrackedFromS1[i].markLost()
    unmatchedTrackedIds = {id(t) for t in unmatchedTrackedFromS1}
    for i in uTracks1:
      trk = allActiveTracks[i]
      if id(trk) not in unmatchedTrackedIds:
        trk.markLost()
    for dIdx in uDets1:
      det = dHigh[dIdx]
      score = det.get("conf", 0.5)
      numCams = len(det.get("cameras", []))
      pos = np.array(det["pos3d"], dtype=np.float64)
      isInside = (
        (worldXMin - 1.5) <= pos[0] <= (worldXMax + 1.5) and
        (worldZMin - 1.5) <= pos[2] <= (worldZMax + 1.5)
      )
      if isInside and (score >= byteTrack3dNewTrackThresh or numCams >= minCamerasForConfirmation):
        newTrack = STrack3D(
          trackId=self.nextId,
          kf=KalmanFilter3D(pos),
          cls=det.get("cls", 0),
          score=score,
          numCameras=numCams,
        )
        self.nextId += 1
        newTrack.markTracked(pos, score, numCams)
        self.trackedTracks.append(newTrack)
    newTracked = []
    newLost = []
    for trk in self.trackedTracks + self.lostTracks:
      p = trk.kf.pos
      isOutside = (
        p[0] < (worldXMin - 3.0) or p[0] > (worldXMax + 3.0) or
        p[2] < (worldZMin - 3.0) or p[2] > (worldZMax + 3.0)
      )
      if isOutside:
        trk.markRemoved()
        continue
      isTransient = (trk.hits <= 1 and trk.numCameras < minCamerasForConfirmation)
      maxBuf = 2 if isTransient else byteTrack3dTrackBuffer
      if trk.timeSinceUpdate > maxBuf:
        trk.markRemoved()
      elif trk.state == TrackState.Tracked:
        newTracked.append(trk)
      elif trk.state == TrackState.Lost:
        newLost.append(trk)
    self.trackedTracks = newTracked
    self.lostTracks = newLost
    output = []
    for trk in self.trackedTracks + self.lostTracks:
      output.append({
        "trackId": trk.trackId,
        "pos": trk.kf.pos.tolist(),
        "vel": trk.kf.vel.tolist(),
        "cls": trk.cls,
        "score": round(trk.score, 2),
        "age": trk.age,
        "hits": trk.hits,
        "numCameras": trk.numCameras,
        "trail": trk.trail.copy(),
        "lost": (trk.state == TrackState.Lost),
      })
    return output
Tracker3D = ByteTrack3D