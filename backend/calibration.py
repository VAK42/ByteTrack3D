import xml.etree.ElementTree as ET
import numpy as np
import cv2
from typing import Optional
from pathlib import Path
from config import (
  extrinsicDir,
  intrinsicDir,
  cameraIds,
  cameraCalibMap,
)
def parseXmlMatrix(filePath: Path, tag: str) -> np.ndarray:
  tree = ET.parse(filePath)
  root = tree.getroot()
  node = root.find(tag)
  if node is None:
    raise ValueError(f"Tag <{tag}> Not Found In {filePath}")
  dataNode = node.find("data")
  text = dataNode.text if dataNode is not None else node.text
  if text is None:
    raise ValueError(f"Empty Data In Tag <{tag}> In {filePath}")
  values = [float(x) for x in text.strip().split()]
  return np.array(values, dtype=np.float64)
def loadCameraCalibration(camId: str) -> dict:
  calibName = cameraCalibMap.get(camId, camId)
  extrFile = extrinsicDir / f"extr{calibName}.xml"
  intrFile = intrinsicDir / f"intr{calibName}.xml"
  rvec = parseXmlMatrix(extrFile, "rvec").reshape(3, 1)
  tvec = parseXmlMatrix(extrFile, "tvec").reshape(3, 1)
  tvecM = tvec / 100.0 if np.max(np.abs(tvec)) > 50.0 else tvec.copy()
  rMatrix, _ = cv2.Rodrigues(rvec)
  K = parseXmlMatrix(intrFile, "camera_matrix").reshape(3, 3)
  rt = np.hstack([rMatrix, tvecM])
  pMatrix = K @ rt
  return {
    "camId": camId,
    "K": K,
    "R": rMatrix,
    "rvec": rvec,
    "t": tvecM,
    "P": pMatrix,
  }
def loadAllCalibrations() -> dict[str, dict]:
  calibrations = {}
  for camId in cameraIds:
    try:
      calibrations[camId] = loadCameraCalibration(camId)
    except Exception as e:
      print(f"Could Not Load Calibration For {camId}: {e}")
  return calibrations
def computeGroundHomographies(calibrations: dict[str, dict]) -> dict[str, np.ndarray]:
  homographies = {}
  for camId, cal in calibrations.items():
    pMatrix = cal["P"]
    hMatrix = np.column_stack([pMatrix[:, 0], pMatrix[:, 1], pMatrix[:, 3]])
    homographies[camId] = hMatrix
  return homographies
def projectToGround(footUv: tuple[float, float], hMatrix: np.ndarray) -> np.ndarray:
  pVec = np.array([footUv[0], footUv[1], 1.0], dtype=np.float64)
  try:
    hInv = np.linalg.inv(hMatrix)
    gp = hInv @ pVec
    if abs(gp[2]) > 1e-6:
      return gp[:2] / gp[2]
  except np.linalg.LinAlgError:
    pass
  return np.array([0.0, 0.0], dtype=np.float64)
def triangulate3d(
  imagePoints: dict[str, tuple[float, float]],
  calibrations: dict[str, dict],
) -> Optional[np.ndarray]:
  camIds = [c for c in imagePoints.keys() if c in calibrations]
  if len(camIds) < 2:
    return None
  if len(camIds) == 2:
    p1 = calibrations[camIds[0]]["P"]
    p2 = calibrations[camIds[1]]["P"]
    pt1 = np.array(imagePoints[camIds[0]], dtype=np.float64).reshape(2, 1)
    pt2 = np.array(imagePoints[camIds[1]], dtype=np.float64).reshape(2, 1)
    x4 = cv2.triangulatePoints(p1, p2, pt1, pt2)
    if abs(x4[3, 0]) > 1e-6:
      return (x4[:3, 0] / x4[3, 0]).flatten()
    return None
  aRows = []
  for camId in camIds:
    pMatrix = calibrations[camId]["P"]
    u, v = imagePoints[camId]
    aRows.append(u * pMatrix[2] - pMatrix[0])
    aRows.append(v * pMatrix[2] - pMatrix[1])
  aMatrix = np.array(aRows, dtype=np.float64)
  _, _, vt = np.linalg.svd(aMatrix)
  x4 = vt[-1]
  if abs(x4[3]) > 1e-6:
    return x4[:3] / x4[3]
  return None