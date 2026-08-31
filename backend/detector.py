import numpy as np
import cv2
import os
from pathlib import Path
from config import (
  yoloModelPath,
  yoloConfThreshold,
  yoloTargetClasses,
  cameraIds,
)
sessionInstance = None
inputName = None
outputName = None
camTrackCounters: dict[str, int] = {cam: 1 for cam in cameraIds}
def getSession():
  global sessionInstance, inputName, outputName
  if sessionInstance is None:
    import onnxruntime as ort
    providers = [("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_mem_pattern = True
    so.intra_op_num_threads = 6
    so.inter_op_num_threads = 6
    sessionInstance = ort.InferenceSession(str(yoloModelPath), sess_options=so, providers=providers)
    inputName = sessionInstance.get_inputs()[0].name
    outputName = sessionInstance.get_outputs()[0].name
    print(f"Loading YOLO Model From {yoloModelPath} On AMD GPU DirectML")
  return sessionInstance
DetectionList = list[dict]
def detectAndTrack(frameBgr: np.ndarray, camId: str) -> DetectionList:
  session = getSession()
  origH, origW = frameBgr.shape[:2]
  scale = min(640.0 / origH, 640.0 / origW)
  newW, newH = int(origW * scale), int(origH * scale)
  resized = cv2.resize(frameBgr, (newW, newH))
  canvas = np.zeros((640, 640, 3), dtype=np.uint8)
  canvas[:newH, :newW] = resized
  rgb = canvas[:, :, ::-1].transpose(2, 0, 1)
  tensor = np.ascontiguousarray(rgb[None, ...], dtype=np.float32) / 255.0
  outputs = session.run(None, {inputName: tensor})[0][0]
  detections: DetectionList = []
  trackCounter = camTrackCounters[camId]
  for box in outputs:
    score = float(box[4])
    clsId = int(box[5])
    if score < yoloConfThreshold:
      continue
    if clsId not in yoloTargetClasses:
      continue
    x1 = max(0.0, min(float(origW), float(box[0]) / scale))
    y1 = max(0.0, min(float(origH), float(box[1]) / scale))
    x2 = max(0.0, min(float(origW), float(box[2]) / scale))
    y2 = max(0.0, min(float(origH), float(box[3]) / scale))
    footU = (x1 + x2) / 2.0
    footV = y2
    detections.append({
      "camTrackId": trackCounter,
      "camId": camId,
      "boxXyxy": [x1, y1, x2, y2],
      "footUv": (footU, footV),
      "conf": score,
      "cls": clsId,
    })
    trackCounter += 1
  camTrackCounters[camId] = trackCounter
  return detections
def buildBatchTensor(framesPerCam: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str], dict[str, tuple[int, int]], dict[str, float]]:
  session = getSession()
  inputDim = int(session.get_inputs()[0].shape[2])
  camList = list(framesPerCam.keys())
  if not camList:
    return np.zeros((0, 3, inputDim, inputDim), dtype=np.float32), [], {}, {}
  batchCount = len(camList)
  batchTensor = np.zeros((batchCount, 3, inputDim, inputDim), dtype=np.float32)
  scales = {}
  origShapes = {}
  firstFrame = framesPerCam[camList[0]]
  origH, origW = firstFrame.shape[:2]
  scale = min(float(inputDim) / origH, float(inputDim) / origW)
  newW, newH = int(origW * scale), int(origH * scale)
  framesList = [framesPerCam[c] for c in camList]
  blob = cv2.dnn.blobFromImages(framesList, scalefactor=1.0 / 255.0, size=(newW, newH), swapRB=True, crop=False)
  batchTensor[:, :, :newH, :newW] = blob
  for camId in camList:
    origShapes[camId] = (origH, origW)
    scales[camId] = scale
  return batchTensor, camList, origShapes, scales
ioBindingInstance = None
def getIoBinding(session):
  global ioBindingInstance
  if ioBindingInstance is None:
    ioBindingInstance = session.io_binding()
  return ioBindingInstance
def runInferenceOnBatchTensor(batchTensor: np.ndarray, camList: list[str], origShapes: dict[str, tuple[int, int]], scales: dict[str, float]) -> dict[str, DetectionList]:
  if not camList:
    return {}
  session = getSession()
  ioBinding = getIoBinding(session)
  ioBinding.bind_cpu_input(inputName, batchTensor)
  ioBinding.bind_output(outputName)
  session.run_with_iobinding(ioBinding)
  outputs = ioBinding.copy_outputs_to_cpu()[0]
  results: dict[str, DetectionList] = {}
  for i, camId in enumerate(camList):
    origH, origW = origShapes[camId]
    scale = scales[camId]
    camOutputs = outputs[i]
    detections: DetectionList = []
    trackCounter = camTrackCounters[camId]
    for box in camOutputs:
      score = float(box[4])
      clsId = int(box[5])
      if score < yoloConfThreshold or clsId not in yoloTargetClasses:
        continue
      x1 = max(0.0, min(float(origW), float(box[0]) / scale))
      y1 = max(0.0, min(float(origH), float(box[1]) / scale))
      x2 = max(0.0, min(float(origW), float(box[2]) / scale))
      y2 = max(0.0, min(float(origH), float(box[3]) / scale))
      footU = (x1 + x2) / 2.0
      footV = y2
      detections.append({
        "camTrackId": trackCounter,
        "camId": camId,
        "boxXyxy": [x1, y1, x2, y2],
        "footUv": (footU, footV),
        "conf": score,
        "cls": clsId,
      })
      trackCounter += 1
    camTrackCounters[camId] = trackCounter
    results[camId] = detections
  return results
def detectAndTrackBatch(framesPerCam: dict[str, np.ndarray]) -> dict[str, DetectionList]:
  batchTensor, camList, origShapes, scales = buildBatchTensor(framesPerCam)
  return runInferenceOnBatchTensor(batchTensor, camList, origShapes, scales)
def loadFrame(camId: str, frameIdx: int, imageDir) -> np.ndarray | None:
  camDir = Path(imageDir) / camId
  candidates = sorted(camDir.glob("*.png")) + sorted(camDir.glob("*.jpg"))
  if frameIdx >= len(candidates):
    return None
  img = cv2.imread(str(candidates[frameIdx]))
  return img