import cv2
import numpy as np
import imutils

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
IMG_WIDTH = 300
IMG_HEIGHT = 300
FRAME_WIDTH = 950
COLOR_RED = (0, 0, 255)

net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "weights.caffemodel")


def face_detection(frame):
    frame = imutils.resize(frame, width=FRAME_WIDTH)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < CONF_THRESHOLD:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        face = frame[startY:endY, startX:endX]
        return face
