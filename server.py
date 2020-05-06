from imutils import build_montages
from datetime import datetime
from imagezmq import imagezmq
import numpy as np
import argparse
import imutils
import cv2
import requests

url = "http://192.168.43.204:8080/shot.jpg"
rpiName = "test"


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
    help="prototxt path")
ap.add_argument("-m", "--model", required=True,
    help="model path")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
    help="minimum probability to filter weak detections")
ap.add_argument("-mW", "--montageW", required=True, type=int,
    help="5")
ap.add_argument("-mH", "--montageH", required=True, type=int,
    help="5")
args = {
    "prototxt": r"C:\Users\ASUS\Documents\Kuliah\Semester 8\Skripsi\imagezmq-streaming\MobileNetSSD_deploy.prototxt",
    "model": r"C:\Users\ASUS\Documents\Kuliah\Semester 8\Skripsi\imagezmq-streaming\MobileNetSSD_deploy.caffemodel",
    "montageW" : 5,
    "montageH" : 5,
    "confidence": 0.2
}

imageHub = imagezmq.ImageHub()

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

CONSIDER = set(["motorbike", "car"])
objCount = {obj: 0 for obj in CONSIDER}
frameDict = {}

lastActive = {}
lastActiveCheck = datetime.now()

ESTIMATED_NUM_PIS = 4
ACTIVE_CHECK_PERIOD = 10
ACTIVE_CHECK_SECONDS = ESTIMATED_NUM_PIS * ACTIVE_CHECK_PERIOD

mW = args["montageW"]
mH = args["montageH"]
print("[INFO] detecting: {}...".format(", ".join(obj for obj in
    CONSIDER)))

while True:
    # (rpiName, frame) = imageHub.recv_image()
    # imageHub.send_reply(b'OK')
    #
    # if rpiName not in lastActive.keys():
    # 	print("[INFO] receiving data from {}...".format(rpiName))
    #
    # lastActive[rpiName] = datetime.now()

    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    input = cv2.imdecode(img_arr, -1)

    frame = imutils.resize(input, width=400)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
        0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    objCount = {obj: 0 for obj in CONSIDER}

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:
            idx = int(detections[0, 0, i, 1])

            if CLASSES[idx] in CONSIDER:
                objCount[CLASSES[idx]] += 1
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    (255, 0, 0), 2)

    cv2.putText(frame, rpiName, (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    label = ", ".join("{}: {}".format(obj, count) for (obj, count) in
        objCount.items())
    cv2.putText(frame, label, (10, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 2)

    frameDict[rpiName] = frame

    montages = build_montages(frameDict.values(), (w, h), (mW, mH))

    for (i, montage) in enumerate(montages):
        cv2.imshow("Sistem Pendeteksi Kemacetan Jalanan ({})".format(i),
            montage)

    key = cv2.waitKey(1) & 0xFF

    if (datetime.now() - lastActiveCheck).seconds > ACTIVE_CHECK_SECONDS:
        for (rpiName, ts) in list(lastActive.items()):
            if (datetime.now() - ts).seconds > ACTIVE_CHECK_SECONDS:
                print("[INFO] lost connection to {}".format(rpiName))
                lastActive.pop(rpiName)
                frameDict.pop(rpiName)

        lastActiveCheck = datetime.now()

    if key == ord("q"):
        break

cv2.destroyAllWindows()