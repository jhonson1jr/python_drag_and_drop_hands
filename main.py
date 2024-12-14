import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8) # definindo precisão

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1) # invertendo o video
    hands, img = detector.findHands(img)

    cv2.imshow('Image', img)
    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.waitKey(1)