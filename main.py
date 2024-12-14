import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(maxHands=4, detectionCon=0.8) # definindo precisão
cor_retangulo = 255, 0 ,255

cx, cy, w ,h = 100, 100, 200, 200
while True:
    success, img = cap.read()
    #img = cv2.flip(img, 1) # invertendo o video
    hands, img = detector.findHands(img, draw=True, flipType=True)
    # print(hands)
    if hands: # se tem coordenadas de mão identificados
        for hand in hands:
            if hand['lmList']:
                lmList = hand['lmList']
                x8, y8 = lmList[8][0], lmList[8][1] # Ponta do dedo indicador (landmark 8)
                x12, y12 = lmList[12][0], lmList[12][1] # Ponta do dedo médio (landmark 12)
                distanciaDedos, _, _ = detector.findDistance((x8, y8), (x12, y12), img) # analisando a distancia entre os dedos indicador e medio
                cursor = lmList[8] # posição correspondente a ponta do dedo indicador
                if distanciaDedos < 30 : # dedos aproximados será o nosso "click"
                    if cx-w//2 < cursor[0] < cx+w//2 and cy-h//2 < cursor[1] < cy+h//2:
                        cor_retangulo = 0, 255, 0
                        cx, cy, _ = cursor
                    else:
                        cor_retangulo = 255, 0, 255

    imgCopia = img.copy()
    cv2.rectangle(imgCopia, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), cor_retangulo, cv2.FILLED)
    alpha = 0.6  # Nível de transparência (0.0 a 1.0)
    cv2.addWeighted(imgCopia, alpha, img, 1 - alpha, 0, img)     # Mescla o retângulo transparente na imagem original
    cv2.imshow('Image', img)
    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.waitKey(1)