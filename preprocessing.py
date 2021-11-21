# pip install cvzone
# pip install mediapipe
import cv2
from cvzone.HandTrackingModule import HandDetector
from PIL import Image
import numpy as np

def detect_hand(img):
    detector = HandDetector(detectionCon=0.8, maxHands=2)
    # img = cv2.imread(img)

    hands = detector.findHands(img, draw=False)
    print(len(hands))

    if hands:
        # Hand 1
        # print(hands[0][0])
        hand1 = hands[0]

        bbox1 = hand1["bbox"]  # Bounding Box info x,y,w,h

        centerPoint1 = hand1["center"]  # center of the hand cx,cy
        handType1 = hand1["type"]  # Hand Type Left or Right

        fingers1 = detector.fingersUp(hand1)

        # if len(hands) == 2:
        #     hand2 = hands[1]
        #
        #     bbox2 = hand2["bbox"]  # Bounding Box info x,y,w,h
        #     centerPoint2 = hand2["center"]  # center of the hand cx,cy
        #     handType2 = hand2["type"]  # Hand Type Left or Right
        #
        #     fingers2 = detector.fingersUp(hand2)

        bbox = bbox1
        if len(hands) == 2:
            hand2 = hands[1]

            bbox2 = hand2["bbox"]  # Bounding Box info x,y,w,h

            # print(fingers1, fingers2)
            # length, info, img = detector.findDistance(lmList1&#91;8], lmList2&#91;8], img) # with draw
            bbox = [min(bbox1[0], bbox2[0]), min(bbox1[1], bbox2[1]), bbox1[2] + bbox2[2] + 20,
                    max(bbox1[3], bbox2[3]) + 20]
        startx = 0
        starty = 0
        endx = img.shape[0]
        endy = img.shape[1]

        if bbox[1] - 20 >= 0:
            starty = bbox[1] - 20
        if bbox[1] + bbox[3] + 20 < endy:
            endy = bbox[1] + bbox[3] + 20
        if bbox[0] - 20 >= 0:
            startx = bbox[0] - 20
        if bbox[0] + bbox[2] + 20 < endx:
            endx = bbox[0] + bbox[2] + 20

        crop_img = img[starty:endy, startx: endx]
        c=cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20), (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20), (0, 255, 0),
                      3)
        # cv2.imwrite("bbox.jpg",c)

        grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (5, 5), 2)
        th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        res = cv2.flip(res, 1)
        # cv2.imshow(res)
        grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        return res,c
    return None,None

def image_preprocessing(res):
  res=cv2.resize(res,(100,100),interpolation=cv2.INTER_CUBIC)
  test_image = res.reshape((100, 100, 1))
  test_image = np.expand_dims(res, axis = 0)
  # print(test_image.shape)
  return test_image
#
# d=detect_hand("pallav.jpg")
# print(d[1])
# cv2.imwrite("bbox.jpg",d[1])