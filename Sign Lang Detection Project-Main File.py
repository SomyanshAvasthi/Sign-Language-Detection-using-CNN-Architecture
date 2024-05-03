import cv2
from cvzone.HandTrackingModule import HandDetector
import tensorflow as tf
from keras.models import load_model
import numpy as np
import math
from keras.utils.image_utils import img_to_array
from keras.optimizers import Adam
import time
adam = Adam(learning_rate=0.00001)

map_characters = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9', 9: 'A', 10: 'B', 11: 'C', 12: 'D',
                  13: 'E', 14: 'F', 15: 'G', 16: 'H', 17: 'I', 18: 'J', 19: 'K', 20: 'L', 21: 'M', 22: 'N', 23: 'O', 24: 'P',
                  25: 'Q', 26: 'R', 27: 'S', 28: 'T', 29: 'U', 30: 'V', 31: 'W', 32: 'X', 33: 'Y', 34: 'Z'}


imgpath=cv2.imread("SignLanguage/Indian/2/0.jpg")



def edge_detection(image):
    minValue = 70
    blur = cv2.GaussianBlur(image,(5,5),2)
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return res

def preprocessor_predict(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = edge_detection(img)
    img = img_to_array(img)
    cv2.imshow("Imgpreprocess",img)
 
    img = cv2.resize(img, (64,64))
    img = img.reshape(-1, 64, 64, 1)
    # imglist=[img,[]]
    
    return predictor(img)
    


def predictor(img):
    model=load_model('SignLanguage/ISL_Predictor.h5',compile=False)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    pred=model.predict(img)
    return pred









cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
detector=HandDetector(maxHands=2)
offset=15
imgSize=300



while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    cv2.imshow("Originalimg",img)
    if hands:
        hand=hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap: wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize
        cv2.imshow("ImgWhite",imgWhite)
        predtor=preprocessor_predict(imgWhite)
        pred_classes = np.argmax(predtor,axis = 1)
        print(map_characters.get(pred_classes[0]))
        

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()