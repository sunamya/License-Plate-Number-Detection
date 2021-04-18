#!/usr/bin/env python
# coding: utf-8
import cv2
import numpy as np
import imutils
from keras.models import load_model
from keras.models import model_from_json

characters = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

json_file = open('MobileNets_character_recognition.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights('model.h5')

model = loaded_model

# # Step 1: Identifying Location of the plate
img=cv2.imread('germany_car_plate.jpg')
img = cv2.resize(img, (620,480) )
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
gray1 = cv2.bilateralFilter(gray, 13,15,15)
edged = cv2.Canny(gray, 30,200)
cnts, new  = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
img1 = img.copy()
cv2.drawContours(img1, cnts, -1, (0,255,0), 3)

cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:50]
NumberPlateCnt = 0 
img2 = img.copy()
cv2.drawContours(img2, cnts, -1, (0,255,0), 3)

count = 0
idx =7
for c in cnts:
        peri = cv2.arcLength(c, True)
        box = cv2.approxPolyDP(c,  0.02*peri, True)
        if len(box) == 4:  
            NumberPlateCnt = box 
            x, y, w, h = cv2.boundingRect(c) 
            new_img = gray[y:y + h, x:x + w] 
            idx+=1
            cv2.drawContours(img, [NumberPlateCnt], -1, (0,255,0), 3)
            break

mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[NumberPlateCnt],0,255,-1,)
new_image = cv2.bitwise_and(img,img,mask=mask)

# Variable Cropped contains the identified license plate

(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = gray[topx:bottomx+1, topy:bottomy+1]

# # Step 2: Identifying individual character on plate i.e Character Segmentation

blur = cv2.GaussianBlur(Cropped,(7,7),0)
binary = cv2.threshold(blur, 180, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),key=lambda b: b[1][i], reverse=reverse))
    return cnts
digit_w, digit_h = 50,100

# # Step 3: Using Deep Learning Model for Character Recognition
def detect(image):
    height, width = image.shape
    image = cv2.resize(image, dsize=(width*3,height*2), interpolation=cv2.INTER_CUBIC)
    ret,thresh = cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)
    kernel = np.ones((5,5), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    m = list()
    pchl = list()
    gsblur=cv2.GaussianBlur(img_dilation,(5,5),0)
    roi = cv2.resize(gsblur, dsize=(28,28), interpolation=cv2.INTER_CUBIC)
    roi = np.array(roi)
    t = np.copy(roi)
    t = t / 255.0
    t = 1-t
    t = t.reshape(1,784)
    m.append(roi)
    pred = model.predict_classes(t)
    pchl.append(pred)
    pcw = list()
    interp = 'bilinear'
    for i in range(len(pchl)):
        pcw.append(characters[pchl[i][0]])
    predstring = ''.join(pcw)
    return predstring


# # Final Function to identify the individual characters on plate 
st=list()
for c in sort_contours(cont):
    (x, y, w, h) = cv2.boundingRect(c)
    ratio = h/w
    if ratio<=4: 
        if h/Cropped.shape[0]>=0.3:
            curr_num = binary[y:y+h,x:x+w]
            curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
            _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            st.append(detect(curr_num))
st1=''
st1=st1.join(st)
print("Number is : ",st1)
