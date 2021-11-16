import numpy as np
import pandas as pd
import cv2.cv2 as cv2
import os
from dlib import dlib
from PIL import Image, ImageTk
from PIL import Image, ImageDraw, ImageFont

def faces_recognition():
    global detector,predictor
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW) # 連接網路攝影機
    #count_f = 0 # 計算圖片數量
   # tmp_for_distance = [] # 儲存某個人的臉部資訊
            
    while True:
        bgr_image = capture.read()[1] # 讀取三維圖        
        facesInformation = [] # 儲存某個人的臉部資訊
        rgb_image = cv2.cvtColor(bgr_image,cv2.COLOR_BGR2RGB)
        face_locations = detector(rgb_image, 0)
        
        for i in range(len(face_locations)):
            shape = predictor(rgb_image,face_locations[i]) #取得68輪廓點
            facesInformation.append(["unkown",face_locations[i],shape])
            # 先 取人臉
            # 改人臉大小
            #face_image = Image.fromarray(rgb_image) # 將OpenCV圖檔轉換成PIL
            #faceTK = ImageTk.PhotoImage(image=face_image) # 轉換成ImageTK
            point1 = (facesInformation[i][1].left(), facesInformation[i][1].top()) #左上座標
            point2 = (facesInformation[i][1].right(), facesInformation[i][1].bottom()) #右下座標   
            # 後 畫框框
            try:
                for j in range(68):
                    cv2.circle(rgb_image,(facesInformation[i][2].part(j).x,facesInformation[i][2].part(j).y),2,(0,255,0), -1, 3)
                    #print(facesInformation[i][2].part(i).x)
            except:
                pass
            cv2.rectangle(rgb_image, point2,point1, (255,255,0), 2)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR) #將視窗轉換成BGR(RGB變換成BGR)
        cv2.imshow("easymode", bgr_image)
        if cv2.waitKey(1) & 0xFF == (ord('q') or ord('Q')):
            break
            
        if cv2.getWindowProperty("easymode", 0) < 0:
            break

    capture.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    detector = dlib.get_frontal_face_detector() #演算法為HOG，用途是找出圖片中的人臉位置
    predictor = dlib.shape_predictor('./recognition_model/shape_predictor_68_face_landmarks.dat') # 從人臉中找出68個輪廓點
    faces_recognition()