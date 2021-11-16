import numpy as np
import pandas as pd
import cv2.cv2 as cv2
import os
from dlib import dlib
from PIL import Image, ImageTk
from PIL import Image, ImageDraw, ImageFont
def read_Data():
    known = pd.read_csv("faces.csv",encoding="utf-8")
    if os.path.exists("faces.csv"):
        known = pd.read_csv("faces.csv",encoding="utf-8")
        knownFeature = []
        for i in range(len(known)):
            try:
                feature = known['特徵'][i].replace('\n','').replace('[','').replace(',','').replace(']','').split()
                feature = list(map(eval,feature))
                knownFeature.append([known['姓名'][i],feature])
                #print(knownFeature)
            except:
                pass
    return knownFeature

def faces_recognition():
    def cv2ImgAddText(img, text, left, top, textColor, textSize=20):
        if (isinstance(img, np.ndarray)):  #判斷是否OpenCV圖片類型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        fontText = ImageFont.truetype( 
                "font/mingliu.ttc", textSize, encoding="utf-8")
        textColor=(textColor[2],textColor[1],textColor[0])
        draw.text( (left, top) , text , textColor , font=fontText)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    def return_euclidean_distance(feature_1, feature_2): 
        #print(type(feature_2))
        #print(feature_2)
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist
    global detector,predictor,facerec,knownFeature
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
            facesInformation.append(["unkown",face_locations[i],facerec.compute_face_descriptor(rgb_image, shape)]) #取得128D特徵值
        if face_locations != []:
            for j in range(len(facesInformation)):
                for k in range(len(knownFeature)):                 
                    # 如果 person_X 資料不為空值
                    if str(knownFeature[i][0]) != " ":
                        e_distance = return_euclidean_distance(facesInformation[j][2], knownFeature[k][1]) # 計算歐基里德距離
                        #print(e_distance_tmp)
                        if e_distance < 0.3: # 提早結束搜尋
                            facesInformation[j][0] = knownFeature[k][0]
                            print("可能是:",facesInformation[j][0])
                        else:    
                            e_distance = -1
                    else:
                        break
                # 先 取人臉
                # 改人臉大小
                #face_image = Image.fromarray(rgb_image) # 將OpenCV圖檔轉換成PIL
                #faceTK = ImageTk.PhotoImage(image=face_image) # 轉換成ImageTK
                point1 = (facesInformation[j][1].left(), facesInformation[j][1].top()) #左上座標
                point2 = (facesInformation[j][1].right(), facesInformation[j][1].bottom()) #右下座標   
                # 後 畫框框
                cv2.rectangle(rgb_image, point2,point1, (255,255,0), 2)
                rgb_image = cv2ImgAddText(rgb_image, facesInformation[j][0], 
                                    facesInformation[j][1].left() * 1.1 ,facesInformation[j][1].top() * 0.85, (255,255,0), 25)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR) #將視窗轉換成BGR(RGB變換成BGR)
        cv2.imshow("簡單版", bgr_image)
        if cv2.waitKey(1) & 0xFF == (ord('q') or ord('Q')):
            break
            
        if cv2.getWindowProperty("簡單版", 0) < 0:
            break

    capture.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    knownFeature = read_Data()
    detector = dlib.get_frontal_face_detector() #演算法為HOG，用途是找出圖片中的人臉位置
    predictor = dlib.shape_predictor('./recognition_model/shape_predictor_68_face_landmarks.dat') # 從人臉中找出68個輪廓點
    facerec = dlib.face_recognition_model_v1("./recognition_model/dlib_face_recognition_resnet_model_v1.dat")
    faces_recognition()