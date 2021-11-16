import numpy as np
import pandas as pd
import cv2.cv2 as cv2
import os
import shutil
import datetime
from dlib import dlib
from PIL import Image, ImageTk
from PIL import Image, ImageDraw, ImageFont
from skimage import io


def add_new(name,content="",face_name="",student_class=""):
    try:
        stu_data = pd.read_csv(name+'.csv',encoding='utf-8')
        stu_data = stu_data.append({"姓名":face_name,"特徵":content},ignore_index=True)
        stu_data.to_csv("faces.csv",index=False,encoding='utf_8_sig')
        print("新增成功")
    except:
        pass

def train_face():
    #last_ID = csv_file[csv_file["學生姓名"]==content.get()]["學生ID"][0]    
    # Get the mean/average features of face/personX, it will be a list with a length of 128D
    while True:
        face_name = input("請輸入姓名:")
        if not os.path.exists("./faces/train/"+face_name):
            os.mkdir("./faces/train/"+face_name)
            break
        else:
            print("已建立過資料夾，請重新輸入")
    faces_recognition(face_name)
    features_mean_personX = return_features_mean_personX('./faces/train/' + face_name)
    add_new('faces',list(features_mean_personX),face_name)
    

def return_128d_features(path_img):
    global facerec,predictor
    img_rd = io.imread(path_img)
    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB)
    img_gray=cv2.resize(img_gray,(0, 0), fx=0.5, fy=0.5)
    faces = detector(img_gray, 1)
    print(faces)
    print("%-40s %-20s" % ("檢測到人臉的圖像 / image with faces detected:", path_img), '\n')

    # 因為有可能攔截下來的人臉再去檢測，檢測不出來人臉了
    # 所以要確保是檢測到人臉的人臉圖像拿去算特徵
    if len(faces) != 0:
        shape = predictor(img_gray, faces[0])
        face_descriptor = facerec.compute_face_descriptor(img_gray, shape)
    else:
        face_descriptor = 0
        print("no face")

    return face_descriptor

def return_features_mean_personX(path_faces_personX):
    features_list_personX = []
    photos_list = os.listdir(path_faces_personX)
    if photos_list:
        for i in range(len(photos_list)):
            # 調用return_128d_features()得到128d特徵
            print("%-40s %-20s" % ("正在讀的人臉圖像 / image to read:", path_faces_personX + "/" + photos_list[i]))
            features_128d = return_128d_features(path_faces_personX + "/" + photos_list[i])
            #  print(features_128d)
            # 遇到沒有檢測出人臉的圖片跳過
            if features_128d == 0:
                i += 1
            else:
                features_list_personX.append(features_128d)
    else:
        print("文件夾內圖像文件為空 / Warning: No images in " + path_faces_personX + '/', '\n')

    # 計算128D特徵的均值
    # personX的N張圖像x 128D-> 1 x 128D
    if features_list_personX:
        features_mean_personX = np.array(features_list_personX).mean(axis=0)
    else:
        features_mean_personX = np.array(['0'])

    return features_mean_personX 

def faces_recognition(face_name):
    global detector
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW) # 連接網路攝影機
    getFace_count_start = datetime.datetime.now()
    face_count = 0
    while True:
        getFace_count_end = datetime.datetime.now()
        bgr_image = capture.read()[1] # 讀取三維圖        
        facesInformation = [] # 儲存某個人的臉部資訊
        rgb_image = cv2.cvtColor(bgr_image,cv2.COLOR_BGR2RGB)
        face_locations = detector(rgb_image, 0)
        
        for i in range(len(face_locations)):
            facesInformation.append(face_locations[i])
            point1 = (facesInformation[i].left(), facesInformation[i].top()) #左上座標
            point2 = (facesInformation[i].right(), facesInformation[i].bottom()) #右下座標   
            # 後 畫框框
            cv2.rectangle(rgb_image, point2,point1, (255,255,0), 2)
            face_image = rgb_image[face_locations[i].top():face_locations[i].bottom(),face_locations[i].left():face_locations[i].right()]
            pil_image = Image.fromarray(face_image)
            if (getFace_count_end - getFace_count_start) > datetime.timedelta(milliseconds=500):
                face_count += 1
                t = datetime.datetime.now().strftime('%Y_%m_%d_%H-%M-%S-%f')
                pil_image.save('faces/train/'+face_name+'/'+ t + '.png')
                getFace_count_start = getFace_count_end

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR) #將視窗轉換成BGR(RGB變換成BGR)
        cv2.imshow("easymode", bgr_image)
        if cv2.waitKey(1) & 0xFF == (ord('q') or ord('Q')):
            break
        
        if cv2.getWindowProperty("easymode", 0) < 0:
            break
        if face_count > 10:
            break
if __name__ == '__main__':
    detector = dlib.get_frontal_face_detector() #演算法為HOG，用途是找出圖片中的人臉位置
    predictor = dlib.shape_predictor('./recognition_model/shape_predictor_68_face_landmarks.dat') # 從人臉中找出68個輪廓點
    facerec = dlib.face_recognition_model_v1("./recognition_model/dlib_face_recognition_resnet_model_v1.dat")
    train_face()