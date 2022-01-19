import dlib
import cv2
import os
import random
import time
import numpy as np
from tqdm import tqdm
from imutils import face_utils 

def mask_point_detection(image, detector, predictor): 
    """
    OpenCV ,dlib library를  이용해 이미지의 마스크영역 검출
    Args:
      image : 원본 이미지
      detector, predictor : dlib class, file
    Return:
      point : 마스크영역
    """
  
    image = cv2.imread(image) 
    image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
      shape = predictor(gray, rect) 
      shape = face_utils.shape_to_np(shape)
      
      (x, y, w, h) = face_utils.rect_to_bb(rect)
      cv2.rectangle(image, (shape[2][0]-30, shape[2][1]-30), (shape[16][0]+30, shape[9][1]+30), (0, 255, 0), 1)
      [x_left, y_left, x_right, y_right] =  shape[2][0]-25, shape[2][1]-25, shape[16][0]+25, shape[9][1]+25
      point = [x_left, y_left, x_right, y_right]  # box point
      
      for (x, y) in shape: 
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    return point

def image_processing(image, mask_image, point, maskname):
    """
    원본 이미지, 마스크를 씌운 합성이미지, Binary이미지를 PATH에 저장
    Args:
      image : 원본 이미지
      mask_image : 마스크 객체 이미지
      point : 얼굴인식 point
      maskname : 이미지 이름 (검은색, 파란색, 흰색)
    """
    image = cv2.imread(image) 
    mask_image = cv2.imread(mask_image)
    
    save_train_dir = PATH+'/train'
    
    x_left, y_left, x_right, y_right = point[0], point[1], point[2], point[3]

    mask_threshold = {'black1.jpeg':230, 'black2.jpeg':230,'white1.jpeg':50,
                      'white2.jpeg':242, 'blue1.jpeg':240}
    
    if maskname in ['black1.jpeg', 'black2.jpg', 'white2.jpeg', 'blue1.jpeg']:
        image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_AREA)
        original_image = image.copy()
        mask_image = cv2.resize(mask_image, dsize=(x_right-x_left, y_right-y_left), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)   #마스크이미지의 색상을 그레이로 변경
        ret, mask = cv2.threshold(gray, mask_threshold[maskname], 255, cv2.THRESH_BINARY) #배경은 흰색으로, 그림을 검정색으로 변경
        mask_inv = cv2.bitwise_not(mask)
        # cv2_imshow(mask) 
        # cv2_imshow(mask_inv) 

        crop = image[y_left:y_right, x_left:x_right]          
        mask_face_crop = cv2.copyTo(mask_image, mask_inv, crop)   # black mask > mask_inv대신
        image[y_left:y_right, x_left:x_right] = mask_face_crop 

        binary_image= np.zeros([image.shape[0], image.shape[1]], dtype=np.uint8)
        binary_image[y_left:y_right, x_left:x_right] = mask_inv

    elif maskname in ['white1.jpeg']:
        image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_AREA)
        original_image = image.copy()
        mask_image = cv2.resize(mask_image, dsize=(x_right-x_left, y_right-y_left), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY) 
        ret, mask = cv2.threshold(gray, mask_threshold[maskname], 255, cv2.THRESH_BINARY) 
        mask_inv = cv2.bitwise_not(mask)

        crop = image[y_left:y_right, x_left:x_right]
        mask_face_crop = cv2.copyTo(mask_image, mask, crop)    # white mask >  mask
        image[y_left:y_right, x_left:x_right] = mask_face_crop 

        binary_image= np.zeros([image.shape[0], image.shape[1]], dtype=np.uint8)
        binary_image[y_left:y_right, x_left:x_right] = mask

    binary_image = binary_image.reshape(224, 224, 1)
    binary_image1 = np.concatenate([binary_image, binary_image], axis=2)
    binary_image2 = np.concatenate([binary_image1, binary_image], axis=2) # 3channel binary image
    merge_image = np.concatenate([original_image, image],axis = 1)        # 원본, 합성이미지 concat
    merge_image = np.concatenate([merge_image, binary_image2],axis = 1)    # merge image, 3channel binary image concat
    
    # cv2.imshow("mask_image", mask_image)
    # cv2.imshow("mask_inv", mask_inv)     
    # cv2.imshow("mask_face_crop", mask_face_crop)
    # cv2.imshow("merge_image", merge_image)
    # cv2.imshow("binary_image", binary_image)
    
    # cv2.waitKey(0)

    #  image save
    if not os.path.exists(save_train_dir):
        os.mkdir(save_train_dir)
    cv2.imwrite(save_train_dir+'/train{num}.jpg'.format(num=i+1), merge_image)  
    

random.seed(42)

# face, mask image PATH
PATH = os.getcwd()
face_path = PATH+'/image'
mask_path = PATH+'/mask'

face_file_list = os.listdir(face_path)  #[:30000]
mask_file_list = os.listdir(mask_path)
face_file_list = sorted(face_file_list)
mask_file_list = sorted(mask_file_list)

fail_image=[]

detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor(PATH + '/shape_predictor_68_face_landmarks.dat')

for i, imagename in tqdm(enumerate(face_file_list), total=len(face_file_list), desc='save', ncols=80):
    image = face_path+'/'+imagename
    maskname = random.choice(mask_file_list)       # mask image rondom choice
    mask_image = mask_path+'/'+maskname
    try:
        point = mask_point_detection(image, detector, predictor)
        image_processing(image, mask_image, point, maskname)
    except (UnboundLocalError, ValueError):        # 마스크검출 영역이 마스크 사이즈 보다 작은면 pass
        fail_image.append(imagename)
        
    time.sleep(0.15)