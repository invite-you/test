# Install required libs

### please update Albumentations to version>=0.3.0 for `Lambda` transform support
!pip install -U albumentations>=0.3.0
!pip install segmentation-models


#!pip install -U torch torchvision cython
#!pip install -U 'git+https://github.com/facebookresearch/fvcore.git' 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
###################

import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt

import segmentation_models as sm

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
    

####################

def predict_TTA(model, image):
    def predict_rotatationt90_image(model, image, times=1):
        image90 = np.rot90(image, times)
        pr_mask = model.predict(np.expand_dims(image90, axis = 0)).squeeze()
        return np.rot90(pr_mask, -times)
    pred_masks = []
    for i in range(1,4):
        pred_masks.append(predict_rotatationt90_image(model, image, times=i))
    
    image_flip = np.flip(image, 0)
    for i in range(1,4):
        pre_mask = predict_rotatationt90_image(model, image_flip, times=i)
        pred_masks.append(np.flip(pre_mask, 0))
    
    pred = np.stack(pred_masks, axis=0).mean(axis=0)
    return pred


def connect_nearby_contours(pred_mask, extend_pixel):
    def find_if_close(cnt1,cnt2):
        row1,row2 = cnt1.shape[0],cnt2.shape[0]
        for i in range(row1):
            for j in range(row2):
                dist = np.linalg.norm(cnt1[i]-cnt2[j])
                if abs(dist) < 50 :
                    return True
                elif i==row1-1 and j==row2-1:
                    return False

    mask = np.argmax(pred_mask, axis = 2) 
    mask_8bit = np.uint8((abs(mask-4)/4) * 255)

    threshold_level = 1 # Set as you need...
    ret, binarized = cv2.threshold(mask_8bit, threshold_level, 255, cv2.THRESH_BINARY)

    # 컨투어 범위 더 넓게
    kernel = np.ones((extend_pixel, extend_pixel),np.uint8)
    binarized = cv2.dilate(binarized, kernel,iterations = 1)

    contours,hier = cv2.findContours(binarized, cv2.RETR_EXTERNAL, 2)
    if 0 == len(contours): return []
    LENGTH = len(contours)
    status = np.zeros((LENGTH,1))

    for i,cnt1 in enumerate(contours):
        x = i    
        if i != LENGTH-1:
            for j,cnt2 in enumerate(contours[i+1:]):
                x = x+1
                dist = find_if_close(cnt1,cnt2)
                if dist == True:
                    val = min(status[i],status[x])
                    status[x] = status[i] = val
                else:
                    if status[x]==status[i]:
                        status[x] = i+1

    unified = []
    maximum = int(status.max())+1
    for i in range(maximum):
        pos = np.where(status==i)[0]
        if pos.size != 0:
            cont = np.vstack(contours[i] for i in pos)
            hull = cv2.convexHull(cont)
            unified.append(hull)
    return unified



def resizeAndPad(img, size, padColor=0):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img



##################

BACKBONE = 'efficientnetb0'
pretrained_weight = '/content/gdrive/My Drive/findShip/models/trainer_UNet_efficientnetb0_CLAHE.h5'

CLASSES = ['container', 'oil tanker', 'aircraft carrier', 'maritime vessels']


n_classes = len(CLASSES) + 1  # case for binary and multiclass segmentation

activation = 'softmax'

efficientnetb0_model = sm.Unet(BACKBONE, weights=pretrained_weight, classes=n_classes, activation=activation)
efficientnetb0_preprocess_input = sm.get_preprocessing(BACKBONE)


BACKBONE = 'resnet34'
pretrained_weight = '/content/gdrive/My Drive/findShip/models/trainer_UNet_resnet34_CLAHE.h5'

resnet34_model = sm.Unet(BACKBONE, weights=pretrained_weight, classes=n_classes, activation=activation)
resnet34_preprocess_input = sm.get_preprocessing(BACKBONE)


models=[(efficientnetb0_model, efficientnetb0_preprocess_input), (resnet34_model, resnet34_preprocess_input)]

##############

from tqdm import tqdm
import glob
import math

allpreds={}

image_size = 768
base_path = "/content/gdrive/My Drive/findShip/test_images/1192.png"

path = glob.glob(base_path)
files = [file for file in path if file.endswith(".png")]

count = 0

img = cv2.imread("/content/gdrive/My Drive/findShip/test_images/1192.png")
#plt.imshow(img, interpolation='nearest')  
height, width, _ = img.shape

loopx = math.ceil(width / image_size)
loopy = math.ceil(height / image_size)


all_contours = []
"""
for x in range(0, loopx+1):    
    minX = image_size * x
    maxX = image_size * (x+1)
    if not x == 0: 
        minX = minX - 420
        maxX = maxX - 420
    if maxX > width:
        minX = width - image_size
        maxX = width        
    for y in range(0, loopy+1):
        minY = image_size * y
        maxY = image_size * (y+1)
        if not y == 0: 
            minY = minY - 420
            maxY = maxY - 420
        if maxY > height:
            minY = height - image_size
            maxY = height
        print(minY,maxY, minX,maxX)
"""
crop_size = [(0, 768), (348, 1116), (768, 1536), (1116, 1884), (1464, 2232), (1812, 2580), (2160, 2928), (2232,3000)]
for minY,maxY in crop_size:
    for minX,maxX in crop_size:

        #plt.figure()
        #plt.imshow(img[minY:maxY, minX:maxX])

        # CSV 만들기 시작
        #outputs = predictor(img[minY:maxY, minX:maxX])
        target_image = img[minY:maxY, minX:maxX]
        print(target_image.shape)

        plt.figure(figsize = (6,8))
        plt.imshow(target_image)

        # For Test
        for model, preprocess_input in models:
            pre_image = preprocess_input(target_image)


            pr_mask = predict_TTA(model, pre_image)
            print(pr_mask.shape)
            
            mask = np.argmax(pr_mask, axis = 2) 
            mask_8bit = np.uint8((abs(mask-4)/4) * 255)

            threshold_level = 1 # Set as you need...
            ret, binarized = cv2.threshold(mask_8bit, threshold_level, 255, cv2.THRESH_BINARY)

            visualize(
                image=target_image,
                container=pr_mask[..., 0].squeeze(),
                oil_tanker=pr_mask[..., 1].squeeze(),
                aircraft_carrier=pr_mask[..., 2].squeeze(),
                maritime_vessels=pr_mask[..., 3].squeeze(),
                result=binarized,
                    )
            
            contours = connect_nearby_contours(pr_mask, 30)
            for contour in contours:
                res = 1
                for x in contour.shape: res*=x                    
                pxs = contour.reshape(res)[0::2]
                pys = contour.reshape(res)[1::2]
                pxs += minX
                pys += minY
                moved_poly = np.array([[x,y] for x, y in zip(pxs,pys)])
                all_contours.append(moved_poly)
                
            """
            for contour in contours:
                bbox = cv2.boundingRect(contour)
                x, y, width, height = bbox[0], bbox[1], bbox[2], bbox[3]
                #cv2.rectangle(target_image, (x,y), (x+width, y+height), (255,0,0), 2)

                # MASK cnn에 들어갈 이미지
                target_image = resizeAndPad(target_image[y:y+height , x:x+width, :], (768, 768), 0)
                #target_image = cv2.resize(target_image[y:y+height , x:x+width, :], dsize=(768, 768), interpolation=cv2.INTER_CUBIC)
                plt.figure(figsize = (5,5))
                plt.imshow(target_image)
            """

#################


test_image = cv2.imread("/content/gdrive/My Drive/findShip/test_images/1192.png")

cv2.drawContours(test_image, np.array(all_contours) ,-1,(0,255,0),5)

from shapely.geometry import Polygon

p1 = Polygon([])
for cnt in range(1, len(all_contours)):
    p1 = p1.union(Polygon(all_contours[cnt]))

for p in p1:
    bbox = np.array(p.bounds, np.int32)
    x, y, width, height = bbox[0], bbox[1], bbox[2], bbox[3]    
    cv2.rectangle(test_image, (x,y), (width, height), (255,0,0), 5)


plt.figure(figsize = (20,20))
plt.imshow(test_image)
