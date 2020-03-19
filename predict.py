!pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
!pip install -r requirements.txt
!pip3 install keras==2.1.3.
!python3 setup.py install




!python3 samples/coco/coco.py evaluate --dataset="/content/gdrive/My Drive/findShip/custom_coco_all/" --year=2017 --model="/content/gdrive/My Drive/findShip/custom_coco_all/log/coco20200318T1703/mask_rcnn_coco_0008.h5"



VALIDATION_STEPS = 10





##NMS

def nms_maskrcnn(scores, classes, bboxes):
    scores, classes, bboxes = scores.copy(), classes.copy(), bboxes.copy()
    def IoA(poly1, poly2):
        return poly1.intersection(poly2).area / poly2.area

    #scores, classes, bboxes
    sidx = np.argsort(-np.array(scores))

    thr = 0.3
    delpos=[]
    for i in range(len(sidx)):
        for j in range(i+1, len(sidx)):        
            higher = sidx[i]
            lower = sidx[j]
            # 삭제 될 인덱스라 검토 불필요
            if lower in delpos: continue
            hpoly, lpoly = Polygon(bboxes[higher]), Polygon(bboxes[lower])
            if hpoly.area < 100.0: continue
            if lpoly.area < 100.0: continue
            #print(IoA(hpoly, lpoly))
            # 두 폴리곤이 상관관계가 없음
            if IoA(hpoly, lpoly) < thr: 
                continue
            #print(higher, lower)
            delpos.append(lower)
            
    bboxes = np.delete(bboxes, delpos, axis=0)
    classes = np.delete(classes, delpos)
    scores = np.delete(scores, delpos)
    """    
    # for debugging
    draw_image = pred_image.copy()
    for i, box in enumerate(b):    
        box = np.int0(box)
        cv2.drawContours(draw_image, [box],-1,(0,255,0),6)

    plt.figure(figsize = (10,10))
    plt.imshow(draw_image)
    """
    return scores, classes, bboxes

### TTA

def predict_mask_TTA(predictor, image):
    def predict_rotatationt90_image(predictor, base_image, times=1, flip=False):
        image = base_image.copy()
        if flip: image = np.flip(image, 0)
        image90 = np.rot90(image, times)
        outputs = predictor(image90)
        ###
        """
        v = Visualizer(image90[:, :, ::-1], MetadataCatalog.get('airbus'), scale=1.3)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.figure(figsize = (10,10))
        plt.imshow(v.get_image()[:, :, ::-1])
        """
        ###
        predictions = outputs["instances"].to("cpu")

        scores = predictions.scores if predictions.has("scores") else None
        if len(scores) == 0: return [], [], [] # 아무것도 없으면 빈값 리턴
        classes = predictions.pred_classes if predictions.has("pred_classes") else None
        pmasks = np.asarray(predictions.pred_masks)
        omasks = []
        for mask in pmasks:
            # 돌렸던 마스크를 원상복귀
            mask = np.rot90(mask, -times)
            if flip: mask = np.flip(mask, 0)
            #mask.astype("uint8") 
            omasks.append(mask)
        return scores, classes, omasks

    scores, classes, masks = [], [], []
    for i in range(1,5):
        score, cls, mask = predict_rotatationt90_image(predictor, image, times=i)
        if not len(score) == 0: 
            scores.extend(score.tolist())
            classes.extend(cls.tolist())
            masks.extend(mask)
        score, cls, mask = predict_rotatationt90_image(predictor, image, times=i, flip=True)    
        if not len(score) == 0: 
            scores.extend(score.tolist())
            classes.extend(cls.tolist())
            masks.extend(mask)

    return scores, classes, masks

"""
image_path = "/content/gdrive/My Drive/findShip/test_images/1192.png"
minY=494
maxY=734
minX=680
maxX=1095
image = cv2.imread(image_path)
pred_image, pad = resizeAndPad(image[minY:maxY, minX:maxX, :], (768, 768))
scores, classes, masks = predict_mask_TTA(predictor, pred_image)

draw_image = image.copy()
for mask in  masks:
    #print("find ship")
    mask = mask.astype("uint8") 
    # Maskrcnn에 넣기위해 동일 사이즈로 만든 것을 원복 한다.
    pad_top, pad_bot, pad_left, pad_right = pad
    w, h = mask.shape[:2]
    mask = mask[pad_top:(h-pad_bot), pad_left:(w-pad_right)]
    mask = cv2.resize(mask, dsize=(maxX-minX, maxY-minY), interpolation=cv2.INTER_CUBIC)
        
    # TODO: 컨투어를 스무스 하게 만들기
    contours, _  = cv2.findContours(
                mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    # 작은 컨투어는 삭제
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    contour = max(contour_sizes, key=lambda x: x[0])[1]
    
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect) 
    # 전체 이미지 사이즈에 맞게 box 위치를 잡는다
    box = move_origin_position([box], minX, minY)[0]
    cv2.drawContours(draw_image, [np.int0(box)],-1,(0,255,0),6)

plt.figure(figsize = (16,16))
plt.imshow(draw_image)
"""


## RUNNER

def calc_contours(image_path, all_contours):
    preds=[]
    image = cv2.imread(image_path)
    draw_image = image.copy()
    # TODO: 너무 작은 area는 제외할 필요가 있음
    p1 = Polygon([])
    chker = 0
    for cnt in range(1, len(all_contours)):
        p2 = Polygon(all_contours[cnt])
        
        if p2.area < 100.0:
            continue
        p1 = p1.union(p2)
        chker += 1
    if chker == 0: return []

    if not isinstance(p1, list):
        p1 = [p1]

    for p in p1:
        if 0 == len(p.bounds): continue
        # 배가 있을 것 같은 부분을 자른다
        bbox = np.array(p.bounds, np.int32)
        minX, minY, maxX, maxY = bbox[0], bbox[1], bbox[2], bbox[3]    
        # 인스턴스 세그먼테이션 처리
        #print("##")
        #print(image[minX:maxX, minY:maxY, :].shape)
        pred_image, pad = resizeAndPad(image[minY:maxY, minX:maxX, :], (768, 768))
        
        scores, classes, masks = predict_mask_TTA(predictor, pred_image)
        # 아무 객체 없음
        if len(masks) == 0: continue
        # 회전된 바운딩 박스 구하기
        #print(len(masks))
        rbboxes=[]
        for mask in  masks:
            #print("find ship")
            mask = mask.astype("uint8") 
            # Maskrcnn에 넣기위해 동일 사이즈로 만든 것을 원복 한다.
            pad_top, pad_bot, pad_left, pad_right = pad
            w, h = mask.shape[:2]
            mask = mask[pad_top:(h-pad_bot), pad_left:(w-pad_right)]
            mask = cv2.resize(mask, dsize=(maxX-minX, maxY-minY), interpolation=cv2.INTER_CUBIC)
            
            
            
            # TODO: 컨투어를 스무스 하게 만들기
            contours, _  = cv2.findContours(
                        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            # 작은 컨투어는 삭제
            contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
            contour = max(contour_sizes, key=lambda x: x[0])[1]
            
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect) 
            # 전체 이미지 사이즈에 맞게 box 위치를 잡는다
            box = move_origin_position([box], minX, minY)[0]
            rbboxes.append(box)

        scores, classes, rbboxes = nms_maskrcnn(scores, classes, rbboxes)
        for score, cls, box in  zip(scores, classes, rbboxes):
            preds.append({'file_name': os.path.basename(image_path),
                                'class_id': cls,
                                'confidence': score,
                                'point1_x': box[0][0],
                                'point1_y': box[0][1],
                                'point2_x': box[1][0],
                                'point2_y': box[1][1],
                                'point3_x': box[2][0],
                                'point3_y': box[2][1],
                                'point4_x': box[3][0],
                                'point4_y': box[3][1],
                                })
            box = np.int0(box)# 실제 쓸때는 제외
            cv2.drawContours(draw_image, [box],-1,(0,255,0), 6)

    ###### 전체 이미지에 대고 한번 더
    pred_image, pad = resizeAndPad(image, (768, 768))
    
    scores, classes, masks = predict_mask_TTA(predictor, pred_image)
    # 아무 객체 없음
    if len(masks) == 0: 
        #plt.figure(figsize = (14,14))
        #plt.imshow(draw_image)
        return preds
    # 회전된 바운딩 박스 구하기
    #print(len(masks))
    rbboxes=[]
    for mask in  masks:
        #print("find ship")
        mask = mask.astype("uint8") 
        # Maskrcnn에 넣기위해 동일 사이즈로 만든 것을 원복 한다.
        pad_top, pad_bot, pad_left, pad_right = pad
        w, h = mask.shape[:2]
        mask = mask[pad_top:(h-pad_bot), pad_left:(w-pad_right)]
        mask = cv2.resize(mask, dsize=(maxX-minX, maxY-minY), interpolation=cv2.INTER_CUBIC)
        
        
        
        # TODO: 컨투어를 스무스 하게 만들기
        contours, _  = cv2.findContours(
                    mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        # 작은 컨투어는 삭제
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        contour = max(contour_sizes, key=lambda x: x[0])[1]
        
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect) 
        # 전체 이미지 사이즈에 맞게 box 위치를 잡는다
        box = move_origin_position([box], minX, minY)[0]
        rbboxes.append(box)

    scores, classes, rbboxes = nms_maskrcnn(scores, classes, rbboxes)
    for score, cls, box in  zip(scores, classes, rbboxes):
        preds.append({'file_name': os.path.basename(image_path),
                            'class_id': cls,
                            'confidence': score,
                            'point1_x': box[0][0],
                            'point1_y': box[0][1],
                            'point2_x': box[1][0],
                            'point2_y': box[1][1],
                            'point3_x': box[2][0],
                            'point3_y': box[2][1],
                            'point4_x': box[3][0],
                            'point4_y': box[3][1],
                            })
        box = np.int0(box)# 실제 쓸때는 제외
        cv2.drawContours(draw_image, [box],-1,(0,255,0), 6)

    #plt.figure(figsize = (14,14))
    #plt.imshow(draw_image)
    return preds



from tqdm import tqdm
import glob
import math

allpreds=[]
count = 0

image_size = 768
base_path = "/content/gdrive/My Drive/findShip/test_images/"

path = glob.glob(base_path + "*.png")
path = sorted(path, key=lambda a: int(os.path.basename(a).split('.')[0]) )

for image_path in path:
    #if count > 10: break
    count+=1
    #image_path = "/content/gdrive/My Drive/findShip/test_images/1192.png"
    print(image_path)
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    loopx = math.ceil(width / image_size)
    loopy = math.ceil(height / image_size)

    crop_size = [(0, 768), (744, 1512), (1488, 2256), (2232, 3000)]
    all_contours = []
    %%timeit
    for minY,maxY in crop_size:
        for minX,maxX in crop_size:

            #plt.figure()
            #plt.imshow(image[minY:maxY, minX:maxX])

            # CSV 만들기 시작
            #outputs = predictor(image[minY:maxY, minX:maxX])
            target_image = image[minY:maxY, minX:maxX]
            #print(target_image.shape)

            #plt.figure(figsize = (6,8))
            #plt.imshow(target_image)

            # For Test
            for model, preprocess_input in models:
                pre_image = preprocess_input(target_image)

                pr_mask = predict_TTA(model, pre_image)
                #print(pr_mask.shape)
                
                mask = np.argmax(pr_mask, axis = 2) 
                mask_8bit = np.uint8((abs(mask-4)/4) * 255)

                threshold_level = 1 # Set as you need...
                ret, binarized = cv2.threshold(mask_8bit, threshold_level, 255, cv2.THRESH_BINARY)

                """
                visualize(
                    container=pr_mask[..., 0].squeeze(),
                    oil_tanker=pr_mask[..., 1].squeeze(),
                    aircraft_carrier=pr_mask[..., 2].squeeze(),
                    maritime_vessels=pr_mask[..., 3].squeeze(),
                    result=binarized,
                        )
                """

                contours = connect_nearby_contours(pr_mask, 30)
                all_contours.extend(
                    move_origin_position(contours, minX, minY))

    allpreds.extend( calc_contours(image_path, all_contours) )
    print(len(allpreds))
