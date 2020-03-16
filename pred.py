
from tqdm import tqdm
import glob
import math
import csv

allpreds={}

image_size = 768
base_path = "/content/gdrive/My Drive/findShip/test_images/"
crop_size = [(0, 768), (744, 1512), (1488, 2256), (2232, 3000)]

path = glob.glob(base_path+'*.png')
#files = [file for file in path if file.endswith(".png")]

count = 0

csv_path = "/content/gdrive/My Drive/findShip/result"
with open(csv_path, 'w') as f:
    csv_writer = csv.DictWriter(f, ['file_name', 'class_id', 'confidence', 'point1_x', 'point1_y', 'point2_x', 'point2_y',
                            'point3_x', 'point3_y', 'point4_x', 'point4_y'])
    csv_writer.writeheader()

    for image_path in path:
        print(image_path)
        count +=1
        if count > 10: break
        #image = cv2.imread(image_path)
        image = cv2.imread("/content/gdrive/My Drive/findShip/test_images/1192.png")
        height, width, _ = image.shape

        loopx = math.ceil(width / image_size)
        loopy = math.ceil(height / image_size)

        all_contours = []
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
                    
        # 이미지내 추출한 컨투어만 필요
        # TODO: 너무 작은 area는 제외할 필요가 있음
        p1 = Polygon([])
        for cnt in range(1, len(all_contours)):
            p2 = Polygon(all_contours[cnt])
            
            if p2.area < 1000.0:
                continue
            p1 = p1.union(p2)

        draw_image = image.copy() ####### 이미지 출력
        for p in p1:
            # 배가 있을 것 같은 부분을 자른다
            pbbox = np.array(p.bounds, np.int32)
            bminX, bminY, bmaxX, bmaxY = pbbox[0], pbbox[1], pbbox[2], pbbox[3]    
            pred_image, pad = resizeAndPad(image[bminX:bmaxX, bminY:bmaxY, :], (768, 768))
            # 인스턴스 세그먼테이션 처리            
            poutputs = predictor(pred_image)
            # 결과 정리
            predictions = poutputs["instances"].to("cpu")

            pscores = predictions.scores if predictions.has("scores") else None
            pclasses = predictions.pred_classes if predictions.has("pred_classes") else None
            pmasks = np.asarray(predictions.pred_masks)

            for pscore, pcls, pmask in  zip(pscores, pclasses, pmasks):
                pmask = pmask.astype("uint8") 
                # Maskrcnn에 넣기위해 동일 사이즈로 만든 것을 원복 한다.
                pad_top, pad_bot, pad_left, pad_right = pad
                ww, hh = pmask.shape[:2]
                pmask = pmask[pad_top:(hh-pad_bot), pad_left:(ww-pad_right)]
                pmask = cv2.resize(pmask, dsize=(bmaxY-bminY, bmaxX-bminX), interpolation=cv2.INTER_CUBIC)
                
                """
                visualize(
                    pred= pred_image,
                    result=mask,
                            )
                """     
                # TODO: 컨투어를 스무스 하게 만들기
                pcontours, _  = cv2.findContours(
                            pmask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                # 작은 컨투어는 삭제
                pcontour_sizes = [(cv2.contourArea(pcontour), pcontour) for pcontour in pcontours]
                pcontour = max(pcontour_sizes, key=lambda x: x[0])[1]
                
                prect = cv2.minAreaRect(pcontour)
                pbox = cv2.boxPoints(prect) 
                # 전체 이미지 사이즈에 맞게 box 위치를 잡는다
                pbox = move_origin_position([pbox], minY, minX)[0]
                
                det_dict = {'file_name': os.path.basename(image_path),
                                'class_id': pcls,
                                'confidence': pscore,
                                'point1_x': pbox[0][0],
                                'point1_y': pbox[0][1],
                                'point2_x': pbox[1][0],
                                'point2_y': pbox[1][1],
                                'point3_x': pbox[2][0],
                                'point3_y': pbox[2][1],
                                'point4_x': pbox[3][0],
                                'point4_y': pbox[3][1],
                                }
                csv_writer.writerow(det_dict)

                pbox = np.int0(pbox)######## 이미지 출력 실제 쓸때는 제외
                cv2.drawContours(draw_image, [pbox],-1,(0,255,0),6)

        plt.figure(figsize = (20,20))####### 이미지 출력
        plt.imshow(draw_image)
        break


   
########################


from tqdm import tqdm
import glob
import math

allpreds={}

image_size = 768
base_path = "/content/gdrive/My Drive/findShip/test_images/"

path = glob.glob(base_path)
files = [file for file in path if file.endswith(".png")]

count = 0
image_path = "/content/gdrive/My Drive/findShip/test_images/1192.png"
image = cv2.imread(image_path)
height, width, _ = image.shape

loopx = math.ceil(width / image_size)
loopy = math.ceil(height / image_size)



crop_size = [(0, 768), (744, 1512), (1488, 2256), (2232, 3000)]
all_contours = []
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
            print(pr_mask.shape)
            
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

preds = calc_contours(image_path, all_contours)




################

draw_image = image.copy()

# TODO: 너무 작은 area는 제외할 필요가 있음
p1 = Polygon([])
for cnt in range(1, len(all_contours)):
    p2 = Polygon(all_contours[cnt])
    
    if p2.area < 1000.0:
        continue
    p1 = p1.union(p2)

for p in p1:
    # 배가 있을 것 같은 부분을 자른다
    bbox = np.array(p.bounds, np.int32)
    minX, minY, maxX, maxY = bbox[0], bbox[1], bbox[2], bbox[3]    
    # 인스턴스 세그먼테이션 처리
    print("##")
    print(image[minX:maxX, minY:maxY, :].shape)
    pred_image, pad = resizeAndPad(image[minX:maxX, minY:maxY, :], (768, 768))
    outputs = predictor(pred_image)
    # 결과 정리
    predictions = outputs["instances"].to("cpu")

    scores = predictions.scores if predictions.has("scores") else None
    classes = predictions.pred_classes if predictions.has("pred_classes") else None
    masks = np.asarray(predictions.pred_masks)
    print(len(masks))
    for score, cls, mask in  zip(scores, classes, masks):
        print("find ship")
        mask = mask.astype("uint8") 
        # Maskrcnn에 넣기위해 동일 사이즈로 만든 것을 원복 한다.
        pad_top, pad_bot, pad_left, pad_right = pad
        w, h = mask.shape[:2]
        mask = mask[pad_top:(h-pad_bot), pad_left:(w-pad_right)]
        mask = cv2.resize(mask, dsize=(maxY-minY, maxX-minX), interpolation=cv2.INTER_CUBIC)
        
        
        visualize(
            pred= pred_image,
            result=mask,
                    )
             
        # TODO: 컨투어를 스무스 하게 만들기
        contours, _  = cv2.findContours(
                    mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        # 작은 컨투어는 삭제
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        contour = max(contour_sizes, key=lambda x: x[0])[1]
        
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect) 
        # 전체 이미지 사이즈에 맞게 box 위치를 잡는다
        box = move_origin_position([box], minY, minX)[0]
        
        box = np.int0(box)# 실제 쓸때는 제외
        cv2.drawContours(draw_image, [box],-1,(0,255,0),6)

plt.figure(figsize = (20,20))
plt.imshow(draw_image)



#############

def calc_contours(image_path, all_contours):
    preds=[]
    image = cv2.imread(image_path)
    # TODO: 너무 작은 area는 제외할 필요가 있음
    p1 = Polygon([])
    for cnt in range(1, len(all_contours)):
        p2 = Polygon(all_contours[cnt])
        
        if p2.area < 1000.0:
            continue
        p1 = p1.union(p2)

    for p in p1:
        # 배가 있을 것 같은 부분을 자른다
        bbox = np.array(p.bounds, np.int32)
        minX, minY, maxX, maxY = bbox[0], bbox[1], bbox[2], bbox[3]    
        # 인스턴스 세그먼테이션 처리
        print("##")
        print(image[minX:maxX, minY:maxY, :].shape)
        pred_image, pad = resizeAndPad(image[minX:maxX, minY:maxY, :], (768, 768))
        outputs = predictor(pred_image)
        # 결과 정리
        predictions = outputs["instances"].to("cpu")

        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None
        masks = np.asarray(predictions.pred_masks)
        print(len(masks))
        for score, cls, mask in  zip(scores, classes, masks):
            print("find ship")
            mask = mask.astype("uint8") 
            # Maskrcnn에 넣기위해 동일 사이즈로 만든 것을 원복 한다.
            pad_top, pad_bot, pad_left, pad_right = pad
            w, h = mask.shape[:2]
            mask = mask[pad_top:(h-pad_bot), pad_left:(w-pad_right)]
            mask = cv2.resize(mask, dsize=(maxY-minY, maxX-minX), interpolation=cv2.INTER_CUBIC)
            
            
            visualize(
                pred= pred_image,
                result=mask,
                        )
                
            # TODO: 컨투어를 스무스 하게 만들기
            contours, _  = cv2.findContours(
                        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            # 작은 컨투어는 삭제
            contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
            contour = max(contour_sizes, key=lambda x: x[0])[1]
            
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect) 
            # 전체 이미지 사이즈에 맞게 box 위치를 잡는다
            box = move_origin_position([box], minY, minX)[0]
            
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
            cv2.drawContours(draw_image, [box],-1,(0,255,0),6)

    plt.figure(figsize = (20,20))
    plt.imshow(draw_image)
