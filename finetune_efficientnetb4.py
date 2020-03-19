import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import pickle
import segmentation_models as sm
import tensorflow
import albumentations as A

from sklearn.model_selection import StratifiedKFold
from segmentation_models import Unet
from segmentation_models.utils import set_trainable


# segmentation_models could also use `tf.keras` if you do not have Keras installed
# or you could switch to other framework using `sm.set_framework('tf.keras')`


## 그간 학습했던 웨이트 불러오기
load_weights = r'/content/gdrive/My Drive/findShip/models/efficientnetb4_finetune_01_0.6304.h5'

base_dir = r"/content/gdrive/My Drive/findShip/"
# TODO: 전체 테스트셋을 하나의 파일로 만들기
annt_file = os.path.join(base_dir, 'custom_coco_all', 'annotations', "instances_train2017.json")

BACKBONE = 'efficientnetb4'
BATCH_SIZE = 2
CLASSES = ['aircraft carrier', 'container', 'oil tanker', 'maritime vessels']
LR = 0.002
EPOCHS = 20

save_path = r'/content/gdrive/My Drive/findShip/models/%s_finetune_{epoch:02d}_{val_loss:.4f}.h5' % BACKBONE



# helper function for data visualization
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
    
# helper function for data visualization    
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x
    

# classes for data loading and preprocessing
# classes for data loading and preprocessing
class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
        [{'id': 1, 'name': 'aircraft carrier', 'supercategory': 'ship'},
        {'id': 2, 'name': 'container', 'supercategory': 'ship'},
        {'id': 3, 'name': 'oil tanker', 'supercategory': 'ship'},
        {'id': 4, 'name': 'maritime vessels', 'supercategory': 'ship'}]
    """
    
    CLASSES = ['aircraft carrier', 'container', 'oil tanker', 'maritime vessels']
    
    def __init__(
            self, 
            annt_coco_path,  
            images_coco_path,  
            indexes=[],
            augmentation=None, 
            preprocessing=None,
    ):
        self.images_coco_path = images_coco_path
        
        self.coco = COCO(annt_coco_path)
        
        self.indexes = indexes

        # convert str names to class values on masks
        self.classes = {'container': 2, 'oil tanker': 3, 'aircraft carrier': 1, 'maritime vessels': 4}
        self.class_values = list(self.classes.values())#[self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, index):
        # 이미지를 로드 한다
        image_path = self.coco.imgs[index]['file_name']
        image = cv2.imread(os.path.join(self.images_coco_path, image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 마스크를 만든다
        width, height = image.shape[:2]
        img = Image.new('L', (width, height), 0)

        for annot in self.coco.imgToAnns[index]:
            label, polygons = annot['category_id'], annot['segmentation']
            for polygon in polygons:
                polygon = np.array(polygon)
                polygon = [ (x, y) for x, y in zip(polygon[0::2], polygon[1::2])]
                ImageDraw.Draw(img).polygon(polygon, fill=int(label))
                        
        mask = np.array(img)
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        if not 0 == len(self.indexes):
            return len(self.indexes)
        return len(self.coco.imgs)
    
    
class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return batch
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)   
            
         
def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# define heavy augmentations
def get_training_augmentation():
    train_transform = [

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        #A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=359, shift_limit=0.1, p=0.7, border_mode=0),

        #A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        #A.RandomCrop(height=320, width=320, always_apply=True),

        A.IAAAdditiveGaussianNoise(p=0.2),
        A.IAAPerspective(p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.8,
        ),

        A.OneOf(
            [
                A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.8,
        ),

        A.OneOf(
            [
                A.RandomContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.8,
        ),
        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.RandomRotate90()
        #A.PadIfNeeded(384, 480)
    ]
    return A.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)
    
    
    
import os
from pycocotools.coco import COCO

coco = COCO(annt_file)
len(coco.anns)

preprocess_input = sm.get_preprocessing(BACKBONE)


model = Unet(backbone_name=BACKBONE, weights=load_weights,
             input_shape=(None, None, 3), classes=len(CLASSES)+1, activation='softmax')

optim = keras.optimizers.Adam(LR)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
dice_loss = sm.losses.DiceLoss(class_weights=np.array([1, 1, 1, 1, 0.5])) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]


model.compile(optim, total_loss, metrics)


callbacks = [
    keras.callbacks.ModelCheckpoint(save_path, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1),
    keras.callbacks.ReduceLROnPlateau(),
]


train_ant = os.path.join(base_dir, 'custom_coco_all', 'annotations', 'instances_train2017.json')
train_images = os.path.join(base_dir, 'custom_coco_all', 'train2017')
test_ant = os.path.join(base_dir, 'custom_coco_all', 'annotations', 'instances_val2017.json')
test_images = os.path.join(base_dir, 'custom_coco_all', 'val2017')

train_dataset = Dataset(train_ant, train_images,
                        augmentation=get_training_augmentation(),
                        preprocessing=get_preprocessing(preprocess_input))
valid_dataset = Dataset(test_ant, test_images,
                        augmentation=get_validation_augmentation(),
                        preprocessing=get_preprocessing(preprocess_input))

train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

# train model
history = model.fit_generator(
    train_dataloader, 
    steps_per_epoch=len(train_dataloader), 
    epochs=EPOCHS, 
    callbacks=callbacks, 
    workers=10,
    validation_data=valid_dataloader, 
    validation_steps=len(valid_dataloader),
)

