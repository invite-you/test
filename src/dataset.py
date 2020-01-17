import os
import json
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
import cv2

class ShipDataset(Dataset):
    def __init__(self, root_dir, label_name):
        self.root_dir = root_dir

        #label_filepath = os.path.join(root_dir, 'labels.json')
        label_filepath = os.path.join(root_dir, label_name)
        with open(label_filepath) as f:
            data = json.load(f)

        self.labels = list()
        self.size = len(data['features'])

        for idx in range(len(data['features'])):
            properties = data['features'][idx]['properties']
            
            anot = dict()            
            anot['answer'] = properties['type_id']
            anot['image_filename'] = properties['image_id']
            anot['bbox'] = np.array([float(num) for num in properties['bounds_imcoords'].split(",")])
            
            self.labels.append(anot)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        try:
            img = self.load_image(idx)
            annot = self.load_annotations(idx)
            sample = {'img': img, 'annot': annot}
        except:
            return {'img': {}, 'annot': {}}
        return sample

    def load_image(self, idx):
        image_filename = self.labels[idx]['image_filename']
        file_path = os.path.join(self.root_dir, 'images', image_filename)
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        # if len(img.shape) == 2:
        #     img = skimage.color.gray2rgb(img)

        return img.astype(np.float32) / 255.

    def load_annotations(self, idx):

        annotations = np.zeros((0, 5))
        anot = self.labels[idx]

        annotation = np.zeros((1, 5))
        annotation[0,:4] = anot['bbox'][[0, 1, 4, 5]]
        annotation[0, 4] = np.array(anot['answer'])
        
        annotations = np.append(annotations, annotation, axis=0)

        return annotations

    def num_classes(self):
        return 4


def collater(data):
    if len(data) == 0: return {}
    
    imgs = [s['img'] for s in data if len(s['img']) != 0]
    annots = [s['annot'] for s in data if len(s['annot']) != 0]
    #scales = [s['scale'] for s in data]
    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)
    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1
        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = torch.tensor(annot.astype(np.float64))

    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded}# {'img': imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, common_size=512):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = common_size / height
            resized_height = common_size
            resized_width = int(width * scale)
        else:
            scale = common_size / width
            resized_height = int(height * scale)
            resized_width = common_size

        image = cv2.resize(image, (resized_width, resized_height))

        new_image = np.zeros((common_size, common_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}
