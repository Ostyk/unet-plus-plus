import os

import cv2
import numpy as np
import torch
import torch.utils.data
import json


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, subset, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
#         self.img_ids = img_ids
#         self.img_dir = img_dir
#         self.mask_dir = mask_dir
#         self.img_ext = img_ext
#         self.mask_ext = mask_ext
        self.subset = subset
        self.num_classes = num_classes
        self.transform = transform
        
        mask_path = os.path.join(root, 'polygons.json')
        with open(mask_path) as f:
            self.polygons = json.load(f)

        self.root = os.path.join(root, self.subset)
        self.imgs = sorted([i for i in os.listdir(self.root) if i.endswith('.png')])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        
        
        img_path = os.path.join(self.root, self.imgs[idx])
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_RGB2BGR) 

        polygons = self.polygons[self.imgs[idx]]['polygons']
        img_id = self.imgs[idx]
        mask = self.load_mask(img, polygons)
        mask = np.expand_dims(mask, -1)
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)
        
        return img, mask, {'img_id': img_id}
    
    
    
    @staticmethod
    def load_rgb(path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_RGB2BGR)
    
    @staticmethod
    def load_mask(img, polygons):
        """
        Transforms polygons of a single image into a 2D binary numpy array
        
        :param img: just to get the corresponding shape of the output array
        :param polygons: - dict
        
        :return mask: numpy array with drawn over and touching polygons
        """
        mask = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
        for curr_pol in polygons:
            cv2.fillPoly(mask, [np.array(curr_pol, 'int32')], 255)
        return mask
