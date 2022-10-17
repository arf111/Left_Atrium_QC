import datetime
import os
import random

import numpy as np
import torch
from monai.data import Dataset, DataLoader, NrrdReader
from monai.networks.nets import get_efficientnet_image_size
from monai.transforms import Affine
from skimage.exposure import equalize_adapthist
from torchvision.transforms import Compose, RandomVerticalFlip, RandomHorizontalFlip, Resize
from tqdm import tqdm
import matplotlib.pyplot as plt

from PIL import Image
from data.custom_transformer import CropPad


class AtriaDataset(Dataset):
    """Read the nrrd files from dataset folder and return the data and label"""

    def __init__(self, root_dir, input_h=224, input_w=224, orientation=0, sequence_length=1, if_clahe=False,
                 transform=False):
        dataset_dir = os.path.join(root_dir, 'training_set')
        self.patient_list = os.listdir(dataset_dir)
        self.patient_path_list = sorted([os.path.join(dataset_dir, pid) for pid in self.patient_list])
        self.data_size = len(self.patient_path_list)

        self.transform = transform
        self.input_h = input_h
        self.input_w = input_w

        self.if_clahe = if_clahe
        self.orientation = orientation
        self.sequence_length = sequence_length

    def __getitem__(self, index):
        # read the nrrd file
        nrrd_reader = NrrdReader()

        img_obj = nrrd_reader.read(os.path.join(self.patient_path_list[index], 'lgemri.nrrd'))
        img, _ = nrrd_reader.get_data(img_obj)

        slice_id = 0

        # slicing the image at given orientation
        if self.orientation == 0:
            img = img.transpose((2, 1, 0))
            slice_id = np.random.randint(0, img.shape[0])

        img = img[[slice_id], :, :]
        # temp_input = np.zeros(img.shape, dtype=np.float)

        # for i in range(img.shape[0]):
        #     if self.if_clahe:
        #         new_input = equalize_adapthist(img[i])
        #         temp_input[i] = new_input
        #
        # if self.if_clahe:
        #     img = temp_input

        new_input = self.pair_transform(img)

        # normalize data
        new_input_mean = np.mean(new_input, axis=(1, 2), keepdims=True)
        new_input -= new_input_mean
        new_std = np.std(new_input, axis=(1, 2), keepdims=True)
        new_input /= new_std + 0.00000000001

        input = torch.from_numpy(new_input).float()
        target = torch.randint(0, 5, (1,))

        return {'input': input, 'target': target}

    def __len__(self):
        return self.data_size

    def pair_transform(self, image, input_h=256, input_w=256):
        # print('fd:',image.shape)
        result_image = np.zeros((image.shape[0], input_h, input_w), dtype=np.float32)

        # data augmentation
        # data_aug = Compose([RandomVerticalFlip(), RandomHorizontalFlip(), Affine()])
        # if self.transform:
        #     image = data_aug(image)

        CPad = CropPad(input_h, input_w)

        for i in range(result_image.shape[0]):
            result_image[i] = CPad(image[i])

        return result_image


if __name__ == '__main__':
    root_dir = '../dataset'
    torch.cuda.set_device(0)

    for rnd in range(100):
        train_dataset = AtriaDataset(root_dir, if_clahe=True, input_h=512, input_w=480, orientation=0, transform=False)
        train_loader = DataLoader(dataset=train_dataset, num_workers=4, batch_size=4, shuffle=True)
        n = 0
        for epoch_iter, data in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
            print(epoch_iter)
            print(data['input'].shape)  # 10*224*224*1
            print(data['target'])

            if n == 0:
                plt.imshow(data['input'][0, 0, :, :], cmap='gray')
                plt.show()
                plt.imshow(data['input'][1, 0, :, :], cmap='gray')
                plt.show()  #
            n += 1

    image_size = get_efficientnet_image_size("efficientnet-b0")