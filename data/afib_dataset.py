import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from monai.data import DataLoader, Dataset
from PIL import Image
from torchvision.transforms import (Compose, RandomHorizontalFlip,
                                    RandomResizedCrop, ToTensor)
from tqdm import tqdm

from util.utils import get_dataloader


class AfibDataset(Dataset):
    """Read the files from dataset folder and return the data and label"""

    def __init__(self, image_paths, labels, transform=None):
        self.patient_mri_images_path_list = image_paths
        self.labels = labels

        self.data_size = len(self.patient_mri_images_path_list)

        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(str(self.patient_mri_images_path_list[index]))

        target = {"quality_for_fibrosis_assessment": torch.tensor(int(self.labels[index]["quality_for_fibrosis_assessment"]) - 1),
                  "enhancement_of_fibrosis_tissue": torch.tensor(int(self.labels[index]["enhancement_of_fibrosis_tissue"]) - 1),
                  "sharpness": torch.tensor(int(self.labels[index]["sharpness"]) - 1),
                  "myocardium_nulling": torch.tensor(int(self.labels[index]["myocardium_nulling"]) - 1),
                  "overall": torch.tensor(int(self.labels[index]["overall"]) - 1)}

        # Image.values are in range [0, 255], shape is (640, 640) or (512, 512)

        if self.transform:
            img = self.transform(img)

        # convert to float
        img = img.float()

        return {'input': img, 'target': target, "patient_id": self.labels[index]["patient_id"]}  # Number of unique class labels (class labels should start at 0).

    def __len__(self):
        return self.data_size


if __name__ == '__main__':
    root_dir = Path('../dataset/afib_data')
    # device
    device = "cpu"

    train_transform = Compose([RandomResizedCrop(256), RandomHorizontalFlip(), ToTensor()])

    for rnd in range(100):
        train_loader, val_loader = get_dataloader(root_dir, AfibDataset)
        n = 0
        for epoch_iter, data in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
            # print(epoch_iter)
            # move to device
            inputs = data['input'].to(device)
            overall_labels = data['target']['overall'].to(device)
            quality_for_fibrosis_assessment_labels = data['target']['quality_for_fibrosis_assessment'].to(device)
            enhancement_of_fibrosis_tissue_labels = data['target']['enhancement_of_fibrosis_tissue'].to(device)
            sharpness_labels = data['target']['sharpness'].to(device)
            myocardium_nulling_labels = data['target']['myocardium_nulling'].to(device)
            
            print(inputs.shape)  # 10*224*224*1
            print(overall_labels.shape)  # 10*1

            if n == 0:
                plt.imshow(inputs[0, 0, :, :], cmap='gray')
                plt.show()

            n += 1
