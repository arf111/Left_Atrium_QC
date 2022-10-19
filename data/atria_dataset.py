import os

import pandas as pd
import torch
from monai.data import Dataset, DataLoader
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor, Normalize, \
    RandomResizedCrop
from tqdm import tqdm
import matplotlib.pyplot as plt

from PIL import Image


class AtriaDataset(Dataset):
    """Read the files from dataset folder and return the data and label"""

    def __init__(self, root_dir, split_name, transform=None):
        dataset_dir = os.path.join(root_dir, split_name)
        dataframe = pd.read_csv(os.path.join(root_dir, f'{split_name}_labels.csv'))

        self.patient_list = os.listdir(dataset_dir)
        self.patient_mri_images_path_list = sorted(
            [(os.path.join(dataset_dir, pid, f"{i}.png"), int(dataframe[dataframe['filenames'] == pid][f"{i}"])) for i in range(88)
             for pid in dataframe['filenames']])

        self.data_size = len(self.patient_mri_images_path_list)

        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.patient_mri_images_path_list[index][0])
        target = torch.tensor(self.patient_mri_images_path_list[index][1], dtype=torch.int8)

        if self.transform:
            img = self.transform(img)

        return {'input': img, 'target': target}

    def __len__(self):
        return self.data_size


if __name__ == '__main__':
    root_dir = '../dataset'
    torch.cuda.set_device(0)

    train_transform = Compose([RandomResizedCrop(256), RandomHorizontalFlip(),
                               ToTensor(), Normalize((0.5,), (0.5,))])

    for rnd in range(100):
        train_dataset = AtriaDataset(root_dir, split_name="training_set", transform=train_transform)
        train_loader = DataLoader(dataset=train_dataset, num_workers=4, batch_size=4, shuffle=True)
        n = 0
        for epoch_iter, data in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
            # print(epoch_iter)
            print(data['input'].shape)  # 10*224*224*1
            print(data['target'].shape)

            if n == 0:
                plt.imshow(data['input'][0, 0, :, :], cmap='gray')
                plt.show()
                plt.imshow(data['input'][1, 0, :, :], cmap='gray')
                plt.show()  #
            n += 1
