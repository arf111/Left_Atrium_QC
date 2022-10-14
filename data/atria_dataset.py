import monai.data
import torch.utils.data as data
from monai.data import Dataset, DataLoader, NrrdReader


class AtriaDataset(Dataset):
    """Read the nrrd files from dataset folder and return the data and label"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

    def __getitem__(self, index):
        data = self.data[index]
        data = NrrdReader().read(data)
        if self.transform:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.data)


