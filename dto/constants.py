import torch
from torchvision.transforms import (Resize, Compose, RandomHorizontalFlip,
                                    RandomResizedCrop, ToTensor)

DATASET_SCORE_FILE = "Image_Quality_Assessment_Whole_Set_Kholmovski.csv"
DISPLAY_IMG_SIZE = 512
RESIZE_IMG = 256
BATCH_SIZE = 32
NUM_CLASSES = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_TRANSFORM = Compose([Resize(RESIZE_IMG), RandomHorizontalFlip(), ToTensor()])
VAL_TRANSFORM = Compose([Resize(RESIZE_IMG), ToTensor()])