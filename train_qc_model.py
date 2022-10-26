import torch
from monai.data import DataLoader
from torch import optim
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, ToTensor, Normalize, Resize, \
    PILToTensor

from data.atria_dataset import AtriaDataset
import segmentation_models_pytorch as smp

from model.QCModel import QCModel
from train import train_model

NUM_CLASSES = 5
RESIZE_IMG = 256
NUM_EPOCHS = 25


def get_dataloader(root_dir):
    train_transform = Compose([Resize(RESIZE_IMG), RandomHorizontalFlip(), ToTensor()])
    val_transform = Compose([Resize(RESIZE_IMG), ToTensor()])

    train_dataset = AtriaDataset(root_dir, split_name="training_set2", transform=train_transform)
    val_dataset = AtriaDataset(root_dir, split_name="testing_set2", transform=val_transform)

    train_loader = DataLoader(dataset=train_dataset, num_workers=8, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, num_workers=8, batch_size=32, shuffle=True)

    return train_loader, val_loader


if __name__ == "__main__":
    root_dir = 'dataset'

    train_loader, val_loader = get_dataloader(root_dir)

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = smp.Unet(
        encoder_name="resnet50",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=3,  # model output channels (number of classes in your dataset)
    )
    # model
    qc_model = QCModel(model.encoder, NUM_CLASSES).to(device)

    # optimizer
    optimizer = optim.SGD(qc_model.parameters(), lr=0.001, momentum=0.99, weight_decay=0.0005)

    # scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    dataloaders = {"train": train_loader, "test": val_loader}
    dataset_sizes = {"train": len(train_loader.dataset), "test": len(val_loader.dataset)}

    best_model = train_model(model=qc_model, dataloaders=dataloaders, optimizer=optimizer,
                             scheduler=scheduler, device=device, num_epochs=NUM_EPOCHS, num_classes=NUM_CLASSES)
