import torch
from monai.data import DataLoader
from torch import optim

from data.atria_dataset import AtriaDataset
import segmentation_models_pytorch as smp

from model.QCModel import QCModel
from train import train_model

NUM_CLASSES = 5

if __name__ == "__main__":
    root_dir = 'dataset'
    train_dataset = AtriaDataset(root_dir, split_name="training_set", if_clahe=True, input_h=512, input_w=480, orientation=0, transform=False)
    train_loader = DataLoader(dataset=train_dataset, num_workers=4, batch_size=4, shuffle=True)

    test_dataset = AtriaDataset(root_dir, split_name="testing_set", if_clahe=True, input_h=512, input_w=480, orientation=0, transform=False)
    test_loader = DataLoader(dataset=test_dataset, num_workers=4, batch_size=4, shuffle=True)

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

    dataloaders = {"train": train_loader, "test": test_loader}
    dataset_sizes = {"train": len(train_dataset), "test": len(test_dataset)}

    best_model = train_model(model=qc_model, dataloaders=dataloaders, dataset_sizes=dataset_sizes, optimizer=optimizer,
                scheduler=scheduler, device=device, num_epochs=1, num_classes=NUM_CLASSES)
