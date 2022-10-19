import copy
import time
from typing import Dict

import torch
from coral_pytorch.dataset import corn_label_from_logits
from coral_pytorch.losses import corn_loss
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader


def train_model(model: nn.Module, optimizer, scheduler, dataloaders: Dict[str, DataLoader],
                device, dataset_sizes: Dict[str, int], num_epochs=25, num_classes=5):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_mse = 100000.0

    all_epochs_mse = {"train": [], "test": []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            mae, mse = 0., 0.

            # Iterate over data.
            for data in dataloaders[phase]:
                inputs, labels = data['input'], data['target']
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.flatten()
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = corn_label_from_logits(outputs).float()

                    loss = corn_loss(outputs, labels, num_classes)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                mae += torch.sum(torch.abs(preds - labels))
                mse += torch.sum((preds - labels) ** 2)
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_mae = mae.item() / dataset_sizes[phase]
            epoch_mse = mse.item() / dataset_sizes[phase]
            all_epochs_mse[phase].append(epoch_mse)

            print(f'{phase} Loss: {epoch_loss:.4f}, MAE: {epoch_mae},  MSE: {epoch_mse:.4f}')

            # deep copy the model
            if phase == 'test' and epoch_mse < best_mse:
                best_mse = epoch_mse
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val MSE: {best_mse:4f}')

    # plot loss
    plt.plot(all_epochs_mse["train"], label="train")
    plt.plot(all_epochs_mse["test"], label="test")
    plt.legend()
    plt.show()

    # save model weights
    torch.save(best_model_wts, "model/saved_models/best_model.pth")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
