import copy
import time
from typing import Dict

import torch
from coral_pytorch.dataset import corn_label_from_logits
from coral_pytorch.losses import corn_loss, coral_loss
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

from util.utils import get_true_vs_predicted_values, plot_confusion_matrix


def compute_loss_and_metrics(model: nn.Module, dataloader: DataLoader, device, num_classes=5):
    model.eval()
    with torch.no_grad():
        mae, mse, running_loss = 0.0, 0.0, 0.0
        for i, data in enumerate(dataloader):
            features, targets = data['input'], data['target']
            features = features.to(device)
            targets = targets.to(device)
            targets = targets.flatten()
            logits = model(features)
            loss = corn_loss(logits, targets, num_classes)
            # statistics
            mse += torch.sum((corn_label_from_logits(logits).float() - targets) ** 2)
            mae += torch.sum(torch.abs(corn_label_from_logits(logits).float() - targets))
            running_loss += loss.item() * features.size(0)
        return running_loss, mae, mse


def train_model(model: nn.Module, model_name: str, optimizer, scheduler, dataloaders: Dict[str, DataLoader],
                device, num_epochs=25, num_classes=5):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_mse = 100000.0

    all_epochs_mse = {"train": [], "val": []}
    all_epochs_loss = {"train": [], "val": []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

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
                    # preds = corn_label_from_logits(outputs).float()

                    loss = corn_loss(outputs, labels, num_classes)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            running_loss, mae, mse = compute_loss_and_metrics(model, dataloaders[phase], device, num_classes)

            epoch_loss = running_loss / len(dataloaders[phase])
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_mae = mae.item() / len(dataloaders[phase])
            epoch_mse = mse.item() / len(dataloaders[phase])
            all_epochs_mse[phase].append(epoch_mse)
            all_epochs_loss[phase].append(epoch_loss)

            print(f'{phase} Loss: {epoch_loss:.4f}, MAE: {epoch_mae},  MSE: {epoch_mse:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_mse < best_mse:
                best_mse = epoch_mse
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val MSE: {best_mse:4f}')

    # plot metrics
    # plt.plot(all_epochs_mse["train"], label="train")
    # plt.plot(all_epochs_mse["val"], label="val")
    # plt.legend()
    # plt.title("Mean Squared Error")
    # plt.xlabel("Epochs")
    # plt.ylabel("MSE")
    # plt.savefig(f"{model_name}_mse_metric.png")
    # plt.show()

    # plot loss
    # plt.plot(all_epochs_loss["train"], label="train")
    # plt.plot(all_epochs_loss["val"], label="val")
    # plt.legend()
    # plt.title("Loss")
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.savefig(f"{model_name}_CORN_loss.png")
    # plt.show()

    # save model weights
    torch.save(best_model_wts, f"model/saved_models/{model_name}_best.pth")

    # load best model weights
    model.load_state_dict(best_model_wts)

    # transfer to cpu
    model.to("cpu")

    # Plot the confusion matrix of training set
    train_true_labels, train_pred_labels = get_true_vs_predicted_values(dataloaders['train'], model)
    # plot_confusion_matrix(train_true_labels, train_pred_labels, f"{model_name}_train_confusion_matrix")

    # Plot the confusion matrix of testing set
    val_true_labels, val_pred_labels = get_true_vs_predicted_values(dataloaders['val'], model)
    # plot_confusion_matrix(val_true_labels, val_pred_labels, f"{model_name}_test_confusion_matrix")

    return best_mse, all_epochs_mse, all_epochs_loss, train_true_labels, train_pred_labels, val_true_labels, val_pred_labels
