import copy
from statistics import mode
import time
from typing import Dict

import torch
from coral_pytorch.dataset import corn_label_from_logits
from coral_pytorch.losses import corn_loss
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

from collections import defaultdict
from util.utils import get_true_vs_predicted_values, plot_confusion_matrix


def compute_loss_and_metrics(model: nn.Module, dataloader: DataLoader, device, num_classes=5):
    model.eval()
    with torch.no_grad():
        # {"patient_id": {"sharpness_attribute": [], "myocardium_nulling_attribute": [], "fibrosis_tissue_enhancement_attribute": [], "overall": [], true_labels: int}}
        patient_dict = defaultdict(lambda: defaultdict(list))
        mae_dict, mse_dict = {}, {}
        running_loss = 0.0

        for i, data in enumerate(dataloader):
            inputs, targets = data['input'], data['target']
            inputs = inputs.to(device)
            
            overall_labels = targets['overall'].to(device)
            sharpness_labels = targets['sharpness'].to(device)
            myocardium_nulling_labels = targets['myocardium_nulling'].to(device)
            fibrosis_tissue_enhancement_labels = targets['enhancement_of_fibrosis_tissue'].to(device)    
           
            logits_of_sharpness_attribute, logits_of_myocardium_nulling_attribute, logits_of_fibrosis_tissue_enhancement_attribute, \
                logits_of_overall = model(inputs)
                    
            loss_of_sharpness_attribute = corn_loss(logits_of_sharpness_attribute, sharpness_labels, num_classes)
            loss_of_myocardium_nulling_attribute = corn_loss(logits_of_myocardium_nulling_attribute, myocardium_nulling_labels, num_classes)
            loss_of_fibrosis_tissue_enhancement_attribute = corn_loss(logits_of_fibrosis_tissue_enhancement_attribute, fibrosis_tissue_enhancement_labels, num_classes)
            loss_of_overall_quality = corn_loss(logits_of_overall, overall_labels, num_classes)

            attribute_loss = loss_of_sharpness_attribute + loss_of_myocardium_nulling_attribute + loss_of_fibrosis_tissue_enhancement_attribute
            loss = attribute_loss + loss_of_overall_quality
            # statistics
            predicted_logits_of_sharpness_attribute = corn_label_from_logits(logits_of_sharpness_attribute).float().tolist()
            predicted_logits_of_myocardium_nulling_attribute = corn_label_from_logits(logits_of_myocardium_nulling_attribute).float().tolist()
            predicted_logits_of_fibrosis_tissue_enhancement_attribute = corn_label_from_logits(logits_of_fibrosis_tissue_enhancement_attribute).float().tolist()
            predicted_logits_of_overall = corn_label_from_logits(logits_of_overall).float().tolist()
            # append all the attributes in a list based on patient_id
            for index, patient_id in enumerate(data["patient_id"]):
                patient_dict[patient_id]['pred_sharpness_attribute'].append(predicted_logits_of_sharpness_attribute[index])
                patient_dict[patient_id]['pred_myocardium_nulling_attribute'].append(predicted_logits_of_myocardium_nulling_attribute[index])
                patient_dict[patient_id]['pred_fibrosis_tissue_enhancement_attribute'].append(predicted_logits_of_fibrosis_tissue_enhancement_attribute[index])
                patient_dict[patient_id]['pred_overall'].append(predicted_logits_of_overall[index])

                patient_dict[patient_id]['true_sharpness_attribute'].append(sharpness_labels[index].float().item())
                patient_dict[patient_id]['true_myocardium_nulling_attribute'].append(myocardium_nulling_labels[index].float().item())
                patient_dict[patient_id]['true_fibrosis_tissue_enhancement_attribute'].append(fibrosis_tissue_enhancement_labels[index].float().item())
                patient_dict[patient_id]['true_overall'].append(overall_labels[index].float().item())

            running_loss += loss.item() * inputs.size(0)

        # compute mode from the list of predictions
        for patient_id in patient_dict.keys():
            predicted_patients_sharpness_attribute = mode(patient_dict[patient_id]['pred_sharpness_attribute'])
            predicted_patients_myocardium_nulling_attribute = mode(patient_dict[patient_id]['pred_myocardium_nulling_attribute'])
            predicted_patients_fibrosis_tissue_enhancement_attribute = mode(patient_dict[patient_id]['pred_fibrosis_tissue_enhancement_attribute'])
            predicted_patients_overall = mode(patient_dict[patient_id]['pred_overall'])

            true_patients_sharpness_attribute = patient_dict[patient_id]['true_sharpness_attribute'][0]
            true_patients_myocardium_nulling_attribute = patient_dict[patient_id]['true_myocardium_nulling_attribute'][0]
            true_patients_fibrosis_tissue_enhancement_attribute = patient_dict[patient_id]['true_fibrosis_tissue_enhancement_attribute'][0]
            true_patients_overall = patient_dict[patient_id]['true_overall'][0]

            mse_dict['sharpness_attribute'] = mse_dict.get('sharpness_attribute', 0) + \
                (predicted_patients_sharpness_attribute - true_patients_sharpness_attribute) ** 2
            mse_dict['myocardium_nulling_attribute'] = mse_dict.get('myocardium_nulling_attribute', 0) + \
                (predicted_patients_myocardium_nulling_attribute - true_patients_myocardium_nulling_attribute) ** 2
            mse_dict['fibrosis_tissue_enhancement_attribute'] = mse_dict.get('fibrosis_tissue_enhancement_attribute', 0) + \
                (predicted_patients_fibrosis_tissue_enhancement_attribute - true_patients_fibrosis_tissue_enhancement_attribute) ** 2
            mse_dict['overall'] = mse_dict.get('overall', 0) + \
                (predicted_patients_overall - true_patients_overall) ** 2

            mae_dict['sharpness_attribute'] = mae_dict.get('sharpness_attribute', 0) + \
                abs(predicted_patients_sharpness_attribute - true_patients_sharpness_attribute)
            mae_dict['myocardium_nulling_attribute'] = mae_dict.get('myocardium_nulling_attribute', 0) + \
                abs(predicted_patients_myocardium_nulling_attribute - true_patients_myocardium_nulling_attribute)
            mae_dict['fibrosis_tissue_enhancement_attribute'] = mae_dict.get('fibrosis_tissue_enhancement_attribute', 0) + \
                abs(predicted_patients_fibrosis_tissue_enhancement_attribute - true_patients_fibrosis_tissue_enhancement_attribute)
            mae_dict['overall'] = mae_dict.get('overall', 0) + \
                abs(predicted_patients_overall - true_patients_overall)

        return running_loss, mae_dict, mse_dict


def train_model(model: nn.Module, model_name: str, optimizer, scheduler, dataloaders: Dict[str, DataLoader],
                device, num_epochs=25, num_classes=5):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_mse = 100000.0

    all_epochs_mse = {"train": [], "val": []}
    all_epochs_mae = {"train": [], "val": []}
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

                overall_labels = labels['overall'].to(device)
                sharpness_labels = labels['sharpness'].to(device)
                myocardium_nulling_labels = labels['myocardium_nulling'].to(device)
                fibrosis_tissue_enhancement_labels = labels['enhancement_of_fibrosis_tissue'].to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad() # zero the parameter gradients

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    logits_of_sharpness_attribute, logits_of_myocardium_nulling_attribute, logits_of_fibrosis_tissue_enhancement_attribute, \
                        logits_of_overall = model(inputs)
                    
                    loss_of_sharpness_attribute = corn_loss(logits_of_sharpness_attribute, sharpness_labels, num_classes)
                    loss_of_myocardium_nulling_attribute = corn_loss(logits_of_myocardium_nulling_attribute, \
                        myocardium_nulling_labels, num_classes)
                    loss_of_fibrosis_tissue_enhancement_attribute = corn_loss(logits_of_fibrosis_tissue_enhancement_attribute, \
                        fibrosis_tissue_enhancement_labels, num_classes)
                    
                    loss_of_overall_quality = corn_loss(logits_of_overall, overall_labels, num_classes)
                    
                    all_attribute_loss = loss_of_sharpness_attribute + loss_of_myocardium_nulling_attribute + loss_of_fibrosis_tissue_enhancement_attribute
                    loss = all_attribute_loss + loss_of_overall_quality

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            running_loss, mae_dict, mse_dict = compute_loss_and_metrics(model, dataloaders[phase], device, num_classes)
            epoch_sharpness_attribute_mse = mse_dict['sharpness_attribute'] / len(dataloaders[phase])
            epoch_myocardium_nulling_attribute_mse = mse_dict['myocardium_nulling_attribute'] / len(dataloaders[phase])
            epoch_fibrosis_tissue_enhancement_attribute_mse = mse_dict['fibrosis_tissue_enhancement_attribute'] / len(dataloaders[phase])
            epoch_overall_mse = mse_dict['overall'] / len(dataloaders[phase])

            epoch_sharpness_attribute_mae = mae_dict['sharpness_attribute'] / len(dataloaders[phase])
            epoch_myocardium_nulling_attribute_mae = mae_dict['myocardium_nulling_attribute'] / len(dataloaders[phase])
            epoch_fibrosis_tissue_enhancement_attribute_mae = mae_dict['fibrosis_tissue_enhancement_attribute'] / len(dataloaders[phase])
            epoch_overall_mae = mae_dict['overall'] / len(dataloaders[phase])

            epoch_loss = running_loss / len(dataloaders[phase])
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]
            all_epochs_mse[phase].append({"sharpness_attribute": epoch_sharpness_attribute_mse,
             "myocardium_nulling_attribute": epoch_myocardium_nulling_attribute_mse,
             "fibrosis_tissue_enhancement_attribute": epoch_fibrosis_tissue_enhancement_attribute_mse,
             "overall": epoch_overall_mse})

            all_epochs_mae[phase].append({"sharpness_attribute": epoch_sharpness_attribute_mae,
             "myocardium_nulling_attribute": epoch_myocardium_nulling_attribute_mae,
             "fibrosis_tissue_enhancement_attribute": epoch_fibrosis_tissue_enhancement_attribute_mae,
             "overall": epoch_overall_mae})
             
            all_epochs_loss[phase].append(epoch_loss)

            print(f'{phase} Loss: {epoch_loss:.4f}, sharpness_attribute MSE: {epoch_sharpness_attribute_mse:.4f}, myocardium_nulling_attribute MSE: {epoch_myocardium_nulling_attribute_mse:.4f}, fibrosis_tissue_enhancement_attribute MSE: {epoch_fibrosis_tissue_enhancement_attribute_mse:.4f}, overall MSE: {epoch_overall_mse:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_overall_mse < best_mse:
                best_mse = epoch_overall_mse
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
    print("Saving model weights...")
    torch.save(best_model_wts, f"model/saved_models/{model_name}_best.pth")

    # load best model weights
    model.load_state_dict(best_model_wts)

    # transfer to cpu
    model.to("cpu")

    # Plot the confusion matrix of training set
    print("Getting training set labels and predictions...")
    train_true_labels, train_pred_labels = get_true_vs_predicted_values(dataloaders['train'], model)
    # plot_confusion_matrix(train_true_labels, train_pred_labels, f"{model_name}_train_confusion_matrix")

    # Plot the confusion matrix of testing set
    print("Getting validation set labels and predictions...")
    val_true_labels, val_pred_labels = get_true_vs_predicted_values(dataloaders['val'], model)
    # plot_confusion_matrix(val_true_labels, val_pred_labels, f"{model_name}_test_confusion_matrix")

    return best_mse, all_epochs_mse, all_epochs_loss, train_true_labels, train_pred_labels, val_true_labels, val_pred_labels
