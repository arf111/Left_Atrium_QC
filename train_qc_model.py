import itertools
from pathlib import Path

import pandas as pd
import torch
from torch import optim

from data.afib_dataset import AfibDataset
from data.atria_dataset import AtriaDataset
import segmentation_models_pytorch as smp

from model.QCModel import QCModel
from train import train_model
from util.utils import get_images_with_labels, get_dataloader
from tqdm import tqdm

NUM_CLASSES = 5
RESIZE_IMG = 256
NUM_EPOCHS = 60

resnet_families = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
mobile_net_families = ['mobilenet_v2', 'timm-mobilenetv3_large_075', 'timm-mobilenetv3_large_100',
                       'timm-mobilenetv3_large_minimal_100', 'timm-mobilenetv3_small_075',
                       'timm-mobilenetv3_small_100', 'timm-mobilenetv3_small_minimal_100']
efficient_net_families = ['efficientnet-b0', 
                         'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4',
                          'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7'
                        ]

model_families_dict = {'efficientnet': efficient_net_families}

if __name__ == "__main__":
    root_dir = Path('dataset/afib_data')

    train_loader, val_loader = get_dataloader(root_dir, AfibDataset) # training = 444

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dict_of_model_with_loss_and_mse = {}

    for model_family, model_families in tqdm(model_families_dict.items()):
        for model_name in tqdm(model_families):
            print("\nTraining model: ", model_name)
            model = smp.Unet(
                encoder_name=model_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
                in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=3,  # model output channels (number of classes in your dataset)
            )
            # model

            qc_model = QCModel(model.encoder, model_family, NUM_CLASSES).to(device)

            # optimizer
            optimizer = optim.SGD(qc_model.parameters(), lr=0.001, momentum=0.99, weight_decay=0.0005)

            # scheduler
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

            dataloaders = {"train": train_loader, "val": val_loader}
            dataset_sizes = {"train": len(train_loader.dataset), "val": len(val_loader.dataset)}

            best_mse, all_epochs_mse, all_epochs_loss, train_true_labels, train_pred_labels, val_true_labels, val_pred_labels = train_model(
                model=qc_model, model_name=model_name,
                dataloaders=dataloaders, optimizer=optimizer,
                scheduler=scheduler, device=device,
                num_epochs=NUM_EPOCHS, num_classes=NUM_CLASSES)

            print("Making dict of model with loss and mse")
            dict_of_model_with_loss_and_mse[f"{model_name}_train_loss"] = all_epochs_loss['train']
            dict_of_model_with_loss_and_mse[f"{model_name}_val_loss"] = all_epochs_loss['val']
            
            dict_of_model_with_loss_and_mse[f"{model_name}_train_sharpness_attribute_mse"] = [x['sharpness_attribute'] for x in all_epochs_mse['train']]
            dict_of_model_with_loss_and_mse[f"{model_name}_val_sharpness_attribute_mse"] = [x['sharpness_attribute'] for x in all_epochs_mse['val']]
            dict_of_model_with_loss_and_mse[f"{model_name}_train_myocardium_nulling_attribute_mse"] = [x['myocardium_nulling_attribute'] for x in all_epochs_mse['train']]
            dict_of_model_with_loss_and_mse[f"{model_name}_val_myocardium_nulling_attribute_mse"] = [x['myocardium_nulling_attribute'] for x in all_epochs_mse['val']]
            dict_of_model_with_loss_and_mse[f"{model_name}_train_fibrosis_tissue_enhancement_attribute_mse"] = [x['fibrosis_tissue_enhancement_attribute'] for x in all_epochs_mse['train']]
            dict_of_model_with_loss_and_mse[f"{model_name}_val_fibrosis_tissue_enhancement_attribute_mse"] = [x['fibrosis_tissue_enhancement_attribute'] for x in all_epochs_mse['val']]
            dict_of_model_with_loss_and_mse[f"{model_name}_train_overall_mse"] = [x['overall'] for x in all_epochs_mse['train']]
            dict_of_model_with_loss_and_mse[f"{model_name}_val_overall_mse"] = [x['overall'] for x in all_epochs_mse['val']]

            dict_of_model_with_loss_and_mse[f"{model_name}_train_true_labels_sharpness"] = train_true_labels["sharpness"]
            dict_of_model_with_loss_and_mse[f"{model_name}_train_true_labels_myocardium_nulling"] = train_true_labels["myocardium_nulling"]
            dict_of_model_with_loss_and_mse[f"{model_name}_train_true_labels_fibrosis_tissue_enhancement"] = train_true_labels["enhancement_of_fibrosis_tissue"]
            dict_of_model_with_loss_and_mse[f"{model_name}_train_true_labels_overall"] = train_true_labels["overall"]
            
            dict_of_model_with_loss_and_mse[f"{model_name}_train_pred_labels_sharpness"] = train_pred_labels["sharpness"]
            dict_of_model_with_loss_and_mse[f"{model_name}_train_pred_labels_myocardium_nulling"] = train_pred_labels["myocardium_nulling"]
            dict_of_model_with_loss_and_mse[f"{model_name}_train_pred_labels_fibrosis_tissue_enhancement"] = train_pred_labels["enhancement_of_fibrosis_tissue"]
            dict_of_model_with_loss_and_mse[f"{model_name}_train_pred_labels_overall"] = train_pred_labels["overall"]
            
            dict_of_model_with_loss_and_mse[f"{model_name}_val_true_labels_sharpness"] = val_true_labels["sharpness"]
            dict_of_model_with_loss_and_mse[f"{model_name}_val_true_labels_myocardium_nulling"] = val_true_labels["myocardium_nulling"]
            dict_of_model_with_loss_and_mse[f"{model_name}_val_true_labels_fibrosis_tissue_enhancement"] = val_true_labels["enhancement_of_fibrosis_tissue"]
            dict_of_model_with_loss_and_mse[f"{model_name}_val_true_labels_overall"] = val_true_labels["overall"]

            dict_of_model_with_loss_and_mse[f"{model_name}_val_pred_labels_sharpness"] = val_pred_labels["sharpness"]
            dict_of_model_with_loss_and_mse[f"{model_name}_val_pred_labels_myocardium_nulling"] = val_pred_labels["myocardium_nulling"]
            dict_of_model_with_loss_and_mse[f"{model_name}_val_pred_labels_fibrosis_tissue_enhancement"] = val_pred_labels["enhancement_of_fibrosis_tissue"]
            dict_of_model_with_loss_and_mse[f"{model_name}_val_pred_labels_overall"] = val_pred_labels["overall"]

            print("Finished dict_of_model_with_loss_and_mse")
            print("-----------------------------------------------------------------------------------------------------")
    for keys, values in dict_of_model_with_loss_and_mse.items():
        if len(values) < 444:
            # fill up the values with 0
            dict_of_model_with_loss_and_mse[keys] = values + [0] * (444 - len(values))

    df = pd.DataFrame(dict_of_model_with_loss_and_mse)
    df.to_csv('results/afib_results.csv')
