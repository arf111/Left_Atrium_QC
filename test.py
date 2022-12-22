from pathlib import Path
from statistics import mode
import numpy as np

import torch
from coral_pytorch.dataset import corn_label_from_logits

import pandas as pd
import pingouin as pg
import segmentation_models_pytorch as smp
from matplotlib import pyplot as plt
from tqdm import tqdm

from collections import defaultdict
from data.afib_dataset import AfibDataset
from model.QCModel import QCModel
from util.utils import get_test_dataloader, get_true_vs_predicted_values, plot_confusion_matrix

resnet_families = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
mobile_net_families = ['mobilenet_v2', 'timm-mobilenetv3_large_075', 'timm-mobilenetv3_large_100',
                       'timm-mobilenetv3_large_minimal_100', 'timm-mobilenetv3_small_075',
                       'timm-mobilenetv3_small_100', 'timm-mobilenetv3_small_minimal_100']
efficient_net_families = ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4',
                          'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7']

model_families_dict = {'efficientnet': efficient_net_families}
NUM_CLASSES = 5

def test_model(dataloader, model, device):
    with torch.no_grad():
        patient_dict = defaultdict(lambda: defaultdict(list))
        mae_dict, mse_dict = {}, {}

        for i, data in enumerate(dataloader):
            inputs, targets = data['input'], data['target']
            inputs = inputs.to(device)
            
            overall_labels = targets['overall'].to(device)
            sharpness_labels = targets['sharpness'].to(device)
            myocardium_nulling_labels = targets['myocardium_nulling'].to(device)
            fibrosis_tissue_enhancement_labels = targets['enhancement_of_fibrosis_tissue'].to(device)    
           
            logits_of_sharpness_attribute, logits_of_myocardium_nulling_attribute, logits_of_fibrosis_tissue_enhancement_attribute, \
                logits_of_overall = model(inputs)
                    
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
        
        return mae_dict, mse_dict


def plot_true_vs_predicted_values(test_true_labels, test_pred_labels, model_name, title):
    # plot histogram
    fig, ax = plt.subplots()
    ax.hist(test_true_labels, bins=20, alpha=0.5, label='true')
    ax.hist(test_pred_labels, bins=20, alpha=0.5, label='predicted')
    ax.set_title('Histogram of true and predicted values of ' + model_name)
    ax.set_xlabel('True and predicted values')
    ax.set_ylabel('Frequency')
    ax.legend(loc='upper right')
    fig.savefig(f'./results/test_histogram/{model_name}_{title}_histogram.png')


if __name__ == '__main__':
    root_dir = Path('dataset/afib_data')

    test_loader = get_test_dataloader(root_dir, AfibDataset)

    device = "cpu"
    #
    model_mae = {}
    model_mse = {}
    
    model_names = []
    true_labels = []
    pred_labels = []
    
    # df = pd.read_csv('results/afib_results.csv')
    for model_family, model_families in tqdm(model_families_dict.items()):
        for model_name in tqdm(model_families):
            model = smp.Unet(
                encoder_name=model_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
                in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=3,  # model output channels (number of classes in your dataset)
            )
            # model
            qc_model = QCModel(model.encoder, model_family, NUM_CLASSES).to(device)
            qc_model.load_state_dict(torch.load(f'model/saved_models/{model_name}_best.pth'))
            qc_model.eval()
            
            test_true_labels, test_pred_labels = get_true_vs_predicted_values(test_loader, qc_model)
            # plot histogram of true vs predicted values for efficientnet-b1 model
            # efficient_net_true_labels = df['efficientnet-b1_val_true_labels'][:112].values
            # efficient_net_pred_labels = df['efficientnet-b1_val_pred_labels'][:112].values
            
            # true_labels_dict["sharpness"].append(true_patients_sharpness_attribute)
            #         true_labels_dict["myocardium_nulling"].append(true_patients_myocardium_nulling_attribute)
            #         true_labels_dict["enhancement_of_fibrosis_tissue"].append(true_patients_fibrosis_tissue_enhancement_attribute)
            #         true_labels_dict["overall"].append(true_patients_overall)

            #         predicted_labels_dict["sharpness"].append(predicted_patients_sharpness_attribute)
            #         predicted_labels_dict["myocardium_nulling"].append(predicted_patients_myocardium_nulling_attribute)
            #         predicted_labels_dict["enhancement_of_fibrosis_tissue"].append(predicted_patients_fibrosis_tissue_enhancement_attribute)
            #         predicted_labels_dict["overall"].append(predicted_patients_overall)
            
            # plot histogram of true vs predicted values for efficientnet-b1 model
            for attribute in test_true_labels:
                plot_true_vs_predicted_values(test_true_labels[attribute], test_pred_labels[attribute], model_name, attribute)
                plot_confusion_matrix(test_true_labels[attribute], test_pred_labels[attribute], model_name, attribute)
            
            # make a dataframe of the results
            # df = pd.DataFrame({'model_name': model_names, 'true_labels': true_labels, 'pred_labels': pred_labels})
            
            # make a dataframe
            # df_mae = pd.DataFrame(model_mae, index=[0])
            # df_mse = pd.DataFrame(model_mse, index=[0])
            # # df_mae.to_csv('results/mae.csv')
            # df_mse.to_csv('results/mse.csv')
