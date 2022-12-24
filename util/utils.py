import os
from pathlib import Path
from statistics import mode
from typing import Dict, List

from collections import defaultdict
import numpy as np

import pandas as pd
import torch
from PIL import Image
from coral_pytorch.dataset import corn_label_from_logits
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import cv2

from dto.constants import TRAIN_TRANSFORM, VAL_TRANSFORM, RESIZE_IMG, BATCH_SIZE, DATASET_SCORE_FILE


def get_image_from_path(path: Path):
    img = np.array(Image.open(path)).astype(np.float32)
    img = cv2.resize(img, (RESIZE_IMG, RESIZE_IMG))

    # convert gray to rgb
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # normalize img
    max_val = img_color.max()
    img_float = img_color / max_val

    # convert img to PIL
    img = Image.fromarray(img)

    input_tensor = VAL_TRANSFORM(img)

    return img, img_float, input_tensor, max_val


def get_images_with_labels(path, dataframe):
    all_images = []
    labels = []
    for f in path.iterdir():
        if f.is_dir():
            for img in f.iterdir():
                if img.suffix == '.png' and dataframe[dataframe['Study ID'] == f.stem]['Start'].values[0] <= \
                        int(img.stem) <= dataframe[dataframe['Study ID'] == f.stem]['End'].values[0]:
                    all_images.append(img)
                    labels.append({"quality_for_fibrosis_assessment": dataframe[dataframe['Study ID'] == f.stem][
                        "Quality for Fibrosis Assessment"].values[0],
                                   "enhancement_of_fibrosis_tissue": dataframe[dataframe['Study ID'] == f.stem][
                                       'Enhancement of Fibrous Tissue'].values[0],
                                   "sharpness": dataframe[dataframe['Study ID'] == f.stem]['Sharpness'].values[0],
                                   "myocardium_nulling":
                                       dataframe[dataframe['Study ID'] == f.stem]['Myocardium Nulling'].values[0],
                                   "overall": int(
                                       dataframe[dataframe['Study ID'] == f.stem][
                                           'Quality for Fibrosis Assessment'].values[0]),
                                   "patient_id": f.stem})
    return all_images, labels


def get_test_dataloader(root_dir, AfibDataset):
    test_transform = VAL_TRANSFORM

    dataset_dir = Path(root_dir)
    dataset_scores_path = dataset_dir / f"All_IQ_Scores/{DATASET_SCORE_FILE}"
    dataframe = pd.read_csv(dataset_scores_path)
    dataframe = dataframe.dropna(subset=['Quality for Fibrosis Assessment'])
    all_images, labels = get_images_with_labels(root_dir / 'test', dataframe)

    test_dataset = AfibDataset(all_images, labels, transform=test_transform)

    test_loader = DataLoader(dataset=test_dataset, num_workers=8, batch_size=BATCH_SIZE, shuffle=True)

    return test_loader


def get_dataloader(root_dir, AfibDataset):
    train_transform = TRAIN_TRANSFORM
    val_transform = VAL_TRANSFORM

    dataset_dir = Path(root_dir)
    dataset_scores_path = dataset_dir / f"All_IQ_Scores/{DATASET_SCORE_FILE}"
    dataframe = pd.read_csv(dataset_scores_path)
    dataframe = dataframe.dropna(subset=['Quality for Fibrosis Assessment'])
    all_images, labels = get_images_with_labels(root_dir / 'training', dataframe)

    train_image_paths, valid_image_paths, train_labels, valid_labels = train_test_split(all_images, labels,
                                                                                        test_size=0.2, random_state=42)

    train_dataset = AfibDataset(train_image_paths, train_labels, transform=train_transform)
    val_dataset = AfibDataset(valid_image_paths, valid_labels, transform=val_transform)

    train_loader = DataLoader(dataset=train_dataset, num_workers=8, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, num_workers=8, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, val_loader


def get_true_vs_predicted_values(dataloader: DataLoader, model: torch.nn.Module):
    """
    Get the true vs predicted plot
    :param dataloader: dataloader of the dataset
    :param model: model to test
    :param device: device to use
    :return: list, list, list, list
    """
    model.eval()
    with torch.no_grad():
        patient_dict = defaultdict(lambda: defaultdict(list))
        true_labels_dict = defaultdict(list)
        predicted_labels_dict = defaultdict(list)
        for i, data in enumerate(dataloader):
            features, targets = data['input'], data['target']
            logits_of_sharpness_attribute, logits_of_myocardium_nulling_attribute, logits_of_fibrosis_tissue_enhancement_attribute, \
                logits_of_overall = model(features)

            overall_labels = targets['overall']
            sharpness_labels = targets['sharpness']
            myocardium_nulling_labels = targets['myocardium_nulling']
            fibrosis_tissue_enhancement_labels = targets['enhancement_of_fibrosis_tissue']

            predicted_logits_of_sharpness_attribute = corn_label_from_logits(
                logits_of_sharpness_attribute).float().tolist()
            predicted_logits_of_myocardium_nulling_attribute = corn_label_from_logits(
                logits_of_myocardium_nulling_attribute).float().tolist()
            predicted_logits_of_fibrosis_tissue_enhancement_attribute = corn_label_from_logits(
                logits_of_fibrosis_tissue_enhancement_attribute).float().tolist()
            predicted_logits_of_overall = corn_label_from_logits(logits_of_overall).float().tolist()

            for index, patient_id in enumerate(data["patient_id"]):
                patient_dict[patient_id]['pred_sharpness_attribute'].append(
                    predicted_logits_of_sharpness_attribute[index] + 1)
                patient_dict[patient_id]['pred_myocardium_nulling_attribute'].append(
                    predicted_logits_of_myocardium_nulling_attribute[index] + 1)
                patient_dict[patient_id]['pred_fibrosis_tissue_enhancement_attribute'].append(
                    predicted_logits_of_fibrosis_tissue_enhancement_attribute[index] + 1)
                patient_dict[patient_id]['pred_overall'].append(predicted_logits_of_overall[index] + 1)

                patient_dict[patient_id]['true_sharpness_attribute'].append(sharpness_labels[index].float().item() + 1)
                patient_dict[patient_id]['true_myocardium_nulling_attribute'].append(
                    myocardium_nulling_labels[index].float().item() + 1)
                patient_dict[patient_id]['true_fibrosis_tissue_enhancement_attribute'].append(
                    fibrosis_tissue_enhancement_labels[index].float().item() + 1)
                patient_dict[patient_id]['true_overall'].append(overall_labels[index].float().item() + 1)

            # completed data points
            print(f"Completed {i + 1} / {len(dataloader)} batches", end="\r")

        # compute mode from the list of predictions
        for patient_id in patient_dict.keys():
            predicted_patients_sharpness_attribute = mode(patient_dict[patient_id]['pred_sharpness_attribute'])
            predicted_patients_myocardium_nulling_attribute = mode(
                patient_dict[patient_id]['pred_myocardium_nulling_attribute'])
            predicted_patients_fibrosis_tissue_enhancement_attribute = mode(
                patient_dict[patient_id]['pred_fibrosis_tissue_enhancement_attribute'])
            predicted_patients_overall = mode(patient_dict[patient_id]['pred_overall'])

            true_patients_sharpness_attribute = patient_dict[patient_id]['true_sharpness_attribute'][0]
            true_patients_myocardium_nulling_attribute = patient_dict[patient_id]['true_myocardium_nulling_attribute'][
                0]
            true_patients_fibrosis_tissue_enhancement_attribute = \
            patient_dict[patient_id]['true_fibrosis_tissue_enhancement_attribute'][0]
            true_patients_overall = patient_dict[patient_id]['true_overall'][0]

            true_labels_dict["sharpness"].append(true_patients_sharpness_attribute)
            true_labels_dict["myocardium_nulling"].append(true_patients_myocardium_nulling_attribute)
            true_labels_dict["enhancement_of_fibrosis_tissue"].append(
                true_patients_fibrosis_tissue_enhancement_attribute)
            true_labels_dict["overall"].append(true_patients_overall)

            predicted_labels_dict["sharpness"].append(predicted_patients_sharpness_attribute)
            predicted_labels_dict["myocardium_nulling"].append(predicted_patients_myocardium_nulling_attribute)
            predicted_labels_dict["enhancement_of_fibrosis_tissue"].append(
                predicted_patients_fibrosis_tissue_enhancement_attribute)
            predicted_labels_dict["overall"].append(predicted_patients_overall)
    return true_labels_dict, predicted_labels_dict


def get_model_output(image_filename, model):
    """
    Get the output of a model on a single image
    :param image_filename: filename of the image to visualize
    :param model: model to visualize
    :return: int, the predicted class
    """
    # Load the image
    image = Image.open(image_filename)
    test_transform = Compose([Resize(RESIZE_IMG), ToTensor(), Normalize((0.5,), (0.5,))])
    image = test_transform(image)
    image = image.unsqueeze(0)

    # Get the model output
    model.eval()
    with torch.no_grad():
        output = model(image)

    # Get the predicted score
    preds = corn_label_from_logits(output).float()

    return preds.item()


def plot_confusion_matrix(true_labels, pred_labels, model_name, title):
    """
    Plot the confusion matrix
    :param title: str, title of the plot
    :param true_labels: list of true labels
    :param pred_labels: list of predicted labels
    :return: None
    """
    # check if path exists
    path = os.path.abspath(os.path.join("results/confusion_matrix/", os.pardir))

    disp = ConfusionMatrixDisplay.from_predictions(true_labels, pred_labels)
    disp.plot(cmap="OrRd")
    plt.title(title)
    plt.savefig(f"{path}/confusion_matrix/{model_name}_{title}_cf_matrix.png")
    # plt.show()
