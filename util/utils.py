from pathlib import Path
from typing import Dict, List

from collections import defaultdict

import pandas as pd
import torch
from PIL.Image import Image
from coral_pytorch.dataset import corn_label_from_logits
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip

RESIZE_IMG = 256
BATCH_SIZE = 32

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
                                       dataframe[dataframe['Study ID'] == f.stem]['Overall Image Quality'].values[0])})
    return all_images, labels


def get_test_dataloader(root_dir, AfibDataset):
    test_transform = Compose([Resize(RESIZE_IMG), ToTensor()])

    dataset_dir = Path(root_dir)
    dataset_scores_path = dataset_dir / "All_IQ_Scores/Image_Quality_Assessment_BAO.csv"
    dataframe = pd.read_csv(dataset_scores_path)
    dataframe = dataframe.dropna(subset=['Overall Image Quality'])
    all_images, labels = get_images_with_labels(root_dir / 'test', dataframe)

    test_dataset = AfibDataset(all_images, labels, transform=test_transform)

    test_loader = DataLoader(dataset=test_dataset, num_workers=8, batch_size=BATCH_SIZE, shuffle=True)

    return test_loader


def get_dataloader(root_dir, AfibDataset):
    train_transform = Compose([Resize(RESIZE_IMG), RandomHorizontalFlip(), ToTensor()])
    val_transform = Compose([Resize(RESIZE_IMG), ToTensor()])

    dataset_dir = Path(root_dir)
    dataset_scores_path = dataset_dir / "All_IQ_Scores/Image_Quality_Assessment_BAO.csv"
    dataframe = pd.read_csv(dataset_scores_path)
    dataframe = dataframe.dropna(subset=['Overall Image Quality'])
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
        true_labels_dict = defaultdict(list)
        predicted_labels_dict = defaultdict(list)
        for i, data in enumerate(dataloader):
            features, targets = data['input'], data['target']
            logits_of_sharpness_attribute, logits_of_myocardium_nulling_attribute, logits_of_fibrosis_tissue_enhancement_attribute, \
                        logits_of_overall = model(features)
            logits_dict = {"sharpness": logits_of_sharpness_attribute, "myocardium_nulling": logits_of_myocardium_nulling_attribute, \
                            "enhancement_of_fibrosis_tissue": logits_of_fibrosis_tissue_enhancement_attribute, "overall": logits_of_overall}

            true_labels_dict["sharpness"].extend(targets["sharpness"])
            true_labels_dict["myocardium_nulling"].extend(targets["myocardium_nulling"])
            true_labels_dict["enhancement_of_fibrosis_tissue"].extend(targets["enhancement_of_fibrosis_tissue"])
            true_labels_dict["overall"].extend(targets["overall"])
            
            predicted_labels_dict["sharpness"].extend(corn_label_from_logits(logits_dict["sharpness"]))
            predicted_labels_dict["myocardium_nulling"].extend(corn_label_from_logits(logits_dict["myocardium_nulling"]))
            predicted_labels_dict["enhancement_of_fibrosis_tissue"].extend(corn_label_from_logits(logits_dict["enhancement_of_fibrosis_tissue"]))
            predicted_labels_dict["overall"].extend(corn_label_from_logits(logits_dict["overall"]))

            # completed data points
            print(f"Completed {i + 1} / {len(dataloader)} batches", end="\r")

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


def plot_confusion_matrix(true_labels, pred_labels, title):
    """
    Plot the confusion matrix
    :param title: str, title of the plot
    :param true_labels: list of true labels
    :param pred_labels: list of predicted labels
    :return: None
    """
    cm = confusion_matrix(true_labels, pred_labels)
    # make confusion matrix color blue
    cm_display = ConfusionMatrixDisplay(cm, display_labels=[1, 2, 3, 4, 5])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[int(i) for i in range(1, cm.shape[0] + 1)])
    disp.plot(cmap="OrRd")
    plt.title(title)
    plt.savefig(f"{title}.png")
    plt.show()
