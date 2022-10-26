import torch
from PIL.Image import Image
from coral_pytorch.dataset import corn_label_from_logits
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

RESIZE_IMG = 256


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
        true_labels = []
        predicted_labels = []
        for i, data in enumerate(dataloader):
            features, targets = data['input'], data['target']
            logits = model(features)
            predicted_labels.append(corn_label_from_logits(logits).float())
            true_labels.append(targets)

        return [value.item() for elem in true_labels for value in elem], [value.item() for elem in predicted_labels for
                                                                          value in elem]


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
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[int(i) for i in range(1, max(true_labels) + 1)])
    disp.plot()
    plt.title(title)
    plt.savefig(f"{title}.png")
    plt.show()
