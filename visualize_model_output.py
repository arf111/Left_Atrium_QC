import numpy as np
import torch
import segmentation_models_pytorch as smp

from model.QCModel import QCModel
from train_qc_model import get_dataloader
from util.utils import get_true_vs_predicted_values, plot_confusion_matrix

if __name__ == "__main__":
    # Load the weights of the model
    model = smp.Unet(
        encoder_name="resnet50",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=3,  # model output channels (number of classes in your dataset)
    )
    # model
    qc_model = QCModel(model.encoder, 5)
    qc_model.load_state_dict(torch.load('model/saved_models/best_model.pth'))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get the model output
    # predicted_score = get_model_output("dataset/testing_set/8ISA3BO39HPSNJI0R8NH/38.png", qc_model)
    # print(predicted_score)

    train_loader, val_loader = get_dataloader("dataset")

    # Plot the confusion matrix of training set
    true_labels, pred_labels = get_true_vs_predicted_values(train_loader, qc_model)
    plot_confusion_matrix(true_labels, pred_labels, "train_confusion_matrix")

    # Plot the confusion matrix of testing set
    true_labels, pred_labels = get_true_vs_predicted_values(val_loader, qc_model)
    plot_confusion_matrix(true_labels, pred_labels, "test_confusion_matrix")
