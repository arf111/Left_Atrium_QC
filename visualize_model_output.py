import numpy as np
import torch
from PIL import Image
from coral_pytorch.dataset import corn_label_from_logits
from torchvision.transforms import ToTensor, Compose, Resize, Normalize
import segmentation_models_pytorch as smp

from model.QCModel import QCModel

RESIZE_IMG = 256


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

    # Get the predicted class
    preds = corn_label_from_logits(output).float()

    return preds.item()


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

    # Get the model output
    predicted_score = get_model_output("dataset/testing_set/4URSJYI2QUH1T5S5PP47/80.png", qc_model)
    print(predicted_score)
