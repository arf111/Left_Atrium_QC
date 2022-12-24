from pathlib import Path
import numpy as np
import torch
import segmentation_models_pytorch as smp
from dto.constants import NUM_CLASSES
from coral_pytorch.dataset import corn_label_from_logits
from explainer.base_regressor_model import BaseRegressorModel
from explainer.regression_classifier_output import RegressorClassifierOutput
from model.QCModel import QCModel
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import matplotlib.pyplot as plt

from util.utils import get_image_from_path

if __name__ == '__main__':
    model_name = 'efficientnet-b1'
    model_family = 'efficientnet'
    layer_names = ['features', 'adaptive_pool', 'attribute_sharpness']
    img_test_path =  Path('dataset/afib_data/test/IQ001/23.png')
    true_label = 4

    model = smp.Unet(
                encoder_name=model_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
                in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=3,  # model output channels (number of classes in your dataset)
            )
            # model
    qc_model = QCModel(model.encoder, model_family, NUM_CLASSES, True).to("cpu")
    qc_model.load_state_dict(torch.load(f'model/saved_models/{model_name}_best.pth'))
    
    base_regressor_model = BaseRegressorModel(qc_model, ['features', 'adaptive_pool', 'attribute_fibrosis_tissue_enhancement'])
    
    target_layers = [qc_model.features[-1]]

    img, img_float, input_tensor, img_float_max_val = get_image_from_path(img_test_path)

    logits = base_regressor_model(input_tensor)

    predicted_rank = corn_label_from_logits(logits)

    mri_rank_target = [RegressorClassifierOutput(rank=predicted_rank, num_classes=NUM_CLASSES)]

    cam = GradCAM(model=base_regressor_model, target_layers=target_layers, use_cuda=False) 

    mri_grayscale_cam = cam(input_tensor=input_tensor, targets=mri_rank_target)

    # squeeze the grayscale cam's 1st dimension
    mri_grayscale_cam = mri_grayscale_cam.squeeze(0)
    
    mri_cam_image = show_cam_on_image(img_float, mri_grayscale_cam, use_rgb=False)
    # plot the img_float and mri_cam_image side by side
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(np.uint8(img_float * 255))
    ax[1].imshow(mri_cam_image)
    ax[0].text(ax[0].get_xlim()[0], ax[0].get_ylim()[1], f'True Rank: {true_label}', color='red', fontsize=20)
    ax[1].text(ax[1].get_xlim()[0], ax[1].get_ylim()[1], f'Predicted Rank: {predicted_rank.item()}', color='red', fontsize=20)
    plt.show()
    plt.savefig('cam.png')