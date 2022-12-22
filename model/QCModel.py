import torch
from torch import nn
import segmentation_models_pytorch as smp

from model.attribute_classifier import AttributeClassifier
from model.overall_classifier import OverallClassifier


class QCModel(nn.Module):
    def __init__(self, encoder, encoder_name, num_classes):
        super(QCModel, self).__init__()

        encoder_layers = None
        features = None

        if encoder_name == "efficientnet":
            features = encoder
        else:
            encoder_layers = encoder.children()
            features = nn.Sequential(*encoder_layers)

        for p in features.parameters():
            p.requires_grad = False

        self.features = nn.Sequential(
            features
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.attribute_sharpness = AttributeClassifier(encoder.out_channels[-1], num_classes)
        self.attribute_myocardium_nulling = AttributeClassifier(encoder.out_channels[-1], num_classes)
        self.attribute_fibrosis_tissue_enhancement = AttributeClassifier(encoder.out_channels[-1], num_classes)

        no_of_attribute_out_channels = self.attribute_sharpness.output_layer.out_features * 3  # 3 attributes

        self.overall_classifier = OverallClassifier(no_of_attribute_out_channels, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.features(x)  # (batch_size, 2048, 8, 8)
        if type(x) == list:
            x = x[-1]

        x = self.adaptive_pool(x)  # (batch_size, 2048, 1, 1)
        feature_space = x.view(x.size(0), -1)  # flatten (batch_size, 2048)

        logits_of_sharpness_attribute = self.attribute_sharpness(feature_space)  # (batch_size, num_classes-1)
        logits_of_myocardium_nulling_attribute = self.attribute_myocardium_nulling(
            feature_space)  # (batch_size, num_classes-1)
        logits_of_fibrosis_tissue_enhancement_attribute = self.attribute_fibrosis_tissue_enhancement(
            feature_space)  # (batch_size, num_classes-1)

        logits_of_attributes = torch.cat((logits_of_sharpness_attribute, logits_of_myocardium_nulling_attribute,
                                          logits_of_fibrosis_tissue_enhancement_attribute),
                                         dim=1)  # (batch_size, 3 * (num_classes-1))

        logits_of_attributes = self.relu(logits_of_attributes)  # Non-linearity

        logits_of_overall = self.overall_classifier(logits_of_attributes)  # (batch_size, num_classes-1)

        return logits_of_sharpness_attribute, logits_of_myocardium_nulling_attribute, \
            logits_of_fibrosis_tissue_enhancement_attribute, logits_of_overall


if __name__ == '__main__':
    model = smp.Unet(
        encoder_name="resnet50",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=3,  # model output channels (number of classes in your dataset)
    )

    qc_model = QCModel(model.encoder, "resnet50", 5)

    input = torch.randn(4, 1, 512, 480)  # (batch_size, channels, height, width)
    logits_of_sharpness_attribute, logits_of_myocardium_nulling_attribute, logits_of_fibrosis_tissue_enhancement_attribute, logits_of_overall = qc_model(
        input)
    print(logits_of_sharpness_attribute.shape)
    print(logits_of_myocardium_nulling_attribute.shape)
    print(logits_of_fibrosis_tissue_enhancement_attribute.shape)
    print(logits_of_overall.shape)
