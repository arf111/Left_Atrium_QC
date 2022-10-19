import torch
from torch import nn
import segmentation_models_pytorch as smp


class QCModel(nn.Module):
    def __init__(self, encoder, num_classes):
        super(QCModel, self).__init__()
        encoder_layers = encoder.children()
        features = nn.Sequential(*encoder_layers)

        for p in features.parameters():
            p.requires_grad = False

        self.model = nn.Sequential(
            features,
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

        self.output_layer = nn.Linear(in_features=2048, out_features=num_classes - 1)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)  # flatten

        logits = self.output_layer(x)

        return logits


if __name__ == '__main__':
    model = smp.Unet(
        encoder_name="resnet50",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=3,  # model output channels (number of classes in your dataset)
    )

    qc_model = QCModel(model.encoder, 5)

    input = torch.randn(4, 1, 512, 480)
    output = qc_model(input)
    print(output.shape)
