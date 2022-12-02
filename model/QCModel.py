import torch
from torch import nn
import segmentation_models_pytorch as smp


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
        
        linear1_out = int(encoder.out_channels[-1] / 4)
        self.linear1 = nn.Linear(encoder.out_channels[-1], linear1_out)

        linear2_out = int(linear1_out / 4)
        self.linear2 = nn.Linear(linear1_out, linear2_out)
        
        self.relu = nn.ReLU()
        
        self.output_layer = nn.Linear(linear2_out, num_classes - 1)

    def forward(self, x):
        x = self.features(x)  # (batch_size, 2048, 16, 15)
        if type(x) == list:
            x = x[-1]

        x = self.adaptive_pool(x)  # (batch_size, 2048, 1, 1)
        x = x.view(x.size(0), -1)  # flatten (batch_size, 2048)
        x = self.linear1(x)  # output shape: (batch_size, 512)
        x = self.relu(x)  # Non-linearity
        x = self.linear2(x)  # output shape: (batch_size, 128)
        x = self.relu(x)  # Non-linearity

        logits = self.output_layer(x)  # output shape: (batch_size, num_classes-1)

        return logits


if __name__ == '__main__':
    model = smp.Unet(
        encoder_name="resnet50",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=3,  # model output channels (number of classes in your dataset)
    )

    qc_model = QCModel(model.encoder, "resnet50", 5)

    input = torch.randn(4, 1, 512, 480)
    output = qc_model(input)
    print(output.shape)

