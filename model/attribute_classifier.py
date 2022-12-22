from torch import nn


class AttributeClassifier(nn.Module):
    def __init__(self, encoder_out_channels, num_classes):
        super(AttributeClassifier, self).__init__()

        self.linear1 = nn.Linear(encoder_out_channels, encoder_out_channels // 4)

        self.linear2 = nn.Linear(self.linear1.out_features, encoder_out_channels // 16)

        self.relu = nn.ReLU()

        self.output_layer = nn.Linear(self.linear2.out_features, num_classes - 1)

    def forward(self, x):
        x = self.linear1(x)  # output shape: (batch_size, encoder_out_channels // 4)
        x = self.relu(x)  # Non-linearity
        x = self.linear2(x)  # output shape: (batch_size, encoder_out_channels // 16)
        x = self.relu(x)  # Non-linearity

        logits = self.output_layer(x)  # output shape: (batch_size, num_classes-1)

        return logits
