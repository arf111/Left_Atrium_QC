from torch import nn


class OverallClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(OverallClassifier, self).__init__()

        self.linear1 = nn.Linear(in_channels, 20)

        self.relu = nn.ReLU()

        self.output_layer = nn.Linear(self.linear1.out_features, num_classes - 1)

    def forward(self, x):
        x = self.linear1(x)  # output shape: (batch_size, 20)
        x = self.relu(x)  # Non-linearity

        logits = self.output_layer(x)  # output shape: (batch_size, num_classes-1)

        return logits
