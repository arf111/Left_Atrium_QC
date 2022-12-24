from typing import List
from torch import nn
from model.QCModel import QCModel

class BaseRegressorModel(nn.Module):
    def __init__(self, model: QCModel, layer_names: List[str]):
        super(BaseRegressorModel, self).__init__()
        # make a model from the layer names
        self.model = nn.Sequential(*[getattr(model, name) for name in layer_names])
    
    def forward(self, x):
        # add extra dimension for the batch size
        x = x.unsqueeze(0)

        # get first layer output
        x = self.model[0](x)

        if type(x) == list:
            x = x[-1] # shape: (batch_size, 320, 8, 8)

        # get second layer output
        x = self.model[1](x) # shape: (batch_size, 320, 1, 1)

        # flatten the output
        x = x.view(x.size(0), -1) # shape: (batch_size, 320)

        # get third layer output
        x = self.model[2](x) # shape: (batch_size, 1)

        return x