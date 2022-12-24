import torch

class RegressorClassifierOutput:
    def __init__(self, rank, num_classes: int):
        if rank >= num_classes:
            raise ValueError(f"Rank {rank} must be less than number of classes {num_classes}")
            
        self.rank = rank - 1 # rank is 1-indexed, but we need 0-indexed

    def __call__(self, model_output: torch.Tensor):
        probas = torch.sigmoid(model_output) # sigmoid activation function for logits to get probabilities (0-1) for each class label (dim=1) for each example (dim=0) in the batch of logits (dim=0) 
        if len(probas.shape) == 1:
            probas = probas.unsqueeze(0)
        
        probas = torch.cumprod(probas, dim=1) # cumulative product along dim=1 (classes)
        
        return probas[:, self.rank]