import torch
from coral_pytorch.dataset import corn_label_from_logits


def test_model(dataloader, model, device):
    with torch.no_grad():
        mae, mse, acc, num_examples = 0., 0., 0., 0

        for i, (features, targets) in enumerate(dataloader):
            features = features.to(device)
            targets = targets.float().to(device)
            targets = targets.flatten()

            logits = model(features)
            predicted_labels = corn_label_from_logits(logits).float()

            num_examples += targets.size(0)
            mae += torch.sum(torch.abs(predicted_labels - targets))
            mse += torch.sum((predicted_labels - targets) ** 2)

        mae = mae / len(dataloader)
        mse = mse / len(dataloader)
        return mae, mse
