import torch
import numpy as np
from torch.utils.data import DataLoader

def inference(
    test_loader : DataLoader, 
    model : torch.nn.Module,
    device : str = "cpu",
    ):

    model.eval()
    model.to(device)
    total_pred = []

    for batch_idx, data in enumerate(test_loader):
        with torch.no_grad():
            data = data.to(device)
            output = model(data)
            total_pred.extend(output.detach().cpu().numpy().tolist())
        
    total_pred = np.array(total_pred)

    return total_pred