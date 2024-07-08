import torch
import os
import pickle

from datasets import EmbryoDatasetWithAdjacency
from models import BaselineCNN

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = EmbryoDatasetWithAdjacency(clinical_endpoint='simplified_outcome')
    loader = torch.utils.data.DataLoader(dataset)
    model = BaselineCNN().to(device)

    # Rip head off model
    model.net.fc = torch.nn.Identity()

    # Make bottlenecks
    model.eval()    
    for i, x in enumerate(loader):
        stack, target = x['stack'], x['label']
        stack = stack.to(device).float()
        bottleneck = model(stack)
        torch.save(
            {
                'stack': bottleneck.squeeze(), 
                'label': target[0]
            },
            os.path.join('data', 'bottlenecks', f'{i}.bottleneck')
        )
