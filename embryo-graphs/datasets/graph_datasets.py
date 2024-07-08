import torch
import torchvision.transforms as T
import numpy as np

from torch_geometric.data import Dataset, Data
from torch_geometric.utils import dense_to_sparse

from .adjacency_dataset import EmbryoDatasetWithAdjacency

class EmbryoGraphDataset(Dataset):
    def __init__(self, dataset_path='data/adjacency_and_bbox_dataset.csv', plane_size=(400, 400), clinical_endpoint=None, 
                 padding=0.1, num_folds=0, fold=0, keep_one_fold_for_testing=False, invert_selection=False):
        super().__init__()
        self.dataset = EmbryoDatasetWithAdjacency(
            dataset_path=dataset_path, 
            plane_size=plane_size, 
            clinical_endpoint=clinical_endpoint, 
            num_folds=num_folds, 
            fold=fold,
            keep_one_fold_for_testing=False,
            invert_selection=invert_selection
        )
        self.padding = padding

    def __getitem__(self, idx):
        embryo = self.dataset[idx]
        # Create nodes
        nodes = []
        for cx, cy, w, h, d in embryo['bboxes']:
            x = int(cx - w/2)
            y = int(cy - h/2)
            img = embryo['stack'][d-1, :, :]
            # Add metadata
            img = torch.cat([
                img.unsqueeze(0),
                img.unsqueeze(0),
                img.unsqueeze(0)
                # torch.ones(img.shape).unsqueeze(0) * w,
                # torch.ones(img.shape).unsqueeze(0) * h
            ], axis=0).float()
            # Compute padding
            p_x, p_y = int(np.floor(w * self.padding)), int(np.floor(h * self.padding))
            # Crop out bbox and resize
            _, img_h, img_w = img.shape
            y0, y1 = np.clip(y-p_y, 0, img_h), np.clip(y+h+p_y, 0, img_h)
            x0, x1 = np.clip(x-p_x, 0, img_w), np.clip(x+w+p_x, 0, img_w)
            img = img[:, y0:y1, x0:x1].unsqueeze(0)
            # Resize to standard size
            img = T.Resize((128, 128))(img)
            # Add to nodes
            nodes.append(img)
        nodes = torch.cat(nodes, axis=0)
        # Create edges
        dist = torch.Tensor(embryo['distance'])
        edge_index, edge_attr = dense_to_sparse(dist)

        # Create graph
        graph = Data(x=nodes, edge_index=edge_index, edge_attr=edge_attr, y=embryo['label']) if self.dataset.clinical_endpoint else Data(x=nodes, edge_index=edge_index, edge_attr=edge_attr)
        return graph

    def __len__(self):
        return len(self.dataset)


class EmbryoGraphDatasetMetadataOnly(Dataset):
    def __init__(self, dataset_path='data/adjacency_and_bbox_dataset.csv', plane_size=(400, 400), clinical_endpoint=None, 
                 padding=0.1, num_folds=0, fold=0, invert_selection=False):
        super().__init__()
        self.dataset = EmbryoDatasetWithAdjacency(
            dataset_path=dataset_path, 
            plane_size=plane_size, 
            clinical_endpoint=clinical_endpoint,
            load_stacks=False, 
            num_folds=num_folds, 
            fold=fold,
            invert_selection=invert_selection
        )
        self.padding = padding

    def __getitem__(self, idx):
        embryo = self.dataset[idx]
        # Create nodes
        nodes = []
        w_max = max([w for _, _, w, _, _ in embryo['bboxes']])
        h_max = max([h for _, _, _, h, _ in embryo['bboxes']])
        for i, box in enumerate(embryo['bboxes']):
            cx, cy, w, h, d = box
            degree = float(sum(embryo['adjacency'][i]))
            data = torch.tensor([[w / w_max, h / h_max, degree / 4]])
            # Add to nodes
            nodes.append(data)
        nodes = torch.cat(nodes, axis=0)
        # Create edges
        dist = torch.Tensor(embryo['distance'])
        edge_index, edge_attr = dense_to_sparse(dist)

        # Create graph
        graph = Data(x=nodes, edge_index=edge_index, edge_attr=edge_attr, y=embryo['label']) if self.dataset.clinical_endpoint else Data(x=nodes, edge_index=edge_index, edge_attr=edge_attr)
        return graph

    def __len__(self):
        return len(self.dataset)


class EmbryoGraphDatasetNoNodeFeatures(Dataset):
    def __init__(self, dataset_path='data/adjacency_and_bbox_dataset.csv', plane_size=(400, 400), clinical_endpoint=None, 
                 padding=0.1, num_folds=0, fold=0, invert_selection=False, keep_one_fold_for_testing=False):
        super().__init__()
        self.dataset = EmbryoDatasetWithAdjacency(
            dataset_path=dataset_path, 
            plane_size=plane_size, 
            clinical_endpoint=clinical_endpoint,
            load_stacks=False, 
            num_folds=num_folds, 
            fold=fold,
            invert_selection=invert_selection,
            keep_one_fold_for_testing=keep_one_fold_for_testing
        )
        self.padding = padding

    def __getitem__(self, idx):
        embryo = self.dataset[idx]
        # Create nodes
        nodes = []
        w_max = max([w for _, _, w, _, _ in embryo['bboxes']])
        h_max = max([h for _, _, _, h, _ in embryo['bboxes']])
        for i, box in enumerate(embryo['bboxes']):
            cx, cy, w, h, d = box
            degree = float(sum(embryo['adjacency'][i]))
            # Use node degree as feat
            data = torch.tensor([[degree]])
            # Add to nodes
            nodes.append(data)
        nodes = torch.cat(nodes, axis=0)
        # Create edges
        dist = torch.Tensor(embryo['distance'])
        edge_index, edge_attr = dense_to_sparse(dist)

        # Create graph
        graph = Data(x=nodes, edge_index=edge_index, edge_attr=edge_attr, y=embryo['label']) if self.dataset.clinical_endpoint else Data(x=nodes, edge_index=edge_index, edge_attr=edge_attr)
        return graph

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    d = EmbryoGraphDataset()
    e = d[0]
    fig, ax = plt.subplots()
    ax.imshow(e.x[5, 0])
    plt.savefig('asdf.jpg')
    print(e.edge_index)
    print(e.edge_attr)

    d = EmbryoGraphDatasetNoNodeFeatures()
    print(d[0].x)