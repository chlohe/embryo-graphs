from torch.utils.data import Dataset
from . import normalise_stack
import pandas as pd
import torch
import numpy as np
import json
import os.path as osp
import glob

from skimage import io
from skimage.transform import resize
from skimage.draw import disk
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class EmbryoDatasetWithAdjacency(Dataset):
    def __init__(self, dataset_path='data/adjacency_and_bbox_dataset.csv', plane_size=(400, 400), 
                 clinical_endpoint=None, load_stacks=True, num_folds=0, fold=0, keep_one_fold_for_testing=False,
                 invert_selection=False):
        self.data = pd.read_csv(dataset_path)
        # Parse out the object data
        self.data['adjacency'] = self.data['adjacency'].apply(json.loads)
        self.data['distance'] = self.data['distance'].apply(json.loads)
        self.data['label'] = self.data['label'].apply(json.loads)
        # Add the stack to the dataset
        self.data['stack'] = self.data['t8_path'].apply(self.__get_stack__)
        # Only provide distances for edges
        self.data['distance'] = self.data.apply(lambda x: np.multiply(x['distance'], x['adjacency']), axis=1)

        if clinical_endpoint is not None:
            self.data = self.data[self.data[clinical_endpoint].notna()]

        # Shuffle em!
        self.data = self.data.sample(frac=1, random_state=1).reset_index()

        self.plane_size = plane_size
        self.clinical_endpoint = clinical_endpoint
        self.load_stacks = load_stacks

        # Remove the correct fold (if applicable)
        if num_folds > 1:
            assert fold < num_folds
            fold_size = int(np.floor(len(self.data) / num_folds))
            if invert_selection:
                # Only select the fold
                selected_data = self.data[
                    fold*fold_size:(fold+1)*fold_size
                ]
                if keep_one_fold_for_testing and fold == num_folds-1:
                    raise Exception('Trying to access fold that is reserved for testing!')
            else:
                # Select everything EXCEPT for the fold
                selected_data = self.data.iloc[list(range(0, fold*fold_size)) + list(range((fold+1)*fold_size, len(self.data) - (fold_size if keep_one_fold_for_testing else 0)))]
            self.data = selected_data

        # Generate mask for hiding the well outline
        rr, cc = disk((plane_size[0]/2, plane_size[1]/2), plane_size[0]/2)
        self.circle_mask = np.zeros(plane_size)
        self.circle_mask[rr, cc] = 1

    def __get_stack__(self, path):
        return dict(sorted({
            int(osp.splitext(osp.basename(plane_path))[0][1:]): plane_path
            for plane_path in glob.glob(osp.join(path, '*.jpg'))
        }.items()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        result = {
            'adjacency': row['adjacency'],
            'distance': row['distance'],
            'bboxes': row['label']
        }
        if self.clinical_endpoint is not None:
            result['label'] = row[self.clinical_endpoint]
        stack = row['stack']
        if self.load_stacks:
            # Load planes
            plane_imgs = [io.imread(plane_path, as_gray=True)
                          for _, plane_path in stack.items()]
            plane_imgs = [img[50:-50, 50:-50] for img in plane_imgs]
            plane_imgs = [resize(img, self.plane_size) for img in plane_imgs]
            # Mask out the edges
            plane_imgs = [img * self.circle_mask for img in plane_imgs]
            # Normalise
            plane_imgs = normalise_stack(plane_imgs, self.plane_size)
            # Combine all the masks into a single image
            combined_img = np.zeros(
                (self.plane_size[0], self.plane_size[1], len(plane_imgs))
            )
            for i in range(len(plane_imgs)):
                combined_img[:, :, i] = plane_imgs[i]
            result['stack'] = torch.from_numpy(combined_img).permute(2, 0, 1)
        return result


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    d = EmbryoDatasetWithAdjacency(clinical_endpoint='simplified_grade')
    fig, ax = plt.subplots()
    e = d[1]
    ax.imshow(e['stack'][5, :, :])
    colors = ['red', 'yellow', 'green', 'blue',
              'purple', 'pink', 'white', 'black']
    for i, cell in enumerate(e['bboxes']):
        cx, cy, w, h, d = cell
        x = int(cx - w/2)
        y = int(cy - h/2)
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor=colors[i], facecolor='none')
        ax.add_patch(rect)
    plt.savefig('asdf.jpg')
