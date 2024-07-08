import os
import glob
import json
import torch
import numpy as np
import albumentations as A
import pickle

from torch.utils.data import Dataset
from skimage import io
from skimage.transform import resize
from skimage.draw import disk, ellipse, polygon

from .stack_utils import normalise_stack

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class T4SegmentationDataset2D(Dataset):

    def __init__(self, data_dir='/datasets/train/stacks/static', label_dir='/datasets/train/seg/static', plane_count=11,
                 plane_size=(400, 400), use_normalisation=True, use_augmentations=False, augmentations=None, num_folds=1, 
                 fold=0, invert_selection=False, preshuffle=True, verbose=False):
        self.use_augmentations = use_augmentations
        self.use_normalisation = use_normalisation
        self.verbose = verbose
        self.plane_count = plane_count
        self.plane_size = plane_size
        if augmentations is None:
            # Default to these
            self.augmentations = [
                A.RandomRotate90(p=0.3),
                A.Flip(p=0.3),
                A.GaussNoise(var_limit=0.005, p=0.3),
                A.Cutout(num_holes=30, max_h_size=20, max_w_size=20, p=0.3)
            ]
        else:
            self.augmentations = augmentations

        clinic_dirs = glob.glob(os.path.join(data_dir, '*'))
        clinic_label_dirs = glob.glob(os.path.join(label_dir, '*'))
        # Find clinics with both stack data AND labels
        clinics_intersection = [os.path.basename(clinic_dir) for clinic_dir in clinic_dirs if os.path.basename(
            clinic_dir) in list(map(os.path.basename, clinic_label_dirs))]
        embryo_dirs = [(os.path.basename(clinic_dir), os.path.basename(embryo_dir)) for clinic_dir in clinics_intersection for embryo_dir in glob.glob(
            os.path.join(data_dir, clinic_dir, '*'))]
        embryo_label_dirs = [(os.path.basename(clinic_dir), os.path.splitext(os.path.basename(embryo_dir))[0]) for clinic_dir in clinics_intersection for
                             embryo_dir in glob.glob(os.path.join(label_dir, clinic_dir, '*'))]
        # Find embryos that have both stack data AND labels
        embryo_intersection = list(
            set(embryo_dirs).intersection(embryo_label_dirs)
        )

        self.data = [
            (
                dict(sorted({
                    int(os.path.splitext(os.path.basename(plane_path))[0][1:]): plane_path
                    for plane_path in glob.glob(os.path.join(data_dir, clinic, embryo, '*.jpg'))
                }.items())),
                os.path.join(label_dir, clinic, f'{embryo}.json')
            )
            for clinic, embryo in embryo_intersection
        ]
        self.data = sorted(self.data, key=lambda x: x[1])

        # Preshuffling
        # Now listen up punk. This is really cancerous but I'm tired and I cba to come up with something
        # elegant. So basically what we gonna do is we generate a permutation and save it. Then we shuffle
        # the data according to said permutation. If the length of the data list and the cached permutation
        # don't match, we regenerate it :) Honestly this is Marica using a .mat file as an iterator vibes.
        self.data_permutation = [i for i in range(len(self.data))]
        if preshuffle:
            # Check if the cached version exists
            try:
                with open('dataset_permutation.pkl', 'rb') as f:
                    self.data_permutation = pickle.load(f)
                assert len(self.data_permutation) == len(self.data)
            except Exception as e:
                # If not, just regenerate the permutation
                if self.verbose:
                    print('Could not load permutation file. Reshuffling data.')
                self.data_permutation = [i for i in range(len(self.data))]
                np.random.shuffle(self.data_permutation)
                with open('dataset_permutation.pkl', 'wb+') as f:
                    pickle.dump(self.data_permutation, f)
        # Finally, shuffle the data according to the permutation indices
        self.data = [self.data[i] for i in self.data_permutation]
                
        # Remove the correct fold (if applicable)
        if num_folds > 1:
            assert fold < num_folds
            fold_size = int(np.floor(len(self.data) / num_folds))
            if invert_selection:
                # Only select the fold
                selected_data = self.data[
                    fold*fold_size:(fold+1)*fold_size
                ]
            else:
                # Select everything EXCEPT for the fold
                selected_data = self.data[0:fold*fold_size]
                selected_data.extend(self.data[(fold+1)*fold_size:-1])
            self.data = selected_data

        # Generate mask for hiding the well outline
        rr, cc = disk((plane_size[0]/2, plane_size[1]/2), plane_size[0]/2)
        self.circle_mask = np.zeros(plane_size)
        self.circle_mask[rr, cc] = 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        stack, label_path = self.data[idx]
        # Load planes
        plane_imgs = [io.imread(plane_path, as_gray=True)
                      for _, plane_path in stack.items()]
        plane_imgs = [img[50:-50, 50:-50] for img in plane_imgs]
        plane_imgs = [resize(img, self.plane_size) for img in plane_imgs]
        # Mask out the edges
        plane_imgs = [img * self.circle_mask for img in plane_imgs]
        # Load label
        masks, depths = self.__load_label__(label_path)
        # Normalise
        if self.use_normalisation:
            plane_imgs = normalise_stack(plane_imgs, self.plane_size)
        # Combine all the masks into a single image so we can run augmentation on it
        combined_img = np.zeros(
            (self.plane_size[0], self.plane_size[1], len(plane_imgs))
        )
        for i in range(len(plane_imgs)):
            combined_img[:, :, i] = plane_imgs[i]
        # Generate metadata
        boxes = []
        for mask in masks:
            pos = np.where(mask)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        # Augment
        if self.use_augmentations:
            combined_img, masks, boxes, depths = self.__augment__(combined_img, masks, boxes, depths)
        boxes = torch.as_tensor(boxes).float()
        label = {
            'boxes': boxes,
            'labels': torch.ones(len(masks), dtype=torch.int64),
            'masks': torch.as_tensor(np.array(masks), dtype=torch.uint8),
            'depths': torch.as_tensor(depths).float(),
            'image_id': torch.tensor(0),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros((len(masks),)).int()
        }
        return combined_img, label

    def __augment__(self, combined_img, masks, boxes, depths):
        # Choose augmentations randomly
        aug = A.Compose(
            self.augmentations,
            bbox_params=A.BboxParams(
                format='pascal_voc', 
                min_area=10, 
                min_visibility=0.1,
                label_fields=['depths']
            )
        )
        # Convert mask to int so that albumentations works on it
        masks = [x.astype(int) for x in masks]
        augmented = aug(image=combined_img, masks=masks, bboxes=boxes, depths=depths)
        return augmented['image'], augmented['masks'], augmented['bboxes'], augmented['depths']

    def __load_label__(self, label_path):
        cells = []
        # Parse out the JSON
        with open(label_path, 'r') as f:
            label_json = json.loads(f.read())
            imgs = label_json['_via_img_metadata']
            for img, attrs in imgs.items():
                depth = int(os.path.splitext(img)[0][1:])
                regions = attrs['regions']
                for region in regions:
                    shape = region['shape_attributes']
                    cells.append((shape, depth))
        # Calculate factors for scaling labels
        scale_factor_x = self.plane_size[0] / \
            (int(attrs['size']['width']) - 100)
        scale_factor_y = self.plane_size[1] / \
            (int(attrs['size']['height']) - 100)
        # Generate the cells
        masks = []
        depths = []
        for shape, depth in cells:
            # Generate the masks
            shape_type = shape['name']
            if shape_type == 'ellipse':
                mask = self.__generate_ellipse_mask__(
                    cx=(shape['cx']-50) * scale_factor_x,
                    cy=(shape['cy']-50) * scale_factor_y,
                    rx=shape['rx'] * scale_factor_x,
                    ry=shape['ry'] * scale_factor_y,
                    theta=shape['theta']
                )
                masks.append(mask)
                depths.append(self.__normalise_focal_depth__(depth))
            elif shape_type == 'polygon':
                mask = self.__generate_polygon_mask__(
                    xs=[(x-50)*scale_factor_x for x in shape['all_points_x']],
                    ys=[(y-50)*scale_factor_y for y in shape['all_points_y']]
                )
                masks.append(mask)
                depths.append(self.__normalise_focal_depth__(depth))
            else:
                print(f'Unknown shape type: {shape_type}')
        return masks, depths

    def __generate_ellipse_mask__(self, cx, cy, rx, ry, theta):
        mask = np.zeros((self.plane_size[0], self.plane_size[1]))
        rr, cc = ellipse(cx, cy, rx, ry, shape=self.plane_size, rotation=theta)
        mask[cc, rr] = 1
        return mask

    def __generate_polygon_mask__(self, xs, ys):
        mask = np.zeros((self.plane_size[0], self.plane_size[1]))
        rr, cc = polygon(xs, ys, shape=self.plane_size)
        mask[cc, rr] = 1
        return mask

    def __normalise_focal_depth__(self, x):
        return (x + 75) / 150


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    ncolors = 256
    color_array = plt.get_cmap('jet')(range(ncolors))

    # change alpha values
    color_array[:, -1] = np.linspace(0, 1, ncolors)

    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(
        name='jet_alpha', colors=color_array)

    # register this new colormap with matplotlib
    plt.register_cmap(cmap=map_object)
    dataset = T4SegmentationDataset2D(use_augmentations=True)
    # for i in range(0, len(dataset), 2):
    i = 0
    img, label = dataset[i]
    # print(img.shape, label)
    fig, ax = plt.subplots()
    ax.imshow(img[:, :, 5])
    for i, mask in enumerate(label['masks']):
        # Show OG image and mask
        ax.imshow(
            mask, cmap='jet_alpha', alpha=0.3)
    
    masks = label['masks']
    # print(masks)
    plt.savefig('asdf.jpg')