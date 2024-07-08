import numpy as np

from skimage import exposure


def normalise_stack(plane_imgs, plane_size):
    """ Apply normalisation across the whole stack by combining planes
        into single image and applying normalisation to that. """
    # Combine the images
    combined_img = np.zeros((plane_size[0] * len(plane_imgs), plane_size[1])    )
    for i in range(len(plane_imgs)):
        combined_img[i * plane_size[0]:(i + 1) * plane_size[0], 0:plane_size[1]] = plane_imgs[i]
    # Normalise combined image
    combined_img = exposure.equalize_adapthist(
        combined_img, clip_limit=0.01)
    # Unpack the combined image into planes
    return [combined_img[i * plane_size[0]:(i + 1) * plane_size[0], 0:plane_size[1]]
            for i in range(len(plane_imgs))]
