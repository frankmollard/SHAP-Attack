"""
data_utils.py
-------------
Image loading and preprocessing helpers for the cat/dog subset of the
Animals and Humans dataset (Kaggle: frankmollard/animals-and-humans).

Used in:
  Mollard, Becker, Röhrbein (2024).
  "Adversarial Evasion Attacks on Computer Vision using SHAP Values."
  DOI: 10.13140/RG.2.2.28762.40647
"""

import numpy as np
from PIL import Image


# Default dataset paths (Kaggle environment)
DATA_PATH      = "/kaggle/input/datasets/frankmollard/animals-and-humans"
IMAGE_FILE     = "imgs_dcwhBig.npy"
CATEGORY_FILE  = "categories_dcwhBig.npy"
IMAGE_SHAPE    = (64, 64, 3)


def load_cat_dog(data_path: str = DATA_PATH,
                 image_file: str = IMAGE_FILE,
                 category_file: str = CATEGORY_FILE):
    """
    Load and preprocess the cat/dog subset of the Animals and Humans dataset.

    Images are normalised from uint8 [0, 255] to float32 [0, 1] and
    returned in flattened form (N, 12288).

    Parameters
    ----------
    data_path     : str — directory containing the .npy files
    image_file    : str — filename of the image array
    category_file : str — filename of the category label array

    Returns
    -------
    images : np.ndarray, shape (N, 12288), float32 in [0, 1]
    labels : np.ndarray, shape (N,),      int    (1=cat, 0=dog)
    """
    categories = np.load(f"{data_path}/{category_file}")
    all_images  = np.load(f"{data_path}/{image_file}")

    mask = np.array([c in {"cat", "dog"} for c in categories])
    images     = all_images[mask] / 255.0
    labels     = np.array([1 if c == "cat" else 0
                           for c in categories[mask]], dtype=np.int32)

    print(f"Loaded {images.shape[0]} images  "
          f"({labels.sum()} cats, {(1 - labels).sum()} dogs)")
    return images, labels


def to_uint8(flat_image: np.ndarray,
             shape: tuple = IMAGE_SHAPE) -> np.ndarray:
    """
    Convert a normalised flat image back to a displayable uint8 array.

    Parameters
    ----------
    flat_image : np.ndarray — pixel values in [0, 1]
    shape      : tuple      — target (H, W, C), default (64, 64, 3)

    Returns
    -------
    np.ndarray, shape (H, W, C), dtype uint8
    """
    return (flat_image * 255).reshape(shape).astype(np.uint8)


def show_image(flat_image: np.ndarray, shape: tuple = IMAGE_SHAPE):
    """Display a single normalised flat image using PIL."""
    Image.fromarray(to_uint8(flat_image, shape)).show()
