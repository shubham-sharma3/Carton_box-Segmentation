import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """Custom Dataset for loading images and their corresponding segmentation masks."""

    def __init__(self, image_dir: str, mask_dir: str, transform=None):
        """
        Initialize the dataset.

        Args:
            image_dir (str): Directory containing input images.
            mask_dir (str): Directory containing segmentation masks.
            transform (callable, optional): Transformations to apply to images and masks.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return len(self.images)

    def __getitem__(self, index: int) -> tuple:
        """
        Retrieve an image and its corresponding mask by index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, mask) where image is the input image and mask is the segmentation mask.
        """
        # Construct file paths
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", ".png"))

        # Load and preprocess image and mask
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0  # Normalize mask to binary [0, 1]

        # Apply transformations if provided
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask