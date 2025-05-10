from torch.utils.data import DataLoader
from dataset import SegmentationDataset


def get_loaders(
    train_dir: str,
    train_maskdir: str,
    val_dir: str,
    val_maskdir: str,
    batch_size: int,
    train_transform,
    val_transform,
    num_workers = 4,
):
    """
    Create data loaders for training and validation datasets.

    Args:
        train_dir (str): Directory containing training images.
        train_maskdir (str): Directory containing training masks.
        val_dir (str): Directory containing validation images.
        val_maskdir (str): Directory containing validation masks.
        batch_size (int): Number of samples per batch.
        train_transform (callable): Transformations for training data.
        val_transform (callable): Transformations for validation data.
        num_workers (int, optional): Number of subprocesses for data loading. Defaults to 4.

    Returns:
        tuple: (train_loader, val_loader) containing DataLoader objects for training and validation.
    """
    # Initialize training dataset
    train_ds = SegmentationDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    # Create training data loader
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

    # Initialize validation dataset
    val_ds = SegmentationDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    # Create validation data loader
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    return train_loader, val_loader