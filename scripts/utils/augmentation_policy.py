import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

def train_augmentations():
    """
    Applies augmentation policy for semantic image segmentation as described in the KITTI dataset assignment.
    
    Args:
        image (numpy.ndarray): Input image (H, W, C)
        mask (numpy.ndarray): Input segmentation mask (H, W)
    
    Returns:
        tuple: Augmented image and mask
    """
    # Define augmentation pipeline
    transform = A.Compose([
        #Resize
        A.Resize(height=512, 
                 width=512,
                 interpolation=cv2.INTER_AREA ),

        # Geometric transformations
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,),
        
        # Noise and blur
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        ToTensorV2(),
        
    ])
    
    # Apply augmentations
    return transform

def val_augmentations():
    """
    Applies augmentation policy for semantic image segmentation as described in the KITTI dataset assignment.
    
    Args:
        image (numpy.ndarray): Input image (H, W, C)
        mask (numpy.ndarray): Input segmentation mask (H, W)
    
    Returns:
        tuple: Augmented image and mask
    """
    # Define augmentation pipeline
    transform = A.Compose([
        #Resize
        A.Resize(height=512, 
                 width=512,
                 interpolation=cv2.INTER_AREA ),

        # Normalize
        A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,),
        
        ToTensorV2(),
        
    ])
    
    return transform