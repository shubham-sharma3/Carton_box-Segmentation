import argparse
import os
import cv2
import numpy as np
import torch
import torch.optim as optim
import torchvision
import onnxruntime as ort

from model import UNET
from utils.get_device import get_device
from utils.augmentation_policy import val_augmentations
from config.params import *


def main():
    """Perform inference on a single image using either a checkpoint or ONNX model."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Save prediction examples from validation set.")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["onnx", "checkpoint"],
        default="checkpoint",
        help="Model type to use: 'onnx' or 'checkpoint'",
    )
    args = parser.parse_args()

    # Set up device
    device = get_device()

    # Load and preprocess image
    test_img = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    if test_img is None:
        raise FileNotFoundError(f"Could not load image at {IMAGE_PATH}")
    image_shape = test_img.shape
    print(f"Image shape: {image_shape}")

    # Apply validation augmentations
    transform = val_augmentations()
    augmented = transform(image=test_img)
    image = augmented["image"]
    image = image.unsqueeze(0).to(device)  # Shape: [1, C, H, W]

    # Initialize output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.model_type == "onnx":
        # Load ONNX model
        if not os.path.exists(ONNX_MODEL_PATH):
            raise FileNotFoundError(f"ONNX model not found at {ONNX_MODEL_PATH}")
        
        # Select execution provider based on device
        providers = (
            ["CUDAExecutionProvider"] if device == "cuda" else
            ["CoreMLExecutionProvider"] if device == "mps" else
            ["CPUExecutionProvider"]
        )
        session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)
        
        # Run inference
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        logits = session.run([output_name], {input_name: image.cpu().numpy()})[0]
        logits = torch.from_numpy(logits).to(device)
        print(f"Logits shape: {logits.shape}")

        # Resize logits to match input size if necessary
        target_size = (image.shape[2], image.shape[3])  # (H, W)
        if logits.shape[2:] != target_size:
            logits = torch.nn.functional.interpolate(
                logits, size=target_size, mode="bilinear", align_corners=False
            )
        
        # Generate binary predictions
        preds = (torch.sigmoid(logits) > 0.5).float()[0].cpu().numpy()
        preds = (preds * 255).astype(np.uint8)
        
        # Save prediction as image
        cv2.imwrite(f"{OUTPUT_DIR}/pred_image1.png", preds)
    else:
        # Load PyTorch model from checkpoint
        model = UNET(in_channels=3, out_channels=1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        checkpoint = torch.load(
            f"{CHECKPOINT_DIR}/checkpoint_epoch_10.pth.tar", weights_only=True
        )
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint["epoch"]
        dice_score = checkpoint["dice_score"]

        # Run inference
        model.eval()
        with torch.no_grad():
            preds = torch.sigmoid(model(image))
            preds = (preds > 0.5).float()

            # Save predictions
            torchvision.utils.save_image(preds, f"{OUTPUT_DIR}/pred_image.png")
            pred_img = cv2.imread(f"{OUTPUT_DIR}/pred_image.png")
            up_pred_img = cv2.resize(
                pred_img,
                (image_shape[1], image_shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )
            print(f"Upsampled prediction shape: {up_pred_img.shape}")
            cv2.imwrite(f"{OUTPUT_DIR}/up_pred_image.png", up_pred_img)


if __name__ == "__main__":
    main()