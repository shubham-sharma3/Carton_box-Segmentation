import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import UNET
from utils.get_device import get_device
from utils.data_loader import get_loaders
from utils.eval_metric import check_accuracy
from utils.augmentation_policy import train_augmentations, val_augmentations
from config.params import *

# Select device for training
DEVICE = get_device()

def train_fn(loader, model, optimizer, loss_fn, scaler=None, writer=None, epoch=0):
    """
    Train the model for one epoch.

    Args:
        loader (DataLoader): Training data loader.
        model (nn.Module): The model being trained.
        optimizer (Optimizer): Optimizer for model parameters.
        loss_fn (Loss): Loss function.
        scaler (GradScaler, optional): For mixed-precision training.
        writer (SummaryWriter, optional): For logging metrics to TensorBoard.
        epoch (int): Current epoch number.

    Returns:
        float: Average loss for the epoch.
    """
    loop = tqdm(loader)
    running_loss = 0.0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # Forward pass
        if DEVICE == "cuda":
            with torch.amp.autocast('cuda'):
                predictions = model(data)
                loss = loss_fn(predictions, targets)
        else:
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Logging
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(loader)
    if writer:
        writer.add_scalar("Training Loss", avg_loss, epoch)

    return avg_loss

def export_to_onnx(model, input_shape=(1, 3, 512, 512), filename="best_model.onnx"):
    """
    Export the trained model to ONNX format.

    Args:
        model (nn.Module): Model to export.
        input_shape (tuple): Input tensor shape.
        filename (str): Output ONNX file path.
    """
    model.eval()
    dummy_input = torch.randn(input_shape).to(DEVICE)
    torch.onnx.export(
        model,
        dummy_input,
        filename,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=12
    )
    print(f"=> Exported model to {filename}")

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """
    Saves a model checkpoint to the specified file.
    
    Args:
        state (dict): Dictionary containing model state_dict and other metadata.
        filename (str): Path to save the checkpoint file.
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)

def main():
    # Create required output directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(os.path.join(OUTPUT_DIR, "runs"))

    # Load augmentations
    train_transform = train_augmentations()
    val_transform = val_augmentations()

    # Initialize model, loss, optimizer
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Load data
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR, TRAIN_MASK_DIR,
        VAL_IMG_DIR, VAL_MASK_DIR,
        BATCH_SIZE, train_transform,
        val_transform, NUM_WORKERS
    )

    best_dice = 0.0
    best_model_path = os.path.join(OUTPUT_DIR, "new_best_model.onnx")

    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, writer, epoch)

        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            print("Evaluating on validation set...")
            metrics = check_accuracy(val_loader, model)
            accuracy = metrics["accuracy"]
            dice_score = metrics["dice_score"]

            writer.add_scalar("Validation Accuracy", accuracy, epoch)
            writer.add_scalar("Validation Dice Score", dice_score, epoch)
            writer.add_scalar("Train Loss", train_loss, epoch)

            # Save checkpoint
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "dice_score": dice_score,
            }
            save_checkpoint(checkpoint, filename=os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch + 1}.pth.tar"))

            # Export best model
            if dice_score > best_dice:
                best_dice = dice_score
                print(f"New best Dice score: {best_dice:.4f}. Exporting to ONNX...")
                export_to_onnx(model, filename=best_model_path)

    writer.close()

if __name__ == "__main__":
    main()
