import torch
from utils.get_device import get_device


def check_accuracy(loader, model) -> dict:
    """
    Evaluate model performance on a dataset using pixel-wise accuracy and Dice score.

    Args:
        loader (DataLoader): DataLoader for the dataset to evaluate.
        model (torch.nn.Module): Model to evaluate.

    Returns:
        dict: Dictionary containing 'accuracy' (pixel-wise accuracy in %) and 'dice_score'.
    """
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    device = get_device()
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            # Calculate pixel-wise accuracy
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

            # Calculate Dice score
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    # Compute average metrics
    accuracy = num_correct / num_pixels * 100 if num_pixels > 0 else 0.0
    dice_avg = dice_score / len(loader) if len(loader) > 0 else 0.0

    # Print evaluation results
    print(f"Got {num_correct}/{num_pixels} with acc {accuracy:.2f}%")
    print(f"Dice score: {dice_avg:.4f}")

    model.train()
    return {"accuracy": accuracy.item(), "dice_score": dice_avg.item()}