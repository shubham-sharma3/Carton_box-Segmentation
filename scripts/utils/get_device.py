import torch


def get_device() -> torch.device:
    """
    Determine the best available device for PyTorch computations.

    Returns:
        torch.device: The selected device ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")