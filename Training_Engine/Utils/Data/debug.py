# Libs >>>
import math
import torch
import random
from typing import List, Any, Tuple, Union
from torch.utils.data import DataLoader


# Func >>>
def retrieve_samples(
    data_loader: DataLoader,
    num_samples: int = 50,
    selection_method: str = "random",
    seed: int = 42,
    return_labels: bool = False,
) -> Union[List[Any], List[Tuple[Any, Any]]]:
    """
    Retrieves a list of images from a DataLoader without iterating through it.

    Parameters:
        data_loader (DataLoader): The DataLoader instance from which to retrieve samples.
        num_samples (int, optional): The number of samples to retrieve. Defaults to 50.
        selection_method (str, optional): The method to select samples.
            Can be 'random', 'from_start', or 'from_end'. Defaults to 'random'.
        seed (int, optional): The seed for random number generator for reproducibility.
            Only used when selection_method is 'random'. Defaults to 42.
        return_labels (bool, optional): Whether to return labels along with images.
            Defaults to False.

    Returns:
        List[Any] or List[Tuple[Any, Any]]:
            - If return_labels is False: A list of image tensors.
            - If return_labels is True: A list of tuples, each containing an image tensor and its corresponding label.

    Raises:
        IndexError: If the dataset does not have enough samples.
        ValueError: If an invalid selection_method is provided.
    """
    dataset = data_loader.dataset
    total_samples = len(dataset)

    if num_samples > total_samples:
        raise IndexError(
            f"Cannot retrieve {num_samples} samples from a dataset of size {total_samples}"
        )

    if selection_method == "random":
        random.seed(seed)
        indices = random.sample(range(total_samples), num_samples)
    elif selection_method == "from_start":
        indices = range(num_samples)
    elif selection_method == "from_end":
        indices = range(total_samples - num_samples, total_samples)
    else:
        raise ValueError(
            "Invalid selection_method. Choose 'random', 'from_start', or 'from_end'."
        )

    samples = [dataset[i] for i in indices]

    if isinstance(samples[0], (tuple, list)):
        if return_labels:
            return samples
        else:
            return [sample[0] for sample in samples]
    else:
        return samples


def make_grid(
    tensor,
    nrow=8,
    padding=2,
    normalize=False,
    value_range=None,
    scale_each=False,
    pad_value=0,
    format="CHW",
):
    """
    Arrange a tensor of images into a grid layout.

    Parameters:
        tensor (Tensor): 4D tensor of shape (B, C, H, W) or 3D tensor of shape (C, H, W).
        nrow (int): Number of images displayed in each row of the grid.
        padding (int): Padding between images.
        normalize (bool): Whether to normalize tensor values to the range [0, 1].
        value_range (tuple): Range (min, max) where the tensor will be normalized if normalize is True.
        scale_each (bool): Whether to normalize each image in the batch individually.
        pad_value (float): Value used to pad the grid image.
        format (str): Output format of the grid, 'CHW' or 'HWC'.

    Returns:
        Tensor: Grid image tensor in the specified format.

    Example:
        grid = make_grid(tensor, nrow=4, padding=2, normalize=True, format='HWC')
    """
    # Check tensor dimensions and adjust if necessary
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() != 4:
        raise ValueError("Input tensor should be 3D or 4D.")

    B, C, H, W = tensor.size()
    nimg = B
    nrows = nrow
    ncols = int(math.ceil(float(nimg) / float(nrows)))

    # Normalize the tensor if required
    if normalize:
        if scale_each:
            tensors = [(x - x.min()) / (x.max() - x.min()) for x in tensor]
            tensor = torch.stack(tensors)
        else:
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        if value_range is not None:
            tensor = tensor * (value_range[1] - value_range[0]) + value_range[0]

    # Calculate padding sizes
    pad_h = padding * (ncols - 1)
    pad_w = padding * (nrows - 1)

    # Create a grid tensor filled with pad_value
    grid_height = H * nrows + pad_h
    grid_width = W * ncols + pad_w
    grid = torch.full(
        (C, grid_height, grid_width),
        pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )

    # Place each image into the grid
    for i in range(nimg):
        row = i // nrows
        col = i % nrows
        top = row * (H + padding)
        left = col * (W + padding)
        grid[:, top : top + H, left : left + W] = tensor[i]

    # Convert to desired format
    if format == "HWC":
        grid = grid.permute(1, 2, 0)
    elif format != "CHW":
        raise ValueError("Invalid format specified. Choose 'CHW' or 'HWC'.")

    return grid
