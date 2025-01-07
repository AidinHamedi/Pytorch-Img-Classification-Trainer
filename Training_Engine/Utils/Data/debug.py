# Libs >>>
import random
from typing import Union, Tuple, Any
from torch.utils.data import DataLoader

# Func >>>
def retrieve_samples(
    data_loader: DataLoader,
    num_samples: int = 50,
    selection_method: str = 'random',
    seed: int = 42
) -> Union[Tuple[Any, ...], Any]:
    """
    Retrieves samples from a DataLoader without iterating through it.
    
    Parameters:
        data_loader (DataLoader): The DataLoader instance from which to retrieve samples.
        num_samples (int, optional): The number of samples to retrieve. Defaults to 50.
        selection_method (str, optional): The method to select samples.
            Can be 'random', 'from_start', or 'from_end'. Defaults to 'random'.
        seed (int, optional): The seed for random number generator for reproducibility.
            Only used when selection_method is 'random'. Defaults to 42.
    
    Returns:
        Union[Tuple[Any, ...], Any]: A single sample or a tuple of samples depending on the dataset.
    
    Raises:
        ValueError: If an invalid selection_method is provided.
        IndexError: If the dataset does not have enough samples.
    """
    dataset = data_loader.dataset
    
    # Check if the dataset supports __len__ and __getitem__
    if not hasattr(dataset, '__len__') or not hasattr(dataset, '__getitem__'):
        raise ValueError("Dataset does not support __len__ or __getitem__")
    
    total_samples = len(dataset)
    
    if num_samples > total_samples:
        raise IndexError(f"Cannot retrieve {num_samples} samples from a dataset of size {total_samples}")
    
    if selection_method == 'random':
        random.seed(seed)
        indices = random.sample(range(total_samples), num_samples)
        samples = [dataset[i] for i in indices]
    elif selection_method == 'from_start':
        indices = range(num_samples)
        samples = [dataset[i] for i in indices]
    elif selection_method == 'from_end':
        indices = range(total_samples - num_samples, total_samples)
        samples = [dataset[i] for i in indices]
    else:
        raise ValueError("Invalid selection_method. Choose 'random', 'from_start', or 'from_end'.")
    
    # If the dataset returns a single item, return a list. If it returns multiple, return a list of tuples.
    if isinstance(samples[0], (tuple, list)):
        return tuple(zip(*samples))
    else:
        return samples