# Libs
import numpy as np


# Funcs >>>
def calculate_normalization_params(images, batch_size=32):
    if images.ndim == 3:
        images = np.expand_dims(images, axis=0)

    mean = np.zeros(images.shape[-1], dtype=np.float64)
    squared_mean = np.zeros(images.shape[-1], dtype=np.float64)
    total_pixels = 0

    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size].astype(np.float64)
        batch_pixels = np.prod(batch.shape[:-1])  # exclude channel dimension
        total_pixels += batch_pixels

        batch_mean = np.mean(batch, axis=(0, 1, 2))
        batch_squared_mean = np.mean(np.square(batch), axis=(0, 1, 2))

        mean += batch_mean * batch_pixels
        squared_mean += batch_squared_mean * batch_pixels

    mean /= total_pixels
    squared_mean /= total_pixels

    epsilon = 1e-10
    variance = np.maximum(squared_mean - np.square(mean), epsilon)
    std = np.sqrt(variance)

    return {"mean": mean, "std": std}


def normalize_image(image, norm_params):
    """
    Normalize a single grayscale image using the provided parameters.
    The output will be in the range [0, 1].

    :param image: numpy array of shape (height, width)
    :param norm_params: dict with mean and std
    :return: normalized image
    """
    # Z-score normalization
    normalized = (image - norm_params["mean"]) / norm_params["std"]

    # Scale to [0, 1] range using a sigmoid-like function
    normalized = 1 / (1 + np.exp(-normalized))

    return normalized


def compute_class_weights_one_hot(y: np.ndarray, weighting: str = "linear"):
    """Computes normalized class weights for multi-label binary one-hot encoded labels.

    This computes the inverse frequency of each class in the provided
    multi-label binary one-hot encoded labels, applies the specified weighting scheme,
    and returns the normalized class weights.

    Parameters:
    y (np.ndarray): Multi-label binary one-hot encoded labels.
    weighting (str): The weighting scheme to apply. Options are 'square', 'sqrt', '1p5_Power', '1p2_Power', 'cube', 'harmonic', 'log', and 'linear'.

    Returns:
    np.ndarray: The normalized class weights.

    Intended for computing loss weighting to handle class imbalance in multi-label classification.
    """
    # Count the number of samples in each class
    class_sample_counts = y.sum(axis=0)

    # Compute the inverse of each class count
    class_weights = 1.0 / class_sample_counts.astype(np.float32)

    # Apply the specified weighting scheme
    if weighting == "square":
        class_weights = np.square(class_weights)
    elif weighting == "none":
        class_weights = np.ones_like(class_sample_counts)
    elif weighting == "sqrt":
        class_weights = np.sqrt(class_weights)
    elif weighting == "cube":
        class_weights = np.power(class_weights, 3)
    elif weighting == "1p5_Power":
        class_weights = np.power(class_weights, 1.5)
    elif weighting == "1p2_Power":
        class_weights = np.power(class_weights, 1.2)
    elif weighting == "log":
        class_weights = np.log(class_weights)
    elif weighting == "harmonic":
        class_weights = 1 / class_weights
    elif weighting != "linear":
        raise ValueError(f"Unknown weighting scheme '{weighting}'")

    # Normalize the class weights so that they sum to 1
    class_weights_normalized = class_weights / np.sum(class_weights)

    # Return the normalized class weights
    return class_weights_normalized
