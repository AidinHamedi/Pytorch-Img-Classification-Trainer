# Libs >>>
import os
import cv2
import torch
import numpy as np
from PIL import Image
from rich import print
from rich.progress import Progress
from rich.progress import (
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    SpinnerColumn,
    MofNCompleteColumn,
)

# Configuration and Constants >>>
BACKEND_SUPPORT = {
    "opencv": {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"},
    "pil": {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"},
}


# Core Functions >>>
def load_image_opencv(
    img_path: str,
    img_size: tuple = None,
    color_mode: str = "rgb",
    raise_on_error: bool = False,
) -> np.ndarray:
    """
    Load an image using OpenCV with multi-format support and optional resizing.

    Args:
        img_path (str): Path to the image file.
        img_size (tuple, optional): Target image dimensions (width, height). Defaults to None.
        color_mode (str, optional): Color mode ('rgb', 'bgr', 'grayscale', 'hsv', 'lab'). Defaults to "rgb".
        raise_on_error (bool, optional): Whether to raise an error if the image fails to load. Defaults to False.

    Returns:
        np.ndarray: Loaded image as a NumPy array.

    Raises:
        ValueError: If the color mode is unsupported.
        FileNotFoundError: If the image fails to load and `raise_on_error` is True.
    """
    img = cv2.imread(img_path)
    if img is None:
        if raise_on_error:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        return None

    match color_mode:
        case "grayscale":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img_size:
                img = cv2.resize(img, img_size)
            return np.expand_dims(img, axis=-1)
        case "rgb":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        case "bgr":
            pass  # Already in BGR format
        case "hsv":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        case "lab":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        case _:
            raise ValueError(f"Unsupported color mode: {color_mode}")

    if img_size:
        img = cv2.resize(img, img_size)
    return img


def load_image_pil(
    img_path: str,
    img_size: tuple = None,
    color_mode: str = "rgb",
    raise_on_error: bool = False,
) -> np.ndarray:
    """
    Load an image using PIL with multi-format support and optional resizing.

    Args:
        img_path (str): Path to the image file.
        img_size (tuple, optional): Target image dimensions (width, height). Defaults to None.
        color_mode (str, optional): Color mode ('rgb', 'bgr', 'grayscale', 'hsv', 'lab', 'rgba'). Defaults to "rgb".
        raise_on_error (bool, optional): Whether to raise an error if the image fails to load. Defaults to False.

    Returns:
        np.ndarray: Loaded image as a NumPy array.

    Raises:
        ValueError: If the color mode is unsupported.
        FileNotFoundError: If the image fails to load and `raise_on_error` is True.
    """
    try:
        img = Image.open(img_path)
    except Exception as e:
        if raise_on_error:
            raise FileNotFoundError(f"Failed to load image: {img_path}") from e
        return None

    match color_mode:
        case "grayscale":
            img = img.convert("L")
            if img_size:
                img = img.resize(img_size)
            return np.expand_dims(np.array(img), axis=-1)
        case "rgb":
            img = img.convert("RGB")
        case "bgr":
            img = img.convert("RGB")
            img = np.array(img)[:, :, ::-1]  # Convert RGB to BGR
        case "hsv":
            img = img.convert("HSV")
        case "lab":
            img = img.convert("LAB")
        case "rgba":
            img = img.convert("RGBA")
        case _:
            raise ValueError(f"Unsupported color mode: {color_mode}")

    if img_size:
        img = img.resize(img_size)
    return np.array(img)


def is_supported_file(filename: str, backend: str = "opencv") -> bool:
    """
    Validate if the file extension is supported by the specified backend.

    Args:
        filename (str): Name of the file.
        backend (str, optional): Backend to check support for ('opencv', 'pil'). Defaults to "opencv".

    Returns:
        bool: True if the file extension is supported, False otherwise.
    """
    ext = os.path.splitext(filename)[1].lower()
    return ext in BACKEND_SUPPORT.get(backend, set())


# Main Dataset Loader >>>
def load_dataset(
    directory: str,
    img_size: tuple = (224, 224),
    color_mode: str = "grayscale",
    dtype: np.dtype = np.uint8,
    backend: str = "opencv",
    raise_on_error: bool = False,
    **kwargs,
) -> tuple:
    """
    Load an image dataset with multi-backend support.

    Args:
        directory (str): Root directory containing class folders.
        img_size (tuple, optional): Target image dimensions (width, height). Defaults to (224, 224).
        color_mode (str, optional): Color mode ('rgb', 'bgr', 'grayscale', 'hsv', 'lab', 'rgba', 'rgba_drop'). Defaults to "grayscale".
        dtype (np.dtype, optional): Output array data type. Defaults to np.uint8.
        backend (str, optional): Image loading backend ('opencv', 'pil'). Defaults to "opencv".
        raise_on_error (bool, optional): Whether to raise an error on any problem. Defaults to False.
        **kwargs: Additional configuration options.
            - print_sig: Custom print signature.
            - progbar_desc: Progress bar description.

    Returns:
        tuple: (image_data: np.ndarray, labels: np.ndarray)

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If the directory is empty or the backend is unsupported.
    """
    # Check if the directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Get the list of subdirectories (class folders)
    label_names = sorted(os.listdir(directory))
    if not label_names:
        raise ValueError(f"Directory is empty: {directory}")

    # Validate the backend
    if backend not in {"opencv", "pil"}:
        raise ValueError(f"Unsupported backend: {backend}")

    # Print a message indicating the backend being used
    print(
        f"[bold green]Loading [white]data using [yellow]{backend} [white]backend from: [yellow]{directory}"
    )

    # Select the appropriate image loader based on the backend
    loaders = {"opencv": load_image_opencv, "pil": load_image_pil}
    load_func = loaders[backend]

    # Initialize lists to store image data and labels
    x_data, y_data = [], []

    # Calculate the total number of images for the progress bar
    total_images = sum(len(files) for _, _, files in os.walk(directory))

    # Use tqdm to show a progress bar while loading images
    with Progress(
        SpinnerColumn(finished_text="-"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    ) as pbar:
        # Initialize the progress bar
        loading_task = pbar.add_task(
            kwargs.get("progbar_desc", "[bold green]Loading images"),
            total=total_images,
        )

        # Iterate over each class folder
        for label_idx, label_name in enumerate(label_names):
            label_dir = os.path.join(directory, label_name)
            if not os.path.isdir(label_dir):
                continue  # Skip if it's not a directory

            # Iterate over each image in the class folder
            for img_name in os.listdir(label_dir):
                # Skip unsupported file formats
                if not is_supported_file(img_name, backend):
                    continue

                # Load the image using the selected backend
                img_path = os.path.join(label_dir, img_name)
                img = load_func(img_path, img_size, color_mode, raise_on_error)

                # If the image is loaded successfully, add it to the dataset
                if img is not None:
                    x_data.append(img.astype(dtype))
                    y_data.append(label_idx)
                elif raise_on_error:
                    raise ValueError(f"Failed to load image: {img_path}")

                # Update the progress bar
                pbar.update(loading_task, advance=1)

    # Convert lists to NumPy arrays and return
    return np.array(x_data), np.array(y_data)


# PyTorch DataLoader >>>
class Torch_ImgDataloader(torch.utils.data.Dataset):
    """
    PyTorch image dataloader with multi-backend support and customizable preprocessing.

    Args:
        data_pairs (list): List of [one-hot label, image path] pairs.
        backend (str, optional): Image loading backend ('opencv', 'pil'). Defaults to "pil".
        color_mode (str, optional): Color mode ('rgb', 'bgr', 'grayscale', 'hsv', 'lab', 'rgba', 'rgba_drop'). Defaults to "rgb".
        transforms (callable, optional): Custom transform pipeline. Defaults to None.
        normalize (bool, optional): Whether to normalize pixel values to [0, 1]. Defaults to True.
        dtype (torch.dtype, optional): Tensor data type. Defaults to torch.float32.
        transform_timing (str, optional): When to apply transforms ('pre_norm', 'post_norm'). Defaults to "post_norm".
        raise_on_error (bool, optional): Whether to raise an error on any problem. Defaults to False.

    Raises:
        ValueError: If the backend is unsupported or an image fails to load.
    """

    def __init__(
        self,
        data_pairs,
        backend="pil",
        color_mode="rgb",
        transforms=None,
        normalize=True,
        dtype=torch.float32,
        transform_timing="post_norm",
        raise_on_error=False,
    ):
        # Validate the backend
        if backend not in {"opencv", "pil"}:
            raise ValueError(f"Unsupported backend: {backend}")

        # Initialize instance variables
        self.data_pairs = data_pairs
        self.backend = backend
        self.color_mode = color_mode
        self.transforms = transforms
        self.normalize = normalize
        self.dtype = dtype
        self.transform_timing = transform_timing
        self.raise_on_error = raise_on_error
        self.load_func = load_image_opencv if backend == "opencv" else load_image_pil

    def _process_image(self, img):
        """
        Process the loaded image through the transformation pipeline.

        Args:
            img (np.ndarray): Raw image array.

        Returns:
            torch.Tensor: Processed and transformed image tensor.

        Raises:
            ValueError: If the image is None or has an unsupported shape.
        """
        if img is None:
            raise ValueError("Image is None, cannot process.")

        # Convert the NumPy array to a PyTorch tensor
        img = torch.from_numpy(img).type(self.dtype, non_blocking=True)

        # Process the image based on the color mode
        if self.color_mode == "grayscale":
            img = img.unsqueeze(0)  # Add channel dimension for grayscale
        elif self.color_mode in {"rgb", "bgr", "hsv", "lab"}:
            if img.shape[2] == 4:  # RGBA
                img = img[:, :, :3]  # Drop alpha channel
            img = img.permute(2, 0, 1)  # Change from HWC to CHW format
        elif self.color_mode == "rgba":
            img = img.permute(2, 0, 1)  # Change from HWC to CHW format (retain alpha)
        elif self.color_mode == "rgba_drop":
            if img.shape[2] == 4:  # RGBA
                img = img[:, :, :3]  # Drop alpha channel
            img = img.permute(2, 0, 1)  # Change from HWC to CHW format
        else:
            raise ValueError(f"Unsupported color mode: {self.color_mode}")

        # Apply transforms before normalization if specified
        if self.transforms and self.transform_timing == "pre_norm":
            img = self.transforms(img)

        # Normalize pixel values to [0, 1] if specified
        if self.normalize:
            img = img / 255.0

        # Apply transforms after normalization if specified
        if self.transforms and self.transform_timing == "post_norm":
            img = self.transforms(img)

        return img

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data_pairs)

    def __getitem__(self, idx):
        """Return the processed image tensor and label for the given index."""
        label, img_path = self.data_pairs[idx]
        img = self.load_func(
            img_path, color_mode=self.color_mode, raise_on_error=self.raise_on_error
        )
        if img is None and self.raise_on_error:
            raise ValueError(f"Failed to load image: {img_path}")
        img_tensor = self._process_image(img)
        return img_tensor, label
