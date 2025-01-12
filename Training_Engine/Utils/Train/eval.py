# Libs >>>
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    cohen_kappa_score,
    matthews_corrcoef,
)
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
from typing import Dict, Optional, Callable, Union, Tuple


# Main >>>
def loss_reduction(loss_fn, y_pred, y):
    # Check if the loss function has a reduction attribute
    if hasattr(loss_fn, "reduction") and loss_fn.reduction == "none":
        # Calculate individual losses
        losses = loss_fn(y_pred, y)
        # Apply reduction to get a single scalar value
        loss = losses.mean()
    else:
        # Calculate the loss directly
        loss = loss_fn(y_pred, y)

    return loss


def calc_metrics(y, y_pred, loss_fn, averaging="macro"):
    """
    Calculate various metrics for multi-class classification.

    Args:
        y (torch.Tensor): Ground truth labels, shape (batch_size, num_classes)
        y_pred (torch.Tensor): Model predictions, shape (batch_size, num_classes)
        loss_fn (callable): The loss function used during training

    Returns:
        dict: A dictionary containing various evaluation metrics
    """
    # Define a small epsilon value
    epsilon = 1e-10

    # Function to safely calculate a metric
    def safe_metric_calculation(metric_fn, *args, **kwargs):
        try:
            return metric_fn(*args, **kwargs)
        except Exception:
            return epsilon

    # Convert tensors to numpy arrays (first convert to fp32 to make it support a dtype like bfloat16)
    y = y.type(torch.float32, non_blocking=True).numpy()
    y_pred = y_pred.type(torch.float32, non_blocking=True).numpy()

    # Convert predictions to class labels
    y_pred_labels = y_pred.argmax(axis=1)
    y_labels = y.argmax(axis=1)

    # Calculating the metrics
    metrics_dict = {
        "Loss": float(
            safe_metric_calculation(
                loss_reduction, loss_fn, torch.tensor(y_pred), torch.tensor(y)
            )
        ),
        f"F1 Score ({averaging})": safe_metric_calculation(
            f1_score, y_labels, y_pred_labels, average=averaging
        ),
        f"Precision ({averaging})": safe_metric_calculation(
            precision_score, y_labels, y_pred_labels, average=averaging, zero_division=0
        ),
        f"Recall ({averaging})": safe_metric_calculation(
            recall_score, y_labels, y_pred_labels, average=averaging
        ),
        "AUROC": float(
            safe_metric_calculation(roc_auc_score, y, y_pred, multi_class="ovr")
        ),
        "Accuracy": safe_metric_calculation(accuracy_score, y_labels, y_pred_labels),
        "Cohen's Kappa": float(
            safe_metric_calculation(cohen_kappa_score, y_labels, y_pred_labels)
        ),
        "Matthews Correlation Coefficient": float(
            safe_metric_calculation(matthews_corrcoef, y_labels, y_pred_labels)
        ),
    }

    return metrics_dict


def eval(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    loss_fn: Optional[Callable] = None,
    verbose: bool = True,
    return_preds: bool = False,
    Progressbar: Progress = None,
    **kwargs,
) -> Union[Dict[str, float], Tuple[Dict[str, float], torch.Tensor, torch.Tensor]]:
    """
    Evaluates the model on the provided dataloader for multi-class classification.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader containing evaluation data.
        model (torch.nn.Module): The PyTorch model to evaluate.
        loss_fn (Optional[Callable]): The loss function for evaluation (e.g., CrossEntropyLoss). If None, loss is not calculated.
        device (torch.device): The device to run the evaluation on.
        verbose (bool, optional): Whether to show progress bar. Defaults to True.
        return_preds (bool, optional): Whether to return model predictions and original labels. Defaults to False.
        Progressbar (Progress, optional): The progress bar object. Defaults to None.
        **kwargs: Additional keyword arguments.
            - progbar_desc (str): Custom description for the progress bar.

    Returns:
        Union[Dict[str, float], Tuple[Dict[str, float], torch.Tensor, torch.Tensor]]: A dictionary containing various evaluation metrics, and optionally the model predictions and original labels.

    Example:
        >>> eval_metrics = eval(test_dataloader, model, nn.CrossEntropyLoss(), device)
        >>> print(f"Test Accuracy: {eval_metrics['Accuracy']:.2f}%")
    """
    model.eval()
    all_y = []
    all_y_pred = []

    # Initialize progress bar if not provided
    if Progressbar is None:
        Progressbar = Progress(
            SpinnerColumn(finished_text="-"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            disable=not verbose,
        )
        Progressbar.start()
        cleanup_progress = (
            True  # Flag to indicate we need to stop the progress bar later
        )
    else:
        cleanup_progress = False  # No cleanup needed if Progressbar was provided

    # Add a task to the progress bar
    task = Progressbar.add_task(
        kwargs.get("progbar_desc", "Evaluation"), total=len(dataloader)
    )

    # Process the dataloader
    with torch.no_grad():
        for x, y in dataloader:
            y_pred = model(x.to(device, non_blocking=True))
            all_y.append(y)
            all_y_pred.append(y_pred.cpu())
            Progressbar.update(task, advance=1)

    # Clean up the progress bar if it was created here
    Progressbar.stop_task(task)
    if cleanup_progress:
        Progressbar.stop()

    # Concatenate results
    all_y = torch.cat(all_y)
    all_y_pred = torch.cat(all_y_pred)

    # Calculate metrics
    metrics = calc_metrics(all_y, all_y_pred, loss_fn.cpu() if loss_fn else None)

    # Return results
    if return_preds:
        return metrics, all_y_pred, all_y
    else:
        return metrics


def calculate_stability(data, window_size=30):
    """
    Calculate the stability of a dataset based on the R² between the data and its moving average.

    Parameters:
        data (list or np.array): The input data (e.g., batch losses).
        window_size (int): Size of the moving average window for detrending.

    Returns:
        stability_score (float): A score representing stability (0 = full noise, 1 = perfect stability).
    """
    data = np.asarray(data)
    if window_size > len(data):
        window_size = len(data)

    # Calculate moving average
    moving_avg = np.convolve(data, np.ones(window_size) / window_size, mode="valid")

    # Pad the moving average to match the original data length
    pad_width = (window_size - 1) // 2
    moving_avg_padded = np.pad(
        moving_avg, (pad_width, len(data) - len(moving_avg) - pad_width), mode="edge"
    )

    # Calculate residuals
    residuals = data - moving_avg_padded

    # Calculate total sum of squares (SS_tot)
    mean_data = np.mean(data)
    SS_tot = np.sum((data - mean_data) ** 2)

    # Calculate residual sum of squares (SS_res)
    SS_res = np.sum(residuals**2)

    # Calculate R²
    if SS_tot == 0:
        r_squared = 1.0  # Perfectly stable data
    else:
        r_squared = 1 - (SS_res / SS_tot)
        r_squared = max(0, r_squared)

    return r_squared
