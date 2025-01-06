# Libs >>>
import torch
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

    # Convert tensors to numpy arrays
    y = y.numpy()
    y_pred = y_pred.numpy()

    # Convert predictions to class labels
    y_pred_labels = y_pred.argmax(axis=1)
    y_labels = y.argmax(axis=1)

    # Calculating the metrics
    metrics_dict = {
        "Loss": safe_metric_calculation(
            loss_reduction, loss_fn, torch.tensor(y_pred), torch.tensor(y)
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
        "AUROC": safe_metric_calculation(roc_auc_score, y, y_pred, multi_class="ovr"),
        "Accuracy": safe_metric_calculation(accuracy_score, y_labels, y_pred_labels),
        "Cohen's Kappa": safe_metric_calculation(
            cohen_kappa_score, y_labels, y_pred_labels
        ),
        "Matthews Correlation Coefficient": safe_metric_calculation(
            matthews_corrcoef, y_labels, y_pred_labels
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

    with torch.no_grad():
        # Use provided Progressbar if not None, else create a new one
        if Progressbar is None:
            # Create a new Progress instance with the desired columns and disable if not verbose
            pbar = Progress(
                SpinnerColumn(finished_text="-"),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
                TimeElapsedColumn(),
                disable=not verbose,
            )
            # Use the new Progress instance as a context manager
            with pbar:
                task = pbar.add_task(
                    kwargs.get("progbar_desc", "Evaluation"), total=len(dataloader)
                )
                for x, y in dataloader:
                    y_pred = model(x.to(device, non_blocking=True))

                    all_y.append(y)
                    all_y_pred.append(y_pred.cpu())

                    pbar.update(task, advance=1)
        else:
            # Use the provided Progressbar without creating a new context
            task = Progressbar.add_task(
                kwargs.get("progbar_desc", "Evaluation"), total=len(dataloader)
            )
            for x, y in dataloader:
                y_pred = model(x.to(device, non_blocking=True))

                all_y.append(y)
                all_y_pred.append(y_pred.cpu())

                Progressbar.update(task, advance=1)

    all_y = torch.cat(all_y)
    all_y_pred = torch.cat(all_y_pred)

    metrics = calc_metrics(all_y, all_y_pred, loss_fn.cpu() if loss_fn else None)

    if return_preds:
        return metrics, all_y_pred, all_y
    else:
        return metrics