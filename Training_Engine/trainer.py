# Libs >>>
import time
import torch
from torch import nn
from rich.console import Console
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
from contextlib import contextmanager
from torch.amp import GradScaler, autocast

# Modules >>>
from .Utils.Base.dynamic_args import DynamicArg
from .Utils.Base.device import get_device
from .Utils.Train.early_stopping import EarlyStopping
from .Utils.Train.eval import calc_metrics, eval as eval_model

# Conf >>>
epoch_verbose_prefix = " | "


# Prep >>>
@contextmanager
def console_prefix(console, prefix=" | "):
    # Save the original print method
    original_print = console.print

    # Define a new print method with the prefix
    def custom_print(*args, **kwargs):
        # Add the prefix to the output
        prefixed_args = [
            f"{prefix}{arg}" if isinstance(arg, str) else arg for arg in args
        ]
        original_print(*prefixed_args, **kwargs)

    # Temporarily override the console.print method
    console.print = custom_print
    try:
        yield
    finally:
        # Restore the original print method
        console.print = original_print


# Main >>>
def fit(
    model: nn.Module,
    train_dataloader: DynamicArg,
    test_dataloader: DynamicArg,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    max_epochs: int = 512,
    early_stopping: dict = {
        "patience": 24,
        "monitor": "Cohen's Kappa",
        "mode": "max",
        "min_delta": 0.00001,
    },
    callbacks: list = [],
    train_eval_portion: float = 0.1,
    gradient_accumulation: bool = False,
    gradient_accumulation_steps: int = 4,
    mixed_precision: bool = True,
    mixed_precision_dtype: torch.dtype = torch.bfloat16,
    force_cpu: bool = False,
):
    # Init rich
    console = Console()

    # Start msg
    console.print("[bold green]Initializing...")

    # Get device
    device = get_device(verbose=True, CPU_only=force_cpu)
    device_str = str(device)

    # Move to device
    model = model.to(device, non_blocking=True)

    # Make the early stopping
    early_stopping = EarlyStopping(
        monitor_name=early_stopping["monitor"],
        mode=early_stopping["mode"],
        patience=early_stopping["patience"],
        min_delta=early_stopping["min_delta"],
        verbose=True,
    )

    # Train vars
    mpt_scaler = GradScaler(device=device_str, enabled=mixed_precision)
    Metrics_hist = {}

    # Get eval dataloader
    if test_dataloader.mode == "static":
        test_dataloader_ins = test_dataloader.get_value()
    if train_dataloader.mode == "static":
        train_dataloader_ins = train_dataloader.get_value()

    # Make the train loop
    for epoch in range(1, max_epochs + 1):
        # Epoch msg
        console.print(
            f"\n[bold bright_white]Epoch [green]{epoch}[bold]/[cyan]{max_epochs} [yellow]-->"
        )
        with console_prefix(console, prefix=epoch_verbose_prefix):
            console.print("[green]Preparing...")
            # Epoch prep
            epoch_start_time = time.time()

            # Get env vars
            env_vars = {
                k: v
                for k, v in locals().items()
                if k != "env_vars"
                and isinstance(
                    v, (int, float, str, bool, bytes, list, tuple, dict, set)
                )
            }

            # Get dataloaders
            if test_dataloader.mode == "dynamic":
                test_dataloader.set_env_args(env_vars)
                test_dataloader_ins = test_dataloader.get_value()
            if train_dataloader.mode == "dynamic":
                train_dataloader.set_env_args(env_vars)
                train_dataloader_ins = train_dataloader.get_value()

            # Prep
            model.train()
            loss_fn = loss_fn.to(device, non_blocking=True)
            train_dataloader_len = train_dataloader_ins.__len__()
            train_total_batches = (
                int(train_dataloader_len / gradient_accumulation_steps)
                if gradient_accumulation
                else train_dataloader_len
            )
            train_eval_data_len = round(train_total_batches * train_eval_portion)
            train_eval_data = []
            batch_idx = 0

            # Verbose
            console.print(f"Train eval data len: [cyan]{train_eval_data_len}")

            # Progress bar
            progress_bar = Progress(
                TextColumn(epoch_verbose_prefix),
                SpinnerColumn(finished_text="-"),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
                TimeElapsedColumn(),
            )
            training_task = progress_bar.add_task(
                "Training",
                total=train_total_batches,
            )
            
            # Show the progress bar
            progress_bar.start()
            
            # Train
            for idx, (x, y) in enumerate(train_dataloader_ins):
                # Forward pass + mixed precision
                with autocast(device_type=device_str, enabled=mixed_precision):
                    y_pred = model(x.to(device, non_blocking=True))
                    loss = loss_fn(y_pred, y.to(device, non_blocking=True))

                # Normalize the loss if using gradient accumulation
                if gradient_accumulation:
                    loss = loss / gradient_accumulation_steps

                # Backward pass + mixed precision
                if mixed_precision:
                    mpt_scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Model param update (Train step)
                if not gradient_accumulation or (
                    (idx + 1) % gradient_accumulation_steps == 0
                ):
                    # Update the batch_idx
                    batch_idx += 1
                    
                    # Gradient unscale
                    if mixed_precision and (
                        False  # TODO
                    ):
                        mpt_scaler.unscale_(optimizer)

                    # Optimizer step
                    if mixed_precision:
                        mpt_scaler.step(optimizer)
                        mpt_scaler.update()
                    else:
                        optimizer.step()

                    # Zero grad
                    for param in model.parameters():
                        param.grad = None

                    # Train Eval Data
                    if batch_idx >= (train_total_batches - train_eval_data_len):
                        progress_bar.set_postfix_str("Recording Eval Data...")
                        train_eval_data.append(
                            {"y_pred": y_pred.detach().cpu(), "y": y.detach().cpu()}
                        )
                        
                    # Progress bar update
                    progress_bar.update(training_task, advance=1)
                    
            # Close progress bar
            progress_bar.stop()
