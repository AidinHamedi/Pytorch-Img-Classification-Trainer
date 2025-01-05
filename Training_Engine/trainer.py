# Libs >>>
import torch
from torch import nn
from rich.console import Console

# Modules >>>
from .Utils.Base.dynamic_args import DynamicArg
from .Utils.Base.device import get_device
from .Utils.Train.early_stopping import EarlyStopping

# Conf >>>

# Prep >>>


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
    verbose: bool = True,
    force_cpu: bool = False,
):
    # Init rich
    console = Console()

    # Start msg
    console.print("[bold green]Initializing...")

    # Get device
    device = get_device(verbose=verbose, CPU_only=force_cpu)

    # Move to device
    model = model.to(device, non_blocking=True)
    loss_fn = loss_fn.to(device, non_blocking=True)

    # Make the early stopping
    early_stopping = EarlyStopping(
        monitor_name=early_stopping["monitor"],
        mode=early_stopping["mode"],
        patience=early_stopping["patience"],
        min_delta=early_stopping["min_delta"],
        verbose=verbose,
    )

    # Make the temp vars
    Mettrics_hist = {}

    # Make the train loop
    for epoch in range(1, max_epochs + 1):
        # Epoch msg
        console.print(
            f"\n[bold bright_white]Epoch [green]{epoch}[bold]/[cyan]{max_epochs} [yellow]-->"
        )

        # Get the dataloaders
