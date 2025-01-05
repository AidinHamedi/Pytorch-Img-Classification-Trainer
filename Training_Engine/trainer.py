# Libs >>>
import torch
from torch import nn
from rich.console import Console

# Modules >>>
from .Utils.Base.dynamic_args import DynamicArg
from .Utils.Base.device import get_device


# Conf >>>

# Prep >>>

# Main >>>
def fit(
    model: nn.Module,
    train_dataloader: DynamicArg,
    test_dataloader:  DynamicArg,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    max_epochs: int = 512, 
    early_stopping: dict = {
        "patience": 24,
        "monitor": "Cohen's Kappa",
        "mode": "max",
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
    
    # Make the temp vars
    total_steps = 0
    
    # Make the train loop
    for epoch in range(max_epochs):
        # Epoch msg
        