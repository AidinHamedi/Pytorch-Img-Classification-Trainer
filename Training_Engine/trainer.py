# Libs >>>
import gc
import os
import shutil
import time
import shortuuid
from contextlib import (
    contextmanager,
    suppress,
)

import numpy as np
import pytorch_optimizer as TP_optim
import torch
from rich import box
from rich.console import (
    Console,
    ConsoleOptions,
    RenderableType,
    RenderResult,
)
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.segment import Segment
from rich.style import Style
from rich.table import Table
from torch import nn
from torch.compiler import cudagraph_mark_step_begin
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from .Utils.Base.device import check_device, get_device
from .Utils.Base.dynamic_args import DA_Manager, DynamicArg

# Modules >>>
from .Utils.Base.other import format_seconds
from .Utils.Data.debug import (
    make_grid,
)
from .Utils.Data.debug import (
    retrieve_samples as dl_retrieve_samples,
)
from .Utils.Train.early_stopping import EarlyStopping
from .Utils.Train.eval import (
    calc_metrics,
    calculate_stability,
)
from .Utils.Train.eval import (
    eval as eval_model,
)
from .Utils.Train.grad_mod import apply_gradient_modifiers

# Conf >>>
epoch_verbose_prefix = " | "


# Prep >>>
class PrefixedRenderable:
    def __init__(
        self, renderable: RenderableType, prefix: str, prefix_style: Style = None
    ):
        self.renderable = renderable
        self.prefix = prefix
        self.prefix_style = prefix_style or Style()

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        segments = console.render(self.renderable, options)
        prefix_segments = console.render_str(self.prefix, style=self.prefix_style)
        for line in Segment.split_lines(segments):
            yield from prefix_segments
            yield from line
            yield Segment("\n")


@contextmanager
def console_prefix(console, prefix=" | ", prefix_style: Style = None):
    original_print = console.print

    def custom_print(*args, **kwargs):
        new_args = []
        for arg in args:
            if isinstance(arg, RenderableType):
                new_args.append(PrefixedRenderable(arg, prefix, prefix_style))
            else:
                new_args.append(arg)
        original_print(*new_args, **kwargs)

    console.print = custom_print
    try:
        yield
    finally:
        console.print = original_print


# Main >>>
def fit(
    model: nn.Module,
    train_dataloader: DynamicArg,
    test_dataloader: DynamicArg,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    max_epochs: int = 512,
    early_stopping_cnf: dict = {
        "patience": 24,
        "monitor": "Cohen's Kappa",
        "mode": "max",
        "min_delta": 0.00001,
    },
    train_eval_portion: float = 0.1,
    gradient_accumulation: bool = True,
    gradient_accumulation_steps: DynamicArg = DynamicArg(
        default_value=4, mode="static"
    ),
    mixed_precision: bool = True,
    mixed_precision_dtype: torch.dtype = torch.float16,
    lr_scheduler: dict = {
        "scheduler": None,
        "enable": False,
        "batch_mode": False,
    },
    opt_features: dict = {"gradient centralization": True},
    grad_mod_exclude_layer_types: list = None,
    experiment_name: str = "!auto",
    model_export_path: str = "./Models",
    model_trace_input: torch.Tensor = None,
    cuda_compile: bool = True,
    cuda_compile_config: dict = {
        "dynamic": False,
        "fullgraph": True,
        "backend": "cudagraphs",
    },
    log_debugging: bool = True,
    force_cpu: bool = False,
):
    """
    Trains a PyTorch model using the provided training and test data loaders.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        train_dataloader (DynamicArg): Data loader for the training dataset. (Warning: This is a DynamicArg object)
        test_dataloader (DynamicArg): Data loader for the test dataset. (Warning: This is a DynamicArg object)
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        loss_fn (torch.nn.Module): Loss function used for training.
        max_epochs (int, optional): Maximum number of epochs to train. Defaults to 512.
        early_stopping_cnf (dict, optional): Configuration for early stopping. Defaults to
            {"patience": 24, "monitor": "Cohen's Kappa", "mode": "max", "min_delta": 0.00001}.
        train_eval_portion (float, optional): Portion of training data to use for evaluation. Defaults to 0.1.
        gradient_accumulation (bool, optional): Whether to use gradient accumulation. Defaults to True.
        gradient_accumulation_steps (DynamicArg, optional): Number of steps for gradient accumulation. Defaults to DynamicArg(default_value=4, mode="static").
            (Warning: This is a DynamicArg object)
        mixed_precision (bool, optional): Whether to use mixed precision training. Defaults to True.
        mixed_precision_dtype (torch.dtype, optional): Data type for mixed precision. Defaults to torch.float16.
        lr_scheduler (dict, optional): Configuration for learning rate scheduler. Defaults to
            {"scheduler": None, "enable": False, "metrics": None, "batch_mode": False}.
            - "scheduler" (torch.optim.lr_scheduler): The learning rate scheduler.
            - "enable" (bool): Whether to enable the learning rate scheduler.
            - "batch_mode" (bool): Whether to use batch mode for the learning rate scheduler.
        opt_features (dict, optional): Optimization features to use. Defaults to {"gradient centralization": True}.
            - "gradient centralization": bool
            - "gradient normalization": bool
            - "adaptive gradient clipping" [bool, float, float] (eps, alpha)
        grad_mod_exclude_layer_types (list, optional): List of layer types to exclude from gradient modification. Defaults to None.
        experiment_name (str, optional): Name of the experiment. Defaults to "!auto".
        model_export_path (str, optional): Path to save the trained model. Defaults to "./Models".
        model_trace_input (torch.Tensor, optional): Input tensor for tracing the model. Defaults to None.
        cuda_compile (bool, optional): Whether to use CUDA compilation. Defaults to True.
        cuda_compile_config (dict, optional): Configuration for CUDA compilation. Defaults to {"dynamic": True, "fullgraph": True, "backend": "cudagraphs"}.
        log_debugging (bool, optional): Whether to log debugging information. Defaults to True.
        force_cpu (bool, optional): Force training on CPU. Defaults to False.

    Returns:
        dict: A dictionary containing the best trained model and training metrics history.
            - "best_model" (nn.Module): The best model based on early stopping criteria.
            - "metrics_hist" (dict): Training and validation metrics history.

    Notes:
        - The function uses Rich library for rich text and progress bar visualization.
        - TensorBoard logging is used for metrics and debugging information.
        - Early stopping is applied to prevent overfitting.
    """
    # Init rich
    console = Console()

    # Make experiment name
    if experiment_name == "!auto":
        experiment_name = f"{time.strftime('%Y-%m-%d %H-%M-%S')}"
    else:
        experiment_name = f"{shortuuid.ShortUUID().random(length=8)}~{experiment_name}"  # Avoid duplicates

    # Start msg
    console.print(
        f"[bold green]Initializing... [default](Experiment name: [yellow]{experiment_name}[default])"
    )

    # Make var to hold the start time
    start_time = time.time()

    # Get device
    if not mixed_precision:
        device = get_device(verbose=True, CPU_only=force_cpu)
    else:
        device = check_device(model)
        console.print(
            f"Chosen device: [bold green]{device}[default], [yellow](using the device that model is currently on)"
        )
    device_str = str(device)

    # Move to device
    model = model.to(device, non_blocking=True)

    # Make the tensorboard writer
    tb_log_dir = f"./Logs/runs/{experiment_name}"
    console.print(f"Tensorboard log dir: [green]{tb_log_dir}")
    if log_debugging:
        tbw_data = SummaryWriter(log_dir=f"{tb_log_dir}/Data", max_queue=25)
    tbw_val = SummaryWriter(log_dir=f"{tb_log_dir}/Val", flush_secs=45)
    tbw_train = SummaryWriter(log_dir=f"{tb_log_dir}/Train", flush_secs=45)

    # Save the model in tensorboard (Must be before model compile)
    if log_debugging:
        if model_trace_input is None:
            raise ValueError(
                "model_trace_input must be provided for tensorboard model logging"
            )
        tbw_data.add_graph(model, model_trace_input.to(device))

    # Enable onednn fusion
    if device_str == "cuda":
        torch.jit.enable_onednn_fusion(True)
        torch.backends.cudnn.benchmark = True
        # Compile model
        if cuda_compile:
            console.print(f"Compiling model with: [green]{cuda_compile_config}")
            torch.set_float32_matmul_precision("high")
            torch._dynamo.config.cache_size_limit = 32
            model = torch.compile(model, **cuda_compile_config)
            console.print(
                "[red]Warning[reset]: The first time you run this model, it will be slow! (Using torch compile)"
            )
            if log_debugging:
                console.print(
                    "[red]Warning[reset]: When using torch compile some log_debugging features may not work properly!"
                )
    elif cuda_compile:
        console.print(
            "[red]Warning[reset]: cuda_compile is only available for cuda devices!"
        )

    # Make the model save path
    model_save_path = f"{model_export_path}/{experiment_name}"
    os.makedirs(model_save_path, exist_ok=True)
    console.print(f"Model save path: [green]{model_save_path}")

    # Train mods
    train_mods = []
    console.print("[yellow]Train mods:")
    for key, value in {
        "gradient centralization": [TP_optim.centralize_gradient, False],
        "gradient normalization": [TP_optim.normalize_gradient, False],
        "adaptive gradient clipping": [TP_optim.agc, True],
    }.items():
        if key in opt_features:
            if isinstance(opt_features[key], (list, tuple)) and opt_features[key][0]:
                train_mods.append([*value, *opt_features[key][1:]])
            elif opt_features[key]:
                train_mods.append([*value])
            console.print(f" - {key}: {opt_features[key]}")
        else:
            console.print(f" - [gray]{key}[reset]: [yellow]Not given")

    # Make a function to handel the lr scheduler
    def _lr_scheduler_step():
        if lr_scheduler["enable"]:
            lr_scheduler["scheduler"].step()

    # Make the early stopping
    early_stopping = EarlyStopping(
        monitor_name=early_stopping_cnf["monitor"],
        mode=early_stopping_cnf["mode"],
        patience=early_stopping_cnf["patience"],
        min_delta=early_stopping_cnf["min_delta"],
        verbose=True,
    )

    # Train vars
    mpt_scaler = GradScaler(device=device_str, enabled=mixed_precision)
    metrics_hist = {"Train": [], "Val": []}
    train_total_fp = 0

    # Dynamic args manager
    da_manager = DA_Manager()

    # Make the train loop
    try:
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

                # Update dynamic args
                da_manager.update(env_vars)

                # Get dataloaders
                test_dataloader_ins = da_manager.auto_get(test_dataloader)
                train_dataloader_ins = da_manager.auto_get(train_dataloader)

                # Get dynamic args
                gradient_accumulation_steps_ins = da_manager.auto_get(
                    gradient_accumulation_steps
                )

                # Log debug images
                if log_debugging:
                    tbw_data.add_image(
                        "Train-Dataloader",
                        make_grid(
                            torch.stack(
                                dl_retrieve_samples(
                                    train_dataloader_ins,
                                    num_samples=9,
                                    selection_method="random",
                                    seed=42,
                                )
                            ),
                            nrow=3,
                            padding=2,
                            normalize=True,
                            value_range=(0, 1),
                            pad_value=0,
                            format="CHW",
                        ),
                        epoch - 1,
                    )

                # Prep
                model.train()
                loss_fn = loss_fn.to(device, non_blocking=True)
                train_dataloader_len = train_dataloader_ins.__len__()
                train_total_batches = (
                    int(train_dataloader_len / gradient_accumulation_steps_ins)
                    if gradient_accumulation
                    else train_dataloader_len
                )
                train_eval_data_len = round(train_total_batches * train_eval_portion)
                train_loss_data = []
                train_eval_data = []
                batch_idx = 0

                # Verbose
                console.print(
                    f"Train batch size: [cyan]{train_dataloader_ins.batch_size * (gradient_accumulation_steps_ins if gradient_accumulation else 1)}"
                )
                console.print(f"Train eval data len: [cyan]{train_eval_data_len}")
                console.print(f"Learning rate: [cyan]{optimizer.param_groups[0]['lr']}")

                # Progress bar
                progress_bar = Progress(
                    TextColumn(epoch_verbose_prefix),
                    SpinnerColumn(finished_text="[yellow]⠿"),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(show_speed=True),
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
                for fp_idx, (x, y) in enumerate(train_dataloader_ins):
                    # Forward pass + mixed precision
                    with autocast(
                        device_type=device_str,
                        enabled=mixed_precision,
                        dtype=mixed_precision_dtype,
                    ):
                        if (
                            cuda_compile
                            and device_str == "cuda"
                            and cuda_compile_config.get("backend", "") == "cudagraphs"
                        ):
                            cudagraph_mark_step_begin()
                        y_pred = model(x.to(device, non_blocking=True))
                        loss = loss_fn(y_pred, y.to(device, non_blocking=True))

                    # Normalize the loss if using gradient accumulation
                    if gradient_accumulation:
                        loss = loss / gradient_accumulation_steps_ins

                    # Backward pass + mixed precision
                    if mixed_precision:
                        mpt_scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    # Store the loss
                    train_loss_data.append(
                        loss.item()
                        * (
                            gradient_accumulation_steps_ins
                            if gradient_accumulation
                            else 1
                        )
                    )

                    # Model param update (Train step)
                    if not gradient_accumulation or (
                        (fp_idx + 1) % gradient_accumulation_steps_ins == 0
                    ):
                        # Update the batch_idx
                        batch_idx += 1

                        # Gradient unscale (For supporting grad modifiers like gradient clipping)
                        if mixed_precision or len(train_mods) > 0:
                            mpt_scaler.unscale_(optimizer)

                        # Apply gradient modifiers
                        if len(train_mods) > 0:
                            apply_gradient_modifiers(
                                model,
                                train_mods,
                                grad_mod_exclude_layer_types,
                            )

                        # Optimizer step
                        if mixed_precision:
                            mpt_scaler.step(optimizer)
                            mpt_scaler.update()
                        else:
                            optimizer.step()

                        # Zero grad
                        optimizer.zero_grad()

                        # LR Scheduler step
                        if lr_scheduler.get("batch_mode", False):
                            _lr_scheduler_step()

                        # Train Eval Data
                        if batch_idx >= (train_total_batches - train_eval_data_len):
                            train_eval_data.append(
                                {
                                    "y_pred": y_pred.detach().to(
                                        "cpu", non_blocking=True
                                    ),
                                    "y": y.to("cpu", non_blocking=True),
                                }
                            )

                            # Progress bar update
                            progress_bar.update(
                                training_task,
                                advance=1,
                                description="Training (Recording Eval Data)"
                                if batch_idx != train_total_batches
                                else "Training",
                            )
                        else:
                            # Progress bar update
                            progress_bar.update(training_task, advance=1)

                # Close task
                progress_bar.stop_task(training_task)

                # Move the loss function to the cpu
                loss_fn = loss_fn.cpu()

                # LR Scheduler step
                if not lr_scheduler.get("batch_mode", False):
                    _lr_scheduler_step()

                # Val
                train_eval = calc_metrics(
                    torch.cat([item["y"] for item in train_eval_data]),
                    torch.cat([item["y_pred"] for item in train_eval_data]),
                    loss_fn,
                )
                test_eval = eval_model(
                    test_dataloader_ins,
                    model,
                    device,
                    loss_fn=loss_fn,
                    Progressbar=progress_bar,
                )

                # Close progress bar
                progress_bar.stop()

                # Clean up
                torch.cuda.empty_cache()
                gc.collect()

                # Saving the results
                metrics_hist["Train"].append(train_eval)
                metrics_hist["Val"].append(test_eval)

                # Make the results table
                eval_table = Table(box=box.ROUNDED, highlight=True)
                eval_table.add_column("Set", justify="center", style="bold green")
                for metric in test_eval:
                    eval_table.add_column(metric, justify="center")
                for metric_set in [[train_eval, "Train"], [test_eval, "Val"]]:
                    eval_table.add_row(
                        metric_set[1],
                        *[
                            f"{metric_set[0][metric]:.5f}"
                            if isinstance(metric_set[0][metric], float)
                            else metric_set[0][metric]
                            for metric in test_eval
                        ],
                    )
                console.print(eval_table)

                # Tensorboard logging
                for metric in train_eval:
                    tbw_train.add_scalar(f"Metrics/{metric}", train_eval[metric], epoch)
                    tbw_val.add_scalar(f"Metrics/{metric}", test_eval[metric], epoch)
                for i, batch_loss in enumerate(train_loss_data, start=1):
                    tbw_train.add_scalar(
                        "Metrics/Iter-Loss",
                        batch_loss,
                        train_total_fp + i,
                    )
                tbw_data.add_histogram("Loss/Train", np.asarray(train_loss_data), epoch)
                if lr_scheduler.get("enable", False):
                    tbw_train.add_scalar(
                        "Other/Train-LR",
                        lr_scheduler["scheduler"].get_last_lr()[0],
                        epoch,
                    )
                tbw_train.add_scalar(
                    "Metrics/Train stability",
                    calculate_stability(
                        train_loss_data,
                        window_size=round(train_dataloader_len / 2.5),
                    ),
                    epoch,
                )

                # Log parameters
                if log_debugging:
                    for name, param in model.named_parameters():
                        param_tag, param_type = (
                            ">".join(name.replace(".", ">").split(">")[:-1]),
                            name.replace(".", ">").split(">")[-1],
                        )
                        # Add a check before adding histogram
                        if param.data.numel() > 0 and not torch.all(
                            torch.isnan(param.data)
                        ):
                            tbw_data.add_histogram(
                                f"Train-Parameters|>>{param_tag}/{param_type}",
                                param.data.cpu(),
                                epoch,
                            )

                # Log train time
                tbw_data.add_scalar(
                    "Other/Epoch_time (minutes)",
                    (time.time() - epoch_start_time) / 60,
                    epoch,
                )

                # Update some vars
                train_total_fp += train_dataloader_len

                # Show time elapsed
                console.print(
                    f"Epoch time: [cyan]{format_seconds(time.time() - epoch_start_time)}"
                )

                # Early stopping
                early_stopping.update(
                    epoch, test_eval[early_stopping_cnf["monitor"]], model
                )
                if early_stopping.should_stop:
                    print("Stopping the training early...")
                    # End
                    break

                # Save the latest model
                torch.save(model, os.path.join(model_save_path, "latest_model.pth"))

    # Handel errors
    except KeyboardInterrupt:
        console.print(
            "\n\n[bold red]KeyboardInterrupt detected.[yellow] Stopping the training..."
        )
    except Exception:
        console.print("\n\n[bold red]An error occurred during training.")
        console.print_exception(show_locals=False)

    # Close the progress bar
    with suppress(Exception):
        progress_bar.stop()
        console.print("[underline]Successfully closed the progress bar.")

    # Load the best model + save it / delete the save path
    with suppress(Exception):
        if epoch > 2:
            early_stopping.load_best_model(model, raise_error=True, verbose=False)
            console.print("[underline]Successfully loaded the best model.")
            torch.save(model, os.path.join(model_save_path, "best_model.pth"))
            console.print("[underline]Successfully saved the best model.")
        else:
            console.print(
                "Training was too short, deleting the model save path... (delete it manually if no confirmation is given)"
            )
            shutil.rmtree(model_save_path)
            console.print("[underline]Successfully deleted the model save path.")

    # Close the tensorboard writers
    with suppress(Exception):
        tbw_val.close()
        tbw_train.close()
        console.print("[underline]Successfully closed the tensorboard writers.")
        tbw_data.close()

    # Delete short tensorboard logs
    with suppress(Exception):
        if not epoch > 2:
            console.print(
                "Tensorboard logs are too short, deleting them... (delete them manually if no confirmation is given)"
            )
            shutil.rmtree(tb_log_dir)
            console.print("[underline]Successfully deleted the short tensorboard logs.")

    # Print the final metrics
    if epoch > 2:
        console.print(
            f"[yellow]Best model from epoch [green]{early_stopping.best_epoch}[yellow] metrics: "
        )
        with console_prefix(console, prefix="   "):
            result_table = Table(box=box.ROUNDED, highlight=True)
            result_table.add_column("Set", justify="center", style="bold green")
            for metric in metrics_hist["Val"][early_stopping.best_epoch - 1]:
                result_table.add_column(metric, justify="center")
            for metric_set in [
                [metrics_hist["Train"][early_stopping.best_epoch - 1], "Train"],
                [metrics_hist["Val"][early_stopping.best_epoch - 1], "Val"],
            ]:
                result_table.add_row(
                    metric_set[1],
                    *[
                        f"{metric_set[0][metric]:.5f}"
                        if isinstance(metric_set[0][metric], float)
                        else metric_set[0][metric]
                        for metric in test_eval
                    ],
                )
            console.print(result_table)

    # Print the final time
    console.print(
        f"Training completed in: [cyan]{format_seconds(time.time() - start_time)}"
    )

    # return model, metrics_hist
    return {
        "best_model": model,
        "metrics_hist": metrics_hist,
    }
