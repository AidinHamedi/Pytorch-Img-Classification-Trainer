# Libs >>>
import os
import gc
import sys
import time
import torch
import shutil
from torch import nn
from rich import box
from functools import wraps
from rich.table import Table
from rich.style import Style
from rich.console import (
    Console,
    RenderableType,
    ConsoleOptions,
    RenderResult,
)
from rich.segment import Segment
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
from contextlib import suppress
from contextlib import contextmanager
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# Modules >>>
from .Utils.Base.other import format_seconds
from .Utils.Base.device import get_device, check_device
from .Utils.Base.dynamic_args import DynamicArg, DA_Manager
from .Utils.Data.debug import (
    make_grid,
    retrieve_samples as dl_retrieve_samples,
)
from .Utils.Train.early_stopping import EarlyStopping
from .Utils.Train.eval import calc_metrics, eval as eval_model

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


# Clean up >>>
def cleanup(local_vars: dict):
    # Make a console
    console = Console()

    # Prefix
    cleanup_prefix = "[Cleanup] "

    with console_prefix(console, cleanup_prefix):
        # Clean up memory
        gc.collect()
        torch.cuda.empty_cache()

        # Close the progress bar
        with suppress(Exception):
            local_vars["progress_bar"].stop()
            console.print("Successfully closed the progress bar.")

        # Save the best model
        with suppress(Exception):
            if local_vars.get("epoch", 0) <= 1:
                torch.save(
                    local_vars["early_stopping"].load_best_model(
                        local_vars["model"], raise_error=True
                    ),
                    os.path.join(local_vars["model_save_path"], "best_model.pth"),
                )
                console.print("Successfully saved the best model.")

        # Close the tensorboard writers
        with suppress(Exception):
            local_vars["tbw_val"].close()
            local_vars["tbw_train"].close()
            console.print("Successfully closed the tensorboard writers.")
            local_vars["tbw_data"].close()

        # Delete short tensorboard logs
        with suppress(Exception):
            if local_vars.get("epoch", 0) <= 1:
                console.print(
                    "Tensorboard logs are too short, deleting them... (delete them manually if no confirmation is given)"
                )
                shutil.rmtree(local_vars["tb_log_dir"])
                console.print("Successfully deleted the short tensorboard logs.")

        # Delete the model save path
        with suppress(Exception):
            if local_vars.get("epoch", 0) <= 1:
                console.print(
                    "Training was too short, deleting the model save path... (delete it manually if no confirmation is given)"
                )
                shutil.rmtree(local_vars["model_save_path"])
                console.print("Successfully deleted the model save path.")

        # End
        console.print("Done.")


# Error handling >>>
def error_handler(log_errors: bool = True, allow_keyboard_interrupt: bool = True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Make a rich console
            console = Console()

            # Prefix
            error_handler_prefix = "[Error Handler]"

            # Get the locals of the function
            local_vars = {}

            def trace_func(frame, event, arg):
                if event == "return":
                    local_vars.update(frame.f_locals)
                return trace_func

            sys.settrace(trace_func)

            # Run the function
            try:
                results = func(*args, **kwargs)
            except KeyboardInterrupt:
                if allow_keyboard_interrupt:
                    console.print(f"\n\n{error_handler_prefix} KeyboardInterrupt!")
                else:
                    console.print_exception(show_locals=True)
                return "!KeyboardInterrupt"
            except Exception:
                if log_errors:
                    console.print_exception(show_locals=True)
                return "!Exception"
            finally:
                # Sleep for a bit to let the console catch up
                time.sleep(2)

                # Reset the trace function
                sys.settrace(None)

                # Do the cleanup (assuming cleanup is defined elsewhere)
                cleanup(local_vars)

            # Return the results
            return results

        return wrapper

    return decorator


# Main >>>
@error_handler()
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
    callbacks: list = [],
    train_eval_portion: float = 0.1,
    gradient_accumulation: bool = True,
    gradient_accumulation_steps: DynamicArg = DynamicArg(
        default_value=4, mode="static"
    ),
    mixed_precision: bool = True,
    mixed_precision_dtype: torch.dtype = torch.bfloat16,
    experiment_name: str = "!auto",
    model_export_path: str = "./models",
    log_debugging: bool = True,
    force_cpu: bool = False,
):
    # Init rich
    console = Console()

    # Make experiment name
    if experiment_name == "!auto":  # TODO Check for duplicates
        experiment_name = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}"

    # Start msg
    console.print(
        f"[bold green]Initializing... [default](Experiment name: [yellow]{experiment_name}[default])"
    )

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
    tb_log_dir = f"./logs/runs/{experiment_name}"
    console.print(f"Tensorboard log dir: [green]{tb_log_dir}")
    if log_debugging:
        tbw_data = SummaryWriter(log_dir=f"{tb_log_dir}/Data", max_queue=25)
    tbw_val = SummaryWriter(log_dir=f"{tb_log_dir}/Val", flush_secs=45)
    tbw_train = SummaryWriter(log_dir=f"{tb_log_dir}/Train", flush_secs=45)

    # Make the model save path
    model_save_path = f"{model_export_path}/{experiment_name}"
    os.makedirs(model_save_path, exist_ok=True)
    console.print(f"Model save path: [green]{model_save_path}")

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

    # Dynamic args manager
    da_manager = DA_Manager()

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
                    epoch,
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
            console.print(f"Train eval data len: [cyan]{train_eval_data_len}")
            console.print(f"Learning rate: [cyan]{optimizer.param_groups[0]['lr']}")

            # Progress bar
            progress_bar = Progress(
                TextColumn(epoch_verbose_prefix),
                SpinnerColumn(finished_text="[yellow]⠿"),
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
            for fp_idx, (x, y) in enumerate(train_dataloader_ins):
                # Forward pass + mixed precision
                with autocast(
                    device_type=device_str,
                    enabled=mixed_precision,
                    dtype=mixed_precision_dtype,
                ):
                    y_pred = model(x.to(device, non_blocking=True))
                    loss = loss_fn(y_pred, y.to(device, non_blocking=True))

                # Store the loss
                train_loss_data.append(loss.item())

                # Normalize the loss if using gradient accumulation
                if gradient_accumulation:
                    loss = loss / gradient_accumulation_steps_ins

                # Backward pass + mixed precision
                if mixed_precision:
                    mpt_scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Model param update (Train step)
                if not gradient_accumulation or (
                    (fp_idx + 1) % gradient_accumulation_steps_ins == 0
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
                        train_eval_data.append(
                            {"y_pred": y_pred.detach().cpu(), "y": y.detach().cpu()}
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

            # Show time elapsed
            console.print(
                f"Epoch time: [cyan]{format_seconds(time.time() - epoch_start_time)}s"
            )

            # Early stopping
            early_stopping.update(
                epoch, test_eval[early_stopping_cnf["monitor"]], model
            )
            if early_stopping.should_stop:
                console.print("Stopping the training early...")
                # End
                break

            # Save the latest model
            torch.save(model, os.path.join(model_save_path, "latest_model.pth"))
