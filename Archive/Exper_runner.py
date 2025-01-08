# Libs >>>
import torch
import typer
from rich import pretty
from typing import Optional

# Conf >>>


# Prep >>>
pretty.install()


# CLI Validation >>>
def validate_img_format_option(value):
    # This callback ensures the value is one of the allowed options
    if value not in {"rgb", "rgb"}:
        raise typer.BadParameter("Invalid option. Choose either `rgb` or `grayscale`.")
    return value


def validate_dl_backend_option(value):
    # This callback ensures the value is one of the allowed options
    if value not in {"pil", "opencv"}:
        raise typer.BadParameter("Invalid option. Choose either `pil` or `opencv`.")
    return value


def validate_dtype_option(values):
    # This callback ensures the value is one of the allowed options
    if values not in {"float32", "float64", "float16"}:
        raise typer.BadParameter(
            "Invalid option. Choose either `float32`, `float64`, `float16`."
        )
    return values


def validate_class_weighting_option(value):
    # This callback ensures the value is one of the allowed options
    if value not in {
        "linear",
        "square",
        "sqrt",
        "1p5_Power",
        "1p2_Power",
        "cube",
        "harmonic",
        "log",
    }:
        raise typer.BadParameter(
            "Invalid option. Choose either `square`, `sqrt`, `1p5_Power`, `1p2_Power`, `cube`, `harmonic`, `log`, and `linear`."
        )
    return value


# Main >>>
def main(
    experiment_dir: str = typer.Argument(
        help="The dir containing the experiment configs"
    ),
    main_data_dir: str = typer.Argument(
        help="The dataset dir, will be used as the train set if eval_data_dir is given"
    ),
    eval_data_dir: Optional[str] = typer.Option(
        None,
        help="This is the validation set dir, will be used if `auto-split` is false",
    ),
    auto_split: bool = typer.Option(
        True,
        help="Auto split dataset (Will auto split the data in `main_data_dir` to Train and Test, Wont use `eval_data_dir`)",
    ),
    split_ratio: Optional[float] = typer.Option(
        0.8,
        help="Split ratio (Train & Test). Required if `auto-split` is True, otherwise ignored.",
    ),
    img_format: Optional[str] = typer.Option(
        "rgb",
        help="The loading format of the images, `rgb` or `grayscale`",
        callback=validate_img_format_option,
    ),
    experiment_output_dir: Optional[str] = typer.Option(
        "./output", help="The output dir for report of the training"
    ),
    tensorboard_log_dir: Optional[str] = typer.Option(
        "./tensorboard_logs", help="The tensorboard log dir"
    ),
    dl_backend: Optional[str] = typer.Option(
        "pil",
        help="The backend for loading the images, `pil` or `opencv`",
        callback=validate_dl_backend_option,
    ),
    dtype: Optional[str] = typer.Option(
        "float32",
        help="The image loader data type: `float64`, `float32`, `float16`",
        callback=validate_dtype_option,
    ),
    class_weighting_method: Optional[str] = typer.Option(
        "linear",
        help="The class weighting method: `square`, `sqrt`, `1p5_Power`, `1p2_Power`, `cube`, `harmonic`, `log`, and `linear`",
    ),
):
    # Init msg
    print("[bold green]Starting...")


# Start >>>
if __name__ == "__main__":
    typer.run(main)
