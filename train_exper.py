# Libs >>>
import torch
from torch import nn
from rich import print
import pytorch_optimizer as TP_optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as v2_transforms

# Modules >>>
from Training_Engine.Utils.Base.device import get_device
from Training_Engine.Utils.Data.data_loader import Torch_ImgDataloader, make_data_pairs
from Training_Engine.Utils.Base.dynamic_args import DynamicArg
from Training_Engine.trainer import fit

# Data loading Conf >>>
main_data_dir = "./Database/Data"  # Main dataset dir
eval_data_dir = (
    "./Database/Validation"  # Eval dataset dir (Not needed if "auto_split" is True)
)
img_res = [224, 224]  # img loading resolution
img_format = "rgb"  # rgb, grayscale
dl_backend = "pil"  # pil or opencv. opencv some times doesn't work properly
dtype = torch.float32  # data type
auto_split = False  # Auto split dataset (Will auto split the data in "main_data_dir" to Train and Test, Wont use "eval_data_dir")
split_ratio = 0.8  # Split (Train&Test) ~ auto_split==True
class_weighting_method = "linear"  # class weighting method
dataLoader_num_workers = 8

# Train Conf >>>
train_batchsize = 64
eval_batchsize = 32
train_gradient_accumulation = None
dataLoader_num_workers = 8


# Main >>>
def train(extra_args: dict):
    # Init msg
    print("[bold green]Starting...")

    # Make data pairs
    data_pairs = make_data_pairs(
        train_dir=main_data_dir,
        val_dir=eval_data_dir,
        auto_split=auto_split,
        split_ratio=split_ratio,
        class_weighting_method=class_weighting_method,
    )
    print("[yellow]Data pairs info:")
    for key in data_pairs["stats"]:
        print(f" - {key}: {data_pairs['stats'][key]}")

    # Make the eval dataloader
    eval_dataloader = DataLoader(
        dataset=Torch_ImgDataloader(
            data_pairs["data_pairs"]["eval"],
            backend=dl_backend,
            color_mode=img_format,
            dtype=dtype,
            transforms=v2_transforms.Resize(img_res),
        ),
        batch_size=eval_batchsize,
        shuffle=False,
        num_workers=dataLoader_num_workers,
        persistent_workers=True,
        prefetch_factor=3,
        timeout=120,
        pin_memory=True,
        drop_last=False,
    )

    # Make the train dataloader
    def gen_train_dataloader(**env_args):
        train_dataloader = DataLoader(
            dataset=Torch_ImgDataloader(
                data_pairs["data_pairs"]["train"],
                backend=dl_backend,
                color_mode=img_format,
                dtype=dtype,
                transforms=v2_transforms.Compose(
                    [
                        v2_transforms.Resize(img_res),
                        v2_transforms.RandAugment(
                            num_ops=2,
                            magnitude=round(min((env_args["epoch"]) / (50 / 16), 30)),
                        ),
                    ]
                ),
            ),
            batch_size=train_batchsize,
            shuffle=True,
            num_workers=dataLoader_num_workers,
            persistent_workers=True,
            prefetch_factor=3,
            timeout=120,
            pin_memory=True,
            drop_last=True,
        )
        return train_dataloader

    # Make the model
    print("[bold green]Making the model...")
    from efficientnet_pytorch import EfficientNet

    model = EfficientNet.from_name(
        extra_args["model"],
        include_top=True,
        num_classes=data_pairs["num_classes"],
        in_channels=3 if img_format == "rgb" else 1,
    ).to(
        get_device()
    )  # Have to move the model to device before making the optimizer (if using mixed precision not doing this will cause error)

    # Make the optimizer
    optimizer_params = [
        {
            "params": [
                param
                for name, param in model.named_parameters()
                if all(
                    keyword not in name for keyword in ["bias", "bn", "mixing_ratio"]
                )
            ]
        },
        {
            "params": [
                param
                for name, param in model.named_parameters()
                if any(keyword in name for keyword in ["bias", "bn", "mixing_ratio"])
            ],
            "weight_decay": 0,
        },
    ]
    optimizer = TP_optim.GrokFastAdamW(
        optimizer_params,
        lr=0.01,
        weight_decay=0.05,
    )
    optimizer = TP_optim.Lookahead(optimizer, k=5, alpha=0.5, pullback_momentum="none")

    # Make the loss function
    loss_fn = nn.CrossEntropyLoss()

    # Train the model
    print("[bold green]Training the model...")
    fit(
        model,
        DynamicArg(gen_train_dataloader),
        DynamicArg(mode="static", default_value=eval_dataloader),
        optimizer,
        loss_fn,
        gradient_accumulation=bool(train_gradient_accumulation),
        gradient_accumulation_steps=DynamicArg(
            default_value=train_gradient_accumulation, mode="static"
        ),
        early_stopping_cnf={
            "patience": 8,
            "monitor": "Cohen's Kappa",
            "mode": "max",
            "min_delta": 0.00001,
        },
        opt_features = {
            "gradient normalization": extra_args["gradient_normalization"],
            "gradient centralization": extra_args["gradient_centralization"],
            "adaptive gradient clipping": [extra_args["agc"], 0.01],
        }
    )
