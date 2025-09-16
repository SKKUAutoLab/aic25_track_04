#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Interface for training."""

import box
import torch

import mon

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Train -----
def train(args: dict | box.Box) -> str:
    # Start
    mon.print_run_summary(args)

    # Device
    device = mon.set_device(args.device)

    # Seed
    mon.set_random_seed(args.seed)

    # Data I/O
    # data_name, data_loader = mon.parse_data_loader(args.data, args.root, True, verbose=False)
    args["datamodule"] |= {
        "root"   : mon.parse_data_dir(args.root, args.datamodule.get("root", "")),
        "devices": device,
    }
    datamodule: mon.DataModule = mon.DATAMODULES.build(config=args.datamodule)
    datamodule.prepare_data()
    datamodule.setup(stage="train")
    train_dataloader = datamodule.train_dataloader
    val_dataloader   = datamodule.val_dataloader

    # Pretrained
    pretrained = args.tuning
    if args.resume and args.resume.is_weights_file(exist=True):
        pretrained = args.resume
    if args.weights and args.weights.is_weights_file(exist=True):
        pretrained = args.weights
    if pretrained and pretrained.is_weights_file(exist=True):
        mon.console.log(f"Pretrained: {pretrained}.")
    else:
        mon.console.log(f"Pretrained: {None}, training from scratch.")

    # Model
    model: torch.nn.Module = None
    if pretrained and pretrained.is_weights_file(exist=True):
        model.load_state_dict(torch.load(pretrained, weights_only=True))
    model = model.to(device)
    model.train()

    # Trainer
    # Optimizer
    # Scheduler
    # Loss

    # Train
    with mon.create_progress_bar() as pbar:
        for _ in pbar.track(
            sequence    = range(args.epochs),
            total       = args.epochs,
            description = f"[bright_yellow]Training"
        ):
            pass

    # Finish
    return str(args.save_dir)


# ----- Main -----
def main() -> str:
    args = mon.parse_train_args(model_root=current_dir, verbose=False)
    train(args)


if __name__ == "__main__":
    main()
