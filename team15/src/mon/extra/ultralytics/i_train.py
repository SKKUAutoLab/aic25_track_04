#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the framework: "Ultralytics".

References:
    - https://github.com/ultralytics/ultralytics
"""

import box

import mon
from ultralytics import YOLO
from ultralytics import settings

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Train -----
def train(args: dict | box.Box) -> str:
    # Start
    mon.print_run_summary(args)

    # Device
    device = [args.device] if isinstance(args.device, str | int) else args.device
    device = [int(d) for d in device]

    # Seed
    mon.set_random_seed(args.seed)

    # Data I/O
    # References: https://docs.ultralytics.com/quickstart/#modifying-settings
    settings.update({"datasets_dir": str(args.root)})

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
    cfg      = args.cfg
    cfg.mode = "train"
    if pretrained and pretrained.is_weights_file(exist=True):
        cfg.model = pretrained
    if not mon.Path(cfg.data).is_yaml_file():
        cfg.data = str(current_dir / "ultralytics" / "cfg" / "datasets" / f"{cfg.data}")
    cfg.epochs   = args.epochs
    cfg.batch    = args.batch_size
    cfg.device   = device
    cfg.project  = str(args.save_dir.parent)
    cfg.name     = str(args.save_dir.name)
    cfg.exist_ok = args.exist_ok
    cfg.verbose  = args.verbose
    cfg.seed     = args.seed
    cfg.save_txt = False  # Disable saving val/test results to txt files (too many files for indexing)

    model = YOLO(cfg.model)
    model.info()

    # Train
    _ = model.train(**cfg)

    # Finish
    return str(args.save_dir)


# ----- Main -----
def main() -> str:
    args = mon.parse_train_args(model_root=current_dir)
    train(args)


if __name__ == "__main__":
    main()
