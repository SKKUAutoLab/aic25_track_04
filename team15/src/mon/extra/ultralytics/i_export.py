#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the framework: "Ultralytics".

References:
    - https://github.com/ultralytics/ultralytics
"""

import box

import mon
from ultralytics import YOLO

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Export -----
def export(args: dict | box.Box) -> str:
    # Start
    mon.print_run_summary(args)

    # Pretrained
    pretrained = None
    if args.weights and args.weights.is_weights_file(exist=True):
        pretrained = args.weights
    if pretrained and pretrained.is_weights_file(exist=True):
        mon.console.log(f"Pretrained: {pretrained}.")
    else:
        raise ValueError(f"Invalid weights file: {pretrained}.")

    # Export
    model = YOLO(pretrained)
    model.info()
    model.export(format="onnx")

    # Finish
    return str(args.save_dir)


# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    export(args)


if __name__ == "__main__":
    main()
