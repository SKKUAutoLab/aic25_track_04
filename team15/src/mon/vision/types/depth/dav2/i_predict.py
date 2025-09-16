#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Depth Anything V2. A More Capable Foundation Model for
Monocular Depth Estimation," NeurIPS 2024.

References:
    - https://github.com/DepthAnything/Depth-Anything-V2
"""

import box
import matplotlib
import numpy as np
import torch
import torch.optim

import mon
from depth_anything_v2.dpt import DepthAnythingV2

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Utils -----
def benchmark(model: torch.nn.Module):
    flops, params = mon.compute_efficiency_score(model=model)
    mon.console.log(f"Params: {params:.4f}")
    mon.console.log(f"FLOPs : {flops:.4f}")


# ----- Predict -----
@torch.no_grad()
def predict(args: dict | box.Box) -> str:
    # Start
    mon.print_run_summary(args)

    # Device
    device = mon.set_device(args.device)

    # Seed
    mon.set_random_seed(args.seed)

    # Data I/O
    data_name, data_loader = mon.parse_data_loader(args.data, args.root, False, verbose=False)

    # Pretrained
    pretrained = args.resume
    if args.weights and args.weights.is_weights_file(exist=True):
        pretrained = args.weights
    if pretrained and pretrained.is_weights_file(exist=True):
        mon.console.log(f"Pretrained: {pretrained}.")
    else:
        raise ValueError(f"Invalid weights file: {pretrained}.")

    # Model
    model = DepthAnythingV2(
        encoder      = args.network.encoder,
        features     = args.network.features,
        out_channels = args.network.out_channels,
        device       = device
    )
    model.load_state_dict(torch.load(str(pretrained), map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    
    # Benchmark
    if args.benchmark:
        benchmark(model)
    
    # Predict
    cmap   = matplotlib.colormaps.get_cmap("Spectral_r")
    timers = mon.TimeProfiler()
    timers.total.tick()
    with mon.create_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(data_loader),
            total       = len(data_loader),
            description = f"[bright_yellow]Predicting"
        ):
            # Preprocess
            timers.preprocess.tick()
            path   = mon.Path(datapoint["meta"]["path"])
            image  = datapoint["image"]
            timers.preprocess.tock()

            # Infer
            timers.infer.tick()
            outputs = model.infer_image(image, args.imgsz[0])
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min()) * 255.0
            outputs = outputs.astype(np.uint8)
            depth_g = np.repeat(outputs[..., np.newaxis], 3, axis=-1)
            depth_c = (cmap(outputs)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            timers.postprocess.tock()

            # Save
            if args.save_image:
                out_dir  = mon.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.save_image(depth_g, out_path)

            if args.save_debug:
                out_dir  = mon.parse_output_dir(args.save_dir, data_name, mon.SAVE_DEBUG_DIR, path, args.keep_subdirs, args.save_nearby)
                if args.save_nearby:
                    out_dir = out_dir.parent / f"{out_dir.stem}_c"
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.save_image(depth_c, out_path)
    timers.total.tock()

    # Finish
    timers.print()
    return str(args.save_dir)


# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
