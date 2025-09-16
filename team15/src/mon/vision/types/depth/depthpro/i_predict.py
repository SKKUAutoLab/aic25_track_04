#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Depth Pro: Sharp Monocular Metric Depth in Less Than
a Second,".

References:
    - https://github.com/apple/ml-depth-pro
"""

import box
import matplotlib
import numpy as np
import torch.optim

import mon
import src.depth_pro as depth_pro

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
    config                      = depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT
    config.patch_encoder_preset = args.network.patch_encoder_preset
    config.image_encoder_preset = args.network.image_encoder_preset
    config.decoder_features     = args.network.decoder_features
    config.use_fov_head         = args.network.use_fov_head
    config.fov_encoder_preset   = args.network.fov_encoder_preset
    config.checkpoint_uri       = pretrained

    model, transform = depth_pro.create_model_and_transforms(config=config, device=device)
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
            path           = mon.Path(datapoint["meta"]["path"])
            image, _, f_px = depth_pro.load_rgb(str(path))
            image          = transform(image)
            timers.preprocess.tock()

            # Infer
            timers.infer.tick()
            outputs = model.infer(image, f_px=f_px)
            timers.infer.tock()
            
            # Postprocess
            timers.postprocess.tick()
            depth          = outputs["depth"]
            focallength_px = outputs["focallength_px"]
            depth          = depth.detach().cpu().numpy().squeeze()
            depth          = (depth - depth.min()) / (depth.max() - depth.min())
            depth_i        = 1.0 - depth
            depth_g        = (depth_i * 255).astype(np.uint8)
            depth_c        = (cmap(depth_i)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
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
