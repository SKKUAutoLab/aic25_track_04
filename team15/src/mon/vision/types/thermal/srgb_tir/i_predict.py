#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Edge-guided Multi-domain RGB-to-TIR image Translation for
Training Vision Tasks with Challenging Labels," ICRA 2023.

References:
    - https://github.com/RPM-Robotics-Lab/sRGB-TIR
"""

import sys

import box
import torch
import torch.optim
from torch.autograd import Variable

import mon
from trainer import MUNIT_Trainer, UNIT_Trainer
from utils import get_config, pytorch03_to_pytorch04

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Utils -----
def benchmark(trainer: torch.nn.Module):
    flops_e, params_e = mon.compute_efficiency_score(model=trainer.gen_a)
    flops_d, params_d = mon.compute_efficiency_score(model=trainer.gen_b)
    flops  = flops_e + flops_d
    params = params_e + params_d
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
    data_name, data_loader = mon.parse_data_loader(args.data, args.root, True, verbose=False)

    # Pretrained
    pretrained = args.resume
    if args.weights and args.weights.is_weights_file(exist=True):
        pretrained = args.weights
    if pretrained and pretrained.is_weights_file(exist=True):
        mon.console.log(f"Pretrained: {pretrained}.")
    else:
        raise ValueError(f"Invalid weights file: {pretrained}.")

    # Model
    cfg_path = current_dir / "option" / args.cfg
    cfgs     = get_config(str(cfg_path))
    cfgs["vgg_model_path"] = args.save_dir
    if args.trainer == "MUNIT":
        style_dim   = cfgs["gen"]["style_dim"]
        style_fixed = Variable(torch.randn(args["num_style"], style_dim, 1, 1).to(device), volatile=False)
        trainer     = MUNIT_Trainer(cfgs)
    elif args.trainer == "UNIT":
        style_dim   = None
        style_fixed = None
        trainer     = UNIT_Trainer(cfgs)
    else:
        sys.exit("Only support MUNIT|UNIT")
    try:
        mon.console.log(f"Pretrained: {pretrained}.")
        state_dict = torch.load(str(pretrained))
        trainer.gen_a.load_state_dict(state_dict["a"])
        trainer.gen_b.load_state_dict(state_dict["b"])
    except:
        state_dict = pytorch03_to_pytorch04(torch.load(str(pretrained)), args.trainer)
        trainer.gen_a.load_state_dict(state_dict["a"])
        trainer.gen_b.load_state_dict(state_dict["b"])
    trainer = trainer.to(device)
    trainer.train()
    encode  = trainer.gen_a.encode if args.a2b else trainer.gen_b.encode  # encode function
    decode  = trainer.gen_b.decode if args.a2b else trainer.gen_a.decode  # decode function
    
    # Benchmark
    if args.benchmark:
        benchmark(trainer)
    
    # Predict
    timers = mon.TimeProfiler()
    timers.total.tick()
    with (mon.create_progress_bar() as pbar):
        for i, datapoint in pbar.track(
            sequence    = enumerate(data_loader),
            total       = len(data_loader),
            description = f"[bright_yellow]Predicting"
        ):
            # Preprocess
            timers.preprocess.tick()
            path   = mon.Path(datapoint["meta"]["path"])
            image  = datapoint["image"]
            h0, w0 = mon.image_size(image)
            if args.resize and h0 != args.imgsz[0] and w0 != args.imgsz[1]:
                image = mon.resize(image, size=args.imgsz)
            image  = image.to(device)
            timers.preprocess.tock()
            
            # Infer
            timers.infer.tick()
            if args.trainer == "MUNIT":
                content, _ = encode(image)
                if args.synchronized:
                    style = style_fixed
                else:
                    style = Variable(torch.randn(args.num_style, style_dim, 1, 1).to(device), volatile=False)
                for j in range(args.num_style):
                    s       = style[j].unsqueeze(0)
                    outputs = decode(content, s)
                    outputs = (outputs + 1) / 2.0
            elif args.trainer == "UNIT":
                content, _ = encode(image)
                outputs    = decode(content)
                outputs    = (outputs + 1) / 2.0
            else:
                sys.exit("Only support MUNIT|UNIT")
            timers.infer.tock()
            
            # Postprocess
            timers.postprocess.tick()
            if args.resize and h0 != args.imgsz[0] and w0 != args.imgsz[1]:
                outputs = mon.resize(outputs, (h0, w0))
            timers.postprocess.tock()

            # Save
            if args.save_image:
                out_dir  = mon.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.save_image(outputs, out_path)
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
