#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "DEIM: DETR with Improved Matching for Fast
Convergence," CVPR 2025.

References:
    - https://github.com/ShihuaHuang95/DEIM
"""

import os
import sys

import box
import torch

import mon

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from engine.core import YAMLConfig

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Utils -----
class Model(torch.nn.Module):

    def __init__(self, cfg, export_postprocessor: bool = True):
        super().__init__()
        self.model = cfg.model.deploy()
        if export_postprocessor:
            self.postprocessor = cfg.postprocessor.deploy()
        else:
            self.postprocessor = None

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        if self.postprocessor is not None:
            outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs


class ModelEnsemble(torch.nn.Module):

    def __init__(self, cfg: list, export_postprocessor: bool = True):
        super().__init__()
        if not isinstance(cfg, list | tuple):
            raise TypeError(f"[cfg] must be a list or tuple of configurations, got {type(cfg)}.")

        self.models = torch.nn.ModuleList([c.model.deploy() for c in cfg])
        if export_postprocessor:
            self.postprocessor = cfg[0].postprocessor.deploy()
        else:
            self.postprocessor = None

    def forward(self, images, orig_target_sizes):
        outputs = {
            "pred_logits": [],
            "pred_boxes" : [],
        }
        for model in self.models:
            y = model(images)
            outputs["pred_logits"].append(y["pred_logits"])
            outputs["pred_boxes"].append(y["pred_boxes"])

        outputs["pred_logits"] = torch.stack(outputs["pred_logits"]).mean(0)    # Mean ensemble
        outputs["pred_boxes"]  = torch.stack(outputs["pred_boxes"]).mean(0)     # Mean ensemble
        if self.postprocessor is not None:
            outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs


# ----- Export -----
@torch.no_grad()
def export_onnx(model: Model, path: mon.Path, args: dict | box.Box) -> mon.Path:
    opset    = args.opset
    simplify = args.simplify
    imgsz    = args.imgsz[0] if isinstance(args.imgsz, list | tuple) else args.imgsz
    data     = torch.rand(32, 3, imgsz, imgsz)
    size     = torch.tensor([[imgsz, imgsz]])
    _        = model(data, size)
    dynamic_axes = {
        "images"           : {0: "N"},
        "orig_target_sizes": {0: "N"}
    }

    if args.get("export_postprocessor", True):
        output_names = ["labels", "boxes", "scores"]
    else:
        output_names = ["outputs"]

    torch.onnx.export(
        model,
        (data, size),
        path,
        input_names         = ["images", "orig_target_sizes"],
        output_names        = output_names,
        dynamic_axes        = dynamic_axes,
        opset_version       = opset,
        verbose             = False,
        do_constant_folding = True,
    )

    check = True
    if check:
        import onnx
        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)
        mon.console.log("Check export onnx model done...")

    if simplify:
        import onnx
        import onnxsim
        dynamic = True
        # input_shapes = {'images': [1, 3, 640, 640], 'orig_target_sizes': [1, 2]} if dynamic else None
        input_shapes = {"images": data.shape, "orig_target_sizes": size.shape} if dynamic else None
        onnx_model_simplify, check = onnxsim.simplify(path, test_input_shapes=input_shapes)
        onnx.save(onnx_model_simplify, path)
        print(f"Simplify onnx model {check}...")


@torch.no_grad()
def export(args: dict | box.Box) -> str:
    # Start
    mon.print_run_summary(args)

    # Device
    device = mon.set_device(args.device)

    # Seed
    mon.set_random_seed(args.seed)

    # Pretrained
    pretrained = args.resume
    if args.weights and (isinstance(args.weights, list | tuple) or args.weights.is_weights_file(exist=True)):
        pretrained = args.weights
    if pretrained and (isinstance(pretrained, list | tuple) or pretrained.is_weights_file(exist=True)):
        mon.console.log(f"Pretrained: {pretrained}.")
    else:
        raise ValueError(f"Invalid weights file: {pretrained}.")
    if not isinstance(pretrained, list | tuple):
        pretrained = [pretrained]

    # Model
    cfg         = args.cfg
    updated_cfg = args.updated_cfg
    if not isinstance(args.cfg, list | tuple):
        cfg         = [args.cfg]
        updated_cfg = [args.updated_cfg]
    if len(cfg) != len(pretrained):
        raise ValueError(f"Number of configurations ({len(cfg)}) does not match number of pretrained weights ({len(pretrained)}).")

    for i in range(len(cfg)):
        cfg_path        = current_dir / "option" / cfg[i]
        updated_cfg[i] |= {"resume": str(pretrained[i])} if pretrained[i] else {}
        updated_cfg[i] |= {
            "device": device,
            "seed"  : args.seed,
        }
        cfg[i] = YAMLConfig(cfg_path=str(cfg_path), root=str(args.root), **updated_cfg[i])

        if "HGNetv2" in cfg[i].yaml_cfg:
            cfg[i].yaml_cfg["HGNetv2"]["pretrained"] = False

        if pretrained[i]:
            checkpoint = torch.load(pretrained[i], map_location="cpu")
            if "ema" in checkpoint:
                state = checkpoint["ema"]["module"]
            else:
                state = checkpoint["model"]
        else:
            raise AttributeError("Only support resume to load model.state_dict by now.")

        # Load train mode state and convert to deploy mode
        cfg[i].model.load_state_dict(state)

    if len(cfg) == 1:  # Single model
        model = Model(cfg[0], export_postprocessor=args.export_postprocessor)
    else:
        model = ModelEnsemble(cfg, export_postprocessor=args.export_postprocessor)
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Export ONNX model (always export ONNX first)
    # save_dir  = pretrained.parent if args.save_nearby  else args.save_dir
    # file_stem = args.fullname     if args.use_fullname else pretrained.stem
    save_dir  = args.save_dir
    file_stem = args.fullname
    onnx_file = save_dir / f"{file_stem}.onnx"
    export_onnx(model, onnx_file, args)
    mon.console.log(f"Exported ONNX model to: {onnx_file}.")


# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    export(args)


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", "-c", type=str, default="options/deim_dfine/dfine_hgnetv2_l_coco80.yml")
    # parser.add_argument("--resume", "-r", type=str)
    # parser.add_argument("--check",        action="store_true", default=True)
    # parser.add_argument("--simplify",     action="store_true", default=True)
    # args = parser.parse_args()
    # main(args)
    main()
