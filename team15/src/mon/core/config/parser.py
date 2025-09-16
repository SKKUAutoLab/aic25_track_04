#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Parses config arguments and command line arguments."""

__all__ = [
    "parse_default_args",
    "parse_predict_args",
    "parse_train_args",
]

import argparse
import socket
import box
from mon.core import pathlib, type_extensions
from mon.core.config import utils
from mon.core.config.core import CLI_OPTIONS
from mon.core.device import parse_device


# ----- Parse Args -----
def parse_default_args(name: str = "main") -> dict | box.Box:
    """Parse default arguments."""
    parser = argparse.ArgumentParser(description=name)
    
    for opt_name, opt_params in CLI_OPTIONS.items():
        action      = opt_params.get("action",      "store")
        default     = opt_params.get("default",     None)
        opt_type    = opt_params.get("type",        None)
        choices     = opt_params.get("choices",     None)
        required    = opt_params.get("required",    False)
        help_text   = opt_params.get("help",        "")
        prompt_only = opt_params.get("prompt_only", False)  # Use in interactive CLI only, not parse_args
        
        if prompt_only:
            continue
        '''
        if opt_type == bool and default is None:
            default = False
        if action == "store_true" and default is None:
            default = False
        if action == "store_false" and default is None:
            default = True
        '''
        
        kwargs = {
            "action"  : action,
            "default" : default,
            "required": required,
            "help"    : help_text,
        }
        if action in ["store_true", "store_false"]:
            kwargs.pop("default")
            # kwargs["default"] = False if action == "store_true" else True
        if opt_type:
            kwargs["type"] = opt_type
        if choices:
            kwargs["choices"] = choices
        flag = f"--{opt_name.replace('_', '-')}"
        parser.add_argument(flag, **kwargs)

    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Additional arguments")
    return box.Box(vars(parser.parse_args()))


def parse_train_args(model_root: pathlib.Path = None, verbose: bool = False) -> dict | box.Box:
    """Parse arguments for training."""
    from mon.nn import parse_weights_file

    # Get args
    cli        = parse_default_args()
    cli.root   = pathlib.Path(cli.root) if cli.root else None
    cli.config = utils.parse_config_file(cli.config, cli.root, model_root=model_root, weights_path=cli.weights)
    args       = utils.load_config(cli.config, verbose=verbose)
    args       = type_extensions.merge_dicts(args, cli)  # Prioritize cli -> args

    # Parse arguments
    if args.save_dir in [None, ""]:
        if args.use_fullname:
            args.save_dir = pathlib.parse_save_dir(args.root/"run"/"train", args.arch, args.model, args.fullname)
            # args.save_dir = pathlib.parse_save_dir(root/"run"/"train", arch, fullname, None)
        else:
            args.save_dir = pathlib.parse_save_dir(args.root/"run"/"train", args.arch, args.model, args.data)
    else:
        args.save_dir = pathlib.Path(args.save_dir)
        # if str("run/train") not in str(args.save_dir):
        #     args.save_dir = pathlib.Path(f"run/train/{args.save_dir}")
        # if str(args.root) not in str(args.save_dir):
        #     args.save_dir = args.root / args.save_dir

    args.hostname = socket.gethostname().lower()
    args.weights  = parse_weights_file(args.root, args.weights)
    args.resume   = parse_weights_file(args.root, args.resume)
    args.tuning   = parse_weights_file(args.root, args.tuning)
    args.device   = parse_device(args.device)

    # Save config file
    if not args.exist_ok:
        pathlib.delete_dir(paths=args.save_dir)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    if args.config and args.config.is_config_file():
        pathlib.copy_file(src=args.config, dst=args.save_dir / f"{args.config.name}")

    return args


def parse_predict_args(model_root: pathlib.Path = None, verbose: bool = False) -> dict | box.Box:
    """Parse arguments for predicting."""
    from mon.nn import parse_weights_file
    from mon import vision

    # Get args
    cli        = parse_default_args()
    cli.root   = pathlib.Path(cli.root) if cli.root else None
    cli.config = utils.parse_config_file(cli.config, cli.root, model_root=model_root, weights_path=cli.weights)
    args       = utils.load_config(cli.config, verbose=verbose)
    args       = type_extensions.merge_dicts(args, cli)  # Prioritize cli -> args

    # Parse arguments
    if args.save_dir in [None, ""]:
        if args.use_fullname or args.save_nearby:
            args.save_dir = pathlib.parse_save_dir(args.root/"run"/"predict", args.arch, args.fullname, None)
        else:
            args.save_dir = pathlib.parse_save_dir(args.root/"run"/"predict", args.arch, args.model, args.data)
    else:
        args.save_dir = pathlib.Path(args.save_dir)
        # args.save_dir = args.save_dir.replace("run/train/", "")
        # if str("run/predict") not in str(args.save_dir):
        #     args.save_dir = pathlib.Path(f"run/predict/{args.save_dir}")
        # if str(args.root) not in str(args.save_dir):
        #     args.save_dir = args.root / args.save_dir

    args.hostname = socket.gethostname().lower()
    args.weights  = parse_weights_file(args.root, args.weights)
    args.resume   = parse_weights_file(args.root, args.resume)
    args.tuning   = parse_weights_file(args.root, args.tuning)
    args.device   = parse_device(args.device)
    args.imgsz    = vision.image_size(args.imgsz)

    # Save config file
    if not args.exist_ok:
        pathlib.delete_dir(paths=args.save_dir)
    if not args.save_nearby and (args.save_result or args.save_image or args.save_debug):
        args.save_dir.mkdir(parents=True, exist_ok=True)
        if args.config and args.config.is_config_file():
            pathlib.copy_file(src=args.config, dst=args.save_dir / f"{args.config.name}")

    return args
