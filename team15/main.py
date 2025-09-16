#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements main running pipeline."""

import os
import subprocess

import box

import menu_rich
import mon

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Train -----
def run_train(args: dict | box.Box):
    # Get args
    args.root = mon.Path(args.root)
    assert args.root.exists()
    
    # Parse arguments
    use_extra_model = mon.is_extra_model(args.model)
    model_root      = mon.parse_model_dir(args.arch, args.model)
    args.model      = mon.parse_model_name(args.model)
    args.fullname   = args.fullname if args.fullname not in [None, "None", ""] else mon.Path(args.config).stem
    args.config     = mon.parse_config_file(args.config, args.root, model_root=model_root, weights_path=args.weights)
    assert args.config not in [None, "None", ""]
    args.weights    = mon.to_str(args.weights, ",")

    kwargs, flags = {}, []
    kwargs |= {"--root"           : str(args.root)}
    kwargs |= {"--task"           : str(args.task)}
    kwargs |= {"--mode"           : args.mode}
    kwargs |= {"--arch"           : args.arch}
    kwargs |= {"--model"          : args.model}
    kwargs |= {"--config"         : args.config}
    # kwargs |= {"--data"           : args.data}
    kwargs |= {"--fullname"       : args.fullname}
    kwargs |= {"--save-dir"       : str(args.save_dir)}
    kwargs |= {"--weights"        : args.weights}
    kwargs |= {"--device"         : args.device}
    kwargs |= {"--seed"           : args.seed}
    # kwargs |= {"--imgsz"          : args.imgsz}
    kwargs |= {"--epochs"         : args.epochs}
    kwargs |= {"--batch-size"     : args.batch_size}
    flags  += ["--torchrun"]     if args.torchrun     else []
    flags  += ["--save-result"]  if args.save_result  else []
    flags  += ["--save-image"]   if args.save_image   else []
    flags  += ["--save-debug"]   if args.save_debug   else []
    flags  += ["--use-fullname"] if args.use_fullname else []
    flags  += ["--keep-subdirs"] if args.keep_subdirs else []
    flags  += ["--save-nearby"]  if args.save_nearby  else []
    flags  += ["--exist-ok"]     if args.exist_ok     else []
    flags  += ["--verbose"]      if args.verbose      else []

    # Parse script file
    python_call = ["python"]
    env         = {**os.environ}
    if use_extra_model:
        script_file = mon.EXTRA_MODELS[args.arch][args.model]["model_dir"] / "i_train.py"
        if args.torchrun:
            device_     = mon.parse_device(args.device)
            python_call = [
                "python", "-m", "torch.distributed.run",
                f"--nproc_per_node={len(device_)}",
                f"--master_port={args.master_port}",
                f"--master_addr={args.master_addr}",
            ]
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(device_)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(device_), **env}
    else:
        script_file = current_dir / "train.py"

    # Parse arguments
    args_call: list[str] = []
    for k, v in kwargs.items():
        if v is None:
            continue
        elif isinstance(v, list | tuple):
            args_call_ = [f"{k}={v_}" for v_ in v]
        else:
            args_call_ = [f"{k}={v}"]
        args_call += args_call_
    
    # Run training
    if script_file.is_py_file():
        print("\n")
        command = (
            python_call +
            [script_file] +
            args_call +
            flags
        )
        result = subprocess.run(command, cwd=current_dir, env=env)
        print(result)
    else:
        raise ValueError(f"Cannot find Python training script file at: {script_file}.")
    

# ----- Predict -----
def run_predict(args: dict | box.Box):
    # Get args
    args.root = mon.Path(args.root)
    assert args.root.exists()

    # Parse arguments
    use_extra_model = mon.is_extra_model(args.model)
    model_root      = mon.parse_model_dir(args.arch, args.model)
    args.model      = mon.parse_model_name(args.model)
    args.data       = mon.to_list(args.data)
    args.fullname   = args.fullname if args.fullname not in [None, "None", ""] else args.model
    args.config     = mon.parse_config_file(args.config, args.root, model_root=model_root, weights_path=args.weights)
    args.config     = args.config or ""
    args.weights    = mon.to_str(args.weights, ",")
    
    for d in args.data:
        kwargs, flags = {}, []
        kwargs |= {"--root"           : str(args.root)}
        kwargs |= {"--task"           : str(args.task)}
        kwargs |= {"--mode"           : args.mode}
        kwargs |= {"--arch"           : args.arch}
        kwargs |= {"--model"          : args.model}
        kwargs |= {"--config"         : args.config}
        kwargs |= {"--data"           : d}
        kwargs |= {"--fullname"       : args.fullname}
        kwargs |= {"--save-dir"       : str(args.save_dir)}
        kwargs |= {"--weights"        : args.weights}
        kwargs |= {"--device"         : args.device}
        kwargs |= {"--seed"           : args.seed}
        kwargs |= {"--imgsz"          : args.imgsz}
        flags  += ["--resize"]       if args.resize       else []
        flags  += ["--benchmark"]    if args.benchmark    else []
        flags  += ["--save-result"]  if args.save_result  else []
        flags  += ["--save-image"]   if args.save_image   else []
        flags  += ["--save-debug"]   if args.save_debug   else []
        flags  += ["--use-fullname"] if args.use_fullname else []
        flags  += ["--keep-subdirs"] if args.keep_subdirs else []
        flags  += ["--save-nearby"]  if args.save_nearby  else []
        flags  += ["--exist-ok"]     if args.exist_ok     else []
        flags  += ["--verbose"]      if args.verbose      else []

        # Parse script file
        if use_extra_model:
            script_file = mon.EXTRA_MODELS[args.arch][args.model]["model_dir"] / "i_predict.py"
            python_call = ["python"]
        else:
            script_file = current_dir / "predict.py"
            python_call = ["python"]
        
        # Parse arguments
        args_call: list[str] = []
        for k, v in kwargs.items():
            if v is None:
                continue
            elif isinstance(v, list | tuple):
                args_call_ = [f"{k}={v_}" for v_ in v]
            else:
                args_call_ = [f"{k}={v}"]
            args_call += args_call_
        
        # Run prediction
        if script_file.is_py_file():
            print("\n")
            command = (
                python_call +
                [script_file] +
                args_call +
                flags
            )
            result = subprocess.run(command, cwd=current_dir)
            print(result)
        else:
            raise ValueError(f"Cannot find Python predicting script file at: {script_file}.")
        

# ----- Main -----
def main():
    defaults = mon.parse_default_args("main")
    menu     = menu_rich.RunmlCLI(defaults)
    args     = menu.prompt_args()
    
    # Run
    if args.mode in ["train"]:
        run_train(args=args)
    elif args.mode in ["predict", "speed"]:
        run_predict(args=args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}.")
        

if __name__ == "__main__":
    main()
