#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pathlib

from utils import export_trt


# ----- Main -----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-file",   type=str,                    help="Path to .onnx model file")
    parser.add_argument("--engine-file", type=str,                    help="Output path for the TensorRT engine file")
    parser.add_argument("--imgsz",       type=int, default=960,       help="Image size for preprocessing")
    parser.add_argument("--opset",       type=int, default=16,        help="ONNX opset version")
    parser.add_argument("--trt-p",       type=str, default="fp16n32", help="TRT floating point precision")
    parser.add_argument("--dla-core",    type=int, default=0,         help="TRT DLA core to use (-1 = no DLA)")
    parser.add_argument("--verbose",     action="store_true",         help="Verbosity")
    args = parser.parse_args()

    # Parse input arguments
    onnx_file   = pathlib.Path(args.onnx_file)
    engine_file = pathlib.Path(args.engine_file)
    imgsz       = args.imgsz
    opset       = args.opset
    trt_p       = args.trt_p
    dla_core    = args.dla_core
    verbose     = args.verbose

    _ = export_trt(
        onnx_file   = onnx_file,
        engine_file = engine_file,
        imgsz       = imgsz,
        opset       = opset,
        trt_p       = trt_p,
        dla_core    = dla_core,
        verbose     = verbose
    )


if __name__ == "__main__":
    main()
