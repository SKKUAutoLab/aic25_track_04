#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the framework: "Ultralytics".

References:
    - https://github.com/ultralytics/ultralytics
"""

import json
from datetime import datetime

import box
import torch

import mon
import ultralytics.engine.results
from ultralytics import YOLO
from ultralytics import settings

Results      = ultralytics.engine.results.Results
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Utils -----


# ----- Predict -----
@torch.no_grad()
def predict(args: dict | box.Box) -> str:
    # Start
    mon.print_run_summary(args)

    # Device
    device = [args.device] if isinstance(args.device, str | int) else args.device
    device = [int(d) for d in device]

    # Seed
    mon.set_random_seed(args.seed)
    
    # Data I/O
    data_name, data_loader = mon.parse_data_loader(args.data, args.root, False, verbose=False)
    # References: https://docs.ultralytics.com/quickstart/#modifying-settings
    settings.update({"datasets_dir": str(args.root)})

    # Pretrained
    pretrained = args.resume
    if args.weights and args.weights.is_weights_file(exist=True):
        pretrained = args.weights
    if pretrained and pretrained.is_weights_file(exist=True):
        mon.console.log(f"Pretrained: {pretrained}.")
    else:
        raise ValueError(f"Invalid weights file: {pretrained}.")

    # Model
    cfg      = args.cfg
    cfg.mode = "predict"
    if pretrained and pretrained.is_weights_file(exist=True):
        cfg.model = pretrained

    model = YOLO(cfg.model)

    # Predict
    # COCO JSON Format
    json_path   = args.save_dir / f"{data_name}.json"
    info        = {
        "year"        : f"{datetime.now().year}",
        "version"     : "1",
        "description" : f"{data_name} predictions",
        "contributor" : "Long H. Pham",
        "url"         : "",
        "date_created": f"{datetime.now()}"
    }
    licenses    = []
    categories  = []
    images      = []
    annotations = []
    ann_id      = 0

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
            h0, w0 = mon.image_size(image)
            timers.preprocess.tock()

            # Infer
            timers.infer.tick()
            outputs = model(
                source  = image,
                imgsz   = cfg.imgsz,
                save    = False,
                device  = device,
                verbose = False,
                conf    = cfg.conf_thres,
                iou     = cfg.iou_thres,
                classes = cfg.classes,
            )
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            outputs: Results = outputs[0]  # batch_size = 1
            timers.postprocess.tock()

            # Save
            if args.save_result:
                out_dir   = mon.parse_output_dir(args.save_dir, data_name, mon.SAVE_LABEL_DIR, path, args.keep_subdirs, args.save_nearby)
                json_path = out_dir.parent / f"{data_name}.json"

                # Append image
                images.append({"id": i, "file_name": path.name, "height": h0, "width": w0})

                # Append annotations
                if outputs.boxes is None or len(outputs.boxes) == 0:
                    continue
                labels = outputs.boxes.cls.cpu().numpy().astype(int)
                boxes  = outputs.boxes.xywh.cpu().numpy().astype(float)
                scores = outputs.boxes.conf.cpu().numpy().astype(float)
                masks  = outputs.masks
                obbs   = outputs.obbs
                if masks is not None:
                    masks = masks.xy.cpu().numpy().astype(float)
                if obbs is not None:
                    obbs = obbs.xywhr.cpu().numpy().astype(float)
                for j, (c, b, s) in enumerate(zip(labels, boxes, scores)):
                    b            = b.flatten().tolist()
                    segmentation = masks[j].flatten().tolist() if masks else []
                    obb          = obbs[j].flatten().tolist()  if obbs  else []
                    annotations.append({
                        "id"          : ann_id,
                        "image_id"    : i,
                        "category_id" : c,
                        "bbox"        : [b[0], b[1], b[2], b[3]],
                        "segmentation": [segmentation],
                        "obb"         : obb,
                        "area"        : float(b[2] * b[3]),
                        "score"       : s,
                        "iscrowd"     : 0,
                    })
                    ann_id += 1
    timers.total.tock()

    # Save
    if args.save_result:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        # Write to JSON file
        json_data = {
            "info"       : info,
            "licenses"   : licenses,
            "categories" : categories,
            "images"     : images,
            "annotations": annotations
        }
        with open(str(json_path), "w") as f:
            json.dump(json_data, f, indent=None)

    # Finish
    timers.print()
    return str(args.save_dir)


# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
