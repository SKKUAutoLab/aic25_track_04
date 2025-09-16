#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json

import cv2

import mon

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]
data_dir     = current_file.parents[1]
if data_dir.has_subdir("data"):
    data_dir = data_dir / "data"


# ----- Utils -----
def get_image_id(filename: str) -> int:
    filename     = mon.Path(filename).stem
    scene_list   = ["M", "A", "E", "N"]
    camera_index = int(filename.split("_")[0].split("camera")[1])
    scene_index  = scene_list.index(filename.split("_")[1])
    frame_index  = int(filename.split("_")[2])
    image_id     = int(str(camera_index) + str(scene_index) + str(frame_index))
    return int(image_id)


# ----- Convert -----
def convert_label_to_coco(predict_dir: str | mon.Path):
    image_dir   = data_dir / "data" / "fisheye8k" / "test" / "image"
    predict_dir = data_dir / "data" / "fisheye8k" / "test" / predict_dir
    # predict_dir = mon.Path(predict_dir)
    json_file   = predict_dir.parent / f"{predict_dir.stem}.json"
    json_file.parent.mkdir(parents=True, exist_ok=True)

    assert mon.Path(image_dir).is_dir()
    # assert mon.Path(predict_dir).is_dir()

    annotations = []
    image_files = sorted([f for f in list(image_dir.rglob("*")) if f.is_image_file()])
    with mon.create_progress_bar() as pbar:
        for i, image_file in pbar.track(
            sequence    = enumerate(image_files),
            total       = len(image_files),
            description = f"[bright_yellow]Processing "
        ):
            # Append image
            image    = cv2.imread(str(image_file))
            h, w, _  = image.shape
            image_id = get_image_id(image_file.name)

            # Append annotations
            label_file = predict_dir / f"{image_file.stem}.txt"
            if not label_file.is_txt_file(exist=True):
                continue

            # Read the YOLO label file and convert bbox format
            bs = mon.load_hbb(path=label_file, fmt=mon.BBoxFormat.YOLO2COCO, imgsz=(h, w))
            if len(bs) == 0:
                continue

            # Append annotations
            for b in bs:
                if b[5] <= 0.8:
                    continue
                annotations.append({
                    "image_id"   : image_id,
                    "category_id": int(b[4]),
                    "bbox"       : [
                        round(float(b[0]), 32),
                        round(float(b[1]), 32),
                        round(float(b[2]), 32),
                        round(float(b[3]), 32)
                    ],
                    "score"      : float(b[5]),
                })

    # Write to JSON file
    with open(str(json_file), "w") as f:
        # fjson.dump(annotations, f, float_format=".32f", indent=None)
        json.dump(annotations, f, indent=None)


# ----- Main -----
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict-dir", type=str, required=True)
    args = parser.parse_args()

    convert_label_to_coco(args.predict_dir)
