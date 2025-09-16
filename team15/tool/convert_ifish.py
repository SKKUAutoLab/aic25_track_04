#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Transform normal images with bboxes to fisheye images."""

import cv2
import numpy as np

import mon

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]
data_dir     = current_file.parents[1]
if data_dir.has_subdir("data"):
    data_dir = data_dir / "data"


# ----- Convert -----
def convert_ifish(
    data        : str,
    split       : str,
    extra_data  : str,
    distortion  : float = 1.0,
    area_thres  : int   = 32,
    aspect_thres: float = 0.1
):
    image_dir         = data_dir / data / split / extra_data / "image"
    label_dir         = data_dir / data / split / extra_data / "label"
    image_fisheye_dir = data_dir / data / split / extra_data / "image_fisheye"
    label_fisheye_dir = data_dir / data / split / extra_data / "label_fisheye"

    if not image_dir.exists():
        raise FileNotFoundError(f"[image_dir] does not exist: {image_dir}.")
    if not label_dir.exists():
        raise FileNotFoundError(f"[label_dir] does not exist: {label_dir}.")

    transform = mon.iFishTransform(distortion, area_thres, aspect_thres, p=1)

    # Process each image
    image_files = sorted([f for f in list(image_dir.rglob("*")) if f.is_image_file()])
    with mon.create_progress_bar() as pbar:
        for i, image_file in pbar.track(
            sequence    = enumerate(image_files),
            total       = len(image_files),
            description = f"[bright_yellow]Processing"
        ):
            # Read image
            image = cv2.imread(str(image_file))
            image = mon.resize(image, 1920, side="long", interpolation="bicubic")
            h, w  = mon.image_size(image)

            # Read YOLO label file
            label_file = label_dir / f"{image_file.stem}.txt"
            if not label_file.is_txt_file(exist=True):
                continue

            bs = mon.load_hbb(path=label_file, fmt=mon.BBoxFormat.YOLO, imgsz=(h, w))
            if len(bs) == 0:
                continue

            # Split image and bounding boxes
            sis, sbs = mon.split_image_and_hbbs(image, bs, 2)

            # Transform each sub-image and bounding box
            for j, (si, sb) in enumerate(zip(sis, sbs)):
                transformed   = transform(image=si, bboxes=sb)
                transformed_i = transformed["image"]
                transformed_b = transformed["bboxes"]

                image_fisheye_file = image_fisheye_dir / f"{image_file.stem}_fisheye_{j}.jpg"
                image_fisheye_file.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(image_fisheye_file), transformed_i)

                label_fisheye_file = label_fisheye_dir / f"{image_file.stem}_fisheye_{j}.txt"
                label_fisheye_file.parent.mkdir(parents=True, exist_ok=True)
                with open(label_fisheye_file, "w") as f:
                    for b in transformed_b:
                        f.write(f"{int(b[4])} {b[0]:.32f} {b[1]:.32f} {b[2]:.32f} {b[3]:.32f}\n")


# ----- Main -----
if __name__ == "__main__":
    convert_ifish("fisheye8k", "extra", "visdrone", distortion=0.5, area_thres=10, aspect_thres=0.3)
