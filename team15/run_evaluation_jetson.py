#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import pathlib
import time

import cv2
import torch

from utils import changeId, get_model, postprocess_result, preprocess_image

conf_thres = {
    #     bus,  bike, car,  pedestrian, truck
    "M": [0.10, 0.10, 0.10, 0.10, 0.10],
    "A": [0.10, 0.10, 0.10, 0.10, 0.10],
    "E": [0.10, 0.10, 0.10, 0.10, 0.10],
    "N": [0.10, 0.10, 0.10, 0.10, 0.10],
}
"""
conf_thres = {
    #     bus,  bike, car,  pedestrian, truck
    "M": [0.40, 0.40, 0.40, 0.40, 0.40],
    "A": [0.40, 0.40, 0.40, 0.40, 0.40],
    "E": [0.40, 0.40, 0.40, 0.40, 0.40],
    "N": [0.40, 0.40, 0.40, 0.40, 0.40],
}
"""


# ----- Main -----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-folder",       type=str,   default="/data/Fisheye1K_eval/images",                 help="Path to image folder")
    parser.add_argument("--output-json",        type=str,   default="/data/Fisheye1K_eval/predictions.json",       help="Output JSON file for predictions")
    parser.add_argument("--ground-truths-path", type=str,   default="/data/Fisheye1K_eval/groundtruth.json",       help="Path to ground truths JSON file")
    parser.add_argument("--model-path",         type=str,   default="/models/deim_dfine_s_960_cv2_fp16n32.engine", help="Path to the model")
    parser.add_argument("--imgsz",              type=int,   default=960,                                           help="Image size for preprocessing")
    parser.add_argument("--max-fps",            type=float, default=25.0,                                          help="Maximum FPS for evaluation")
    parser.add_argument("--device",             type=str,   default="cuda",                                        help="Device to run the model on (e.g., 'cuda' or 'cpu')")
    args = parser.parse_args()

    # Parse input arguments
    image_folder = args.image_folder
    output_json  = pathlib.Path(args.output_json)
    imgsz        = args.imgsz
    device       = torch.device(args.device) if args.device != "cpu" else "cpu"
    max_fps      = float(args.max_fps)

    # Prepare model
    model = get_model(args.model_path, imgsz=imgsz, device=device)
    model.warmup(1000)  # Warm up the model
    print(f"Model warmup completed.")

    # List images
    image_files = sorted([
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    print(f"Found {len(image_files)} images.")

    # Predict
    print("Prediction started")
    predictions      = []
    preprocess_time  = 0
    inference_time   = 0
    postprocess_time = 0
    total_time       = 0
    start_time       = time.time()
    for image_path in image_files:
        img = cv2.imread(image_path)  # BGR
        if img is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            continue

        t0      = time.time()
        img     = preprocess_image(image=img, imgsz=imgsz, device=device)
        t1      = time.time()
        with torch.no_grad():
            results = model(img)
        t2      = time.time()
        results = postprocess_result(image_path, results, conf_thres)  # [boxes, classes, scores]
        predictions.append((image_path, results))
        t3      = time.time()

        # print(f"Processed {os.path.basename(image_path)}: {len(results[0])} objects detected.")
        preprocess_time  += (t1 - t0)
        inference_time   += (t2 - t1)
        postprocess_time += (t3 - t2)
        total_time       += (t3 - t0)
    end_time     = time.time()
    elapsed_time = end_time - start_time
    print(f"Processed {len(image_files)} images in {elapsed_time:.2f} seconds.")
    print(f"Avg Image Preprocess Time: {preprocess_time / len(image_files) * 1000:.2f} ms")
    print(f"Avg Inference Time: {inference_time / len(image_files) * 1000:.2f} ms")
    print(f"Avg Postprocessing Time: {postprocess_time / len(image_files) * 1000:.2f} ms")
    print(f"Avg Processing Time: {total_time / len(image_files) * 1000:.2f} ms")

    # Prepare predictions for JSON output
    predictions_json = []
    for image_path, results in predictions:
        image_name = pathlib.Path(image_path).stem
        boxes, classes, scores = results
        # Add predictions to JSON format
        for b, c, s in zip(boxes, classes, scores):
            predictions_json.append({
                "image_id"   : changeId(image_name),
                "category_id": int(c),
                "bbox"       : [
                    round(float(b[0]), 32),
                    round(float(b[1]), 32),
                    round(float(b[2]), 32),
                    round(float(b[3]), 32)
                ],
                "score"      : float(s),
            })

    # Save predictions to JSON
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(predictions_json, f, indent=None)
        # json_data = {"predictions": predictions_json}
        # fjson.dump(json_data, f, float_format=".32f", indent=None)

    fps     = len(image_files)  / total_time
    normfps = min(fps, max_fps) / max_fps

    # f1 = f1_score(args.output_json, args.ground_truths_path)
    # harmonic_mean = 2 * f1 * normfps / (f1 + normfps)

    print(f"\n--- Evaluation Complete ---")
    print(f"Total inference time: {elapsed_time:.2f} seconds")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"FPS: {fps:.2f}")
    print(f"Normalized FPS: {normfps:.4f}")
    # print(f"F1-score: {f1:.4f}")
    # print(f"Metric (harmonic mean of F1-score and NormalizedFPS): {harmonic_mean:.4f}")


if __name__ == "__main__":
    main()
