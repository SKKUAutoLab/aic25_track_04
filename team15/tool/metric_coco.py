#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import mon

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]
root_dir     = current_file.parents[1]


def measure_metric(input_json: str, target_json: str):
    assert  input_json and  mon.Path(input_json).is_json_file()
    assert target_json and mon.Path(target_json).is_json_file()

    coco_gt = COCO(str(target_json))
    coco_dt = coco_gt.loadRes(str(input_json))
    imgIds  = sorted(coco_gt.getImgIds())

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = imgIds
    # coco_eval.params.catIds = [0, 1, 2, 3, 4]  # class specified
    # coco_eval.params.maxDets[2] = len(imgIds)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print("----------------------------------------")
    print("AP    : ", coco_eval.stats[0])
    print("AP50  : ", coco_eval.stats[1])
    print("AP75  : ", coco_eval.stats[2])
    print("APs   : ", coco_eval.stats[3])
    print("APm   : ", coco_eval.stats[4])
    print("APl   : ", coco_eval.stats[5])
    print("AR@1  : ", coco_eval.stats[6])
    print("AR@10 : ", coco_eval.stats[7])
    print("AR@100: ", coco_eval.stats[8])
    print("ARs   : ", coco_eval.stats[9])
    print("ARm   : ", coco_eval.stats[10])
    print("ARl   : ", coco_eval.stats[11])
    print("F1    : ", coco_eval.stats[20])
    print("----------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="metric_coco")
    parser.add_argument("--input-json",  type=str, help="Input JSON file.")
    parser.add_argument("--target-json", type=str, help="Ground-truth JSON file.")
    args = parser.parse_args()
    measure_metric(**vars(args))
