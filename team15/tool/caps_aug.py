#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""CAPS: Cut-And-Paste Augmentation for Static Camera."""

import copy
import random
from typing import Any

import cv2
import numpy as np
import torch

import mon
import ultralytics

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]
data_dir     = current_file.parents[1]
if data_dir.has_subdir("data"):
    data_dir = data_dir / "data"


# ----- Utils -----
def group_image_and_label(data: str, split: str):
    """Group images and labels by their file names."""
    image_dir = data_dir / data / split / "image"
    label_dir = data_dir / data / split / "label"

    if not image_dir.exists():
        raise FileNotFoundError(f"[image_dir] does not exist: {image_dir}.")
    if not label_dir.exists():
        raise FileNotFoundError(f"[label_dir] does not exist: {label_dir}.")

    image_files = sorted([f for f in list(image_dir.rglob("*")) if f.is_image_file()])
    with mon.create_progress_bar() as pbar:
        for i, image_file in pbar.track(
            sequence    = enumerate(image_files),
            total       = len(image_files),
            description = f"[bright_yellow]Grouping"
        ):
            stem   = image_file.stem
            parts  = stem.split("_")[0:2]  # Exclude the last part (e.g., "_01")
            camera = "_".join(parts)       # Join the remaining parts

            new_image_file = image_dir.parent / camera / "image" / image_file.name
            new_image_file.parent.mkdir(parents=True, exist_ok=True)
            mon.copy_file(image_file, new_image_file)

            label_file     = label_dir / f"{stem}.txt"
            new_label_file = label_dir.parent / camera / "label" / label_file.name
            mon.copy_file(label_file, new_label_file)


def concat_image_and_label(data: str, split: str):
    """Concatenate images and labels into a single directory."""
    parent_dir = data_dir / data / split
    image_dir  = parent_dir / "image"
    label_dir  = parent_dir / "label"

    if not parent_dir.exists():
        raise FileNotFoundError(f"[parent_dir] does not exist: {parent_dir}.")

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for subdir in sorted(parent_dir.subdirs()):
        if subdir.stem == "image" or subdir.stem == "label":
            continue

        sub_image_dir = parent_dir / subdir.stem / "image_caps"
        sub_label_dir = parent_dir / subdir.stem / "label_caps"

        if not sub_image_dir.exists():
            mon.error_console.log(f"[sub_image_dir] does not exist: {sub_image_dir}.")
            continue
        if not sub_label_dir.exists():
            mon.error_console.log(f"[sub_label_dir] does not exist: {sub_label_dir}.")
            continue

        image_files = sorted([f for f in list(sub_image_dir.rglob("*")) if f.is_image_file()])
        with mon.create_progress_bar() as pbar:
            for i, image_file in pbar.track(
                sequence    = enumerate(image_files),
                total       = len(image_files),
                description = f"[bright_yellow]Concatenating"
            ):
                new_image_file = image_dir / f"{image_file.stem}.jpg"
                new_image_file.parent.mkdir(parents=True, exist_ok=True)
                mon.copy_file(image_file, new_image_file)

                label_file     = sub_label_dir / f"{image_file.stem}.txt"
                new_label_file = label_dir / f"{image_file.stem}.txt"
                new_label_file.parent.mkdir(parents=True, exist_ok=True)
                mon.copy_file(label_file, new_label_file)


# ----- Augmentation -----
class Label:
    """Candidate object for ``CAPS``.

    Args:
        bbox: Bounding box in XYXY format.
        mask: Instance segmentation mask for the object.
        image_path: Path to the image file containing the object.
    """

    def __init__(self, bbox: np.ndarray, mask: np.ndarray, image_path: mon.Path):
        if len(bbox) < 5:
            raise ValueError(f"[bbox] must have at least 5 elements, got {len(bbox)}.")

        self.bbox       = bbox
        self.mask       = mask
        self.image_path = image_path
        self.class_id   = int(bbox[4])
        self.used       = 0


# noinspection PyMethodMayBeStatic
class CutAndPasteStaticAugmentation:
    """CAPS: Cut-And-Paste Augmentation for Static/Surveillance Camera.

    Attributes:
        _suffix: Suffix for the new image and label names.

    Args:
        image_dir: Directory containing the images.
        label_dir: Directory containing the YOLO-format labels.
        num_classes: Number of classes in the dataset.
        sam_model: Path to the SAM model file for auto-annotation (Ultralytics).
        ratio: Ratio of objects to sample per class. Default is ``None``, which
            means all objects will be sampled.
        iou_thres: IoU threshold for filtering candidate labels. Default is 0.0.
        max_tries: Maximum number of tries to find a valid candidate. Default is 10.
        style_transfer: Whether to apply style transfer to the pasted objects.
            Default is ``False``.
        device: Device to run the SAM model on. Default is ``torch.device("cuda")``.
        verbose: Verbosity mode. Default is ``True``.
    """

    _suffix = "caps"  # Suffix for the new image and label files

    def __init__(
        self,
        image_dir     : mon.Path,
        label_dir     : mon.Path,
        num_classes   : int,
        sam_model     : str          = "sam2.1_l.pt",
        ratio         : Any          = None,
        iou_thres     : float        = 0.0,
        max_tries     : int          = 10,
        style_transfer: bool         = False,
        device        : torch.device = torch.device("cuda"),
        verbose       : bool         = True
    ):
        if not image_dir.exists():
            raise FileNotFoundError(f"[image_dir] does not exist: {image_dir}.")
        if not label_dir.exists():
            raise FileNotFoundError(f"[label_dir] does not exist: {label_dir}.")

        mon.console.rule(f"[bold red]{image_dir.parent.stem}")

        # Assign attributes
        self._image_dir     = image_dir
        self._label_dir     = label_dir
        self.new_image_dir  = image_dir.parent / f"{image_dir.stem}_{self._suffix}"
        self.new_label_dir  = label_dir.parent / f"{label_dir.stem}_{self._suffix}"
        self._num_classes   = num_classes
        self.sam_model      = sam_model
        self.iou_thres      = iou_thres
        self.max_tries      = max_tries
        self.style_transfer = style_transfer
        self.device         = device
        self.verbose        = verbose
        self._run           = 0  # Number of times this augmentation has been run (for naming output files)

        # Initialize attributes
        labels, candidates = self._load_data()
        self._all_labels   = copy.deepcopy(labels)
        self._labels       = copy.deepcopy(labels)
        self._candidates   = self._group_obj_per_class(candidates)
        self._counts       = self._count_obj_per_class(self.all_labels)
        self.ratio         = ratio

        if self.verbose:
            mon.console.log(f"Subdir              : {image_dir.parent.stem}")
            mon.console.log(f"Number of objects   : {len(self.labels)}")
            mon.console.log(f"Number of candidates: {len(candidates)}")
            for k, v in self.candidates.items():
                mon.console.log(f"  |_ Class {k}: {len(v):<10}")

    # ----- Properties -----
    @property
    def sam_model(self) -> ultralytics.SAM:
        """Return the SAM model."""
        return self._sam_model

    @sam_model.setter
    def sam_model(self, model: Any):
        """Set the SAM model."""
        if isinstance(model, str | mon.Path):
            self._sam_model = ultralytics.SAM(model)
        elif isinstance(model, ultralytics.SAM):
            self._sam_model = model
        else:
            raise TypeError(f"[sam_model] must be a string or an instance of ultralytics.SAM, got {type(model)}.")

    @property
    def ratio(self) -> dict[int, float]:
        """Return the ratio of objects to sample per class."""
        return self._ratio

    @ratio.setter
    def ratio(self, ratio: Any):
        """Set the ratio of objects to sample per class."""
        if ratio is None:
            ratio = [1.0] * self._num_classes
        elif isinstance(ratio, int | float):
            ratio = [ratio] * self._num_classes
        elif ratio in ["even", "uniform"]:
            ratio = [1.0 / self._num_classes] * self._num_classes

        if len(ratio) != self._num_classes:
            raise ValueError(f"[ratio] must have the same length as the number of classes, got {len(ratio)} != {self._num_classes}.")
        self._ratio = {k: ratio[int(k)] for k, v in self.counts.items()}

    @property
    def ratio_per_class(self) -> dict[int, float]:
        """Calculate the percentage of each class."""
        counts      = self.counts
        total_count = sum(counts.values())
        ratios      = {k: float(v / total_count) for k, v in counts.items()}
        return ratios

    @property
    def num_samples_per_class(self) -> dict[int, int]:
        """Calculate the number of objects to sample per class."""
        ratio         = self.ratio
        curr_ratio    = self.ratio_per_class
        sample_ratios = {k: float(ratio[k] - curr_ratio[k]) for k, _ in ratio.items()}
        num_samples   = {}
        for k, v in sample_ratios.items():
            if k in self.candidates and v > 0:
                num_samples[k] = int(v * len(self.candidates[k]))
            else:
                num_samples[k] = 0
        return num_samples

    @property
    def all_labels(self) -> list[Label]:
        """Return all labels."""
        return self._all_labels

    @property
    def labels(self) -> list[Label]:
        """Return the labels for the current run."""
        return self._labels

    @property
    def candidates(self) -> dict[int, list[Label]]:
        """Return the candidates grouped by class."""
        return self._candidates

    @property
    def counts(self) -> dict[int, int]:
        """Return the counts of objects per class."""
        return self._counts

    # ----- Initialize -----
    def _load_data(self) -> tuple[list[Label], list[Label]]:
        """Load all labels from the label directory."""
        labels      = []
        candidates  = []
        image_files = sorted([f for f in list(self._image_dir.rglob("*")) if f.is_image_file()])
        with mon.create_progress_bar() as pbar:
            for i, image_file in pbar.track(
                sequence    = enumerate(image_files),
                total       = len(image_files),
                description = f"[bright_yellow]Loading"
            ):
                # Read image
                image = cv2.imread(str(image_file))
                h, w, = mon.image_size(image)

                # Read YOLO label file
                label_file = self._label_dir / f"{image_file.stem}.txt"
                if not label_file.is_txt_file(exist=True):
                    continue

                bs = mon.load_hbb(path=label_file, fmt=mon.BBoxFormat.CXCYWHN2XYXY, imgsz=(h, w))
                for b in bs:
                    labels.append(Label(b, None, image_file))

                # Determine candidates objects
                filtered_bs = mon.hbb_filter_iou(bs, iou_thres=self.iou_thres)
                masks       = self._generate_instance_masks(image, filtered_bs)
                for b, m in zip(filtered_bs, masks):
                    if m is None:  # Skip if no mask is found
                        continue
                    candidates.append(Label(b, m, image_file))

        return labels, candidates

    def _generate_instance_masks(self, image: np.ndarray, bbox: np.ndarray) -> list[np.ndarray]:
        sam_results = self.sam_model(image, bboxes=bbox[:, 0:4], device=torch.device("cuda"), verbose=False)
        sam_results = sam_results[0]

        semantic_mask = np.zeros(image.shape, dtype=np.uint8)
        if sam_results.masks is not None:
            for m in sam_results.masks.xy:
                semantic_mask = cv2.fillPoly(semantic_mask, [np.array(m, dtype=np.int32)], (1, 1, 1))

        # Dilate the mask to ensure it covers the object
        # kernel        = np.ones((3, 3), np.uint8)
        # semantic_mask = cv2.dilate(semantic_mask, kernel, iterations=1)

        masks = []
        for b in bbox:
            x1, y1, x2, y2 = b[0:4].astype(int)
            area  = (x2 - x1) * (y2 - y1)
            m     = semantic_mask[y1:y2, x1:x2]
            count = np.count_nonzero(m) / 3.0  # Count non-zero pixels in a single channel
            if count < float(area * 0.6):      # Skip if the mask is too small
                masks.append(None)
            else:
                masks.append(m)

        return masks

    # ----- Sampling -----
    def process(self):
        # Copy and paste objects from candidate to each image
        image_files = sorted([f for f in list(self._image_dir.rglob("*")) if f.is_image_file()])
        with mon.create_progress_bar() as pbar:
            for i, image_file in pbar.track(
                sequence    = enumerate(image_files),
                total       = len(image_files),
                description = f"[bright_yellow]Processing"
            ):
                labels, new_labels = self._get_new_samples(image_file)
                if len(new_labels) == 0:
                    continue

                # Update
                self.all_labels.extend(new_labels)
                self._counts = self._count_obj_per_class(self.all_labels)

                # Copy and paste objects to the image
                image = cv2.imread(str(image_file))
                h, w  = mon.image_size(image)
                image = self._copy_and_paste_labels(image, new_labels)

                new_image_file = self.new_image_dir / f"{image_file.stem}_{self._suffix}_{self._run}.jpg"
                new_image_file.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(new_image_file), image)

                # Save new label file
                bs = [l.bbox for l in labels]
                bs = mon.convert_hbb(bs, fmt=mon.BBoxFormat.XYXY2CXCYWHN, imgsz=(h, w))
                new_label_file = self.new_label_dir / f"{image_file.stem}_{self._suffix}_{self._run}.txt"
                new_label_file.parent.mkdir(parents=True, exist_ok=True)
                with open(new_label_file, "w") as f:
                    for b in bs:
                        f.write(f"{int(b[4])} {b[0]:.32f} {b[1]:.32f} {b[2]:.32f} {b[3]:.32f}\n")

        self._run += 1

    def _get_new_samples(self, image_file: mon.Path) -> tuple[list[Label], list[Label]]:
        """Get new samples for the given image file."""
        # Get all Labels for this image
        labels     = [l for l in self.labels if l.image_path == image_file]
        new_labels = []

        # Determine number of samples to paste
        samples_per_class = self.num_samples_per_class
        # if self.verbose:
            # mon.console.log(f"Objects per class: {self.counts} | Samples per class: {samples_per_class}")

        samples_per_class = dict(sorted(samples_per_class.items(), key=lambda item: item[1], reverse=True))
        for class_id, num_sample in samples_per_class.items():
            candidates = self.candidates[class_id]
            # min_used   = min([b.used for b in candidates])
            num_sample = 0
            tries      = 0

            while num_sample < samples_per_class[class_id]:
                # Stop if too many tries (e.g., no more candidates)
                if tries >= self.max_tries:
                    break

                # Randomly select a candidate
                choice    = random.randint(0, len(candidates) - 1)
                new_label = candidates[choice]

                # Skip if the sample has been used too many times
                # if new_label.used > min_used:
                #     continue

                # Determine if the sample can be pasted
                bs  = [l.bbox for l in labels]
                iou = mon.hbb_iou(new_label.bbox, bs)
                if np.any(iou > self.iou_thres):
                    tries += 1
                    continue

                # Add the sample to the image
                labels.append(new_label)
                new_labels.append(new_label)
                num_sample += 1
                candidates[choice].used += 1

                # Reset tries after a successful paste
                tries = 0

        return labels, new_labels

    def _copy_and_paste_labels(self, image: np.ndarray, new_labels: list[Label]) -> np.ndarray:
        # Group labels by image_file
        grouped_labels = {}
        for l in new_labels:
            if l.image_path not in grouped_labels:
                grouped_labels[l.image_path] = [l]
            else:
                grouped_labels[l.image_path].append(l)

        # Copy and paste each label from the source image to the target image
        dst = image.copy()
        for image_file, labels in grouped_labels.items():
            source = cv2.imread(str(image_file))
            if self.style_transfer:
                source = mon.color_transfer(source=source, target=image)
            for l in labels:
                dst = self._copy_and_paste_single_label(src=source, dst=dst, label=l)

        return dst

    def _copy_and_paste_single_label(self, src: np.ndarray, dst: np.ndarray, label: Label) -> np.ndarray:
        """Copy and paste a single ``label`` from ``src`` to ``dst`` image."""
        x1, y1, x2, y2    = label.bbox[:4].astype(int)
        mask              = label.mask
        roi_src           = src[y1:y2, x1:x2]
        roi_dst           = dst[y1:y2, x1:x2]
        dst[y1:y2, x1:x2] = roi_src * mask + roi_dst * (1.0 - mask)
        return dst

    # ----- Utils -----
    def _group_obj_per_class(self, data: list[Label]) -> dict[int, list[Label]]:
        """Group objects per class."""
        groups = {c: [] for c in range(self._num_classes)}
        for b in data:
            c = b.class_id
            if c not in groups:
                groups[c] = []
            groups[c].append(b)
        return groups

    def _count_obj_per_class(self, data: list[Label]) -> dict[int, int]:
        """Count objects per class."""
        counts = {c: 0 for c in range(self._num_classes)}
        for b in data:
            c = b.class_id
            if c not in counts:
                counts[c] = 0
            counts[c] += 1
        return counts

    def _count_obj_per_image(self, data: list[Label], image_file: mon.Path) -> dict[int, int]:
        """Count objects per image."""
        counts = {}
        bs = [b.bbox for b in data if b.image_path == image_file]
        for b in bs:
            c = int(b[4])
            if c not in counts:
                counts[c] = 0
            counts[c] += 1
        return counts


# ----- Convert -----
def convert_all(data: str, split: str, runs: int = 1):
    data_root = data_dir / data / split
    for subdir in sorted(data_root.subdirs()):
        image_dir = data_dir / data / split / subdir.stem / "image"
        label_dir = data_dir / data / split / subdir.stem / "label"

        augment   = CutAndPasteStaticAugmentation(
            image_dir      = image_dir,
            label_dir      = label_dir,
            num_classes    = 5,
            sam_model      = "sam2.1_l.pt",
            ratio          = 1,
            iou_thres      = 0.00001,
            max_tries      = 100,
            style_transfer = False,
        )
        for _ in range(runs):
            augment.process()


def convert_one(data: str, split: str, subdir: str, runs: int = 1):
    image_dir = data_dir / data / split / subdir / "image"
    label_dir = data_dir / data / split / subdir / "label"

    augment   = CutAndPasteStaticAugmentation(
        image_dir      = image_dir,
        label_dir      = label_dir,
        num_classes    = 5,
        sam_model      = "sam2.1_l.pt",
        ratio          = 1,
        iou_thres      = 0.00001,
        max_tries      = 100,
        style_transfer = False,
    )
    for _ in range(runs):
        augment.process()


# ----- Main -----
if __name__ == "__main__":
    # group_image_and_label("fisheye8k", "cameras")
    #convert_all("fisheye8k", "cameras", 1)
    #convert_one("fisheye8k", "cameras", "camera1_A", 1)
    #convert_one("fisheye8k", "cameras", "camera2_A", 1)
    #convert_one("fisheye8k", "cameras", "camera3_A", 1)
    #convert_one("fisheye8k", "cameras", "camera3_N", 1)
    #convert_one("fisheye8k", "cameras", "camera4_A", 1)
    #convert_one("fisheye8k", "cameras", "camera4_E", 1)
    #convert_one("fisheye8k", "cameras", "camera4_M", 1)
    #convert_one("fisheye8k", "cameras", "camera4_N", 1)
    #convert_one("fisheye8k", "cameras", "camera5_A", 1)
    #convert_one("fisheye8k", "cameras", "camera6_A", 1)
    #convert_one("fisheye8k", "cameras", "camera7_A", 1)
    #convert_one("fisheye8k", "cameras", "camera8_A", 1)
    #convert_one("fisheye8k", "cameras", "camera9_A", 1)
    #convert_one("fisheye8k", "cameras", "camera10_A", 1)
    #convert_one("fisheye8k", "cameras", "camera11_M", 1)
    #convert_one("fisheye8k", "cameras", "camera12_A", 1)
    #convert_one("fisheye8k", "cameras", "camera13_A", 1)
    #convert_one("fisheye8k", "cameras", "camera14_A", 1)
    #convert_one("fisheye8k", "cameras", "camera15_A", 1)
    #convert_one("fisheye8k", "cameras", "camera16_A", 1)
    #convert_one("fisheye8k", "cameras", "camera17_A", 1)
    #convert_one("fisheye8k", "cameras", "camera18_A", 1)
    #convert_one("fisheye8k", "cameras", "camera19_A", 1)
    #convert_one("fisheye8k", "cameras", "camera20_A", 1)
    #convert_one("fisheye8k", "cameras", "camera21_A", 1)
    #convert_one("fisheye8k", "cameras", "camera22_A", 1)
    #convert_one("fisheye8k", "cameras", "camera23_A", 1)
    #convert_one("fisheye8k", "cameras", "camera24_A", 1)
    #convert_one("fisheye8k", "cameras", "camera25_A", 1)
    #convert_one("fisheye8k", "cameras", "camera26_A", 1)
    #convert_one("fisheye8k", "cameras", "camera27_A", 1)
    #convert_one("fisheye8k", "cameras", "camera28_A", 1)
    #convert_one("fisheye8k", "cameras", "camera29_A", 1)
    #convert_one("fisheye8k", "cameras", "camera29_N", 1)
    concat_image_and_label("fisheye8k", "cameras")
