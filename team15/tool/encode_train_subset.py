#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Encode and decode subset IDs for Pretrained or FishEye8K datasets."""

from math import comb

import mon

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]

subsets      = {
    "visdrone"             : 0,
    "fisheye8k:train"      : 1,
    "fisheye8k:train_fixed": 2,
    "fisheye8k:val"        : 3,
    "fisheye8k:val_fixed"  : 4,
    "fisheye8k:syn"        : 5,
    "fisheye8k:test1"      : 6,
    "fisheye8k:test2"      : 7,
}


def encode_subset_2letter(item: str = None, subcategory: str = None, combo: list[str] = None) -> str:
    """Encodes a dataset item, subcategory, or combination into a 2-letter code using a mathematical scheme.

    Args:
        item (str, optional): Dataset name ('visdrone' or 'fisheye8k').
        subcategory (str, optional): Subcategory for 'fisheye8k' (e.g., 'train', 'test_01').
        combo (list, optional): List of items/subcategories (e.g., ['visdrone', 'fisheye8k:train']).

    Returns:
        str: A 2-letter code (e.g., 'AA', 'DY').

    Raises:
        ValueError: If inputs are invalid or code space is exhausted.

    Examples:
        >>> encode_dataset_2letter(item="visdrone")
        'AA'
        >>> encode_dataset_2letter(item="fisheye8k", subcategory="train")
        'AB'
        >>> encode_dataset_2letter(combo=["visdrone", "fisheye8k:train"])
        'DY'
    """
    # Helper: Convert value to 2-letter code
    def value_to_code(v: int) -> str:
        if not 0 <= v <= 675:
            raise ValueError(f"Value out of range: {v}")
        l1 = chr(65 + (v // 26))  # A=65 in ASCII
        l2 = chr(65 + (v % 26))
        return l1 + l2

    # Helper: Compute combinatorial rank of a subset
    def subset_rank(subset: list[int]) -> int:
        subset = sorted(subset)
        rank = 0
        for i, s in enumerate(subset, 1):
            rank += comb(s, i)
        return rank

    # Single item encoding
    if item and not combo:
        if item == "visdrone":
            if subcategory:
                raise ValueError("visdrone has no subcategories")
            key = "visdrone"
        elif item == "fisheye8k":
            if not subcategory:
                raise ValueError("fisheye8k requires a subcategory")
            key = f"fisheye8k:{subcategory}"
            if key not in subsets:
                raise ValueError(f"Invalid subcategory: {subcategory}")
        else:
            raise ValueError(f"Invalid item: {item}")

        index = subsets[key]
        if index > 99:
            raise ValueError("Index exceeds single-item range")
        return value_to_code(index)

    # Combination encoding
    if combo:
        if item or subcategory:
            raise ValueError("Use 'combo' for combinations, not 'item' or 'subcategory'")
        if not combo or len(combo) < 2:
            raise ValueError("Combo must contain at least two items")

        # Validate and convert combo to indices
        subset = []
        for c in combo:
            if c not in subsets:
                if c.startswith("fisheye8k:"):
                    raise ValueError(f"Invalid subcategory: {c}")
                raise ValueError(f"Invalid item: {c}")
            subset.append(subsets[c])

        # Compute rank and value
        k = len(subset)
        rank = subset_rank(subset)
        v = 100 + ((k * 1000 + rank) % 576)  # Include size to avoid collisions
        return value_to_code(v)

    raise ValueError("Must provide either 'item' or 'combo'")


if __name__ == "__main__":
    print(encode_subset_2letter(item="visdrone"), ": visdrone")
    print(encode_subset_2letter(item="fisheye8k", subcategory="train"),       ": fisheye8k:train")
    # print(encode_subset_2letter(item="fisheye8k", subcategory="train_fixed"), ": fisheye8k:train_fixed")
    # print(encode_subset_2letter(item="fisheye8k", subcategory="val"),         ": fisheye8k:val")
    # print(encode_subset_2letter(item="fisheye8k", subcategory="val_fixed"),   ": fisheye8k:val_fixed")
    # print(encode_subset_2letter(item="fisheye8k", subcategory="syn"),         ": fisheye8k:syn")
    # print(encode_subset_2letter(item="fisheye8k", subcategory="test1"),       ": fisheye8k:test1")
    print(encode_subset_2letter(item="fisheye8k", subcategory="test2"),       ": fisheye8k:test2")

    print(encode_subset_2letter(combo=["fisheye8k:train", "fisheye8k:val"]),                                     ": fisheye8k:train + fisheye8k:val")
    print(encode_subset_2letter(combo=["fisheye8k:train", "fisheye8k:val", "fisheye8k:test1"]),                  ": fisheye8k:train + fisheye8k:val + fisheye8k:test1")
    print(encode_subset_2letter(combo=["fisheye8k:train", "fisheye8k:val", "fisheye8k:test1", "fisheye8k:syn"]), ": fisheye8k:train + fisheye8k:val + fisheye8k:test1 + fisheye8k:syn")
    print(encode_subset_2letter(combo=["fisheye8k:train", "fisheye8k:val", "fisheye8k:test2"]),                  ": fisheye8k:train + fisheye8k:val + fisheye8k:test2")
    print(encode_subset_2letter(combo=["fisheye8k:train", "fisheye8k:val", "fisheye8k:test2", "fisheye8k:syn"]), ": fisheye8k:train + fisheye8k:val + fisheye8k:test2 + fisheye8k:syn")

    print(encode_subset_2letter(combo=["fisheye8k:train_fixed", "fisheye8k:val_fixed"]),                                     ": fisheye8k:train_fixed + fisheye8k:val_fixed")
    print(encode_subset_2letter(combo=["fisheye8k:train_fixed", "fisheye8k:val_fixed", "fisheye8k:test1"]),                  ": fisheye8k:train_fixed + fisheye8k:val_fixed + fisheye8k:test1")
    print(encode_subset_2letter(combo=["fisheye8k:train_fixed", "fisheye8k:val_fixed", "fisheye8k:test1", "fisheye8k:syn"]), ": fisheye8k:train_fixed + fisheye8k:val_fixed + fisheye8k:test1 + fisheye8k:syn")
    print(encode_subset_2letter(combo=["fisheye8k:train_fixed", "fisheye8k:val_fixed", "fisheye8k:test2"]),                  ": fisheye8k:train_fixed + fisheye8k:val_fixed + fisheye8k:test2")
    print(encode_subset_2letter(combo=["fisheye8k:train_fixed", "fisheye8k:val_fixed", "fisheye8k:test2", "fisheye8k:syn"]), ": fisheye8k:train_fixed + fisheye8k:val_fixed + fisheye8k:test2 + fisheye8k:syn")

    print(encode_subset_2letter(combo=["visdrone, fisheye8k:train", "fisheye8k:val"]), ": visdrone + fisheye8k:train + fisheye8k:val")
