#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements distance functions between points or arrays."""

__all__ = [
	"angle_between_arrays",
	"chebyshev_distance",
	"cosine_distance",
	"distance_between_bbox_and_polygon",
	"distance_between_bbox_center_and_polygon",
	"distance_between_points",
	"euclidean_distance",
	"get_distance_function",
	"hausdorff_distance",
	"haversine_distance",
	"manhattan_distance",
]

import math
from typing import Callable

import cv2
import numpy as np


# ----- Angle between two arrays -----
def angle_between_arrays(x: np.ndarray, y: np.ndarray) -> float | bool:
	"""Calculate the angle between two arrays."""
	vec1 = np.array([x[-1][0] - x[0][0], x[-1][1] - x[0][1]])
	vec2 = np.array([y[-1][0] - y[0][0], y[-1][1] - y[0][1]])
	L1   = np.sqrt(vec1.dot(vec1))
	L2   = np.sqrt(vec2.dot(vec2))
	if L1 == 0 or L2 == 0:
		return False
	cos   = vec1.dot(vec2) / (L1 * L2)
	angle = np.arccos(cos) * 360 / (2 * np.pi)
	return angle


# ----- Distance between two arrays -----
def chebyshev_distance(x: np.ndarray, y: np.ndarray) -> float:
	"""Calculate the Chebyshev distance."""
	n   = x.shape[0]
	ret = -1 * np.inf
	for i in range(n):
		d = abs(x[i] - y[i])
		if d > ret:
			ret = d
	return ret


def cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
	"""Calculate the Cosine distance."""
	n      = x.shape[0]
	xy_dot = 0.0
	x_norm = 0.0
	y_norm = 0.0
	for i in range(n):
		xy_dot += x[i] * y[i]
		x_norm += x[i] * x[i]
		y_norm += y[i] * y[i]
	return 1.0 - xy_dot / (math.sqrt(x_norm) * math.sqrt(y_norm))


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
	"""Calculate of Euclidean distance."""
	n   = x.shape[0]
	ret = 0.0
	for i in range(n):
		ret += (x[i] - y[i]) ** 2
	return math.sqrt(ret)


def hausdorff_distance(x: np.ndarray, y: np.ndarray) -> float:
	"""Calculate Hausdorff distance.
	
	``euclidean_distance``, ``manhattan_distance``, ``chebyshev_distance``,
	``cosine_distance``, ``haversine_distance`` could be use for this function.
	"""
	cmax = 0.0
	for i in range(len(x)):
		cmin = np.inf
		for j in range(len(y)):
			# euclidean_distance, manhattan_distance, chebyshev_distance, cosine_distance, haversine_distance
			d = euclidean_distance(x[i, :], y[j, :])
			if d < cmin:
				cmin = d
			if cmin < cmax:
				break
		if cmax < cmin < np.inf:
			cmax = cmin
	return cmax


def haversine_distance(x: np.ndarray, y: np.ndarray) -> float:
	"""Calculate Haversine distance."""
	r       = 6378.0
	radians = np.pi / 180.0
	lat_x   = radians * x[0]
	lon_x   = radians * x[1]
	lat_y   = radians * y[0]
	lon_y   = radians * y[1]
	dlon    = lon_y - lon_x
	dlat    = lat_y - lat_x
	a       = (pow(math.sin(dlat / 2.0), 2.0) + math.cos(lat_x) * math.cos(lat_y) * pow(math.sin(dlon / 2.0), 2.0))
	return r * 2 * math.asin(math.sqrt(a))


def manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
	"""Calculate Manhattan distance."""
	n   = x.shape[0]
	ret = 0.0
	for i in range(n):
		ret += abs(x[i] - y[i])
	return ret


def get_distance_function(name: str) -> Callable[[np.ndarray, np.ndarray], float]:
	"""Get the distance function by name."""
	if name in ["chebyshev"]:
		return chebyshev_distance
	elif name in ["cosine"]:
		return cosine_distance
	elif name in ["euclidean"]:
		return euclidean_distance
	elif name in ["haversine"]:
		return haversine_distance
	elif name in ["hausdorff"]:
		return hausdorff_distance
	elif name in ["manhattan"]:
		return manhattan_distance
	else:
		raise ValueError(f"")


# ----- Distance between points -----
def distance_between_points(x: np.ndarray, y: np.ndarray) -> float:
	"""Calculate Euclidean distance between two points in [x, y] format."""
	return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))


# ----- Distance between bounding box and polygon -----
def distance_between_bbox_and_polygon(bbox_xyxy: np.ndarray, polygon: np.ndarray) -> float:
	"""Compute the distance between a bounding box and a polygon.
	
	Args:
		bbox_xyxy: The bounding box coordinates in ``XYXY`` format.
		polygon: The polygon as a list of points in [x, y] format.
			
	Returns:
		positive if the bounding box is inside the ROI,
		zero if the bounding box is on the edge of the ROI, and
		negative if the bounding box is outside the ROI.
	"""
	tl = cv2.pointPolygonTest(polygon, (bbox_xyxy[0], bbox_xyxy[1]), True)
	tr = cv2.pointPolygonTest(polygon, (bbox_xyxy[2], bbox_xyxy[1]), True)
	br = cv2.pointPolygonTest(polygon, (bbox_xyxy[2], bbox_xyxy[3]), True)
	bl = cv2.pointPolygonTest(polygon, (bbox_xyxy[0], bbox_xyxy[3]), True)
	if tl > 0 and tr > 0 and br > 0 and bl > 0:
		return min(tl, tr, br, bl)
	elif tl < 0 and tr < 0 and br < 0 and bl < 0:
		return min(tl, tr, br, bl)
	else:
		return 0
		

def distance_between_bbox_center_and_polygon(bbox_xyxy: np.ndarray, polygon: np.ndarray) -> float:
	"""Compute the distance between a bounding box center and a polygon.
	
	Args:
		bbox_xyxy: The bounding box coordinates in ``XYXY`` format.
		polygon: The polygon as a list of points in [x, y] format.
	
	Returns:
		positive if the bounding box is inside the ROI,
		zero if the bounding box is on the edge of the ROI, and
		negative if the bounding box is outside the ROI.
	"""
	cx = (bbox_xyxy[0] + bbox_xyxy[2]) / 2
	cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2
	return int(cv2.pointPolygonTest(polygon, (cx, cy), True))
