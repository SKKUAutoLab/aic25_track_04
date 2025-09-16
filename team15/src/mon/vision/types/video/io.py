#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements input/output operations for videos."""

__all__ = [
    "VideoWriter",
    "VideoWriterCV",
    "VideoWriterFFmpeg",
    "load_video_ffmpeg",
    "write_video_ffmpeg",
]

from abc import ABC, abstractmethod

import cv2
import ffmpeg
import numpy as np
import torch

from mon import core
from mon.nn import _size_2_t
from mon.vision.types import image as I


# ----- Reading -----
def load_video_ffmpeg(
    process,
    height   : int,
    width    : int,
    to_tensor: bool = False,
    normalize: bool = False,
) -> torch.Tensor | np.ndarray:
    """Read video frame bytes via ``ffmpeg``.

    Args:
        process: Subprocess managing ``ffmpeg`` instance as ``subprocess.Popen``.
        height: Video frame height as ``int``.
        width: Video frame width as ``int``.
        to_tensor: Convert to ``torch.Tensor`` if ``True``. Default is ``False``.
        normalize: Normalize to [0.0, 1.0] if ``True``. Default is ``False``.

    Returns:
        Frame as ``numpy.ndarray`` in [H, W, C] with range [0, 255],
        ``torch.Tensor`` in [1, C, H, W] with range [0.0, 1.0], or ``None`` if no data.

    Raises:
        ValueError: If read bytes do not match expected frame size.
    """
    # RGB24: 3 bytes per pixel
    img_size = height * width * 3
    in_bytes = process.stdout.read(img_size)
    if len(in_bytes) == 0:
        image = None
    else:
        if len(in_bytes) != img_size:
            raise ValueError(f"[in_bytes] length [{len(in_bytes)}] != expected size [{img_size}].")
        image = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )
        if to_tensor:
            image = I.image_to_tensor(image, normalize)
    return image


# ----- Writing -----
def write_video_ffmpeg(
    process,
    frame      : torch.Tensor | np.ndarray,
    denormalize: bool = False
):
    """Write frame to video via ``ffmpeg``.

    Args:
        process: Subprocess managing ``ffmpeg`` as ``subprocess.Popen``.
        frame: Frame/image as ``torch.Tensor`` or ``numpy.ndarray`` in [1, C, H, W].
        denormalize: Convert to [0, 255] if ``True``. Default is ``False``.

    Raises:
        ValueError: If ``frame`` is not a ``torch.Tensor`` or ``numpy.ndarray``.
    """
    if isinstance(frame, np.ndarray):
        if I.is_image_normalized(frame):
            frame = I.denormalize_image(frame)
        if I.is_image_channel_first(frame):
            frame = I.image_to_channel_last(frame)
    elif isinstance(frame, torch.Tensor):
        frame = I.image_to_array(frame, denormalize)
    else:
        raise ValueError(f"[frame] must be a torch.Tensor or np.ndarray, got [{type(frame).__name__}].")
    process.stdin.write(
        frame
        .astype(np.uint8)
        .tobytes()
    )
    return None


class VideoWriter(ABC):
    """Base class for video writers.

    Args:
        dst: Directory to save images.
        imgsz: Output size in [H, W] format. Default is ``[480, 640]``.
        frame_rate: Frame rate of output video as ``float``. Default is ``10``.
        denormalize: Convert to [0, 255] if ``True``. Default is ``False``.
        verbose: Enable verbosity if ``True``. Default is ``False``.
    """
    
    def __init__(
        self,
        dst		   : core.Path,
        imgsz      : tuple[int, int] = [480, 640],
        frame_rate : float = 10,
        denormalize: bool  = False,
        verbose    : bool  = False,
        *args, **kwargs
    ):
        self.dst         = core.Path(dst)
        self.denormalize = denormalize
        self.index       = 0
        self.image_size  = I.image_size(imgsz)
        self.frame_rate  = frame_rate
        self.verbose     = verbose
        self.init()
        
    def __len__(self) -> int:
        """Count written frames.

        Returns:
            Number of written frames as ``int``.
        """
        return self.index
    
    def __del__(self):
        """Close video writer."""
        self.close()
    
    @abstractmethod
    def init(self):
        """Initialize output handler."""
        pass
    
    @abstractmethod
    def close(self):
        """Close video writer."""
        pass
    
    @abstractmethod
    def write(
        self,
        frame      : torch.Tensor | np.ndarray,
        path       : core.Path = None,
        denormalize: bool = False
    ):
        """Write frame to ``dst``.

        Args:
            frame: Video frame as ``torch.Tensor`` or ``numpy.ndarray``.
            path: Image file path. Default is ``None``.
            denormalize: Convert to [0, 255] if ``True``. Default is ``False``.
        """
        pass
    
    @abstractmethod
    def write_batch(
        self,
        frames     : list[torch.Tensor] | list[np.ndarray],
        paths      : list[core.Path] = None,
        denormalize: bool = False
    ):
        """Write batch of frames to ``dst``.

        Args:
            frames: List of video frames as ``list[torch.Tensor] | list[np.ndarray]``.
            paths: List of file paths as ``list[core.Path]``. Default is ``None``.
            denormalize: Convert to [0, 255] if ``True``. Default is ``False``.
        """
        pass
    

class VideoWriterCV(VideoWriter):
    """Write images to video using ``cv2``.

    Args:
        dst: Directory to save video.
        imgsz: Output size in [H, W] format. Default is ``[480, 640]``.
        frame_rate: Frame rate of video as ``float``. Default is ``30``.
        fourcc: Video codec as ``str``. One of ``"mp4v"``, ``"xvid"``, ``"mjpg"``,
            ``"wmv"``. Default is ``"mp4v"``.
        denormalize: Convert to [0, 255] if ``True``. Default is ``False``.
        verbose: Enable verbosity if ``True``. Default is ``False``.
    """
    
    def __init__(
        self,
        dst		   : core.Path,
        imgsz      : tuple[int, int] = [480, 640],
        frame_rate : float = 30,
        fourcc     : str   = "mp4v",
        denormalize: bool  = False,
        verbose    : bool  = False,
        *args, **kwargs
    ):
        self.fourcc       = fourcc
        self.video_writer = None
        super().__init__(
            dst			= dst,
            imgsz= imgsz,
            frame_rate  = frame_rate,
            denormalize = denormalize,
            verbose     = verbose,
            *args, **kwargs
        )
    
    def init(self):
        """Initialize video writer."""
        if self.dst.is_dir():
            video_file = self.dst / f"result.mp4"
        else:
            video_file = self.dst.parent / f"{self.dst.stem}.mp4"
        video_file.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*self.fourcc)
        self.video_writer = cv2.VideoWriter(
            filename  = str(video_file),
            fourcc    = fourcc,
            fps       = float(self.frame_rate),
            frameSize =self.image_size[::-1],  # Must be in [W, H]
            isColor   = True
        )
        
        if self.video_writer is None:
            raise FileNotFoundError(f"[video_file] cannot be created at {video_file}.")
    
    def close(self):
        """Close video writer."""
        if self.video_writer:
            self.video_writer.release()
    
    def write(
        self,
        frame      : torch.Tensor | np.ndarray,
        path       : core.Path = None,
        denormalize: bool = False
    ):
        """Write frame to video.

        Args:
            frame: Image as ``torch.Tensor`` or ``numpy.ndarray``.
            path: File path. Default is ``None``.
            denormalize: Convert to [0, 255] if ``True``. Default is ``False``.
        """
        denormalize = denormalize or self.denormalize
        image       = I.image_to_array(frame, denormalize)
        # IMPORTANT: Image must be in a BGR format
        image       = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.video_writer.write(image)
        self.index += 1
    
    def write_batch(
        self,
        frames     : list[torch.Tensor] | list[np.ndarray],
        paths      : list[core.Path] = None,
        denormalize: bool = False
    ):
        """Write batch of frames to video.

        Args:
            frames: List of images as ``list[torch.Tensor] | list[np.ndarray]``.
            paths: List of file paths as ``list[core.Path]``. Default is ``None``.
            denormalize: Convert to [0, 255] if ``True``. Default is ``False``.
        """
        if paths is None:
            paths = [None for _ in range(len(frames))]
        for frame, path in zip(frames, paths):
            self.write(frame, path, denormalize)


class VideoWriterFFmpeg(VideoWriter):
    """Write images to video using ``ffmpeg``.

    Args:
        dst: Directory to save video.
        imgsz: Output size in [H, W] format. Default is ``[480, 640]``.
        frame_rate: Frame rate of video as ``float``. Default is ``10``.
        pix_fmt: Video codec as ``str``. Default is ``"yuv420p"``.
        denormalize: Convert to [0, 255] if ``True``. Default is ``False``.
        verbose: Enable verbosity if ``True``. Default is ``False``.
    """
    
    def __init__(
        self,
        dst		   : core.Path,
        imgsz      : tuple[int, int] = [480, 640],
        frame_rate : float = 10,
        pix_fmt    : str   = "yuv420p",
        denormalize: bool  = False,
        verbose    : bool  = False,
        *args, **kwargs
    ):
        self.pix_fmt        = pix_fmt
        self.ffmpeg_process = None
        self.ffmpeg_kwargs  = kwargs
        super().__init__(
            dst			= dst,
            imgsz= imgsz,
            frame_rate  = frame_rate,
            denormalize = denormalize,
            verbose     = verbose,
            *args, **kwargs
        )
    
    def init(self):
        """Initialize video writer."""
        if self.dst.is_dir():
            video_file = self.dst / "result.mp4"
        else:
            video_file = self.dst.parent / f"{self.dst.stem}.mp4"
        video_file.parent.mkdir(parents=True, exist_ok=True)

        s = f"{self.image_size[1]}x{self.image_size[0]}"  # WxH for ffmpeg
        stream = (
            ffmpeg
            .input(
                filename = "pipe:",
                format   = "rawvideo",
                pix_fmt  = "rgb24",
                s        = s
            )
            .output(
                filename = str(video_file),
                pix_fmt  = self.pix_fmt,
                **self.ffmpeg_kwargs
            )
            .overwrite_output()
        )
        if not self.verbose:
            stream = stream.global_args("-loglevel", "quiet")
        self.ffmpeg_process = stream.run_async(pipe_stdin=True)
    
    def close(self):
        """Close video writer."""
        if self.ffmpeg_process:
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.terminate()
            self.ffmpeg_process.wait()
            self.ffmpeg_process = None
    
    def write(
        self,
        frame      : torch.Tensor | np.ndarray,
        path       : core.Path = None,
        denormalize: bool = False
    ):
        """Write frame to video.

        Args:
            frame: Image as ``torch.Tensor`` or ``numpy.ndarray``.
            path: File path. Default is ``None``.
            denormalize: Convert to [0, 255] if ``True``. Default is ``False``.
        """
        denormalize = denormalize or self.denormalize
        write_video_ffmpeg(self.ffmpeg_process, frame, denormalize)
        self.index += 1
    
    def write_batch(
        self,
        frames     : list[torch.Tensor] | list[np.ndarray],
        paths      : list[core.Path] = None,
        denormalize: bool = False,
    ):
        """Write batch of frames to video.

        Args:
            frames: List of images as ``list[torch.Tensor] | list[np.ndarray]``.
            paths: List of file paths as ``list[core.Path]``. Default is ``None``.
            denormalize: Convert to [0, 255] if ``True``. Default is ``False``.
        """
        if paths is None:
            paths = [None for _ in range(len(frames))]
        for frame, path in zip(frames, paths):
            self.write(frame, path, denormalize)
