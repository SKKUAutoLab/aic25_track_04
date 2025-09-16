#!/bin/bash

# sudo chmod +x install.sh
# ./install.sh

clear
echo "${HOSTNAME}"

# ----- Input -----


# ----- Directory & File -----
current_file=$(readlink -f "${0}")
current_dir=$(dirname "${current_file}")
root_dir=$current_dir


# ----- Main -----
conda activate mon

pip install \
  emoji fjson loguru plum-dispatch psutil pyhumps python-box requests rich tabulate tqdm typing-extensions validators \
  PyYAML filterpy numpy protobuf scipy xmltodict \
  einops fvcore lightning onnx onnxruntime openvino scikit-learn tensorboard thop torch-fidelity torchfile torchmetrics transformers \
  Pillow PyTurboJPEG albumentations  faster-coco-eval ffmpeg-python kornia opencv-contrib-python opencv-python pillow_heif pycocotools pyiqa rawpy scikit-image

# ----- Done -----
exit 0
