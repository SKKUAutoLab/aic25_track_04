#!/bin/bash

echo "${HOSTNAME}"
clear

# ----- Input -----
task="detect"
mode="train"
arch="deim"
model="deim_dfine_s"
imgsz=960
#imgsz=1280
device="cuda:0"
batch_size=16

fullname="${model}_${imgsz}_cv2"

# ----- Directory & File -----
current_file=$(readlink -f "${0}")
current_dir=$(dirname "${current_file}")
#
deim_dir="${current_dir}/mon/src/mon/vision/detect/deim"
config="${deim_dir}/config/fisheye8k/${fullname}.yaml"

# ----- Validation -----
check_file() {
    [[ ! -f "$1" ]] && { echo "File not found: $1"; exit 1; }
}

check_dir() {
    [[ ! -d "$1" ]] && { echo "Directory not found: $1"; exit 1; }
}

create_dir() {
    [[ ! -d "$1" ]] && { echo "Creating directory: $1"; mkdir -p "$1"; }
}

check_file "${config}"

# ----- Main -----
cd "${current_dir}" || exit
python -W ignore main.py \
    --root "${current_dir}" \
    --task "${task}" \
    --mode "${mode}" \
    --arch "${arch}" \
    --model "${model}" \
    --config "${config}" \
    --device "${device}" \
    --batch-size "${batch_size}" \
    --save-result \
    --save-image \
    --exist-ok \
    --verbose \
    "$@"

# ----- Done -----
cd "${current_dir}" || exit
exit 0
