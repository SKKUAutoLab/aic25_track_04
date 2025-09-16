#!/bin/bash

echo "${HOSTNAME}"
clear

# ----- Input -----
arch="deim"
model="deim_dfine_s"
#imgsz=960
imgsz=1280

fullname="${model}_${imgsz}_cv2"

# ----- Directory & File -----
current_file=$(readlink -f "${0}")
current_dir=$(dirname "${current_file}")
#
deim_dir="${current_dir}/mon/src/mon/vision/detect/deim"
config="${deim_dir}/config/fisheye8k/${fullname}.yaml"
weights="${current_dir}/run/train/${arch}/${model}/${fullname}/best_stg2_f1.pth"
save_dir="${current_dir}/models"

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
check_file "${weights}"

# ----- Main -----
cd "${deim_dir}" || exit

echo "Exporting ONNX: ${weights}."
python -W ignore i_export.py \
    --root "${current_dir}" \
    --arch "${arch}" \
    --model "${model}" \
    --config "${config}" \
    --fullname "${fullname}" \
    --save-dir "${save_dir}" \
    --weights "${weights}" \
    --use-fullname \
    --exist-ok \
    --verbose \
    "$@"

# ----- Done -----
cd "${current_dir}" || exit
exit 0
