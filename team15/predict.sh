#!/bin/bash

echo "${HOSTNAME}"
clear

# ----- Input -----
model="deim_dfine_s"
imgsz=960
#imgsz=1280
trt_p="fp16n32"
#trt_p="fp32"

fullname="${model}_${imgsz}_cv2_${trt_p}"

# ----- Directory & File -----
current_file=$(readlink -f "${0}")
current_dir=$(dirname "${current_file}")
#
image_folder="${current_dir}/Fisheye1K_eval/images"
output_json="${current_dir}/Fisheye1K_eval/predictions.json"
groundtruth_json="${current_dir}/Fisheye1K_eval/groundtruth.json"
model_path="${current_dir}/models/${fullname}.engine"
output_dir=$(dirname "${output_json}")

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

check_dir "${image_folder}"
#check_file "${groundtruth_json}"
check_file "${model_path}"
create_dir "${output_dir}"

# ----- Main -----
cd "${current_dir}" || exit

nvpmodel -q         # Check current power mode
sudo jetson_clocks  # Set the Jetson to maximum frequency (CPU + GPU) within the power code constraints.

echo "Running model: ${model_path}"
python -W ignore run_evaluation_jetson.py \
    --image-folder "${image_folder}" \
    --output-json "${output_json}" \
    --ground-truths-path "${groundtruth_json}" \
    --model-path "${model_path}" \
    --imgsz "${imgsz}" \
    "$@"

# ----- Done -----
cd "${current_dir}" || exit
exit 0
