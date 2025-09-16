#!/bin/bash

echo "${HOSTNAME}"
clear

# ----- Input -----
model="deim_dfine_s"
imgsz=960
#imgsz=1280
trt_p="fp16n32"
#trt_p="fp32"
dla_core=0

fullname="${model}_${imgsz}_cv2"

# ----- Directory & File -----
current_file=$(readlink -f "${0}")
current_dir=$(dirname "${current_file}")
#
onnx_file="${current_dir}/models/${fullname}.onnx"
engine_file="${current_dir}/models/${fullname}_${trt_p}.engine"
engine_dir=$(dirname "${engine_file}")

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

check_file "${onnx_file}"
create_dir "${engine_dir}"

# Skip if engine file already exists
[[ -f "${engine_file}" ]] && { echo "Engine file already exists: ${engine_file}"; exit 0; }

# ----- Main -----
cd "${current_dir}" || exit

echo "Converting ONNX -> TRT: ${engine_file}."
python -W ignore export_trt.py \
    --onnx-file "${onnx_file}" \
    --engine-file "${engine_file}" \
    --imgsz "${imgsz}" \
    --opset "16" \
    --trt-p "${trt_p}" \
    --dla-core "${dla_core}" \
    --verbose \
    "$@"

# ----- Done -----
cd "${current_dir}" || exit
exit 0
