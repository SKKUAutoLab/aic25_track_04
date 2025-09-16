# Team 15: AI City Challenge 2025 Track 4

---
## PREREQUISITES 

- Jetson AGX Orin 32GB: nvidia-jetpack (6.1) 
- Operating System: Ubuntu 22.04.5 LTS 
- Docker version 27.5.1
- Please ensure both [Docker & NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.13.5/install-guide.html) are installed on your machine.

**NOTICE: Due to file size limitation, please download the full inference code (including the data and supporting files) through this link: [Google Drive](https://drive.google.com/file/d/1OcAkx-xlIdAeBY7wl0Gf2F1GgIpyelXJ/view?usp=sharing).**

---
##  INFERENCE GUIDE

### Build the Docker Image & Container

- Build the Docker image and container using the following commands:
  ```bash
  # On local machine
  cd <path_to_project_dir_on_local_machine>/team15
  sudo jetson_clocks
  
  sudo docker build -f Dockerfile -t team15_jetson .
  sudo docker run --name team15_jetson -v "$(pwd):/team15" --runtime=nvidia --privileged --shm-size=32g -it team15_jetson
  ```
  - We mount the current directory to `/team15` in the container for easier access to the project files.
  - We tested with NVIDIA Jetson AGX Orin 32GB with 6.1 Jetpack, so `shm-size=32g`. If you are using a different hardware configuration, you may need to adjust the `--shm-size` parameter accordingly.


### Prepare Model & Data

- You need to place the ONNX models in the `/team15/models/` directory. 
- We have included two ONNX models:
  - **Final submission:** `/team15/models/deim_dfine_s_960_cv2.onnx`
  - Best F1-score : `/team15/models/deim_dfine_s_1280_cv2.onnx`


- Also, prepare the test dataset in the: `/team15/data/Fisheye1K_eval/`
  - For your convenience, we have included the `Fisheye1K_eval` dataset in the Docker:
  ```bash
  unzip /team15/data/Fisheye1K_eval.zip -d /team15/data/
  ```


### Inference (Final Submission)

- First, export ONNX model to TensorRT: `/team15/models/deim_dfine_s_960_cv2_fp16n32.engine`
- Second, run inference and save the predictions to: `/team15/data/Fisheye1K_eval/predictions.json`

```bash
# On local machine
sudo jetson_clocks

# Enter the Docker container
./export_trt.sh
# It can take a while to export the model, so please be patient.

./predict.sh
```


### Inference (Leaderboard Best F1-Score)

```bash
# On local machine
sudo jetson_clocks
  
# Enter the Docker container
./export_trt_1280_fp32.sh
# It can take a while to export the model, so please be patient.

./predict_1280_fp32.sh
 ```


### Misc

- You can modify the `export_trt.sh` and `predict.sh` scripts try different TensorRT optimization:
  - `imgsz`: `960` or `1280`
  - `trt_p`: `fp16n32` or `fp32`

---
## Contact
- If you have any questions, feel free to contact `Long H. Pham` ([longpham3105@gmail.com](longpham3105@gmail.com) or [phlong@skku.edu](phlong@skku.edu))
