# Team 15: AI City Challenge 2025 Track 4

---
## PREREQUISITES 

- Jetson AGX Orin 32GB: nvidia-jetpack (6.1) 
- Operating System: Ubuntu 22.04.5 LTS 
- Docker version 27.5.1
- Please ensure both [Docker & NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.13.5/install-guide.html) are installed on your machine.

**NOTICE: Due to file size limitation, please download the full training code (including the data and supporting files) through this link: [Google Drive](https://drive.google.com/file/d/1jAdPiZaQmFaBV28pUxgVhIywFw3JSUcg/view?usp=sharing).**


---
##  TRAINING GUIDE

### Build Docker Image & Container

- Build the Docker image and container using the following commands:
  ```bash
  cd <path_to_project_dir_on_local_machine>/team15
  
  sudo docker build -f Dockerfile.train -t team15 . && sudo docker image ls
  sudo docker run --name team15 -v "$(pwd):/team15" --gpus all --shm-size=48g -it team15
  ```
  - We mount the current directory to `/team15` in the container for easier modification of the project files.
  - We trained the model using 4x NVIDIA A6000 GPUs with 48GB. 
  - Adjust the `--gpus` and `--shm-size` parameters based on your hardware configuration.


### Prepare Data

- We have included the training data in the `/team15/data` directory:
  ```bash
  # Enter the Docker container then unzip the training data
  unzip /team15/data/fisheye8k.zip -d /team15/data
  ```
  
  
### Train Models

- Our model is trained in 2 stages:
  1. **Pre-training**: This stage trains the model on open-source datasets to learn general features.
  2. **Fine-tuning**: This stage fine-tunes the model on Fisheye8K dataset to adapt it to the specific task.

  ```bash
  # First stage: Pre-training
  ./train_pre.sh

  # Second stage: Fine-tuning
  ./train.sh
  ```
  - Just follow the training instructions and select the appropriate option.


- **NOTICE**: 
  - When fine-tuning, the training process may be interrupted due to multi-gpu training. If this happens, just simply resume the training by running the command again: `./train_distributed.sh`
  - The trained models will be saved in this directory: 
    - `/team15/run/train/deim/deim_dfine_s/deim_dfine_s_960_cv2/best_stg2_f1.pth`
  - **When you train the model, please backup the pretrained model first, or it will be overwritten!**


### Export to ONNX

- After training, export the model to ONNX format for inference. The exported ONNX model will be saved in: `/team15/models/`.
  ```bash
  ./export_onnx.sh
  ```
  

- **NOTICE**:
  - During the challenge, we test all possible model settings and configurations.
  - The highest F1-score model is: `deim_dfine_s_1280_cv2/best_stg2_f1.pth`, **which is not our final submission when considering the inference speed!**
  - You can still export this model for checking the F1-score:
    ```bash
    ./export_onnx_1280.sh
    ```


### Misc

- You can adjust the training parameters in `train.sh` and `train_pre.sh`:
  - `batch_size`: `16` for single GPU and `32` for multi-GPU.
  - `device`:  `cuda:0` for single GPU and `cuda:0,1,2,3` for multi-GPU.
  - `imgsz`: `960`
- You can modify other training parameters in config files located in:
  - `/team15/mon/src/mon/vision/detect/deim/config/fisheye8k/deim_dfine_s_960_cv2.yaml`
  - `/team15/mon/src/mon/vision/detect/deim/config/fisheye8k/deim_dfine_s_1280_cv2.yaml`


---
## Contact
- If you have any questions, feel free to contact `Long H. Pham` ([longpham3105@gmail.com](longpham3105@gmail.com) or [phlong@skku.edu](phlong@skku.edu))
