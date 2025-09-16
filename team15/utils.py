#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections
import json
import os
import pathlib
from collections import OrderedDict

import cv2
import numpy as np
import tensorrt as trt
import torch
import torchvision.transforms as T

transforms      = T.Compose([T.ToTensor(),])
supported_trt_p = ["fp32", "fp16n32", "fp16ln32", "fp16lbn32", "fp16lbln32", "fp16"]
fp32_layers     = {
    "fp16ln32"  : [
        #                Function name                                Weights name
        # Encoder
        # "/model/encoder/encoder.0/layers.0/norm1",      "model.encoder.encoder.0.layers.0.norm1",
        # "/model/encoder/encoder.0/layers.0/norm2",      "model.encoder.encoder.0.layers.0.norm2",
        # Decoder
        "/model/decoder/enc_output/norm",               "model.decoder.enc_output.norm",
        # Decoder block 0
        # "/model/decoder/decoder/layers.0/norm1",        "model.decoder.decoder.layers.0.norm1",
        # "/model/decoder/decoder/layers.0/gateway/norm", "model.decoder.decoder.layers.0.gateway.norm",
        "/model/decoder/decoder/layers.0/norm3",        "model.decoder.decoder.layers.0.norm3",
        # Decoder block 1
        # "/model/decoder/decoder/layers.1/norm1",        "model.decoder.decoder.layers.1.norm1",
        # "/model/decoder/decoder/layers.1/gateway/norm", "model.decoder.decoder.layers.1.gateway.norm",
        "/model/decoder/decoder/layers.1/norm3",        "model.decoder.decoder.layers.1.norm3",
        # Decoder block 2
        # "/model/decoder/decoder/layers.2/norm1",        "model.decoder.decoder.layers.2.norm1",
        # "/model/decoder/decoder/layers.2/gateway/norm", "model.decoder.decoder.layers.2.gateway.norm",
        "/model/decoder/decoder/layers.2/norm3",        "model.decoder.decoder.layers.2.norm3"
        # Decoder block 3
        # "/model/decoder/decoder/layers.3/norm1",        "model.decoder.decoder.layers.3.norm1",
        # "/model/decoder/decoder/layers.3/gateway/norm", "model.decoder.decoder.layers.3.gateway.norm",
        "/model/decoder/decoder/layers.3/norm3",        "model.decoder.decoder.layers.3.norm3",
        # Decoder block 4
        # "/model/decoder/decoder/layers.4/norm1",        "model.decoder.decoder.layers.4.norm1",
        # "/model/decoder/decoder/layers.4/gateway/norm", "model.decoder.decoder.layers.4.gateway.norm",
        "/model/decoder/decoder/layers.4/norm3",        "model.decoder.decoder.layers.4.norm3",
        # Decoder block 5
        # "/model/decoder/decoder/layers.5/norm1",        "model.decoder.decoder.layers.5.norm1",
        # "/model/decoder/decoder/layers.5/gateway/norm", "model.decoder.decoder.layers.5.gateway.norm",
        "/model/decoder/decoder/layers.5/norm3",        "model.decoder.decoder.layers.5.norm3",
    ],
    "fp16lbn32" : [
        #                Function name                                Weights name
        # Encoder
        # "/model/encoder/encoder.0/layers.0/norm1",      "model.encoder.encoder.0.layers.0.norm1",
        # "/model/encoder/encoder.0/layers.0/norm2",      "model.encoder.encoder.0.layers.0.norm2",
        # Decoder
        "/model/decoder/enc_output/norm",               "model.decoder.enc_output.norm",
        # Decoder block 0
        # "/model/decoder/decoder/layers.0/norm1",        "model.decoder.decoder.layers.0.norm1",
        # "/model/decoder/decoder/layers.0/gateway/norm", "model.decoder.decoder.layers.0.gateway.norm",
        # "/model/decoder/decoder/layers.0/norm3",        "model.decoder.decoder.layers.0.norm3",
        # Decoder block 1
        # "/model/decoder/decoder/layers.1/norm1",        "model.decoder.decoder.layers.1.norm1",
        # "/model/decoder/decoder/layers.1/gateway/norm", "model.decoder.decoder.layers.1.gateway.norm",
        # "/model/decoder/decoder/layers.1/norm3",        "model.decoder.decoder.layers.1.norm3",
        # Decoder block 2
        # "/model/decoder/decoder/layers.2/norm1",        "model.decoder.decoder.layers.2.norm1",
        # "/model/decoder/decoder/layers.2/gateway/norm", "model.decoder.decoder.layers.2.gateway.norm",
        # "/model/decoder/decoder/layers.2/norm3",        "model.decoder.decoder.layers.2.norm3"
        # Decoder block 3
        # "/model/decoder/decoder/layers.3/norm1",        "model.decoder.decoder.layers.3.norm1",
        # "/model/decoder/decoder/layers.3/gateway/norm", "model.decoder.decoder.layers.3.gateway.norm",
        # "/model/decoder/decoder/layers.3/norm3",        "model.decoder.decoder.layers.3.norm3",
        # Decoder block 4
        # "/model/decoder/decoder/layers.4/norm1",        "model.decoder.decoder.layers.4.norm1",
        # "/model/decoder/decoder/layers.4/gateway/norm", "model.decoder.decoder.layers.4.gateway.norm",
        # "/model/decoder/decoder/layers.4/norm3",        "model.decoder.decoder.layers.4.norm3",
        # Decoder block 5
        "/model/decoder/decoder/layers.5/norm1",        "model.decoder.decoder.layers.5.norm1",
        "/model/decoder/decoder/layers.5/gateway/norm", "model.decoder.decoder.layers.5.gateway.norm",
        "/model/decoder/decoder/layers.5/norm3",        "model.decoder.decoder.layers.5.norm3",
    ],
    "fp16lbln32": [
        #                Function name                                Weights name
        # Encoder
        # "/model/encoder/encoder.0/layers.0/norm1",      "model.encoder.encoder.0.layers.0.norm1",
        # "/model/encoder/encoder.0/layers.0/norm2",      "model.encoder.encoder.0.layers.0.norm2",
        # Decoder
        "/model/decoder/enc_output/norm",               "model.decoder.enc_output.norm",
        # Decoder block 0
        # "/model/decoder/decoder/layers.0/norm1",        "model.decoder.decoder.layers.0.norm1",
        # "/model/decoder/decoder/layers.0/gateway/norm", "model.decoder.decoder.layers.0.gateway.norm",
        # "/model/decoder/decoder/layers.0/norm3",        "model.decoder.decoder.layers.0.norm3",
        # Decoder block 1
        # "/model/decoder/decoder/layers.1/norm1",        "model.decoder.decoder.layers.1.norm1",
        # "/model/decoder/decoder/layers.1/gateway/norm", "model.decoder.decoder.layers.1.gateway.norm",
        # "/model/decoder/decoder/layers.1/norm3",        "model.decoder.decoder.layers.1.norm3",
        # Decoder block 2
        # "/model/decoder/decoder/layers.2/norm1",        "model.decoder.decoder.layers.2.norm1",
        # "/model/decoder/decoder/layers.2/gateway/norm", "model.decoder.decoder.layers.2.gateway.norm",
        # "/model/decoder/decoder/layers.2/norm3",        "model.decoder.decoder.layers.2.norm3"
        # Decoder block 3
        # "/model/decoder/decoder/layers.3/norm1",        "model.decoder.decoder.layers.3.norm1",
        # "/model/decoder/decoder/layers.3/gateway/norm", "model.decoder.decoder.layers.3.gateway.norm",
        # "/model/decoder/decoder/layers.3/norm3",        "model.decoder.decoder.layers.3.norm3",
        # Decoder block 4
        # "/model/decoder/decoder/layers.4/norm1",        "model.decoder.decoder.layers.4.norm1",
        # "/model/decoder/decoder/layers.4/gateway/norm", "model.decoder.decoder.layers.4.gateway.norm",
        # "/model/decoder/decoder/layers.4/norm3",        "model.decoder.decoder.layers.4.norm3",
        # Decoder block 5
        # "/model/decoder/decoder/layers.5/norm1",        "model.decoder.decoder.layers.5.norm1",
        # "/model/decoder/decoder/layers.5/gateway/norm", "model.decoder.decoder.layers.5.gateway.norm",
        "/model/decoder/decoder/layers.5/norm3",        "model.decoder.decoder.layers.5.norm3",
    ],
}  # Not working yet


# ----- DEIM Utils -----
class TRTInference:

    def __init__(
        self,
        engine_path   : str,
        device        : torch.device | str = "cuda",
        backend       : str  = "torch",
        max_batch_size: int  = 32,
        imgsz         : int  = 640,
        verbose       : bool = False
    ):
        self.engine_path    = engine_path
        self.device         = device
        self.backend        = backend
        self.max_batch_size = max_batch_size
        self.imgsz          = imgsz

        self.logger         = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)

        self.engine         = self.load_engine(engine_path)
        self.context        = self.engine.create_execution_context()
        self.bindings       = self.get_bindings(self.engine, self.context, self.max_batch_size, self.device)
        self.bindings_addr  = OrderedDict((n, v.ptr) for n, v in self.bindings.items())
        self.input_names    = self.get_input_names()
        self.output_names   = self.get_output_names()

    def load_engine(self, path):
        trt.init_libnvinfer_plugins(self.logger, "")
        with open(path, "rb") as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def get_input_names(self):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names

    def get_output_names(self):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names

    def get_bindings(self, engine, context, max_batch_size=32, device=None) -> OrderedDict:
        Binding  = collections.namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        bindings = OrderedDict()

        for i, name in enumerate(engine):
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            if shape[0] == -1:
                shape[0] = max_batch_size
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    context.set_input_shape(name, shape)

            data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, shape, data, data.data_ptr())
        return bindings

    def run_torch(self, blob):
        for n in self.input_names:
            if self.bindings[n].shape != blob[n].shape:
                self.context.set_input_shape(n, blob[n].shape)
                self.bindings[n] = self.bindings[n]._replace(shape=blob[n].shape)

            assert self.bindings[n].data.dtype == blob[n].dtype, "{} dtype mismatch".format(n)

        self.bindings_addr.update({n: blob[n].data_ptr() for n in self.input_names})
        self.context.execute_v2(list(self.bindings_addr.values()))
        outputs = {n: self.bindings[n].data for n in self.output_names}
        return outputs

    def __call__(self, blob):
        if self.backend == "torch":
            return self.run_torch(blob)
        else:
            raise NotImplementedError("Only 'torch' backend is implemented.")

    def warmup(self, n: int):
        dummy     = torch.randn(1, 3, self.imgsz, self.imgsz).to(self.device)
        orig_size = torch.tensor([self.imgsz, self.imgsz])[None].to(self.device)
        blob      = {
            "images"           : dummy,
            "orig_target_sizes": orig_size,
        }
        for _ in range(n):
            _ = self(blob)


@torch.no_grad()
def export_trt(
    onnx_file  : pathlib.Path,
    engine_file: pathlib.Path,
    imgsz      : int  = 640,
    opset      : int  = 16,
    trt_p      : str  = "fp32",
    dla_core   : int  = 0,
    verbose    : bool = True
) -> pathlib.Path:
    onnx_file   = pathlib.Path(onnx_file)
    engine_file = pathlib.Path(engine_file)
    imgsz       = imgsz[0] if isinstance(imgsz, list | tuple) else imgsz

    if not onnx_file.exists():
        raise FileNotFoundError(f"Invalid ONNX file: {onnx_file}.")

    if trt_p not in supported_trt_p:
        raise ValueError(f"[fp] must be one of {supported_trt_p}, got {trt_p}.")

    # Setup
    logger        = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    builder       = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network       = builder.create_network(network_flags)
    parser        = trt.OnnxParser(network, logger)

    # Load ONNX model
    print(f"Loading ONNX model from: {onnx_file}.")
    with open(onnx_file, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX file!")

    # Create builder config
    config = builder.create_builder_config()
    memory_pool_limit = 8 << 30  # 1 << 30  1GB
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, memory_pool_limit)
    config.builder_optimization_level = 5   # Maximum optimization level

    if trt_p in ["fp16n32", "fp16ln32", "fp16lbn32", "fp16lbln32", "fp16",
                 "int8n32", "int8"]:
        if dla_core not in [None, -1]:  # Use DLA core
            config.DLA_core = dla_core
            config.default_device_type = trt.DeviceType.DLA
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
            print("Apply DLA core.")

    if trt_p in ["fp16n32", "fp16ln32", "fp16lbn32", "fp16lbln32", "fp16"]:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("Apply FP16 optimization.")
        else:
            print("Apply FP32 optimization.")
    elif trt_p in ["int8n32", "int8"]:
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("Apply INT8 optimization.")
        else:
            print("Apply FP32 optimization.")

    # Create optimization profile
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "images",
        min=(1,  3, imgsz, imgsz),
        opt=(16, 3, imgsz, imgsz),
        max=(32, 3, imgsz, imgsz)
    )
    profile.set_shape("orig_target_sizes", min=(1, 2), opt=(1, 2), max=(1, 2))
    config.add_optimization_profile(profile)

    # Retain FP32 for specific layers
    if opset == 16:
        if trt_p in ["fp16n32", "fp16ln32", "fp16lbn32", "fp16lbln32"]:
            layer_names = ["layernorm", "norm", "rms"]
            # norm_layers = fp32_layers.get(trt_p, [])
            for i in range(network.num_layers):
                layer = network.get_layer(i)
                # Heuristic: match common LayerNorm-related names
                if any(kw in layer.name.lower() for kw in layer_names):
                    '''
                    if len(norm_layers) > 0:
                        if any(nl in layer.name for nl in norm_layers):
                            print(f"Apply FP32 on LayerNorm-related layer: {layer.name}.")
                            layer.precision = trt.DataType.FLOAT
                            layer.set_output_type(0, trt.DataType.FLOAT)
                    else:
                        print(f"Apply FP32 on LayerNorm-related layer: {layer.name}.")
                        layer.precision = trt.DataType.FLOAT
                        layer.set_output_type(0, trt.DataType.FLOAT)
                    '''
                    print(f"Apply FP32 on LayerNorm-related layer: {layer.name}.")
                    layer.precision = trt.DataType.FLOAT
                    layer.set_output_type(0, trt.DataType.FLOAT)

    print("Building TensorRT engine...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build the engine.")

    print(f"Saving engine to {engine_file}")
    with open(engine_file, "wb") as f:
        f.write(serialized_engine)
    print("Engine export complete.")


# ----- Utils -----
def f1_score(predictions_path: str, ground_truths_path: str):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval_modified import COCOeval
    coco_gt      = COCO(ground_truths_path)
    gt_image_ids = coco_gt.getImgIds()

    with open(predictions_path, "r") as f:
        detection_data = json.load(f)
    filtered_detection_data = [item for item in detection_data if item["image_id"] in gt_image_ids]
    with open("./temp.json", "w") as f:
        json.dump(filtered_detection_data, f)

    coco_dt   = coco_gt.loadRes("./temp.json")
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Assuming the F1 score is at index 20 in the stats array
    return coco_eval.stats[20]  # Return the F1 score from the evaluation stats
    # return 0.85  # Simulated constant value for demo purposes


def get_model(model_path: str, imgsz: int = 640, device: torch.device = torch.device("cuda")):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} does not exist.")

    # You need to implement your model here
    # Load the exported TensorRT model
    trt_model = TRTInference(model_path, max_batch_size=1, device=device, imgsz=imgsz)

    return trt_model


def preprocess_image(image: np.ndarray, imgsz: int = 640, device: torch.device = torch.device("cuda")) -> dict:
    image     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB
    h0, w0, _ = image.shape
    image     = cv2.resize(image, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
    # image     = transforms(image).unsqueeze(0)  # Add batch dimension
    image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float().div(255.0).unsqueeze(0)
    blob      = {
        "images"           : image.to(device),
        "orig_target_sizes": torch.tensor([w0, h0])[None].to(device),
    }
    return blob


def postprocess_result(image_path: str, results: dict, conf_thres: dict = None) -> tuple[list, list, list]:
    if not results or len(results) == 0:
        return [], [], []
    boxes   = results["boxes" ].cpu().numpy().astype(np.float32)[0]  #.tolist()   # shape: (N, 4), XYXY format
    classes = results["labels"].cpu().numpy().astype(np.int32)[0]    #.tolist()   # shape: (N,)
    scores  = results["scores"].cpu().numpy().astype(np.float32)[0]  #.tolist()   # shape: (N,)

    if conf_thres:
        image_name = pathlib.Path(image_path).stem
        scene_id   = image_name.split("_")[1]
        conf_thres = conf_thres[scene_id]
        mask = np.zeros_like(scores, dtype=bool)
        for cls_id, thresh in enumerate(conf_thres[:5]):  # Limit to 5 classes for safety
            mask |= (classes == cls_id) & (scores >= thresh)
        boxes   = boxes[mask]
        classes = classes[mask]
        scores  = scores[mask]

    return boxes, classes, scores


def changeId(id: str) -> int:
    sceneList = ["M", "A", "E", "N"]
    cameraId  = int(id.split("_")[0].split("camera")[1])
    sceneId   = sceneList.index(id.split("_")[1])
    frameId   = int(id.split("_")[2])
    imageId   = int(str(cameraId) + str(sceneId) + str(frameId))
    return imageId
