import os

import torch

T_RAGX_DEVICE_ENV = "T_RAGX_DEVICE"


def is_mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def get_default_device_name() -> str:
    env_device = os.environ.get(T_RAGX_DEVICE_ENV)
    if env_device:
        return env_device
    if is_cuda_available():
        return "cuda"
    if is_mps_available():
        return "mps"
    return "cpu"


def get_torch_device(device=None) -> torch.device:
    if device is None:
        device = get_default_device_name()
    return torch.device(device)


def get_comet_predict_kwargs() -> dict:
    device_name = get_default_device_name()
    if device_name == "cuda":
        return {"gpus": 1, "accelerator": "auto"}
    if device_name == "mps":
        return {"gpus": 1, "accelerator": "mps"}
    return {"gpus": 0, "accelerator": "cpu"}


def get_llama_cpp_gpu_layers() -> int:
    if os.environ.get(T_RAGX_DEVICE_ENV) == "cpu":
        return 0
    if is_mps_available() or is_cuda_available():
        return -1
    return 0
