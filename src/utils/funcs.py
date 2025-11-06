
import json
import torch 


def load_json(path):
	d = {}
	with open(path, mode="r") as f:
		d = json.load(f)
	return d


def get_available_device():
    if torch.backends.mps.is_available():  # 检查 Apple MPS 是否可用
        return torch.device("mps")
    elif torch.cuda.is_available():        # 检查 NVIDIA CUDA 是否可用
        return torch.device("cuda")
    else:
        return torch.device("cpu")