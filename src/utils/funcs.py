
import json
import torch 


def load_json(path):
	d = {}
	with open(path, mode="r") as f:
		d = json.load(f)
	return d


def get_available_device():
    if torch.backends.mps.is_available():  # Apple Silicon (M1/M2/M3/M4)
        print("device is mps")
        return torch.device("mps")
    elif torch.cuda.is_available():        # NVIDIA GPU
        print("device is cuda")
        return torch.device("cuda")
    else:                                  # CPU fallback
        print("device is cpu")
        return torch.device("cpu")