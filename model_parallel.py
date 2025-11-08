import time 
from concurrent.futures import ThreadPoolExecutor
import torch
import os
import sys
import pandas as pd
import shutil 
# from rts import Sync
import os
import torch
import torch.nn.functional as F
from PIL import Image
import cv2 
import csv 
from torchvision import transforms
sys.path.append(os.path.join(os.path.dirname(__file__), 'src')) 
from detect import Detector  
from tqdm import tqdm 
_cached_model = None
_cached_model_path = None
_cached_device = None

import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from torchvision import transforms

def extract_frames_in_memory(video_path, num_total_frames=32, num_sampled=8):
    """
    从视频中读取帧到内存（不落盘），输出为 list[Tensor]
    """
    try:
        # 1️⃣ 用 decord 打开视频
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        if total_frames == 0:
            print(f"视频无帧: {video_path}")
            return []

        # 2️⃣ 均匀采样 32 帧（或补齐）
        indices = np.linspace(0, total_frames - 1, num_total_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()  # shape: (T, H, W, 3)

        # 如果帧数不足，则重复最后一帧补齐
        if frames.shape[0] < num_total_frames:
            last = np.repeat(frames[-1][None, :, :, :], num_total_frames - frames.shape[0], axis=0)
            frames = np.concatenate([frames, last], axis=0)

        # 3️⃣ 再次下采样成 num_sampled 帧（如 8）
        step = num_total_frames // num_sampled
        frames = frames[::step][:num_sampled]  # shape: (num_sampled, H, W, 3)

        # 4️⃣ 中心裁剪 + 归一化
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 5️⃣ 转成 Tensor 列表（每帧一个 Tensor）
        tensor_frames = [test_transform(Image.fromarray(frame)) for frame in frames]
        return tensor_frames

    except Exception as e:
        print(f"解码失败 {video_path}: {e}")
        return []
    
    
def process_single_video(video_path):
    frames = extract_frames_in_memory(video_path)
    return frames  # shape: [num_frames, 3, 224, 224]

def batch_predict(video_list, model, device, batch_size=680):
    model.eval()
    results = []
    with torch.no_grad():
        for i in tqdm(range(0, len(video_list), batch_size), desc="Processing videos", ncols=100):
            batch_videos = video_list[i:i+batch_size]
            video_tensors = [torch.stack(v, dim=0) for v in batch_videos]
            video_tensors = torch.stack(video_tensors, dim=0).to(device)  # [B, T, 3, 224, 224]

            output = model(video_tensors)
            prob = F.softmax(output, dim=1)
            fake_conf = prob[:, 1].cpu().numpy().tolist()
            print(len(fake_conf))
            for conf in fake_conf:
                results.append((1 if conf > 0.5 else 0, round(conf, 4)))
    return results

def main_parallel(to_pred_dir, result_save_path, workers=30):
    filenames = sorted(os.listdir(to_pred_dir))
    filepaths = [os.path.join(to_pred_dir, f) for f in filenames]

    # 1️⃣ CPU 多线程提取帧
    with ThreadPoolExecutor(max_workers=workers) as executor:
        video_tensors = list(
            tqdm(
                executor.map(process_single_video, filepaths),
                total=len(filepaths),
                desc="Extracting frames",
                ncols=100,
                unit="video"
            )
        )

    # 2️⃣ GPU 批量推理
    global _cached_model, _cached_model_path, _cached_device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_path = "output/DeCoF_ivy_base_11_07_22_49_35/weights/50_0.9926_val.tar" 
    # === 1. 模型懒加载 + 缓存 ===
    if _cached_model is None or _cached_model_path != model_path:
        print(f"Loading model from {model_path} on {device}...")
        model = Detector().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.net_all.load_state_dict(checkpoint["model"])
        model.eval()
        _cached_model = model
        _cached_model_path = model_path
        _cached_device = device
    else:
        model = _cached_model
        device = _cached_device
    model = _cached_model
    t = time.time()
    print(f"batch predict begin")
    results = batch_predict(video_tensors, model, torch.device('cuda'))
    print(f"predict time is {time.time()-t}")

    # 3️⃣ 写出结果
    with open(result_save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "confidence"])
        for filename, (label, conf) in zip(filenames, results):
            writer.writerow([filename, label, conf])
    
    
    return pd.read_csv(result_save_path)



# ---------------------------------------------
# 使用示例，请勿修改
# ---------------------------------------------
if __name__ == "__main__":
   
    pred_dir = sys.argv[1]
    result_save_path = sys.argv[2]
    result = main_parallel(pred_dir, result_save_path)