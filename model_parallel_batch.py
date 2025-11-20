import time 
from concurrent.futures import ThreadPoolExecutor
import torch
import os
import sys
import pandas as pd
import torch.nn.functional as F
from PIL import Image
import csv 
from torchvision import transforms
from tqdm import tqdm 
import numpy as np
from decord import VideoReader, cpu
# from rts import Sync


# 模型缓存全局变量
_cached_model = None
_cached_model_path = None
_cached_device = None

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from detect import Detector  # 用户自定义模块


# ===================== 视频帧提取 =====================
def extract_frames_in_memory(video_path, num_total_frames=32, num_sampled=8):
    """
    从视频中读取帧到内存（不落盘），输出为 list[Tensor]
    """
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        if total_frames == 0:
            print(f"[警告] 视频无帧: {video_path}")
            return []

        indices = np.linspace(0, total_frames - 1, num_total_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()  # (T, H, W, 3)

        if frames.shape[0] < num_total_frames:
            last = np.repeat(frames[-1][None, :, :, :], num_total_frames - frames.shape[0], axis=0)
            frames = np.concatenate([frames, last], axis=0)

        step = num_total_frames // num_sampled
        frames = frames[::step][:num_sampled]

        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        tensor_frames = [test_transform(Image.fromarray(frame)) for frame in frames]
        return tensor_frames
    except Exception as e:
        print(f"[错误] 解码失败 {video_path}: {e}")
        return []


# ===================== 单视频帧处理函数 =====================
def process_single_video(video_path):
    frames = extract_frames_in_memory(video_path)
    return frames


# ===================== 批量推理函数 =====================
def batch_predict(video_tensors, model, device):
    model.eval()
    results = []
    with torch.no_grad():
        video_tensors = [torch.stack(v, dim=0) for v in video_tensors if len(v) > 0]
        if len(video_tensors) == 0:
            return []
        video_tensors = torch.stack(video_tensors, dim=0).to(device)  # [B, T, 3, 224, 224]

        output = model(video_tensors)
        prob = F.softmax(output, dim=1)
        fake_conf = prob[:, 1].cpu().numpy().tolist()
        for conf in fake_conf:
            results.append((1 if conf > 0.5 else 0, round(conf, 4)))

    del video_tensors
    torch.cuda.empty_cache()
    return results


# ===================== 主函数 =====================
def main_parallel(to_pred_dir, result_save_path, workers=20, batch_size=512):
    filenames = sorted(os.listdir(to_pred_dir))
    filepaths = [os.path.join(to_pred_dir, f) for f in filenames]
    # sync = Sync(0, len(filenames))
    # === 模型加载（仅一次） ===
    global _cached_model, _cached_model_path, _cached_device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "output/DeCoF_ivy_base_11_07_22_49_35/weights/50_0.9926_val.tar"

    if _cached_model is None or _cached_model_path != model_path:
        print(f"[模型加载] 从 {model_path} 加载中...")
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

    # === 输出CSV初始化 ===
    with open(result_save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "confidence"])

    # === 主循环：分批提取 + 推理 + 写出 ===
    start_time = time.time()
    total_videos = len(filepaths)
    index = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for i in tqdm(range(0, total_videos, batch_size),
                      desc="Processing videos (streaming mode)", ncols=100):
            batch_files = filepaths[i:i + batch_size]
            batch_names = filenames[i:i + batch_size]

            # 提取帧（并行CPU）
            video_tensors = list(executor.map(process_single_video, batch_files))

            # GPU 推理
            batch_results = batch_predict(video_tensors, model, device)

            # 写入结果（逐批写盘）
            with open(result_save_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for name, res in zip(batch_names, batch_results):
                    label, conf = res
                    writer.writerow([name, label, conf])
            index += len(batch_results)
            # sync.update(index)
            # 显存清理
            torch.cuda.empty_cache()

    print(f"✅ 全部完成，总耗时 {time.time() - start_time:.2f} 秒")
    return pd.read_csv(result_save_path)


# ===================== 程序入口 =====================
if __name__ == "__main__":
    pred_dir = sys.argv[1]
    result_save_path = sys.argv[2]
    result = main_parallel(pred_dir, result_save_path)
