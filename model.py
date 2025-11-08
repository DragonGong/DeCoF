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

_cached_model = None
_cached_model_path = None
_cached_device = None


def extract_and_process_video(video_path, output_dir, num_total_frames=32, num_sampled=8):
    # 如果 output_dir 已存在，先删除整个目录
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return False

    frames = []
    while len(frames) < num_total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()

    if len(frames) == 0:
        print(f"视频无帧: {video_path}")
        return False

    if len(frames) < num_total_frames:
        print(f"警告: 视频 {video_path} 帧数不足32 ({len(frames)})，将用最后一帧补足")
        while len(frames) < num_total_frames:
            frames.append(frames[-1])

    step = num_total_frames // num_sampled  # 4
    sampled_indices = [i * step for i in range(num_sampled)]

    os.makedirs(output_dir, exist_ok=True)

    for idx, frame_idx in enumerate(sampled_indices):
        frame = frames[frame_idx]
        h, w = frame.shape[:2]
        crop_size = min(h, w)
        start_h = (h - crop_size) // 2
        start_w = (w - crop_size) // 2
        cropped = frame[start_h:start_h + crop_size, start_w:start_w + crop_size]

        img_pil = Image.fromarray(cropped)
        img_path = os.path.join(output_dir, f"{idx:03d}.jpg")
        img_pil.save(img_path, quality=95)

    return output_dir


def predict_todo(file_path, model_path='model.pth'):
    global _cached_model, _cached_model_path, _cached_device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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

    # === 2. 提取帧 ===
    frame_folder_path = extract_and_process_video(file_path, "output_tmp")
    n_frames = len([f for f in os.listdir(frame_folder_path) if f.endswith('.jpg')])
    if n_frames == 0:
        # 清理临时目录（可选）
        shutil.rmtree("output_tmp", ignore_errors=True)
        return 0, 1.0

    # === 3. 构建 transform ===
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # === 4. 加载帧 ===
    frames = []
    for i in range(n_frames):
        img_path = os.path.join(frame_folder_path, f"{str(i).zfill(3)}.jpg")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Frame {img_path} not found.")
        img = Image.open(img_path).convert("RGB")
        frames.append(test_transform(img))

    video_tensor = torch.stack(frames, dim=0).unsqueeze(0).to(device)

    # === 5. 推理 ===
    with torch.no_grad():
        output = model(video_tensor)
        prob = F.softmax(output, dim=1)
        confidence_fake = prob[0, 1].item()

    # === 6. 清理临时文件（重要！避免磁盘爆满）===
    shutil.rmtree("output_tmp", ignore_errors=True)

    label = 1 if confidence_fake > 0.5 else 0
    confidence = round(confidence_fake, 4)
    return (label, confidence)

def main(to_pred_dir, result_save_path):
    filenames = sorted(os.listdir(to_pred_dir))
    filepaths = [os.path.join(to_pred_dir, filename) for filename in filenames]
    total = len(filenames)
    # sync = Sync(0, total)

    save_dir = os.path.dirname(result_save_path)
    os.makedirs(save_dir, exist_ok=True)

    # 不检查已有内容
    with open(result_save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "confidence"])  # 写 header
        f.flush()

        for i, (filename, filepath) in enumerate(zip(filenames, filepaths)):
            try:
                label, confidence = predict_todo(filepath, model_path="output/DeCoF_ivy_base_11_07_22_49_35/weights/50_0.9926_val.tar")
                writer.writerow([filename, label, f"{confidence:.4f}"])
                f.flush()  # 关键：立即落盘
            except Exception as e:
                print(f"Error on {filename}: {e}", file=sys.stderr)
                # 当 fake 处理”
                writer.writerow([filename, 0, 1.0])
                f.flush()

            # sync.update(i + 1)  

    return pd.read_csv(result_save_path)
    # 如果因程序异常中断、或答题超时程序被动杀停，导致无csv文件生成，则本次执行无成绩。可考虑文件实时落地
    # 如果程序异常中断，但有csv文件生成，则本次执行有成绩，且占用当日1次执行成功次数，但可能由于csv写入不完整导致成绩不完整

# ---------------------------------------------
# 使用示例，请勿修改
# ---------------------------------------------
if __name__ == "__main__":
   
    pred_dir = sys.argv[1]
    result_save_path = sys.argv[2]
    result = main(pred_dir, result_save_path)