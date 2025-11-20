import time 
import os
import sys
import pandas as pd
import shutil 
import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
sys.path.append(os.path.join(os.path.dirname(__file__), 'src')) 
from detect import Detector  
import cv2 


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
    """
    对视频进行 Deepfake 检测预测。
    
    Args:
        file_path (str): 输入视频的路径（如 'xxx.mp4'）
        model_path (str): 模型权重文件路径（.pth 文件，需包含 "model" 键）

    Returns:
        tuple: (label, confidence)
            - label: int, 0 表示 real, 1 表示 fake（可根据你的数据集定义调整）
            - confidence: float, 属于 fake 的置信度，范围 [0,1]，保留4位小数
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # === 1. 加载模型 ===
    model = Detector().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.net_all.load_state_dict(checkpoint["model"])
    model.eval()

    # === 2. 调用用户提供的函数提取帧 ===
    t = time.time()
    frame_folder_path = extract_and_process_video(file_path,"output_tmp")  # 返回帧所在的文件夹路径
    print(f"extract time is {time.time()-t} ")
    n_frames = len([f for f in os.listdir(frame_folder_path) if f.endswith('.jpg')])
    if n_frames == 0:
        return 0 , 1 
        # raise ValueError(f"No frames found in {frame_folder_path}. Check your extract_frames implementation.")
    
    # === 3. 构建测试阶段的 transform（与 Dataset 中完全一致）===
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # === 4. 加载所有帧 ===
    frames = []
    for i in range(n_frames):
        img_path = os.path.join(frame_folder_path, f"{str(i).zfill(3)}.jpg")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Frame {img_path} not found. Ensure all required frames exist.")
        
        img = Image.open(img_path).convert("RGB")
        img_tensor = test_transform(img)  # [C, H, W]
        frames.append(img_tensor)

    # Stack to [T, C, H, W]
    video_tensor = torch.stack(frames, dim=0).unsqueeze(0).to(device)  # [1, T, C, H, W]

    # === 5. 推理 ===
    with torch.no_grad():
        output = model(video_tensor)  # shape: [1, 2]
        prob = F.softmax(output, dim=1)
        confidence_fake = prob[0, 1].item()  # 第1类为 fake

    # === 6. 返回结果 ===
    label = 1 if confidence_fake > 0.5 else 0
    confidence = round(confidence_fake, 4)

    return (label, confidence)

if __name__ == "__main__":
    r , c = predict_todo(file_path="input_test/0a169eb3345847089028dbac81331cf6a25f4107e80cce3676036d3dcc748eec.mp4" ,model_path= "output/DeCoF_ivy_base_11_07_22_49_35/weights/9_0.9745_val.tar")
    print(f"result is {r},confidence is {c}")