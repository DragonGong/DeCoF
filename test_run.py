import os
import sys
import pandas as pd

import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from detect import Detector  



def extract_frames(video_path):
    """
    提取视频中的帧。
    
    Args:
        video_path (str): 视频文件的路径。
        
    Returns:
        str: 包含提取帧的文件夹路径。
    """
    raise NotImplementedError("This function should be implemented by the user.")

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
    frame_folder_path = extract_frames(file_path)  # 返回帧所在的文件夹路径
    n_frames = len([f for f in os.listdir(frame_folder_path) if f.endswith('.jpg')])
    if n_frames == 0:
        raise ValueError(f"No frames found in {frame_folder_path}. Check your extract_frames implementation.")
    
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
    r , c = predict_todo(file_path="" ,model_path= "")
    print(f"result is {r},confidence is {c}")