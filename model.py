import os
import sys
import pandas as pd

from rts import Sync
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

def main(to_pred_dir, result_save_path):
    filenames = os.listdir(to_pred_dir)
    filepaths = [os.path.join(to_pred_dir, filename) for filename in filenames]
    sync = Sync(0, len(filenames))
    
    # 加载模型，选手自行实现
    # TODO
    # 可以修改整体写法和逻辑，需保证生成结果的csv结构不变

    results = []
    for i, (filename, filepath) in enumerate(zip(filenames, filepaths)):
        (label, confidence) = predict_todo(filepath, model_path="your_model.pth")
        # 自行打印调试信息
        results.append([filename, label, confidence])
        sync.update(i+1)
        
    results = pd.DataFrame(results, columns=["filename", "label", "confidence"])
    # 输出文件为csv格式
    save_dir = os.path.dirname(result_save_path)
    os.makedirs(save_dir, exist_ok=True)
    results.to_csv(result_save_path, index=None, float_format='%.4f')
    return results
    # 如果因程序异常中断、或答题超时程序被动杀停，导致无csv文件生成，则本次执行无成绩。可考虑文件实时落地
    # 如果程序异常中断，但有csv文件生成，则本次执行有成绩，且占用当日1次执行成功次数，但可能由于csv写入不完整导致成绩不完整

# ---------------------------------------------
# 使用示例，请勿修改
# ---------------------------------------------
if __name__ == "__main__":
   
    pred_dir = sys.argv[1]
    result_save_path = sys.argv[2]
    result = main(pred_dir, result_save_path)