import os
import cv2
from PIL import Image
import argparse
from tqdm import tqdm
def extract_and_process_video(video_path, output_dir, num_total_frames=32, num_sampled=8):
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

    return True

def is_valid_video_file(filename):
    """忽略 macOS 的 ._ 文件，只保留 .mp4"""
    if filename.startswith('._'):
        return False
    if not filename.lower().endswith('.mp4'):
        return False
    return True

def main(input_folder, output_root, dataset_name="Text2Video_Zero", split="Test", label="0_real"):
    all_files = os.listdir(input_folder)
    video_files = [f for f in all_files if is_valid_video_file(f)]

    if not video_files:
        print("输入文件夹中没有找到有效的 .mp4 视频文件（已忽略 ._ 开头的文件）！")
        return

    print(f"共找到 {len(video_files)} 个有效视频文件，将放入 {label} 类别下。")

    # 构建目标路径: data/{dataset}/{split}/{label}/{video_id}/
    base_output_dir = os.path.join(output_root, dataset_name, split, label)

    for video_file in tqdm(video_files, desc="Processing videos", ncols=100, unit="video"):
        video_id = os.path.splitext(video_file)[0]  # 仅文件名，不含扩展名
        video_path = os.path.join(input_folder, video_file)
        output_subdir = os.path.join(base_output_dir, video_id)  # 注意：这里不再拼 label_

        # print(f"处理: {video_file} -> {output_subdir}")
        success = extract_and_process_video(video_path, output_subdir)
        if not success:
            print(f"跳过: {video_file}")

    print("✅ 所有视频处理完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="生成符合 DeCoF 项目要求的训练数据结构（正确嵌套 label 目录）"
    )
    parser.add_argument("--input", required=True, help="包含MP4视频的输入文件夹")
    parser.add_argument("--output", default="./data", help="输出根目录（默认 ./data）")
    parser.add_argument("--dataset", default="Text2Video_Zero", help="数据集名称")
    parser.add_argument("--split", default="Test", choices=["Train", "Val", "Test"], help="划分类型")
    parser.add_argument("--label", default="0_real", choices=["0_real", "1_fake"], help="标签目录名")

    args = parser.parse_args()

    main(
        input_folder=args.input,
        output_root=args.output,
        dataset_name=args.dataset,
        split=args.split,
        label=args.label
    )