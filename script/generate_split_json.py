import os
import argparse

def collect_video_ids_from_label_dirs(base_split_dir):
    """
    从 base_split_dir 下的 0_real 和 1_fake 目录中收集所有视频ID（文件夹名）
    """
    video_ids = set()

    for label in ["0_real", "1_fake"]:
        label_dir = os.path.join(base_split_dir, label)
        if os.path.exists(label_dir) and os.path.isdir(label_dir):
            for item in os.listdir(label_dir):
                item_path = os.path.join(label_dir, item)
                if os.path.isdir(item_path):  # 确保是文件夹
                    video_ids.add(item)

    return sorted(video_ids)

def write_json_like_txt(video_ids, output_json_path):
    """
    写成每行一个ID的“伪JSON”文件（实际是纯文本）
    """
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        for vid in video_ids:
            f.write(vid + '\n')
    print(f"✅ 已写入 {len(video_ids)} 个视频ID 到 {output_json_path}")

def main(data_root, dataset_name, split_name, output_split_dir):
    """
    data_root: 数据根目录，如 ./data
    dataset_name: 如 Text2Video_Zero
    split_name: 如 Test / Train / Val
    output_split_dir: 输出划分文件的目录，如 ./split_ori
    """
    base_split_dir = os.path.join(data_root, dataset_name, split_name)
    if not os.path.exists(base_split_dir):
        raise ValueError(f"路径不存在: {base_split_dir}")

    video_ids = collect_video_ids_from_label_dirs(base_split_dir)
    if not video_ids:
        print("⚠️ 未找到任何视频文件夹！请检查路径。")
        return

    output_file = os.path.join(output_split_dir, f"{split_name.lower()}.json")
    write_json_like_txt(video_ids, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 DeCoF 项目所需的 .json 划分文件（伪JSON格式）")
    parser.add_argument("--data_root", default="./data", help="数据根目录，默认 ./data")
    parser.add_argument("--dataset", default="Text2Video_Zero", help="数据集名称")
    parser.add_argument("--split", required=True, choices=["Train", "Val", "Test"], help="要生成的划分：Train / Val / Test")
    parser.add_argument("--output_split_dir", default="./split", help="输出 .json 文件的目录")

    args = parser.parse_args()

    main(
        data_root=args.data_root,
        dataset_name=args.dataset,
        split_name=args.split,
        output_split_dir=args.output_split_dir
    )