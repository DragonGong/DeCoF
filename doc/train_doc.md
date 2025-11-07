- 计算设备：首先检查有没有mps，再检查cuda，然后再cpu



首先运行：

~~~
# zsh or bash bash/train_ds_m.sh 
# python script/train_ds_m.py --input '/Volumes/My Passport/dataset/decor/test_train' --output ./data --dataset test_data --split Train --label 1_fake
# python script/train_ds_m.py --input '/Volumes/My Passport/dataset/decor/test_val' --output ./data --dataset test_data --split Train --label 0_real
# python script/train_ds_m.py --input '/Volumes/My Passport/dataset/decor/test_val/fake' --output ./data --dataset test_data --split Val --label 1_fake
python script/train_ds_m.py --input '/Volumes/My Passport/dataset/decor/test_val/real' --output ./data --dataset test_data --split Val --label 0_real
~~~

来生成训练数据，原始的输入是文件夹，test_data是数据集名称，里面都是视频文件。



~~~
#zsh or bash bash/split_generate.sh
python script/generate_split_json.py  --data_root ./data --dataset test_data --split Val --output_split_dir datas/split
~~~

输入dataroot，然后是训练数据的名称，然后是split 到Train 还是Val 还有Test 

然后再base.json修改

~~~
{
    "epoch":5,
    "batch_size":32,
    "image_size":224,
    "n_frames":8,
    "subdatasets_name":"test_data" # 训练数据名称来指定训练哪个数据
}
~~~

然后再

src/utils/initialize.py

中改data的root,这是项目bug



然后直接bash/train.sh

~~~
CUDA_VISIBLE_DEVICES=0 python src/train.py src/configs/base.json -n DeCoF_test
~~~

-n 是模型名称

多卡

~~~
torchrun --nproc_per_node=4 src/train_multi_gpu.py src/configs/base.json -n DeCoF_ivy 
~~~



