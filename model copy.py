import os
import sys
import pandas as pd

from rts import Sync

# ---------------------------------------------
# 预测函数示例, 选手需自行实现
# ---------------------------------------------
def predict_todo(file_path, model_path='model.pth'):
    # 选手自行实现预测逻辑
    # TODO

    # label 作为检测结果标签，仅做标记参考，不作为判分指标，real fake 1 0 均可
    # confidence精度尽量高，建议.4f，以confidence计算eer
    label = 1
    confidence = 0.8005
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