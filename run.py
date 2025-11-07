import os, sys

def main(to_pred_dir, submit_path):
    run_py = os.path.abspath(__file__)
    dir = os.path.dirname(run_py)
    run_sh = os.path.join(dir, "run.sh")

    #1> 默认python是3.7, 如果使用python主版本是3.7, 可以不执行os.system, 而直接执行python的代码逻辑

    #2> 如果使用非3.7版本的python, 则需要通过os.system调用shell脚本选择其他python主版本执行
    cmd = f"/bin/bash {run_sh} {to_pred_dir} {submit_path}"
    os.system(cmd)

if __name__ == "__main__":
    to_pred_dir = sys.argv[1]
    submit_path = sys.argv[2]
    main(to_pred_dir, submit_path)