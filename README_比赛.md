### 平台运行环境
`python==3.9`
~~~
cd /home/workspace/model
python3.9 -m pip install -r requirements.txt
~~~

测试
~~~
python3.9 /home/workspace/model/run.py /home/workspace/input /home/workspace/output/result.csv
~~~
### 本地开发
- 新建自己的分支 如 `feat/xxx`
- 写完代码后记得合一下main分支，然后提交代码并申请pr，提交完后与owner联系完成合并
  - 安装了新的依赖记得更新requirements.txt
- main分支不可直接推送
### 本地测试
~~~
python run.py ./input_test ./output_test/result.csv
~~~

### 注意事项(在实验平台中)
- 所有代码及模型文件必须放在`/home/workspace/model`此路径下，如该路径不存在，请自行创建
- `/home/workspace/model`下的组织结构有如下要求：
  - `/home/workspace/model/run.py`（！！！必须有，从示例中复制，不可修改）
  - `/home/workspace/model/run.sh`（！！！必须有，从示例中复制，慎重修改，只建议修改环境变量和python启动版本）
  - `/home/workspace/model/model.py`（可改名，可修改，可引入其他代码块。如果改名，请对应修改run.sh中的内容）
  - `/home/workspace/model/requirements.txt`（可省略，用于在实训环境中安装依赖）
- `model.py`解析
  - 模型主推理逻辑，根据选手自身需求添加、修改。
  - 示例代码中包含：
    - 引入：`from rts import Sync`，引入Sync对象，将把选手更新的推理进度同步到web页面。
    - 初始化：`sync = Sync(0, len(prompts))`，初始化Sync对象，用于同步推理进度。 第二个入参为最大进度，请根据实际情况修改。
    - 更新：`sync.update(i+1)`，更新Sync对象，用于更新推理进度，推理进度更新频率取决于选手`sync.update`频率。
- `run.sh`解析
  - 该脚本文件中仅建议修改推理所python版本，需要哪个版本就启用哪条命令
  - 该脚本还允许选手使用环境变量，请根据实际需求来设置环境变量。
- 依赖安装：请在镜像制作前完成依赖安装

