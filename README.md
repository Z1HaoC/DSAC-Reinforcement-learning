Distributional Soft Actor-Critic (DSAC)
系统要求
Windows 7 及以上版本或 Linux 系统
Python 3.8
安装路径必须为英文
安装步骤
bash
# 请确保安装路径中不包含中文字符，否则可能导致执行失败
# 克隆DSAC-T仓库
git clone git@github.com:Jingliang-Duan/DSAC-T
cd DSAC-T

# 创建conda环境
conda env create -f DSAC2.0_environment.yml
conda activate DSAC2.0

# 安装DSAC2.0
pip install -e.



训练
以下是在两个环境上运行 DSAC-T 的示例。通过运行以下命令训练策略：
bash
cd example_train

# 训练摆锤任务
python main.py

# 训练人形机器人任务。执行此文件前，需先安装Mujoco和Mujoco-py
python dsac_mlp_humanoidconti_offserial.py
训练完成后，结果将存储在 “DSAC-T/results” 文件夹中。
仿真
在 “DSAC-T/results” 文件夹中，选择要应用政策进行仿真的文件夹路径，并挑选合适的 PKL 文件用于仿真。
bash
python run_policy.py
# 在Windows系统上运行此文件前，可能需要执行“pip install imageio-ffmpeg”
运行后，仿真视频以及状态与动作曲线图表将存储在 “DSAC-T/figures” 文件夹中。
致谢
感谢清华大学车辆与运载学院智能驾驶实验室（iDLab）的所有成员，他们为 DSAC-T 做出了卓越贡献并提供了有益建议。
