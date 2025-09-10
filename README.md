以下是适配规范格式的中文版本`README.md`，确保内容清晰、结构合理，便于在代码仓库中展示：


# DSAC-T

## 参考
Distributional Soft Actor-Critic (DSAC)

## 系统要求
- 操作系统：Windows 7及以上版本 或 Linux
- Python 版本：3.8
- 安装路径：必须为英文（不含中文字符）


## 安装步骤
```bash
# 注意：安装路径中不可包含中文字符，否则可能导致运行失败
# 克隆仓库
git clone git@github.com:Jingliang-Duan/DSAC-T
cd DSAC-T

# 创建并激活conda环境
conda env create -f DSAC2.0_environment.yml
conda activate DSAC2.0

# 安装DSAC2.0
pip install -e.
```


## 训练
以下是在两个环境中运行DSAC-T的示例，执行以下命令训练策略：

```bash
cd example_train

# 训练摆锤任务
python main.py

# 训练人形机器人任务（执行前需先安装Mujoco和Mujoco-py）
python dsac_mlp_humanoidconti_offserial.py
```

训练结果将保存在 `DSAC-T/results` 文件夹中。


## 仿真
在 `DSAC-T/results` 文件夹中，选择需要应用策略的文件夹路径，并指定对应的PKL文件进行仿真：

```bash
python run_policy.py
# 注意：Windows系统运行前可能需要执行：pip install imageio-ffmpeg
```

运行后，仿真视频及状态-动作曲线图将保存在 `DSAC-T/figures` 文件夹中。


## 致谢
感谢清华大学车辆与运载学院智能驾驶实验室（iDLab）的所有成员，他们为DSAC-T的开发提供了卓越贡献和宝贵建议。
