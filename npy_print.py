import numpy as np

def read_and_print_npy(file_path):
    try:
        # 设置 allow_pickle 为 True
        data = np.load(file_path, allow_pickle=True)

        # 打印数组数据
        print("Array Data:")
        print(data)

        # 如果需要，你可以访问数组的形状、数据类型等信息
        print("\nArray Shape:", data.shape)
        print("Array Data Type:", data.dtype)

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# 替换 'your_file.npy' 为你实际的文件路径
file_path = '/home/yz/DSAC-T-main/results/DSAC2/evaluator/iter8100_ep3.npy'

# 调用函数读取并打印 npy 文件
read_and_print_npy(file_path)
