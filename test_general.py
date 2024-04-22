from utils import *


def read_txt_to_list(file_path):
    try:
        # 打开文件
        with open(file_path, 'r') as file:
            # 读取文件内容
            data = file.read()
            # 按换行符分割文本并转换成列表
            data_list = data.split('\n')
            return data_list
    except FileNotFoundError:
        print("文件不存在！")
        return None

# 调用函数并传入文件路径
file_path = "txt/N-spring2D_N=10_dt=0.001.txt"  # 替换成你的文件路径
data_list = read_txt_to_list(file_path)
for i in range(len(data_list)-1):
    data_list[i] = float(data_list[i])
data_list = data_list[:-1]
print(data_list)

plot1(data_list, 100, 10, "N-spring2D_N=5_dt=0.005")

print(max(data_list))