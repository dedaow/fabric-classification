import os
import pandas as pd


def data_augmentation(folder_path, name, step_size, group_size, save_folder):

    # 获取文件夹中包含"train"的CSV文件列表
    csv_files = [file for file in os.listdir(
        folder_path) if name in file and file.endswith('.csv')]

    # 创建一个空的DataFrame来存储合并后的数据
    combined_data = pd.DataFrame()

    # 逐个读取并合并CSV文件
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        combined_data = pd.concat([combined_data, df])

    nums = len(combined_data) // group_size * group_size - group_size
    for i in range(0, nums, step_size):
        # 计算当前分组的起始索引和结束索引
        start_idx = i
        end_idx = i + group_size

        # 获取当前分组的数据
        group_data = combined_data.iloc[start_idx:end_idx]

        # 生成新文件名
        file_name = f"{name}_{i}.csv"
        file_path = os.path.join(save_folder, file_name)

        # 将当前分组的数据保存到新文件中
        group_data.to_csv(file_path, index=False)


if __name__ == "__main__":
    # 设置包含CSV文件的文件夹路径
    folder_path = 'data/train'
    save_folder = 'data/train_aug'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    material_names = ['glass', 'white_foam', 'blue_foam', 'woven_fabric',
                      'gray_sponge', 'ribbed_fabric', 'yarn_screen', 'glossy_wood', 'rough_wood',
                      'twill', 'Popline', 'Polyester', 'cotton',
                      'olyester', 'Dobby', 'Nylon', 'Stretch',
                      'Crimp', 'plush', 'Combed', 'Tulle ']
    step_size = 10
    group_size = 300
    for name in material_names:
        data_augmentation(folder_path, name, step_size,
                          group_size, save_folder)
