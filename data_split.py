import os
import shutil
from glob import glob
import numpy as np


def split_data(src_directory, dst_directory, names, train_ratio=0.9, test_ratio=0.1):
    # assert train_ratio + test_ratio + val_ratio == 1, "Ratios must sum up to 1"
    assert 0.99999999 <= train_ratio + test_ratio <= 1.0000001, "Ratios must sum up to 1"

    # 确保目标目录存在
    for folder in ['train', 'test']:
        os.makedirs(os.path.join(dst_directory, folder), exist_ok=True)

    # 遍历每种类型的数据
    for type in names:
        # 获取所有相应颜色的文件
        files = glob(os.path.join(src_directory, f'{type}*.csv'))
        # files = glob(src_directory + '/*.csv')
        # 打乱文件顺序
        np.random.shuffle(files)

        # 计算分割点
        train_end = int(len(files) * train_ratio)

        # 分配文件到训练、测试和验证集
        train_files = files[:train_end]
        test_files = files[train_end:]

        # 移动文件到相应的目标文件夹
        for file in train_files:
            shutil.copy(file, os.path.join(dst_directory,
                        'train', os.path.basename(file)))
        for file in test_files:
            shutil.copy(file, os.path.join(
                dst_directory, 'test', os.path.basename(file)))


# 源目录和目标目录
src_directory = 'csv_files'
dst_directory = 'data'  # 目标目录改为 /data

material_names = ['glass', 'white_foam', 'blue_foam', 'woven_fabric',
                  'gray_sponge', 'ribbed_fabric', 'yarn_screen', 'glossy_wood', 'rough_wood',
                  'twill', 'Popline', 'Polyester', 'cotton',
                  'olyester', 'Dobby', 'Nylon', 'Stretch',
                  'Crimp', 'plush', 'Combed', 'Tulle ']
split_data(src_directory, dst_directory, material_names, train_ratio=0.9,
           test_ratio=0.1)
