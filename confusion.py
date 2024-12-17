
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from model import CNN1D_A as CNN1D
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from train import CustomDataset



from matplotlib.colors import LinearSegmentedColormap

def get_confusion_matrix(true_labels, pred_labels, output_name='train'):
    cm = confusion_matrix(true_labels, pred_labels)
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_percent = (cm / cm_sum) * 100
    cm_percent = np.nan_to_num(cm_percent, nan=0)  # 将NaN替换为0

    # 创建一个与cm_percent形状相同的空字符串数组用于注解
    annot_array = [['' for _ in range(cm.shape[1])] for _ in range(cm.shape[0])]

    # 仅将非零值的位置替换为格式化后的字符串
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm_percent[i, j] != 0:
                annot_array[i][j] = '{:.1f}'.format(cm_percent[i, j])

                # 创建包含更暗红色调的自定义颜色映射
    colors = ["white", "#ffe0e0", "#ff9999", "#ff6666", "#ff3333"]  # 更暗的红色系列
    n_bins = len(colors)
    cmap = LinearSegmentedColormap.from_list("", colors, N=n_bins)

    #cmap = LinearSegmentedColormap.from_list("", colors, N=n_bins)

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 7))  # 设置图表尺寸
    sns.heatmap(cm_percent, annot=annot_array, fmt='', cmap=cmap,
                xticklabels=True, yticklabels=True, annot_kws={'size': 12},
                linewidths=0.5, linecolor='black')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix (in %)')
    plt.tight_layout()
    plt.savefig(output_name + '_confusion_matrix.png')  # 保存图表
    plt.show()

if __name__ == '__main__':

    material_names = ['glass', 'white_foam', 'blue_foam', 'woven_fabric',
                      'gray_sponge', 'ribbed_fabric', 'yarn_screen', 'glossy_wood', 'rough_wood',
                      'twill', 'Popline', 'Polyester', 'cotton',
                      'olyester', 'Dobby', 'Nylon', 'Stretch',
                      'Crimp', 'plush', 'Combed', 'Tulle ']
    # Define hyperparameters
    input_size = 300
    num_classes = 21
    learning_rate = 0.001
    batch_size = 500
    # train_data_path = r'C:\Users\lenovo\Desktop\goodjob\current_posess1\data\train_aug'
    train_data_path = 'data/train'
    test_data_path = 'data/test'

    # Define device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Initialize the model, loss function, and optimizer
    # Move model to GPU if available
    model = CNN1D(input_size, num_classes)
    model.load_state_dict(torch.load(
        'cnn1d_model_epock_18.pth', map_location='cpu'))
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    # 创建train_loader和val_loader
    train_dataset = CustomDataset(
        train_data_path, material_names)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = CustomDataset(
        test_data_path, material_names)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    # train
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_accuracy = 0
        true_labels = []
        pred_labels = []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(
                device)  # Move data to GPU if available

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            total_loss += loss.item()
            pred = outputs.argmax(1)
            total_accuracy += (pred == targets).sum().item()
            true_labels.extend(targets.cpu().numpy())
            pred_labels.extend(pred.cpu().numpy())
        # Print loss after every epoch
        average_loss = total_loss / len(train_loader)
        average_accuracy = total_accuracy / len(train_loader.dataset)

        print(
            f'Train: Accuracy: {average_accuracy*100:.2f} Loss: {average_loss:.4f}')
        for i in range(21):
            if i not in true_labels:
                print(f'Train: {i} is not in true_labels')

            if i not in pred_labels:
                print(f'Train: {i} is not in pred_labels')
        get_confusion_matrix(true_labels, pred_labels, 'train')

    # test
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_accuracy = 0
        true_labels = []
        pred_labels = []
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(
                device)  # Move data to GPU if available

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            total_loss += loss.item()
            pred = outputs.argmax(1)
            total_accuracy += (pred == targets).sum().item()
            true_labels.extend(targets.cpu().numpy())
            pred_labels.extend(pred.cpu().numpy())
        # Print loss after every epoch
        average_loss = total_loss / len(test_loader)
        average_accuracy = total_accuracy / len(test_loader.dataset)

        print(
            f'Test:  Accuracy: {average_accuracy*100:.2f} Loss: {average_loss:.4f}')

        for i in range(21):
            if i not in true_labels:
                print(f'Test: {i} is not in true_labels')

            if i not in pred_labels:
                print(f'Test: {i} is not in pred_labels')
        get_confusion_matrix(true_labels, pred_labels, 'test')
