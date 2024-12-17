import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from model import CNN1D_A as CNN1D
from torch.utils.tensorboard import SummaryWriter

import os
from typing import List, Optional, Tuple
from torch.utils.data import Dataset
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, root_dir, names, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.datas = []
        self.labels = []

        for filename in os.listdir(root_dir):

            for i, name in enumerate(names):
                if name in filename:
                    label = i
                    self.labels.append(label)
                    break
            self.datas.append(os.path.join(root_dir, filename))

    def __len__(self):
        return len(self.datas)
    def __getitem__(self, idx):
        df = pd.read_csv(self.datas[idx])
        df.fillna(0, inplace=True)
        data = np.squeeze(df.values)
        # if (data.ndim > 1):
        #     print(f'abnormal:{df.shape}')
        # else:
        #     print(f'normal:{df.shape}')
        # if (data[:, 0].mean() != 0):
        #     data = data[:, 0]
        # else:
        #     data = data[:, 1]
        data_cleaned = np.zeros(data.shape[0], dtype=float)
        if (data.ndim > 1):
            for i in range(data.shape[0]):
                if (data[i, 0] != 0):
                    data_cleaned[i] = data[i, 0]
                elif (data[i, 1] != 0):
                    data_cleaned[i] = data[i, 1]
                else:
                    data_cleaned[i] = 0
            data = data_cleaned
        data += 1e-10
        label = self.labels[idx]

        # 计算均值和标准差
        # file_name = self.datas[idx]
        mean = data.mean()
        std = data.std()
        # if (std == 0):
        #     print(self.datas[idx])

        # 使用计算得到的均值和标准差进行标准化
        normalized_data = (data - mean) / std
        # if self.transform:
        #     data = self.transform(data)
        normalized_data = torch.from_numpy(normalized_data).float()

        return normalized_data, label


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
    num_epochs = 60
    # train_data_path = r'C:\Users\lenovo\Desktop\goodjob\current_posess1\data\train_aug'
    # val_data_path = r'C:\Users\lenovo\Desktop\goodjob\current_posess1\data\test'
    train_data_path = 'data/train_aug' 
    val_data_path = 'data/test'

    # tensorboard
    writer = SummaryWriter('log/train')

    # Define device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Initialize the model, loss function, and optimizer
    # Move model to GPU if available
    model = CNN1D(input_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 创建train_loader和val_loader
    train_dataset = CustomDataset(
        train_data_path, material_names)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = CustomDataset(
        val_data_path, material_names)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_accuracy = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(
                device)  # Move data to GPU if available

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_accuracy += (outputs.argmax(1) == targets).sum().item()
        # Print loss after every epoch
        average_loss = total_loss / len(train_loader)
        average_accuracy = total_accuracy / len(train_loader.dataset)
        writer.add_scalar('Loss/train', average_loss, epoch)
        writer.add_scalar('Accuracy/train', average_accuracy, epoch)
        print(
            f'Train: Epoch [{epoch+1}/{num_epochs}], Accuracy: {average_accuracy*100:.2f} Loss: {average_loss:.4f}')

        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_accuracy = 0
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(
                    device)  # Move data to GPU if available

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                total_loss += loss.item()
                total_accuracy += (outputs.argmax(1) == targets).sum().item()
            # Print loss after every epoch
            average_loss = total_loss / len(val_loader)
            average_accuracy = total_accuracy / len(val_loader.dataset)
            writer.add_scalar('Loss/val', average_loss, epoch)
            writer.add_scalar('Accuracy/val', average_accuracy, epoch)
            print(
                f'Val:   Epoch [{epoch+1}/{num_epochs}], Accuracy: {average_accuracy*100:.2f} Loss: {average_loss:.4f}')
            if average_accuracy > 0.90:
                # save the trained model
                torch.save(model.state_dict(),
                           f'cnn1d_model_epock_{epoch+1}.pth')
                exit()
    torch.save(model.state_dict(), 'cnn1d_model.pth')
