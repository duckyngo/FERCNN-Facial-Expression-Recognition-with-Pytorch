import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import enum


class DataTypes(enum.Enum):
    Training = 1
    PublicTest = 2
    PrivateTest = 3


class FER2013(Dataset):

    def __init__(self, csv_path, data_type=DataTypes.Training, transform=None):

        self.transform = transform
        pd_fer = pd.read_csv(csv_path)

        if data_type == DataTypes.Training:
            self.pd_data = pd_fer.loc[pd_fer['Usage'] == 'Training']
            self.images = self.pd_data['pixels'].values
            self.labels = self.pd_data['emotion'].values
        elif data_type == DataTypes.PublicTest:
            self.pd_data = pd_fer.loc[pd_fer['Usage'] == 'PublicTest']
            self.images = self.pd_data['pixels'].values
            self.labels = self.pd_data['emotion'].values
        else:
            self.pd_data = pd_fer.loc[pd_fer['Usage'] == 'PrivateTest']
            self.images = self.pd_data['pixels'].values
            self.labels = self.pd_data['emotion'].values

    def __len__(self):
        return self.pd_data.shape[0]

    def __getitem__(self, index):
        image = np.fromstring(self.images[index], dtype=int, sep=' ')
        label = self.labels[index]

        image = image.reshape((1, 48, 48))

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'target': label
        }


# if __name__ == '__main__':
#
#     # data_root = 'data/fer2013.csv'
#     data_root = '/mnt/d1/Workspace/fer_cnn/data/fer2013.csv'
#     test = FER2013(data_root, data_type=DataTypes.Training)
#     print(test.__len__())
#     print(test.__getitem__(3))

