import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import time


class Key_Dataset(Dataset):
    def __init__(self, data_path, label_path, frames=8, num_classes=100, train=False, val=False, test=False, transform=None):
        super(Key_Dataset, self).__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.transform = transform
        self.frames = frames
        self.num_classes = num_classes
        self.signers = 50
        self.repetition = 5
        self.train = train
        self.val = val
        self.test = test
        if self.train:
            self.videos_per_folder = int(250 * 0.7)
        if self.val:
            self.videos_per_folder = int(250 * 0.2)
        if self.test:
            self.videos_per_folder = int(250 * 0.1)
        self.data_folder = []
        try:
            obs_path = [os.path.join(self.data_path, item) for item in os.listdir(self.data_path)]
            self.data_folder = sorted([item for item in obs_path if os.path.isdir(item)])
        except Exception as e:
            print("Something wrong with your data path!!!")
            raise
        self.labels = {}
        try:
            label_file = open(self.label_path, 'r', encoding='UTF-8')
            for line in label_file.readlines():
                line = line.strip()
                line = line.split('\t')
                self.labels[line[0]] = line[1]
        except Exception as e:
            raise

    def read_images(self, folder_path):
        files = os.listdir(folder_path)
        assert len(files) >= self.frames, "Too few images in your data folder: " + str(folder_path)
        files.sort(key=lambda x: int(x[:-4]))
        images = []

        for file in files:
            image = cv2.imread(os.path.join(folder_path, file))
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)

        images = torch.stack(images, dim=0)
        images = images.permute(1, 0, 2, 3)
        return images

    def __len__(self):
        return self.num_classes * self.videos_per_folder

    def __getitem__(self, idx):
        top_folder = self.data_folder[int(idx / self.videos_per_folder)]
        selected_folders = [os.path.join(top_folder, item) for item in os.listdir(top_folder)]
        selected_folders = sorted([item for item in selected_folders if os.path.isdir(item)])
        selected_folder = selected_folders[idx % self.videos_per_folder]
        images = self.read_images(selected_folder)
        label = torch.LongTensor([int(idx / self.videos_per_folder)])
        return {'data': images, 'label': label, 'images': images}

    def label_to_word(self, label):
        if isinstance(label, torch.Tensor):
            return self.labels['{:06d}'.format(label.item())]
        elif isinstance(label, int):
            return self.labels['{:06d}'.format(label)]
