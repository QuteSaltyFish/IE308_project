'''
used to read the data from the data folder
'''
import torch as t
import torchvision as tv
from torchvision import transforms
import os
from PIL import Image
import json


class TrainingData(t.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.config = json.load(open('config.json'))
        self.data_root = self.config["Taining_Dir"]
        self.label_root = self.config["Label_Dir"]
        self.data_names = os.listdir(self.data_root)
        self.label_names = os.listdir(self.label_root)
        self.DEVICE = t.device(self.config["DEVICE"])
        self.init_transform()
        self.gray = self.config["gray"]

    def init_transform(self):
        """
        The preprocess of the img and label
        """
        self.transform = transforms.Compose([
            transforms.Resize([self.config['H'], self.config['W']]),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_root, self.data_names[index]))
        label = Image.open(os.path.join(
            self.label_root+'/'+self.label_names[index]))
        if self.gray:
            img = img.convert("L")
            label = label.convert("L")
        img, label = self.transform(img), self.transform(label)
        label = img-label
        return img, label

    def __len__(self):
        return len(self.data_names)


class TestingData(t.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.config = json.load(open('config.json'))
        self.test_root = self.config["Test_Dir"]
        self.test_names = os.listdir(self.test_root)
        self.DEVICE = t.device(self.config["DEVICE"])
        self.init_transform()

    def init_transform(self):
        """
        The preprocess of the img and label
        """
        self.transform = transforms.Compose([
            transforms.Resize([self.config['H'], self.config['W']]),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.test_root, self.test_names[index]))
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.test_names)


if __name__ == "__main__":
    train_data = TrainingData()
    test_data = TestingData()
    img, label = train_data[1]
    print(img.shape, img.device)
    img = test_data[1]
    print(img.shape, img.device)
    print(len(test_data))
    tv.transforms.ToPILImage()(label).show()
