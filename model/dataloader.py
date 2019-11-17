'''
used to read the data from the data folder
'''
import torch as t 
import torchvision as tv 
from torchvision import transforms
import os
from PIL import Image
class data(t.utils.data.Dataset):
    def __init__(self, data_dir, label_dir):
        super().__init__()   
        self.data_root = data_dir
        self.label_root = label_dir
        self.data_names = os.listdir(data_dir)
        self.label_names = os.listdir(label_dir)

    def __getitem__(self, index):
        img =  Image.open(self.data_root+'/'+ self.data_names[index])
        label = Image.open(self.label_root+'/'+self.label_names[index])
        return img, label


if __name__ == "__main__":
    data = data('data/train_noise','data/train_origin')
    img, label = data[1]
    img.show()
    label.show()