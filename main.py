import json
import torch as t 
import torchvision as tv 
from torchvision import transforms
from model import Mydataloader
import torch.utils.data.dataloader as DataLoader
import os
import json



if __name__ == "__main__":
    config = json.load(open("config.json"))
    os.environ["CUDA_VISIBLE_DEVICES"] = config["GPU"]
    DEVICE = t.device(config["DEVICE"])
    train_data = Mydataloader.TrainingData()
    test_data = Mydataloader.TestingData() 

    train_loader = DataLoader.DataLoader(train_data, batch_size=config["batch_size"], shuffle = True, num_workers= config["num_workers"])
    test_loader = DataLoader.DataLoader(test_data, batch_size=config["batch_size"], shuffle = False, num_workers= config["num_workers"])

    # Test the train_loader
    for batch_idx, [data, label] in enumerate(train_loader):
        data, label = data.to(DEVICE), label.to(DEVICE)
        print(data.shape, data.device)
        print(label.shape, label.device)
        break
    
    # Test the test_loader
    for batch_idx, data in enumerate(test_loader):
        data = data.to(DEVICE)
        print(data.shape, data.device) 
        break