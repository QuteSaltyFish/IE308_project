import json
import torch as t 
import torchvision as tv 
from torchvision import transforms
from model import Mydataloader
import torch.utils.data.dataloader as DataLoader
import os
import json
from PIL import Image

from model.DnCnn import DnCNN

if __name__ == "__main__":
    config = json.load(open("config.json"))
    os.environ["CUDA_VISIBLE_DEVICES"] = config["GPU"]
    DEVICE = t.device(config["DEVICE"])
    LR = config['lr']
    EPOCH = config['epoch']
    train_data = Mydataloader.TrainingData()
    test_data = Mydataloader.TestingData() 

    train_loader = DataLoader.DataLoader(train_data, batch_size=config["batch_size"], shuffle = True, num_workers= config["num_workers"])
    test_loader = DataLoader.DataLoader(test_data, batch_size=1, shuffle = False, num_workers= config["num_workers"])

    model = DnCNN().to(DEVICE)

    optimizer = t.optim.Adam(model.parameters())
    
    criterian = t.nn.MSELoss()

    # Test the train_loader
    model = model.train()
    for _ in range(EPOCH):
        for batch_idx, [data, label] in enumerate(train_loader):
            data, label = data.to(DEVICE), label.to(DEVICE)
            # print(data.shape, data.device)
            # print(label.shape, label.device)
            out = model(data)
            loss = criterian(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(loss)
    

    model = model.eval()
    with t.no_grad():
        # Test the test_loader
        for batch_idx, [data, label] in enumerate(train_loader):
            data = data.to(DEVICE)
            out = model(data)
            tv.transforms.ToPILImage()(out[0].squeeze().cpu()).show()
            tv.transforms.ToPILImage()(data[0].squeeze().cpu()).show()
            print(data.shape, data.device) 


    model = model.eval()
    with t.no_grad():
        # Test the test_loader
        for batch_idx, data in enumerate(test_loader):
            data = data.to(DEVICE)
            out = model(data)
            tv.transforms.ToPILImage()(out.squeeze().cpu()).show()
            tv.transforms.ToPILImage()(data.squeeze().cpu()).show()
            print(data.shape, data.device) 
            