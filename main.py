import json
import torch as t
import torchvision as tv
from torchvision import transforms
from model import Mydataloader
import torch.utils.data.dataloader as DataLoader
import os
import json
from PIL import Image
import time
from model.DnCnn import DnCNN
from model.func import save_model, load_model
if __name__ == "__main__":
    time_start = time.time()

    config = json.load(open("config.json"))
    os.environ["CUDA_VISIBLE_DEVICES"] = config["GPU"]
    DEVICE = t.device(config["DEVICE"])
    LR = config['lr']
    EPOCH = config['epoch']
    train_data = Mydataloader.TrainingData()
    test_data = Mydataloader.TestingData()

    train_loader = DataLoader.DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    test_loader = DataLoader.DataLoader(
        test_data, batch_size=1, shuffle=False, num_workers=config["num_workers"])

    model = DnCNN(n_channels=8).to(DEVICE)
    # Multi GPU setting
    # model = t.nn.DataParallel(model,device_ids=[0,1])

    optimizer = t.optim.Adam(model.parameters())

    criterian = t.nn.MSELoss()

    # Test the train_loader
    model = model.train()
    # model = load_model(model, 0)
    for epoch in range(EPOCH):
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
        save_model(model, epoch)
    time_end = time.time()
    print('time cost', time_end-time_start)

    model = model.eval()
    with t.no_grad():
        # Test the test_loader
        for batch_idx, [data, label] in enumerate(train_loader):
            data = data.to(DEVICE)
            out = model(data)
            # out = out + data
            # monitor the upper and lower boundary of output
            out_max = t.max(out)
            out_min = t.min(out)
            out = (out - out_min) / (out_max - out_min)
            tv.transforms.ToPILImage()(out[0].squeeze().cpu()).save('result/output.jpg')
            tv.transforms.ToPILImage()(data[0].squeeze().cpu()).save('result/input.jpg')
            print(data.shape, data.device)

    model = model.eval()
    with t.no_grad():
        # Test the test_loader
        for batch_idx, data in enumerate(test_loader):
            data = data.to(DEVICE)
            out = model(data)
            # monitor the upper and lower boundary of output
            out_max = t.max(out)
            out_min = t.min(out)
            out = (out - out_min) / (out_max - out_min)
            tv.transforms.ToPILImage()(out.squeeze().cpu()).save('result/test_output.jpg')
            tv.transforms.ToPILImage()(data.squeeze().cpu()).save('result/test_input.jpg')
            print(data.shape, data.device)
