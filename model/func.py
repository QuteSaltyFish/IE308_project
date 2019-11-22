import json
import os

import torch as t
import torch.utils.data.dataloader as DataLoader
import torchvision as tv

from model import Mydataloader
from model.DnCnn import DnCNN


def save_model(model, epoch):
    dir = 'saved_model/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    t.save(model.state_dict(), dir + '{}.pkl'.format(epoch))


def load_model(model, epoch):
    path = 'saved_model/''{}.pkl'.format(epoch)
    model.load_state_dict(t.load(path))
    return model


def eval_model_new_thread(epoch, gpu):
    config = json.load(open("config.json"))
    path = 'result/nohup_result'
    if not os.path.exists(path):
        os.makedirs(path)
    python_path = config['python_path']
    os.system('nohup {} -u test_eval.py --epoch={} --gpu={} > {} 2>&1 &'.format(python_path, epoch, gpu,
                                                                                path + '/{}.out'.format(epoch)))
    os.system('nohup {} -u train_eval.py --epoch={} --gpu={} > {} 2>&1 &'.format(python_path, epoch, gpu,
                                                                                 path + '/{}.out'.format(epoch)))


def eval_model(epoch, gpu='0'):
    """
    evaluate the model using multi threading
    :param epoch: the model stored in the nth epoch
    :param gpu: which gpu U tried to use
    :return:
    """
    config = json.load(open('config.json'))
    DEVICE = config['DEVICE'] + ':' + gpu
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    train_data = Mydataloader.TrainingData()
    train_loader = DataLoader.DataLoader(train_data, batch_size=1, shuffle=False, num_workers=config["num_workers"])
    test_data = Mydataloader.TestingData()
    test_loader = DataLoader.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=config["num_workers"])

    criterian = t.nn.MSELoss()
    model = DnCNN(n_channels=8).to(DEVICE)
    # Test the train_loader
    model = load_model(model, epoch)
    model = model.eval()

    with t.no_grad():
        # Test the test_loader
        for batch_idx, [data, label] in enumerate(train_loader):
            data = data.to(DEVICE)
            out = model(data)
            # monitor the upper and lower boundary of output
            out_max = t.max(out)
            out_min = t.min(out)
            out = (out - out_min) / (out_max - out_min)
            DIR = 'result/train_result/epoch_{}'.format(epoch)
            if not os.path.exists(DIR):
                os.makedirs(DIR)
            OUTPUT = t.cat([data, out], dim=3)
            tv.transforms.ToPILImage()(OUTPUT.squeeze().cpu()).save(DIR + '/idx_{}.jpg'.format(batch_idx))

    with t.no_grad():
        # Test the test_loader
        for batch_idx, data in enumerate(test_loader):
            data = data.to(DEVICE)
            out = model(data)
            # monitor the upper and lower boundary of output
            out_max = t.max(out)
            out_min = t.min(out)
            out = (out - out_min) / (out_max - out_min)
            DIR = 'result/test_result/epoch_{}'.format(epoch)
            if not os.path.exists(DIR):
                os.makedirs(DIR)
            OUTPUT = t.cat([data, out], dim=3)
            tv.transforms.ToPILImage()(OUTPUT.squeeze().cpu()).save(DIR + '/idx_{}.jpg'.format(batch_idx))

def Sobel(data, DEVICE='cuda'):
    Gx = t.tensor([
        [-1.0, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]).to(DEVICE)
    Gy = t.tensor([
        [1.0, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ]).to(DEVICE)

    data = t.nn.ZeroPad2d(1)(data)

    kernal_size = 3
    output = t.zeros([data.shape[0], data.shape[1],data.shape[2] - kernal_size + 1, data.shape[3] - kernal_size + 1],
                     dtype=t.float, device=DEVICE)
    for batch_size in range(output.shape[0]):
        for k in range(output.shape[1]):
            for i in range(0, output.shape[2] - 1):
                for j in range(0, output.shape[3] - 1):
                    # print(data[k,i-1:i+1,j-1:j+1])
                    output[batch_size, k, i, j] = t.abs(t.sum(Gx * data[batch_size, k, i:i + kernal_size, j:j + kernal_size])) + t.abs(
                        t.sum(Gy * data[batch_size, k, i:i + kernal_size, j:j + kernal_size]))
    return output
if __name__ == '__main__':
    eval_model_new_thread(0 ,1)
