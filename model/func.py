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


def canny(img, TL=0.1, TH=0.3, DEVICE='cuda'):
    sigma1 = sigma2 = 1
    sum = 0

    gaussian = t.zeros([5, 5], dtype=t.float, device=DEVICE)
    for i in range(5):
        for j in range(5):
            gaussian[i, j] = np.exp(-1 / 2 * ((i - 3) ** 2 / sigma1 ** 2
                                              + ((j - 3) ** 2 / sigma2 ** 2))) / (
                                     2 * np.pi * sigma1 * sigma2)
            sum = sum + gaussian[i, j]

    gaussian = gaussian / sum

    # print(gaussian)

    def rgb2gray(rgb):
        rgb = rgb.permute(0, 2, 3, 1)
        return t.matmul(rgb[..., :3], t.tensor([0.299, 0.587, 0.114], device=DEVICE))

    # step1.高斯滤波
    gray = rgb2gray(img)
    Batch_size, W, H = gray.shape
    new_gray = t.zeros([W - 5, H - 5], dtype=t.float, device=DEVICE)
    for batch in range(Batch_size):
        for i in range(W - 5):
            for j in range(H - 5):
                new_gray[i, j] = t.sum(gray[batch, i:i + 5, j:j + 5] * gaussian)  # 与高斯矩阵卷积实现滤波

    # plt.imshow(new_gray, cmap="gray")

    # step2.增强 通过求梯度幅值
    W1, H1 = new_gray.shape
    dx = t.zeros([W1 - 1, H1 - 1], dtype=t.float, device=DEVICE)
    dy = t.zeros([W1 - 1, H1 - 1], dtype=t.float, device=DEVICE)
    d = t.zeros([W1 - 1, H1 - 1], dtype=t.float, device=DEVICE)
    for i in range(W1 - 1):
        for j in range(H1 - 1):
            dx[i, j] = new_gray[i, j + 1] - new_gray[i, j]
            dy[i, j] = new_gray[i + 1, j] - new_gray[i, j]
            d[i, j] = t.sqrt(dx[i, j] ** 2 + dy[i, j] ** 2)  # 图像梯度幅值作为图像强度值

    # plt.imshow(d, cmap="gray")

    # setp3.非极大值抑制 NMS
    W2, H2 = d.shape
    NMS = d.clone()
    NMS[0, :] = NMS[W2 - 1, :] = NMS[:, 0] = NMS[:, H2 - 1] = 0
    for i in range(1, W2 - 1):
        for j in range(1, H2 - 1):

            if d[i, j] == 0:
                NMS[i, j] = 0
            else:
                gradX = dx[i, j]
                gradY = dy[i, j]
                gradTemp = d[i, j]

                # 如果Y方向幅度值较大
                if t.abs(gradY) > t.abs(gradX):
                    weight = t.abs(gradX) / t.abs(gradY)
                    grad2 = d[i - 1, j]
                    grad4 = d[i + 1, j]
                    # 如果x,y方向梯度符号相同
                    if gradX * gradY > 0:
                        grad1 = d[i - 1, j - 1]
                        grad3 = d[i + 1, j + 1]
                    # 如果x,y方向梯度符号相反
                    else:
                        grad1 = d[i - 1, j + 1]
                        grad3 = d[i + 1, j - 1]

                # 如果X方向幅度值较大
                else:
                    weight = t.abs(gradY) / t.abs(gradX)
                    grad2 = d[i, j - 1]
                    grad4 = d[i, j + 1]
                    # 如果x,y方向梯度符号相同
                    if gradX * gradY > 0:
                        grad1 = d[i + 1, j - 1]
                        grad3 = d[i - 1, j + 1]
                    # 如果x,y方向梯度符号相反
                    else:
                        grad1 = d[i - 1, j - 1]
                        grad3 = d[i + 1, j + 1]

                gradTemp1 = weight * grad1 + (1 - weight) * grad2
                gradTemp2 = weight * grad3 + (1 - weight) * grad4
                if gradTemp >= gradTemp1 and gradTemp >= gradTemp2:
                    NMS[i, j] = gradTemp
                else:
                    NMS[i, j] = 0

    # step4. 双阈值算法检测、连接边缘
    W3, H3 = NMS.shape
    DT = t.zeros([W3, H3], dtype=t.float, device=DEVICE)
    # 定义高低阈值
    TL = TL * t.max(NMS)
    TH = TH * t.max(NMS)
    for i in range(1, W3 - 1):
        for j in range(1, H3 - 1):
            if (NMS[i, j] < TL):
                DT[i, j] = 0
            elif (NMS[i, j] > TH):
                DT[i, j] = 1
            elif ((NMS[i - 1, j - 1:j + 1] < TH).any() or (NMS[i + 1, j - 1:j + 1]).any()
                  or (NMS[i, [j - 1, j + 1]] < TH).any()):
                DT[i, j] = 1
    return DT


def Sobel(data, DEVICE='cuda'):
    Gx = t.tensor([
        [-1.0, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], device=DEVICE)
    Gy = t.tensor([
        [1.0, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ], device=DEVICE)

    data = t.nn.ZeroPad2d(1)(data)

    kernal_size = 3
    output = t.zeros([data.shape[0], data.shape[1], data.shape[2] - kernal_size + 1, data.shape[3] - kernal_size + 1],
                     dtype=t.float, device=DEVICE)
    for batch in range(output.shape[0]):
        for k in range(output.shape[1]):
            for i in range(0, output.shape[2] - 1):
                for j in range(0, output.shape[3] - 1):
                    # print(data[k,i-1:i+1,j-1:j+1])
                    output[batch, k, i, j] = t.abs(
                        t.sum(Gx * data[batch, k, i:i + kernal_size, j:j + kernal_size])) + t.abs(
                        t.sum(Gy * data[batch, k, i:i + kernal_size, j:j + kernal_size]))
    return output
if __name__ == '__main__':
    eval_model_new_thread(0 ,1)
