import torch as t
import numpy as np
import cv2
from PIL import Image
import torchvision as tv
from torchvision import transforms
import json
from time import time


class morphology():
    def __init__(self, data, DEVICE=None):
        if DEVICE == None:
            self.DEVICE = t.device("cuda" if t.cuda.is_available() else 'cpu')
        else:
            self.DEVICE = t.device(DEVICE)
        # self.img = Image.open(dir)
        # self.img_gray = self.img.convert("L")
        # self.img_tensor = transforms.ToTensor()(self.img).to(self.DEVICE)
        self.img_gray_tensor = data
        self.load_config()

    def load_data(self, data):
        self.img_gray_tensor = t.tensor(
            data, dtype=t.float, device=self.DEVICE).unsqueeze(0)

    def load_config(self):
        file = open('config.json', "rb")
        self.config = json.load(file)

    def Erode(self, kernel, save_img=True, new_data=None):
        kernel = t.tensor(kernel, dtype=t.float, device=self.DEVICE)
        kernel_size = kernel.shape
        padding = [kernel_size[0]//2, kernel_size[0] //
                   2, kernel_size[1]//2, kernel_size[1]//2]
        if new_data is None:
            data = t.nn.ConstantPad2d(padding, 2)(self.img_gray_tensor)
        else:
            data = t.nn.ConstantPad2d(padding, 2)(new_data)
        # print(data)
        output = t.zeros([data.shape[0], data.shape[1]-kernel_size[0]+1,
                          data.shape[2]-kernel_size[1]+1], dtype=t.float).to(self.DEVICE)

        for k in range(output.shape[0]):
            for i in range(output.shape[1]):
                for j in range(output.shape[2]):
                    # print(data[k,i-1:i+1,j-1:j+1])
                    output[k, i, j] = t.min(
                        data[k, i:i+kernel_size[0], j:j+kernel_size[1]]-kernel)
        if save_img:
            out_img = transforms.ToPILImage()(output.cpu())
            out_img.save('result/Erode.gif')
            return out_img
        else:
            return output

    def Bi_Erode(self, kernel, save_img=True, new_data=None):
        kernel = t.tensor(kernel, dtype=t.float, device=self.DEVICE)
        kernel[kernel > 0] = 1
        kernel_size = kernel.shape
        padding = [kernel_size[0]//2, kernel_size[0] //
                   2, kernel_size[1]//2, kernel_size[1]//2]
        if new_data is None:
            data = t.nn.ConstantPad2d(padding, 2)(self.img_gray_tensor)
        else:
            data = t.nn.ConstantPad2d(padding, 2)(new_data)
        # print(data)
        output = t.zeros([data.shape[0], data.shape[1]-kernel_size[0]+1,
                          data.shape[2]-kernel_size[1]+1], dtype=t.float).to(self.DEVICE)

        for k in range(output.shape[0]):
            for i in range(output.shape[1]):
                for j in range(output.shape[2]):
                    # print(data[k,i-1:i+1,j-1:j+1])
                    output[k, i, j] = t.min(
                        data[k, i:i+kernel_size[0], j:j+kernel_size[1]]*kernel)
        if save_img:
            out_img = transforms.ToPILImage()(output.cpu())
            out_img.save('result/Bi_Erode.gif')
            return out_img
        else:
            return output

    def Dilate(self, kernel, save_img=True, new_data=None):
        kernel = t.tensor(kernel, dtype=t.float, device=self.DEVICE)
        kernel_size = kernel.shape
        padding = [kernel_size[0] // 2, kernel_size[0] //
                   2, kernel_size[1] // 2, kernel_size[1] // 2]

        if new_data is None:
            data = t.nn.ConstantPad2d(padding, -1)(self.img_gray_tensor)
        else:
            data = t.nn.ConstantPad2d(padding, -1)(new_data)
        # print(data)
        output = t.zeros([data.shape[0], data.shape[1]-kernel_size[0]+1,
                          data.shape[2]-kernel_size[1]+1], dtype=t.float).to(self.DEVICE)

        for k in range(output.shape[0]):
            for i in range(output.shape[1]):
                for j in range(output.shape[2]):
                    # print(data[k,i-1:i+1,j-1:j+1])
                    output[k, i, j] = t.max(
                        data[k, i:i+kernel_size[0], j:j+kernel_size[1]]+kernel)
        if save_img:
            out_img = transforms.ToPILImage()(output.cpu())
            out_img.save('result/Dilate.gif')
            return out_img
        else:
            return output

    def Bi_Dilate(self, kernel, save_img=True, new_data=None):
        kernel = t.tensor(kernel, dtype=t.float, device=self.DEVICE)
        kernel[kernel > 0] = 1
        kernel_size = kernel.shape
        padding = [kernel_size[0] // 2, kernel_size[0] //
                   2, kernel_size[1] // 2, kernel_size[1] // 2]

        if new_data is None:
            data = t.nn.ConstantPad2d(padding, -1)(self.img_gray_tensor)
        else:
            data = t.nn.ConstantPad2d(padding, -1)(new_data)
        # print(data)
        output = t.zeros([data.shape[0], data.shape[1]-kernel_size[0]+1,
                          data.shape[2]-kernel_size[1]+1], dtype=t.float).to(self.DEVICE)

        for k in range(output.shape[0]):
            for i in range(output.shape[1]):
                for j in range(output.shape[2]):
                    # print(data[k,i-1:i+1,j-1:j+1])
                    output[k, i, j] = t.max(
                        data[k, i:i+kernel_size[0], j:j+kernel_size[1]]*kernel)
        if save_img:
            out_img = transforms.ToPILImage()(output.cpu())
            out_img.save('result/Bi_Dilate.gif')
            return out_img
        else:
            return output

    def Bi_Opening(self, kernel, save_img=True, new_data=None):
        output = self.Bi_Erode(kernel, save_img=False, new_data=new_data)
        output = self.Bi_Dilate(kernel, save_img=False, new_data=output)
        if save_img:
            out_img = transforms.ToPILImage()(output.cpu())
            out_img.save('result/Bi_Opening.gif')
            return out_img
        else:
            return output

    def Opening(self, kernel, save_img=True, new_data=None):
        output = self.Erode(kernel, save_img=False, new_data=new_data)
        output = self.Dilate(kernel, save_img=False, new_data=output)
        if save_img:
            out_img = transforms.ToPILImage()(output.cpu())
            out_img.save('result/Opening.gif')
            return out_img
        else:
            return output

    def Bi_Closing(self, kernel, save_img=True, new_data=None):
        output = self.Bi_Dilate(kernel, save_img=False, new_data=new_data)
        output = self.Bi_Erode(kernel, save_img=False, new_data=output)
        if save_img:
            out_img = transforms.ToPILImage()(output.cpu())
            out_img.save('result/Bi_Closing.gif')
            return out_img
        else:
            return output

    def Closing(self, kernel, save_img=True, new_data=None):
        output = self.Dilate(kernel, save_img=False, new_data=new_data)
        output = self.Erode(kernel, save_img=False, new_data=output)
        if save_img:
            out_img = transforms.ToPILImage()(output.cpu())
            out_img.save('result/Closing.gif')
            return out_img
        else:
            return output

    def edge(self, kernel, save_img=True, new_data=None):
        if new_data is None:
            data = self.img_gray_tensor
        else:
            data = new_data
        data = self.to2(data)
        pic_erode = self.Bi_Erode(kernel, False)
        pic_dilate = self.Bi_Dilate(kernel, False)

        edge = pic_dilate - pic_erode

        if save_img:
            out_img = transforms.ToPILImage()(edge.cpu())
            out_img.save('result/Edge_detection.gif')
            return out_img
        else:
            return edge

    def edge_ex(self, kernel, save_img=True, new_data=None):
        if new_data is None:
            data = self.img_gray_tensor
        else:
            data = new_data
        data = self.to2(data)
        # pic_erode = self.Erode(kernel, False)
        pic_dilate = self.Bi_Dilate(kernel, False)
        edge = pic_dilate - data

        if save_img:
            out_img = transforms.ToPILImage()(edge.cpu())
            out_img.save('result/Edge_detection_external.gif')
            return out_img
        else:
            return edge

    def edge_in(self, kernel, save_img=True, new_data=None):
        if new_data is None:
            data = self.img_gray_tensor
        else:
            data = new_data
        data = self.to2(data)
        pic_erode = self.Bi_Erode(kernel, False)
        # pic_dilate = self.Dilate(kernel, False)
        edge = data - pic_erode

        if save_img:
            out_img = transforms.ToPILImage()(edge.cpu())
            out_img.save('result/Edge_detection_internal.gif')
            return out_img
        else:
            return edge

    def grad(self, kernel, save_img=True, new_data=None):
        pic_erode = self.Erode(kernel, False)
        pic_dilate = self.Dilate(kernel, False)

        grad = (pic_dilate - pic_erode)/2

        if save_img:
            out_img = transforms.ToPILImage()(grad.cpu())
            out_img.save('result/grad.gif')
            return out_img
        else:
            return grad

    def grad_ex(self, kernel, save_img=True, new_data=None):
        if new_data is None:
            data = self.img_gray_tensor
        else:
            data = new_data
        pic_erode = self.Erode(kernel, False)
        pic_dilate = self.Dilate(kernel, False)
        grad = pic_erode - data

        if save_img:
            out_img = transforms.ToPILImage()(grad.cpu())
            out_img.save('result/grad_internal.gif')
            return out_img
        else:
            return grad

    def grad_in(self, kernel, save_img=True, new_data=None):
        if new_data is None:
            data = self.img_gray_tensor
        else:
            data = new_data
        pic_erode = self.Erode(kernel, False)
        # pic_dilate = self.Dilate(kernel, False)
        grad = data - pic_erode

        if save_img:
            out_img = transforms.ToPILImage()(grad.cpu())
            out_img.save('result/grad_internal.gif')
            return out_img
        else:
            return grad

    def Smooth(self, kernel, save_img=True, new_data=None):
        output = self.Opening(kernel, save_img=False, new_data=new_data)
        output = self.Closing(kernel, save_img=False, new_data=output)
        if save_img:
            out_img = transforms.ToPILImage()(output.cpu())
            out_img.save('result/Smooth.gif')
            return out_img
        else:
            return output

    def to2(self, data, th=0.5):
        data[data <= th] = 0
        data[data >= th] = 1
        return data

    def MReconstruction(self, kernel, loop=4, save_img=True, new_data=None, th=0.5, noise=False):
        if new_data is None:
            data = self.to2(self.img_gray_tensor, th)
        else:
            data = self.to2(new_data, th)

        # noise = t.randn_like(data, device=self.DEVICE)*0.01
        # data += noise

        # Store the picture before the reconstruction
        out_img = transforms.ToPILImage()(data.cpu())
        out_img.save('result/Before_Reconstruction.gif')

        M = self.to2(self.Opening(kernel, save_img=False, new_data=data))
        out_img = transforms.ToPILImage()(M.cpu())
        out_img.save('result/Before_Reconstruction_M.gif')
        assert (t.sum(M < 0) == 0)
        assert (t.sum(M > 1) == 0)
        while (True):
            T = M

            for i in range(loop):
                M = self.to2(self.Bi_Dilate(kernel, False, M))
                assert (t.sum(M < 0) == 0)
                assert (t.sum(M > 1) == 0)
            M = self.to2(M * data)
            out_img = transforms.ToPILImage()(M.cpu())
            out_img.save('tmp.gif')

            criterion = t.sum(M != T)
            print(criterion)
            if t.sum(M != T) == 0:
                break

        if save_img:
            out_img = transforms.ToPILImage()(M.cpu())
            out_img.save('result/Reconstruction.gif')
            return out_img
        else:
            return M

    def Conditional_Dilate(self, condition, kernel, save_img=True, new_data=None):
        out = self.Dilate(kernel, save_img=False, new_data=new_data)

        condition = self.to2(condition)

        out *= condition
        if save_img:
            out_img = transforms.ToPILImage()(out.cpu())
            out_img.save('result/Conditional_Dilate.gif')
            return out_img
        else:
            return out


if __name__ == "__main__":
    start = time()
    print("Start: " + str(start))
    stop = time()
    print("Stop: " + str(stop))
    print(str(stop-start) + "秒")

    passdata = t.tensor([
        [209.0,  125,  191, 9, 168, 246, 158, 14],
        [232, 205, 101, 113, 42, 141, 122, 136],
        [33, 37, 168, 98, 31, 36, 91, 200],
        [234, 108, 44, 196, 128, 39, 213, 240],
        [162, 235, 181, 204, 246, 66, 150, 34],
        [25, 203, 9, 48, 88, 216, 141, 146],
        [72, 246, 71, 126, 150, 66, 235, 121]
    ]) / 255
    kernel = t.tensor([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ])/255.0
    kernel2 = t.tensor([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ])
    model = hw2('/home/wangmingke/Desktop/HomeWork/CV_HW2/src/Noised_img.gif')
    # model = hw2('/home/wangmingke/Desktop/HomeWork/CV_HW2/src/img.jpg', 'cpu')

    # img = tv.transforms.ToTensor()(Image.open("/home/wangmingke/Desktop/HomeWork/CV_HW2/src/img.jpg").convert('L'))
    # condition = t.zeros_like(img)
    # _, H, W = condition.shape
    # condition[0, H // 4:H * 3 // 4, W // 4:W * 3 // 4] = 1
    # print(t.sum(condition>0))
    # # model.load_data(data)
    model.Erode(kernel)
    model.Bi_Erode(kernel2)
    model.Dilate(kernel)
    model.Bi_Dilate(kernel2)
    model.edge(kernel)
    model.Opening(kernel)
    model.Closing(kernel)
    # model.MReconstruction(kernel, th=0.45)
    model.grad(kernel)
    # model.Conditional_Dilate(condition, kernel)
    stop = time()
    print("Stop: " + str(stop))
    print(str(stop-start) + "秒")
