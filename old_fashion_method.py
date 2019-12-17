# %%
import numpy as np
from PIL import Image
import torch as t
import torchvision as tv
import matplotlib.pyplot as plt
from model.func import canny
import cv2
import numpy as np
from model.morphology import morphology
import sys
from model import fillfront
from model import priorities
from model import bestpatch
from model import update
from model.PSNR import PSNR
from model.ssim import SSIM
import argparse
import skimage

def find_marker(name):
    original_img = cv2.imread('data/train_noise/' + name + '.jpg')
    noisefree = cv2.imread('data/train_origin/' + name + '.jpg')
    pic = cv2.resize(original_img, (256, 256), interpolation=cv2.INTER_CUBIC)
    noisefree = cv2.resize(noisefree, (256, 256),
                           interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('result/noise.jpg', pic)
    cv2.imwrite('result/origin.jpg', noisefree)
    TH = 100
    hsv = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)
    r, g, b = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    Image.fromarray(r)
    # %%
    data = r
    data[data < TH] = 0
    data[data >= TH] = 255
    Image.fromarray(data)

    data = tv.transforms.ToTensor()(data)
    mor = morphology(data, 'cpu')
    kernel = t.ones(7, 7)
    closing = mor.Bi_Closing(kernel, save_img=False)
    noise = closing - data
    mor = morphology(noise, 'cpu')
    kernel = t.ones(3, 3) / 255
    mask = mor.Bi_Opening(kernel, save_img=False)

    mor = morphology(mask, 'cpu')
    kernel = t.ones(3, 3)
    mask = mor.Bi_Dilate(kernel, save_img=False)
    tv.transforms.ToPILImage()(1-mask).save('result/mask.jpg')


def inplant(name):
    name = name.split(sep='/')
    name = name[-1].split('.')[0]
    cheminimage = "result/origin.jpg"
    cheminmasque = "result/mask.jpg"
    taillecadre = 3
    image = cv2.imread(cheminimage, 1)
    masque = cv2.imread(cheminmasque, 0)
    xsize, ysize, channels = image.shape    # samesize for filter and image

    # then we check the sizes

    x, y = masque.shape

    if x != xsize or y != ysize:
        print("the image size and the filer size must be the same!")
        exit()

    tau = 170  # value to separate mask values
    omega = []
    confiance = np.copy(masque)
    masque = np.copy(masque)
    for x in range(xsize):
        for y in range(ysize):
            v = masque[x, y]
            if v < tau:
                omega.append([x, y])
                image[x, y] = [255, 255, 255]
                masque[x, y] = 1
                confiance[x, y] = 0.
            else:
                masque[x, y] = 0
                confiance[x, y] = 1.

    cv2.imwrite('result/' + name + "_avec_masque.png", image)
    source = np.copy(confiance)
    original = np.copy(confiance)
    dOmega = []
    normale = []

    im = np.copy(image)
    result = np.ndarray(shape=image.shape)

    data = np.ndarray(shape=image.shape[:2])
    Lap = np.array([[1.,  1.,  1.], [1., -8.,  1.], [1.,  1.,  1.]])
    kerx = np.array([[0.,  0.,  0.], [-1.,  0.,  1.], [0.,  0.,  0.]])
    kery = np.array([[0., -1.,  0.], [0.,  0.,  0.], [0.,  1.,  0.]])

    bool = True  # flag for the while loop
    print("Algorithm in operation")
    k = 0

    niveau_de_gris = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    gradientX = np.float32(cv2.convertScaleAbs(
        cv2.Scharr(niveau_de_gris, cv2.CV_32F, 1, 0)))

    gradientY = np.float32(cv2.convertScaleAbs(
        cv2.Scharr(niveau_de_gris, cv2.CV_32F, 0, 1)))
    while bool:
        print(k)
        k += 1
        xsize, ysize = source.shape

        niveau_de_gris = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

        gradientX = np.float32(cv2.convertScaleAbs(
            cv2.Scharr(niveau_de_gris, cv2.CV_32F, 1, 0)))

        gradientY = np.float32(cv2.convertScaleAbs(
            cv2.Scharr(niveau_de_gris, cv2.CV_32F, 0, 1)))

        for x in range(xsize):
            for y in range(ysize):
                if masque[x][y] == 1:
                    gradientX[x][y] = 0
                    gradientY[x][y] = 0
        gradienX, gradientY = gradientX/255, gradientY/255

        dOmega, normale = fillfront.IdentifyTheFillFront(masque, source)

        confiance, data, index = priorities.calculPriority(
            im, taillecadre, masque, dOmega, normale, data, gradientX, gradientY, confiance)

        list, pp = bestpatch.calculPatch(
            dOmega, index, im, original, masque, taillecadre)

        im, gradientX, gradientY, confiance, source, masque = update.update(
            im, gradientX, gradientY, confiance, source, masque, dOmega, pp, list, index, taillecadre)

        # check if we are finished
        bool = False
        for x in range(xsize):
            for y in range(ysize):
                if source[x, y] == 0:
                    bool = True

        # we save the process in folder process and the final pic
        cv2.imwrite('result/' + name + "_resultat.jpg", im)
        cv2.imwrite('result/process/' + name + "_" + str(k) + ".jpg", im)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str,
                        help="The name of the pic to be test")
    args = parser.parse_args()
    name = args.name
    #name = 'SK BR 1102 XU LI YING F70Y_20160817_141835_image'
    # name = 'SK BR938 SHI SI MING F31Y_20160715_111151_image'
    find_marker(name)
    inplant(name)
    
    # compare the psnr and ssim 
    origin_noise = cv2.imread('data/train_noise/' + name + '.jpg')
    origin_noise = cv2.resize(origin_noise, (256, 256), interpolation=cv2.INTER_CUBIC)

    original_img = cv2.imread('data/train_origin/' + name + '.jpg')
    original_img = cv2.resize(original_img, (256, 256), interpolation=cv2.INTER_CUBIC)

    result = cv2.imread('result/' + name + '_resultat.jpg')

    # before
    print("Before the operation:")
    print("The PSNR between the two img of the two is {}".format(skimage.measure.compare_psnr(origin_noise, original_img, 255)))
    print("The SSIM between the two img of the two is {}".format(skimage.measure.compare_ssim(origin_noise, original_img, multichannel=True)))
    print('-'*20)
    print("After the operation:")
    print("The PSNR between the two img of the two is {}".format(skimage.measure.compare_psnr(result, original_img, 255)))
    print("The SSIM between the two img of the two is {}".format(skimage.measure.compare_ssim(result, original_img, multichannel=True)))


# %%
