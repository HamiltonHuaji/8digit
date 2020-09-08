#!/usr/bin/env python3

from PIL import Image, ImageChops
from math import *
import numpy as np
import random
import cv2


def ImgOffset(img, xoffset, yoffset):
    width, height = img.size
    c = ImageChops.offset(img, xoffset, yoffset)
    c.paste((0, 0, 0), (0, 0, xoffset, height))
    c.paste((0, 0, 0), (0, 0, width, yoffset))
    return c


def ImgResize(img, xScaleFactor, yScaleFactor):
    ImgSize = img.size  # 获得图像原始尺寸
    NewSize = [int(ImgSize[0]*xScaleFactor),
               int(ImgSize[1]*yScaleFactor)]  # 获得图像新尺寸，保持长宽比
    img = img.resize(NewSize)  # 利用PIL的函数进行图像resize，类似matlab的imresize函数
    ImgSize = img.size  # 获得图像新尺寸
    return img.crop((ImgSize[0]/2-16, ImgSize[1]/2-32,
                     ImgSize[0]/2+16, ImgSize[1]/2+32))  # 剪切到原始尺寸


origin = [Image.open(f"{_}.png") for _ in range(10)]

d = {}

for i in range(8192):
    num = random.randint(0, 9)
    current = origin[num].copy()
    current = current.rotate(random.randint(-15, 15))
    current = ImgOffset(current, random.randint(-4, 4), random.randint(-4, 4))
    current = ImgResize(current, random.randint(
        50, 150) / 100, random.randint(50, 150) / 100)
    d[f"dataset/{str(i).zfill(4)}.png"] = num
    current.save(f"dataset/{str(i).zfill(4)}.png")

print(f"answer={d}")
