#!/usr/bin/env python3

from PIL import Image
from math import *
import numpy as np
import random

origin = [Image.open(f"{_}.png") for _ in range(10)]

d = {}

for i in range(4096):
    num = random.randint(0, 9)
    current = origin[num].copy()
    current = current.rotate(random.randint(-20, 20))
    d[f"dataset/{str(i).zfill(4)}.png"] = num
    current.save(f"dataset/{str(i).zfill(4)}.png")

print(f"answer={d}")