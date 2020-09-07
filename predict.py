#!/usr/bin/env python3

import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model("8digit.h5")

def predict_single(img : np.ndarray):
    assert img.shape == (64, 32, 3), "输入图像的尺寸不对, 应为64*32*3"
    return np.argmax(model.predict(np.expand_dims(img, 0)))
def predict_many(imgs : np.ndarray):
    assert imgs.shape[1:] == (64, 32, 3), "输入图像的尺寸不对, 应为64*32*3"
    return np.argmax(model.predict(imgs))

if __name__ == "__main__":
    from answer import answer
    test_set = answer
    for _ in test_set:
        assert test_set[_] == predict_single(np.array(Image.open(_))), f"模型预测错误: {_}"
    print("模型正确")
