#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        img = N + img
        #img[img > 255] = 255                       # 避免有值超过255而反转
        #img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img