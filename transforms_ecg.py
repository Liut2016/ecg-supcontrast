#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from scipy.interpolate import CubicSpline      # for warping
from transforms3d.axangles import axangle2mat  # for rotation
import pywt
from scipy import signal
import pandas as pd

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

class dataReshape(object):
    def __init__(self, len):
        self.len = len
        pass
    def __call__(self, data):
        if self.len == 3:
            data = data.squeeze(2)
        elif self.len == 4:
            data = data.unsqueeze(2)
        return data

## This example using cubic splice is not the best approach to generate random curves.
## You can use other aprroaches, e.g., Gaussian process regression, Bezier curve, etc.
def GenerateRandomCurves(X, sigma=0.2, knot=4):
    # X (C, L)
    # out (C, L) np.ndarry
    from scipy.interpolate import CubicSpline

    xx = (np.ones((X.shape[0], 1)) * (np.arange(0, X.shape[1], (X.shape[1] - 1) / (knot + 1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, X.shape[0]))
    x_range = np.arange(X.shape[1])

    cs = []
    for i in range(X.shape[0]):
        cs.append(CubicSpline(xx[:, i], yy[:, i]))

    return np.array([cs_i(x_range) for cs_i in cs])

class GenerateRandomCurvesClass(object):
    def __init__(self, sigma=0.2, knot=4):
        self.sigma = sigma
        self.knot = knot

    def __call__(self, tensor):
        res = GenerateRandomCurves(tensor, self.sigma, self.knot)
        res = torch.from_numpy(res)
        return res

def DistortTimesteps(X, sigma=0.2):
    # X: (C, L)
    # out: (C, L) np.ndarry
    tt = GenerateRandomCurves(X, sigma).transpose() # Regard these samples aroun 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale = [(X.shape[1] - 1) / tt_cum[-1, i] for i in range(X.shape[0])]
    for i in range(X.shape[0]):
        tt_cum[:,i] = tt_cum[:,i]*t_scale[i]
    return tt_cum.transpose()

class DistortTimestepsClass(object):
    def __init__(self, sigma=0.2):
        self.sigma = sigma

    def __call__(self, tensor):
        x = DistortTimesteps(tensor, self.sigma)
        x = torch.from_numpy(x)
        return x

def RandSampleTimesteps(X, nSample=1000):
    # X: (C, L)
    # out: (C, L) np.ndarry
    tt = np.zeros((nSample,X.shape[0]), dtype=int)
    for i in range(X.shape[0]):
        tt[1:-1,i] = np.sort(np.random.randint(1,X.shape[1]-1,nSample-2))
    tt[-1,:] = X.shape[1]-1
    return tt.transpose()

class RandSampleTimestepsClass(object):
    def __init__(self, nSample=1000):
        self.nSample = nSample

    def __call__(self, tensor):
        x = RandSampleTimesteps(tensor, self.nSample)
        x = torch.from_numpy(x)
        return x

def WTfilt_1d(sig):
    coeffs = pywt.wavedec(data=sig, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    # 如果按照原来的写法loss会变成NaN
    #threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1[0]))))
    # 将高频信号cD1、cD2置零
    cD1.fill(0)
    cD2.fill(0)
    # 将其他中低频信号按软阈值公式滤波
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata

class WTfilt_1d_Class(object):
    def __init__(self):
        pass
    def __call__(self, tensor):
        x = WTfilt_1d(tensor)
        x = torch.from_numpy(x)
        return x

class Jitter(object):
    """
    Args:
        sigma
    """

    def __init__(self, sigma=0.05):
        self.sigma = sigma

    def __call__(self, tensors):
        """
        Args:
            tensor (Tensor): Tensor of size (C, L) to be scaled.

        Returns:
            Tensor: Scaled Tensor.
        """

        myNoise = torch.normal(mean=torch.zeros(tensors.shape), std=self.sigma)

        # print("This is Jitter")
        # print(type(tensors + myNoise))

        return (tensors + myNoise).float()

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={0})'.format(self.sigma)


class Scaling(object):
    """
    Args:
        sigma
    """

    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, tensors):
        """
        Args:
            tensor (Tensor): Tensor of size (C, L) to be scaled.

        Returns:
            Tensor: Scaled Tensor.
        """

        scalingFactor = torch.normal(mean=torch.ones((tensors.shape[0], 1)), std=self.sigma)
        myNoise = torch.matmul(scalingFactor, torch.ones((1, tensors.shape[1])))

        # print("This is Scaling")
        # print(type(tensors * myNoise))

        return (tensors * myNoise).float()

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={0})'.format(self.sigma)


class MagWarp(object):
    """
    Args:
        sigma
    """

    def __init__(self, sigma=0.2):
        self.sigma = sigma

    def __call__(self, tensors):
        """
        Args:
            tensor (Tensor): Tensor of size (C, L) to be scaled.

        Returns:
            Tensor: Scaled Tensor.
        """
        # print("This is MagWarp")
        # print(type(tensors * torch.from_numpy(GenerateRandomCurves(tensors, self.sigma))))

        return tensors * torch.from_numpy(GenerateRandomCurves(tensors, self.sigma))

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={0})'.format(self.sigma)


class TimeWarp(object):
    """
    Args:
        sigma
    """

    def __init__(self, sigma=0.2):
        self.sigma = sigma

    def __call__(self, tensors):
        """
        Args:
            tensor (Tensor): Tensor of size (C, L) to be scaled.

        Returns:
            Tensor: Scaled Tensor.
        """

        tt_new = DistortTimesteps(tensors, self.sigma)
        X_new = np.zeros(tensors.shape)
        x_range = np.arange(tensors.shape[1])
        for i in range(tensors.shape[0]):
            X_new[i, :] = np.interp(x_range, tt_new[i, :], tensors[i, :])

        # print("This is TimeWarp")
        # print(type(torch.from_numpy(X_new)))

        return torch.from_numpy(X_new).float()

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={0})'.format(self.sigma)


class Rotation(object):
    """
    Args:

    """

    def __init__(self):
        pass

    def __call__(self, tensors):
        """
        Args:
            tensor (Tensor): Tensor of size (C, L) to be scaled.

        Returns:
            Tensor: Scaled Tensor.
        """

        #axis = torch.Tensor(tensors.shape[0]).uniform_(-1, 1)
        #angle = torch.Tensor().uniform_(-np.pi, np.pi)


        axis = torch.Tensor(1).uniform_(-1, 1)
        angle = torch.Tensor(1).uniform_(-np.pi, np.pi)

        x = axangle2mat(axis, angle)
        x = torch.from_numpy(x)
        return torch.matmul(x, tensors).float()
        # print("This is Rotation")
        # print(type(torch.matmul(axangle2mat(axis, angle), tensors)))

        #return torch.matmul(axangle2mat(axis, angle), tensors).float()

    def __repr__(self):
        return self.__class__.__name__


class Permutation(object):
    """
    Args:
        nPerm:
        minSegLength:

    """

    def __init__(self, nPerm=4, minSegLength=10):
        self.nPerm = nPerm
        self.minSegLength = minSegLength

    def __call__(self, tensors):
        """
        Args:
            tensor (Tensor): Tensor of size (C, L) to be scaled.

        Returns:
            Tensor: Scaled Tensor.
        """

        X_new = torch.zeros(tensors.shape, dtype=torch.int64)
        idx = torch.randperm(self.nPerm)
        bWhile = True
        while bWhile == True:
            segs = torch.zeros(self.nPerm + 1, dtype=torch.int64)
            segs[1:-1] = torch.sort(torch.randint(self.minSegLength, tensors.shape[1] - self.minSegLength, (self.nPerm - 1,))).values
            segs[-1] = tensors.shape[1]
            if torch.min(segs[1:] - segs[0:-1]) > self.minSegLength:
                bWhile = False
        pp = 0
        for ii in range(self.nPerm):
            x_temp = tensors[:, segs[idx[ii]]:segs[idx[ii] + 1]]
            X_new[:, pp:pp + x_temp.shape[1]] = x_temp
            pp += x_temp.shape[1]

        # print("This is Permutation")
        # print(type(X_new))

        return (X_new).float()

    def __repr__(self):
        return self.__class__.__name__


class RandSampling(object):
    """
    Args:
        nSample:

    """

    def __init__(self, nSample=1000):
        self.nSample = nSample

    def __call__(self, tensors):
        """
        Args:
            tensor (Tensor): Tensor of size (C, L) to be scaled.

        Returns:
            Tensor: Scaled Tensor.
        """

        tt = RandSampleTimesteps(tensors, self.nSample)
        X_new = np.zeros(tensors.shape)
        for i in range(tensors.shape[0]):
            X_new[i, :] = np.interp(np.arange(tensors.shape[1]), tt[i, :], tensors[i, tt[i, :]])

        # print("This is RandSampling")
        # print(type(torch.from_numpy(X_new)))

        return (torch.from_numpy(X_new).float())

    def __repr__(self):
        return self.__class__.__name__


class filter_and_detrend(object):
    """
    Args:

    """
    def __init__(self):
        pass

    def __call__(self, data):
        """
        Args:
            data: 12 lead ECG data . For example,the shape of data is (12,5000)

        Returns:
            Tensor: 12 lead ECG data after filtered and detrended
        """

        filtered_data = pd.DataFrame()
        for k in range(12):
            try:
                filtered_data[k] = signal.detrend(WTfilt_1d(data[k]))
            except ValueError:
                ##有些数据全是0，记录下来，无法进行detrend处理
                filtered_data[k] = WTfilt_1d(data[k])

        return (filtered_data.values).T

    def __repr__(self):
        return self.__class__.__name__