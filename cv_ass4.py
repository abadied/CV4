# -*- coding: utf-8 -*-

import numpy as np
import cv2
import scipy
from scipy import ndimage
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
import math
import scipy.signal
import scipy.io as sio
from math import pow
from mpl_toolkits.axes_grid1 import make_axes_locatable


img = cv2.imread('hw4_data/hw4_data/mandrill.png', 1)
c_i = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow('mandrill', img)


# ==================
# Question 1
# ==================
def Q1():
    # an isotropic 7×7 Gaussian with σ = 3;
    img_f1 = cv2.GaussianBlur(img, (7, 7), 3)
    cv2.imshow('GaussianBlur 7 X 7', img_f1)

    # an isotropic 21×21 Gaussian with σ = 10
    img_f2 = cv2.GaussianBlur(img, (21, 21), 10)
    cv2.imshow('GaussianBlur 21 X 21', img_f2)

    # a uniform 21 × 21 blur kernel
    img_f3 = cv2.blur(img,(21,21))
    cv2.imshow('uniform 21 × 21 blur kernel', img_f3)

    fig = plt.figure()
    fig.set_size_inches(20, 10)
    ax1 = fig.add_subplot(141)
    ax1.title.set_text('Mandrill')
    ax2 = fig.add_subplot(142)
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax2.title.set_text('An isotropic 7×7 Gaussian with σ = 3')
    ax3 = fig.add_subplot(143)
    ax2.imshow(cv2.cvtColor(cv2.GaussianBlur(img,(7,7),3), cv2.COLOR_BGR2RGB))
    ax3.title.set_text('An isotropic 21×21 Gaussian with σ = 10')
    ax3.imshow(cv2.cvtColor(cv2.GaussianBlur(img,(21,21),10), cv2.COLOR_BGR2RGB))
    ax4 = fig.add_subplot(144)
    ax4.title.set_text('Uniform 21 × 21 blur kernel')
    ax4.imshow(cv2.cvtColor(cv2.blur(img,(21,21)), cv2.COLOR_BGR2RGB))
    #plt.show()


# ==================
# Question 2
# ==================
def build_H(M, N, h):
    h = h.T
    H = np.zeros((M * N, M * N))
    for i in range(0, M * N):
        for j in range(0, M * N):
            if j == i - 1 - N and (i % N)-1 >= 0:
                H[i][j] = h[0][0]
            elif j == i-N:
                H[i][j] = h[0][1]
            elif j == i + 1 - N and (i % N)+1 < N:
                H[i][j] = h[0][2]
            elif j == i-1 and (i % N)-1 >= 0:
                H[i][j] = h[1][0]
            elif j == i:
                H[i][j] = h[1][1]
            elif j == i + 1 and (i % N)+1 < N:
                H[i][j] = h[1][2]
            elif j == i - 1 + N and (i % N)-1 >= 0:
                H[i][j] = h[2][0]
            elif j == i + N:
                H[i][j] = h[2][1]
            elif j == i + 1 + N and (i % N)+1 < N:
                H[i][j] = h[2][2]
    return H


def Q2():
    filter_h1 = 1/float(9)*np.ones((3, 3))
    filter_h2 = 1/float(8)*np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1]).reshape(3,3)
    filter_h3 = np.transpose(filter_h2)

    M = 12
    N = 16
    H_h1 = build_H(M, N, filter_h1)
    H_h2 = build_H(M, N, filter_h2)
    H_h3 = build_H(M, N, filter_h3)

    fig = plt.figure(2)
    fig.set_size_inches(18, 12)
    ax = fig.add_subplot(131)
    img = ax.imshow(H_h1, interpolation="None", cmap='binary_r')
    ax.set_title('H1')
    ax = fig.add_subplot(132)
    img = ax.imshow(H_h2, interpolation="None", cmap='binary_r')
    ax.set_title('H2')
    ax = fig.add_subplot(133)
    img = ax.imshow(H_h3, interpolation="None", cmap='binary_r')
    ax.set_title('H3')
    plt.colorbar(img, fraction=0.046, pad=0.04)
    plt.show()


# ==================
# Question 3
# ==================
def Q3():
    mdict = sio.loadmat("hw4_data/hw4_data/are_these_separable_filters.mat")
    print([k for k in mdict.keys() if not k.startswith("__")])

    for i, key in enumerate([k for k in mdict.keys() if not k.startswith("__")]):
        k = mdict[key]
        U, S, V = scipy.linalg.svd(k)
        compare = np.zeros(len(S))
        if S[0] > pow(10, -12):
            S[0] = 0
            print("{} seperable: {}".format(key, np.allclose(S, compare, atol=1e-12)))


# ==================
# Question 4
# ==================
def Q4():
    bil_mdict = sio.loadmat('hw4_data/hw4_data/bilateral.mat')
    img_noise = bil_mdict['img_noisy']
    fig = plt.figure(4, figsize=(20, 10))
    ax = fig.add_subplot(121)
    img = ax.imshow(img_noise, cmap='gray')
    ax.set_title('Original')
    fig.colorbar(img)

    ax = fig.add_subplot(122)
    img = ax.imshow(cv2.bilateralFilter(img_noise, 16, 0.35, 5), cmap='gray')
    ax.set_title('Bilateral Filter')
    fig.colorbar(img)


# ==================
# Question 5
# ==================
def Q5():
    img_ascent = cv2.cvtColor(cv2.imread('hw4_data/hw4_data/ascent.jpg'), cv2.COLOR_BGR2GRAY)
    cv2.imshow('ascent img', img_ascent)

    def calc_der(theta):
        x, y = np.mgrid[-11:11, -11:11]
        g_xy = (1/np.sqrt(2 * np.pi * pow(sigma, 2))) * np.exp((np.power(x, 2) + np.power(y, 2)) / float(-2 * (np.power(sigma, 2))))
        u = np.cos(theta)
        v = np.sin(theta)
        g_dx = - (x * g_xy) / float(np.power(sigma, 2))
        g_dy = - (y * g_xy) / float(np.power(sigma, 2))
        return u * g_dx + v * g_dy

    dirs = [35, 45, 90, 115]
    sigma = 1.2

    fig = plt.figure(5, figsize=(20, 10))
    plt.subplots_adjust(top=1.0, wspace=0.7)
    for i, theta in enumerate(dirs):
        fil_calc = calc_der(math.radians(theta)).T
        ax = fig.add_subplot(2, 5, i+1)
        ax.set_title('Theta:{}'.format(theta))
        im = ax.imshow(fil_calc, interpolation="None", cmap='jet')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="10%", pad=0.05)
        fig.colorbar(im, cax=cax)
        convolved = scipy.signal.convolve2d(img_ascent, fil_calc, 'same')
        ax2 = fig.add_subplot(1, 5, i+1)
        ax2.imshow(convolved, cmap='gray')


# ==================
# Question 6
# ==================
def Q6():
    mdict_of = sio.loadmat('hw4_data/hw4_data/imgs_for_optical_flow.mat')
    img_of = mdict_of['img1']
    I_blur = cv2.GaussianBlur(img_of, (7,7), 0.2)

    fig = plt.figure(7, figsize=(15, 10))
    ax = fig.add_subplot(1, 6, 1)
    ax.title.set_text('Original')
    ax.imshow(img_of, cmap='gray')

    ax = fig.add_subplot(162)
    ax.set_title('Blurred')
    ax.imshow(I_blur, cmap='gray')

    h_x1, h_x2 = cv2.getDerivKernels(1, 0, 3)
    I_x = cv2.sepFilter2D(I_blur, -1, h_x1, h_x2)
    ax = fig.add_subplot(163)
    ax.set_title('dx')
    ax.imshow(I_x, cmap='gray')

    h_y1, h_y2 = cv2.getDerivKernels(0, 1, 3)
    I_y = cv2.sepFilter2D(I_blur, -1, h_y1, h_y2)
    ax = fig.add_subplot(164)
    ax.set_title('dy')
    ax.imshow(I_y, cmap='gray')

    h_x1_2, h_x2_2 = cv2.getDerivKernels(2, 0, 3)
    I_x2 = cv2.sepFilter2D(I_blur, -1, h_x1_2, h_x2_2)
    ax = fig.add_subplot(165)
    ax.set_title('dxx')
    ax.imshow(I_x2, cmap='gray')

    h_y1_2, h_y2_2 = cv2.getDerivKernels(0, 2, 3)
    I_y2 = cv2.sepFilter2D(I_blur, -1, h_y1_2, h_y2_2)
    ax = fig.add_subplot(166)
    ax.set_title('dyy')
    ax.imshow(I_y2, cmap='gray')


def main():
    Q1()
    Q2()
    Q3()
    Q4()
    Q5()
    Q6()
    print('Done.')


if __name__ == '__main__':
    main()
