import numpy as np
import cv2

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'LBP'))
sys.path.append(os.path.join(os.getcwd(), 'HOG'))

import LBP.local_binary_pattern as local_binary_pattern
import HOG.hog as hog

def feature_extractor(image, image_channel, feature_type):
    if feature_type.find('lbp') == 0:
        if feature_type == 'lbp18':
            feature = LBP_hist_18(image, image_channel)
        elif feature_type == 'lbp216':
            feature = LBP_hist_216(image, image_channel)
        elif feature_type == 'lbp216_YCbCr':
            feature = LBP_hist_216_YCbCr(image)
        elif feature_type == 'lbp216_HSV':
            feature = LBP_hist_216_HSV(image)
        elif feature_type == 'lbp216_HSL':
            feature = LBP_hist_216_HSL(image)
        elif feature_type == 'lbp216_YCbCr_HSV':
            # feature = LBP_hist_216_YCbCr_HSV(image)
            feature = LBP_hist_216_YCbCr(image)
            feature = np.append(feature, LBP_hist_216_HSV(image))
        elif feature_type == 'lbp216_YCbCr_HSV2':
            feature = LBP_hist_216_YCbCr_HSV(image)
            # feature = LBP_hist_216_YCbCr(image)
            # feature = np.append(feature, LBP_hist_216_HSV(image))
        elif feature_type == 'lbp18_YCbCr_HSV_4':
            feature = LBP_hist_18_YCbCr_HSV_4(image)
    elif feature_type.find('hog') == 0:
        w, h = feature_type.split('.')[-1].split('x')
        w = int(w)
        h = int(h)
        size = (w, h)
        feature = HOG_cv2(image, size)
    else:
        feature = np.reshape(image, (-1, 1))
        feature = feature.squeeze()
    return feature

def LBP_hist_18(image, image_channel):
    R = 1
    P = 8
    if image_channel == 1:
        feature = local_binary_pattern.calc_LBP_hist(image, P, R)
    elif image_channel == 3:
        b, g, r = cv2.split(image)
        b_feature = local_binary_pattern.calc_LBP_hist(b, P, R)
        g_feature = local_binary_pattern.calc_LBP_hist(g, P, R)
        r_feature = local_binary_pattern.calc_LBP_hist(r, P, R)
        feature = b_feature
        feature = np.append(feature, g_feature)
        feature = np.append(feature, r_feature)

    return feature

def LBP_hist_216(image, image_channel):
    R = 2
    P = 16
    if image_channel == 1:
        feature = local_binary_pattern.calc_LBP_hist(image, P, R)
    elif image_channel == 3:
        b, g, r = cv2.split(image)
        b_feature = local_binary_pattern.calc_LBP_hist(b, P, R)
        g_feature = local_binary_pattern.calc_LBP_hist(g, P, R)
        r_feature = local_binary_pattern.calc_LBP_hist(r, P, R)
        feature = b_feature
        feature = np.append(feature, g_feature)
        feature = np.append(feature, r_feature)

    return feature

def LBP_hist_216_YCbCr(image):
    '''
    YCbCr color texture
    :param image: color image
    :return:
    '''
    R = 2
    P = 16
    YCbCr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Y, Cb, Cr = cv2.split(YCbCr_image)
    Y_feature = local_binary_pattern.calc_LBP_hist(Y, P, R)
    Cb_feature = local_binary_pattern.calc_LBP_hist(Cb, P, R)
    Cr_feature = local_binary_pattern.calc_LBP_hist(Cr, P, R)
    feature = Y_feature
    feature = np.append(feature, Cb_feature)
    feature = np.append(feature, Cr_feature)

    return feature

def LBP_hist_216_HSV(image):
    '''
    HSV color texture
    :param image: color image
    :return:
    '''
    R = 2
    P = 16
    HSV_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    H, S, V = cv2.split(HSV_image)
    H_feature = local_binary_pattern.calc_LBP_hist(H, P, R)
    S_feature = local_binary_pattern.calc_LBP_hist(S, P, R)
    V_feature = local_binary_pattern.calc_LBP_hist(V, P, R)
    feature = H_feature
    feature = np.append(feature, S_feature)
    feature = np.append(feature, V_feature)

    return feature

def LBP_hist_216_HSL(image):
    '''
    HSL color texture
    :param image: color image
    :return:
    '''
    R = 2
    P = 16
    HSL_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    H, S, L = cv2.split(HSL_image)
    H_feature = local_binary_pattern.calc_LBP_hist(H, P, R)
    S_feature = local_binary_pattern.calc_LBP_hist(S, P, R)
    L_feature = local_binary_pattern.calc_LBP_hist(L, P, R)
    feature = H_feature
    feature = np.append(feature, S_feature)
    feature = np.append(feature, L_feature)

    return feature

def LBP_hist_216_YCbCr_HSV(image):
    '''
    YCrCb_HSV color texture
    :param image: color image
    :return:
    '''
    R = 2
    P = 16

    YCrCb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(YCrCb_image)
    Y_feature = local_binary_pattern.calc_LBP_hist(Y, P, R)
    Cr_feature = local_binary_pattern.calc_LBP_hist(Cr, P, R)
    Cb_feature = local_binary_pattern.calc_LBP_hist(Cb, P, R)
    feature = Y_feature
    feature = np.append(feature, Cr_feature)
    feature = np.append(feature, Cb_feature)

    HSV_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(HSV_image)
    H_feature = local_binary_pattern.calc_LBP_hist(H, P, R)
    S_feature = local_binary_pattern.calc_LBP_hist(S, P, R)
    V_feature = local_binary_pattern.calc_LBP_hist(V, P, R)
    feature = np.append(feature, H_feature)
    feature = np.append(feature, S_feature)
    feature = np.append(feature, V_feature)

    return feature

def LBP_hist_18_YCbCr_HSV_4(image):
    '''
    YCrCb_HSV color texture
    :param image: color image
    :return:
    '''
    R = 1
    P = 8

    YCrCb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(YCrCb_image)
    Cr_feature = local_binary_pattern.calc_LBP_hist(Cr, P, R)
    Cb_feature = local_binary_pattern.calc_LBP_hist(Cb, P, R)
    feature = Cb_feature
    feature = np.append(feature, Cr_feature)

    HSV_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(HSV_image)
    H_feature = local_binary_pattern.calc_LBP_hist(H, P, R)
    S_feature = local_binary_pattern.calc_LBP_hist(S, P, R)
    feature = np.append(feature, H_feature)
    feature = np.append(feature, S_feature)

    return feature

def LBP_hist_324(image):
    R = 3
    P = 24
    feature = local_binary_pattern.calc_LBP_hist(image, P, R)
    return feature

def LBP_hist_multi_resolution(image):
    feature = []
    P = 8
    R = 1
    feature81 = local_binary_pattern.calc_LBP_hist(image, P, R)
    feature.extend(feature81)

    P = 16
    R = 2
    feature216 = local_binary_pattern.calc_LBP_hist(image, P, R)
    feature.extend(feature216)

    P = 24
    R = 3
    feature324 = local_binary_pattern.calc_LBP_hist(image, P, R)
    feature.extend(feature324)

    feature = np.float32(feature)
    return feature

def HOG_64(image):
    feature = hog.hog(image)
    return feature

def HOG_cv2(image, size):
    h = hog.HOG(size)
    feature = h.calc_hog_feature(image)
    return feature

if __name__ == '__main__':
    image = cv2.imread("../resources/images/face.jpg", cv2.IMREAD_COLOR)

    # feature = feature_extractor(image, 3, 'lbp216_YCbCr_HSV')
    # print(type(feature))
    # print(feature)

    image = cv2.resize(image, (32, 32))
    feature = feature_extractor(image, 1, 'hog_cv2.32x32')
    print(type(feature), len(feature))
    print(feature)

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.resize(image, (64, 64))
    # feature = feature_extractor(image, 1, 'lbp18')
    # print(type(feature), len(feature))
    # print(feature)

    cv2.imshow("image", image)
    cv2.waitKey()