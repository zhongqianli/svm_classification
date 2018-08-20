import cv2
import numpy as np
import sys
import platform

from common.config import Config
from load_data import load_data


def svm_train(x_train, y_train, config):
    print('svm training...')

    data_config = config.parse_data_config()
    print('data config: {0}'.format(data_config))

    svm_config = config.parse_svm_config()

    print('svm config: {0}'.format(svm_config))

    model_name = svm_config['model_name']
    kernel = svm_config['kernel']
    is_auto_train = svm_config['autotrain']
    is_binary_classify = svm_config['binary_classify']
    C = svm_config['c']
    gamma = svm_config['gamma']

    if cv2.__version__[0] == '2':
        model = cv2.SVM()
        if kernel == 'linear':
            params = dict(kernel_type=cv2.SVM_LINEAR,
                          svm_type=cv2.SVM_C_SVC,
                          C=C,
                          gamma=gamma)
        elif kernel == 'rbf':
            params = dict(kernel_type=cv2.SVM_RBF,
                          svm_type=cv2.SVM_C_SVC,
                          C=C,
                          gamma=gamma)
        else:
            model.setKernel(cv2.ml.SVM_LINEAR)

        model.train(x_train, y_train, params=params)
    else:
        model = cv2.ml.SVM_create()
        if kernel == 'linear':
            model.setKernel(cv2.ml.SVM_LINEAR)
        elif kernel == 'rbf':
            model.setKernel(cv2.ml.SVM_RBF)
        else:
            model.setKernel(cv2.ml.SVM_LINEAR)

        model.setGamma(gamma)
        model.setC(C)
        model.setType(cv2.ml.SVM_C_SVC)

        if is_auto_train:
            print('train auto')
            model.trainAuto(x_train, cv2.ml.ROW_SAMPLE, y_train \
                            , kFold=10 \
                            , Cgrid=cv2.ml.SVM_getDefaultGridPtr(cv2.ml.SVM_C) \
                            , gammaGrid=cv2.ml.SVM_getDefaultGridPtr(cv2.ml.SVM_GAMMA) \
                            , pGrid=cv2.ml.SVM_getDefaultGridPtr(cv2.ml.SVM_P) \
                            , nuGrid=cv2.ml.SVM_getDefaultGridPtr(cv2.ml.SVM_NU) \
                            , coeffGrid=cv2.ml.SVM_getDefaultGridPtr(cv2.ml.SVM_COEF) \
                            , degreeGrid=cv2.ml.SVM_getDefaultGridPtr(cv2.ml.SVM_DEGREE) \
                            , balanced=is_binary_classify)
        else:
            model.train(x_train, cv2.ml.ROW_SAMPLE, y_train)


    print('data dims = {0}'.format(x_train.shape[-1]))
    labels = list(set(y_train))
    class_num = len(labels)
    print('class_num = {0}'.format(class_num))
    print('labels = {0}'.format(labels))

    total_count = [list(y_train).count(x) for x in labels]
    error_count = [x*0 for x in range(class_num)]

    for x, y in zip(x_train, y_train):
        if cv2.__version__[0] == '2':
            res = np.int32(model.predict(x))
        else:
            res = np.int32(model.predict(x.reshape(1, -1))[1].ravel())
        for i in range(class_num):
            if y == labels[i]:
                if y != res:
                    error_count[i] += 1
                break

    print('total_count = {0}'.format(total_count))
    print('error_count = {0}'.format(error_count))
    print('accurate = {0}'.format((np.float32(total_count) - np.float32(error_count)) / np.float32(total_count)))
    print('total accurate = {0}'.format(np.sum(np.float32(total_count) - np.float32(error_count)) / np.sum(np.float32(total_count))))

    model.save(model_name)

if __name__ == '__main__':
    print('python' + platform.python_version())
    print('opencv' + cv2.__version__)
    config_file = './config/blink_detect.48x32.tiny/config.binary.json'

    if len(sys.argv) == 1:
        print('usage: python svm_train.py <config_file>')
    else:
        config_file = sys.argv[1]

    config = Config(config_file)

    x_train, y_train= load_data(config, 'train')
    svm_train(x_train, y_train, config)

    from svm_test import svm_test
    x_test, y_test = load_data(config, 'test')
    svm_test(x_test, y_test, config)