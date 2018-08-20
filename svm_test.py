import cv2
import numpy as np
import sys

from common.config import Config
from load_data import load_data


def svm_test(x_test, y_test, config):
    print('svm testing...')

    svm_config = config.parse_svm_config()
    model_name = svm_config['model_name']
    print('model_name: {0}'.format(model_name))

    if cv2.__version__[0] == '2':
        model = cv2.SVM()
        model.load(model_name)
    else:
        model = cv2.ml.SVM_load(model_name)

    print('data dims = {0}'.format(x_test.shape[-1]))
    labels = list(set(y_test))
    class_num = len(labels)
    print('class_num = {0}'.format(class_num))
    print('labels = {0}'.format(labels))

    total_count = [list(y_test).count(x) for x in labels]
    error_count = [x * 0 for x in range(class_num)]

    for x, y in zip(x_test, y_test):
        if cv2.__version__[0] == '2':
            res = np.int32(model.predict(x))
        else:
            res = np.int32(model.predict(x)[1].ravel())
        for i in range(class_num):
            if y == labels[i]:
                if y != res:
                    error_count[i] += 1
                break

    print('total_count = {0}'.format(total_count))
    print('error_count = {0}'.format(error_count))
    print('accurate = {0}'.format((np.float32(total_count) - np.float32(error_count)) / np.float32(total_count)))
    print('total accurate = {0}'.format(np.sum(np.float32(total_count) - np.float32(error_count)) / np.sum(np.float32(total_count))))

if __name__ == '__main__':
    config_file = './config/facelive.tiny/config.json'

    if len(sys.argv) == 1:
        print('usage: python svm_test.py <config_file>')
    else:
        config_file = sys.argv[1]

    config = Config(config_file)

    x_test, y_test= load_data(config, 'test')
    svm_test(x_test, y_test, config)