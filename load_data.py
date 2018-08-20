import cv2
import numpy as np

from common.config import Config
from common.config import Config
from common.read_listfile import read_listfile

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'feature_extractor'))
from feature_extractor.feature_extractor import feature_extractor


def load_data(config, flag = 'all'):
    print('[{0}] load data ...'.format(flag))
    train_label_listfile, test_label_listfile = config.parse_label_listfile()

    data_config = config.parse_data_config()

    if flag == 'train':
        return get_samples_labels(train_label_listfile, data_config, 'train')
    elif flag == 'test':
        return get_samples_labels(test_label_listfile, data_config, 'test')
    else:
        return get_samples_labels(train_label_listfile, data_config, 'train')\
            , get_samples_labels(test_label_listfile, data_config, 'test')

def get_samples_labels(label_listfile, data_config, flag = 'train'):
    samples = []
    labels = []

    image_channel = data_config['image_channel']
    feature_type = data_config['feature_type']

    for label, listfiles in label_listfile.items():
        filename_list = read_listfile(listfiles)
        for filename in filename_list:
            if image_channel == 1:
                image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            elif image_channel == 3:
                image = cv2.imread(filename, cv2.IMREAD_COLOR)

            if image is None:
                print('no image: {0}'.format(filename))
                continue

            feature = feature_extractor(image, image_channel, feature_type)
            if flag == 'train':
                # [[feature1], [feature2], ...]
                samples.append(feature)
            elif flag == 'test':
                # [[[feature1]], [[feature2]], ...]
                samples.append(feature.reshape(1, -1))
            labels.append(label)
    return np.float32(samples), np.int32(labels)

if __name__ == '__main__':
    config_filename = './config/carpet/config.json'
    config = Config(config_filename)
    (x_train, y_train), (x_test, y_test) = load_data(config)
    print(x_train, y_train)
    print(x_test, y_test)

    x_train, y_train = load_data(config, 'train')
    print(x_train, y_train)