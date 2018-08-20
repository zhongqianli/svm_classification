import cv2
import numpy as np

from common.config import Config
from common.config import Config
from common.read_listfile import read_listfile

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'feature_extractor'))
from feature_extractor.feature_extractor import feature_extractor

def liblinear_generate_data(config, data_dir):
    print('liblinear_generate_data ...')
    train_label_listfile, test_label_listfile = config.parse_label_listfile()
    print(train_label_listfile)
    data_config = config.parse_data_config()
    print(data_config)
    # x : [1 2 ...]
    # y : [[feature1], [feature2], ...]
    x_train, y_train = get_samples_labels(train_label_listfile, data_config)
    generate_data(x_train, y_train, data_dir, 'train')

    x_test, y_test = get_samples_labels(test_label_listfile, data_config)
    generate_data(x_test, y_test, data_dir, 'test')

# flag = 'train' or 'test'
def generate_data(X, Y, data_dir, flag = 'train'):
    filename = '{0}/{1}ing_data.txt'.format(data_dir ,flag)
    print(filename)
    for feature, label in zip(X, Y):
        feature_str = ''
        for i, feat in enumerate(feature):
            # feat = round(feat, 6)
            if i == 0:
                feature_str = '{0}:{1}'.format(i + 1, feat)
            else:
                feature_str = '{0} {1}:{2}'.format(feature_str, i + 1, feat)

        with open(filename, 'a+') as file:
            file.write('{0} {1}\n'.format(str(label), feature_str))


def get_samples_labels(label_listfile, data_config):
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

            # [[feature1], [feature2], ...]
            samples.append(feature)
            labels.append(label)
    return np.float32(samples), np.int32(labels)

if __name__ == '__main__':
    config_filename = './config/liblinear_generate_data/config.binary.json'
    config = Config(config_filename)

    liblinear_generate_data(config, './liblinear')