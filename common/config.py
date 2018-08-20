import json
import numpy as np
import os

class Singleton(type):
    def __init__(cls, name, bases, dict):
        super(Singleton, cls).__init__(name, bases, dict)
        cls._instance = None

    def __call__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = super(Singleton, cls).__call__(*args, **kw)
        return cls._instance

class Config(object):
    __metaclass__ = Singleton
    def __init__(self, filename):
        with open(filename, 'r') as f:
            self.config = json.load(f)

    def parse_config(self, key):
        return self.config[key]

    def parse_label_listfile(self):
        train_label_listfile_result = {}
        test_label_listfile_result = {}

        train_label_listfile = self.config['train_label_listfile']
        for k, v in train_label_listfile.items():
            if v:
                train_label_listfile_result[k] = v
        # print 'train_label_listfile: {0}'.format(train_label_listfile)

        test_label_listfile = self.config['test_label_listfile']
        for k, v in test_label_listfile.items():
            if v:
                test_label_listfile_result[k] = v
        # print 'test_label_listfile: {0}'.format(test_label_listfile)

        return train_label_listfile_result, test_label_listfile_result

    def parse_data_config(self):
        data_config = self.config['data']
        return data_config

    def parse_svm_config(self):
        svm_config = self.config['svm']
        return svm_config

if __name__ == '__main__':
    filename = '../config/carpet/config.json'

    config = Config(filename)

    description = config.parse_config('description')
    print(description)

    train_label_listfile, test_label_listfile = config.parse_label_listfile()
    print(train_label_listfile)
    print(test_label_listfile)

    data_config = config.parse_data_config()
    print(data_config)
    print(data_config['image_channel'])
    print(data_config['feature_type'])

    svm_config = config.parse_svm_config()
    print(svm_config)

    config2 = Config(filename)

    print(id(config), id(config2))