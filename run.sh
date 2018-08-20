#!/bin/sh

mkdir model

config_file="./config/blink_detect.48x32/config.json"

python svm_train.py $config_file