from __future__ import division

import os, sys, time, random
import torch
import json
import numpy as np

from config.config import MODEL_PATH

def get_model_path(dataset_name, network_arch, random_seed):
    if not os.path.isdir(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    model_path = os.path.join(MODEL_PATH, "{}_{}_{}".format(dataset_name, network_arch, random_seed))
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    return model_path

def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600*need_hour) / 60)
    need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
    return need_hour, need_mins, need_secs

def time_string():
    ISOTIMEFORMAT='%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
    return string

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def manipulate_net_architecture(model_arch, net, num_classes):
    if model_arch in ["vgg16", "vgg19"]:
        num_ftrs = net.classifier[6].in_features
        net.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)
    elif model_arch in ["resnet18", "resnet50", "resnet101", "resnet152"]:
        num_ftrs = net.fc.in_features
        net.fc = torch.nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError("Network {} not supported".format(model_arch))
    return net
