import functools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp
import seg_hrnet

import utils

class FCN(nn.Module):

    def __init__(self, num_input_channels, num_output_classes, num_filters=64):
        super(FCN,self).__init__()

        self.conv1 = nn.Conv2d(num_input_channels, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters,        kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters,        kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(num_filters, num_filters,        kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(num_filters, num_filters,        kernel_size=3, stride=1, padding=1)
        self.last =  nn.Conv2d(num_filters, num_output_classes, kernel_size=1, stride=1, padding=0)

    def forward(self,inputs):
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.last(x)
        return x


def get_unet():
    return smp.Unet(
        encoder_name='resnet18', encoder_depth=3, encoder_weights=None,
        decoder_channels=(128, 64, 64), in_channels=4, classes=len(utils.NLCD_CLASSES)
    )

def get_fcn():
    return FCN(num_input_channels=4, num_output_classes=len(utils.NLCD_CLASSES), num_filters=64)


def get_hrnet():
    config = {
        "DATASET": {
            "NUM_CLASSES": len(utils.NLCD_CLASSES)
        },
        "MODEL": {
            "EXTRA": {
                "FINAL_CONV_KERNEL": 1,
                "NUM_INPUT_CHANNELS": 4
            }
        }
    }
    config["MODEL"]["EXTRA"].update(seg_hrnet.cfg_cls["hrnet_w32"])
    return seg_hrnet.HighResolutionNet(config)