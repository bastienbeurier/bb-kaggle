from __future__ import division, print_function

import os, json
from glob import glob
import numpy as np
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D  # Conv2D: Keras2
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing import image


vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3,1,1))
def vgg_preprocess(x):
    x = x - vgg_mean
    return x[:, ::-1] # reverse axis rgb->bgr


class Vgg16BN():
    """The VGG 16 Imagenet model with Batch Normalization for the Dense Layers"""


    def __init__(self, size=(224,224), include_top=True, dropout=0.5):
        self.FILE_PATH = 'http://files.fast.ai/models/'
        self.create(size, include_top, dropout=0.5)
        self.get_classes()


    def get_classes(self):
        fname = 'imagenet_class_index.json'
        fpath = get_file(fname, self.FILE_PATH+fname, cache_subdir='models')
        with open(fpath) as f:
            class_dict = json.load(f)
        self.classes = [class_dict[str(i)][1] for i in range(len(class_dict))]

    def ConvBlock(self, layers, filters):
        model = self.model
        for i in range(layers):
            model.add(ZeroPadding2D((1, 1)))
            model.add(Conv2D(filters, kernel_size=(3, 3), activation='relu'))  # Keras2
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


    def FCBlock(self, dropout):
        model = self.model
        model.add(Dense(4096, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))


    def create(self, size, include_top, dropout=0.5):
        if size != (224,224):
            include_top=False

        model = self.model = Sequential()
        model.add(Lambda(vgg_preprocess, input_shape=(3,)+size, output_shape=(3,)+size))

        self.ConvBlock(2, 64)
        self.ConvBlock(2, 128)
        self.ConvBlock(3, 256)
        self.ConvBlock(3, 512)
        self.ConvBlock(3, 512)

        if not include_top:
            fname = 'vgg16_bn_conv.h5'
            model.load_weights(get_file(fname, self.FILE_PATH+fname, cache_subdir='models'))
            return

        model.add(Flatten())
        self.FCBlock(dropout)
        self.FCBlock(dropout)
        model.add(Dense(1000, activation='softmax'))

        fname = 'vgg16_bn.h5'
        model.load_weights(get_file(fname, self.FILE_PATH+fname, cache_subdir='models'))