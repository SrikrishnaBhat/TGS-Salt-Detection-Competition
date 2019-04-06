import numpy as np
import pandas as pd
from tgs_nets.u_net import build_u_net_normal
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

class GAN_class(object):

    def __init__(self, input):
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator(input)
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator(input)

        # The generator takes noise as input and generates imgs
        z = Input(shape=self.img_shape)
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)

    def build_generator(self, input_shape):

        model = Sequential()
        model.add(Conv2D(32, kernel_size=3, input_shape=input_shape, padding="same"))
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(32, kernel_size=3, padding="same"))
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        return model

    def build_discriminator(self, input):
        model = Sequential()
        return model

    def save_generator(self, file_name):
        self.generator.save(file_name)
