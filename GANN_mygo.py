from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D,Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from common import *
import os
from folder_manipulation import *
import matplotlib.pyplot as plt

import sys

import numpy as np


class DCGAN():
    def __init__(self):
        self.img_rows = 100
        self.img_cols = 100
        self.channels = 3

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(10,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        noise_shape = (10,)
        model = Sequential()
        model.add(Dense(10,input_shape = (10,)))
        model.summary()
        model.add(Dense(64 * 25 * 25, activation="relu"))
        model.summary()
        model.add(Reshape((25, 25, 64)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(3, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(10, activation='sigmoid'))

        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        #(X_train, _), (_, _) = mnist.load_data()
        X_train = self.get_data()#[:,:,:,1]
        #print(X_train2.shape)
        labels = []
        for i in range(X_train.shape[0]):
            labels.append([0,0,0,0,0,0,0,0,0,1])
        labels = np.array(labels)
        print(X_train.shape)
        print(labels.shape)
        print('*****************')

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        #X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------
            '''
            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]
            labels2 = labels[idx]

            # Sample noise and generate a half batch of new images
            noise = np.random.normal(0, 1, (half_batch, 100))

            y_generator_data = self.get_random_labels(imgs.shape[0])

            #prediction_data = np.expand_dims(np.array([0,0,0,0,0,0,0,0,0,1]),axis = 0)
            gen_imgs = self.generator.predict(y_generator_data)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, y_generator_data)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 10)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)'''

            # ---------------------
            #  Train Generator
            # ---------------------

            y_generator_data = self.get_random_labels(X_train.shape[0])
            #print(y_generator_data.shape)
            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(y_generator_data, y_generator_data)

            # Plot the progress
            #print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)


    def get_random_labels(self,length):
        y_generator_data = []
        for i in range(length):
            y_generator_data.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        y_generator_data = np.array(y_generator_data)
        return y_generator_data


    def get_data(self):
        path = os.path.join('training_data','3073781011456')
        return dstack_folder(path)[:50]

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = self.get_random_labels(25)
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        # fig.suptitle("DCGAN: Generated digits", fontsize=12)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='brg')
                axs[i, j].axis('off')
                cnt += 1
        os.mkdir("dcgan/images/%d" %epoch)
        count2 = 0
        for image in gen_imgs:
            cv2.imwrite("dcgan/images/%d" %epoch + "/" + str(count2) + '.jpg',(image*127.5)+127.5)
            count2 += 1
        fig.savefig("dcgan/images/mnist_%d.png" % epoch)
        plt.close()

if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=4000, batch_size=32, save_interval=50)