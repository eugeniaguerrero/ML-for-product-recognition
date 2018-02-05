import os
import shutil
import matplotlib.pyplot as plt
#matplotlib inline
#config InlineBackend.figure_format = 'retina'
from data_generator_bandw import *
import keras.backend as K
from keras.datasets import mnist
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.initializers import *
from keras.callbacks import *
from keras.utils.generic_utils import Progbar
from common import *
FOLDER_NUMBER = 0
ITERATION = 0


class WGANN(object):
    def __init__(self):
        RND = 777
        RUN = 'F'
        self.OUT_DIR = 'out/' + RUN
        self.TENSORBOARD_DIR = '/tensorboard/wgans/' + RUN
        # GPU #
        self.GPU = "1"
        # latent vector size
        self.Z_SIZE = IM_WIDTH
        # number of iterations D is trained for per each G iteration
        self.D_ITERS = 5
        self.BATCH_SIZE = 5
        self.ITERATIONS = 25000
        self.NO_CHANNELS = 1
        self.DG_losses = []
        self.D_true_losses = []
        self.D_fake_losses = []

        # write tensorboard summaries
        self.sw = tf.summary.FileWriter(self.TENSORBOARD_DIR)

        # save 10x10 sample of generated images
        self.samples_zz = np.random.normal(0., 1., (100, self.Z_SIZE))

        np.random.seed(RND)

        self.G = self.create_G()
        self.G.summary()
        self.D = self.create_D()
        self.DG = self.create_combined()
        self.DG.get_layer('D').trainable = False  # freeze D in generator training faze
        self.DG.compile(optimizer=RMSprop(lr=0.00005),loss=[self.wasserstein, 'sparse_categorical_crossentropy'])



        if not os.path.isdir(self.OUT_DIR): os.makedirs(self.OUT_DIR)
        ##MAYBE DELETE THIS!!
        K.set_image_dim_ordering('tf')





    # basically return mean(y_pred),
    # but with ability to inverse it for minimization (when y_true == -1)
    def wasserstein(self,y_true, y_pred):
        return K.mean(y_true * y_pred)


    def create_D(self):

        # weights are initlaized from normal distribution with below params
        weight_init = RandomNormal(mean=0., stddev=0.02)

        #CHANGED THIS
        input_image = Input(shape=(IM_WIDTH, IM_WIDTH, self.NO_CHANNELS), name='input_image')

        x = Conv2D(32, (3, 3),padding='same',name='conv_1',kernel_initializer=weight_init)(input_image)
        x = LeakyReLU()(x)
        x = MaxPool2D(pool_size=2)(x)
        x = Dropout(0.3)(x)

        x = Conv2D(64, (3, 3),padding='same',name='conv_2',kernel_initializer=weight_init)(x)
        x = MaxPool2D(pool_size=1)(x)
        x = LeakyReLU()(x)
        x = Dropout(0.3)(x)

        x = Conv2D(128, (3, 3),padding='same',name='conv_3',kernel_initializer=weight_init)(x)
        x = MaxPool2D(pool_size=2)(x)
        x = LeakyReLU()(x)
        x = Dropout(0.3)(x)

        x = Conv2D(256, (3, 3),padding='same',name='conv_4',
            kernel_initializer=weight_init)(x)
        x = MaxPool2D(pool_size=1)(x)
        x = LeakyReLU()(x)
        x = Dropout(0.3)(x)

        features = Flatten()(x)

        output_is_fake = Dense(
            1, activation='linear', name='output_is_fake')(features)

        output_class = Dense(
            10, activation='softmax', name='output_class')(features)

        return Model(inputs=[input_image], outputs=[output_is_fake, output_class], name='D')

    def create_G(self):
        DICT_LEN = 10
        EMBEDDING_LEN = self.Z_SIZE

        # weights are initlaized from normal distribution with below params
        weight_init = RandomNormal(mean=0., stddev=0.02)

        # class#
        input_class = Input(shape=(1, ), dtype='int32', name='input_class')
        # encode class# to the same size as Z to use hadamard multiplication later on
        e = Embedding(
            DICT_LEN, EMBEDDING_LEN,
            embeddings_initializer='glorot_uniform')(input_class)
        embedded_class = Flatten(name='embedded_class')(e)

        # latent var
        input_z = Input(shape=(self.Z_SIZE, ), name='input_z')

        # hadamard product
        h = multiply([input_z, embedded_class], name='h')

        # cnn part
        x = Dense(1024)(h)
        x = LeakyReLU()(x)

        x = Dense(128 * int(IM_WIDTH/4) * int(IM_WIDTH/4))(x)
        x = LeakyReLU()(x)
        x = Reshape((int(IM_WIDTH/4),int(IM_WIDTH/4), 128))(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(256, (5, 5), padding='same', kernel_initializer=weight_init)(x)
        x = LeakyReLU()(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(128, (5, 5), padding='same', kernel_initializer=weight_init)(x)
        x = LeakyReLU()(x)

        x = Conv2D(self.NO_CHANNELS, (2, 2),padding='same', activation='tanh',name='output_generated_image',kernel_initializer=weight_init)(x)
        return Model(inputs=[input_z, input_class], outputs=x, name='G')


    def create_combined(self):


        # # remember D dropout rates
        # for l in D.layers:
        #     if l.name.startswith('dropout'):
        #         l._rate = l.rate

        self.D.compile(
            optimizer=RMSprop(lr=0.00005),
            loss=[self.wasserstein, 'sparse_categorical_crossentropy'])

        input_z = Input(shape=(self.Z_SIZE,), name='input_z_')
        input_class = Input(shape=(1,), name='input_class_', dtype='int32')



        # create combined D(G) model
        output_is_fake, output_class = self.D(self.G(inputs=[input_z, input_class]))
        return Model(inputs=[input_z, input_class], outputs=[output_is_fake, output_class])


    def generate_samples(self,n=0, save=True):
        global FOLDER_NUMBER
        global ITERATION
        generated_classes = np.array(list(range(0, 10)) * 10)
        generated_images = self.G.predict([self.samples_zz, generated_classes.reshape(-1, 1)])
        #print(generated_images.shape)
        if FOLDER_NUMBER == 0:
            if os.path.exists('new_img_folder'):
                shutil.rmtree('new_img_folder')
            os.makedirs('new_img_folder')

        folder_name = os.path.join('new_img_folder', str(FOLDER_NUMBER))


        if os.path.exists(folder_name):
            shutil.rmtree(folder_name)
        os.makedirs(folder_name)
        print("WORKING")
        rr = []
        for c in range(generated_images.shape[0]):
            #CHANGED THIS
            file_name = str(c) + '.png'
            img = (generated_images[c,:,:,:]*127.5)+127.5
            #print(img.shape)
            #if save & ITERATION % 1 == 0:
            cv2.imwrite(os.path.join(folder_name, file_name), img)
            FOLDER_NUMBER += 1
            rr.append(img)
        img = np.hstack(rr)
        #print(img.shape)

        ITERATION += 1
        # plt.imsave(OUT_DIR + '/samples_%07d.png' % n, img, cmap=plt.cm.gray)
        return img


    def update_tb_summary(self,step, sample_images=True, save_image_files=True):

        s = tf.Summary()

        # losses as is
        for names, vals in zip((('D_real_is_fake', 'D_real_class'),
                                ('D_fake_is_fake', 'D_fake_class'), ('DG_is_fake',
                                                                     'DG_class')),
                               (self.D_true_losses, self.D_fake_losses, self.DG_losses)):
            v = s.value.add()
            v.simple_value = vals[-1][1]
            v.tag = names[0]

            v = s.value.add()
            v.simple_value = vals[-1][2]
            v.tag = names[1]

        # D loss: -1*D_true_is_fake - D_fake_is_fake
        v = s.value.add()
        v.simple_value = -self.D_true_losses[-1][1] - self.D_fake_losses[-1][1]
        v.tag = 'D loss (-1*D_real_is_fake - D_fake_is_fake)'

        # generated image
        if sample_images:
            img = self.generate_samples(step, save=save_image_files)
            s.MergeFromString(tf.Session().run(
                tf.summary.image('samples_%07d' % step,
                                 img.reshape([1, *img.shape]))))

        self.sw.add_summary(s, step)
        self.sw.flush()


        # fake = 1
        # real = -1

    def train(self):

        '''
        params = {'dir': 'training_data', 'batch_size': self.BATCH_SIZE,'shuffle': True}
        train_data_gen = DataGenerator(**params).generate()
        params2 = {'dir': 'validation_data', 'batch_size': self.BATCH_SIZE, 'shuffle': True}
        validation_data_gen = DataGenerator(**params).generate()'''

        progress_bar = Progbar(target=self.ITERATIONS)


        self.DG_losses = []
        self.D_true_losses = []
        self.D_fake_losses = []

        for it in range(self.ITERATIONS):

            # load mnist data
            #(X_train, y_train) = train_data_gen.__next__()
            #(X_test, y_test) = validation_data_gen.__next__()
            (X_train, y_train), (X_test, y_test) = mnist.load_data()

            # use all available 70k samples
            X_train = np.concatenate((X_train, X_test))
            y_train = np.concatenate((y_train, y_test))

            # convert to -1..1 range, reshape to (sample_i, 28, 28, 1)
            X_train = (X_train.astype(np.float32) - 127.5) / 127.5
            X_train = np.expand_dims(X_train, axis=3)
            #print(self.D_true_losses)


            if len(self.D_true_losses) > 0:
                progress_bar.update(
                    it,
                    values=[
                        ('D_real_is_fake', np.mean(self.D_true_losses[-5:], axis=0)[1]),
                        ('D_real_class', np.mean(self.D_true_losses[-5:], axis=0)[2]),
                        ('D_fake_is_fake', np.mean(self.D_fake_losses[-5:], axis=0)[1]),
                        ('D_fake_class', np.mean(self.D_fake_losses[-5:], axis=0)[2]),
                        ('D(G)_is_fake', np.mean(self.DG_losses[-5:], axis=0)[1]),
                        ('D(G)_class', np.mean(self.DG_losses[-5:], axis=0)[2])
                    ]
                )

            else:
                progress_bar.update(it)

            # 1: train D on real+generated images

            if (it % 1000) < 25 or it % 500 == 0:  # 25 times in 1000, every 500th
                d_iters = 100
            else:
                d_iters = self.D_ITERS

            for d_it in range(d_iters):

                # unfreeze D
                self.D.trainable = True
                for l in self.D.layers: l.trainable = True

                # # restore D dropout rates
                # for l in D.layers:
                #     if l.name.startswith('dropout'):
                #         l.rate = l._rate

                # clip D weights

                for l in self.D.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -0.01, 0.01) for w in weights]
                    l.set_weights(weights)

                # 1.1: maximize D output on reals === minimize -1*(D(real))
                # load mnist data
                #(X_train, y_train) = train_data_gen.__next__()
                #(X_test, y_test) = validation_data_gen.__next__()

                # IF TESTING MNIST
                (X_train, y_train), (X_test, y_test) = mnist.load_data()
                # use all available 70k samples
                X_train = np.concatenate((X_train, X_test))
                y_train = np.concatenate((y_train, y_test))
                X_train = np.expand_dims(X_train, axis=3)

                # convert to -1..1 range, reshape to (sample_i, 28, 28, 1)
                X_train = (X_train.astype(np.float32) - 127.5) / 127.5

                # draw random samples from real images
                index = np.random.choice(len(X_train), self.BATCH_SIZE, replace=False)
                real_images = X_train[index]
                real_images_classes = y_train[index]

                D_loss = self.D.train_on_batch(real_images, [-np.ones(self.BATCH_SIZE), real_images_classes])
                self.D_true_losses.append(D_loss)

                # 1.2: minimize D output on fakes

                zz = np.random.normal(0., 1., (self.BATCH_SIZE, self.Z_SIZE))
                generated_classes = np.random.randint(0, 10, self.BATCH_SIZE)
                generated_images = self.G.predict([zz, generated_classes.reshape(-1, 1)])

                D_loss = self.D.train_on_batch(generated_images, [np.ones(self.BATCH_SIZE), generated_classes])
                self.D_fake_losses.append(D_loss)

            # 2: train D(G) (D is frozen)
            # minimize D output while supplying it with fakes, telling it that they are reals (-1)

            # freeze D
            self.D.trainable = False
            for l in self.D.layers: l.trainable = False

            # # disable D dropout layers
            # for l in D.layers:
            #     if l.name.startswith('dropout'):
            #         l.rate = 0.

            zz = np.random.normal(0., 1., (self.BATCH_SIZE, self.Z_SIZE))
            generated_classes = np.random.randint(0, 10, self.BATCH_SIZE)

            DG_loss = self.DG.train_on_batch(
                [zz, generated_classes.reshape((-1, 1))],
                [-np.ones(self.BATCH_SIZE), generated_classes])

            self.DG_losses.append(DG_loss)

            if it % 10 == 0:
                self.update_tb_summary(it, sample_images=(it % 10 == 0), save_image_files=True)




wg = WGANN()
wg.train()