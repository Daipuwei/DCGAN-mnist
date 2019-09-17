# -*- coding: utf-8 -*-
# @Time    : 2019/9/15 9:26
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @Blog    : https://daipuweiai.blog.csdn.net/
# @File    : DCGAN.py
# @Software: PyCharm

import os
import numpy as np
from scipy.stats import truncnorm
from keras import Model
from keras import Input
from keras import Sequential
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.datasets import mnist
import matplotlib.pyplot as plt

class DCGAN(object):

    def __init__(self,config):
        """
        这是DCGAN的初始化函数
        :param config: 网络模型参数配置类
        """
        # 初始化网络相关超参数类
        self.config = config

        # 构建生成器与判别器
        self.generotor_model = self.build_generator_model()
        self.discriminator_model = self.build_discriminator_model()

        # 构建DCGAN的优化器,并编译判别器
        self.optimizier = Adam(lr=self.config.init_learning_rate,
                               beta_1=self.config.beta1,
                               decay=1e-8)
        self.discriminator_model.compile(loss='binary_crossentropy',
                                         optimizer=self.optimizier, metrics=['accuracy'])


        # 构建DCGAN模型并进行编译
        dcgan_input = Input(shape=self.config.generator_input_dim)
        dcgan_output = self.discriminator_model(self.generotor_model(dcgan_input))
        self.discriminator_model.trainable = False

        self.dcgan = Model(dcgan_input,dcgan_output)
        self.dcgan.compile(optimizer=self.optimizier, loss='binary_crossentropy', metrics=['accuracy'])

    def build_generator_model(self):
        """
        这是构建生成器网络的函数
        :return:返回生成器模型generotor_model
        """
        model = Sequential()
        model.add(Dense(256*7*7,input_shape=self.config.generator_input_dim))
        model.add(BatchNormalization(momentum=self.config.BatchNormalization_Momentum))
        model.add(Activation('relu'))
        model.add(Reshape((7,7,256)))

        #generotor_model.add(UpSampling2D(size=(2,2)))
        #generotor_model.add(Conv2D(64,5,5,padding='same'))
        model.add(Conv2DTranspose(128,kernel_size=3,strides=2,padding='same'))
        model.add(BatchNormalization(momentum=self.config.BatchNormalization_Momentum))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(64, kernel_size=3, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=self.config.BatchNormalization_Momentum))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(32, kernel_size=3,padding='same'))
        model.add(BatchNormalization(momentum=self.config.BatchNormalization_Momentum))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(self.config.discriminator_input_dim[2], kernel_size=3,padding='same'))
        model.add(Activation('tanh'))

        model.summary()

        noise = Input(shape=self.config.generator_input_dim)
        image = model(noise)

        return Model(noise,image)

    def build_discriminator_model(self):
        """
        这是构造判别器模型的函数
        :return: 返回判别器模型discriminator_model
        """
        model = Sequential()
        model.add(Conv2D(64,kernel_size=3,strides=2,input_shape=self.config.discriminator_input_dim,padding='same'))
        model.add(LeakyReLU(self.config.LeakyReLU_alpha))
        model.add(Dropout(self.config.dropout_prob))

        model.add(Conv2D(128,kernel_size=3,strides=2,padding='same'))
        model.add(LeakyReLU(self.config.LeakyReLU_alpha))
        model.add(Dropout(self.config.dropout_prob))

        model.add(Conv2D(256,kernel_size=3,strides=2,padding='same'))
        model.add(LeakyReLU(self.config.LeakyReLU_alpha))
        model.add(Dropout(self.config.dropout_prob))

        model.add(Conv2D(512,kernel_size=3,strides=2,padding='same'))
        model.add(LeakyReLU(self.config.LeakyReLU_alpha))
        model.add(Dropout(self.config.dropout_prob))

        model.add(Flatten())
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.summary()

        image = Input(shape=self.config.discriminator_input_dim)
        validity = model(image)

        return Model(image,validity)

    def train(self,k,batch_size=256):
        """
        这是DCGAN的训练函数
        :param train_generator:训练数据生成器
        :param batch_size:小批量样本规模
        :param k:训练判别器次数
        :return:
        """
        (x_train, y_train), (X_test, y_test) = mnist.load_data()
        x_train = (x_train.astype(np.float32) - 127.5) / 127.5
        x_train = np.expand_dims(x_train,axis=3)
        for epoch in np.arange(1,self.config.epoch+1):
            half_batch = int(batch_size / 2)
            g_losses = []
            g_accuracy = []
            d_losses = []
            d_accuracy = []

            d_loss = []
            d_acc = []
            for i in np.arange(k):
                # 获取真实图片
                idx = np.random.randint(0, x_train.shape[0], half_batch)
                batch_real_images = x_train[idx]
                # 生成一个batch_size的噪声用于生成图片
                batch_noise =truncnorm.rvs(-1,1,size = (half_batch , self.config.generator_input_dim[0]))
                batch_gen_images = self.generotor_model.predict(batch_noise)
                batch_images = np.concatenate((batch_gen_images,batch_real_images))
                # 构造标签
                batch_gen_images_labels = truncnorm.rvs(0.0,0.3,size=(half_batch ,1))
                batch_real_images_labels = truncnorm.rvs(0.7,1.2,size=(half_batch ,1))
                batch_images_labels = np.concatenate((batch_gen_images_labels,batch_real_images_labels))
                # 训练判别器
                d_result = self.discriminator_model.train_on_batch(batch_images,batch_images_labels)
                d_loss.append(d_result[0])
                d_acc.append(d_result[1])
            d_loss = np.average(d_loss)
            d_acc = np.average(d_acc)

            # 生成一个batch_size的噪声来训练生成器
            batch_noise = truncnorm.rvs(-1,1,size=(half_batch ,self.config.generator_input_dim[0]))
            batch_noise_label = truncnorm.rvs(0.7,1.2,size=(half_batch ,1))
            g_result = self.dcgan.train_on_batch(batch_noise,batch_noise_label)

            g_losses.append(g_result[0])
            g_accuracy.append(g_result[1])
            d_losses.append(d_loss)
            d_accuracy.append(d_acc)

            str = "Epoch:%05d,generator_loss:%.5f,generator_acc:%.5f,discriminator_loss:%.5f,discriminator_accuracy%.5f" \
                  % (epoch, g_result[0],g_result[1],d_loss, d_acc)
            print(str)


            if epoch % self.config.save_interval == 0:
                model_dcgan = "Epoch%05dgenerator_loss%.5fgenerator_accuracy%.5fdiscriminator_loss%.5fdiscriminator_accuracy%.5f.h5" \
                          % (epoch,np.average(g_losses), np.average(g_accuracy),np.average(d_losses),np.average(d_accuracy))
                #self.dcgan.save(os.path.join(self.config.save_weight_dir,'dcgan.h5'))
                self.dcgan.save(os.path.join(self.config.save_weight_dir,model_dcgan))
                self.save_image(epoch)

    def save_image(self,epoch):
        """
        这是保存生成图片的函数
        :param images: 图片集
        :param epoch:周期数
        :return:
        """
        rows, cols = 5, 5
        noise = truncnorm.rvs(-1, 1, size=(rows * cols, self.config.generator_input_dim[0]))
        images = self.generotor_model.predict(noise)

        fig, axs = plt.subplots(rows, cols)
        cnt = 0
        for i in range(rows):
            for j in range(cols):
                axs[i, j].imshow(images[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(self.config.result_path,"mnist-{0:0>5}.png".format(epoch)), dpi=300)
        plt.close()


def run_main():
    """
       这是主函数
    """


if __name__ == '__main__':
    run_main()
 