# -*- coding: utf-8 -*-
# @Time    : 2019/9/15 9:26
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @Blog    : https://daipuweiai.blog.csdn.net/
# @File    : DCGAN.py
# @Software: PyCharm

import os
import numpy as np
import matplotlib.pyplot as plt

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
from keras.utils.generic_utils import Progbar

class DCGAN(object):

    def __init__(self,config,discriminator_weight_path = None,dcgan_weight_path=None):
        """
        这是DCGAN的初始化函数
        :param config: 网络模型参数配置类
        :param discriminator_weight_path: 网络模型参数配置类
        :param dcgan_weight_path: 网络模型参数配置类
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
        if discriminator_weight_path is not None:
            self.discriminator_model.load_weights(discriminator_weight_path,by_name=True)
        self.discriminator_model.compile(loss='binary_crossentropy',
                                         optimizer=self.optimizier)

        # 构建DCGAN模型并进行编译
        dcgan_input = Input(shape=self.config.generator_input_dim)
        dcgan_output = self.discriminator_model(self.generotor_model(dcgan_input))
        self.discriminator_model.trainable = False

        self.dcgan = Model(dcgan_input,dcgan_output)
        if dcgan_weight_path is not None:
            self.dcgan.load_weights(dcgan_weight_path,by_name=True)
        self.dcgan.compile(optimizer=self.optimizier, loss='binary_crossentropy')

    def build_generator_model(self):
        """
        这是构建生成器网络的函数
        :return:返回生成器模型generotor_model
        """
        noise = Input(shape=self.config.generator_input_dim, name="generator_input")

        x = Dense(256*7*7,input_shape=self.config.generator_input_dim,name="dense1")(noise)
        x = BatchNormalization(momentum=self.config.BatchNormalization_Momentum,name="bn1")(x)
        x = Activation('relu',name="relu1")(x)
        x = Reshape((7,7,256),name="reshape")(x)

        x = Conv2DTranspose(128,kernel_size=3,strides=2,padding='same',name="deconv1")(x)
        x = BatchNormalization(momentum=self.config.BatchNormalization_Momentum,name="bn2")(x)
        x = Activation('relu',name="relu2")(x)

        x = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same',name="deconv2")(x)
        x = BatchNormalization(momentum=self.config.BatchNormalization_Momentum,name="bn3")(x)
        x = Activation('relu',name="relu3")(x)

        x = Conv2DTranspose(32, kernel_size=3,padding='same',name="deconv3")(x)
        x = BatchNormalization(momentum=self.config.BatchNormalization_Momentum,name="bn4")(x)
        x = Activation('relu',name="relu4")(x)

        x = Conv2DTranspose(self.config.discriminator_input_dim[2], kernel_size=3,padding='same',name="deconv4")(x)
        x = Activation('tanh',name="generator_output")(x)

        model = Model(noise,x)
        model.summary()

        return model

    def build_discriminator_model(self):
        """
        这是构造判别器模型的函数
        :return: 返回判别器模型discriminator_model
        """
        image = Input(shape=self.config.discriminator_input_dim, name="discriminator_input")

        x = Conv2D(64,kernel_size=3,strides=2,padding='same',name="conv1")(image)
        x = LeakyReLU(self.config.LeakyReLU_alpha,name="leakyrelu1")(x)
        x = Dropout(self.config.dropout_prob,name="dropout1")(x)

        x = Conv2D(128,kernel_size=3,strides=2,padding='same',name="conv2")(x)
        x = LeakyReLU(self.config.LeakyReLU_alpha,name="leakyrelu2")(x)
        x = Dropout(self.config.dropout_prob,name="dropout2")(x)

        x = Conv2D(256,kernel_size=3,strides=2,padding='same',name="conv3")(x)
        x = LeakyReLU(self.config.LeakyReLU_alpha,name="leakyrelu3")(x)
        x = Dropout(self.config.dropout_prob,name="dropout3")(x)

        x = Conv2D(512,kernel_size=3,strides=2,padding='same',name="conv4")(x)
        x = LeakyReLU(self.config.LeakyReLU_alpha,name="leakyrelu4")(x)
        x = Dropout(self.config.dropout_prob,name="dropout4")(x)

        x = Flatten(name="flatten1")(x)
        x = Dense(1,name="dense")(x)
        x = Activation('sigmoid',name="discriminator_output")(x)

        model = Model(image,x)
        model.summary()

        return model

    def train(self,train_datagen,epoch,k,batch_size=256):
        """
        这是DCGAN的训练函数
        :param train_generator:训练数据生成器
        :param epoch:训练周期
        :param batch_size:小批量样本规模
        :param k:训练判别器次数
        :return:
        """
        half_batch = int(batch_size/2)
        length = train_datagen.get_batch_num()
        for ep in np.arange(1,epoch+1):
            dcgan_losses = []
            d_losses = []
            probar = Progbar(length)
            print("Epoch {}/{}".format(ep,epoch))
            iter = 0
            while True:
                # 数据集遍历完成停止循环
                if train_datagen.get_epoch() != ep:
                    break

                iter +=1

                d_loss = []
                for i in np.arange(k):
                    # 获取真实图片及其标签
                    batch_real_images = train_datagen.next_batch()
                    batch_real_images_labels = truncnorm.rvs(0.7, 1.2, size=(half_batch, 1))
                    # 生成一个batch_size的噪声用于生成图片，并制造标签
                    batch_noise = truncnorm.rvs(-1,1,size = (half_batch , self.config.generator_input_dim[0]))
                    batch_gen_images = self.generotor_model.predict(batch_noise)
                    batch_gen_images_labels = truncnorm.rvs(0.0, 0.3, size=(half_batch, 1))

                    # 合并真图与假图及其对应的标签
                    #print(np.shape(batch_gen_images))
                    #print(np.shape(batch_real_images))
                    batch_images = np.concatenate([batch_gen_images, batch_real_images],axis=0)
                    batch_images_labels = np.concatenate((batch_gen_images_labels,batch_real_images_labels))
                    # 训练判别器
                    _d_loss = self.discriminator_model.train_on_batch(batch_images,batch_images_labels)
                    d_loss.append(_d_loss)
                d_loss = np.average(d_loss)

                # 生成一个batch_size的噪声来训练生成器
                batch_noise = truncnorm.rvs(-1,1,size=(half_batch ,self.config.generator_input_dim[0]))
                batch_noise_label = truncnorm.rvs(0.7,1.2,size=(half_batch ,1))
                dcgan_loss = self.dcgan.train_on_batch(batch_noise,batch_noise_label)

                dcgan_losses.append(dcgan_loss)
                d_losses.append(d_loss)

                # 更新进度条
                probar.update(iter,[("dcgan_loss",np.average(dcgan_losses[:iter])),
                                    ("discriminator_loss",np.average(d_losses[:iter]))])

            if int(ep % self.config.save_interval) == 0:
                dcgan_model = "Epoch%ddcgan_loss%.5fdiscriminator_loss%.5f.h5" \
                              % (ep,np.average(dcgan_losses), np.average(d_losses))
                discriminator_model = "Epoch%ddiscriminator_loss%.5f.h5" \
                              % (ep,np.average(d_losses))
                #self.dcgan.save(os.path.join(self.config.save_weight_dir,'dcgan.h5'))
                self.dcgan.save(os.path.join(self.config.save_weight_dir,dcgan_model))
                self.discriminator_model.save(os.path.join(self.config.save_weight_dir, discriminator_model))
                self.save_image(epoch)

    def save_image(self,epoch):
        """
        这是保存生成图片的函数
        :param epoch:周期数,用于图片命名需要
        :return:
        """
        rows, cols = 5, 5
        images = self.generator_batch_images(rows*cols)         # 生成批量数据

        fig, axs = plt.subplots(rows, cols)
        cnt = 0
        for i in range(rows):
            for j in range(cols):
                axs[i, j].imshow(images[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(self.config.result_path,"mnist-{0:0>5}.png".format(epoch)), dpi=600)
        plt.close()

    def generator_batch_images(self,batch_size):
        """
        这是生成批量规模图像的函数
        :param batch_size: 批量规模
        :return:
        """
        # 生成一个batch_size的噪声用于生成图片
        batch_noise = truncnorm.rvs(-1, 1, size=(batch_size, self.config.generator_input_dim[0]))
        batch_gen_images = self.generotor_model.predict(batch_noise)
        return batch_gen_images

    def generator_image(self):
        """
        这是生成批量规模图像的函数
        :param batch_size: 批量规模
        :return:
        """
        image = self.generator_batch_images(1)[0]
        return image