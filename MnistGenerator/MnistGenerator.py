# -*- coding: utf-8 -*-
# @Time    : 2019/9/30 15:53
# @Author  : Dai PuWei
# @Site    : 广州山越有限公司
# @File    : MnistGenerator.py
# @Software: PyCharm

import numpy as np
from keras.datasets import mnist

class MnistGenerator(object):

    def __init__(self,batch_size):
        """
        这个mnist数据集生成器类的初始化函数
        :param batch_size: 小规模样本规模大小
        """
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = (x_train.astype(np.float32) - 127.5) / 127.5
        x_test = (x_test.astype(np.float32) - 127.5) / 127.5
        self.train_images = np.concatenate((x_train,x_test))
        self.train_images = np.expand_dims(self.train_images,axis=-1)
        self.size = len(self.train_images)
        self.batch_num = int(round(self.size/batch_size))
        self.index = np.random.permutation(self.size)
        self.train_images = self.train_images[self.index]

        # print(self.index)
        # print(type(self.index))
        #print(np.shape(self.train_images))
        #print(type(self.train_images))

        self.batch_size = batch_size
        self.epoch = 1
        self.start = 0
        self.end = 0
        self.finish_flag = False

    def _next_batch(self):
        while True:
            batch_images_list = np.array([])
            if self.finish_flag:        # 数据集遍历万一次
                np.random.shuffle(self.index)
                self.finish_flag = False
                self.epoch +=1
            self.end = int(np.min([self.size,self.start+self.batch_size]))
            batch_images_list = np.concatenate((batch_images_list,self.index[self.start:self.end]))
            batch_size = self.end - self.start
            if self.end == self.size:       # 数据集刚好被均分，遍历结束
                self.finish_flag = True
            if batch_size < self.batch_size:            # 小批量规模小于预定闺蜜，基本上发生在最后一组
                np.random.shuffle(self.train_images)
                batch_images_list = np.concatenate((batch_images_list,self.index[0:self.batch_size-batch_size]))
                self.start = self.batch_size-batch_size
                self.epoch += 1
            else:
                self.start = self.end
            #print(batch_images_list)
            yield self.train_images[batch_images_list.astype(np.int32)]

    def next_batch(self):
        datagen = self._next_batch()
        return datagen.__next__()

    def get_batch_num(self):
        return self.batch_num

    def get_epoch(self):
        return self.epoch

def run_main():
    """
    这是主函数
    """
    mnistgen = MnistGenerator(2)
    x = mnistgen.next_batch()
    print(np.shape(x))



if __name__ == '__main__':
    run_main()