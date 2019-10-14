# -*- coding: utf-8 -*-
# @Time    : 2019/9/15 9:39
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @Blog    : https://daipuweiai.blog.csdn.net/
# @File    : Config.py
# @Software: PyCharm

import os

class Config(object):

    def __init__(self):
        self.generator_input_dim = (100,)
        self.discriminator_input_dim = (28,28,1)

        self.BatchNormalization_Momentum = 0.9
        self.dropout_prob = 0.4
        self.LeakyReLU_alpha = 0.2
        self.init_learning_rate = 0.0002
        self.beta1 = 0.5

        self.epoch = 100000
        self.batch_size = 256
        self.save_interval = 1

        self.save_weight_dir = os.path.abspath("./model")
        if not os.path.exists(self.save_weight_dir):
            os.mkdir(self.save_weight_dir)
        self.dataset_path = os.path.abspath("./faces")
        self.train_result_path = os.path.abspath("./train_result")
        if not os.path.exists(self.train_result_path):
            os.mkdir(self.train_result_path)
        self.test_result_path = os.path.abspath("./test_result")
        if not os.path.exists(self.test_result_path):
            os.mkdir(self.test_result_path)

    def get_save_weight_dir(self):
        return self.save_weight_dir

    def get_result_path(self):
        return self.train_result_path

def run_main():
    """
       这是主函数
    """


if __name__ == '__main__':
    run_main()
 