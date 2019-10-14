# -*- coding: utf-8 -*-
# @Time    : 2019/9/15 21:58
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @Blog    : https://daipuweiai.blog.csdn.net/
# @File    : train.py
# @Software: PyCharm

import os
import datetime
from Config.Config import Config
from DCGAN.DCGAN import DCGAN
from MnistGenerator.MnistGenerator import MnistGenerator

class MnistConfig(Config):

    def __init__(self):
        #super(Config, self).__init__()
        Config.__init__(self)
        time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.save_weight_dir = os.path.join(Config.get_save_weight_dir(self),time)
        if not os.path.exists(self.save_weight_dir):
            os.mkdir(self.save_weight_dir)
        self.result_path = os.path.join(Config.get_result_path(self),time)
        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path)
        self.batch_size = 256

        print("模型保存在：{}".format(self.save_weight_dir))
        print("训练结果保存在：{}".format(self.result_path))


def run_main():
    """
       这是主函数
    """

    # 训练模型
    cfg =  MnistConfig()
    dcgan = DCGAN(cfg)
    train_datagen = MnistGenerator(int(cfg.batch_size/2))
    dcgan.train(train_datagen,1000,20,cfg.batch_size)      # 训练模型

if __name__ == '__main__':
    run_main()
 
