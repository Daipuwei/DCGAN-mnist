# -*- coding: utf-8 -*-
# @Time    : 2019/9/15 21:58
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @Blog    : https://daipuweiai.blog.csdn.net/
# @File    : main.py
# @Software: PyCharm

import os
import datetime
from Config.Config import Config
from DCGAN.DCGAN import DCGAN
from ImageGenerator.ImageGenerator import ImageGenerator

class CartooncFaceConfig(Config):

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

def run_main():
    """
       这是主函数
    """
    cfg = CartooncFaceConfig()
    #train_datagen = ImageGenerator(cfg.dataset_path,cfg.batch_size)
    dcgan = DCGAN(cfg)
    dcgan.train(20,cfg.batch_size)


if __name__ == '__main__':
    run_main()
 