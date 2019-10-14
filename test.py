# -*- coding: utf-8 -*-
# @Time    : 2019/9/30 22:14
# @Author  : Dai PuWei
# @Site    : 广州山越有限公司
# @File    : test.py
# @Software: PyCharm

import os
import datetime
import matplotlib.pyplot as plt

from Config.Config import Config
from DCGAN.DCGAN import DCGAN

class MnistConfig(Config):

    def __init__(self):
        # super(Config, self).__init__()
        Config.__init__(self)

def generator_image(dcgan,row,col):
    """
    这是生成手写数字的函数
    :param dcgan: dcgan模型类
    :param row: 行数
    :param col: 列数
    :return:
    """
    images = dcgan.generator_batch_images(row * col)  # 生成批量数据
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    image_path = os.path.join(dcgan.config.test_result_path, "mnist-{}.png".format(time))

    fig, axs = plt.subplots(row, col)
    cnt = 0
    for i in range(row):
        for j in range(col):
            axs[i, j].imshow(images[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(image_path, dpi=600)
    plt.show()
    plt.close()
    print("图片保存路径为:",image_path)

def run_main():
    """
       这是主函数
    """
    # 测试模型
    cfg = MnistConfig()
    dcgan_weight_path = os.path.abspath("./model/20190930225619/Epoch999dcgan_loss0.91778discriminator_loss0.59186.h5")
    discriminator_weight_path = \
        os.path.abspath("./model/20190930225619/Epoch999discriminator_loss0.59186.h5")
    dcgan = DCGAN(cfg,dcgan_weight_path=dcgan_weight_path,discriminator_weight_path=discriminator_weight_path)
    test_result_path = os.path.abspath("./test_result")
    if not os.path.exists(test_result_path):
        os.mkdir(test_result_path)
    generator_image(dcgan,10,10)


if __name__ == '__main__':
    run_main()

