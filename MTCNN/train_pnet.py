# 训练P网络

import Nets
import Train

if __name__ == '__main__':
    net = Nets.PNet()

    trainer = Train.Trainer(net, "/Users/karson/Downloads/Dataset/12")  # 网络、保存参数、训练数据
    trainer()  # 调用训练方法
