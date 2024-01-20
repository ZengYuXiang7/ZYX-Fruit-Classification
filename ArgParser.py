import os
import random
import numpy as np
import torch
import torch.nn as nn
import argparse


# 全局常量
EPS = 1e-6
DATASET_PATH    =   '/Data'
MODEL_SAVE_PATH =   './result/weight'


# 通用型模块
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        

# 任务超参数设置模块
class ArgParser:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self):
        if self.__initialized:
            return
        self.parser = argparse.ArgumentParser()
         
        # 通用性参数
        self.parser.add_argument("--model", default = None, type = str,              # 模型
                    help = "The model needed for the specific task")
        self.parser.add_argument("--dataset", default = None, type = str,            # 数据集
                                help = "Dataset for train/test model.")
        self.parser.add_argument("--seed", default = 3407, type = int,               # 随机数设置
                                 help = "Random seed for initializing training.")
        self.parser.add_argument("--mode", default = "train/test", type = str,       # 模式:训练或测试
                                 help = "Select Train/Test mode")
        
        # 基本参数，learning_rate, batch_size, epoch, 激活函数
        self.parser.add_argument("-lr", "--learning_rate", default = 0.01, type = float,
                                 help = "Epoch for train model.")
        self.parser.add_argument("--batch-size", default = 128, type = int,
                                 help = "Batch size for train/test model.")
        self.parser.add_argument("--epoch", default = 10, type = int,
                                 help = "Epoch for train model.")
        self.parser.add_argument("-A", "--activate", default = 'relu', type = str,
                                 help = "Activata-function for model.")
        
        # 基本参数延伸的更多参数
        self.parser.add_argument("--lr-decay-step", default = 10, type = int,        # 用来决定每隔多少步衰减学习率
                            help = "Used to decide how many steps to decay the learning rate every.")
        self.parser.add_argument("--lr-decay-gamma", default = 0.5, type = float,    # 用来决定每次学习率衰减的幅度是多少
                            help = "Used to decide how much the learning rate decays each time.")
        self.parser.add_argument("--weight-decay",  default = 1e-5, type = float, 
                                 help = "Used to decide weight decay in optimizer.")
        self.parser.add_argument("--input-size", nargs = 4, type = int,              # 设置模型按批处理读入图像的维度，由于无法直接接收元组类型，可以将其设为接收四个 int 参数
                            help = "Set channel for images that the model reads in batches.")
        
        self.parser.add_argument("--load-path", default = None, type = str,          # 模型权重载入路径
                    help = "Load the pre-trained model for evaluation.")
        self.parser.add_argument("--num-workers", default = 4, type = int,
                            help = "Number of threads to loading dataset.")          # 用于加载数据集的线程数
        
        
        # 针对分类任务的特定参数
        self.parser.add_argument("-aug", "--aug-mask", default="000000", type=str,   # 数据增强的方案掩码
                                 help="Mask used to specify data augmentation")
        self.parser.add_argument("-at", "--attack-type", default=None, type=str,     # 对抗训练使用的攻击类型
                                 help="Attack for adversarial train/test.")
        self.parser.add_argument("-an", "--attack-norm", default=None, type=str,     # 对抗训练使用的攻击约束
                                 help="Attack for adversarial train/test.")
        self.parser.add_argument("--test-size", default=0.2, type=float,             # 测试集的比例
                                 help="Test set proportion.")
    
         # 设置分类模型需要分类的个数、编码解码结构的中间维度
        self.parser.add_argument("--num-classes", default = 10, type = int,
                    help = "The number of classes the model needs to classify.") 
        self.parser.add_argument('-dim', "--dim", default = 10, type = int,
                    help = "The dimension of latent space.")
        
        self.__initialized = True

    def parse_args(self):
        return self.parser.parse_args()



if __name__ == '__main__':
    # 测试单例模式
    parser1 = ArgParser()
    parser2 = ArgParser()
    print(parser1 is parser2)