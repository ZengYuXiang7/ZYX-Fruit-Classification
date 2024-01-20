import torch
import torchvision
import matplotlib.pyplot as plt

from ArgParser import *
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 封装一个用于处理蔬果数据集(VegetableFruitDatasetLoader)数据的类对象
class VFDataloader:
    def __init__(self, dataset_path=DATASET_PATH, img_size: int = 224, batch_size: int = 256, num_workers: int = 4):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),  # 调整图像大小
            transforms.ToTensor(),  # 图片转换为Tensor类型
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 图像数据变为正态分布
                                 std=[0.229, 0.224, 0.225])
        ])

    def get_dataloader(self, dataset: str):
        if dataset == 'fruits-360' or dataset == 'fruits360':
            #  来自于 Kaggle 数据集: https://www.kaggle.com/datasets/moltean/fruits
            train_dataset = datasets.ImageFolder(root=f'{self.dataset_path}/fruits-360_dataset/fruits-360/Training',
                                                 transform=self.transform)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                      num_workers=self.num_workers)

            test_dataset = datasets.ImageFolder(root=f'{self.dataset_path}/fruits-360_dataset/fruits-360/Test',
                                                transform=self.transform)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                     num_workers=self.num_workers)
            return train_loader, test_loader
        elif dataset == 'FSC' or dataset == 'FSIFV':
            # 来自于 Kaggle 数据集: https://www.kaggle.com/datasets/raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables
            pass  # 占位符
        elif dataset == 'FRC':
            # 来自于 Kaggle 数据集: https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification?rvi=1
            def binary_target_transform(target):
                # ImageFolder会自动根据文件夹名称映射得到相应的类别编号，但是我们希望将此多分类数据集改造变成二分类数据集
                # 将以 'fresh' 开头映射为 1，其他的映射为 0

                label = train_dataset.classes[target]
                return 1 if label.startswith('fresh') else 0

            train_dataset = datasets.ImageFolder(
                root=os.path.join(self.dataset_path, 'fresh-and-stale-classification/dataset/Train'),
                transform=self.transform,
                target_transform=binary_target_transform  # 使用自定义的标签转换函数
            )
            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True,
                num_workers=self.num_workers
            )
            test_dataset = datasets.ImageFolder(
                root=os.path.join(self.dataset_path, 'fresh-and-stale-classification/dataset/Test'),
                transform=self.transform,
                target_transform=binary_target_transform  # 使用自定义的标签转换函数
            )
            test_loader = DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers
            )
            return train_loader, test_loader

        raise ValueError("Dataset Name Not Found!")


if __name__ == '__main__':
    vfd = VFDataloader()
    train_loader, test_loader = vfd.get_dataloader(dataset='fruits-360')
    images, labels = next(iter(train_loader))

    print(images)
    print(labels)
