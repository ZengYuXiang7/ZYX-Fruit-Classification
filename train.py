import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging

from models import *
from ArgParser import *

from tqdm import tqdm
from metrics import ClsMetrics
from FruitData import VFDataloader


def setup_logging(args):
    log_dir = './log'
    os.makedirs(log_dir, exist_ok=True)  
    log_file = os.path.join(log_dir, f'{args.model}-{args.dataset}-C{args.num_classes}-E{args.epoch}.log')  
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger()


def train_model(args, model, criterion, optimizer, train_loader):
    # 是否训练二分类模型
    is_binary = (args.num_classes == 2)
    logger = setup_logging(args)
    metricser = ClsMetrics(args.num_classes)
    for epoch in range(args.epoch):
        model.train().to('mps')
        running_loss, acc_topk = 0.0, np.array([0.0, 0.0, 0.0])
        progress_bar = tqdm(train_loader, desc = f'Epoch {epoch + 1} / {args.epoch}')
        for images, labels in progress_bar:
            images, labels = images.to('mps'), labels.to('mps')
            
            outputs = model(images)            # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()                    # 反向传播
            optimizer.step()                   # 更新权重
            optimizer.zero_grad()              # 清除梯度

            running_loss += loss.item()
            metricser.stats(torch.argmax(outputs, dim = 1), labels)

            # 多分类任务：计算 acc@1, acc@3, acc@5
            if not is_binary:
                acc_topk += metricser.acc_topk(outputs, labels, topk = (1, 3, 5))
            
            progress_bar.set_postfix(loss = running_loss / len(train_loader))

        avg_loss = running_loss / len(train_loader)
        acc_topk /= len(train_loader)
        
        # 训练过程的信息写入日志文件
        if is_binary:
            logger.info(f"Epoch [{epoch + 1}/{args.epoch}], Loss: {avg_loss:.4f}" + metricser.bin_log())
        else:
            logger.info(f"Epoch [{epoch + 1}/{args.epoch}], Loss: {avg_loss:.4f}, Top-1 Acc: {acc_topk[0]:.2f}, Top-3 Acc: {acc_topk[1]:.2f}, Top-5 Acc: {acc_topk[2]:.2f}")

    # 保存模型
    torch.save(model.state_dict(), f'./res/{args.model}-{args.dataset}-C{args.num_classes}-E{args.epoch}.pth')




'''
备选模型:
--model resnet18
--model resnet34
--model densenet121
--model densenet161
--model efficientnet_b0
--model efficientnet_b1
--model efficientnet_b3
--model efficientnet_b5
--model efficientnet_b7
--model mobilenet_v2
--model mobilenet_v3_small
--model mobilenet_v3_large
'''


# python train.py --input-size 32 3 100 100  -lr 0.01 --epoch 3 --dataset fruits360 --num-classes 131  --model resnet18
# python train.py --input-size 32 3 100 100  -lr 0.01 --epoch 3 --dataset FRC        --num-classes 2  --model resnet18
if __name__ == '__main__':
    args = ArgParser().parse_args()
    seed_everything(args.seed)
    vfd = VFDataloader(img_size = args.input_size[-1], batch_size = args.input_size[0], num_workers = args.num_workers)
    train_loader, test_loader = vfd.get_dataloader(dataset = args.dataset)
    model = get_model(args.model, args.num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = args.learning_rate)
    train_model(args, model, criterion, optimizer, train_loader)
    