import numpy as np
import matplotlib.pyplot as plt

from ArgParser import *


class ClsMetrics:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.confusion_mtx = np.zeros((num_classes, num_classes), dtype=np.int32)

    def set_confusion_mtx(self, confusion_mtx):
        self.confusion_mtx = confusion_mtx
        return self
    
    def get_confusion_mtx(self):
        return self.confusion_mtx[:]

    def clear_confusion_mtx(self):
        # 每一轮 epoch 都要清空统计信息
        self.confusion_mtx = np.zeros((self.num_classes, self.num_classes), dtype=np.int32)
    
    def stats(self, pred, real):
        for t, p in zip(real, pred):
            self.confusion_mtx[t, p] += 1
        return self
    
    def acc_topk(self, output, target, topk=(1,)):
        """计算 top-k 准确率"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size).item())
            return np.array(res)
        
    def multi_metrics(self, percentage = True):
        tp = np.diag(self.confusion_mtx)
        fp = np.sum(self.confusion_mtx, axis=0) - tp
        fn = np.sum(self.confusion_mtx, axis=1) - tp
        tn = np.sum(self.confusion_mtx) - tp - fp - fn

        accuracy = np.sum(tp) / np.sum(self.confusion_mtx)
        precision = np.mean(tp / (tp + fp + EPS))
        recall = np.mean(tp / (tp + fn + EPS))
        f1_score = 2 * precision * recall / (precision + recall + EPS)

        ans = np.array([accuracy, precision, recall, f1_score])
        return ans * 100 if percentage else ans
    
    # 不仅能够用于二分类任务，且当多分类任务拆成多个二分类任务来看的时候同样可以使用这个函数
    # 不过如果多分类任务使用下面的函数计算某个类的性能则将得到一个偏高的评估数值
    def binary_metrics(self, class_index, percentage = True):
        tp = self.confusion_mtx[class_index, class_index]
        fp = np.sum(self.confusion_mtx[:, class_index]) - tp
        fn = np.sum(self.confusion_mtx[class_index, :]) - tp
        tn = np.sum(self.confusion_mtx) - tp - fp - fn

        accuracy = (tp + tn) / np.sum(self.confusion_mtx)
        precision = tp / (tp + fp + EPS)
        recall = tp / (tp + fn + EPS)
        specificity = tn / (tn + fp + EPS)
        f1_score = 2 * precision * recall / (precision + recall + EPS)

        ans = np.array([accuracy, precision, recall, specificity, f1_score])
        return  ans * 100 if percentage else ans 


    def mul_log(self,  percentage = True):
        return "Acc = {:.2f}, Precision = {:.2f}, Recall = {:.2f}, F1_score = {:.2f}".format(*self.multi_metrics(percentage))
    
    def bin_log(self):
        res = '\n' + self.bin_log_ith(0) + '\n' + self.bin_log_ith(1) + '\n'
        # 提取混淆矩阵的元素
        TP = self.confusion_mtx[0, 0]
        FP = self.confusion_mtx[0, 1]
        FN = self.confusion_mtx[1, 0]
        TN = self.confusion_mtx[1, 1]
        # 构建混淆矩阵的输出字符串
        confusion_mtx_str = f"TP = {TP}, FP = {FP}, FN = {FN}, TN = {TN}"
        return res + confusion_mtx_str
        
        
    def bin_log_ith(self, class_index, percentage = True):
        return "Acc = {:.2f}, Prec = {:.2f}, Recall(Sens) = {:.2f}, Spec = {:.2f}, F1 = {:.2f}".format(*self.binary_metrics(class_index, percentage))
    
    
    

if __name__ == '__main__':
    print("Test Case 1: ")
    multiclass_confusion_mtx = np.array([
        [90, 2, 3, 1, 1, 1, 1, 0, 0, 1],
        [1, 85, 2, 1, 3, 2, 0, 2, 2, 2],
        [3, 1, 80, 3, 0, 1, 1, 4, 3, 4],
        [1, 1, 4, 75, 5, 2, 1, 5, 3, 3],
        [2, 1, 0, 3, 80, 3, 4, 2, 2, 3],
        [1, 2, 1, 2, 2, 90, 1, 0, 0, 1],
        [0, 2, 2, 2, 3, 1, 88, 1, 0, 1],
        [1, 3, 4, 4, 2, 0, 1, 85, 0, 0],
        [1, 2, 3, 2, 4, 1, 2, 0, 85, 0],
        [2, 3, 3, 2, 2, 3, 3, 1, 1, 80]
    ], dtype=np.int32)
    MulClf = ClsMetrics(10).set_confusion_mtx(multiclass_confusion_mtx)
    print(MulClf.mul_log())

    # 创建 2x2 的二分类混淆矩阵
    print("Test Case 2: ")
    binary_confusion_mtx = np.array([
        [35, 5],
        [3, 57]
    ], dtype=np.int32)
    BiClf = ClsMetrics(2).set_confusion_mtx(binary_confusion_mtx)
    print(BiClf.bin_log())