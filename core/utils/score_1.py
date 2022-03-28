"""Evaluation Metrics for Semantic Segmentation"""
import torch
import numpy as np

class Evaluator_1(object):
    def __init__(self, num_class = 2):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        pre_image = pre_image.data.cpu().numpy()
        gt_image = gt_image.cpu().numpy()
        pre_image = np.argmax(pre_image, axis=1)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def add_batch_sigmoid(self, gt_image, pre_image):
        pre_image = torch.sigmoid(pre_image)
        pre_image = pre_image.data.cpu().numpy()
        gt_image = gt_image.cpu().numpy()
        pre_image = (pre_image > 0.95).astype(np.uint8)
        pre_image = pre_image.squeeze(1)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def add_batch_semantic(self, gt_image, pre_image):
        pre_image = pre_image.data.cpu().numpy()
        gt_image = gt_image.cpu().numpy()
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def add_batch_metric(self, gt_image, pre_image):
        pre_image = pre_image.data.cpu().numpy()
        gt_image = gt_image.cpu().numpy()
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def f1_miou(self):
        c = self.confusion_matrix.shape[0]
        col_sum = np.sum(self.confusion_matrix, axis=1)# TP+FN
        raw_sum = np.sum(self.confusion_matrix, axis=0)# FP+TN
        # print(self.confusion_matrix)
        TP = []
        for i in range(c):
            TP.append(self.confusion_matrix[i,i])

        TP = np.array(TP)
        FN = col_sum - TP
        FP = raw_sum - TP

        f1_m = []
        iou_m =[]
        acc_m = []
        precision_m = []
        for i in range(c):
            f1 = TP[i] *2 /(TP[i]*2 + FP[i]+FN[i])
            f1_m.append(f1)
            iou = TP[i]  /(TP[i] + FP[i]+FN[i])
            iou_m.append(iou)
            precision = TP[i]  /(TP[i] + FP[i])
            precision_m.append(precision)
            acc = TP[i]  /(TP[i] + FN[i])
            acc_m.append(acc)


        f1_array = np.array(f1_m)
        iou_array = np.array(iou_m)
        precision_m = np.array(precision_m)
        acc_array = np.array(acc_m)
        mF1 = np.mean(f1_array)
        mIoU = np.mean(iou_array)
        mPrecision = np.mean(precision_m)
        mAcc = np.mean(acc_m)

        over_acc = TP.sum() / self.confusion_matrix.sum()
        return f1_array,mF1,iou_array,mIoU,acc_array,mAcc,over_acc

    def f1_miou_test(self):
        c = self.confusion_matrix.shape[0]
        col_sum = np.sum(self.confusion_matrix, axis=1)# TP+FN
        raw_sum = np.sum(self.confusion_matrix, axis=0)# FP+TN
        # print(self.confusion_matrix)
        TP = []
        for i in range(c):
            TP.append(self.confusion_matrix[i,i])

        TP = np.array(TP)
        FN = col_sum - TP
        FP = raw_sum - TP

        f1_m = []
        iou_m =[]
        recall_m = []
        precision_m = []
        for i in range(c):
            f1 = TP[i] *2 /(TP[i]*2 + FP[i]+FN[i])
            f1_m.append(f1)
            iou = TP[i]  /(TP[i] + FP[i]+FN[i])
            iou_m.append(iou)
            precision = TP[i]  /(TP[i] + FP[i])
            precision_m.append(precision)
            recall = TP[i]  /(TP[i] + FN[i])
            recall_m.append(recall)


        f1_array = np.array(f1_m)
        iou_array = np.array(iou_m)
        precision_array = np.array(precision_m)
        recall_array = np.array(recall_m)
        # mF1 = np.mean(f1_array)
        # mIoU = np.mean(iou_array)
        # mPrecision = np.mean(precision_m)
        # mAcc = np.mean(acc_m)

        over_acc = TP.sum() / self.confusion_matrix.sum()
        return f1_array,iou_array,recall_array,precision_array,over_acc

    def f1_miou_test_plot(self):
        c = self.confusion_matrix.shape[0]
        col_sum = np.sum(self.confusion_matrix, axis=1)# TP+FN
        raw_sum = np.sum(self.confusion_matrix, axis=0)# FP+TN
        # print(self.confusion_matrix)
        TP = []
        for i in range(c):
            TP.append(self.confusion_matrix[i,i])

        TP = np.array(TP)
        FN = col_sum - TP
        FP = raw_sum - TP

        f1_m = []
        iou_m =[]
        acc_m = []
        precision_m = []
        for i in range(c):
            f1 = TP[i] *2 /(TP[i]*2 + FP[i]+FN[i])
            f1_m.append(f1)
            iou = TP[i]  /(TP[i] + FP[i]+FN[i])
            iou_m.append(iou)
            precision = TP[i]  /(TP[i] + FP[i])
            precision_m.append(precision)
            acc = TP[i]  /(TP[i] + FN[i])
            acc_m.append(acc)


        f1_array = np.array(f1_m)
        iou_array = np.array(iou_m)
        precision_array = np.array(precision_m)
        acc_array = np.array(acc_m)
        TP_array = np.array(TP)
        FP_array = np.array(FP)
        # mF1 = np.mean(f1_array)
        # mIoU = np.mean(iou_array)
        # mPrecision = np.mean(precision_m)
        # mAcc = np.mean(acc_m)

        over_acc = TP.sum() / self.confusion_matrix.sum()
        return f1_array,iou_array,acc_array,precision_array,over_acc,TP_array,FP_array
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
