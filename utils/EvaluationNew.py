from sklearn import metrics
import numpy as np
import cv2

class Evaluation():
    # 计算各应用评价指标
    def __init__(self, label, pred):
        super(Evaluation, self).__init__()
        self.label = label / 255
        self.pred = pred / 255

    def ConfusionMatrix(self):
        raw = self.label.shape[0]
        col = self.label.shape[1]
        size = raw * col
        union = np.clip(((self.label + self.pred)), 0, 1)
        intersection = (self.label * self.pred)
        TP = int(intersection.sum())
        TN = int(size - union.sum())
        FP = int((self.pred - intersection).sum())
        FN = int((self.label - intersection).sum())

        # c_num_and = TP
        c_num_or = int(union.sum())
        # uc_num_and = TN
        uc_num_or = int(size - intersection.sum())

        return TP, TN, FP, FN, c_num_or, uc_num_or


class Index():
    # 计算各应用评价指标
    def __init__(self, TPSum, TNSum, FPSum, FNSum, c_Sum_or, uc_Sum_or):
        super(Index, self).__init__()
        self.TP = TPSum
        self.TN = TNSum
        self.FP = FPSum
        self.FN = FNSum
        self.c_num_and = TPSum
        self.c_num_or = c_Sum_or
        self.uc_num_and = TNSum
        self.uc_num_or = uc_Sum_or

    def CD_indicators(self):
        # 二分类混淆矩阵解释说明(0为未变化；1为变化)
        #                    0  1(pred)
        # [[a,b]   (label) 0[TP,FP]
        # [c, d]]          1[FN,TN]
        ######################################
        # Report_ConfusionMat = metrics.confusion_matrix(y_true=self.label.flatten(), y_pred=self.pred.flatten())
        # if Report_ConfusionMat.size == 1:
            # return 0.0, 0.0, 0.0
        # else:
        TP = self.TP
        TN = self.TN
        FP = self.FP
        FN = self.FN

        FA = FP / (FP + TN)
        MA = FN / (FN + TP)
        TE = (FP + FN) / (TP + TN + FP + FN)

        return FA*100, MA*100, TE*100

    # return OA, kappa, AA
    def Classification_indicators(self):
        TP = self.TP
        TN = self.TN
        FP = self.FP
        FN = self.FN

        OA = (TP + TN) / (TP + TN + FP + FN)
        kappa = metrics.cohen_kappa_score(label.flatten(), pred.flatten())
        AA = (TP / (TP + FN) + TN / (TN + FP)) / 2

        return OA*100, kappa*100, AA*100

    # Completeness, Correctness, Quality
    def Landsilde_indicators(self):
        TP = self.TP
        TN = self.TN
        FP = self.FP
        FN = self.FN

        Completeness = TP / (TP + FN)
        Correctness = TP / (TP + FP)
        Quality = TP / (TP + FP + FN)

        return Completeness*100, Correctness*100, Quality*100

    # return iou
    def IOU_indicator(self):
        c_num_and = self.c_num_and
        c_num_or = self.c_num_or
        uc_num_and = self.uc_num_and
        uc_num_or = self.uc_num_or

        c_iou = (c_num_and / c_num_or) * 100
        uc_iou = (uc_num_and / uc_num_or) * 100
        mIoU = (c_iou + uc_iou) / 2

        return mIoU, c_iou, uc_iou

    # return Precision, Recall, F1
    def ObjectExtract_indicators(self):
        TP = self.TP
        TN = self.TN
        FP = self.FP
        FN = self.FN

        OA = (TP + TN) / (TP + TN + FP + FN)
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F1 = (2 * Precision * Recall) / (Precision + Recall)

        return OA*100, Precision*100, Recall*100, F1*100