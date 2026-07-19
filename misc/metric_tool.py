import numpy as np

###################       metrics      ###################
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict

    def clear(self):
        self.initialized = False


###################      cm metrics      ###################
class ConfuseMatrixMeter(AverageMeter):
    """Computes and stores the average and current value"""
    def __init__(self, n_class):
        super(ConfuseMatrixMeter, self).__init__()
        self.n_class = n_class

    def update_cm(self, pr, gt, weight=1):
        """获得当前混淆矩阵，并计算当前F1得分，并更新混淆矩阵"""
        val = get_confuse_matrix(num_classes=self.n_class, label_gts=gt, label_preds=pr)
        self.update(val, weight)
        current_score = cm2F1(val)
        return current_score

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict
    
class FairnessMeter(AverageMeter):
    """Computes and stores the average and current value"""
    def __init__(self, n_class,positive_class=1):
        super(FairnessMeter, self).__init__()
        self.n_class = n_class
        self.positive_class = positive_class
        self.initialized = False
        self.sum_prot = None
        self.sum_unprot = None

    def update_cm(self, pr_prot, gt_prot, pr_unprot, gt_unprot, weight=1):
        val_unprot = get_confuse_matrix(num_classes=self.n_class, label_gts=gt_unprot, label_preds=pr_unprot)
        val_prot = get_confuse_matrix(num_classes=self.n_class, label_gts=gt_prot, label_preds=pr_prot)
        if not self.initialized:
            self.sum_prot = val_prot
            self.sum_unprot = val_unprot
            self.initialized = True
        else:
            self.sum_prot += val_prot
            self.sum_unprot += val_unprot

        # Return fairness for the current batch
        return cm2fairness(val_prot, val_unprot, self.positive_class)

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict

def harmonic_mean(xs):
    harmonic_mean = len(xs) / sum((x+1e-6)**-1 for x in xs)
    return harmonic_mean


def cm2F1(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # acc_cls = np.nanmean(recall)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    return mean_F1

def cm2fairness(confusion_matrix_prot,confusion_matrix_unprot,positive_class=1):
    
    eps = np.finfo(np.float32).eps

    tp_prot = np.diag(confusion_matrix_prot)
    tp_unprot = np.diag(confusion_matrix_unprot)

    ### ACCURACY PARITY AP ###

    acc_prot = tp_prot.sum() / (confusion_matrix_prot.sum() + eps)
    acc_unprot = tp_unprot.sum() / (confusion_matrix_unprot.sum() + eps)
    ap = acc_prot / (acc_unprot + eps)

    ### DISPARATE IMPACT DI ###

    success_prot = confusion_matrix_prot.sum(axis=0)[positive_class]
    success_unprot = confusion_matrix_unprot.sum(axis=0)[positive_class]
    
    full_prot = confusion_matrix_prot.sum()
    full_unprot = confusion_matrix_unprot.sum()

    di = (success_prot / full_prot) / (success_unprot / full_unprot + eps)

    ### EQUAL OPPORTUNITY EO ###

    pos_prot = confusion_matrix_prot.sum(axis=1)[positive_class]
    pos_unprot = confusion_matrix_unprot.sum(axis=1)[positive_class]

    tp_class_prot = confusion_matrix_prot[positive_class, positive_class]
    tp_class_unprot = confusion_matrix_unprot[positive_class, positive_class]

    eo = (tp_class_prot / (pos_prot + eps)) / (tp_class_unprot / pos_unprot + eps)


    return {
        'EO': eo,
        'DI': di,
        'AP': ap
    }
    

def cm2score(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # acc_cls = np.nanmean(recall)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2*recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    # ---------------------------------------------------------------------- #
    # 2. Frequency weighted Accuracy & Mean IoU
    # ---------------------------------------------------------------------- #
    iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
    mean_iu = np.nanmean(iu)

    freq = sum_a1 / (hist.sum() + np.finfo(np.float32).eps)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    #
    cls_iou = dict(zip(['iou_'+str(i) for i in range(n_class)], iu))

    cls_precision = dict(zip(['precision_'+str(i) for i in range(n_class)], precision))
    cls_recall = dict(zip(['recall_'+str(i) for i in range(n_class)], recall))
    cls_F1 = dict(zip(['F1_'+str(i) for i in range(n_class)], F1))

    score_dict = {'acc': acc, 'miou': mean_iu, 'mf1':mean_F1}
    score_dict.update(cls_iou)
    score_dict.update(cls_F1)
    score_dict.update(cls_precision)
    score_dict.update(cls_recall)
    return score_dict


def get_confuse_matrix(num_classes, label_gts, label_preds):
    """计算一组预测的混淆矩阵"""
    def __fast_hist(label_gt, label_pred):
        """
        Collect values for Confusion Matrix
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :param label_gt: <np.array> ground-truth
        :param label_pred: <np.array> prediction
        :return: <np.ndarray> values for confusion matrix
        """
        mask = (label_gt >= 0) & (label_gt < num_classes)
        hist = np.bincount(num_classes * label_gt[mask].astype(int) + label_pred[mask],
                           minlength=num_classes**2).reshape(num_classes, num_classes)
        return hist
    confusion_matrix = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_gts, label_preds):
        confusion_matrix += __fast_hist(lt.flatten(), lp.flatten())
    return confusion_matrix

def get_mIoU(num_classes, label_gts, label_preds):
    confusion_matrix = get_confuse_matrix(num_classes, label_gts, label_preds)
    score_dict = cm2score(confusion_matrix)
    return score_dict['miou']

