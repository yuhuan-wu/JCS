import torch
import numpy as np

# adapted from https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/score.py
MAX_IMG_PER_BATCH = 256

class IoUEval:
    def __init__(self, nthresh=255):
        self.nthresh = nthresh
        self.thresh = torch.linspace(1./(nthresh + 1), 1. - 1./(nthresh + 1), nthresh).cuda()
        self.EPSILON = np.finfo(np.float).eps

        self.gt_sum = torch.zeros((nthresh,)).cuda()
        self.pred_sum = torch.zeros((nthresh,)).cuda()
        self.num_images = 0
        self.mae = 0
        self.prec = torch.zeros(self.nthresh).cuda()
        self.recall = torch.zeros(self.nthresh).cuda()
        self.iou = 0.


    def add_batch(self, predict, gth):
        for i in range(predict.shape[0]):
            dt = predict[i]; gt = gth[i]
            self.mae += (dt-gt).abs().mean()
            dt = dt > (dt.mean() * 2)
            gt = gt > 0.5
            intersect = (dt*gt).sum()
            iou = intersect.float() / (dt.sum() + gt.sum() - intersect).float()
            self.iou += iou
            
        self.num_images += predict.shape[0]
        

    def get_metric(self):
        x = self.iou / self.num_images
        y = self.mae / self.num_images
        print(x, y)
        return x, y 
        """
        prec = self.prec / self.num_images
        recall = self.recall / self.num_images
        F_beta = (1 + 0.3) * prec * recall / (0.3 * prec + recall + self.EPSILON)
        MAE = self.mae / self.num_images
        print('total_images: {}'.format(self.num_images))

        return F_beta.max().item(), MAE.item()
        """

