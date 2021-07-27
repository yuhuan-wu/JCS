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
        return x, y
